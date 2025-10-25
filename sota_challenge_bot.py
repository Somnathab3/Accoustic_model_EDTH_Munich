"""
Clean Challenge Bot for Acoustic Drone Detection
Uses state-of-the-art model with optimized inference pipeline
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import torch
import argparse
import time
import tempfile
import csv
from datetime import datetime
import io
import librosa
import soundfile as sf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import the clean inference module
from sota_inference import AcousticDroneClassifier

# Import challenge bot utilities (keep only submission methods)
from src.adrone.serve.challenge_handler import (
    ChallengeResultStorage,
    ChallengeAPIClient
)


class CleanChallengeBot:
    """
    Clean challenge bot with state-of-the-art model
    Focuses on fast, accurate inference and reliable submission
    """
    
    def __init__(
        self,
        model_path: str,
        labels_path: str,
        csv_path: str = "challenge_results/results.csv",
        storage_dir: str = "challenge_results",
        api_base_url: str = None,
        api_token: str = None
    ):
        """
        Initialize challenge bot
        
        Args:
            model_path: Path to trained model checkpoint
            labels_path: Path to labels JSON
            csv_path: Path to CSV file for storing results
            storage_dir: Directory to store results
            api_base_url: Challenge API base URL (defaults to env var or https://edth.helsing.codes)
            api_token: API authentication token (defaults to env var)
        """
        print("Initializing Clean Challenge Bot...")
        print(f"Model: {model_path}")
        print(f"Labels: {labels_path}")
        
        # Load API configuration from environment variables if not provided
        if api_base_url is None:
            api_base_url = os.getenv('API_BASE_URL', 'https://edth.helsing.codes')
        
        if api_token is None:
            api_token = os.getenv('API_TOKEN')
            if api_token is None:
                raise ValueError(
                    "API_TOKEN not found! Please set it in .env file or pass as argument.\n"
                    "Create a .env file with: API_TOKEN=your_token_here"
                )
        
        print(f"API URL: {api_base_url}")
        print(f"API Token: {api_token[:8]}...{api_token[-4:]} (masked)")
        
        # Store API base URL for constructing full URLs
        self.api_base_url = api_base_url
        
        # Initialize classifier with faster inference settings
        self.classifier = AcousticDroneClassifier(
            model_path=model_path,
            labels_path=labels_path,
            device='auto'
        )
        
        # Pre-warm the model for faster first inference
        print("Warming up model for faster inference...")
        dummy_tensor = torch.randn(1, 3, 96, 126).to(self.classifier.device)
        with torch.no_grad():
            _ = self.classifier.model(dummy_tensor)
        
        # Initialize API client
        self.api_client = ChallengeAPIClient(
            api_base_url=api_base_url,
            api_token=api_token
        )
        
        # Setup HTTP session with connection pooling for faster downloads
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # Initialize result storage
        self.storage = ChallengeResultStorage(storage_dir=storage_dir)
        
        # CSV setup
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create CSV with headers if it doesn't exist
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'iteration', 'timestamp', 'challenge_id', 
                    'predicted', 'actual', 'correct_inferred', 
                    'confidence', 'score_awarded', 'total_score',
                    'success', 'api_message',
                    'inference_time', 'total_time'
                ])
        
        # Statistics
        self.iteration = 0
        self.total_time = 0.0
        self.start_time = time.time()
        
        # Smart timing using server's time_until_next_rotation_ms
        self.last_challenge_id = None  # Track last challenge to detect duplicates
        self.wait_for_new_challenge = False  # Flag to wait for new challenge
        self.next_rotation_time = None  # Server-provided time until next challenge (seconds)
        self.last_challenge_time = None  # When we received the timing info
        
        print("✓ Initialization complete\n")
    
    def _classify_from_memory(self, audio_bytes: bytes) -> tuple:
        """
        Classify audio directly from memory (RAM) without saving to disk
        
        Args:
            audio_bytes: Audio file content as bytes
        
        Returns:
            (prediction, confidence, all_probs)
        """
        # Load audio from bytes using librosa
        audio_io = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_io, sr=self.classifier.preprocessor.sample_rate, mono=True)
        
        # Convert to tensor
        waveform = torch.from_numpy(y).unsqueeze(0).float()
        
        # Preprocess
        spectrogram = self.classifier.preprocessor(waveform)
        spectrogram = spectrogram.unsqueeze(0).to(self.classifier.device)
        
        # Inference
        import torch.nn.functional as F
        with torch.no_grad():
            logits = self.classifier.model(spectrogram)
            probabilities = F.softmax(logits, dim=1)
            
            predicted_idx = probabilities.argmax(dim=1).item()
            confidence = probabilities[0, predicted_idx].item()
            prediction = self.classifier.idx_to_class[predicted_idx]
            
            all_probs = {
                self.classifier.idx_to_class[i]: probabilities[0, i].item()
                for i in range(self.classifier.num_classes)
            }
        
        return prediction, confidence, all_probs
    
    def run_single_challenge(self) -> bool:
        """
        Run a single challenge iteration - OPTIMIZED FOR SPEED
        
        Returns:
            True if successful, False otherwise
        """
        try:
            iter_start = time.time()
            
            # Fetch challenge
            challenge = self.api_client.get_challenge()
            challenge_id = challenge['challenge_id']
            wav_url = challenge['wav_url']
            time_until_next_ms = challenge.get('time_until_next_rotation_ms', None)
            
            # Store the timing information from server
            # This is the server's countdown to NEXT challenge rotation from NOW
            # We submit to the CURRENT challenge immediately, then wait this time for the next one
            if time_until_next_ms is not None:
                self.next_rotation_time = time_until_next_ms / 1000.0  # Convert ms to seconds
                self.last_challenge_time = time.time()  # Record when we got this info
                print(f"⏱️  Server reports NEXT challenge will be ready in {self.next_rotation_time:.1f}s")
            else:
                # No timing provided, use default 100s
                self.next_rotation_time = 100.0
                self.last_challenge_time = time.time()
                print(f"⏱️  No timing data, assuming 100s cycle")
            
            # Check if this is the same challenge (already submitted)
            if challenge_id == self.last_challenge_id:
                print(f"⏸️  Same challenge detected ({challenge_id[:8]}...) - waiting for new challenge")
                self.wait_for_new_challenge = True
                return False  # Don't process, wait longer
            
            # Handle relative URLs - prepend base URL if needed
            if wav_url.startswith('/'):
                # API returned relative URL, construct full URL
                full_wav_url = f"{self.api_base_url}{wav_url}"
            else:
                full_wav_url = wav_url
            
            # OPTIMIZATION: Download audio directly to MEMORY (RAM) instead of disk
            # This is much faster than disk I/O
            download_start = time.time()
            response = self.session.get(full_wav_url, stream=True)
            response.raise_for_status()
            audio_bytes = response.content
            download_time = time.time() - download_start
            
            # OPTIMIZATION: Classify directly from memory (no disk I/O)
            inference_start = time.time()
            prediction, confidence, all_probs = self._classify_from_memory(audio_bytes)
            inference_time = time.time() - inference_start
            
            # Submit prediction IMMEDIATELY for speed bonus (don't wait for anything)
            print(f"📤 Submitting: {prediction} (confidence: {confidence:.3f})")
            try:
                result = self.api_client.submit_classification(challenge_id, prediction)
                
                # Print full API response for debugging
                print(f"� API Response:")
                import json
                for key, value in result.items():
                    print(f"   {key}: {value}")
                print()
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a timeout error
                if "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
                    print(f"⚠️  Submission timeout - server may be slow, continuing...")
                    # Don't fail, just record what we can and move on
                    result = {
                        'correct': False,
                        'score_awarded': 0,
                        'total_score': 0,
                        'actual_classification': 'unknown',
                        'success': False,
                        'message': 'timeout',
                        'error': 'timeout'
                    }
                    is_correct = False
                    score_awarded = 0
                    total_score = 0
                    actual_label = 'unknown'
                    success = False
                    api_message = 'timeout'
                    
                    # Still mark as processed to avoid resubmission
                    self.last_challenge_id = challenge_id
                    self.wait_for_new_challenge = False
                    
                    # Update statistics
                    self.iteration += 1
                    total_time = time.time() - iter_start
                    self.total_time += total_time
                    
                    # Write to CSV
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    with open(self.csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            self.iteration,
                            timestamp,
                            challenge_id,
                            prediction,
                            actual_label,
                            is_correct,
                            f"{confidence:.4f}",
                            score_awarded,
                            total_score,
                            success,
                            api_message,
                            f"{inference_time:.4f}",
                            f"{total_time:.4f}"
                        ])
                    
                    # Clean up is not needed - no temp file created
                    # Continue to next challenge
                    return True
                
                # Check if already submitted
                elif "already submitted" in error_msg.lower():
                    print(f"⚠️  Already submitted for challenge {challenge_id[:8]}...")
                    self.last_challenge_id = challenge_id
                    self.wait_for_new_challenge = True
                    
                    # No cleanup needed
                    return False
                else:
                    # Other error, re-raise
                    raise
            
            total_time = time.time() - iter_start
            
            # Now process result and log (after submission to maximize speed)
            # NOTE: API does NOT provide 'correct' or 'actual_classification' fields!
            # We only get: success, message, score_awarded, total_score
            score_awarded = result.get('score_awarded', 0)
            total_score = result.get('total_score', 0)
            success = result.get('success', True)
            api_message = result.get('message', '')
            
            # Infer if correct based on score_awarded (scores > 0 mean correct)
            is_correct = score_awarded > 0
            actual_label = 'unknown'  # API doesn't provide this
            
            # Check if we got any response (score or not)
            # This means it's a real new challenge, not a duplicate
            self.last_challenge_id = challenge_id
            self.wait_for_new_challenge = False
            
            # Update statistics
            self.iteration += 1
            self.total_time += total_time
            
            # Write to CSV immediately (append mode for speed)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.iteration,
                    timestamp,
                    challenge_id,
                    prediction,
                    actual_label,
                    is_correct,
                    f"{confidence:.4f}",
                    score_awarded,
                    total_score,
                    success,
                    api_message,
                    f"{inference_time:.4f}",
                    f"{total_time:.4f}"
                ])
            
            # NOW save audio to disk AFTER submission (for storage/analysis)
            # This doesn't slow down submission since it happens after
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                
                # Save audio bytes to file for storage
                audio_io = io.BytesIO(audio_bytes)
                y, sr = librosa.load(audio_io, sr=16000, mono=True)
                sf.write(tmp_path, y, sr)
                
                # Store result in JSONL (background compatible)
                self.storage.store_result(
                    challenge_id=challenge_id,
                    audio_path=tmp_path,
                    prediction=prediction,
                    confidence=confidence,
                    all_probabilities=all_probs,
                    is_correct=is_correct,
                    score_awarded=score_awarded,
                    actual_label=actual_label,
                    inference_time=inference_time,
                    response_data=result
                )
                
                # Clean up temp file after storing
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            except Exception as e:
                print(f"⚠️  Warning: Could not save audio for storage: {e}")
                # Don't fail - submission was successful
            
            # Print result
            status = "✓" if is_correct else "✗"
            if is_correct:
                print(f"[{self.iteration}] {status} Predicted: {prediction:11s} | "
                      f"Score: +{score_awarded:3d} | Total: {total_score:4d} | "
                      f"Conf: {confidence:.3f} | "
                      f"Time: {total_time:.2f}s")
            else:
                print(f"[{self.iteration}] {status} Predicted: {prediction:11s} | "
                      f"Score: +{score_awarded:3d} (WRONG) | Total: {total_score:4d} | "
                      f"Conf: {confidence:.3f} | "
                      f"Time: {total_time:.2f}s")
            
            # Print API message if present and interesting
            if api_message and api_message not in ['', 'Success']:
                print(f"     API: {api_message}")
            
            # Print probabilities for debugging (only if wrong)
            if not is_correct:
                probs_str = ' | '.join([f'{k}:{v:.3f}' for k, v in all_probs.items()])
                print(f"     [{probs_str}]")
            
            # Print timing breakdown
            print(f"     ⏱️  Download: {download_time:.3f}s | Inference: {inference_time:.3f}s | "
                  f"Submission: {time.time() - (iter_start + download_time + inference_time):.3f}s")
            
            return True
        
        except Exception as e:
            print(f"❌ Error in challenge iteration: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self, max_iterations: int = None, delay: float = 0.0):
        """
        Run the challenge bot with SERVER-SYNCHRONIZED TIMING
        
        Uses time_until_next_rotation_ms from API response for optimal timing
        
        Args:
            max_iterations: Maximum number of iterations (None = infinite)
            delay: Base delay between challenges (0.0 for fastest)
        """
        print("="*60)
        print("🎯 SERVER-SYNCED CHALLENGE BOT")
        print("="*60)
        print(f"Max iterations: {max_iterations if max_iterations else 'Infinite'}")
        print(f"Strategy: Wait for server's time_until_next_rotation_ms")
        print(f"No burst polling - clean single requests at optimal timing")
        print("="*60 + "\n")
        
        iteration_count = 0
        consecutive_failures = 0
        
        try:
            while True:
                # Check if we've reached max iterations
                if max_iterations and iteration_count >= max_iterations:
                    break
                
                # Run challenge
                success = self.run_single_challenge()
                iteration_count += 1
                
                if success:
                    consecutive_failures = 0
                    
                    # Print summary every 10 iterations
                    if iteration_count % 10 == 0:
                        self.print_summary()
                    
                    # USE SERVER-PROVIDED TIMING FOR NEXT CHALLENGE
                    if self.next_rotation_time is not None and self.last_challenge_time is not None:
                        # Calculate how much time has passed since we got the timing info
                        elapsed = time.time() - self.last_challenge_time
                        
                        # Use EXACT server timing - no buffer added
                        # The server's time_until_next_rotation_ms already accounts for everything
                        remaining_time = self.next_rotation_time - elapsed
                        
                        if remaining_time > 1.0:
                            print(f"⏳ Waiting {remaining_time:.1f}s for next challenge (server timing)")
                            print(f"   Will check for new challenge at: {time.strftime('%H:%M:%S', time.localtime(time.time() + remaining_time))}")
                            time.sleep(remaining_time)
                            print(f"✓ Checking for new challenge now...")
                        elif remaining_time > 0:
                            # Small remaining time, still wait
                            print(f"⏱️  Short wait: {remaining_time:.1f}s...")
                            time.sleep(remaining_time)
                        else:
                            # Already past expected time, check immediately
                            print(f"✓ Challenge should be ready (past server timing by {-remaining_time:.1f}s)")
                    else:
                        # No timing info, use base delay
                        if delay > 0:
                            time.sleep(delay)
                else:
                    consecutive_failures += 1
                    
                    # If we're waiting for a new challenge (duplicate detected)
                    if self.wait_for_new_challenge:
                        # USE SERVER TIMING if available
                        if self.next_rotation_time is not None and self.last_challenge_time is not None:
                            elapsed = time.time() - self.last_challenge_time
                            remaining_time = self.next_rotation_time - elapsed  # No buffer
                            
                            if remaining_time > 1.0:
                                # Wait for the full server-reported duration
                                print(f"⏳ Duplicate detected - waiting {remaining_time:.1f}s (server timing)")
                                time.sleep(remaining_time)
                                print(f"✓ Checking for new challenge now...")
                            else:
                                # Already at or past expected time, wait a bit and retry
                                wait_time = max(2.0, remaining_time)
                                print(f"⏱️  Waiting {wait_time:.1f}s for new challenge...")
                                time.sleep(wait_time)
                        else:
                            # NO SERVER TIMING - Use simple 2s retry
                            print(f"🔍 No timing data - checking again in 2s...")
                            time.sleep(2.0)
                        
                        self.wait_for_new_challenge = False  # Reset flag
                    
                    # If multiple consecutive failures, increase delay
                    elif consecutive_failures >= 3:
                        backoff_delay = min(5.0 * (2 ** (consecutive_failures - 3)), 30.0)
                        print(f"⚠️  Multiple failures detected, backing off for {backoff_delay:.1f}s...")
                        time.sleep(backoff_delay)
                    else:
                        # Normal retry with short delay
                        time.sleep(1.0)
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopped by user")
        
        # Final summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        self.print_summary()
        
        # Print detailed statistics
        analysis = self.storage.analyze_performance()
        print(f"\nPerformance Analysis:")
        for key, value in analysis.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        print(f"\n✓ Results saved to: {self.csv_path}")
        print(f"✓ Use 'python analyze_results.py' to view detailed analysis")
    
    def print_summary(self):
        """Print current statistics"""
        stats = self.storage.stats
        
        if stats['total_attempts'] == 0:
            return
        
        accuracy = stats['accuracy']
        avg_time = self.total_time / self.iteration if self.iteration > 0 else 0
        
        print(f"\n{'─'*60}")
        print(f"Attempts: {stats['total_attempts']} | "
              f"Correct: {stats['correct']} | "
              f"Wrong: {stats['wrong']}")
        print(f"Accuracy: {accuracy:.1%} | "
              f"Score: {stats['total_score']} | "
              f"Avg Time: {avg_time:.3f}s")
        print(f"{'─'*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Clean Challenge Bot for Acoustic Drone Detection')
    
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint (default: auto-detect best available)')
    parser.add_argument('--labels', type=str, default=None,
                        help='Path to labels JSON (default: auto-detect)')
    parser.add_argument('--csv', type=str, default='challenge_results/results.csv',
                        help='Path to CSV file for results')
    parser.add_argument('--storage-dir', type=str, default='challenge_results',
                        help='Directory to store results')
    parser.add_argument('--api-url', type=str, default=None,
                        help='Challenge API base URL (default: from .env or https://edth.helsing.codes)')
    parser.add_argument('--api-token', type=str, default=None,
                        help='API authentication token (default: loaded from .env file)')
    parser.add_argument('--max-iterations', type=int, default=None,
                        help='Maximum number of iterations (default: infinite)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between challenges in seconds (default: 0.5 for speed)')
    
    args = parser.parse_args()
    
    # Auto-detect best model if not specified
    if args.model is None:
        # Priority: crnn_combined/crnn_final.pt (BEST - latest trained) > panns_combined/panns_final.pt > models/best_model.pt
        crnn_final_model = Path('models/crnn_combined/crnn_final.pt')
        crnn_best_model = Path('models/crnn_combined/best_model.pt')
        panns_final_model = Path('models/panns_combined/panns_final.pt')
        panns_best_model = Path('models/panns_combined/best_model.pt')
        root_best_model = Path('models/best_model.pt')
        
        if crnn_final_model.exists():
            args.model = str(crnn_final_model)
            print(f"✓ Using LATEST CRNN MODEL: {args.model}")
            print(f"  🎯 This is the complete trained CRNN model from crnn_combined")
            
            # Load model info
            try:
                checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
                model_type = checkpoint.get('model_type', 'crnn')
                num_classes = checkpoint.get('num_classes', 3)
                input_channels = checkpoint.get('input_channels', 3)
                n_mels = checkpoint.get('n_mels', 96)
                print(f"  📊 Model: {model_type.upper()} | Channels: {input_channels} | Classes: {num_classes} | Mels: {n_mels}")
            except:
                print(f"  📊 Model: CRNN (assumed)")
                
        elif crnn_best_model.exists():
            args.model = str(crnn_best_model)
            print(f"✓ Using best checkpoint from crnn_combined: {args.model}")
            print(f"  💡 Note: crnn_final.pt not found, using best checkpoint")
        elif panns_final_model.exists():
            args.model = str(panns_final_model)
            print(f"✓ Using PANNs model (fallback): {args.model}")
            print(f"  💡 Note: CRNN model not found, using PANNs")
        elif panns_best_model.exists():
            args.model = str(panns_best_model)
            print(f"✓ Using best checkpoint from panns_combined: {args.model}")
        elif root_best_model.exists():
            args.model = str(root_best_model)
            print(f"⚡ Using root best checkpoint: {args.model}")
        else:
            print("❌ Error: No model found!")
            print("\nSearched for:")
            print(f"  - {crnn_final_model} (RECOMMENDED - latest trained CRNN)")
            print(f"  - {crnn_best_model} (CRNN best checkpoint)")
            print(f"  - {panns_final_model} (PANNs fallback)")
            print(f"  - {panns_best_model}")
            print(f"  - {root_best_model}")
            print("\nPlease train the model first using:")
            print("  python train_sota_model.py --train-dir data/edth_munich_dataset/data/train "
                  "--val-dir data/edth_munich_dataset/data/val")
            sys.exit(1)
    else:
        if not Path(args.model).exists():
            print(f"❌ Error: Model file not found: {args.model}")
            sys.exit(1)
    
    # Auto-detect labels if not specified
    if args.labels is None:
        # Priority: crnn_combined/labels.json (matching CRNN) > panns_combined/labels.json > models/labels_current.json
        crnn_labels = Path('models/crnn_combined/labels.json')
        panns_labels = Path('models/panns_combined/labels.json')
        labels_current = Path('models/labels_current.json')
        labels_default = Path('models/labels.json')
        
        if crnn_labels.exists():
            args.labels = str(crnn_labels)
            print(f"✓ Using labels from crnn_combined: {args.labels}")
        elif panns_labels.exists():
            args.labels = str(panns_labels)
            print(f"✓ Using labels from panns_combined: {args.labels}")
        elif labels_current.exists():
            args.labels = str(labels_current)
            print(f"✓ Using labels_current: {args.labels}")
        elif labels_default.exists():
            args.labels = str(labels_default)
            print(f"✓ Using root labels: {args.labels}")
        else:
            print("❌ Error: No labels file found!")
            print("\nSearched for:")
            print(f"  - {crnn_labels} (CRNN labels)")
            print(f"  - {panns_labels} (PANNs labels)")
            print(f"  - {labels_current}")
            print(f"  - {labels_default}")
            sys.exit(1)
    else:
        if not Path(args.labels).exists():
            print(f"❌ Error: Labels file not found: {args.labels}")
            sys.exit(1)
    
    print(f"✓ Using labels: {args.labels}")
    print(f"✓ Results will be saved to: {args.csv}")
    print()
    
    # Create and run bot
    bot = CleanChallengeBot(
        model_path=args.model,
        labels_path=args.labels,
        csv_path=args.csv,
        storage_dir=args.storage_dir,
        api_base_url=args.api_url,
        api_token=args.api_token
    )
    
    bot.run(
        max_iterations=args.max_iterations,
        delay=args.delay
    )


if __name__ == '__main__':
    main()
