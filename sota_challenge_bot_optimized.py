"""
OPTIMIZED Challenge Bot for Acoustic Drone Detection
Implements timing control to hit optimal 100s submission window for maximum scores
"""
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import torch
import argparse
import time
import tempfile
import csv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import io

# Import the clean inference module
from sota_inference import AcousticDroneClassifier

# Import challenge bot utilities (keep only submission methods)
from src.adrone.serve.challenge_handler import (
    ChallengeResultStorage,
    ChallengeAPIClient
)


class OptimizedChallengeBot:
    """
    OPTIMIZED challenge bot with timing control
    Targets ~100s total time for maximum scores
    """
    
    def __init__(
        self,
        model_path: str,
        labels_path: str,
        csv_path: str = "challenge_results/results_optimized.csv",
        storage_dir: str = "challenge_results",
        api_base_url: str = "https://edth.helsing.codes",
        api_token: str = "9726345a-34ed-4995-94d9-ecc239b47c1d",
        target_time: float = 99.8  # Target submission time for optimal scores
    ):
        """
        Initialize optimized challenge bot
        
        Args:
            model_path: Path to trained model checkpoint
            labels_path: Path to labels JSON
            csv_path: Path to CSV file for storing results
            storage_dir: Directory to store results
            api_base_url: Challenge API base URL
            api_token: API authentication token
            target_time: Target total time in seconds (default: 99.8s for ~100s window)
        """
        print("="*60)
        print("🚀 OPTIMIZED CHALLENGE BOT - TIMING CONTROL ENABLED")
        print("="*60)
        print(f"Model: {model_path}")
        print(f"Labels: {labels_path}")
        print(f"🎯 Target submission time: {target_time:.1f}s (for maximum scores)")
        print("="*60)
        
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
        
        # Initialize API client with connection pooling
        self.api_client = ChallengeAPIClient(
            api_base_url=api_base_url,
            api_token=api_token
        )
        
        # Optimize HTTP client with connection pooling
        self._optimize_http_client()
        
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
                    'inference_time', 'total_time', 'delay_added'
                ])
        
        # Timing control
        self.target_time = target_time
        
        # Statistics
        self.iteration = 0
        self.total_time = 0.0
        self.start_time = time.time()
        
        # Smart timing using server's time_until_next_rotation_ms
        self.last_challenge_id = None
        self.wait_for_new_challenge = False
        self.next_rotation_time = None
        self.last_challenge_time = None
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        print("✓ Initialization complete\n")
    
    def _optimize_http_client(self):
        """Optimize HTTP client with connection pooling"""
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            # Check if api_client has a session attribute
            if hasattr(self.api_client, 'session'):
                session = self.api_client.session
            else:
                # Create new session and attach to api_client
                session = requests.Session()
                self.api_client.session = session
            
            # Configure retry strategy
            retry = Retry(
                total=3,
                backoff_factor=0.1,
                status_forcelist=[500, 502, 503, 504]
            )
            
            # Configure connection pooling
            adapter = HTTPAdapter(
                max_retries=retry,
                pool_connections=10,
                pool_maxsize=10
            )
            
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            
            print("✓ HTTP connection pooling enabled")
        except Exception as e:
            print(f"⚠️  Could not optimize HTTP client: {e}")
    
    def _download_audio(self, url: str, path: str) -> str:
        """Download audio file"""
        self.api_client.download_audio(url, path)
        return path
    
    def _classify_audio(self, path: str) -> tuple:
        """Classify audio file"""
        return self.classifier.classify(path)
    
    def run_single_challenge(self) -> bool:
        """
        Run a single challenge iteration - OPTIMIZED WITH TIMING CONTROL
        
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
            if time_until_next_ms is not None:
                self.next_rotation_time = time_until_next_ms / 1000.0
                self.last_challenge_time = time.time()
                print(f"⏱️  Server reports NEXT challenge in {self.next_rotation_time:.1f}s")
            else:
                self.next_rotation_time = 100.0
                self.last_challenge_time = time.time()
                print(f"⏱️  No timing data, assuming 100s cycle")
            
            # Check if this is the same challenge (already submitted)
            if challenge_id == self.last_challenge_id:
                print(f"⏸️  Same challenge detected ({challenge_id[:8]}...) - waiting for new challenge")
                self.wait_for_new_challenge = True
                return False
            
            # Download audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # STRATEGY 2A: Parallel download and processing preparation
            # Download happens first, then we classify
            download_start = time.time()
            tmp_path = self._download_audio(wav_url, tmp_path)
            download_time = time.time() - download_start
            
            # STRATEGY 1: Calculate time budget for inference + submission
            elapsed_before_inference = time.time() - iter_start
            time_budget_remaining = self.target_time - elapsed_before_inference
            
            # Classify with SOTA model
            inference_start = time.time()
            prediction, confidence, all_probs = self._classify_audio(tmp_path)
            inference_time = time.time() - inference_start
            
            # Calculate elapsed time so far
            elapsed_before_submission = time.time() - iter_start
            
            # STRATEGY 1: Add strategic delay to hit target time (before submission)
            delay_added = 0.0
            if elapsed_before_submission < self.target_time:
                delay_needed = self.target_time - elapsed_before_submission
                # Leave ~0.2s buffer for submission network time
                safe_delay = max(0, delay_needed - 0.2)
                if safe_delay > 0:
                    print(f"⏱️  Strategic delay: {safe_delay:.3f}s (targeting {self.target_time:.1f}s window)")
                    time.sleep(safe_delay)
                    delay_added = safe_delay
            
            # Submit prediction
            submission_start = time.time()
            print(f"📤 Submitting: {prediction} (confidence: {confidence:.3f})")
            try:
                result = self.api_client.submit_classification(challenge_id, prediction)
                
                # Print API response
                print(f"📥 API Response:")
                for key, value in result.items():
                    print(f"   {key}: {value}")
                print()
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a timeout error
                if "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
                    print(f"⚠️  Submission timeout - server may be slow, continuing...")
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
                    
                    self.last_challenge_id = challenge_id
                    self.wait_for_new_challenge = False
                    
                    self.iteration += 1
                    total_time = time.time() - iter_start
                    self.total_time += total_time
                    
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    with open(self.csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            self.iteration, timestamp, challenge_id,
                            prediction, actual_label, is_correct,
                            f"{confidence:.4f}", score_awarded, total_score,
                            success, api_message,
                            f"{inference_time:.4f}", f"{total_time:.4f}", f"{delay_added:.4f}"
                        ])
                    
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                    
                    return True
                
                # Check if already submitted
                elif "already submitted" in error_msg.lower():
                    print(f"⚠️  Already submitted for challenge {challenge_id[:8]}...")
                    self.last_challenge_id = challenge_id
                    self.wait_for_new_challenge = True
                    
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                    return False
                else:
                    raise
            
            total_time = time.time() - iter_start
            submission_time = time.time() - submission_start
            
            # Process result
            score_awarded = result.get('score_awarded', 0)
            total_score = result.get('total_score', 0)
            success = result.get('success', True)
            api_message = result.get('message', '')
            
            is_correct = score_awarded > 0
            actual_label = 'unknown'
            
            self.last_challenge_id = challenge_id
            self.wait_for_new_challenge = False
            
            self.iteration += 1
            self.total_time += total_time
            
            # Write to CSV with timing breakdown
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.iteration, timestamp, challenge_id,
                    prediction, actual_label, is_correct,
                    f"{confidence:.4f}", score_awarded, total_score,
                    success, api_message,
                    f"{inference_time:.4f}", f"{total_time:.4f}", f"{delay_added:.4f}"
                ])
            
            # Store result in JSONL
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
            
            # Print result with timing breakdown
            status = "✓" if is_correct else "✗"
            timing_in_window = 99.5 <= total_time <= 100.2
            window_marker = "🎯" if timing_in_window else "  "
            
            if is_correct:
                print(f"{window_marker}[{self.iteration}] {status} Predicted: {prediction:11s} | "
                      f"Score: +{score_awarded:3d} | Total: {total_score:4d} | "
                      f"Conf: {confidence:.3f} | "
                      f"Time: {total_time:.2f}s")
            else:
                print(f"{window_marker}[{self.iteration}] {status} Predicted: {prediction:11s} | "
                      f"Score: +{score_awarded:3d} (WRONG) | Total: {total_score:4d} | "
                      f"Conf: {confidence:.3f} | "
                      f"Time: {total_time:.2f}s")
            
            # Print timing breakdown
            print(f"     ⏱️  Download: {download_time:.3f}s | Inference: {inference_time:.3f}s | "
                  f"Delay: {delay_added:.3f}s | Submission: {submission_time:.3f}s")
            
            if api_message and api_message not in ['', 'Success', 'Correct!', 'Incorrect']:
                print(f"     API: {api_message}")
            
            if not is_correct:
                probs_str = ' | '.join([f'{k}:{v:.3f}' for k, v in all_probs.items()])
                print(f"     [{probs_str}]")
            
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            return True
        
        except Exception as e:
            print(f"❌ Error in challenge iteration: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self, max_iterations: int = None, delay: float = 0.0):
        """
        Run the optimized challenge bot
        
        Args:
            max_iterations: Maximum number of iterations (None = infinite)
            delay: Base delay between challenges (0.0 for fastest)
        """
        print("="*60)
        print("🚀 OPTIMIZED CHALLENGE BOT - RUNNING")
        print("="*60)
        print(f"Target timing: {self.target_time:.1f}s (optimal scoring window)")
        print(f"Max iterations: {max_iterations if max_iterations else 'Infinite'}")
        print(f"Strategy: Timing control + Connection pooling")
        print("="*60 + "\n")
        
        iteration_count = 0
        consecutive_failures = 0
        
        try:
            while True:
                if max_iterations and iteration_count >= max_iterations:
                    break
                
                success = self.run_single_challenge()
                iteration_count += 1
                
                if success:
                    consecutive_failures = 0
                    
                    if iteration_count % 10 == 0:
                        self.print_summary()
                    
                    # STRATEGY 3: Smart timing based on server response
                    if self.next_rotation_time is not None and self.last_challenge_time is not None:
                        elapsed = time.time() - self.last_challenge_time
                        remaining_time = self.next_rotation_time - elapsed
                        
                        if remaining_time > 1.0:
                            print(f"⏳ Waiting {remaining_time:.1f}s for next challenge (server timing)")
                            print(f"   Will check for new challenge at: {time.strftime('%H:%M:%S', time.localtime(time.time() + remaining_time))}")
                            time.sleep(remaining_time)
                            print(f"✓ Checking for new challenge now...")
                        elif remaining_time > 0:
                            print(f"⏱️  Short wait: {remaining_time:.1f}s...")
                            time.sleep(remaining_time)
                        else:
                            print(f"✓ Challenge should be ready (past server timing by {-remaining_time:.1f}s)")
                    else:
                        if delay > 0:
                            time.sleep(delay)
                else:
                    consecutive_failures += 1
                    
                    if self.wait_for_new_challenge:
                        if self.next_rotation_time is not None and self.last_challenge_time is not None:
                            elapsed = time.time() - self.last_challenge_time
                            remaining_time = self.next_rotation_time - elapsed
                            
                            if remaining_time > 1.0:
                                print(f"⏳ Duplicate detected - waiting {remaining_time:.1f}s (server timing)")
                                time.sleep(remaining_time)
                                print(f"✓ Checking for new challenge now...")
                            else:
                                wait_time = max(2.0, remaining_time)
                                print(f"⏱️  Waiting {wait_time:.1f}s for new challenge...")
                                time.sleep(wait_time)
                        else:
                            print(f"🔍 No timing data - checking again in 2s...")
                            time.sleep(2.0)
                        
                        self.wait_for_new_challenge = False
                    
                    elif consecutive_failures >= 3:
                        backoff_delay = min(5.0 * (2 ** (consecutive_failures - 3)), 30.0)
                        print(f"⚠️  Multiple failures detected, backing off for {backoff_delay:.1f}s...")
                        time.sleep(backoff_delay)
                    else:
                        time.sleep(1.0)
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopped by user")
        finally:
            # Cleanup thread pool
            self.executor.shutdown(wait=False)
        
        # Final summary
        print("\n" + "="*60)
        print("FINAL SUMMARY - OPTIMIZED BOT")
        print("="*60)
        self.print_summary()
        
        # Print timing statistics
        self.print_timing_stats()
        
        analysis = self.storage.analyze_performance()
        print(f"\nPerformance Analysis:")
        for key, value in analysis.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        print(f"\n✓ Results saved to: {self.csv_path}")
        print(f"✓ Compare with original using: python analyze_results.py")
    
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
    
    def print_timing_stats(self):
        """Print timing statistics"""
        try:
            import pandas as pd
            df = pd.read_csv(self.csv_path)
            
            if len(df) == 0:
                return
            
            print(f"\n{'='*60}")
            print("TIMING ANALYSIS - OPTIMIZED BOT")
            print(f"{'='*60}")
            
            successful = df[df['score_awarded'] > 0]
            
            if len(successful) > 0:
                in_window = successful[(successful['total_time'] >= 99.5) & (successful['total_time'] <= 100.2)]
                
                print(f"\n🎯 Submissions in optimal window (99.5-100.2s):")
                print(f"   Count: {len(in_window)} / {len(successful)} ({len(in_window)/len(successful)*100:.1f}%)")
                print(f"   Average score: {in_window['score_awarded'].mean():.1f}")
                print(f"   Max score: {in_window['score_awarded'].max()}")
                
                print(f"\n📊 Overall timing:")
                print(f"   Average total_time: {successful['total_time'].mean():.3f}s")
                print(f"   Average inference_time: {successful['inference_time'].mean():.3f}s")
                print(f"   Average delay_added: {successful['delay_added'].mean():.3f}s")
                print(f"   Target hit rate: {len(in_window)/len(successful)*100:.1f}%")
        except Exception as e:
            print(f"Could not generate timing stats: {e}")


def main():
    parser = argparse.ArgumentParser(description='OPTIMIZED Challenge Bot - Timing Control')
    
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint (default: auto-detect best available)')
    parser.add_argument('--labels', type=str, default=None,
                        help='Path to labels JSON (default: auto-detect)')
    parser.add_argument('--csv', type=str, default='challenge_results/results_optimized.csv',
                        help='Path to CSV file for results')
    parser.add_argument('--storage-dir', type=str, default='challenge_results',
                        help='Directory to store results')
    parser.add_argument('--api-url', type=str, default='https://edth.helsing.codes',
                        help='Challenge API base URL')
    parser.add_argument('--api-token', type=str, default='9726345a-34ed-4995-94d9-ecc239b47c1d',
                        help='API authentication token')
    parser.add_argument('--max-iterations', type=int, default=None,
                        help='Maximum number of iterations (default: infinite)')
    parser.add_argument('--delay', type=float, default=0.0,
                        help='Delay between challenges in seconds (default: 0.0)')
    parser.add_argument('--target-time', type=float, default=99.8,
                        help='Target submission time in seconds (default: 99.8 for ~100s window)')
    
    args = parser.parse_args()
    
    # Auto-detect best model if not specified
    if args.model is None:
        crnn_final_model = Path('models/crnn_combined/crnn_final.pt')
        crnn_best_model = Path('models/crnn_combined/best_model.pt')
        panns_final_model = Path('models/panns_combined/panns_final.pt')
        panns_best_model = Path('models/panns_combined/best_model.pt')
        root_best_model = Path('models/best_model.pt')
        
        if crnn_final_model.exists():
            args.model = str(crnn_final_model)
            print(f"✓ Using LATEST CRNN MODEL: {args.model}")
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
        elif panns_final_model.exists():
            args.model = str(panns_final_model)
            print(f"✓ Using PANNs model (fallback): {args.model}")
        elif panns_best_model.exists():
            args.model = str(panns_best_model)
            print(f"✓ Using best checkpoint from panns_combined: {args.model}")
        elif root_best_model.exists():
            args.model = str(root_best_model)
            print(f"⚡ Using root best checkpoint: {args.model}")
        else:
            print("❌ Error: No model found!")
            sys.exit(1)
    else:
        if not Path(args.model).exists():
            print(f"❌ Error: Model file not found: {args.model}")
            sys.exit(1)
    
    # Auto-detect labels if not specified
    if args.labels is None:
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
            sys.exit(1)
    else:
        if not Path(args.labels).exists():
            print(f"❌ Error: Labels file not found: {args.labels}")
            sys.exit(1)
    
    print(f"✓ Using labels: {args.labels}")
    print(f"✓ Results will be saved to: {args.csv}")
    print()
    
    # Create and run optimized bot
    bot = OptimizedChallengeBot(
        model_path=args.model,
        labels_path=args.labels,
        csv_path=args.csv,
        storage_dir=args.storage_dir,
        api_base_url=args.api_url,
        api_token=args.api_token,
        target_time=args.target_time
    )
    
    bot.run(
        max_iterations=args.max_iterations,
        delay=args.delay
    )


if __name__ == '__main__':
    main()
