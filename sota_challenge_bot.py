"""
Clean Challenge Bot for Acoustic Drone Detection
Uses state-of-the-art model with optimized inference pipeline
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
        api_base_url: str = "https://edth.helsing.codes",
        api_token: str = "9726345a-34ed-4995-94d9-ecc239b47c1d"
    ):
        """
        Initialize challenge bot
        
        Args:
            model_path: Path to trained model checkpoint
            labels_path: Path to labels JSON
            csv_path: Path to CSV file for storing results
            storage_dir: Directory to store results
            api_base_url: Challenge API base URL
            api_token: API authentication token
        """
        print("Initializing Clean Challenge Bot...")
        print(f"Model: {model_path}")
        print(f"Labels: {labels_path}")
        
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
                    'predicted', 'actual', 'correct', 
                    'confidence', 'score_awarded', 
                    'inference_time', 'total_time'
                ])
        
        # Statistics
        self.iteration = 0
        self.total_time = 0.0
        self.start_time = time.time()
        
        # Smart timing for score-based synchronization
        self.last_score_time = None  # When we last got a score
        self.last_challenge_id = None  # Track last challenge to detect duplicates
        self.wait_for_new_challenge = False  # Flag to wait for new challenge
        self.first_score_received = False  # Track if we've ever received a score
        
        print("✓ Initialization complete\n")
    
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
            
            # Check if this is the same challenge (already submitted)
            if challenge_id == self.last_challenge_id:
                print(f"⏸️  Same challenge detected ({challenge_id[:8]}...) - waiting for new challenge")
                self.wait_for_new_challenge = True
                return False  # Don't process, wait longer
            
            # Download audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            self.api_client.download_audio(wav_url, tmp_path)
            
            # Classify with SOTA model - FAST
            inference_start = time.time()
            prediction, confidence, all_probs = self.classifier.classify(tmp_path)
            inference_time = time.time() - inference_start
            
            # Submit prediction IMMEDIATELY for speed bonus (don't wait for anything)
            try:
                result = self.api_client.submit_classification(challenge_id, prediction)
            except Exception as e:
                error_msg = str(e)
                
                # Check if already submitted
                if "already submitted" in error_msg.lower():
                    print(f"⚠️  Already submitted for challenge {challenge_id[:8]}...")
                    self.last_challenge_id = challenge_id
                    self.wait_for_new_challenge = True
                    
                    # Clean up and return
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                    return False
                else:
                    # Other error, re-raise
                    raise
            
            total_time = time.time() - iter_start
            
            # Now process result and log (after submission to maximize speed)
            is_correct = result.get('correct', False)
            score_awarded = result.get('score_awarded', 0)
            actual_label = result.get('actual_classification', 'unknown')
            
            # Check if we got any response (score or not)
            # This means it's a real new challenge, not a duplicate
            self.last_challenge_id = challenge_id
            self.wait_for_new_challenge = False
            
            # Track first SUCCESSFUL score to enable syncing
            if score_awarded > 0:
                if not self.first_score_received:
                    self.first_score_received = True
                    self.last_score_time = time.time()
                    print(f"🎯 First score received! Now synced with server timing...")
                else:
                    self.last_score_time = time.time()
                    print(f"🎯 Score received! Re-syncing timing...")
            
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
                    f"{inference_time:.4f}",
                    f"{total_time:.4f}"
                ])
            
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
            
            # Print result
            status = "✓" if is_correct else "✗"
            print(f"[{self.iteration}] {status} Predicted: {prediction:11s} | "
                  f"Actual: {actual_label:11s} | "
                  f"Conf: {confidence:.3f} | "
                  f"Score: +{score_awarded:3d} | "
                  f"Time: {total_time:.2f}s")
            
            # Print probabilities for debugging (only if wrong)
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
    
    def run(self, max_iterations: int = None, delay: float = 1.0):
        """
        Run the challenge bot with smart timing
        
        Args:
            max_iterations: Maximum number of iterations (None = infinite)
            delay: Base delay between challenges in seconds
        """
        print("="*60)
        print("CLEAN CHALLENGE BOT - STARTING")
        print("="*60)
        print(f"Max iterations: {max_iterations if max_iterations else 'Infinite'}")
        print(f"Base delay: {delay}s")
        print(f"Pre-sync mode: Check every 1s until first score > 0")
        print(f"Synced mode: Wait 100s cycle after score received")
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
                    
                    # Use base delay
                    if delay > 0:
                        time.sleep(delay)
                else:
                    consecutive_failures += 1
                    
                    # If we're waiting for a new challenge (duplicate detected)
                    if self.wait_for_new_challenge:
                        # BEFORE first score: Check frequently (every 1s) until we get initial sync
                        if not self.first_score_received:
                            check_interval = 1.0
                            print(f"🔍 Checking for new challenge in {check_interval:.0f}s (pre-sync mode)...")
                            time.sleep(check_interval)
                        else:
                            # AFTER first score: Use smart 100s wait based on timing
                            wait_time = 100.0  # Wait 100 seconds for new challenge
                            
                            # Calculate time since last score
                            if self.last_score_time:
                                time_since_score = time.time() - self.last_score_time
                                remaining_wait = max(0, wait_time - time_since_score)
                            else:
                                remaining_wait = wait_time
                            
                            if remaining_wait > 0:
                                print(f"⏳ Waiting {remaining_wait:.0f}s for new challenge (synced with server)...")
                                time.sleep(remaining_wait)
                            else:
                                print(f"⏳ Server cycle complete, checking for new challenge...")
                        
                        self.wait_for_new_challenge = False  # Reset flag
                    
                    # If multiple consecutive failures, increase delay
                    elif consecutive_failures >= 3:
                        backoff_delay = min(delay * (2 ** (consecutive_failures - 2)), 30.0)
                        print(f"⚠️  Multiple failures detected, backing off for {backoff_delay:.1f}s...")
                        time.sleep(backoff_delay)
                    else:
                        # Normal retry with base delay
                        time.sleep(delay)
        
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
    parser.add_argument('--api-url', type=str, default='https://edth.helsing.codes',
                        help='Challenge API base URL')
    parser.add_argument('--api-token', type=str, default='9726345a-34ed-4995-94d9-ecc239b47c1d',
                        help='API authentication token')
    parser.add_argument('--max-iterations', type=int, default=None,
                        help='Maximum number of iterations (default: infinite)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between challenges in seconds (default: 0.5 for speed)')
    
    args = parser.parse_args()
    
    # Auto-detect best model if not specified
    if args.model is None:
        # Priority: panns_final.pt (trained) > best_model.pt (training)
        final_model = Path('models/panns_final.pt')
        best_model = Path('models/best_model.pt')
        
        if final_model.exists():
            args.model = str(final_model)
            print(f"✓ Using final trained model: {args.model}")
        elif best_model.exists():
            args.model = str(best_model)
            print(f"⚡ Using best checkpoint (training in progress): {args.model}")
            print(f"   Will automatically use panns_final.pt when training completes")
        else:
            print("❌ Error: No model found!")
            print("\nSearched for:")
            print(f"  - {final_model} (final trained model)")
            print(f"  - {best_model} (best checkpoint)")
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
        # Try labels_current.json (correct 3-class), then labels.json
        labels_current = Path('models/labels_current.json')
        labels_default = Path('models/labels.json')
        
        if labels_current.exists():
            args.labels = str(labels_current)
        elif labels_default.exists():
            args.labels = str(labels_default)
        else:
            print("❌ Error: No labels file found!")
            print("\nSearched for:")
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
