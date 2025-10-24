"""
Enhanced Challenge Bot with FFT + CNN + DNN Architecture
Uses FFT feature extraction, CNN feature learning, and DNN classification
Stores results and audio samples for continuous improvement
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import librosa
import numpy as np
import json
import time
from pathlib import Path
import tempfile
from datetime import datetime
from adrone.models.fft_cnn_dnn import FFTCNNDNNFusion
from adrone.features.fft_processor import FFTProcessor
from adrone.serve.challenge_handler import (
    ChallengeResultStorage,
    ChallengeAPIClient,
    AdaptiveLearningTracker
)

# Configuration
API_BASE_URL = "https://edth.helsing.codes"
API_TOKEN = "9726345a-34ed-4995-94d9-ecc239b47c1d"
MODEL_PATH = "models/cnn_edth_3class_improved.pt"
LABELS_PATH = "models/labels_edth_3class_improved.json"
STORAGE_DIR = "challenge_results"

# EMERGENCY: Model is broken, use heuristic-based classification
USE_HEURISTIC = True  # Set to False once model is retrained

# Audio processing parameters
SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
MAX_DURATION = 2.0  # seconds


class FFTCNNDNNChallengeBot:
    """Enhanced challenge bot using FFT + CNN + DNN architecture"""
    
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        labels_path: str = LABELS_PATH,
        storage_dir: str = STORAGE_DIR,
        api_base_url: str = API_BASE_URL,
        api_token: str = API_TOKEN
    ):
        self.model_path = model_path
        self.labels_path = labels_path
        
        # Initialize components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model and labels
        self.model, self.labels = self._load_model_and_labels()
        
        # Initialize FFT processor
        self.fft_processor = FFTProcessor(
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            sample_rate=SAMPLE_RATE
        )
        
        # Initialize storage and API client
        self.storage = ChallengeResultStorage(storage_dir)
        self.api_client = ChallengeAPIClient(api_base_url, api_token)
        self.learning_tracker = AdaptiveLearningTracker()
        
        print(f"✓ Model loaded: {model_path}")
        print(f"✓ Classes: {self.labels}")
        print(f"✓ Results storage: {storage_dir}")
    
    def _load_model_and_labels(self):
        """Load trained model and label mapping"""
        # Load labels
        with open(self.labels_path, 'r') as f:
            labels_data = json.load(f)
            labels = labels_data['labels']
        
        # Create model
        model = FFTCNNDNNFusion(
            n_classes=len(labels),
            in_channels=1,
            cnn_feature_dim=512,
            dnn_hidden_dims=[256, 128]
        )
        
        # Load weights
        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        return model, labels
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        Preprocess audio using FFT feature extraction
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=MAX_DURATION)
        
        # Pad or truncate to fixed length
        target_length = int(SAMPLE_RATE * MAX_DURATION)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]
        
        # Extract FFT features (mel spectrogram)
        tensor = self.fft_processor.extract_features_for_model(y)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor
    
    def classify_audio(self, audio_path: str):
        """
        Classify audio file using FFT + CNN + DNN pipeline
        
        Returns:
            prediction, confidence, all_probabilities, inference_time
        """
        start_time = time.time()
        
        # Preprocess with FFT
        audio_tensor = self.preprocess_audio(audio_path)
        
        # Inference
        with torch.no_grad():
            logits = self.model(audio_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_idx].item()
        
        inference_time = time.time() - start_time
        
        prediction = self.labels[predicted_idx]
        all_probs = {
            label: probabilities[0, idx].item()
            for idx, label in enumerate(self.labels)
        }
        
        return prediction, confidence, all_probs, inference_time
    
    def run_single_challenge(self):
        """Run a single challenge iteration"""
        try:
            # Fetch challenge (silent for speed)
            challenge = self.api_client.get_challenge()
            challenge_id = challenge['challenge_id']
            wav_url = challenge['wav_url']
            
            # Download audio (silent for speed)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            self.api_client.download_audio(wav_url, tmp_path)
            
            # Classify with FFT + CNN + DNN (silent for speed)
            prediction, confidence, all_probs, inference_time = self.classify_audio(tmp_path)
            
            # CRITICAL: Submit immediately without any prints for fastest upload!
            result = self.api_client.submit_classification(challenge_id, prediction)
            
            # Now print results AFTER submission
            # Debug: Show all probabilities to diagnose bias
            all_probs_str = " | ".join([f"{k}:{v:.1%}" for k, v in all_probs.items()])
            print(f"📤 {prediction.upper()} [{confidence:.2%}] ({all_probs_str}) -> ", end="")
            
            # Process result
            score_awarded = result.get('score_awarded', 0)
            total_score = result.get('total_score', 0)
            
            # Correct prediction = positive score awarded, Wrong = 0 or negative score
            is_correct = score_awarded > 0
            
            # Print result inline
            if is_correct:
                print(f"✅ CORRECT! +{score_awarded} (Total: {total_score}) [{inference_time:.3f}s]")
            else:
                print(f"❌ WRONG (Total: {total_score}) [{inference_time:.3f}s]")
            
            # Store result (silent for speed)
            self.storage.store_result(
                challenge_id=challenge_id,
                audio_path=tmp_path,
                prediction=prediction,
                confidence=confidence,
                all_probabilities=all_probs,
                is_correct=is_correct,
                score_awarded=score_awarded,
                inference_time=inference_time,
                response_data=result
            )
            
            # Update learning tracker
            self.learning_tracker.add_result(prediction, confidence, is_correct)
            
            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self, max_iterations: int = None, delay: float = 1.0):
        """
        Run challenge bot
        
        Args:
            max_iterations: Maximum challenges to attempt (None = infinite)
            delay: Seconds to wait between challenges
        """
        print("🤖 FFT+CNN+DNN Bot | API: edth.helsing.codes | Storage: challenge_results")
        print("=" * 80)
        
        iteration = 0
        
        try:
            while max_iterations is None or iteration < max_iterations:
                iteration += 1
                print(f"#{iteration:03d} ", end="", flush=True)
                
                success = self.run_single_challenge()
                
                if not success:
                    print(f"   ⚠ Retrying in {delay}s...")
                
                # Show stats every 10 iterations
                if iteration % 10 == 0:
                    stats = self.storage.get_statistics()
                    print(f"      📊 Stats: {stats['correct']}/{stats['total_attempts']} correct ({stats['accuracy']*100:.1f}%), Score: {stats['total_score']}")
                
                # Wait before next challenge
                if max_iterations is None or iteration < max_iterations:
                    time.sleep(delay)
        
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        
        # Final summary
        print("\n" + "="*80)
        print("🏁 Session Complete!")
        stats = self.storage.get_statistics()
        print(f"   Total: {stats['total_attempts']} | Correct: {stats['correct']} | Accuracy: {stats['accuracy']*100:.1f}% | Score: {stats['total_score']}")
        print("="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced Challenge Bot with FFT + CNN + DNN"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum number of challenges (default: infinite)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between challenges in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help=f"Path to model checkpoint (default: {MODEL_PATH})"
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=LABELS_PATH,
        help=f"Path to labels JSON (default: {LABELS_PATH})"
    )
    parser.add_argument(
        "--storage-dir",
        type=str,
        default=STORAGE_DIR,
        help=f"Directory for storing results (default: {STORAGE_DIR})"
    )
    
    args = parser.parse_args()
    
    # Create bot
    bot = FFTCNNDNNChallengeBot(
        model_path=args.model,
        labels_path=args.labels,
        storage_dir=args.storage_dir
    )
    
    # Run
    bot.run(max_iterations=args.max_iterations, delay=args.delay)


if __name__ == "__main__":
    main()
