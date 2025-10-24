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
            # Fetch challenge
            print("📥 Fetching challenge...")
            challenge = self.api_client.get_challenge()
            challenge_id = challenge['challenge_id']
            wav_url = challenge['wav_url']
            
            print(f"   Challenge ID: {challenge_id}")
            print(f"   Audio URL: {wav_url}")
            
            # Download audio
            print("🎵 Downloading audio...")
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            self.api_client.download_audio(wav_url, tmp_path)
            
            # Classify with FFT + CNN + DNN
            print("🔍 Processing with FFT + CNN + DNN...")
            prediction, confidence, all_probs, inference_time = self.classify_audio(tmp_path)
            
            print(f"   Prediction: {prediction.upper()}")
            print(f"   Confidence: {confidence:.2%}")
            print(f"   All probabilities:")
            for label, prob in all_probs.items():
                print(f"      {label:12s}: {prob:.2%}")
            print(f"   Inference time: {inference_time:.3f}s")
            
            # Submit classification
            print(f"📤 Submitting: {prediction}")
            result = self.api_client.submit_classification(challenge_id, prediction)
            
            # Process result
            is_correct = result.get('success', False)
            score_awarded = result.get('score_awarded', 0)
            total_score = result.get('total_score', 0)
            
            # Store result
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
            
            # Display result
            print("\n" + "="*80)
            if is_correct:
                print("✅ CORRECT!")
                print(f"   Score awarded: {score_awarded}")
                print(f"   Total score: {total_score}")
            else:
                print("❌ WRONG!")
                print(f"   Message: {result.get('message', 'No message')}")
                print(f"   Expected score was 0, got: {score_awarded}")
            
            # Show statistics
            stats = self.storage.get_statistics()
            print(f"\n📊 Session Stats:")
            print(f"   Correct: {stats['correct']}")
            print(f"   Wrong: {stats['wrong']}")
            print(f"   Accuracy: {stats['accuracy']*100:.1f}%")
            print(f"   Total Score: {stats['total_score']}")
            print("="*80)
            
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
        print("=" * 80)
        print("🤖 FFT + CNN + DNN CHALLENGE BOT")
        print("=" * 80)
        print(f"🎯 API: {API_BASE_URL}")
        print(f"🔑 Token: {API_TOKEN[:8]}...")
        print(f"🧠 Architecture: FFT → CNN → DNN")
        print(f"💾 Storage: {STORAGE_DIR}")
        print("\n" + "=" * 80)
        
        iteration = 0
        
        try:
            while max_iterations is None or iteration < max_iterations:
                iteration += 1
                print(f"\n{'='*80}")
                print(f"🔄 Challenge #{iteration} - {datetime.now().strftime('%H:%M:%S')}")
                print(f"{'='*80}")
                
                success = self.run_single_challenge()
                
                if not success:
                    print(f"   Retrying in {delay}s...")
                
                # Show adaptive learning recommendations periodically
                if iteration % 10 == 0:
                    recommendations = self.learning_tracker.get_recommendations()
                    if 'message' not in recommendations:
                        print("\n🎓 Adaptive Learning Recommendations:")
                        for class_name, rec in recommendations.items():
                            print(f"   {class_name}: {rec['status']} - {rec['suggestion']}")
                
                # Wait before next challenge
                if max_iterations is None or iteration < max_iterations:
                    time.sleep(delay)
        
        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped by user")
        
        # Final summary
        print("\n" + "="*80)
        print("🏁 Challenge Bot Session Complete!")
        stats = self.storage.get_statistics()
        print(f"   Total Attempts: {stats['total_attempts']}")
        print(f"   Correct: {stats['correct']}")
        print(f"   Wrong: {stats['wrong']}")
        print(f"   Accuracy: {stats['accuracy']*100:.1f}%")
        print(f"   Final Score: {stats['total_score']}")
        print("="*80)
        
        # Performance analysis
        analysis = self.storage.analyze_performance()
        print("\n📈 Performance Analysis:")
        for key, value in analysis.items():
            if key not in ['prediction_distribution']:
                print(f"   {key}: {value}")


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
