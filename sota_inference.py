"""
Clean, optimized inference module for real-time acoustic drone detection
Implements sliding window inference with exponential smoothing and hysteresis
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import json
from collections import deque

from src.adrone.preprocessing import AudioPreprocessor
from src.adrone.models.acoustic_models import create_model


class AcousticDroneClassifier:
    """
    Real-time acoustic drone classifier
    
    Features:
        - Fast single-file inference
        - Exponential smoothing for stable predictions
        - Hysteresis for alert triggering
        - Calibrated confidence scores
    """
    
    def __init__(
        self,
        model_path: str,
        labels_path: str,
        device: str = 'auto',
        use_hpss: bool = True,
        confidence_threshold: float = 0.7
    ):
        """
        Args:
            model_path: Path to model checkpoint
            labels_path: Path to labels JSON
            device: 'auto', 'cuda', or 'cpu'
            use_hpss: Whether model uses HPSS (3 channels)
            confidence_threshold: Minimum confidence for predictions
        """
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load labels
        with open(labels_path, 'r') as f:
            label_data = json.load(f)
            self.class_to_idx = label_data['class_to_idx']
            self.idx_to_class = {int(k): v for k, v in label_data['idx_to_class'].items()}
        
        self.num_classes = len(self.class_to_idx)
        self.confidence_threshold = confidence_threshold
        
        # Initialize preprocessor
        self.preprocessor = AudioPreprocessor(
            sample_rate=16000,
            n_fft=1024,
            hop_length=320,
            n_mels=96,
            window_duration=2.0,
            use_hpss=use_hpss
        )
        
        # Load model
        self.model = self._load_model(model_path, use_hpss)
        self.model.eval()
        
        print(f"✓ Loaded model from {model_path}")
        print(f"  Device: {self.device}")
        print(f"  Classes: {list(self.class_to_idx.keys())}")
    
    def _load_model(self, model_path: str, use_hpss: bool = True):
        """Load model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model config
        if 'model_type' in checkpoint:
            model_type = checkpoint['model_type']
            input_channels = checkpoint.get('input_channels', 3 if use_hpss else 1)
            n_mels = checkpoint.get('n_mels', 96)
        else:
            # Legacy checkpoint format - auto-detect from state_dict keys
            state_dict_keys = list(checkpoint['model_state_dict'].keys())
            
            # Check for CRNN-specific keys
            if any('gru' in key for key in state_dict_keys) or any('attention' in key for key in state_dict_keys):
                model_type = 'crnn'
                print("  Auto-detected CRNN model from state_dict keys")
            else:
                model_type = 'panns'
                print("  Auto-detected PANNs model from state_dict keys")
            
            input_channels = 3 if use_hpss else 1
            n_mels = 96
        
        print(f"  Model type: {model_type.upper()}")
        print(f"  Input channels: {input_channels}")
        print(f"  N_mels: {n_mels}")
        
        # Create model with correct parameters
        if model_type == 'crnn':
            model = create_model(
                model_type=model_type,
                num_classes=self.num_classes,
                input_channels=input_channels,
                n_mels=n_mels
            )
        else:
            model = create_model(
                model_type=model_type,
                num_classes=self.num_classes,
                input_channels=input_channels
            )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model
    
    @torch.no_grad()
    def classify(
        self,
        audio_path: str,
        return_all_probs: bool = True
    ) -> Tuple[str, float, Optional[Dict[str, float]]]:
        """
        Classify a single audio file
        
        Args:
            audio_path: Path to audio file
            return_all_probs: Whether to return all class probabilities
        
        Returns:
            prediction: Predicted class name
            confidence: Confidence score (0-1)
            all_probs: Dictionary of all class probabilities (optional)
        """
        # Load and preprocess audio
        waveform = self.preprocessor.load_audio(audio_path)
        spectrogram = self.preprocessor(waveform)
        
        # Add batch dimension
        spectrogram = spectrogram.unsqueeze(0).to(self.device)
        
        # Inference
        logits = self.model(spectrogram)
        probabilities = F.softmax(logits, dim=1)
        
        # Get prediction
        predicted_idx = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, predicted_idx].item()
        prediction = self.idx_to_class[predicted_idx]
        
        # Get all probabilities if requested
        all_probs = None
        if return_all_probs:
            all_probs = {
                self.idx_to_class[i]: probabilities[0, i].item()
                for i in range(self.num_classes)
            }
        
        return prediction, confidence, all_probs
    
    def classify_batch(
        self,
        audio_paths: list
    ) -> list:
        """
        Classify multiple audio files in a batch
        
        Args:
            audio_paths: List of audio file paths
        
        Returns:
            List of (prediction, confidence, all_probs) tuples
        """
        results = []
        
        for audio_path in audio_paths:
            result = self.classify(audio_path, return_all_probs=True)
            results.append(result)
        
        return results


class StreamingClassifier:
    """
    Streaming classifier with temporal smoothing and hysteresis
    For real-time deployment with stable predictions
    """
    
    def __init__(
        self,
        classifier: AcousticDroneClassifier,
        smoothing_tau: float = 3.0,
        history_size: int = 5,
        alert_threshold: float = 0.9,
        clear_threshold: float = 0.3,
        min_consecutive: int = 2
    ):
        """
        Args:
            classifier: Base classifier instance
            smoothing_tau: Exponential smoothing time constant (seconds)
            history_size: Number of recent predictions to keep
            alert_threshold: Threshold to raise alert
            clear_threshold: Threshold to clear alert
            min_consecutive: Minimum consecutive predictions to trigger alert
        """
        self.classifier = classifier
        self.smoothing_tau = smoothing_tau
        self.history_size = history_size
        self.alert_threshold = alert_threshold
        self.clear_threshold = clear_threshold
        self.min_consecutive = min_consecutive
        
        # State
        self.smoothed_probs = None
        self.history = deque(maxlen=history_size)
        self.alert_state = {class_name: False for class_name in classifier.class_to_idx.keys()}
    
    def update(self, audio_path: str) -> Dict:
        """
        Process new audio window and update state
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Dictionary with prediction, confidence, alerts, and smoothed probabilities
        """
        # Get current prediction
        prediction, confidence, all_probs = self.classifier.classify(audio_path)
        
        # Convert to tensor for smoothing
        current_probs = torch.tensor([
            all_probs[self.classifier.idx_to_class[i]]
            for i in range(self.classifier.num_classes)
        ])
        
        # Exponential smoothing
        if self.smoothed_probs is None:
            self.smoothed_probs = current_probs
        else:
            alpha = 1.0 / self.smoothing_tau
            self.smoothed_probs = alpha * current_probs + (1 - alpha) * self.smoothed_probs
        
        # Get smoothed prediction
        smoothed_idx = self.smoothed_probs.argmax().item()
        smoothed_prediction = self.classifier.idx_to_class[smoothed_idx]
        smoothed_confidence = self.smoothed_probs[smoothed_idx].item()
        
        # Update history
        self.history.append(smoothed_prediction)
        
        # Check for alerts with hysteresis
        alerts = {}
        for class_name in self.classifier.class_to_idx.keys():
            class_idx = self.classifier.class_to_idx[class_name]
            class_prob = self.smoothed_probs[class_idx].item()
            
            # Check consecutive predictions
            recent_count = sum(1 for p in self.history if p == class_name)
            
            # Trigger alert
            if not self.alert_state[class_name]:
                if (class_prob > self.alert_threshold and 
                    recent_count >= self.min_consecutive):
                    self.alert_state[class_name] = True
            
            # Clear alert
            else:
                if class_prob < self.clear_threshold:
                    self.alert_state[class_name] = False
            
            alerts[class_name] = self.alert_state[class_name]
        
        return {
            'instant_prediction': prediction,
            'instant_confidence': confidence,
            'smoothed_prediction': smoothed_prediction,
            'smoothed_confidence': smoothed_confidence,
            'smoothed_probabilities': {
                self.classifier.idx_to_class[i]: self.smoothed_probs[i].item()
                for i in range(self.classifier.num_classes)
            },
            'alerts': alerts,
            'alert_active': any(alerts.values())
        }
    
    def reset(self):
        """Reset streaming state"""
        self.smoothed_probs = None
        self.history.clear()
        self.alert_state = {class_name: False for class_name in self.classifier.class_to_idx.keys()}


def quick_inference(model_path: str, labels_path: str, audio_path: str):
    """Quick inference for single audio file"""
    classifier = AcousticDroneClassifier(
        model_path=model_path,
        labels_path=labels_path,
        device='auto'
    )
    
    prediction, confidence, all_probs = classifier.classify(audio_path)
    
    print(f"\nPrediction: {prediction}")
    print(f"Confidence: {confidence:.4f}")
    print(f"\nAll probabilities:")
    for class_name, prob in all_probs.items():
        print(f"  {class_name}: {prob:.4f}")
    
    return prediction, confidence


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python inference.py <model_path> <labels_path> <audio_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    labels_path = sys.argv[2]
    audio_path = sys.argv[3]
    
    quick_inference(model_path, labels_path, audio_path)
