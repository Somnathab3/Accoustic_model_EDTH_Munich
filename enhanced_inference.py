"""
Enhanced inference module for LIGO-modified matched filter bank model
Supports both baseline CRNN and enhanced matched bank models
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import json

from src.adrone.preprocessing import AudioPreprocessor
from src.adrone.models.acoustic_models import CRNNWithAttention
from src.models.enhanced_models_with_bank import create_enhanced_crnn


class EnhancedCRNN(torch.nn.Module):
    """Enhanced CRNN backbone that accepts 9 input channels"""
    def __init__(self, num_classes: int = 3, input_channels: int = 9, n_mels: int = 96, dropout: float = 0.3):
        super().__init__()
        
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        
        self.gru = torch.nn.GRU(
            input_size=128 * (n_mels // 8),
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        batch, channels, freq, time = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch, time, -1)
        
        x, _ = self.gru(x)
        x = x.mean(dim=1)
        
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class EnhancedAcousticClassifier:
    """
    Unified classifier supporting both baseline and enhanced matched bank models
    
    Automatically detects model type and applies correct preprocessing
    """
    
    def __init__(
        self,
        model_path: str,
        labels_path: str,
        device: str = 'auto',
        use_hpss: bool = True
    ):
        """
        Args:
            model_path: Path to model checkpoint
            labels_path: Path to labels JSON
            device: 'auto', 'cuda', or 'cpu'
            use_hpss: Whether to use HPSS (always True for our models)
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
        
        # Initialize preprocessor (always use HPSS for 3 channels)
        self.preprocessor = AudioPreprocessor(
            sample_rate=16000,
            n_fft=512,
            hop_length=160,
            n_mels=96,
            f_min=20,
            f_max=8000,
            window_duration=2.0,
            use_hpss=True  # Always True for our models
        )
        
        # Load model and detect type
        self.model, self.is_enhanced = self._load_model(model_path)
        self.model.eval()
        
        print(f"✓ Loaded model from {Path(model_path).name}")
        print(f"  Type: {'Enhanced (Matched Filter Bank)' if self.is_enhanced else 'Baseline CRNN'}")
        print(f"  Device: {self.device}")
        print(f"  Classes: {list(self.class_to_idx.keys())}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Parameters: {total_params:,}")
    
    def _load_model(self, model_path: str) -> Tuple[torch.nn.Module, bool]:
        """
        Load model and detect if it's enhanced with matched bank
        
        Returns:
            (model, is_enhanced)
        """
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        # Check if this is an enhanced model with matched filter bank
        # Enhanced models have 'bank' keys in state_dict
        is_enhanced = any('bank' in key for key in state_dict.keys())
        
        if is_enhanced:
            print("  🔬 Detected: Enhanced model with LIGO-style matched filter bank")
            
            # Create enhanced backbone (9 input channels: 3 original + 6 from bank)
            enhanced_backbone = EnhancedCRNN(
                num_classes=self.num_classes,
                input_channels=9,  # 3 HPSS + 6 compressed bank
                n_mels=96,
                dropout=0.3
            )
            
            # Wrap with matched filter bank
            model = create_enhanced_crnn(
                crnn_backbone=enhanced_backbone,
                n_mels=96,
                sr=16000,
                compression=6,  # 75 templates → 6 channels
                trainable_bank=True
            )
        else:
            print("  📊 Detected: Baseline CRNN model")
            
            # Create baseline CRNN (3 input channels: HPSS)
            model = CRNNWithAttention(
                num_classes=self.num_classes,
                input_channels=3,
                n_mels=96,
                dropout=0.3
            )
        
        # Load weights
        model.load_state_dict(state_dict)
        model.to(self.device)
        
        return model, is_enhanced
    
    def classify(self, audio_path: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify an audio file
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            (prediction, confidence, all_probabilities)
        """
        # Load and preprocess audio
        waveform = self.preprocessor.load_audio(audio_path)
        
        # Convert to tensor and preprocess
        waveform_tensor = torch.from_numpy(waveform).unsqueeze(0).float()
        spectrogram = self.preprocessor(waveform_tensor)
        
        # Add batch dimension and move to device
        spectrogram = spectrogram.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(spectrogram)
            probabilities = F.softmax(logits, dim=1)
            
            predicted_idx = probabilities.argmax(dim=1).item()
            confidence = probabilities[0, predicted_idx].item()
            prediction = self.idx_to_class[predicted_idx]
            
            all_probs = {
                self.idx_to_class[i]: probabilities[0, i].item()
                for i in range(self.num_classes)
            }
        
        return prediction, confidence, all_probs
    
    def classify_from_tensor(self, waveform: torch.Tensor) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify from pre-loaded waveform tensor
        
        Args:
            waveform: Audio waveform tensor (1D or 2D)
        
        Returns:
            (prediction, confidence, all_probabilities)
        """
        # Ensure 2D: (batch, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Preprocess
        spectrogram = self.preprocessor(waveform)
        
        # Add batch dimension if needed
        if spectrogram.dim() == 3:
            spectrogram = spectrogram.unsqueeze(0)
        
        spectrogram = spectrogram.to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(spectrogram)
            probabilities = F.softmax(logits, dim=1)
            
            predicted_idx = probabilities.argmax(dim=1).item()
            confidence = probabilities[0, predicted_idx].item()
            prediction = self.idx_to_class[predicted_idx]
            
            all_probs = {
                self.idx_to_class[i]: probabilities[0, i].item()
                for i in range(self.num_classes)
            }
        
        return prediction, confidence, all_probs


# Maintain backward compatibility with old classifier name
AcousticDroneClassifier = EnhancedAcousticClassifier
