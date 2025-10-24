"""
FFT + CNN + DNN Fusion Model
Complete pipeline: FFT features -> CNN feature extraction -> DNN classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """Residual block with batch normalization"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ChannelAttention(nn.Module):
    """Channel attention module"""
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average pool
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        # Max pool
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out.expand_as(x)


class CNNFeatureExtractor(nn.Module):
    """
    CNN for extracting features from FFT-preprocessed audio
    This CNN outputs feature vectors instead of final classifications
    """
    def __init__(self, in_channels: int = 1, feature_dim: int = 512):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Residual blocks
        self.res_block1 = ResidualBlock(32, 64, stride=2)
        self.attention1 = ChannelAttention(64)
        
        self.res_block2 = ResidualBlock(64, 128, stride=2)
        self.attention2 = ChannelAttention(128)
        
        self.res_block3 = ResidualBlock(128, 256, stride=2)
        self.attention3 = ChannelAttention(256)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.feature_dim = feature_dim
    
    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        
        # Residual blocks with attention
        x = self.res_block1(x)
        x = self.attention1(x)
        
        x = self.res_block2(x)
        x = self.attention2(x)
        
        x = self.res_block3(x)
        x = self.attention3(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Project to feature space
        features = self.feature_proj(x)
        
        return features


class DNNClassifier(nn.Module):
    """
    Deep Neural Network classifier that takes CNN features and makes final prediction
    """
    def __init__(self, feature_dim: int, n_classes: int, hidden_dims: list = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        layers = []
        in_dim = feature_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, n_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)


class FFTFeatureExtractor(nn.Module):
    """
    FFT-based statistical feature extractor
    Extracts handcrafted frequency domain features
    """
    def __init__(self, n_fft: int = 2048, sample_rate: int = 16000, feature_dim: int = 256):
        super().__init__()
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        
        # Project FFT features to feature space
        # We'll extract ~50 statistical features from FFT
        self.feature_proj = nn.Sequential(
            nn.Linear(50, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )
        
        self.feature_dim = feature_dim
    
    def extract_fft_statistics(self, mel_spec):
        """
        Extract statistical features from mel spectrogram
        
        Args:
            mel_spec: (B, 1, H, W) mel spectrogram tensor
        Returns:
            (B, 50) statistical features
        """
        batch_size = mel_spec.size(0)
        mel_spec = mel_spec.squeeze(1)  # (B, H, W)
        
        features = []
        
        # Temporal statistics (across time axis)
        features.append(mel_spec.mean(dim=2))  # (B, H) - mean over time
        features.append(mel_spec.std(dim=2))   # (B, H) - std over time
        features.append(mel_spec.max(dim=2)[0])  # (B, H) - max over time
        features.append(mel_spec.min(dim=2)[0])  # (B, H) - min over time
        
        # Frequency statistics (across frequency axis)
        features.append(mel_spec.mean(dim=1))  # (B, W) - mean over freq
        features.append(mel_spec.std(dim=1))   # (B, W) - std over freq
        
        # Flatten all features and take first 50 dims
        all_features = torch.cat(features, dim=1)  # (B, H*4 + W*2)
        
        # Adaptive pooling to get exactly 50 features
        if all_features.size(1) > 50:
            all_features = F.adaptive_avg_pool1d(
                all_features.unsqueeze(1), 50
            ).squeeze(1)
        elif all_features.size(1) < 50:
            padding = torch.zeros(batch_size, 50 - all_features.size(1), 
                                device=all_features.device)
            all_features = torch.cat([all_features, padding], dim=1)
        
        return all_features
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) mel spectrogram from FFT preprocessing
        Returns:
            (B, feature_dim) FFT-based features
        """
        # Extract statistical features from mel spectrogram
        fft_stats = self.extract_fft_statistics(x)
        
        # Project to feature space
        features = self.feature_proj(fft_stats)
        
        return features


class FFTCNNDNNFusion(nn.Module):
    """
    Complete FFT + CNN + DNN fusion model with PARALLEL architecture
    
    Pipeline:
    1. Input (raw audio) → FFT preprocessing → Mel Spectrogram
    2a. Mel Spectrogram → FFT Feature Extractor → FFT Features (256-dim)
    2b. Mel Spectrogram → CNN Feature Extractor → CNN Features (512-dim)  
    3. Concatenate [FFT Features | CNN Features] → Combined Features (768-dim)
    4. Combined Features → DNN Classifier → Final Prediction
    
    Key Innovation: Both FFT statistical features AND CNN learned features
    are computed in parallel, then fused for classification.
    """
    def __init__(
        self,
        n_classes: int,
        in_channels: int = 1,
        fft_feature_dim: int = 256,
        cnn_feature_dim: int = 512,
        dnn_hidden_dims: list = None
    ):
        super().__init__()
        
        # FFT feature extractor (handcrafted frequency features)
        self.fft_extractor = FFTFeatureExtractor(
            n_fft=2048,
            sample_rate=16000,
            feature_dim=fft_feature_dim
        )
        
        # CNN feature extractor (learned spatial-temporal features)
        self.cnn = CNNFeatureExtractor(
            in_channels=in_channels,
            feature_dim=cnn_feature_dim
        )
        
        # Fusion dimension
        fusion_dim = fft_feature_dim + cnn_feature_dim
        
        # DNN classifier (works on fused features)
        self.dnn = DNNClassifier(
            feature_dim=fusion_dim,
            n_classes=n_classes,
            hidden_dims=dnn_hidden_dims
        )
        
        self.n_classes = n_classes
        self.fft_feature_dim = fft_feature_dim
        self.cnn_feature_dim = cnn_feature_dim
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) mel spectrogram from FFT preprocessing
        Returns:
            logits: (B, n_classes) class logits
        """
        # Parallel feature extraction
        fft_features = self.fft_extractor(x)  # (B, fft_feature_dim)
        cnn_features = self.cnn(x)             # (B, cnn_feature_dim)
        
        # Fuse features by concatenation
        fused_features = torch.cat([fft_features, cnn_features], dim=1)  # (B, fusion_dim)
        
        # Classify with DNN
        logits = self.dnn(fused_features)
        
        return logits
    
    def extract_features(self, x):
        """Extract both FFT and CNN features separately"""
        with torch.no_grad():
            fft_features = self.fft_extractor(x)
            cnn_features = self.cnn(x)
            fused_features = torch.cat([fft_features, cnn_features], dim=1)
        return fused_features, fft_features, cnn_features
    
    def predict_from_features(self, features):
        """Predict from pre-extracted fused features"""
        with torch.no_grad():
            logits = self.dnn(features)
        return logits


class MultiScaleCNNDNN(nn.Module):
    """
    Multi-scale CNN with DNN fusion
    Processes audio at multiple temporal scales
    """
    def __init__(self, n_classes: int, in_channels: int = 1):
        super().__init__()
        
        # Scale 1: Fine-grained (small kernels)
        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Scale 2: Medium (medium kernels)
        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Scale 3: Coarse (large kernels)
        self.scale3 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Fusion of scales
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # DNN classifier
        self.dnn = DNNClassifier(
            feature_dim=256,
            n_classes=n_classes,
            hidden_dims=[256, 128, 64]
        )
    
    def forward(self, x):
        # Multi-scale processing
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        
        # Concatenate scales
        x = torch.cat([s1, s2, s3], dim=1)
        
        # Fusion
        x = self.fusion_conv(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        logits = self.dnn(x)
        
        return logits


# Backward compatibility - keep existing model classes
class CNNImproved(nn.Module):
    """
    Original improved CNN (kept for backward compatibility)
    """
    def __init__(self, n_classes: int):
        super().__init__()
        
        # Use the new fusion model internally
        self.fusion_model = FFTCNNDNNFusion(
            n_classes=n_classes,
            in_channels=1,
            cnn_feature_dim=512,
            dnn_hidden_dims=[256, 128]
        )
    
    def forward(self, x):
        return self.fusion_model(x)


class CNNLarge(nn.Module):
    """
    Larger CNN for better performance (kept for backward compatibility)
    """
    def __init__(self, n_classes: int):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
