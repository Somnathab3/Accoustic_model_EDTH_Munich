"""
Deep CNN Architecture for Acoustic Drone Detection
State-of-the-art architecture with:
- Residual connections for better gradient flow
- Attention mechanisms for feature importance
- Batch normalization and dropout for regularization
- Multi-scale feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AttentionBlock(nn.Module):
    """
    Channel attention mechanism to emphasize important features
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """
    Residual block with batch normalization and dropout
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.3):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(dropout)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = F.relu(out)
        
        return out


class MultiScaleBlock(nn.Module):
    """
    Multi-scale feature extraction using different kernel sizes
    Captures both fine-grained and coarse patterns
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        branch_channels = out_channels // 4
        
        # Branch 1: 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: 5x5 conv (replaced with two 3x3)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 4: max pooling + conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        # Concatenate all branches
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class DeepDroneDetectorCNN(nn.Module):
    """
    Deep CNN for acoustic drone detection
    
    Architecture:
    - Initial feature extraction with multi-scale convolutions
    - Multiple residual blocks with increasing depth
    - Channel attention mechanisms
    - Global pooling and fully connected layers
    - Dropout for regularization
    
    Input: [batch, 3, height, width] - 3 channels of acoustic features
    Output: [batch, num_classes] - class probabilities
    """
    
    def __init__(self, num_classes: int = 3, dropout: float = 0.5):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Initial convolution - 3 input channels (mel, mfcc, spectral)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Multi-scale feature extraction
        self.multiscale1 = MultiScaleBlock(64, 128)
        
        # Residual blocks with attention
        self.layer1 = self._make_layer(128, 128, num_blocks=2, stride=1, dropout=dropout)
        self.attention1 = AttentionBlock(128)
        
        self.layer2 = self._make_layer(128, 256, num_blocks=2, stride=2, dropout=dropout)
        self.attention2 = AttentionBlock(256)
        
        self.layer3 = self._make_layer(256, 512, num_blocks=2, stride=2, dropout=dropout)
        self.attention3 = AttentionBlock(512)
        
        # Multi-scale feature extraction at deeper level
        self.multiscale2 = MultiScaleBlock(512, 512)
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512 * 2, 512),  # *2 because we concat avg and max pooling
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.25),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                   num_blocks: int, stride: int, dropout: float):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, dropout))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, dropout))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial feature extraction
        x = self.conv1(x)  # [B, 64, H/4, W/4]
        
        # Multi-scale features
        x = self.multiscale1(x)  # [B, 128, H/4, W/4]
        
        # Residual blocks with attention
        x = self.layer1(x)  # [B, 128, H/4, W/4]
        x = self.attention1(x)
        
        x = self.layer2(x)  # [B, 256, H/8, W/8]
        x = self.attention2(x)
        
        x = self.layer3(x)  # [B, 512, H/16, W/16]
        x = self.attention3(x)
        
        # Deep multi-scale features
        x = self.multiscale2(x)  # [B, 512, H/16, W/16]
        
        # Global pooling (both avg and max for richer representation)
        avg_pool = self.global_avg_pool(x).view(x.size(0), -1)
        max_pool = self.global_max_pool(x).view(x.size(0), -1)
        x = torch.cat([avg_pool, max_pool], dim=1)  # [B, 1024]
        
        # Classification
        x = self.fc(x)  # [B, num_classes]
        
        return x
    
    def get_feature_maps(self, x):
        """
        Extract intermediate feature maps for visualization
        """
        features = {}
        
        x = self.conv1(x)
        features['conv1'] = x
        
        x = self.multiscale1(x)
        features['multiscale1'] = x
        
        x = self.layer1(x)
        x = self.attention1(x)
        features['layer1'] = x
        
        x = self.layer2(x)
        x = self.attention2(x)
        features['layer2'] = x
        
        x = self.layer3(x)
        x = self.attention3(x)
        features['layer3'] = x
        
        return features


class LightweightDroneDetectorCNN(nn.Module):
    """
    Lightweight version for faster training and inference
    Good for testing and rapid prototyping
    """
    
    def __init__(self, num_classes: int = 3, dropout: float = 0.4):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            # Global pooling
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def test_model():
    """Test the model architecture"""
    batch_size = 4
    num_classes = 3
    
    # Test with typical spectrogram dimensions
    # [batch, channels, height, width]
    x = torch.randn(batch_size, 3, 128, 130)  # 3 channels, 128 mel bins, ~130 time frames
    
    print("Testing DeepDroneDetectorCNN...")
    model = DeepDroneDetectorCNN(num_classes=num_classes)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nTesting LightweightDroneDetectorCNN...")
    model_light = LightweightDroneDetectorCNN(num_classes=num_classes)
    output_light = model_light(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_light.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model_light.parameters()):,}")


if __name__ == "__main__":
    test_model()
