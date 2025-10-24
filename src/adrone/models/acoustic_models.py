"""
State-of-the-art Deep CNN Models for Acoustic Drone Detection
Implements three architectures: CRNN-Attention, PANNs-CNN14, and Audio Transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class TemporalFrequencyAttention(nn.Module):
    """
    Temporal-Frequency Attention Module
    Learns to focus on discriminative time-frequency regions (rotor harmonics)
    From "Temporal-Frequency Attention for ESC" (Mu et al., 2021)
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        # Temporal attention
        self.temporal_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.temporal_fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )
        
        # Frequency attention
        self.frequency_pool = nn.AdaptiveAvgPool2d((1, None))
        self.frequency_fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, freq, time)
        Returns:
            Attention-weighted features
        """
        batch, c, f, t = x.shape
        
        # Temporal attention
        temp_pool = self.temporal_pool(x).squeeze(-1).permute(0, 2, 1)  # (batch, freq, channels)
        temp_attn = self.temporal_fc(temp_pool).permute(0, 2, 1).unsqueeze(-1)  # (batch, channels, freq, 1)
        
        # Frequency attention
        freq_pool = self.frequency_pool(x).squeeze(-2).permute(0, 2, 1)  # (batch, time, channels)
        freq_attn = self.frequency_fc(freq_pool).permute(0, 2, 1).unsqueeze(-2)  # (batch, channels, 1, time)
        
        # Combined attention
        attention = temp_attn * freq_attn
        
        return x * attention


class CRNNWithAttention(nn.Module):
    """
    CRNN with Temporal-Frequency Attention (Tier B: Edge-light baseline)
    
    Architecture:
        Conv blocks → BiGRU → TF-Attention → Classification
    
    Designed for:
        - Fast inference (~1-2M params)
        - Strong baseline performance
        - Attention focuses on rotor harmonics
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        input_channels: int = 3,  # Total, harmonic, percussive
        n_mels: int = 96,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Convolutional feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Temporal-frequency attention
        self.attention = TemporalFrequencyAttention(128)
        
        # Recurrent layers for temporal modeling
        self.gru = nn.GRU(
            input_size=128 * (n_mels // 8),  # After 3 maxpool layers
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)  # 256 = 128 * 2 (bidirectional)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, n_mels, time)
        Returns:
            logits: (batch, num_classes)
        """
        # CNN feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Temporal-frequency attention
        x = self.attention(x)
        
        # Reshape for GRU: (batch, time, features)
        batch, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # (batch, time, channels, freq)
        x = x.view(batch, t, -1)  # (batch, time, channels*freq)
        
        # Bidirectional GRU
        x, _ = self.gru(x)
        
        # Take final hidden state (mean over time)
        x = torch.mean(x, dim=1)
        
        # Classification
        x = self.dropout(x)
        logits = self.fc(x)
        
        return logits


class PANNsCNN14(nn.Module):
    """
    PANNs-inspired CNN14 (Tier A: Balanced performance)
    
    Simplified version of Pre-trained Audio Neural Networks
    From "PANNs: Large-Scale Pretrained Audio Neural Networks" (Kong et al., 2020)
    
    Designed for:
        - Strong accuracy with moderate compute
        - Good transfer learning
        - ~5-10M params
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        input_channels: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Initial conv block
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, n_mels, time)
        Returns:
            logits: (batch, num_classes)
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits


class PatchEmbedding(nn.Module):
    """Split spectrogram into patches and embed them"""
    
    def __init__(
        self,
        input_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 384
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            input_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            patches: (batch, num_patches, embed_dim)
        """
        x = self.proj(x)  # (batch, embed_dim, h', w')
        x = x.flatten(2)  # (batch, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch, num_patches, embed_dim)
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder block"""
    
    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with residual
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class AudioTransformer(nn.Module):
    """
    Audio Transformer (Tier S: Accuracy-first)
    
    Simplified version of Audio Spectrogram Transformer (AST)
    From "AST: Audio Spectrogram Transformer" (Gong et al., 2021)
    
    Designed for:
        - Best accuracy on noisy/complex scenes
        - Captures long-range dependencies
        - GPU recommended (~20-30M params)
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        input_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(input_channels, patch_size, embed_dim)
        
        # Positional embedding (learned)
        # For 96 mels and ~100 time frames with patch_size=16: ~36 patches
        self.num_patches = 64  # Approximate, will be adjusted in forward
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, n_mels, time)
        Returns:
            logits: (batch, num_classes)
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1 + num_patches, embed_dim)
        
        # Add positional embedding (interpolate if needed)
        if x.shape[1] != self.pos_embed.shape[1] + 1:
            # Interpolate positional embeddings
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=x.shape[1] - 1,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            pos_embed = self.pos_embed
        
        # Add pos embedding to all patches (not CLS)
        x[:, 1:, :] = x[:, 1:, :] + pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Use CLS token for classification
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        
        return logits


def create_model(
    model_type: str = 'crnn',
    num_classes: int = 3,
    input_channels: int = 3,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_type: 'crnn', 'panns', or 'transformer'
        num_classes: Number of output classes
        input_channels: Number of input channels (1 or 3 with HPSS)
        **kwargs: Additional model-specific arguments (n_mels, dropout, etc.)
    
    Returns:
        Model instance
    """
    if model_type == 'crnn':
        # CRNN needs n_mels parameter
        return CRNNWithAttention(
            num_classes=num_classes,
            input_channels=input_channels,
            **kwargs
        )
    elif model_type == 'panns':
        # PANNs doesn't need n_mels (uses adaptive pooling)
        # Filter out n_mels if present
        panns_kwargs = {k: v for k, v in kwargs.items() if k != 'n_mels'}
        return PANNsCNN14(
            num_classes=num_classes,
            input_channels=input_channels,
            **panns_kwargs
        )
    elif model_type == 'transformer':
        # Transformer doesn't need n_mels (uses patch embedding)
        # Filter out n_mels if present
        transformer_kwargs = {k: v for k, v in kwargs.items() if k != 'n_mels'}
        return AudioTransformer(
            num_classes=num_classes,
            input_channels=input_channels,
            **transformer_kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'crnn', 'panns', or 'transformer'")


if __name__ == '__main__':
    # Test models
    batch_size = 4
    channels = 3
    n_mels = 96
    time_frames = 100
    
    x = torch.randn(batch_size, channels, n_mels, time_frames)
    
    print("Testing CRNN with Attention...")
    model_crnn = create_model('crnn', num_classes=3, input_channels=channels, n_mels=n_mels)
    out = model_crnn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_crnn.parameters()) / 1e6:.2f}M\n")
    
    print("Testing PANNs CNN14...")
    model_panns = create_model('panns', num_classes=3, input_channels=channels)
    out = model_panns(x)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_panns.parameters()) / 1e6:.2f}M\n")
    
    print("Testing Audio Transformer...")
    model_transformer = create_model('transformer', num_classes=3, input_channels=channels, depth=6)
    out = model_transformer(x)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_transformer.parameters()) / 1e6:.2f}M")
