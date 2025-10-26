"""
Enhanced models with LIGO-style matched filter bank integration.

This module provides wrapper classes that augment existing models
(CRNN, PANN, Transformer, SNN) with the matched filter bank layer.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple

# Handle both direct execution and module import
try:
    from .matched_filter_bank import (
        MatchedFilterBank2D,
        EnhancedInputWithMatchedBank,
        create_adaptive_bank_specs
    )
except ImportError:
    from matched_filter_bank import (
        MatchedFilterBank2D,
        EnhancedInputWithMatchedBank,
        create_adaptive_bank_specs
    )


class CRNNWithMatchedBank(nn.Module):
    """
    CRNN enhanced with matched filter bank for low-SNR detection.
    
    Architecture:
        Audio → Mel/HPSS (3 channels) → [Original path + Matched Bank path]
                                       → Concat → CRNN backbone
    """
    
    def __init__(
        self,
        crnn_backbone: nn.Module,
        n_mels: int = 96,
        kernel_time: int = 25,
        bank_specs: Optional[List[Tuple]] = None,
        compression: Optional[int] = 6,
        trainable_bank: bool = True
    ):
        """
        Args:
            crnn_backbone: Original CRNN model (expects modified input channels)
            n_mels: Number of mel bins
            kernel_time: Template temporal size
            bank_specs: Template specifications (None = auto-generate)
            compression: Compress bank outputs to this many channels (None = no compression)
            trainable_bank: Allow templates to be fine-tuned
        """
        super().__init__()
        
        # Enhanced input layer
        self.enhanced_input = EnhancedInputWithMatchedBank(
            in_channels=3,
            n_mels=n_mels,
            kernel_time=kernel_time,
            bank_specs=bank_specs,
            compression=compression,
            trainable_bank=trainable_bank
        )
        
        self.crnn = crnn_backbone
        self.expected_input_channels = self.enhanced_input.out_channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, M, T] - HPSS spectrogram
            
        Returns:
            Class logits [B, num_classes]
        """
        # Apply matched filter bank and concat
        x_enhanced = self.enhanced_input(x)  # [B, C_enhanced, M, T']
        
        # Pass through CRNN
        out = self.crnn(x_enhanced)
        
        return out
    
    def get_template_info(self) -> dict:
        """Get information about template bank."""
        return {
            "n_templates": self.enhanced_input.matched_bank.n_templates,
            "template_names": self.enhanced_input.matched_bank.get_template_names(),
            "out_channels": self.expected_input_channels,
            "compression": self.enhanced_input.compression
        }


class PANNWithMatchedBank(nn.Module):
    """
    PANN (CNN14) enhanced with matched filter bank.
    
    Two integration options:
    1. Input augmentation: concat bank outputs with Mel before first conv
    2. Residual injection: add bank features after first conv layer
    """
    
    def __init__(
        self,
        pann_backbone: nn.Module,
        n_mels: int = 96,
        kernel_time: int = 25,
        bank_specs: Optional[List[Tuple]] = None,
        compression: int = 6,
        integration_mode: str = "input",  # "input" or "residual"
        trainable_bank: bool = True
    ):
        """
        Args:
            pann_backbone: Original PANN model
            n_mels: Number of mel bins
            kernel_time: Template temporal size
            bank_specs: Template specifications
            compression: Bank output compression
            integration_mode: "input" (early fusion) or "residual" (after first conv)
            trainable_bank: Allow template learning
        """
        super().__init__()
        
        self.integration_mode = integration_mode
        
        if integration_mode == "input":
            # Augment input
            self.enhanced_input = EnhancedInputWithMatchedBank(
                in_channels=3,
                n_mels=n_mels,
                kernel_time=kernel_time,
                bank_specs=bank_specs,
                compression=compression,
                trainable_bank=trainable_bank
            )
            self.expected_input_channels = self.enhanced_input.out_channels
            
        else:  # residual
            # Apply bank separately and inject after first conv
            self.matched_bank = MatchedFilterBank2D(
                in_channels=3,
                n_mels=n_mels,
                kernel_time=kernel_time,
                bank_specs=bank_specs,
                trainable=trainable_bank,
                use_relu=True
            )
            
            bank_out_ch = 3 * self.matched_bank.n_templates
            self.bank_projection = nn.Sequential(
                nn.Conv1d(bank_out_ch, compression, kernel_size=1),
                nn.BatchNorm1d(compression),
                nn.ReLU(inplace=True)
            )
            
        self.pann = pann_backbone
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, M, T]
            
        Returns:
            Class logits [B, num_classes]
        """
        if self.integration_mode == "input":
            # Early fusion
            x_enhanced = self.enhanced_input(x)
            out = self.pann(x_enhanced)
            
        else:  # residual
            # Get bank features
            bank_features = self.matched_bank(x)  # [B, C*K, T']
            bank_features = self.bank_projection(bank_features)  # [B, compression, T']
            
            # Pass original through PANN first conv
            # (This requires modifying PANN or accessing intermediate features)
            # For now, simple approach: add as extra input channels
            T_prime = bank_features.shape[2]
            x_trimmed = x[..., :T_prime]
            
            bank_features_2d = bank_features.unsqueeze(2).expand(-1, -1, x.shape[2], -1)
            x_combined = torch.cat([x_trimmed, bank_features_2d], dim=1)
            
            out = self.pann(x_combined)
        
        return out


class TransformerWithMatchedBank(nn.Module):
    """
    Audio Transformer enhanced with matched filter bank.
    
    Integration: Treat bank outputs as additional "channels" in patch embedding.
    The correlation maps boost token salience for weak rotor patterns.
    """
    
    def __init__(
        self,
        transformer_backbone: nn.Module,
        n_mels: int = 96,
        kernel_time: int = 25,
        bank_specs: Optional[List[Tuple]] = None,
        compression: int = 6,
        trainable_bank: bool = True
    ):
        """
        Args:
            transformer_backbone: Original transformer model
            n_mels: Number of mel bins
            kernel_time: Template temporal size
            bank_specs: Template specifications
            compression: Bank output compression
            trainable_bank: Allow template learning
        """
        super().__init__()
        
        self.enhanced_input = EnhancedInputWithMatchedBank(
            in_channels=3,
            n_mels=n_mels,
            kernel_time=kernel_time,
            bank_specs=bank_specs,
            compression=compression,
            trainable_bank=trainable_bank
        )
        
        self.transformer = transformer_backbone
        self.expected_input_channels = self.enhanced_input.out_channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, M, T]
            
        Returns:
            Class logits [B, num_classes]
        """
        # Augment with bank features
        x_enhanced = self.enhanced_input(x)  # [B, C_enhanced, M, T']
        
        # Pass through transformer
        out = self.transformer(x_enhanced)
        
        return out


class SNNWithMatchedBank(nn.Module):
    """
    Spiking Neural Network enhanced with matched filter bank.
    
    Integration: Bank outputs are nonnegative and sparse (SNR-like),
    making them ideal for rate-coded spiking inputs.
    """
    
    def __init__(
        self,
        snn_backbone: nn.Module,
        n_mels: int = 96,
        kernel_time: int = 25,
        bank_specs: Optional[List[Tuple]] = None,
        compression: int = 6,
        trainable_bank: bool = True,
        rate_scale: float = 1.0
    ):
        """
        Args:
            snn_backbone: Original SNN model
            n_mels: Number of mel bins
            kernel_time: Template temporal size
            bank_specs: Template specifications
            compression: Bank output compression
            trainable_bank: Allow template learning
            rate_scale: Scaling factor for spike rates from bank outputs
        """
        super().__init__()
        
        self.matched_bank = MatchedFilterBank2D(
            in_channels=3,
            n_mels=n_mels,
            kernel_time=kernel_time,
            bank_specs=bank_specs,
            trainable=trainable_bank,
            use_relu=True  # Essential for rate coding
        )
        
        bank_out_ch = 3 * self.matched_bank.n_templates
        
        # Scale and compress bank outputs for SNN
        self.rate_encoder = nn.Sequential(
            nn.Conv1d(bank_out_ch, compression, kernel_size=1),
            nn.BatchNorm1d(compression),
            nn.Sigmoid()  # Map to [0, 1] for spike rates
        )
        
        self.rate_scale = rate_scale
        self.snn = snn_backbone
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, M, T]
            
        Returns:
            Class logits [B, num_classes]
        """
        # Get bank features (SNR-like, already ReLU'd)
        bank_features = self.matched_bank(x)  # [B, C*K, T']
        
        # Encode as spike rates
        spike_rates = self.rate_encoder(bank_features)  # [B, compression, T']
        spike_rates = spike_rates * self.rate_scale
        
        # Expand to 2D for SNN input
        T_prime = spike_rates.shape[2]
        x_trimmed = x[..., :T_prime]
        
        spike_rates_2d = spike_rates.unsqueeze(2).expand(-1, -1, x.shape[2], -1)
        x_combined = torch.cat([x_trimmed, spike_rates_2d], dim=1)
        
        # Pass through SNN
        out = self.snn(x_combined)
        
        return out


# ============================================================================
# Factory functions for easy model creation
# ============================================================================

def create_enhanced_crnn(
    crnn_backbone: nn.Module,
    n_mels: int = 96,
    sr: int = 16000,
    compression: int = 6,
    **kwargs
) -> CRNNWithMatchedBank:
    """
    Create CRNN with adaptive matched filter bank.
    
    Args:
        crnn_backbone: CRNN model instance
        n_mels: Number of mel bins
        sr: Sample rate
        compression: Bank output compression
        **kwargs: Additional arguments for matched bank
        
    Returns:
        Enhanced CRNN model
    """
    bank_specs = create_adaptive_bank_specs(n_mels=n_mels, sr=sr)
    
    return CRNNWithMatchedBank(
        crnn_backbone=crnn_backbone,
        n_mels=n_mels,
        bank_specs=bank_specs,
        compression=compression,
        **kwargs
    )


def create_enhanced_pann(
    pann_backbone: nn.Module,
    n_mels: int = 96,
    sr: int = 16000,
    compression: int = 6,
    integration_mode: str = "input",
    **kwargs
) -> PANNWithMatchedBank:
    """
    Create PANN with adaptive matched filter bank.
    
    Args:
        pann_backbone: PANN model instance
        n_mels: Number of mel bins
        sr: Sample rate
        compression: Bank output compression
        integration_mode: "input" or "residual"
        **kwargs: Additional arguments
        
    Returns:
        Enhanced PANN model
    """
    bank_specs = create_adaptive_bank_specs(n_mels=n_mels, sr=sr)
    
    return PANNWithMatchedBank(
        pann_backbone=pann_backbone,
        n_mels=n_mels,
        bank_specs=bank_specs,
        compression=compression,
        integration_mode=integration_mode,
        **kwargs
    )


def create_enhanced_transformer(
    transformer_backbone: nn.Module,
    n_mels: int = 96,
    sr: int = 16000,
    compression: int = 6,
    **kwargs
) -> TransformerWithMatchedBank:
    """
    Create Transformer with adaptive matched filter bank.
    
    Args:
        transformer_backbone: Transformer model instance
        n_mels: Number of mel bins
        sr: Sample rate
        compression: Bank output compression
        **kwargs: Additional arguments
        
    Returns:
        Enhanced Transformer model
    """
    bank_specs = create_adaptive_bank_specs(n_mels=n_mels, sr=sr)
    
    return TransformerWithMatchedBank(
        transformer_backbone=transformer_backbone,
        n_mels=n_mels,
        bank_specs=bank_specs,
        compression=compression,
        **kwargs
    )


def create_enhanced_snn(
    snn_backbone: nn.Module,
    n_mels: int = 96,
    sr: int = 16000,
    compression: int = 6,
    rate_scale: float = 1.0,
    **kwargs
) -> SNNWithMatchedBank:
    """
    Create SNN with adaptive matched filter bank.
    
    Args:
        snn_backbone: SNN model instance
        n_mels: Number of mel bins
        sr: Sample rate
        compression: Bank output compression
        rate_scale: Spike rate scaling
        **kwargs: Additional arguments
        
    Returns:
        Enhanced SNN model
    """
    bank_specs = create_adaptive_bank_specs(n_mels=n_mels, sr=sr)
    
    return SNNWithMatchedBank(
        snn_backbone=snn_backbone,
        n_mels=n_mels,
        bank_specs=bank_specs,
        compression=compression,
        rate_scale=rate_scale,
        **kwargs
    )


if __name__ == "__main__":
    print("=" * 80)
    print("Enhanced Models with LIGO-Style Matched Filter Bank")
    print("=" * 80)
    
    # Create dummy backbones for testing
    class DummyCRNN(nn.Module):
        def __init__(self, in_channels):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(32, 3)
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x).squeeze(-1).squeeze(-1)
            return self.fc(x)
    
    # Test each enhanced model
    x = torch.randn(2, 3, 96, 100)
    
    print("\n1. Enhanced CRNN:")
    crnn = DummyCRNN(in_channels=9)  # 3 original + 6 compressed bank
    model = create_enhanced_crnn(crnn, n_mels=96, sr=16000, compression=6)
    out = model(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out.shape}")
    print(f"   Template info: {model.get_template_info()}")
    
    print("\n2. Enhanced PANN (input mode):")
    pann = DummyCRNN(in_channels=9)
    model = create_enhanced_pann(pann, n_mels=96, sr=16000, compression=6)
    out = model(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out.shape}")
    
    print("\n3. Enhanced Transformer:")
    transformer = DummyCRNN(in_channels=9)
    model = create_enhanced_transformer(transformer, n_mels=96, sr=16000, compression=6)
    out = model(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out.shape}")
    
    print("\n4. Enhanced SNN:")
    snn = DummyCRNN(in_channels=9)
    model = create_enhanced_snn(snn, n_mels=96, sr=16000, compression=6)
    out = model(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out.shape}")
    
    # Count parameters
    print("\n5. Parameter comparison:")
    original_params = sum(p.numel() for p in crnn.parameters())
    enhanced_params = sum(p.numel() for p in model.parameters())
    overhead = enhanced_params - original_params
    print(f"   Original model:  {original_params:,} params")
    print(f"   Enhanced model:  {enhanced_params:,} params")
    print(f"   Bank overhead:   {overhead:,} params ({overhead/original_params*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("✓ All enhanced models tested successfully!")
    print("=" * 80)
