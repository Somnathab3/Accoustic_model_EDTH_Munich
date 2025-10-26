"""
LIGO-inspired Template Bank for Acoustic Drone Detection

This module implements a physics-inspired matched-filter layer that detects
faint, structured patterns (harmonic combs, chirps, amplitude modulation)
similar to how LIGO detects gravitational wave chirps.

Key concepts:
- Chirp kernels: detect RPM drift (slanted ridges in time-frequency)
- Harmonic comb kernels: detect rotor harmonics (vertical ridges at f₀, 2f₀, ...)
- AM kernels: detect amplitude modulation (especially helicopters)
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import List, Tuple, Optional
import math


def make_chirp_kernel(
    n_mels: int,
    n_frames: int,
    f0_bin: int,
    f1_bin: int,
    sigma_freq: float = 2.0,
    sigma_time: float = 0.6
) -> torch.Tensor:
    """
    Create a chirp kernel: a slanted ridge in time-frequency plane.
    
    Args:
        n_mels: Number of mel frequency bins
        n_frames: Temporal kernel size
        f0_bin: Starting frequency bin
        f1_bin: Ending frequency bin
        sigma_freq: Gaussian spread in frequency direction
        sigma_time: Gaussian spread in time direction
        
    Returns:
        Normalized chirp kernel [n_mels, n_frames]
    """
    k = np.zeros((n_mels, n_frames), dtype=np.float32)
    
    # Draw slanted ridge from (f0_bin, 0) → (f1_bin, n_frames-1)
    for t in range(n_frames):
        alpha = t / (n_frames - 1 + 1e-8)
        f = int(round(f0_bin + (f1_bin - f0_bin) * alpha))
        if 0 <= f < n_mels:
            k[f, t] = 1.0
    
    # Gaussian thicken the line (spread across frequency bins)
    k = gaussian_filter(k, sigma=(sigma_freq, sigma_time), mode='constant')
    
    # Whiten: zero-mean, unit-variance (correlation-like)
    k = k - k.mean()
    k = k / (k.std() + 1e-8)
    
    return torch.tensor(k, dtype=torch.float32)


def make_comb_kernel(
    n_mels: int,
    n_frames: int,
    f0_bin: int,
    n_harm: int = 6,
    decay: float = 0.7
) -> torch.Tensor:
    """
    Create harmonic comb kernel: energy at f₀, 2f₀, 3f₀, ...
    
    Args:
        n_mels: Number of mel frequency bins
        n_frames: Temporal kernel size
        f0_bin: Fundamental frequency bin
        n_harm: Number of harmonics to include
        decay: Harmonic amplitude decay factor (< 1)
        
    Returns:
        Normalized comb kernel [n_mels, n_frames]
    """
    k = np.zeros((n_mels, n_frames), dtype=np.float32)
    
    # Place ridges at harmonic multiples with decaying amplitude
    for h in range(1, n_harm + 1):
        f = int(round(h * f0_bin))
        if 0 <= f < n_mels:
            amplitude = decay ** (h - 1)
            k[f, :] = amplitude
    
    # Apply slight frequency blur to account for tuning uncertainty
    k = gaussian_filter(k, sigma=(1.5, 0.3), mode='constant')
    
    # Normalize
    k = k - k.mean()
    k = k / (k.std() + 1e-8)
    
    return torch.tensor(k, dtype=torch.float32)


def make_am_kernel(
    n_mels: int,
    n_frames: int,
    center_bin: int,
    mod_freq: float = 0.2,
    bandwidth: int = 10
) -> torch.Tensor:
    """
    Create amplitude modulation kernel for helicopter detection.
    
    Args:
        n_mels: Number of mel frequency bins
        n_frames: Temporal kernel size
        center_bin: Center frequency bin
        mod_freq: Modulation frequency (normalized to frame rate)
        bandwidth: Frequency bandwidth around center
        
    Returns:
        Normalized AM kernel [n_mels, n_frames]
    """
    k = np.zeros((n_mels, n_frames), dtype=np.float32)
    
    # Create amplitude envelope
    t = np.arange(n_frames)
    envelope = 0.5 * (1 + np.cos(2 * np.pi * mod_freq * t))
    
    # Apply to frequency band
    f_start = max(0, center_bin - bandwidth // 2)
    f_end = min(n_mels, center_bin + bandwidth // 2)
    
    for f in range(f_start, f_end):
        # Gaussian taper
        weight = np.exp(-0.5 * ((f - center_bin) / (bandwidth / 4)) ** 2)
        k[f, :] = weight * envelope
    
    # Normalize
    k = k - k.mean()
    k = k / (k.std() + 1e-8)
    
    return torch.tensor(k, dtype=torch.float32)


class MatchedFilterBank2D(nn.Module):
    """
    LIGO-style matched filter bank for acoustic pattern detection.
    
    This layer applies a bank of template kernels (chirps, harmonic combs, AM)
    to input spectrograms, producing SNR-like correlation maps that highlight
    faint structured patterns.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        n_mels: int = 96,
        kernel_time: int = 25,
        bank_specs: Optional[List[Tuple]] = None,
        trainable: bool = True,
        use_relu: bool = True
    ):
        """
        Args:
            in_channels: Number of input channels (e.g., 3 for HPSS)
            n_mels: Number of mel frequency bins
            kernel_time: Temporal kernel size (frames)
            bank_specs: List of template specifications, e.g.,
                        [("chirp", f0_bin, f1_bin), ("comb", f0_bin), ("am", center_bin)]
            trainable: Whether to make kernels learnable
            use_relu: Apply ReLU to outputs (SNR-like, nonnegative)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.n_mels = n_mels
        self.kernel_time = kernel_time
        self.use_relu = use_relu
        
        # Build default bank if not provided
        if bank_specs is None:
            bank_specs = self._default_bank_specs()
        
        self.bank_specs = bank_specs
        self.n_templates = len(bank_specs)
        
        # Build template kernels
        kernels = []
        for spec in bank_specs:
            kernel = self._create_kernel(spec)
            kernels.append(kernel.unsqueeze(0))  # [1, n_mels, kT]
        
        # Stack and replicate across input channels
        # Shape: [n_templates, 1, n_mels, kT]
        template_bank = torch.stack(kernels, dim=0)
        
        # Replicate for each input channel (for grouped convolution)
        # Shape: [n_templates * in_channels, 1, n_mels, kT]
        weight = template_bank.repeat(in_channels, 1, 1, 1)
        
        # Create grouped 2D convolution (each input channel processed separately)
        self.conv = nn.Conv2d(
            in_channels,
            self.n_templates * in_channels,
            kernel_size=(n_mels, kernel_time),
            groups=in_channels,
            bias=False,
            padding=(0, 0)
        )
        
        # Initialize with template bank
        with torch.no_grad():
            self.conv.weight.copy_(weight)
        
        # Control trainability
        self.conv.weight.requires_grad = trainable
        
    def _default_bank_specs(self) -> List[Tuple]:
        """
        Generate default template bank covering common drone/helicopter patterns.
        
        Returns:
            List of template specifications
        """
        specs = []
        
        # --- Drone templates ---
        # High-frequency harmonic combs (300-800 Hz region on mel scale)
        # Assuming mel bins ~ 20-80 for this range (depends on sr/n_mels)
        drone_f0_bins = [25, 30, 35, 40, 45, 50, 55, 60]
        for f0 in drone_f0_bins:
            specs.append(("comb", f0, 6))  # 6 harmonics
        
        # Drone chirps (RPM variations)
        drone_chirps = [
            (30, 35), (35, 40), (40, 45), (45, 50),  # Upward chirps
            (35, 30), (40, 35), (45, 40), (50, 45),  # Downward chirps
        ]
        for f0, f1 in drone_chirps:
            specs.append(("chirp", f0, f1))
        
        # --- Helicopter templates ---
        # Low-frequency blade-pass fundamentals (10-60 Hz)
        # Assuming mel bins ~ 5-20 for this range
        heli_f0_bins = [5, 8, 10, 12, 15, 18]
        for f0 in heli_f0_bins:
            specs.append(("comb", f0, 8))  # More harmonics for helicopters
        
        # Helicopter AM patterns
        heli_am_centers = [10, 15, 20]
        for center in heli_am_centers:
            specs.append(("am", center, 0.15))  # Slow modulation
        
        return specs
    
    def _create_kernel(self, spec: Tuple) -> torch.Tensor:
        """
        Create kernel from specification.
        
        Args:
            spec: Template specification tuple
            
        Returns:
            Kernel tensor [n_mels, kernel_time]
        """
        kernel_type = spec[0]
        
        if kernel_type == "chirp":
            f0_bin, f1_bin = spec[1], spec[2]
            return make_chirp_kernel(self.n_mels, self.kernel_time, f0_bin, f1_bin)
        
        elif kernel_type == "comb":
            f0_bin = spec[1]
            n_harm = spec[2] if len(spec) > 2 else 6
            return make_comb_kernel(self.n_mels, self.kernel_time, f0_bin, n_harm)
        
        elif kernel_type == "am":
            center_bin = spec[1]
            mod_freq = spec[2] if len(spec) > 2 else 0.2
            return make_am_kernel(self.n_mels, self.kernel_time, center_bin, mod_freq)
        
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply matched filter bank to input spectrogram.
        
        Args:
            x: Input tensor [B, C, M, T] where
               B = batch size
               C = channels (e.g., 3 for HPSS)
               M = mel bins
               T = time frames
               
        Returns:
            Correlation maps [B, C * n_templates, T']
            where T' = T - kernel_time + 1
        """
        # Apply grouped convolution (matched filtering)
        # Output: [B, C * n_templates, 1, T']
        y = self.conv(x)
        
        # Remove spatial dimension
        y = y.squeeze(2)  # [B, C * n_templates, T']
        
        # Apply ReLU for SNR-like nonnegative outputs
        if self.use_relu:
            y = torch.relu(y)
        
        return y
    
    def get_template_names(self) -> List[str]:
        """Get human-readable names for each template."""
        names = []
        for spec in self.bank_specs:
            if spec[0] == "chirp":
                names.append(f"chirp_{spec[1]}->{spec[2]}")
            elif spec[0] == "comb":
                names.append(f"comb_f0={spec[1]}")
            elif spec[0] == "am":
                names.append(f"am_f={spec[1]}")
        return names


class EnhancedInputWithMatchedBank(nn.Module):
    """
    Wrapper that concatenates matched filter bank outputs with original input.
    
    This creates an augmented input: [Mel_HPSS, MatchedBank(Mel_HPSS)]
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        n_mels: int = 96,
        kernel_time: int = 25,
        bank_specs: Optional[List[Tuple]] = None,
        compression: Optional[int] = None,
        trainable_bank: bool = True
    ):
        """
        Args:
            in_channels: Original input channels
            n_mels: Number of mel bins
            kernel_time: Template temporal size
            bank_specs: Template specifications
            compression: If set, compress bank outputs to this many channels
            trainable_bank: Make templates learnable
        """
        super().__init__()
        
        self.matched_bank = MatchedFilterBank2D(
            in_channels=in_channels,
            n_mels=n_mels,
            kernel_time=kernel_time,
            bank_specs=bank_specs,
            trainable=trainable_bank,
            use_relu=True
        )
        
        bank_out_channels = in_channels * self.matched_bank.n_templates
        
        # Optional bottleneck to reduce channel explosion
        self.compression = compression
        if compression is not None:
            self.bottleneck = nn.Sequential(
                nn.Conv1d(bank_out_channels, compression, kernel_size=1, bias=False),
                nn.BatchNorm1d(compression),
                nn.ReLU(inplace=True)
            )
            self.out_channels = in_channels + compression
        else:
            self.bottleneck = None
            self.out_channels = in_channels + bank_out_channels
        
        self.kernel_time = kernel_time
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, M, T]
            
        Returns:
            Concatenated features [B, C_out, M, T']
            where C_out = in_channels + (bank_channels or compression)
        """
        # Apply matched filter bank
        bank_features = self.matched_bank(x)  # [B, C*K, T']
        
        # Optional compression
        if self.bottleneck is not None:
            bank_features = self.bottleneck(bank_features)  # [B, compression, T']
        
        # Trim original input to match temporal dimension
        # x: [B, C, M, T] -> [B, C, M, T']
        T_prime = bank_features.shape[2]
        x_trimmed = x[..., :T_prime]
        
        # Expand bank features to match spatial structure
        # [B, K, T'] -> [B, K, 1, T'] -> [B, K, M, T']
        bank_features = bank_features.unsqueeze(2)
        bank_features = bank_features.expand(-1, -1, x.shape[2], -1)
        
        # Concatenate along channel dimension
        out = torch.cat([x_trimmed, bank_features], dim=1)
        
        return out


def create_adaptive_bank_specs(
    n_mels: int = 96,
    sr: int = 16000,
    n_fft: int = 512,
    drone_f0_range: Tuple[float, float] = (300, 800),
    heli_f0_range: Tuple[float, float] = (10, 60),
    n_drone_templates: int = 12,
    n_heli_templates: int = 8
) -> List[Tuple]:
    """
    Create template bank specs adapted to audio parameters.
    
    Args:
        n_mels: Number of mel bins
        sr: Sample rate
        n_fft: FFT size
        drone_f0_range: Drone fundamental frequency range (Hz)
        heli_f0_range: Helicopter fundamental frequency range (Hz)
        n_drone_templates: Number of drone templates
        n_heli_templates: Number of helicopter templates
        
    Returns:
        List of template specifications
    """
    from librosa import hz_to_mel, mel_to_hz
    
    # Calculate mel frequency range
    mel_min = hz_to_mel(0)
    mel_max = hz_to_mel(sr / 2)
    
    def hz_to_mel_bin(freq_hz: float) -> int:
        """Convert Hz to mel bin index."""
        mel = hz_to_mel(freq_hz)
        bin_idx = int(round((mel - mel_min) / (mel_max - mel_min) * (n_mels - 1)))
        return max(0, min(n_mels - 1, bin_idx))
    
    specs = []
    
    # --- Drone templates ---
    drone_f0_bins = [
        hz_to_mel_bin(f) for f in 
        np.linspace(drone_f0_range[0], drone_f0_range[1], n_drone_templates // 2)
    ]
    
    # Harmonic combs
    for f0_bin in drone_f0_bins:
        specs.append(("comb", f0_bin, 6))
    
    # Chirps (±10% frequency drift)
    for f0_bin in drone_f0_bins:
        drift = max(2, int(0.1 * f0_bin))
        specs.append(("chirp", f0_bin, f0_bin + drift))  # Upward
        specs.append(("chirp", f0_bin, f0_bin - drift))  # Downward
    
    # --- Helicopter templates ---
    heli_f0_bins = [
        hz_to_mel_bin(f) for f in
        np.linspace(heli_f0_range[0], heli_f0_range[1], n_heli_templates // 2)
    ]
    
    # Harmonic combs (more harmonics)
    for f0_bin in heli_f0_bins:
        specs.append(("comb", f0_bin, 8))
    
    # AM patterns
    for f0_bin in heli_f0_bins:
        specs.append(("am", f0_bin, 0.15))
    
    return specs


if __name__ == "__main__":
    # Demo and visualization
    print("=" * 80)
    print("LIGO-Style Matched Filter Bank for Acoustic Drone Detection")
    print("=" * 80)
    
    # Create sample input (batch=2, channels=3, mels=96, time=100)
    x = torch.randn(2, 3, 96, 100)
    
    # Test basic matched filter bank
    print("\n1. Basic MatchedFilterBank2D:")
    bank = MatchedFilterBank2D(in_channels=3, n_mels=96, kernel_time=25)
    y = bank(x)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Templates:    {bank.n_templates}")
    print(f"   Template names: {bank.get_template_names()[:5]}...")
    
    # Test enhanced input wrapper
    print("\n2. EnhancedInputWithMatchedBank (no compression):")
    enhanced = EnhancedInputWithMatchedBank(in_channels=3, n_mels=96, kernel_time=25)
    y_enhanced = enhanced(x)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {y_enhanced.shape}")
    print(f"   Channel expansion: {3} -> {enhanced.out_channels}")
    
    # Test with compression
    print("\n3. EnhancedInputWithMatchedBank (with compression to 6 channels):")
    enhanced_compressed = EnhancedInputWithMatchedBank(
        in_channels=3, n_mels=96, kernel_time=25, compression=6
    )
    y_compressed = enhanced_compressed(x)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {y_compressed.shape}")
    print(f"   Channel expansion: {3} -> {enhanced_compressed.out_channels}")
    
    # Test adaptive specs
    print("\n4. Adaptive template bank specs (16kHz, 96 mels):")
    specs = create_adaptive_bank_specs(n_mels=96, sr=16000)
    print(f"   Total templates: {len(specs)}")
    print(f"   Drone templates: {sum(1 for s in specs if s[1] > 20)}")
    print(f"   Heli templates:  {sum(1 for s in specs if s[1] <= 20)}")
    
    # Visualize some kernels
    print("\n5. Visualizing sample kernels...")
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle("Sample Template Kernels (LIGO-style matched filters)", fontsize=14)
        
        # Chirp kernels
        chirp1 = make_chirp_kernel(96, 25, 30, 40)
        chirp2 = make_chirp_kernel(96, 25, 40, 30)
        
        # Comb kernels
        comb1 = make_comb_kernel(96, 25, 35, n_harm=6)
        comb2 = make_comb_kernel(96, 25, 10, n_harm=8)
        
        # AM kernels
        am1 = make_am_kernel(96, 25, 15, mod_freq=0.15)
        am2 = make_am_kernel(96, 25, 40, mod_freq=0.2)
        
        kernels = [
            (chirp1, "Drone Chirp (up)"),
            (chirp2, "Drone Chirp (down)"),
            (comb1, "Drone Comb (f₀≈35)"),
            (comb2, "Heli Comb (f₀≈10)"),
            (am1, "Heli AM (low freq)"),
            (am2, "Drone AM (mid freq)")
        ]
        
        for ax, (kernel, title) in zip(axes.flat, kernels):
            im = ax.imshow(kernel.numpy(), aspect='auto', origin='lower', cmap='RdBu_r')
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Time (frames)")
            ax.set_ylabel("Mel bin")
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.tight_layout()
        plt.savefig("f:/EDTH/acoustic-drone-detector/visualizations/matched_filter_kernels.png", dpi=150)
        print("   ✓ Saved visualization to visualizations/matched_filter_kernels.png")
        
    except ImportError:
        print("   (matplotlib not available for visualization)")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed! Ready to integrate with CRNN/PANN/Transformer.")
    print("=" * 80)
