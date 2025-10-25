"""
State-of-the-art Audio Preprocessing and Augmentation Pipeline
Implements best practices from ESC-50, AudioSet, and acoustic drone detection literature
"""
import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np
from typing import Optional, Tuple
import random


class AudioPreprocessor(nn.Module):
    """
    Advanced audio preprocessing with log-mel spectrograms and HPSS
    
    Args:
        sample_rate: Target sampling rate (16kHz recommended)
        n_fft: FFT window size (1024 for 16kHz)
        hop_length: Hop length in samples (320 = 20ms at 16kHz)
        n_mels: Number of mel filterbanks (64-96 recommended)
        window_duration: Analysis window in seconds (2.0s recommended)
        use_hpss: Whether to add harmonic-percussive separation channels
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 320,
        n_mels: int = 96,
        window_duration: float = 2.0,
        use_hpss: bool = True,
        f_min: float = 50.0,
        f_max: float = 8000.0
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.window_duration = window_duration
        self.use_hpss = use_hpss
        self.target_length = int(window_duration * sample_rate)
        
        # Mel spectrogram transform
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
            normalized=True
        )
        
        # Amplitude to dB
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype='power',
            top_db=80.0
        )
    
    def load_audio(self, path: str) -> torch.Tensor:
        """
        Load and resample audio to target sample rate using librosa
        
        Uses librosa by default for maximum compatibility across platforms.
        No FFmpeg or TorchCodec required - works out of the box on Windows.
        """
        # Load audio with librosa (reliable, no FFmpeg needed)
        y, sr = librosa.load(path, sr=self.sample_rate, mono=True)
        
        # Convert to tensor and add channel dimension
        waveform = torch.from_numpy(y).unsqueeze(0).float()
        
        return waveform
    
    def pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pad or trim waveform to target length"""
        current_length = waveform.shape[-1]
        
        if current_length > self.target_length:
            # Random crop during training for augmentation
            start = random.randint(0, current_length - self.target_length)
            waveform = waveform[..., start:start + self.target_length]
        elif current_length < self.target_length:
            # Pad with zeros
            pad_length = self.target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        
        return waveform
    
    def compute_hpss(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Harmonic-Percussive Source Separation
        Useful for emphasizing rotor harmonics (harmonic) vs motor noise (percussive)
        """
        # Convert to numpy for librosa
        audio_np = waveform.squeeze().numpy()
        
        # HPSS with moderate margin for drone acoustics
        harmonic, percussive = librosa.effects.hpss(
            audio_np,
            margin=2.0
        )
        
        return torch.from_numpy(harmonic).unsqueeze(0), torch.from_numpy(percussive).unsqueeze(0)
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to log-mel spectrogram with optional HPSS channels
        
        Returns:
            Log-mel spectrogram of shape (channels, n_mels, time)
            channels = 1 (basic) or 3 (with HPSS: total, harmonic, percussive)
        """
        # Pad or trim to target length
        waveform = self.pad_or_trim(waveform)
        
        # Compute mel spectrogram
        mel_spec = self.mel_spec(waveform)
        log_mel = self.amplitude_to_db(mel_spec)
        
        if self.use_hpss:
            # Compute HPSS components
            harmonic, percussive = self.compute_hpss(waveform)
            
            # Compute mel spectrograms for harmonic and percussive
            mel_harmonic = self.mel_spec(harmonic)
            mel_percussive = self.mel_spec(percussive)
            
            log_mel_harmonic = self.amplitude_to_db(mel_harmonic)
            log_mel_percussive = self.amplitude_to_db(mel_percussive)
            
            # Stack as 3-channel input
            log_mel = torch.cat([log_mel, log_mel_harmonic, log_mel_percussive], dim=0)
        
        return log_mel


class SpecAugment(nn.Module):
    """
    SpecAugment: time and frequency masking for robust training
    From "SpecAugment: A Simple Data Augmentation Method for ASR" (Park et al., 2019)
    """
    
    def __init__(
        self,
        freq_mask_param: int = 16,
        time_mask_param: int = 40,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        p: float = 0.5
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.p = p
        
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
    
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment with probability p"""
        if not self.training or random.random() > self.p:
            return spec
        
        # Apply frequency masks
        for _ in range(self.n_freq_masks):
            spec = self.freq_mask(spec)
        
        # Apply time masks
        for _ in range(self.n_time_masks):
            spec = self.time_mask(spec)
        
        return spec


class BackgroundNoiseMixer(nn.Module):
    """
    Mix background noise into audio samples
    Implements SNR curriculum learning (start clean, progressively add noise)
    """
    
    def __init__(
        self,
        noise_paths: Optional[list] = None,
        snr_range: Tuple[float, float] = (0.0, 20.0),
        p: float = 0.5
    ):
        super().__init__()
        self.noise_paths = noise_paths or []
        self.snr_range = snr_range
        self.p = p
        self.noise_cache = []
        
        # Preload noise samples
        if noise_paths:
            self._load_noise_samples()
    
    def _load_noise_samples(self):
        """Preload noise samples for fast mixing using librosa"""
        for path in self.noise_paths[:10]:  # Limit to avoid memory issues
            try:
                y, sr = librosa.load(path, sr=16000, mono=True)
                waveform = torch.from_numpy(y).unsqueeze(0).float()
                self.noise_cache.append(waveform)
            except:
                pass
    
    def add_noise(self, waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Add Gaussian noise at specified SNR"""
        # Compute signal power
        signal_power = torch.mean(waveform ** 2)
        
        # Compute noise power for target SNR
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        
        # Generate and add noise
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise
    
    def forward(self, waveform: torch.Tensor, current_epoch: int = 0, max_epochs: int = 50) -> torch.Tensor:
        """
        Add noise with SNR curriculum
        Early epochs: higher SNR (cleaner)
        Later epochs: lower SNR (noisier)
        """
        if not self.training or random.random() > self.p:
            return waveform
        
        # SNR curriculum: start high, decrease over epochs
        progress = min(current_epoch / max_epochs, 1.0)
        min_snr, max_snr = self.snr_range
        target_snr = max_snr - progress * (max_snr - min_snr)
        
        # Add random variation
        snr_db = target_snr + random.uniform(-2.0, 2.0)
        
        return self.add_noise(waveform, snr_db)


class TimePitchAugmentation(nn.Module):
    """
    Apply small time stretching and pitch shifting
    Simulates RPM variations and Doppler effects
    """
    
    def __init__(
        self,
        time_stretch_range: Tuple[float, float] = (0.95, 1.05),
        pitch_shift_range: Tuple[int, int] = (-2, 2),
        p: float = 0.5
    ):
        super().__init__()
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_range = pitch_shift_range
        self.p = p
    
    def forward(self, waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """Apply time stretch and/or pitch shift"""
        if not self.training or random.random() > self.p:
            return waveform
        
        audio_np = waveform.squeeze().numpy()
        
        # Random time stretch
        if random.random() > 0.5:
            rate = random.uniform(*self.time_stretch_range)
            audio_np = librosa.effects.time_stretch(audio_np, rate=rate)
        
        # Random pitch shift
        if random.random() > 0.5:
            n_steps = random.randint(*self.pitch_shift_range)
            audio_np = librosa.effects.pitch_shift(
                audio_np,
                sr=sample_rate,
                n_steps=n_steps
            )
        
        return torch.from_numpy(audio_np).unsqueeze(0).float()


class MixupAugmentation:
    """
    Mixup: mix two samples and their labels
    From "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)
    """
    
    def __init__(self, alpha: float = 0.2, p: float = 0.5):
        self.alpha = alpha
        self.p = p
    
    def __call__(
        self,
        spec1: torch.Tensor,
        label1: torch.Tensor,
        spec2: torch.Tensor,
        label2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mix two spectrograms and labels
        
        Returns:
            Mixed spectrogram and soft label
        """
        if random.random() > self.p:
            return spec1, label1
        
        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix spectrograms
        mixed_spec = lam * spec1 + (1 - lam) * spec2
        
        # Mix labels (create soft labels)
        if label1.dim() == 0:  # If hard labels
            num_classes = 3  # background, drone, helicopter
            soft_label1 = torch.zeros(num_classes)
            soft_label1[label1] = 1.0
            soft_label2 = torch.zeros(num_classes)
            soft_label2[label2] = 1.0
        else:
            soft_label1 = label1
            soft_label2 = label2
        
        mixed_label = lam * soft_label1 + (1 - lam) * soft_label2
        
        return mixed_spec, mixed_label


class AugmentationPipeline(nn.Module):
    """
    Complete augmentation pipeline for training
    Applies all augmentations in optimal order
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        use_time_pitch: bool = True,
        use_noise: bool = True,
        use_spec_augment: bool = True,
        use_mixup: bool = True
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.use_time_pitch = use_time_pitch
        self.use_noise = use_noise
        self.use_spec_augment = use_spec_augment
        self.use_mixup = use_mixup
        
        # Initialize transforms
        if use_time_pitch:
            self.time_pitch = TimePitchAugmentation(p=0.3)
        
        if use_noise:
            self.noise_mixer = BackgroundNoiseMixer(p=0.4)
        
        if use_spec_augment:
            self.spec_augment = SpecAugment(p=0.5)
        
        if use_mixup:
            self.mixup = MixupAugmentation(alpha=0.2, p=0.3)
    
    def augment_waveform(
        self,
        waveform: torch.Tensor,
        current_epoch: int = 0,
        max_epochs: int = 50
    ) -> torch.Tensor:
        """Apply waveform-level augmentations"""
        # Time and pitch augmentation (subtle)
        if self.use_time_pitch:
            waveform = self.time_pitch(waveform, self.sample_rate)
        
        # Noise mixing with curriculum
        if self.use_noise:
            waveform = self.noise_mixer(waveform, current_epoch, max_epochs)
        
        return waveform
    
    def augment_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply spectrogram-level augmentations"""
        if self.use_spec_augment:
            spec = self.spec_augment(spec)
        
        return spec