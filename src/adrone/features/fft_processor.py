"""
FFT Feature Processor for Audio Analysis
Extracts frequency domain features using Fast Fourier Transform
"""
import numpy as np
import torch
import torch.nn as nn
import librosa


class FFTProcessor:
    """Extract FFT-based features from audio signals"""
    
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        sample_rate: int = 16000,
        max_freq_bins: int = 256
    ):
        """
        Args:
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            n_mels: Number of mel bands
            sample_rate: Audio sample rate
            max_freq_bins: Maximum frequency bins to keep
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.max_freq_bins = max_freq_bins
    
    def compute_fft_features(self, audio: np.ndarray) -> dict:
        """
        Compute multiple FFT-based features from audio
        
        Args:
            audio: Audio waveform (1D numpy array)
            
        Returns:
            Dictionary containing various FFT features
        """
        # Compute FFT
        fft = np.fft.rfft(audio, n=self.n_fft)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        power = magnitude ** 2
        
        # Limit to max frequency bins
        magnitude = magnitude[:self.max_freq_bins]
        phase = phase[:self.max_freq_bins]
        power = power[:self.max_freq_bins]
        
        # Compute STFT for time-frequency representation
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        stft_magnitude = np.abs(stft)
        stft_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)[0]
        
        # MFCC
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=20, n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        return {
            'fft_magnitude': magnitude,
            'fft_phase': phase,
            'fft_power': power,
            'stft_db': stft_db,
            'mel_spec_db': mel_spec_db,
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'spectral_bandwidth': spectral_bandwidth,
            'zcr': zcr,
            'mfcc': mfcc
        }
    
    def extract_features_for_model(self, audio: np.ndarray) -> torch.Tensor:
        """
        Extract and combine FFT features into a single tensor for model input
        
        Args:
            audio: Audio waveform (1D numpy array)
            
        Returns:
            Tensor of shape (channels, height, width) suitable for CNN
        """
        features = self.compute_fft_features(audio)
        
        # Stack time-frequency representations
        # Use mel spectrogram as primary feature
        mel_spec = features['mel_spec_db']
        
        # Normalize mel spectrogram
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        # Convert to tensor and add channel dimension
        tensor = torch.FloatTensor(mel_spec).unsqueeze(0)
        
        return tensor
    
    def extract_multi_channel_features(self, audio: np.ndarray) -> torch.Tensor:
        """
        Extract multiple feature channels for enhanced CNN input
        
        Args:
            audio: Audio waveform (1D numpy array)
            
        Returns:
            Tensor of shape (channels, height, width) with multiple feature channels
        """
        features = self.compute_fft_features(audio)
        
        # Prepare multiple channels
        channels = []
        
        # Channel 1: Mel spectrogram
        mel_spec = features['mel_spec_db']
        mel_spec_norm = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        channels.append(mel_spec_norm)
        
        # Channel 2: MFCC
        mfcc = features['mfcc']
        # Resize MFCC to match mel spectrogram height if needed
        if mfcc.shape[0] != mel_spec.shape[0]:
            # Simple interpolation to resize
            import cv2
            mfcc_resized = cv2.resize(mfcc, (mfcc.shape[1], mel_spec.shape[0]))
            mfcc = mfcc_resized
        mfcc_norm = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        channels.append(mfcc_norm)
        
        # Ensure all channels have the same width
        min_width = min(ch.shape[1] for ch in channels)
        channels = [ch[:, :min_width] for ch in channels]
        
        # Stack channels
        multi_channel = np.stack(channels, axis=0)
        tensor = torch.FloatTensor(multi_channel)
        
        return tensor


class FFTStatisticalFeatures:
    """Extract statistical features from FFT for classical ML or as additional DNN input"""
    
    def __init__(self, n_fft: int = 2048, sample_rate: int = 16000):
        self.n_fft = n_fft
        self.sample_rate = sample_rate
    
    def extract_statistical_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from FFT spectrum
        
        Returns:
            1D feature vector
        """
        # Compute FFT
        fft = np.fft.rfft(audio, n=self.n_fft)
        magnitude = np.abs(fft)
        power = magnitude ** 2
        
        # Frequency bins
        freqs = np.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate)
        
        # Statistical features
        features = []
        
        # Spectral statistics
        features.append(np.mean(magnitude))
        features.append(np.std(magnitude))
        features.append(np.max(magnitude))
        features.append(np.min(magnitude))
        features.append(np.median(magnitude))
        
        # Spectral centroid (weighted mean of frequencies)
        spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-8)
        features.append(spectral_centroid)
        
        # Spectral spread (standard deviation around centroid)
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / (np.sum(magnitude) + 1e-8))
        features.append(spectral_spread)
        
        # Spectral skewness
        spectral_skewness = np.sum(((freqs - spectral_centroid) ** 3) * magnitude) / (
            np.sum(magnitude) * (spectral_spread ** 3 + 1e-8)
        )
        features.append(spectral_skewness)
        
        # Spectral kurtosis
        spectral_kurtosis = np.sum(((freqs - spectral_centroid) ** 4) * magnitude) / (
            np.sum(magnitude) * (spectral_spread ** 4 + 1e-8)
        )
        features.append(spectral_kurtosis)
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumulative_energy = np.cumsum(power)
        total_energy = cumulative_energy[-1]
        rolloff_idx = np.where(cumulative_energy >= 0.85 * total_energy)[0]
        if len(rolloff_idx) > 0:
            spectral_rolloff = freqs[rolloff_idx[0]]
        else:
            spectral_rolloff = freqs[-1]
        features.append(spectral_rolloff)
        
        # Spectral flatness (measure of noisiness)
        geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-8)))
        arithmetic_mean = np.mean(magnitude)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-8)
        features.append(spectral_flatness)
        
        # Power in different frequency bands
        # Low: 0-500 Hz
        low_freq_mask = freqs < 500
        features.append(np.sum(power[low_freq_mask]))
        
        # Mid: 500-2000 Hz
        mid_freq_mask = (freqs >= 500) & (freqs < 2000)
        features.append(np.sum(power[mid_freq_mask]))
        
        # High: 2000+ Hz
        high_freq_mask = freqs >= 2000
        features.append(np.sum(power[high_freq_mask]))
        
        return np.array(features, dtype=np.float32)
