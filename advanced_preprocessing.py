"""
Advanced Audio Preprocessing for Acoustic Drone Detection
Best-in-class features combining multiple representations
"""

import numpy as np
import librosa
import torch
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AudioPreprocessor:
    """
    Advanced audio preprocessing combining multiple acoustic features:
    - Mel Spectrograms (time-frequency representation)
    - MFCCs (cepstral features)
    - Spectral features (contrast, bandwidth, rolloff)
    - Delta and delta-delta features (temporal dynamics)
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        duration: float = 3.0,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mfcc: int = 40,
        fmin: float = 20,
        fmax: float = 8000,
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.fmin = fmin
        self.fmax = fmax
        
    def load_audio(self, audio_path: str) -> np.ndarray:
        """Load and preprocess audio file"""
        try:
            # Load audio at target sample rate
            audio, _ = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            
            # Normalize audio
            if len(audio) > 0:
                audio = librosa.util.normalize(audio)
            
            # Pad or trim to fixed length
            if len(audio) < self.n_samples:
                audio = np.pad(audio, (0, self.n_samples - len(audio)), mode='constant')
            else:
                audio = audio[:self.n_samples]
                
            return audio
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return np.zeros(self.n_samples)
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel spectrogram - captures time-frequency representation
        Good for capturing harmonic content and frequency patterns
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        return mel_spec_db
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCCs - captures timbral texture
        Excellent for distinguishing different sound sources
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Add delta features (first derivative - velocity)
        mfcc_delta = librosa.feature.delta(mfcc)
        
        # Add delta-delta features (second derivative - acceleration)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Stack all MFCC features
        mfcc_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        
        # Normalize
        mfcc_features = (mfcc_features - mfcc_features.mean()) / (mfcc_features.std() + 1e-8)
        
        return mfcc_features
    
    def extract_spectral_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract spectral features - captures spectral characteristics
        """
        # Spectral contrast - distinguishes peaks and valleys in spectrum
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Spectral rolloff - frequency below which 85% of energy is contained
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Spectral bandwidth - width of spectral content
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Stack features
        spectral_features = np.vstack([
            spectral_contrast,
            spectral_rolloff,
            spectral_bandwidth
        ])
        
        # Normalize
        spectral_features = (spectral_features - spectral_features.mean()) / (spectral_features.std() + 1e-8)
        
        return spectral_features
    
    def extract_all_features(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract all features from audio file
        Returns: (mel_spec, mfcc, spectral)
        """
        # Load audio
        audio = self.load_audio(audio_path)
        
        # Extract all features
        mel_spec = self.extract_mel_spectrogram(audio)
        mfcc = self.extract_mfcc(audio)
        spectral = self.extract_spectral_features(audio)
        
        return mel_spec, mfcc, spectral
    
    def extract_combined_features(self, audio_path: str) -> np.ndarray:
        """
        Extract and combine all features into a single representation
        Returns: Combined feature tensor [channels, height, width]
        """
        mel_spec, mfcc, spectral = self.extract_all_features(audio_path)
        
        # Resize all features to the same temporal dimension
        target_width = mel_spec.shape[1]
        
        # Resize MFCC if needed
        if mfcc.shape[1] != target_width:
            mfcc = librosa.util.fix_length(mfcc, size=target_width, axis=1)
        
        # Resize spectral features if needed
        if spectral.shape[1] != target_width:
            spectral = librosa.util.fix_length(spectral, size=target_width, axis=1)
        
        # Stack as separate channels (similar to RGB channels in images)
        # Channel 1: Mel spectrogram (frequency content)
        # Channel 2: MFCCs with deltas (timbral features)
        # Channel 3: Spectral features (spectral characteristics)
        
        # Pad MFCC to match mel_spec height if needed
        if mfcc.shape[0] < mel_spec.shape[0]:
            pad_height = mel_spec.shape[0] - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_height), (0, 0)), mode='constant')
        elif mfcc.shape[0] > mel_spec.shape[0]:
            mfcc = mfcc[:mel_spec.shape[0], :]
            
        # Pad spectral to match mel_spec height if needed
        if spectral.shape[0] < mel_spec.shape[0]:
            pad_height = mel_spec.shape[0] - spectral.shape[0]
            spectral = np.pad(spectral, ((0, pad_height), (0, 0)), mode='constant')
        elif spectral.shape[0] > mel_spec.shape[0]:
            spectral = spectral[:mel_spec.shape[0], :]
        
        # Stack as 3-channel image: [3, height, width]
        combined = np.stack([mel_spec, mfcc, spectral], axis=0)
        
        return combined


class AudioAugmenter:
    """
    Data augmentation for audio to improve model robustness
    """
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def add_noise(self, audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        """Add random Gaussian noise"""
        noise = np.random.randn(len(audio)) * noise_factor
        return audio + noise
    
    def time_shift(self, audio: np.ndarray, shift_max: float = 0.2) -> np.ndarray:
        """Shift audio in time"""
        shift = int(np.random.uniform(-shift_max, shift_max) * len(audio))
        return np.roll(audio, shift)
    
    def pitch_shift(self, audio: np.ndarray, n_steps: float = None) -> np.ndarray:
        """Shift pitch up or down"""
        if n_steps is None:
            n_steps = np.random.uniform(-2, 2)
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
    
    def time_stretch(self, audio: np.ndarray, rate: float = None) -> np.ndarray:
        """Stretch or compress audio in time"""
        if rate is None:
            rate = np.random.uniform(0.8, 1.2)
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def random_augment(self, audio: np.ndarray, p: float = 0.5) -> np.ndarray:
        """Apply random augmentations with probability p"""
        if np.random.random() < p:
            # Choose random augmentation
            aug_type = np.random.choice(['noise', 'shift', 'pitch', 'stretch'])
            
            if aug_type == 'noise':
                audio = self.add_noise(audio)
            elif aug_type == 'shift':
                audio = self.time_shift(audio)
            elif aug_type == 'pitch':
                audio = self.pitch_shift(audio)
            elif aug_type == 'stretch':
                audio = self.time_stretch(audio)
                # Ensure correct length after stretch
                target_len = len(audio)
                if len(audio) < target_len:
                    audio = np.pad(audio, (0, target_len - len(audio)))
                else:
                    audio = audio[:target_len]
        
        return audio


def visualize_features(audio_path: str, save_path: Optional[str] = None):
    """
    Visualize all extracted features for a given audio file
    """
    import matplotlib.pyplot as plt
    
    preprocessor = AudioPreprocessor()
    mel_spec, mfcc, spectral = preprocessor.extract_all_features(audio_path)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot mel spectrogram
    img1 = axes[0].imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Mel Spectrogram')
    axes[0].set_ylabel('Mel Frequency')
    axes[0].set_xlabel('Time Frames')
    plt.colorbar(img1, ax=axes[0])
    
    # Plot MFCC
    img2 = axes[1].imshow(mfcc, aspect='auto', origin='lower', cmap='coolwarm')
    axes[1].set_title('MFCC + Deltas')
    axes[1].set_ylabel('MFCC Coefficients')
    axes[1].set_xlabel('Time Frames')
    plt.colorbar(img2, ax=axes[1])
    
    # Plot spectral features
    img3 = axes[2].imshow(spectral, aspect='auto', origin='lower', cmap='plasma')
    axes[2].set_title('Spectral Features')
    axes[2].set_ylabel('Feature Index')
    axes[2].set_xlabel('Time Frames')
    plt.colorbar(img3, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Test the preprocessor
    import sys
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        print(f"Testing preprocessor on: {audio_path}")
        
        preprocessor = AudioPreprocessor()
        features = preprocessor.extract_combined_features(audio_path)
        
        print(f"Combined features shape: {features.shape}")
        print(f"  - Channels: {features.shape[0]}")
        print(f"  - Height: {features.shape[1]}")
        print(f"  - Width: {features.shape[2]}")
        
        visualize_features(audio_path, save_path='feature_visualization.png')
        print("Saved visualization to feature_visualization.png")
    else:
        print("Usage: python advanced_preprocessing.py <audio_file.wav>")
