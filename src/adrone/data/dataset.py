import csv, numpy as np, torch
from torch.utils.data import Dataset
from ..utils.audio_io import load_wav
from ..features.melspec import melspec
import json
from pathlib import Path
import librosa

class MelDataset(Dataset):
    def __init__(self, csv_path, labels_json, sample_rate=16000, n_mels=64, n_fft=1024, hop_length=320, window_sec=2.0):
        self.items = []
        with open(csv_path, "r") as f:
            r = csv.DictReader(f)
            self.items = list(r)
        with open(labels_json, "r") as f:
            self.labels = json.load(f)["labels"]
        self.lab2idx = {l:i for i,l in enumerate(self.labels)}
        self.sr = sample_rate; self.n_mels=n_mels; self.n_fft=n_fft; self.hop_length=hop_length
        self.window = int(window_sec * sample_rate)

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        y, sr = load_wav(it["path"], target_sr=self.sr)
        if len(y) < self.window:
            pad = self.window - len(y)
            y = np.pad(y, (0,pad))
        else:
            y = y[:self.window]
        S = melspec(y, self.sr, self.n_mels, self.n_fft, self.hop_length)
        x = torch.from_numpy(S).unsqueeze(0)  # [1, n_mels, T]
        y_idx = self.lab2idx[it["label"]]
        return x, torch.tensor(y_idx, dtype=torch.long)


class AudioDataset(Dataset):
    """
    Audio dataset that works with FFT preprocessing for directory-based data
    """
    def __init__(self, data_dir, fft_processor=None, max_duration=2.0, sample_rate=16000):
        """
        Args:
            data_dir: Directory containing class subdirectories with audio files
            fft_processor: FFTProcessor instance for feature extraction
            max_duration: Maximum audio duration in seconds
            sample_rate: Target sample rate
        """
        self.data_dir = Path(data_dir)
        self.fft_processor = fft_processor
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * max_duration)
        
        # Find all audio files organized by class
        self.samples = []
        self.classes = []
        
        # Scan directory structure
        for class_dir in sorted(self.data_dir.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                if class_name not in self.classes:
                    self.classes.append(class_name)
                
                class_idx = self.classes.index(class_name)
                
                # Find all audio files in this class
                for audio_file in class_dir.glob('*.wav'):
                    self.samples.append((str(audio_file), class_idx))
        
        print(f"Found {len(self.samples)} audio files in {len(self.classes)} classes")
        print(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.max_duration)
        
        # Pad or truncate
        if len(y) < self.window_samples:
            y = np.pad(y, (0, self.window_samples - len(y)), mode='constant')
        else:
            y = y[:self.window_samples]
        
        # Extract features using FFT processor
        if self.fft_processor is not None:
            features = self.fft_processor.extract_features_for_model(y)
        else:
            # Fallback to simple mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=self.sample_rate,
                n_mels=64,
                n_fft=1024,
                hop_length=320
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
            features = torch.FloatTensor(mel_spec_db).unsqueeze(0)
        
        return features, torch.tensor(label, dtype=torch.long)
