"""
Dataset class for EDTH Munich Acoustic Drone Detection
Handles loading and preprocessing of audio data
"""
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, List
import json


class AcousticDroneDataset(Dataset):
    """
    Dataset for acoustic drone detection
    
    Directory structure:
        root_dir/
            background/
                sample1.wav
                sample2.wav
            drone/
                sample1.wav
            helicopter/
                sample1.wav
    """
    
    def __init__(
        self,
        root_dir: str,
        preprocessor,
        augmentation_pipeline: Optional = None,
        is_training: bool = True,
        max_samples_per_class: Optional[int] = None
    ):
        """
        Args:
            root_dir: Root directory containing class subdirectories
            preprocessor: AudioPreprocessor instance
            augmentation_pipeline: AugmentationPipeline instance for training
            is_training: Whether this is training data (enables augmentation)
            max_samples_per_class: Limit samples per class (for debugging)
        """
        self.root_dir = Path(root_dir)
        self.preprocessor = preprocessor
        self.augmentation_pipeline = augmentation_pipeline if is_training else None
        self.is_training = is_training
        
        # Class mapping
        self.class_to_idx = {
            'background': 0,
            'drone': 1,
            'helicopter': 2
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Load file paths and labels
        self.samples = []
        self._load_samples(max_samples_per_class)
        
        print(f"Loaded {len(self.samples)} samples from {root_dir}")
        self._print_class_distribution()
    
    def _load_samples(self, max_samples: Optional[int] = None):
        """Load all audio file paths and their labels"""
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist, skipping")
                continue
            
            # Get all .wav files
            audio_files = list(class_dir.glob('*.wav'))
            
            # Limit samples if specified
            if max_samples is not None:
                audio_files = audio_files[:max_samples]
            
            for audio_path in audio_files:
                self.samples.append((str(audio_path), class_idx))
    
    def _print_class_distribution(self):
        """Print class distribution statistics"""
        class_counts = {name: 0 for name in self.class_to_idx.keys()}
        
        for _, label in self.samples:
            class_name = self.idx_to_class[label]
            class_counts[class_name] += 1
        
        print("Class distribution:")
        for class_name, count in class_counts.items():
            percentage = 100.0 * count / len(self.samples) if self.samples else 0
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample
        
        Returns:
            spectrogram: (channels, n_mels, time) tensor
            label: class index
        """
        audio_path, label = self.samples[idx]
        
        # Load audio
        waveform = self.preprocessor.load_audio(audio_path)
        
        # Apply waveform augmentations (if training)
        if self.augmentation_pipeline is not None and self.is_training:
            waveform = self.augmentation_pipeline.augment_waveform(waveform)
        
        # Convert to spectrogram
        spectrogram = self.preprocessor(waveform)
        
        # Apply spectrogram augmentations (if training)
        if self.augmentation_pipeline is not None and self.is_training:
            spectrogram = self.augmentation_pipeline.augment_spectrogram(spectrogram)
        
        return spectrogram, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for balanced training
        Uses inverse effective number weighting
        """
        class_counts = torch.zeros(len(self.class_to_idx))
        
        for _, label in self.samples:
            class_counts[label] += 1
        
        # Inverse frequency weighting
        total_samples = len(self.samples)
        weights = total_samples / (len(self.class_to_idx) * class_counts)
        
        # Normalize
        weights = weights / weights.sum() * len(self.class_to_idx)
        
        return weights
    
    def get_samples_per_class(self) -> torch.Tensor:
        """
        Get number of samples per class for class-balanced loss
        
        Returns:
            Tensor of shape (num_classes,) with sample counts
        """
        class_counts = torch.zeros(len(self.class_to_idx))
        
        for _, label in self.samples:
            class_counts[label] += 1
        
        return class_counts
    
    def save_class_mapping(self, output_path: str):
        """Save class to index mapping as JSON"""
        mapping = {
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class
        }
        
        with open(output_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        print(f"Saved class mapping to {output_path}")


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    preprocessor,
    augmentation_pipeline,
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples_per_class: Optional[int] = None,
    use_balanced_sampler: bool = False
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.Tensor, torch.Tensor]:
    """
    Create training and validation dataloaders
    
    Args:
        use_balanced_sampler: If True, use WeightedRandomSampler for balanced mini-batches
    
    Returns:
        train_loader, val_loader, class_weights, samples_per_class
    """
    # Create datasets
    train_dataset = AcousticDroneDataset(
        root_dir=train_dir,
        preprocessor=preprocessor,
        augmentation_pipeline=augmentation_pipeline,
        is_training=True,
        max_samples_per_class=max_samples_per_class
    )
    
    val_dataset = AcousticDroneDataset(
        root_dir=val_dir,
        preprocessor=preprocessor,
        augmentation_pipeline=None,  # No augmentation for validation
        is_training=False,
        max_samples_per_class=max_samples_per_class
    )
    
    # Get class weights from training set
    class_weights = train_dataset.get_class_weights()
    
    # Get samples per class for class-balanced loss
    samples_per_class = train_dataset.get_samples_per_class()
    
    # Create balanced sampler if requested
    sampler = None
    shuffle = True
    
    if use_balanced_sampler:
        # Compute sample weights (inverse of class frequency)
        sample_weights = torch.zeros(len(train_dataset))
        for idx, (_, label) in enumerate(train_dataset.samples):
            sample_weights[idx] = class_weights[label]
        
        # Create weighted random sampler
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        shuffle = False  # Sampler handles shuffling
        
        print(f"✓ Using balanced sampler for training")
    
    # Determine pin_memory based on CUDA availability
    use_pin_memory = torch.cuda.is_available()
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True  # For stable batch norm
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    return train_loader, val_loader, class_weights, samples_per_class


if __name__ == '__main__':
    # Test dataset loading
    from adrone.preprocessing import AudioPreprocessor, AugmentationPipeline
    
    preprocessor = AudioPreprocessor(use_hpss=True)
    augmentation = AugmentationPipeline()
    
    dataset = AcousticDroneDataset(
        root_dir='data/edth_munich_dataset/data/train',
        preprocessor=preprocessor,
        augmentation_pipeline=augmentation,
        is_training=True,
        max_samples_per_class=10  # Small test
    )
    
    # Test loading a sample
    if len(dataset) > 0:
        spec, label = dataset[0]
        print(f"\nSample spectrogram shape: {spec.shape}")
        print(f"Sample label: {label} ({dataset.idx_to_class[label]})")
        
        # Test class weights
        weights = dataset.get_class_weights()
        print(f"\nClass weights: {weights}")
