"""
Custom Dataset for Acoustic Drone Detection
Handles loading, preprocessing, and augmentation
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import json
from collections import Counter
from advanced_preprocessing import AudioPreprocessor, AudioAugmenter


class AcousticDroneDataset(Dataset):
    """
    PyTorch Dataset for acoustic drone detection
    
    Features:
    - Automatic directory structure scanning
    - Advanced preprocessing with multiple feature types
    - Data augmentation support
    - Class balancing utilities
    """
    
    def __init__(
        self,
        data_dir: str,
        preprocessor: AudioPreprocessor,
        augmenter: Optional[AudioAugmenter] = None,
        augment: bool = False,
        cache_features: bool = False
    ):
        """
        Args:
            data_dir: Root directory containing class subdirectories
            preprocessor: AudioPreprocessor instance
            augmenter: AudioAugmenter instance (optional)
            augment: Whether to apply augmentation
            cache_features: Whether to cache extracted features in memory
        """
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor
        self.augmenter = augmenter
        self.augment = augment
        self.cache_features = cache_features
        
        # Scan directory and build file list
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        self._scan_directory()
        
        # Feature cache
        self.feature_cache = {} if cache_features else None
        
        print(f"Dataset initialized:")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Classes: {self.classes}")
        print(f"  Class distribution: {self.get_class_distribution()}")
        print(f"  Augmentation: {'Enabled' if augment else 'Disabled'}")
        print(f"  Feature caching: {'Enabled' if cache_features else 'Disabled'}")
    
    def _scan_directory(self):
        """Scan directory structure and build sample list"""
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        # Find all class directories
        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        if len(class_dirs) == 0:
            raise ValueError(f"No class directories found in {self.data_dir}")
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            
            # Add class if not already present
            if class_name not in self.classes:
                class_idx = len(self.classes)
                self.classes.append(class_name)
                self.class_to_idx[class_name] = class_idx
                self.idx_to_class[class_idx] = class_name
            else:
                class_idx = self.class_to_idx[class_name]
            
            # Find all audio files in this class
            audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
            for ext in audio_extensions:
                for audio_file in class_dir.glob(f'*{ext}'):
                    self.samples.append({
                        'path': str(audio_file),
                        'class': class_name,
                        'class_idx': class_idx
                    })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample and its label
        
        Returns:
            features: Tensor of shape [3, height, width]
            label: Integer class index
        """
        sample = self.samples[idx]
        audio_path = sample['path']
        label = sample['class_idx']
        
        # Check cache first
        if self.feature_cache is not None and idx in self.feature_cache:
            features = self.feature_cache[idx]
        else:
            # Extract features
            features = self.preprocessor.extract_combined_features(audio_path)
            
            # Cache if enabled
            if self.feature_cache is not None:
                self.feature_cache[idx] = features
        
        # Convert to tensor
        features = torch.from_numpy(features).float()
        label = torch.tensor(label, dtype=torch.long)
        
        return features, label
    
    def get_class_distribution(self) -> dict:
        """Get the distribution of samples per class"""
        labels = [s['class'] for s in self.samples]
        return dict(Counter(labels))
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for balanced training
        Useful for weighted loss functions
        """
        class_counts = np.zeros(len(self.classes))
        for sample in self.samples:
            class_counts[sample['class_idx']] += 1
        
        # Inverse frequency weighting
        total_samples = len(self.samples)
        class_weights = total_samples / (len(self.classes) * class_counts)
        
        return torch.FloatTensor(class_weights)
    
    def get_sample_weights(self) -> List[float]:
        """
        Get per-sample weights for WeightedRandomSampler
        Balances the dataset by oversampling minority classes
        """
        class_weights = self.get_class_weights().numpy()
        sample_weights = [class_weights[s['class_idx']] for s in self.samples]
        return sample_weights
    
    def save_metadata(self, save_path: str):
        """Save dataset metadata to JSON"""
        metadata = {
            'num_samples': len(self.samples),
            'num_classes': len(self.classes),
            'classes': self.classes,
            'class_to_idx': self.class_to_idx,
            'class_distribution': self.get_class_distribution(),
            'data_dir': str(self.data_dir)
        }
        
        with open(save_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {save_path}")


def create_data_loaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampling: bool = True,
    augment_train: bool = True,
    cache_features: bool = False
) -> Tuple[DataLoader, DataLoader, AudioPreprocessor]:
    """
    Create train and validation data loaders
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        use_weighted_sampling: Whether to use weighted sampling for class balance
        augment_train: Whether to augment training data
        cache_features: Whether to cache features in memory
    
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        preprocessor: The preprocessor instance (needed for inference)
    """
    
    # Create preprocessor and augmenter
    preprocessor = AudioPreprocessor(
        sample_rate=22050,
        duration=3.0,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        n_mfcc=40
    )
    
    augmenter = AudioAugmenter(sample_rate=22050) if augment_train else None
    
    # Create datasets
    train_dataset = AcousticDroneDataset(
        data_dir=train_dir,
        preprocessor=preprocessor,
        augmenter=augmenter,
        augment=augment_train,
        cache_features=cache_features
    )
    
    val_dataset = AcousticDroneDataset(
        data_dir=val_dir,
        preprocessor=preprocessor,
        augmenter=None,
        augment=False,
        cache_features=cache_features
    )
    
    # Create samplers
    if use_weighted_sampling:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, preprocessor


def visualize_batch(dataloader: DataLoader, num_samples: int = 4):
    """
    Visualize a batch of samples from the dataloader
    """
    import matplotlib.pyplot as plt
    
    # Get a batch
    features, labels = next(iter(dataloader))
    
    # Get dataset to map labels to class names
    dataset = dataloader.dataset
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
    
    for i in range(min(num_samples, len(features))):
        feature = features[i].numpy()  # [3, height, width]
        label_idx = labels[i].item()
        class_name = dataset.idx_to_class[label_idx]
        
        # Plot each channel
        for j in range(3):
            ax = axes[i, j] if num_samples > 1 else axes[j]
            ax.imshow(feature[j], aspect='auto', origin='lower', cmap='viridis')
            
            if i == 0:
                channel_names = ['Mel Spectrogram', 'MFCC + Deltas', 'Spectral Features']
                ax.set_title(channel_names[j])
            
            if j == 0:
                ax.set_ylabel(f'{class_name}\n(Sample {i+1})')
    
    plt.tight_layout()
    plt.savefig('batch_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Batch visualization saved to batch_visualization.png")


def test_dataset():
    """Test the dataset implementation"""
    
    # Test with the EDTH Munich dataset
    train_dir = "f:/EDTH/acoustic-drone-detector/data/edth_munich_dataset/data/train"
    val_dir = "f:/EDTH/acoustic-drone-detector/data/edth_munich_dataset/data/val"
    
    print("Testing dataset implementation...")
    print("=" * 60)
    
    try:
        train_loader, val_loader, preprocessor = create_data_loaders(
            train_dir=train_dir,
            val_dir=val_dir,
            batch_size=8,
            num_workers=0,  # Use 0 for testing
            use_weighted_sampling=True,
            augment_train=True,
            cache_features=False
        )
        
        print("\nDataLoaders created successfully!")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        
        # Test loading a batch
        print("\nLoading a training batch...")
        features, labels = next(iter(train_loader))
        print(f"Batch features shape: {features.shape}")
        print(f"Batch labels shape: {labels.shape}")
        print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"Labels in batch: {labels.numpy()}")
        
        # Visualize batch
        print("\nVisualizing batch...")
        visualize_batch(train_loader, num_samples=4)
        
        # Save metadata
        train_loader.dataset.save_metadata('train_metadata.json')
        val_loader.dataset.save_metadata('val_metadata.json')
        
        print("\n✅ Dataset test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during dataset test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_dataset()
