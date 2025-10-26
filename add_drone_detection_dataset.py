"""
Split Drone-detection-dataset into train/val and integrate into combined_dataset
"""
import shutil
from pathlib import Path
import random
from tqdm import tqdm

def split_and_integrate_dataset():
    """
    Split Drone-detection-dataset (90 samples: 30 each class) into:
    - Train: 80% (24 samples per class = 72 total)
    - Val: 20% (6 samples per class = 18 total)
    
    Then copy to combined_dataset structure
    """
    
    # Paths
    source_dir = Path('data/Drone-detection-dataset')
    combined_train = Path('data/combined_dataset/train')
    combined_val = Path('data/combined_dataset/val')
    
    # Ensure combined dataset exists
    for split in [combined_train, combined_val]:
        for class_name in ['background', 'drone', 'helicopter']:
            (split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Classes and their file prefixes
    classes = {
        'background': 'BACKGROUND',
        'drone': 'DRONE',
        'helicopter': 'HELICOPTER'
    }
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Split ratio
    train_ratio = 0.8
    
    print("="*70)
    print("SPLITTING AND INTEGRATING DRONE-DETECTION-DATASET")
    print("="*70)
    
    total_train_added = 0
    total_val_added = 0
    
    for class_name, prefix in classes.items():
        # Get all files for this class
        files = sorted(list(source_dir.glob(f'{prefix}_*.wav')))
        
        print(f"\n{class_name.upper()}:")
        print(f"  Total files: {len(files)}")
        
        # Shuffle
        random.shuffle(files)
        
        # Split
        n_train = int(len(files) * train_ratio)
        train_files = files[:n_train]
        val_files = files[n_train:]
        
        print(f"  Train: {len(train_files)} files")
        print(f"  Val: {len(val_files)} files")
        
        # Copy train files
        print(f"  Copying train files...")
        for file_path in tqdm(train_files, desc=f"    Train", leave=False):
            dest = combined_train / class_name / file_path.name
            shutil.copy2(file_path, dest)
        
        # Copy val files
        print(f"  Copying val files...")
        for file_path in tqdm(val_files, desc=f"    Val", leave=False):
            dest = combined_val / class_name / file_path.name
            shutil.copy2(file_path, dest)
        
        total_train_added += len(train_files)
        total_val_added += len(val_files)
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    print(f"Added to combined_dataset/train: {total_train_added} files")
    print(f"Added to combined_dataset/val: {total_val_added} files")
    
    # Count total files now in combined dataset
    print(f"\n{'='*70}")
    print("UPDATED COMBINED DATASET STATISTICS")
    print('='*70)
    
    for split_name, split_path in [('Train', combined_train), ('Val', combined_val)]:
        print(f"\n{split_name}:")
        total = 0
        for class_name in ['background', 'drone', 'helicopter']:
            class_dir = split_path / class_name
            n_files = len(list(class_dir.glob('*.wav')))
            print(f"  {class_name.capitalize():11s}: {n_files:3d} files")
            total += n_files
        print(f"  {'Total':11s}: {total:3d} files")
    
    print(f"\n{'='*70}")
    print("✓ Integration complete!")
    print(f"  Train dir: {combined_train}")
    print(f"  Val dir: {combined_val}")
    print("\nYou can now retrain models using:")
    print("  python train_sota_model.py \\")
    print("    --train-dir data/combined_dataset/train \\")
    print("    --val-dir data/combined_dataset/val \\")
    print("    --output-dir models/crnn_combined_v2")
    print('='*70)


if __name__ == '__main__':
    split_and_integrate_dataset()
