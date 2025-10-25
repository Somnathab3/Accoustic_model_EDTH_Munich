"""
Combine original dataset with challenge results for comprehensive training
"""
import shutil
from pathlib import Path
from tqdm import tqdm

# Paths
ORIGINAL_TRAIN = Path("F:/EDTH/acoustic-drone-detector/data/edth_munich_dataset/data/train")
ORIGINAL_VAL = Path("F:/EDTH/acoustic-drone-detector/data/edth_munich_dataset/data/val")
CHALLENGE_TRAIN = Path("F:/EDTH/acoustic-drone-detector/data/edth_prepared/train")
CHALLENGE_VAL = Path("F:/EDTH/acoustic-drone-detector/data/edth_prepared/val")

# Output paths
COMBINED_DIR = Path("F:/EDTH/acoustic-drone-detector/data/combined_dataset")
COMBINED_TRAIN = COMBINED_DIR / "train"
COMBINED_VAL = COMBINED_DIR / "val"

def combine_datasets():
    """Combine original and challenge datasets"""
    print("="*80)
    print("COMBINING DATASETS: Original + Challenge Samples")
    print("="*80)
    
    # Create directories
    for label in ['background', 'drone', 'helicopter']:
        (COMBINED_TRAIN / label).mkdir(parents=True, exist_ok=True)
        (COMBINED_VAL / label).mkdir(parents=True, exist_ok=True)
    
    stats = {
        'train': {'background': 0, 'drone': 0, 'helicopter': 0},
        'val': {'background': 0, 'drone': 0, 'helicopter': 0}
    }
    
    # Copy original training data
    print("\n📦 Copying original training data...")
    for label in ['background', 'drone', 'helicopter']:
        src_dir = ORIGINAL_TRAIN / label
        dst_dir = COMBINED_TRAIN / label
        
        if src_dir.exists():
            files = list(src_dir.glob('*.wav'))
            for file in tqdm(files, desc=f"  {label}"):
                dst = dst_dir / file.name
                if not dst.exists():
                    shutil.copy2(file, dst)
                    stats['train'][label] += 1
    
    # Copy challenge training data
    print("\n📦 Copying challenge training data...")
    for label in ['background', 'drone', 'helicopter']:
        src_dir = CHALLENGE_TRAIN / label
        dst_dir = COMBINED_TRAIN / label
        
        if src_dir.exists():
            files = list(src_dir.glob('*.wav'))
            for file in tqdm(files, desc=f"  {label}"):
                dst = dst_dir / file.name
                if not dst.exists():
                    shutil.copy2(file, dst)
                    stats['train'][label] += 1
                else:
                    print(f"    ⊙ Skipped duplicate: {file.name}")
    
    # Copy original validation data
    print("\n📦 Copying original validation data...")
    for label in ['background', 'drone', 'helicopter']:
        src_dir = ORIGINAL_VAL / label
        dst_dir = COMBINED_VAL / label
        
        if src_dir.exists():
            files = list(src_dir.glob('*.wav'))
            for file in tqdm(files, desc=f"  {label}"):
                dst = dst_dir / file.name
                if not dst.exists():
                    shutil.copy2(file, dst)
                    stats['val'][label] += 1
    
    # Copy challenge validation data
    print("\n📦 Copying challenge validation data...")
    for label in ['background', 'drone', 'helicopter']:
        src_dir = CHALLENGE_VAL / label
        dst_dir = COMBINED_VAL / label
        
        if src_dir.exists():
            files = list(src_dir.glob('*.wav'))
            for file in tqdm(files, desc=f"  {label}"):
                dst = dst_dir / file.name
                if not dst.exists():
                    shutil.copy2(file, dst)
                    stats['val'][label] += 1
                else:
                    print(f"    ⊙ Skipped duplicate: {file.name}")
    
    # Print summary
    print("\n" + "="*80)
    print("COMBINED DATASET SUMMARY")
    print("="*80)
    
    print("\n📊 Training Set:")
    for label, count in sorted(stats['train'].items()):
        print(f"  {label:12s}: {count:4d} samples")
    print(f"  {'TOTAL':12s}: {sum(stats['train'].values()):4d} samples")
    
    print("\n📊 Validation Set:")
    for label, count in sorted(stats['val'].items()):
        print(f"  {label:12s}: {count:4d} samples")
    print(f"  {'TOTAL':12s}: {sum(stats['val'].values()):4d} samples")
    
    print("\n" + "="*80)
    print("✓ Dataset combination complete!")
    print(f"  Combined dataset location: {COMBINED_DIR}")
    print("="*80)
    
    print("\n📝 Next step - Train with combined dataset:")
    print("\npython train_sota_model.py \\")
    print("  --train-dir data/combined_dataset/train \\")
    print("  --val-dir data/combined_dataset/val \\")
    print("  --model-type panns \\")
    print("  --use-hpss \\")
    print("  --epochs 100 \\")
    print("  --batch-size 32 \\")
    print("  --use-focal-loss \\")
    print("  --use-class-weights \\")
    print("  --output-dir models/panns_combined")

if __name__ == '__main__':
    combine_datasets()
