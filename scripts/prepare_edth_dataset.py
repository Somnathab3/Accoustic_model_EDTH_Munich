"""
Prepare EDTH Munich dataset for 3-class training (drone, helicopter, background).
Creates metadata CSV files compatible with the training pipeline.
Usage: python scripts/prepare_edth_dataset.py
"""
import pandas as pd
from pathlib import Path
import argparse

def create_metadata(data_dir: Path, split: str) -> pd.DataFrame:
    """
    Create metadata CSV for the given split (train/val).
    
    Args:
        data_dir: Path to data directory containing split folders
        split: 'train' or 'val'
    
    Returns:
        DataFrame with columns: file_path, label, class_name
    """
    split_dir = data_dir / split
    
    metadata = []
    
    # Define class mapping: 0=background, 1=drone, 2=helicopter
    class_mapping = {
        'background': 0,
        'drone': 1,
        'helicopter': 2
    }
    
    for class_name, label in class_mapping.items():
        class_dir = split_dir / class_name
        
        if not class_dir.exists():
            print(f"⚠️  Warning: {class_dir} does not exist, skipping...")
            continue
        
        # Find all WAV files
        audio_files = list(class_dir.glob('*.wav'))
        
        print(f"📂 Class '{class_name}' (label={label}): {len(audio_files)} files")
        
        for audio_file in audio_files:
            metadata.append({
                'path': str(audio_file.absolute()),
                'label': class_name  # Use class name as string label for compatibility
            })
    
    df = pd.DataFrame(metadata)
    return df

def main():
    parser = argparse.ArgumentParser(description='Prepare EDTH Munich dataset')
    parser.add_argument('--data-dir', type=str, 
                        default='data/edth_munich_dataset/data',
                        help='Path to EDTH dataset directory')
    parser.add_argument('--output-dir', type=str,
                        default='data/edth_prepared',
                        help='Output directory for metadata CSV files')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("🎯 EDTH Munich Dataset Preparation")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Process training set
    print("=" * 70)
    print("📊 Processing Training Set")
    print("=" * 70)
    train_df = create_metadata(data_dir, 'train')
    train_csv = output_dir / 'metadata_train.csv'
    train_df.to_csv(train_csv, index=False)
    print(f"\n✅ Saved training metadata: {train_csv}")
    print(f"   Total samples: {len(train_df)}\n")
    
    # Process validation set
    print("=" * 70)
    print("📊 Processing Validation Set")
    print("=" * 70)
    val_df = create_metadata(data_dir, 'val')
    val_csv = output_dir / 'metadata_val.csv'
    val_df.to_csv(val_csv, index=False)
    print(f"\n✅ Saved validation metadata: {val_csv}")
    print(f"   Total samples: {len(val_df)}\n")
    
    # Create labels mapping
    labels = {
        'labels': ['background', 'drone', 'helicopter']
    }
    
    import json
    labels_path = output_dir / 'labels.json'
    with open(labels_path, 'w') as f:
        json.dump(labels, f, indent=2)
    print(f"✅ Saved labels mapping: {labels_path}\n")
    
    # Print summary
    print("=" * 70)
    print("📊 Dataset Summary")
    print("=" * 70)
    print(f"Training samples: {len(train_df)}")
    for class_name in ['background', 'drone', 'helicopter']:
        count = len(train_df[train_df['label'] == class_name])
        print(f"  - {class_name}: {count}")
    
    print(f"\nValidation samples: {len(val_df)}")
    for class_name in ['background', 'drone', 'helicopter']:
        count = len(val_df[val_df['label'] == class_name])
        print(f"  - {class_name}: {count}")
    
    print("\n" + "=" * 70)
    print("✅ Dataset preparation complete!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"1. Review the metadata files in {output_dir}")
    print(f"2. Update configs/train_edth.yaml with the new paths")
    print(f"3. Run training: python -m src.adrone.train --config configs/train_edth.yaml")

if __name__ == '__main__':
    main()
