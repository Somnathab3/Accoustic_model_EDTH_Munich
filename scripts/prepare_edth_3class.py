"""
Prepare EDTH Munich 3-class dataset (drone, helicopter, background)
"""
import csv
from pathlib import Path
import json

def prepare_edth_3class():
    # Paths
    data_root = Path("data/edth_munich_dataset/data")
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    
    output_dir = Path("data/edth_prepared")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Define classes
    classes = ["background", "drone", "helicopter"]
    
    # Create labels JSON
    labels_data = {"labels": classes}
    labels_path = output_dir / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels_data, f, indent=2)
    print(f"✅ Created labels file: {labels_path}")
    
    # Create training CSV
    train_csv = output_dir / "metadata_train.csv"
    with open(train_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        
        for class_name in classes:
            class_dir = train_dir / class_name
            if not class_dir.exists():
                print(f"⚠️  Warning: {class_dir} does not exist")
                continue
            
            audio_files = list(class_dir.glob("*.wav"))
            print(f"📁 {class_name} (train): {len(audio_files)} files")
            
            for audio_file in audio_files:
                writer.writerow([str(audio_file.absolute()), class_name])
    
    print(f"✅ Created training CSV: {train_csv}")
    
    # Create validation CSV
    val_csv = output_dir / "metadata_val.csv"
    with open(val_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        
        for class_name in classes:
            class_dir = val_dir / class_name
            if not class_dir.exists():
                print(f"⚠️  Warning: {class_dir} does not exist")
                continue
            
            audio_files = list(class_dir.glob("*.wav"))
            print(f"📁 {class_name} (val): {len(audio_files)} files")
            
            for audio_file in audio_files:
                writer.writerow([str(audio_file.absolute()), class_name])
    
    print(f"✅ Created validation CSV: {val_csv}")
    
    # Print summary
    print("\n" + "="*70)
    print("📊 EDTH 3-Class Dataset Summary")
    print("="*70)
    print(f"Classes: {', '.join(classes)}")
    print(f"Training samples: {len(classes) * 180} (180 per class)")
    print(f"Validation samples: {len(classes) * 60} (60 per class)")
    print(f"\n📍 Output files:")
    print(f"   Labels: {labels_path}")
    print(f"   Train CSV: {train_csv}")
    print(f"   Val CSV: {val_csv}")
    print("="*70)

if __name__ == "__main__":
    prepare_edth_3class()
