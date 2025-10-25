"""
Test Continuous Training Pipeline Setup
Verifies all components are ready before starting
"""
from pathlib import Path
import json
import sys

def check_file(path: Path, description: str, required: bool = True) -> bool:
    """Check if a file exists"""
    exists = path.exists()
    status = "✓" if exists else ("✗" if required else "⚠️")
    print(f"  {status} {description}: {path}")
    if not exists and required:
        print(f"      ERROR: Required file not found!")
        return False
    return True

def check_dir(path: Path, description: str, required: bool = True) -> bool:
    """Check if a directory exists"""
    exists = path.exists() and path.is_dir()
    status = "✓" if exists else ("✗" if required else "⚠️")
    print(f"  {status} {description}: {path}")
    if not exists and required:
        print(f"      ERROR: Required directory not found!")
        return False
    return True

def main():
    print("="*80)
    print("CONTINUOUS TRAINING PIPELINE - SETUP CHECK")
    print("="*80)
    
    all_ok = True
    
    # Check model files
    print("\n📦 Model Files:")
    all_ok &= check_file(
        Path("models/crnn_combined/crnn_final.pt"),
        "Final model (used by bot)",
        required=True
    )
    all_ok &= check_file(
        Path("models/crnn_combined/best_model.pt"),
        "Best checkpoint (for retraining)",
        required=True
    )
    all_ok &= check_file(
        Path("models/crnn_combined/labels.json"),
        "Labels file",
        required=True
    )
    check_file(
        Path("models/crnn_combined/training_history.json"),
        "Training history",
        required=False
    )
    
    # Check data directories
    print("\n📁 Data Directories:")
    all_ok &= check_dir(
        Path("data/edth_munich_dataset/data/train"),
        "Original training data",
        required=True
    )
    all_ok &= check_dir(
        Path("data/edth_munich_dataset/data/val"),
        "Original validation data",
        required=True
    )
    
    # Check combined dataset (will be created if doesn't exist)
    check_dir(
        Path("data/combined_dataset"),
        "Combined dataset (will be created)",
        required=False
    )
    
    # Check challenge results
    print("\n🎯 Challenge Results:")
    check_dir(
        Path("challenge_results"),
        "Challenge results directory",
        required=False
    )
    check_file(
        Path("challenge_results/results.csv"),
        "Results CSV (will be created by bot)",
        required=False
    )
    check_file(
        Path("challenge_results/results.jsonl"),
        "Results JSONL (will be created by bot)",
        required=False
    )
    check_dir(
        Path("challenge_results/audio_samples"),
        "Audio samples (created by bot)",
        required=False
    )
    
    # Check scripts
    print("\n🔧 Scripts:")
    all_ok &= check_file(
        Path("sota_challenge_bot.py"),
        "Challenge bot",
        required=True
    )
    all_ok &= check_file(
        Path("continuous_training_pipeline.py"),
        "Training pipeline",
        required=True
    )
    check_file(
        Path("start_continuous_improvement.ps1"),
        "Quick start script",
        required=False
    )
    
    # Check model configuration
    print("\n⚙️  Model Configuration:")
    try:
        with open("models/crnn_combined/labels.json", 'r') as f:
            labels = json.load(f)
            if 'class_to_idx' in labels:
                classes = list(labels['idx_to_class'].values())
            else:
                classes = labels['labels']
            print(f"  ✓ Classes: {', '.join(classes)}")
    except Exception as e:
        print(f"  ✗ Error reading labels: {e}")
        all_ok = False
    
    # Check dataset sizes
    print("\n📊 Original Dataset Size:")
    for split in ['train', 'val']:
        split_dir = Path(f"data/edth_munich_dataset/data/{split}")
        if split_dir.exists():
            total = 0
            for label in ['background', 'drone', 'helicopter']:
                label_dir = split_dir / label
                if label_dir.exists():
                    count = len(list(label_dir.glob('*.wav')))
                    total += count
                    print(f"  {split}/{label}: {count} samples")
            print(f"  {split} total: {total} samples")
    
    # Final verdict
    print("\n" + "="*80)
    if all_ok:
        print("✅ SETUP CHECK PASSED - Ready to start!")
        print("="*80)
        print("\nNext steps:")
        print("  1. Start the system:")
        print("     .\\start_continuous_improvement.ps1")
        print("\n  OR manually start both processes:")
        print("     Terminal 1: python sota_challenge_bot.py --delay 0.5")
        print("     Terminal 2: python continuous_training_pipeline.py --interval 1200")
        print("\n  2. Monitor progress in both terminal windows")
        print("  3. Check results:")
        print("     - Bot: challenge_results/results.csv")
        print("     - Pipeline: models/crnn_combined/training_history.json")
        return 0
    else:
        print("❌ SETUP CHECK FAILED - Please fix errors above")
        print("="*80)
        print("\nCommon fixes:")
        print("  1. Train the model first:")
        print("     python train_sota_model.py \\")
        print("       --train-dir data/edth_munich_dataset/data/train \\")
        print("       --val-dir data/edth_munich_dataset/data/val \\")
        print("       --model-type crnn \\")
        print("       --output-dir models/crnn_combined")
        print("\n  2. Ensure original dataset is downloaded and extracted")
        return 1

if __name__ == '__main__':
    sys.exit(main())
