"""
Quick Start Script - Train Enhanced Model with Pre-trained Baseline

This script provides a simple one-command way to start training with your existing baseline.
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("="*80)
    print("MATCHED FILTER BANK TRAINING - QUICK START")
    print("="*80)
    print()
    
    # Check if baseline exists
    baseline_path = Path("models/crnn_combined/best_model.pt")
    
    if not baseline_path.exists():
        print("⚠️  WARNING: Baseline model not found!")
        print(f"   Expected: {baseline_path.absolute()}")
        print()
        print("Please either:")
        print("  1. Train a baseline CRNN model first")
        print("  2. Specify a custom baseline with --baseline-checkpoint")
        print()
        return 1
    
    print("✓ Found baseline model:", baseline_path.absolute())
    print()
    
    # Check data directory
    data_dir = Path("data/combined_dataset")
    if not data_dir.exists():
        print("⚠️  WARNING: Data directory not found!")
        print(f"   Expected: {data_dir.absolute()}")
        print()
        return 1
    
    print("✓ Found data directory:", data_dir.absolute())
    print()
    
    # Show what will happen
    print("Training Plan:")
    print("  1. Load existing baseline CRNN (~30 seconds)")
    print("  2. Train enhanced CRNN with matched filter bank (~1.5-2 hours)")
    print("  3. Compare models and evaluate SNR robustness (~5-10 minutes)")
    print()
    print("Total estimated time: ~2 hours")
    print()
    
    # Ask for confirmation
    response = input("Start training? [Y/n]: ").strip().lower()
    
    if response in ['', 'y', 'yes']:
        print()
        print("="*80)
        print("STARTING TRAINING...")
        print("="*80)
        print()
        
        # Build command
        cmd = [
            sys.executable,
            "train_and_compare_matched_bank.py",
            "--data-dir", "data/combined_dataset",
            "--epochs", "30",
            "--batch-size", "32",
            "--lr", "1e-4",
            "--compression", "6",
            "--focal-gamma", "2.0"
        ]
        
        print("Command:", " ".join(cmd))
        print()
        
        # Run training
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
            return 1
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Training failed with error code {e.returncode}")
            return 1
        
        print()
        print("="*80)
        print("✓ TRAINING COMPLETED!")
        print("="*80)
        print()
        print("Results saved to: models/matched_bank_comparison/")
        print()
        print("Next steps:")
        print("  1. Check summary.json for overall results")
        print("  2. View training_comparison.png for training curves")
        print("  3. View snr_comparison.png for low-SNR performance")
        print()
        
        return 0
    
    else:
        print()
        print("Training cancelled.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
