"""
Quick Start Script for Training Acoustic Drone Detector
Simplifies the training process with sensible defaults
"""
import subprocess
import sys
from pathlib import Path
import argparse


def check_data_exists(data_path: str) -> bool:
    """Check if data directory exists and has required structure"""
    path = Path(data_path)
    
    if not path.exists():
        return False
    
    # Check for required subdirectories
    required = ['background', 'drone', 'helicopter']
    for subdir in required:
        if not (path / subdir).exists():
            return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Quick Start Training for Acoustic Drone Detection')
    
    parser.add_argument('--model', type=str, default='panns',
                        choices=['crnn', 'panns', 'transformer'],
                        help='Model to train (default: panns)')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test with 2 epochs and small data')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--gpu', action='store_true',
                        help='Force GPU usage')
    
    args = parser.parse_args()
    
    # Check if data exists
    train_dir = Path('data/edth_munich_dataset/data/train')
    val_dir = Path('data/edth_munich_dataset/data/val')
    
    print("\n" + "="*60)
    print("ACOUSTIC DRONE DETECTION - QUICK START")
    print("="*60)
    
    print("\nChecking data directories...")
    
    if not check_data_exists(train_dir):
        print(f"❌ Training data not found at: {train_dir}")
        print("\nExpected structure:")
        print("  data/edth_munich_dataset/data/train/")
        print("    ├── background/")
        print("    ├── drone/")
        print("    └── helicopter/")
        sys.exit(1)
    
    if not check_data_exists(val_dir):
        print(f"❌ Validation data not found at: {val_dir}")
        print("\nExpected structure:")
        print("  data/edth_munich_dataset/data/val/")
        print("    ├── background/")
        print("    ├── drone/")
        print("    └── helicopter/")
        sys.exit(1)
    
    print(f"✓ Training data found: {train_dir}")
    print(f"✓ Validation data found: {val_dir}")
    
    # Build command
    cmd = [
        sys.executable,
        'train_sota_model.py',
        '--train-dir', str(train_dir),
        '--val-dir', str(val_dir),
        '--model-type', args.model,
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--output-dir', f'models/{args.model}',
        '--use-class-weights',
        '--use-hpss'
    ]
    
    if args.quick_test:
        cmd.append('--quick-test')
        print("\n⚡ Quick test mode enabled (2 epochs, small data)")
    
    if args.gpu:
        cmd.extend(['--device', 'cuda'])
        print("🔥 GPU mode enabled")
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs if not args.quick_test else 2}")
    print(f"  Batch size: {args.batch_size if not args.quick_test else 8}")
    print(f"  Output: models/{args.model}/")
    print(f"  Features: HPSS, Class Weights, Label Smoothing, SpecAugment")
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
        
        print("\n" + "="*60)
        print("✓ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nModel saved to: models/{args.model}/")
        print(f"  - {args.model}_final.pt (model weights)")
        print(f"  - labels.json (class mapping)")
        print(f"  - training_history.json (metrics)")
        print(f"  - training_curves.png (plots)")
        
        print("\n📊 To test the model:")
        print(f"  python sota_inference.py models/{args.model}/{args.model}_final.pt models/{args.model}/labels.json <audio.wav>")
        
        print("\n🤖 To run challenge bot:")
        print(f"  python sota_challenge_bot.py --model models/{args.model}/{args.model}_final.pt --labels models/{args.model}/labels.json")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)


if __name__ == '__main__':
    main()
