"""
State-of-the-Art Training Script for Acoustic Drone Detection
Implements full methodology with curriculum learning, augmentation, and best practices
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import json
from datetime import datetime
from tqdm import tqdm

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.adrone.preprocessing import AudioPreprocessor, AugmentationPipeline
from src.adrone.data.acoustic_dataset import create_dataloaders
from src.adrone.models.acoustic_models import create_model
from src.adrone.training import (
    CombinedLoss,
    cosine_schedule_with_warmup,
    EarlyStopping,
    MetricsTracker
)
from src.adrone.evaluation import evaluate_model, print_evaluation_report


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_epochs: int
) -> tuple:
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{max_epochs} [Train]')
    
    for spectrograms, labels in pbar:
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(spectrograms)
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        
        # Handle soft labels (from mixup)
        if labels.dim() > 1:
            labels = labels.argmax(dim=1)
        
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    max_epochs: int
) -> tuple:
    """Validate the model"""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{max_epochs} [Val]')
    
    with torch.no_grad():
        for spectrograms, labels in pbar:
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(spectrograms)
            loss = criterion(logits, labels)
            
            # Track metrics
            total_loss += loss.item()
            
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(labels.cpu())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    # Compute macro F1
    from sklearn.metrics import f1_score
    all_predictions = torch.cat(all_predictions).numpy()
    all_targets = torch.cat(all_targets).numpy()
    macro_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    
    return avg_loss, accuracy, macro_f1


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    args,
    device: torch.device
):
    """Main training loop"""
    
    # Loss function
    criterion = CombinedLoss(
        use_focal=args.use_focal_loss,
        use_label_smoothing=True,
        class_weights=class_weights.to(device) if args.use_class_weights else None,
        smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    
    scheduler = cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=args.min_lr_ratio
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        mode='max'  # Maximize macro F1
    )
    
    # Metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Training loop
    best_macro_f1 = 0.0
    best_epoch = 0
    
    print(f"\nTraining {args.model_type} for {args.epochs} epochs")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Warmup steps: {num_warmup_steps}")
    print(f"Total steps: {num_training_steps}\n")
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, args.epochs
        )
        
        # Validate
        val_loss, val_acc, val_macro_f1 = validate(
            model, val_loader, criterion,
            device, epoch, args.epochs
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Track metrics
        metrics_tracker.update(
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            val_macro_f1=val_macro_f1,
            lr=current_lr
        )
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Val Macro F1: {val_macro_f1:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_macro_f1': val_macro_f1,
                'val_acc': val_acc,
                'args': vars(args)
            }
            
            torch.save(checkpoint, args.output_dir / 'best_model.pt')
            print(f"  ✓ Saved best model (Macro F1: {val_macro_f1:.4f})")
        
        # Early stopping check
        if early_stopping(val_macro_f1):
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print(f"\nTraining completed!")
    print(f"Best Macro F1: {best_macro_f1:.4f} at epoch {best_epoch+1}")
    
    # Save final metrics
    metrics_tracker.save(args.output_dir / 'training_history.json')
    metrics_tracker.plot(args.output_dir / 'training_curves.png')
    
    return best_epoch, best_macro_f1


def main():
    parser = argparse.ArgumentParser(description='Train acoustic drone detection model')
    
    # Data arguments
    parser.add_argument('--train-dir', type=str, required=True,
                        help='Path to training data directory (e.g., data/combined_dataset/train)')
    parser.add_argument('--val-dir', type=str, required=True,
                        help='Path to validation data directory (e.g., data/combined_dataset/val)')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for models and logs')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='panns',
                        choices=['crnn', 'panns', 'transformer', 'snn', 'hpc_snn'],
                        help='Model architecture to use')
    parser.add_argument('--use-hpss', action='store_true', default=True,
                        help='Use HPSS for 3-channel input')
    
    # SNN-specific arguments
    parser.add_argument('--snn-timesteps', type=int, default=4,
                        help='Number of SNN simulation timesteps (2-8 typical)')
    parser.add_argument('--use-ttfs', action='store_true',
                        help='Use time-to-first-spike encoding (else rate coding)')
    parser.add_argument('--spike-slope', type=float, default=25.0,
                        help='Surrogate gradient slope for SNN (10-50)')
    parser.add_argument('--snn-depth', type=int, default=None,
                        help='Transformer depth for SNN (default: use model default)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=250,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (use lower for SNN, e.g., 5e-5)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup-ratio', type=float, default=0.1,
                        help='Warmup ratio (0.0-1.0)')
    parser.add_argument('--min-lr-ratio', type=float, default=0.01,
                        help='Minimum LR as ratio of initial LR')
    
    # Loss arguments
    parser.add_argument('--use-focal-loss', action='store_true',
                        help='Use focal loss instead of cross entropy')
    parser.add_argument('--focal-gamma', type=float, default=1.0,
                        help='Focal loss gamma parameter')
    parser.add_argument('--label-smoothing', type=float, default=0.05,
                        help='Label smoothing factor')
    parser.add_argument('--use-class-weights', action='store_true', default=True,
                        help='Use class weights for imbalanced data')
    
    # Regularization arguments
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # System arguments
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    
    # Debugging arguments
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples per class (for debugging)')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test with 2 epochs and small data')
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick_test:
        args.epochs = 2
        args.max_samples = 50
        args.batch_size = 8
        print("\n⚡ QUICK TEST MODE ENABLED")
    
    # Auto-adjust hyperparameters for SNN if using defaults
    if args.model_type in ['snn', 'hpc_snn']:
        # Lower learning rate for SNN stability if using default
        if args.lr == 1e-4:  # Default value
            args.lr = 5e-5
            print(f"\n📊 Auto-adjusted LR for SNN: {args.lr:.6f}")
        
        # Reduce batch size if memory constrained
        if args.batch_size > 32:
            args.batch_size = 16
            print(f"📊 Auto-adjusted batch size for SNN: {args.batch_size}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config (convert Path to string for JSON serialization)
    config_dict = vars(args).copy()
    config_dict['output_dir'] = str(args.output_dir)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Update args.output_dir to Path object for rest of code
    args.output_dir = output_dir
    
    print("\n" + "="*60)
    print("ACOUSTIC DRONE DETECTION TRAINING")
    print("="*60)
    
    # Create preprocessor and augmentation
    print("\nInitializing preprocessing pipeline...")
    preprocessor = AudioPreprocessor(
        sample_rate=16000,
        n_fft=1024,
        hop_length=320,
        n_mels=96,
        window_duration=2.0,
        use_hpss=args.use_hpss
    )
    
    augmentation = AugmentationPipeline(
        sample_rate=16000,
        use_time_pitch=True,
        use_noise=True,
        use_spec_augment=True,
        use_mixup=False  # Mixup requires special handling in training loop
    )
    
    # Create dataloaders
    print("Loading datasets...")
    
    # Resolve paths to absolute
    train_dir = Path(args.train_dir).resolve()
    val_dir = Path(args.val_dir).resolve()
    
    # Check if paths exist
    if not train_dir.exists():
        print(f"\n❌ ERROR: Training directory not found: {train_dir}")
        print("\nPlease use one of these formats:")
        print("  Absolute: --train-dir f:/EDTH/acoustic-drone-detector/data/edth_munich_dataset/data/train")
        print("  Relative: --train-dir data/edth_munich_dataset/data/train")
        sys.exit(1)
    
    if not val_dir.exists():
        print(f"\n❌ ERROR: Validation directory not found: {val_dir}")
        print("\nPlease use one of these formats:")
        print("  Absolute: --val-dir f:/EDTH/acoustic-drone-detector/data/edth_munich_dataset/data/val")
        print("  Relative: --val-dir data/edth_munich_dataset/data/val")
        sys.exit(1)
    
    print(f"✓ Training directory: {train_dir}")
    print(f"✓ Validation directory: {val_dir}")
    
    train_loader, val_loader, class_weights = create_dataloaders(
        train_dir=str(train_dir),
        val_dir=str(val_dir),
        preprocessor=preprocessor,
        augmentation_pipeline=augmentation,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples_per_class=args.max_samples
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Class weights: {class_weights}")
    
    # Create model
    print(f"\nCreating {args.model_type} model...")
    input_channels = 3 if args.use_hpss else 1
    
    # Build model kwargs
    model_kwargs = {
        'num_classes': 3,
        'input_channels': input_channels,
    }
    
    # Add n_mels for CRNN (other models don't need it)
    if args.model_type == 'crnn':
        model_kwargs['n_mels'] = 96
    
    # Add SNN-specific parameters
    if args.model_type in ['snn', 'hpc_snn']:
        model_kwargs['snn_timesteps'] = args.snn_timesteps
        model_kwargs['use_ttfs'] = args.use_ttfs
        model_kwargs['spike_slope'] = args.spike_slope
        
        # Set depth if specified
        if args.snn_depth is not None:
            model_kwargs['depth'] = args.snn_depth
        
        print(f"  SNN Configuration:")
        print(f"    - Timesteps: {args.snn_timesteps}")
        print(f"    - Encoding: {'TTFS' if args.use_ttfs else 'Rate Coding'}")
        print(f"    - Spike slope: {args.spike_slope}")
        if args.snn_depth:
            print(f"    - Transformer depth: {args.snn_depth}")
    
    model = create_model(model_type=args.model_type, **model_kwargs)
    
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    
    # Train
    print("\nStarting training...")
    best_epoch, best_macro_f1 = train(
        model, train_loader, val_loader,
        class_weights, args, device
    )
    
    # Final evaluation
    print("\nPerforming final evaluation...")
    
    # Load best model
    checkpoint = torch.load(args.output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Comprehensive evaluation
    results = evaluate_model(
        model, val_loader, device,
        compute_roc=True,
        compute_calibration=True
    )
    
    print_evaluation_report(results['metrics'], results['confusion_matrix'])
    
    # Save final model (weights only for deployment)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': args.model_type,
        'input_channels': input_channels,
        'num_classes': 3,
        'n_mels': 96,
        'best_epoch': best_epoch,
        'best_macro_f1': best_macro_f1,
        'metrics': results['metrics']
    }, args.output_dir / f'{args.model_type}_final.pt')
    
    # Save class mapping
    class_mapping = {
        'class_to_idx': {'background': 0, 'drone': 1, 'helicopter': 2},
        'idx_to_class': {0: 'background', 1: 'drone', 2: 'helicopter'}
    }
    
    with open(args.output_dir / 'labels.json', 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"  Best model saved to: {args.output_dir / 'best_model.pt'}")
    print(f"  Final model saved to: {args.output_dir / f'{args.model_type}_final.pt'}")
    print(f"  Labels saved to: {args.output_dir / 'labels.json'}")
    print(f"  Training history saved to: {args.output_dir / 'training_history.json'}")


if __name__ == '__main__':
    main()
