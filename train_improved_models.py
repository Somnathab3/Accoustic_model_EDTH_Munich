"""
Improved Training Script for PANNS, Transformer, and SNN
Incorporates: Focal Loss, Knowledge Distillation, Balanced Sampling, Progressive Unfreezing
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import json
from datetime import datetime
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.adrone.preprocessing import AudioPreprocessor, AugmentationPipeline
from src.adrone.data.acoustic_dataset import create_dataloaders
from src.adrone.models.acoustic_models import create_model
from src.adrone.training import (
    CombinedLoss,
    DistillationLoss,
    ClassBalancedLoss,
    cosine_schedule_with_warmup,
    EarlyStopping,
    MetricsTracker
)
from src.adrone.evaluation import evaluate_model, print_evaluation_report


def load_teacher_model(teacher_path: str, device: torch.device):
    """Load CRNN teacher model for knowledge distillation"""
    print(f"\n📚 Loading teacher model from {teacher_path}")
    
    checkpoint = torch.load(teacher_path, map_location=device, weights_only=False)
    
    # Get model config
    model_type = checkpoint.get('model_type', 'crnn')
    input_channels = checkpoint.get('input_channels', 3)
    num_classes = checkpoint.get('num_classes', 3)
    n_mels = checkpoint.get('n_mels', 96)
    
    # Create teacher model
    model_kwargs = {
        'num_classes': num_classes,
        'input_channels': input_channels,
    }
    if model_type == 'crnn':
        model_kwargs['n_mels'] = n_mels
    
    teacher_model = create_model(model_type=model_type, **model_kwargs)
    teacher_model.load_state_dict(checkpoint['model_state_dict'])
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    
    print(f"✓ Teacher model loaded: {model_type.upper()}")
    print(f"  Best F1: {checkpoint.get('best_macro_f1', 'N/A')}")
    
    return teacher_model


def train_epoch_with_kd(
    model: nn.Module,
    teacher_model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    kd_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_epochs: int,
    use_kd: bool = True
) -> tuple:
    """Train for one epoch with knowledge distillation"""
    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{max_epochs} [Train]')
    
    for spectrograms, labels in pbar:
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        student_logits = model(spectrograms)
        
        # Compute loss
        if use_kd and teacher_model is not None:
            # Knowledge distillation
            with torch.no_grad():
                teacher_logits = teacher_model(spectrograms)
            
            # Hard labels for KD
            hard_labels = labels.argmax(dim=1) if labels.dim() > 1 else labels
            loss = kd_criterion(student_logits, teacher_logits, hard_labels)
        else:
            # Standard loss
            loss = criterion(student_logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        
        # Handle soft labels
        if labels.dim() > 1:
            labels = labels.argmax(dim=1)
        
        predictions = student_logits.argmax(dim=1)
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
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    # Compute per-class metrics
    from sklearn.metrics import f1_score, recall_score, precision_score
    all_predictions = torch.cat(all_predictions).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    macro_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    per_class_recall = recall_score(all_targets, all_predictions, average=None, zero_division=0)
    per_class_f1 = f1_score(all_targets, all_predictions, average=None, zero_division=0)
    
    return avg_loss, accuracy, macro_f1, per_class_recall, per_class_f1


def train(
    model: nn.Module,
    teacher_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    samples_per_class: torch.Tensor,
    args,
    device: torch.device
):
    """Main training loop with KD and all improvements"""
    
    # Loss function
    if args.use_class_balanced:
        criterion = CombinedLoss(
            use_class_balanced=True,
            samples_per_class=samples_per_class.to(device),
            num_classes=3,
            smoothing=args.label_smoothing
        )
    elif args.use_focal_loss:
        criterion = CombinedLoss(
            use_focal=True,
            class_weights=class_weights.to(device) if args.use_class_weights else None,
            focal_gamma=args.focal_gamma
        )
    else:
        criterion = CombinedLoss(
            use_label_smoothing=True,
            class_weights=class_weights.to(device) if args.use_class_weights else None,
            smoothing=args.label_smoothing
        )
    
    # Knowledge distillation loss
    kd_criterion = None
    if args.use_kd and teacher_model is not None:
        kd_criterion = DistillationLoss(
            temperature=args.kd_temperature,
            alpha=args.kd_alpha,
            base_criterion=criterion
        )
        print(f"\n✓ Knowledge Distillation enabled (T={args.kd_temperature}, α={args.kd_alpha})")
    
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
        mode='max'
    )
    
    # Metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Training loop
    best_macro_f1 = 0.0
    best_drone_recall = 0.0
    best_epoch = 0
    
    print(f"\n{'='*70}")
    print(f"Training {args.model_type.upper()} with Improvements")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Loss type: {'Focal' if args.use_focal_loss else 'Class-Balanced' if args.use_class_balanced else 'CE+LabelSmoothing'}")
    print(f"Balanced sampling: {args.balanced_sampling}")
    print(f"Knowledge distillation: {args.use_kd}")
    print(f"{'='*70}\n")
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch_with_kd(
            model, teacher_model, train_loader, criterion, kd_criterion,
            optimizer, device, epoch, args.epochs, use_kd=args.use_kd
        )
        
        # Validate
        val_loss, val_acc, val_macro_f1, per_class_recall, per_class_f1 = validate(
            model, val_loader, criterion, device, epoch, args.epochs
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
        
        # Class names
        class_names = ['background', 'drone', 'helicopter']
        drone_recall = per_class_recall[1]  # Drone is class 1
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Val Macro F1: {val_macro_f1:.4f}")
        print(f"  Per-class Recall: {dict(zip(class_names, per_class_recall))}")
        print(f"  ⚠️  DRONE RECALL: {drone_recall:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model (by macro F1)
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_macro_f1': val_macro_f1,
                'val_acc': val_acc,
                'per_class_recall': per_class_recall.tolist(),
                'per_class_f1': per_class_f1.tolist(),
                'args': vars(args)
            }
            
            torch.save(checkpoint, args.output_dir / 'best_model.pt')
            print(f"  ✓ Saved best model (Macro F1: {val_macro_f1:.4f})")
        
        # Save best drone recall model
        if drone_recall > best_drone_recall:
            best_drone_recall = drone_recall
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_macro_f1': val_macro_f1,
                'drone_recall': drone_recall,
                'per_class_recall': per_class_recall.tolist()
            }
            
            torch.save(checkpoint, args.output_dir / 'best_drone_recall.pt')
            print(f"  ✓ Saved best drone recall model (Recall: {drone_recall:.4f})")
        
        # Early stopping check
        if early_stopping(val_macro_f1):
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Best Macro F1: {best_macro_f1:.4f} at epoch {best_epoch+1}")
    print(f"Best Drone Recall: {best_drone_recall:.4f}")
    print(f"{'='*70}\n")
    
    # Save final metrics
    metrics_tracker.save(args.output_dir / 'training_history.json')
    metrics_tracker.plot(args.output_dir / 'training_curves.png')
    
    return best_epoch, best_macro_f1, best_drone_recall


def main():
    parser = argparse.ArgumentParser(description='Train improved acoustic drone detection models')
    
    # Data arguments
    parser.add_argument('--train-dir', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--val-dir', type=str, required=True,
                        help='Path to validation data directory')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for models and logs')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, required=True,
                        choices=['panns', 'transformer', 'snn', 'hpc_snn'],
                        help='Model architecture to train')
    parser.add_argument('--use-hpss', action='store_true', default=True,
                        help='Use HPSS for 3-channel input')
    
    # SNN-specific arguments
    parser.add_argument('--snn-timesteps', type=int, default=10,
                        help='Number of SNN simulation timesteps')
    parser.add_argument('--use-ttfs', action='store_true',
                        help='Use time-to-first-spike encoding')
    parser.add_argument('--spike-slope', type=float, default=35.0,
                        help='Surrogate gradient slope for SNN')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup-ratio', type=float, default=0.1,
                        help='Warmup ratio')
    parser.add_argument('--min-lr-ratio', type=float, default=0.01,
                        help='Minimum LR ratio')
    
    # Loss arguments
    parser.add_argument('--use-focal-loss', action='store_true', default=True,
                        help='Use focal loss (recommended for drone recall)')
    parser.add_argument('--use-class-balanced', action='store_true',
                        help='Use class-balanced loss instead of focal')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter')
    parser.add_argument('--label-smoothing', type=float, default=0.05,
                        help='Label smoothing factor')
    parser.add_argument('--use-class-weights', action='store_true', default=True,
                        help='Use class weights')
    
    # Balanced sampling
    parser.add_argument('--balanced-sampling', action='store_true', default=True,
                        help='Use balanced sampler for equal class representation')
    
    # Knowledge distillation arguments
    parser.add_argument('--use-kd', action='store_true', default=True,
                        help='Use knowledge distillation from CRNN teacher')
    parser.add_argument('--teacher-model', type=str,
                        default='models/crnn_combined/crnn_final.pt',
                        help='Path to teacher model checkpoint')
    parser.add_argument('--kd-temperature', type=float, default=3.0,
                        help='KD temperature (2-5 typical)')
    parser.add_argument('--kd-alpha', type=float, default=0.5,
                        help='KD alpha (weight for soft targets)')
    
    # Regularization
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    
    # System arguments
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    
    args = parser.parse_args()
    
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
    output_dir = Path(args.output_dir) / f'{args.model_type}_improved'
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir
    
    print(f"\n{'='*70}")
    print(f"IMPROVED {args.model_type.upper()} TRAINING")
    print(f"{'='*70}\n")
    
    # Create preprocessor and augmentation
    print("Initializing preprocessing pipeline...")
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
        use_mixup=False,
        use_notch_filter=True,  # New!
        use_band_limiting=True  # New!
    )
    
    # Create dataloaders
    print("Loading datasets...")
    train_dir = Path(args.train_dir).resolve()
    val_dir = Path(args.val_dir).resolve()
    
    if not train_dir.exists():
        print(f"\n❌ ERROR: Training directory not found: {train_dir}")
        sys.exit(1)
    
    if not val_dir.exists():
        print(f"\n❌ ERROR: Validation directory not found: {val_dir}")
        sys.exit(1)
    
    print(f"✓ Training directory: {train_dir}")
    print(f"✓ Validation directory: {val_dir}")
    
    train_loader, val_loader, class_weights, samples_per_class = create_dataloaders(
        train_dir=str(train_dir),
        val_dir=str(val_dir),
        preprocessor=preprocessor,
        augmentation_pipeline=augmentation,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_balanced_sampler=args.balanced_sampling
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Class weights: {class_weights}")
    print(f"Samples per class: {samples_per_class}")
    
    # Load teacher model for KD
    teacher_model = None
    if args.use_kd:
        teacher_path = Path(args.teacher_model)
        if teacher_path.exists():
            try:
                teacher_model = load_teacher_model(str(teacher_path), device)
            except Exception as e:
                print(f"⚠️  Warning: Could not load teacher model: {e}")
                print("  Continuing without knowledge distillation")
                args.use_kd = False
        else:
            print(f"⚠️  Warning: Teacher model not found: {teacher_path}")
            print("  Continuing without knowledge distillation")
            args.use_kd = False
    
    # Create student model
    print(f"\nCreating {args.model_type} student model...")
    input_channels = 3 if args.use_hpss else 1
    
    model_kwargs = {
        'num_classes': 3,
        'input_channels': input_channels,
    }
    
    if args.model_type in ['snn', 'hpc_snn']:
        model_kwargs['snn_timesteps'] = args.snn_timesteps
        model_kwargs['use_ttfs'] = args.use_ttfs
        model_kwargs['spike_slope'] = args.spike_slope
        print(f"  SNN config: timesteps={args.snn_timesteps}, TTFS={args.use_ttfs}, slope={args.spike_slope}")
    
    model = create_model(model_type=args.model_type, **model_kwargs)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    
    # Train
    print("\nStarting training with improvements...")
    best_epoch, best_macro_f1, best_drone_recall = train(
        model, teacher_model, train_loader, val_loader,
        class_weights, samples_per_class, args, device
    )
    
    # Final evaluation
    print("\nPerforming final evaluation...")
    checkpoint = torch.load(args.output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    results = evaluate_model(
        model, val_loader, device,
        compute_roc=True,
        compute_calibration=True
    )
    
    print_evaluation_report(results['metrics'], results['confusion_matrix'])
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': args.model_type,
        'input_channels': input_channels,
        'num_classes': 3,
        'n_mels': 96,
        'best_epoch': best_epoch,
        'best_macro_f1': best_macro_f1,
        'best_drone_recall': best_drone_recall,
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
    print(f"  Best model: {args.output_dir / 'best_model.pt'}")
    print(f"  Final model: {args.output_dir / f'{args.model_type}_final.pt'}")
    print(f"  Drone recall: {args.output_dir / 'best_drone_recall.pt'}")


if __name__ == '__main__':
    main()
