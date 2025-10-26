"""
Complete Training and Comparison Script: Baseline CRNN vs Enhanced CRNN with Matched Filter Bank

This script trains both models and provides detailed comparison metrics.

Usage:
    python train_and_compare_matched_bank.py --data-dir data/combined_dataset --epochs 30
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import logging
from pathlib import Path
import json
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent))

# Import your existing models
from src.adrone.models.acoustic_models import CRNNWithAttention

# Import matched filter bank components
from src.models.enhanced_models_with_bank import create_enhanced_crnn
from src.training.matched_bank_training import (
    MatchedBankTrainingWrapper,
    FocalLoss
)

# Import your data loading utilities
try:
    from src.adrone.data.acoustic_dataset import create_dataloaders, AcousticDroneDataset
    from src.adrone.preprocessing.audio_transforms import AudioPreprocessor, AugmentationPipeline
    DATA_LOADER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import data utilities: {e}")
    DATA_LOADER_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedCRNN(nn.Module):
    """
    Modified CRNN to accept 9 input channels (3 original + 6 from matched bank)
    """
    def __init__(
        self,
        num_classes: int = 3,
        input_channels: int = 9,  # Changed from 3 to 9
        n_mels: int = 96,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Convolutional feature extraction (modified for 9 channels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Recurrent layers for temporal modeling
        self.gru = nn.GRU(
            input_size=128 * (n_mels // 8),
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Reshape for GRU: (batch, time, features)
        batch, channels, freq, time = x.shape
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, freq)
        x = x.reshape(batch, time, -1)  # (batch, time, channels*freq)
        
        # GRU temporal modeling
        x, _ = self.gru(x)
        
        # Global average pooling over time
        x = x.mean(dim=1)  # (batch, 256)
        
        # Classification
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, use_wrapper=False, wrapper=None):
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    running_loss_breakdown = {}
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        if use_wrapper and wrapper is not None:
            # Use matched bank wrapper (includes curriculum augmentation)
            logits, loss, loss_dict = wrapper.forward_with_augmentation(data, target)
            
            # Track loss breakdown
            for k, v in loss_dict.items():
                running_loss_breakdown[k] = running_loss_breakdown.get(k, 0) + v
        else:
            # Standard training
            logits = model(data)
            loss = criterion(logits, target)
        
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    # Average loss breakdown
    if running_loss_breakdown:
        for k in running_loss_breakdown:
            running_loss_breakdown[k] /= len(train_loader)
    
    return epoch_loss, epoch_acc, running_loss_breakdown


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validating"):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc, all_preds, all_targets


def evaluate_at_snr(model, test_loader, snr_db, device, class_names):
    """Evaluate model at specific SNR level"""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=f"Eval @ SNR={snr_db} dB"):
            data, target = data.to(device), target.to(device)
            
            # Add noise at specified SNR
            if snr_db is not None:
                # Generate white noise
                noise = torch.randn_like(data)
                
                # Calculate signal and noise power
                signal_power = (data ** 2).mean(dim=(1, 2, 3), keepdim=True)
                noise_power = (noise ** 2).mean(dim=(1, 2, 3), keepdim=True)
                
                # Calculate desired SNR ratio
                snr_linear = 10 ** (snr_db / 10.0)
                
                # Scale noise to achieve target SNR
                noise_scale = torch.sqrt(signal_power / (noise_power * snr_linear + 1e-8))
                scaled_noise = noise * noise_scale
                
                # Add noise to signal
                data = data + scaled_noise
            
            output = model(data)
            _, predicted = output.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Compute metrics
    accuracy = accuracy_score(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, 
                                   target_names=class_names, 
                                   output_dict=True,
                                   zero_division=0)
    
    return {
        'snr_db': snr_db,
        'accuracy': accuracy * 100,
        'recall': report['macro avg']['recall'] * 100,
        'precision': report['macro avg']['precision'] * 100,
        'f1': report['macro avg']['f1-score'] * 100,
        'per_class': report
    }


def plot_comparison(baseline_history, enhanced_history, output_dir):
    """Plot training comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Baseline CRNN vs Enhanced CRNN with Matched Filter Bank', 
                 fontsize=16, fontweight='bold')
    
    epochs_baseline = range(1, len(baseline_history['train_loss']) + 1)
    epochs_enhanced = range(1, len(enhanced_history['train_loss']) + 1)
    
    # Training loss
    axes[0, 0].plot(epochs_baseline, baseline_history['train_loss'], 'b-', label='Baseline', linewidth=2)
    axes[0, 0].plot(epochs_enhanced, enhanced_history['train_loss'], 'g-', label='Enhanced', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training accuracy
    axes[0, 1].plot(epochs_baseline, baseline_history['train_acc'], 'b-', label='Baseline', linewidth=2)
    axes[0, 1].plot(epochs_enhanced, enhanced_history['train_acc'], 'g-', label='Enhanced', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Validation loss
    axes[1, 0].plot(epochs_baseline, baseline_history['val_loss'], 'b-', label='Baseline', linewidth=2)
    axes[1, 0].plot(epochs_enhanced, enhanced_history['val_loss'], 'g-', label='Enhanced', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Validation Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Validation accuracy
    axes[1, 1].plot(epochs_baseline, baseline_history['val_acc'], 'b-', label='Baseline', linewidth=2)
    axes[1, 1].plot(epochs_enhanced, enhanced_history['val_acc'], 'g-', label='Enhanced', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_comparison.png', dpi=150, bbox_inches='tight')
    logger.info(f"Saved training comparison to {output_dir / 'training_comparison.png'}")
    plt.close()


def plot_snr_comparison(baseline_snr, enhanced_snr, output_dir):
    """Plot SNR robustness comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SNR Robustness: Baseline vs Enhanced', fontsize=16, fontweight='bold')
    
    snr_levels = sorted(baseline_snr.keys())
    
    metrics = [
        ('accuracy', 'Accuracy (%)', axes[0, 0]),
        ('recall', 'Recall (%)', axes[0, 1]),
        ('precision', 'Precision (%)', axes[1, 0]),
        ('f1', 'F1 Score (%)', axes[1, 1])
    ]
    
    for metric_key, ylabel, ax in metrics:
        baseline_values = [baseline_snr[snr][metric_key] for snr in snr_levels]
        enhanced_values = [enhanced_snr[snr][metric_key] for snr in snr_levels]
        
        ax.plot(snr_levels, baseline_values, 'bo-', label='Baseline', linewidth=2, markersize=8)
        ax.plot(snr_levels, enhanced_values, 'gs-', label='Enhanced (Matched Bank)', linewidth=2, markersize=8)
        
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ylabel, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add improvement annotations
        for i, snr in enumerate(snr_levels):
            improvement = enhanced_values[i] - baseline_values[i]
            if abs(improvement) > 1:
                mid_y = (baseline_values[i] + enhanced_values[i]) / 2
                color = 'green' if improvement > 0 else 'red'
                ax.annotate(f'{improvement:+.1f}%', 
                           xy=(snr, mid_y),
                           fontsize=9, 
                           color=color,
                           fontweight='bold',
                           ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'snr_comparison.png', dpi=150, bbox_inches='tight')
    logger.info(f"Saved SNR comparison to {output_dir / 'snr_comparison.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train and compare baseline vs enhanced CRNN")
    
    # Data
    parser.add_argument("--data-dir", type=str, default="data/combined_dataset",
                        help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default="models/matched_bank_comparison",
                        help="Output directory")
    
    # Training
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    
    # Matched bank config
    parser.add_argument("--compression", type=int, default=6,
                        help="Bank compression channels")
    parser.add_argument("--use-curriculum", action="store_true", default=True,
                        help="Use curriculum SNR training")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma")
    
    # Evaluation
    parser.add_argument("--snr-levels", type=float, nargs="+",
                        default=[30, 20, 15, 10, 5, 0],
                        help="SNR levels to evaluate")
    
    # Misc
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--baseline-checkpoint", type=str, 
                        default="models/crnn_combined/best_model.pt",
                        help="Path to existing baseline model checkpoint")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info(f"Configuration: {json.dumps(vars(args), indent=2)}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info("Loading dataset...")
    data_dir = Path(args.data_dir)
    
    if DATA_LOADER_AVAILABLE:
        # Create preprocessor
        preprocessor = AudioPreprocessor(
            sample_rate=16000,
            n_mels=96,
            use_hpss=True,
            n_fft=512,
            hop_length=160,
            f_min=20,
            f_max=8000
        )
        
        # Create augmentation pipeline
        augmentation = AugmentationPipeline(
            sample_rate=16000,
            use_time_pitch=True,
            use_noise=True,
            use_spec_augment=True,
            use_mixup=True
        )
        
        train_loader, val_loader, class_weights, samples_per_class = create_dataloaders(
            train_dir=str(data_dir / "train"),
            val_dir=str(data_dir / "val"),
            preprocessor=preprocessor,
            augmentation_pipeline=augmentation,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_balanced_sampler=True
        )
        
        # Use test loader as val loader for SNR evaluation
        test_loader = val_loader
        
        # Get class names from dataset
        train_dataset = train_loader.dataset
        class_names = train_dataset.classes if hasattr(train_dataset, 'classes') else ['drone', 'helicopter', 'background']
    else:
        logger.error("Data loader not available. Please check imports.")
        return
    
    num_classes = len(class_names)
    logger.info(f"Classes: {class_names}")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # =========================================================================
    # PART 1: Load Existing Baseline CRNN
    # =========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("LOADING EXISTING BASELINE CRNN")
    logger.info("="*80)
    
    baseline_checkpoint_path = Path(args.baseline_checkpoint)
    
    if not baseline_checkpoint_path.exists():
        logger.error(f"Baseline checkpoint not found: {baseline_checkpoint_path}")
        logger.error("Please provide a valid --baseline-checkpoint path")
        return
    
    # Create baseline model
    baseline_model = CRNNWithAttention(
        num_classes=num_classes,
        input_channels=3,  # Original HPSS channels
        n_mels=96,
        dropout=0.3
    ).to(device)
    
    # Load checkpoint
    logger.info(f"Loading baseline from: {baseline_checkpoint_path}")
    checkpoint = torch.load(baseline_checkpoint_path, map_location=device, weights_only=False)
    baseline_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Count parameters
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    logger.info(f"Baseline model parameters: {baseline_params:,}")
    
    # Load training history if available
    baseline_history_path = baseline_checkpoint_path.parent / "training_history.json"
    if baseline_history_path.exists():
        with open(baseline_history_path, "r") as f:
            baseline_history = json.load(f)
        logger.info(f"Loaded baseline training history ({len(baseline_history.get('train_loss', []))} epochs)")
    else:
        baseline_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        logger.info("No training history found for baseline (will only show final metrics)")
    
    # Evaluate baseline on validation set
    logger.info("\nEvaluating baseline model on validation set...")
    criterion = FocalLoss(gamma=args.focal_gamma)
    baseline_val_loss, baseline_val_acc, _, _ = validate(baseline_model, val_loader, criterion, device)
    
    logger.info(f"Baseline validation accuracy: {baseline_val_acc:.2f}%")
    logger.info(f"Baseline validation loss: {baseline_val_loss:.4f}")
    
    # Try to get original training val_acc from checkpoint
    baseline_trained_val_acc = checkpoint.get('val_acc', baseline_val_acc)
    
    # Copy baseline checkpoint to output directory for comparison
    baseline_output_path = output_dir / "baseline_crnn.pt"
    if not baseline_output_path.exists():
        import shutil
        shutil.copy(baseline_checkpoint_path, baseline_output_path)
        logger.info(f"✓ Copied baseline checkpoint to {baseline_output_path}")
    
    # Save baseline history to output directory
    with open(output_dir / "baseline_history.json", "w") as f:
        json.dump(baseline_history, f, indent=2)
    
    # =========================================================================
    # PART 2: Train Enhanced CRNN with Matched Filter Bank
    # =========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING ENHANCED CRNN WITH MATCHED FILTER BANK")
    logger.info("="*80)
    
    # Create enhanced backbone (accepts 9 channels)
    enhanced_backbone = EnhancedCRNN(
        num_classes=num_classes,
        input_channels=9,  # 3 original + 6 compressed bank
        n_mels=96,
        dropout=0.3
    )
    
    # Wrap with matched filter bank
    enhanced_model = create_enhanced_crnn(
        crnn_backbone=enhanced_backbone,
        n_mels=96,
        sr=16000,
        compression=args.compression,
        trainable_bank=True
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in enhanced_model.parameters())
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    overhead = total_params - baseline_params
    logger.info(f"Enhanced model parameters: {total_params:,}")
    logger.info(f"Parameter overhead: {overhead:,} ({overhead/baseline_params*100:.1f}%)")
    
    # Create training wrapper
    wrapper = MatchedBankTrainingWrapper(
        model=enhanced_model,
        num_classes=num_classes,
        focal_gamma=args.focal_gamma,
        template_margin=0.5,
        template_margin_weight=0.1,
        use_energy_gating=True,
        curriculum_config={
            "initial_snr_db": 30.0,
            "final_snr_db": 0.0,
            "curriculum_epochs": args.epochs // 3
        } if args.use_curriculum else None
    )
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        enhanced_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Loss function (for validation)
    criterion = FocalLoss(gamma=args.focal_gamma)
    
    # Training history
    enhanced_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    enhanced_checkpoint = output_dir / "enhanced_crnn.pt"
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_acc, loss_breakdown = train_epoch(
            enhanced_model, train_loader, optimizer, criterion,
            device, epoch, use_wrapper=True, wrapper=wrapper
        )
        
        logger.info(f"Loss breakdown: {loss_breakdown}")
        
        # Validate
        val_loss, val_acc, _, _ = validate(
            enhanced_model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Step curriculum
        wrapper.step_epoch()
        
        # Log
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save history
        enhanced_history['train_loss'].append(train_loss)
        enhanced_history['train_acc'].append(train_acc)
        enhanced_history['val_loss'].append(val_loss)
        enhanced_history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': enhanced_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': enhanced_history
            }, enhanced_checkpoint)
            logger.info(f"✓ Saved new best enhanced model (val_acc: {val_acc:.2f}%)")
    
    training_time = time.time() - start_time
    logger.info(f"\nEnhanced training completed in {training_time/60:.2f} minutes")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save final history
    with open(output_dir / "enhanced_history.json", "w") as f:
        json.dump(enhanced_history, f, indent=2)
    
    # =========================================================================
    # PART 3: Compare Models
    # =========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("COMPARISON: BASELINE vs ENHANCED")
    logger.info("="*80)
    
    # Load best enhanced model
    enhanced_best = torch.load(output_dir / "enhanced_crnn.pt", weights_only=False)
    enhanced_model.load_state_dict(enhanced_best['model_state_dict'])
    
    # Baseline is already loaded (no need to reload)
    
    # Plot training curves (only if baseline history is available)
    if baseline_history.get('train_loss') and baseline_history.get('val_acc'):
        plot_comparison(baseline_history, enhanced_history, output_dir)
    else:
        logger.info("Skipping training curve comparison (baseline history not available)")
    
    # Final validation performance
    logger.info("\nFinal Validation Performance:")
    _, baseline_final_val_acc, baseline_preds, baseline_targets = validate(
        baseline_model, val_loader, criterion, device
    )
    _, enhanced_val_acc, enhanced_preds, enhanced_targets = validate(
        enhanced_model, val_loader, criterion, device
    )
    
    logger.info(f"Baseline:  {baseline_final_val_acc:.2f}%")
    logger.info(f"Enhanced:  {enhanced_val_acc:.2f}%")
    logger.info(f"Improvement: {enhanced_val_acc - baseline_final_val_acc:+.2f}%")
    
    # Classification reports
    logger.info("\nBaseline Classification Report:")
    print(classification_report(baseline_targets, baseline_preds, target_names=class_names))
    
    logger.info("\nEnhanced Classification Report:")
    print(classification_report(enhanced_targets, enhanced_preds, target_names=class_names))
    
    # =========================================================================
    # PART 4: SNR Robustness Evaluation
    # =========================================================================
    
    if test_loader is not None and len(test_loader) > 0:
        logger.info("\n" + "="*80)
        logger.info("SNR ROBUSTNESS EVALUATION")
        logger.info("="*80)
        
        baseline_snr_results = {}
        enhanced_snr_results = {}
        
        for snr in args.snr_levels:
            logger.info(f"\nEvaluating at SNR = {snr} dB...")
            
            # Baseline
            baseline_metrics = evaluate_at_snr(
                baseline_model, test_loader, snr, device, class_names
            )
            baseline_snr_results[snr] = baseline_metrics
            
            # Enhanced
            enhanced_metrics = evaluate_at_snr(
                enhanced_model, test_loader, snr, device, class_names
            )
            enhanced_snr_results[snr] = enhanced_metrics
            
            # Log comparison
            logger.info(f"  Baseline: Acc={baseline_metrics['accuracy']:.2f}%, "
                       f"Recall={baseline_metrics['recall']:.2f}%")
            logger.info(f"  Enhanced: Acc={enhanced_metrics['accuracy']:.2f}%, "
                       f"Recall={enhanced_metrics['recall']:.2f}%")
            logger.info(f"  Improvement: {enhanced_metrics['accuracy'] - baseline_metrics['accuracy']:+.2f}%")
        
        # Save SNR results
        with open(output_dir / "snr_results.json", "w") as f:
            json.dump({
                'baseline': baseline_snr_results,
                'enhanced': enhanced_snr_results
            }, f, indent=2)
        
        # Plot SNR comparison
        plot_snr_comparison(baseline_snr_results, enhanced_snr_results, output_dir)
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    
    enhanced_params = sum(p.numel() for p in enhanced_model.parameters())
    
    summary = {
        'baseline': {
            'params': baseline_params,
            'checkpoint': str(baseline_checkpoint_path),
            'trained_val_acc': baseline_trained_val_acc,
            'current_val_acc': baseline_final_val_acc
        },
        'enhanced': {
            'params': enhanced_params,
            'best_val_acc': enhanced_best['val_acc'],
            'final_val_acc': enhanced_val_acc,
            'param_overhead_pct': (enhanced_params - baseline_params) / baseline_params * 100
        },
        'improvement': {
            'val_acc': enhanced_val_acc - baseline_final_val_acc,
            'val_acc_pct': (enhanced_val_acc - baseline_final_val_acc) / baseline_final_val_acc * 100
        }
    }
    
    logger.info(f"\nBaseline CRNN (Pre-trained):")
    logger.info(f"  Parameters: {summary['baseline']['params']:,}")
    logger.info(f"  Checkpoint: {summary['baseline']['checkpoint']}")
    logger.info(f"  Original Val Acc: {summary['baseline']['trained_val_acc']:.2f}%")
    logger.info(f"  Current Val Acc: {summary['baseline']['current_val_acc']:.2f}%")
    
    logger.info(f"\nEnhanced CRNN (with Matched Filter Bank):")
    logger.info(f"  Parameters: {summary['enhanced']['params']:,} (+{summary['enhanced']['param_overhead_pct']:.1f}%)")
    logger.info(f"  Best Val Acc: {summary['enhanced']['best_val_acc']:.2f}%")
    
    logger.info(f"\nImprovement:")
    logger.info(f"  Absolute: {summary['improvement']['val_acc']:+.2f}%")
    logger.info(f"  Relative: {summary['improvement']['val_acc_pct']:+.1f}%")
    
    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n✓ All results saved to {output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
