"""
Improved training script with advanced techniques:
- Learning rate scheduling
- Early stopping
- Mixed precision training
- Better augmentation
- Comprehensive metrics tracking
- Confusion matrix
- Model checkpointing
"""
import argparse, json, torch, torch.nn as nn, torch.optim as optim, numpy as np, random
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from .data.dataset import MelDataset
from .models.cnn_small import CNNSmall
from .models.cnn_improved import CNNImproved, CNNLarge
from .models.fft_cnn_dnn import FFTCNNDNNFusion, MultiScaleCNNDNN
from tqdm import tqdm
import os, yaml
from pathlib import Path
import time
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def plot_confusion_matrix(cm, classes, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_metrics(model, dataloader, device, classes):
    """Calculate comprehensive metrics including confusion matrix"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Calculate per-class metrics
    report = classification_report(all_targets, all_preds, target_names=classes, output_dict=True)
    
    # Overall accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs
    }

def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None, use_amp=False):
    """Train for one epoch with optional mixed precision"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total else 0
    
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Validating"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            
            total_loss += loss.item()
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total else 0
    
    return avg_loss, accuracy

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def main():
    ap = argparse.ArgumentParser(description="Improved training with advanced techniques")
    ap.add_argument("--config", type=str, default="configs/train_edth.yaml")
    ap.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    ap.add_argument("--early-stop", action="store_true", help="Use early stopping")
    ap.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    args = ap.parse_args()

    # Load config
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["seed"])

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"🚀 TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"Mixed Precision: {'Enabled' if args.amp else 'Disabled'}")
        torch.cuda.empty_cache()
    
    print(f"Epochs: {cfg['epochs']}")
    print(f"Batch Size: {cfg['batch_size']}")
    print(f"Learning Rate: {cfg['lr']}")
    print(f"Early Stopping: {'Enabled' if args.early_stop else 'Disabled'}")
    print(f"{'='*70}\n")

    # Load datasets
    labels_json = cfg.get("labels_json", "data/processed/labels.json")
    train_ds = MelDataset(cfg["train_csv"], labels_json, cfg["sample_rate"], 
                         cfg["n_mels"], cfg["n_fft"], cfg["hop_length"], cfg["window_sec"])
    val_ds = MelDataset(cfg["val_csv"], labels_json, cfg["sample_rate"], 
                       cfg["n_mels"], cfg["n_fft"], cfg["hop_length"], cfg["window_sec"])
    
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, 
                         num_workers=cfg["num_workers"], pin_memory=(device=="cuda"))
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, 
                       num_workers=cfg["num_workers"], pin_memory=(device=="cuda"))
    
    print(f"📊 Dataset Info:")
    print(f"   Training samples: {len(train_ds)}")
    print(f"   Validation samples: {len(val_ds)}")
    print(f"   Classes: {train_ds.labels}")
    print(f"   Number of classes: {len(train_ds.labels)}\n")

    # Create model based on config
    n_classes = len(train_ds.labels)
    model_type = cfg.get("model_type", "improved")
    
    if model_type == "fft_cnn_dnn" or model_type == "fusion":
        model = FFTCNNDNNFusion(
            n_classes=n_classes,
            in_channels=1,
            fft_feature_dim=256,
            cnn_feature_dim=512,
            dnn_hidden_dims=[256, 128]
        )
        print(f"🏗️  Using FFT + CNN + DNN Fusion (Parallel Architecture)")
        print(f"     FFT Path: 256-dim statistical features")
        print(f"     CNN Path: 512-dim learned features")
        print(f"     Fusion: 768-dim combined → DNN classifier")
    elif model_type == "multiscale":
        model = MultiScaleCNNDNN(n_classes=n_classes, in_channels=1)
        print(f"🏗️  Using Multi-Scale CNN + DNN")
    elif model_type == "improved":
        model = CNNImproved(n_classes)
        print(f"🏗️  Using Improved CNN with Residual Blocks and Attention")
    elif model_type == "large":
        model = CNNLarge(n_classes)
        print(f"🏗️  Using Large CNN")
    else:
        model = CNNSmall(n_classes)
        print(f"🏗️  Using Small CNN")
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔧 Model Info:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}\n")

    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    
    # Learning rate scheduler - ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Mixed precision scaler
    scaler = GradScaler() if args.amp and device == "cuda" else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True) if args.early_stop else None
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    
    # Best model tracking
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Create output directory
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    print(f"🎯 Starting Training...\n")
    start_time = time.time()
    
    for epoch in range(cfg["epochs"]):
        epoch_start = time.time()
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{cfg['epochs']}")
        print(f"{'='*70}")
        
        # Show GPU memory
        if device == "cuda":
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"💾 GPU Memory - Allocated: {mem_allocated:.2f} GB, Reserved: {mem_reserved:.2f} GB")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_dl, optimizer, criterion, device, scaler, args.amp
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_dl, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"   Learning Rate: {current_lr:.6f}")
        print(f"   Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1
            
            print(f"\n   ✨ New best model! Val Acc: {val_acc:.4f}")
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, cfg["model_out"])
            
            # Save labels
            with open(cfg["labels_out"], "w") as f:
                json.dump({"labels": train_ds.labels}, f, indent=2)
            
            # Calculate and save confusion matrix for best model
            print("   Computing detailed metrics...")
            metrics = calculate_metrics(model, val_dl, device, train_ds.labels)
            
            # Save confusion matrix plot
            cm_path = output_dir / "confusion_matrix_best.png"
            plot_confusion_matrix(metrics['confusion_matrix'], train_ds.labels, cm_path)
            
            # Save classification report
            report_path = output_dir / "classification_report_best.txt"
            with open(report_path, "w") as f:
                f.write(f"Best Model Performance (Epoch {best_epoch})\n")
                f.write(f"{'='*70}\n\n")
                f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n\n")
                f.write("Per-Class Metrics:\n")
                f.write(f"{'='*70}\n")
                for class_name in train_ds.labels:
                    if class_name in metrics['classification_report']:
                        class_metrics = metrics['classification_report'][class_name]
                        f.write(f"\n{class_name}:\n")
                        f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
                        f.write(f"  Recall:    {class_metrics['recall']:.4f}\n")
                        f.write(f"  F1-Score:  {class_metrics['f1-score']:.4f}\n")
                        f.write(f"  Support:   {class_metrics['support']}\n")
        
        # Early stopping check
        if early_stopping is not None:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
                break
    
    total_time = time.time() - start_time
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"🎉 TRAINING COMPLETED!")
    print(f"{'='*70}")
    print(f"Total Training Time: {total_time/60:.2f} minutes")
    print(f"Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Final Learning Rate: {history['lr'][-1]:.6f}")
    print(f"\n💾 Model saved to: {cfg['model_out']}")
    print(f"📊 Confusion matrix saved to: {output_dir / 'confusion_matrix_best.png'}")
    print(f"📋 Classification report saved to: {output_dir / 'classification_report_best.txt'}")
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"📈 Training history saved to: {history_path}")
    
    # Plot training curves
    plot_training_curves(history, output_dir / "training_curves.png")
    print(f"📊 Training curves saved to: {output_dir / 'training_curves.png'}")
    
    print(f"{'='*70}\n")

def plot_training_curves(history, save_path):
    """Plot and save training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[0, 1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 0].plot(history['lr'], marker='o', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Accuracy Difference (Overfitting indicator)
    acc_diff = np.array(history['train_acc']) - np.array(history['val_acc'])
    axes[1, 1].plot(acc_diff, marker='o', color='red')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Train Acc - Val Acc')
    axes[1, 1].set_title('Overfitting Indicator')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
