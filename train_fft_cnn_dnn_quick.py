"""
Quick Training Script for FFT + CNN + DNN Model
Uses existing EDTH dataset structure with CSV files
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

from adrone.models.fft_cnn_dnn import FFTCNNDNNFusion
from adrone.data.dataset import MelDataset
from sklearn.metrics import classification_report, confusion_matrix


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, labels):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, label_batch in tqdm(val_loader, desc='Validation'):
            inputs = inputs.to(device)
            label_batch = label_batch.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, label_batch)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += label_batch.size(0)
            correct += predicted.eq(label_batch).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label_batch.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total
    
    # Generate classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=labels,
        digits=4
    )
    
    return epoch_loss, epoch_acc, report


def main():
    print("="*80)
    print("Training FFT + CNN + DNN Model on EDTH Dataset")
    print("="*80)
    
    # Configuration
    DATA_DIR = Path("data/edth_prepared")
    TRAIN_CSV = DATA_DIR / "metadata_train.csv"
    VAL_CSV = DATA_DIR / "metadata_val.csv"
    LABELS_JSON = DATA_DIR / "labels.json"
    OUTPUT_DIR = Path("models")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Check if data exists
    if not TRAIN_CSV.exists():
        print(f"\n❌ Training data not found: {TRAIN_CSV}")
        print("Please prepare the dataset first:")
        print("  python scripts/prepare_edth_3class.py")
        return
    
    # Load labels
    with open(LABELS_JSON, 'r') as f:
        labels_data = json.load(f)
        labels = labels_data['labels']
    
    print(f"Classes: {labels}")
    print(f"Number of classes: {len(labels)}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = MelDataset(
        csv_path=str(TRAIN_CSV),
        labels_json=str(LABELS_JSON),
        sample_rate=16000,
        n_mels=64,
        n_fft=1024,
        hop_length=320,
        window_sec=2.0
    )
    
    val_dataset = MelDataset(
        csv_path=str(VAL_CSV),
        labels_json=str(LABELS_JSON),
        sample_rate=16000,
        n_mels=64,
        n_fft=1024,
        hop_length=320,
        window_sec=2.0
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    print("\nCreating FFT + CNN + DNN model...")
    model = FFTCNNDNNFusion(
        n_classes=len(labels),
        in_channels=1,
        cnn_feature_dim=512,
        dnn_hidden_dims=[256, 128]
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, verbose=True
    )
    
    # Training loop
    print(f"\n{'='*80}")
    print("Starting Training")
    print(f"{'='*80}\n")
    
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        
        # Validate
        val_loss, val_acc, report = validate(model, val_loader, criterion, device, labels)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # Save model
            model_path = OUTPUT_DIR / "cnn_edth_3class_improved.pt"
            torch.save(model.state_dict(), model_path)
            print(f"✓ Saved best model to {model_path} (Val Acc: {val_acc*100:.2f}%)")
            
            # Save labels
            labels_path = OUTPUT_DIR / "labels_edth_3class_improved.json"
            with open(labels_path, 'w') as f:
                json.dump({'labels': labels}, f, indent=2)
            print(f"✓ Saved labels to {labels_path}")
            
            # Save classification report
            report_path = OUTPUT_DIR / "classification_report_best.txt"
            with open(report_path, 'w') as f:
                f.write(f"Best Validation Accuracy: {val_acc*100:.2f}%\n")
                f.write(f"Epoch: {epoch+1}\n\n")
                f.write(report)
            print(f"✓ Saved classification report")
    
    # Final summary
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"\nModel saved to: {OUTPUT_DIR / 'cnn_edth_3class_improved.pt'}")
    print(f"Labels saved to: {OUTPUT_DIR / 'labels_edth_3class_improved.json'}")
    print(f"\nTo use the model:")
    print(f"  python challenge_bot_fft_cnn_dnn.py")
    print("="*80)
    
    # Save training history
    history_path = OUTPUT_DIR / 'training_history_improved.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")


if __name__ == '__main__':
    main()
