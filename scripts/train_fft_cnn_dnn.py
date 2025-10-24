"""
Training Script for FFT + CNN + DNN Model
Trains the fusion model on drone acoustic data
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

from adrone.models.fft_cnn_dnn import FFTCNNDNNFusion, MultiScaleCNNDNN
from adrone.features.fft_processor import FFTProcessor
from adrone.data.dataset import AudioDataset
from sklearn.metrics import classification_report, confusion_matrix


class FFTCNNDNNTrainer:
    """Trainer for FFT + CNN + DNN fusion model"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        labels: list,
        device: str = 'cuda',
        learning_rate: float = 0.001,
        output_dir: str = 'models'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.labels = labels
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=5, factor=0.5
        )
        
        # Tracking
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
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
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='Validation'):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def train(self, num_epochs: int):
        """Train the model"""
        print(f"\n{'='*80}")
        print(f"Training FFT + CNN + DNN Model")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Classes: {self.labels}")
        print(f"{'='*80}\n")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 80)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            
            # Validate
            val_loss, val_acc, all_preds, all_labels = self.validate()
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best_model.pt', epoch, val_acc)
                print(f"✓ Saved best model (Val Acc: {val_acc*100:.2f}%)")
                
                # Save classification report
                report = classification_report(
                    all_labels, all_preds,
                    target_names=self.labels,
                    digits=4
                )
                report_path = self.output_dir / 'classification_report_best.txt'
                with open(report_path, 'w') as f:
                    f.write(report)
                print(f"✓ Saved classification report")
        
        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"Best Validation Accuracy: {self.best_val_acc*100:.2f}%")
        print(f"{'='*80}")
        
        return self.history
    
    def save_checkpoint(self, filename: str, epoch: int, val_acc: float):
        """Save model checkpoint"""
        checkpoint_path = self.output_dir / filename
        torch.save(self.model.state_dict(), checkpoint_path)
        
        # Save metadata
        metadata = {
            'epoch': epoch,
            'val_acc': val_acc,
            'labels': self.labels,
            'model_architecture': 'FFTCNNDNNFusion',
            'timestamp': datetime.now().isoformat()
        }
        metadata_path = self.output_dir / filename.replace('.pt', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train FFT + CNN + DNN Model')
    parser.add_argument('--train-dir', type=str, required=True,
                        help='Training data directory')
    parser.add_argument('--val-dir', type=str, required=True,
                        help='Validation data directory')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--model-type', type=str, default='fusion',
                        choices=['fusion', 'multiscale'],
                        help='Model architecture type')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize FFT processor
    fft_processor = FFTProcessor(
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        sample_rate=16000
    )
    
    # Load datasets
    print("Loading datasets...")
    # Note: You'll need to implement AudioDataset to use FFT preprocessing
    # For now, using placeholder
    train_dataset = AudioDataset(
        args.train_dir,
        fft_processor=fft_processor,
        max_duration=2.0
    )
    val_dataset = AudioDataset(
        args.val_dir,
        fft_processor=fft_processor,
        max_duration=2.0
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Get labels
    labels = train_dataset.classes
    n_classes = len(labels)
    
    print(f"Classes: {labels}")
    print(f"Number of classes: {n_classes}")
    
    # Create model
    if args.model_type == 'fusion':
        model = FFTCNNDNNFusion(
            n_classes=n_classes,
            in_channels=1,
            cnn_feature_dim=512,
            dnn_hidden_dims=[256, 128]
        )
    else:
        model = MultiScaleCNNDNN(
            n_classes=n_classes,
            in_channels=1
        )
    
    print(f"Model: {args.model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = FFTCNNDNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        labels=labels,
        device=device,
        learning_rate=args.lr,
        output_dir=args.output_dir
    )
    
    # Train
    history = trainer.train(num_epochs=args.epochs)
    
    # Save training history
    history_path = Path(args.output_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Training history saved to {history_path}")


if __name__ == '__main__':
    main()
