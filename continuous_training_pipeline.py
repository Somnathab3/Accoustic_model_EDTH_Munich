"""
Continuous Training Pipeline for Acoustic Drone Detection
Automatically improves the model by:
1. Collecting correct predictions from challenge results
2. Adding them to the combined dataset
3. Retraining from the last best checkpoint
4. Updating the model in-place (no renaming)
5. Running in 20-minute cycles

This enables dynamic model improvement while the challenge bot continues running.
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import json
import shutil
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple
import argparse
from tqdm import tqdm

# Import training utilities
from adrone.preprocessing import AudioPreprocessor, AugmentationPipeline
from adrone.data.acoustic_dataset import create_dataloaders
from adrone.models.acoustic_models import create_model
from adrone.training import (
    CombinedLoss,
    cosine_schedule_with_warmup,
    EarlyStopping,
    MetricsTracker
)
from adrone.evaluation import evaluate_model


class ContinuousTrainingPipeline:
    """
    Pipeline for continuous model improvement
    """
    
    def __init__(
        self,
        challenge_results_dir: str = "challenge_results",
        combined_data_dir: str = "data/combined_dataset",
        model_dir: str = "models/crnn_combined",
        original_train_dir: str = "data/edth_munich_dataset/data/train",
        original_val_dir: str = "data/edth_munich_dataset/data/val",
        train_ratio: float = 0.8,
        cycle_interval: int = 1200  # 20 minutes in seconds
    ):
        """
        Initialize the continuous training pipeline
        
        Args:
            challenge_results_dir: Directory containing challenge results
            combined_data_dir: Directory for combined training dataset
            model_dir: Directory containing the model to update
            original_train_dir: Original training data directory
            original_val_dir: Original validation data directory
            train_ratio: Ratio of new samples to add to training vs validation
            cycle_interval: Time between training cycles in seconds (default: 20 min)
        """
        self.challenge_dir = Path(challenge_results_dir)
        self.audio_samples_dir = self.challenge_dir / "audio_samples"
        self.results_csv = self.challenge_dir / "results.csv"
        self.results_jsonl = self.challenge_dir / "results.jsonl"
        
        self.combined_dir = Path(combined_data_dir)
        self.combined_train = self.combined_dir / "train"
        self.combined_val = self.combined_dir / "val"
        
        self.model_dir = Path(model_dir)
        self.best_model_path = self.model_dir / "best_model.pt"
        self.final_model_path = self.model_dir / "crnn_final.pt"
        self.labels_path = self.model_dir / "labels.json"
        self.history_path = self.model_dir / "training_history.json"
        
        self.original_train_dir = Path(original_train_dir)
        self.original_val_dir = Path(original_val_dir)
        
        self.train_ratio = train_ratio
        self.cycle_interval = cycle_interval
        
        # Track processed samples to avoid duplicates
        self.processed_samples_file = self.challenge_dir / "processed_samples.json"
        self.processed_samples = self.load_processed_samples()
        
        # Statistics
        self.cycle_count = 0
        self.total_samples_added = 0
        
    def load_processed_samples(self) -> set:
        """Load set of already processed challenge IDs"""
        if self.processed_samples_file.exists():
            with open(self.processed_samples_file, 'r') as f:
                data = json.load(f)
                return set(data.get('processed_ids', []))
        return set()
    
    def save_processed_samples(self):
        """Save set of processed challenge IDs"""
        with open(self.processed_samples_file, 'w') as f:
            json.dump({
                'processed_ids': list(self.processed_samples),
                'last_update': datetime.now().isoformat(),
                'total_processed': len(self.processed_samples)
            }, f, indent=2)
    
    def collect_new_correct_samples(self) -> Dict[str, list]:
        """
        Collect new correct predictions from challenge results
        
        Returns:
            Dictionary mapping labels to list of (filename, info) tuples
        """
        print(f"\n{'='*80}")
        print("COLLECTING NEW CORRECT SAMPLES")
        print(f"{'='*80}")
        
        new_samples = {
            'background': [],
            'drone': [],
            'helicopter': []
        }
        
        # Load results from JSONL (most complete data)
        if not self.results_jsonl.exists():
            print("⚠️  No results.jsonl found")
            return new_samples
        
        jsonl_data = []
        with open(self.results_jsonl, 'r') as f:
            for line in f:
                if line.strip():
                    jsonl_data.append(json.loads(line))
        
        print(f"Loaded {len(jsonl_data)} total results from JSONL")
        
        # Filter for correct predictions (score > 0) that haven't been processed
        new_correct = 0
        for entry in jsonl_data:
            challenge_id = entry.get('challenge_id')
            score = entry.get('score_awarded', 0)
            
            # Skip if already processed or incorrect
            if challenge_id in self.processed_samples or score <= 0:
                continue
            
            prediction = entry.get('prediction')
            audio_file = entry.get('audio_file')
            confidence = entry.get('confidence', 0.0)
            
            if prediction and audio_file and prediction in new_samples:
                audio_path = Path(audio_file)
                if audio_path.exists():
                    new_samples[prediction].append({
                        'path': audio_path,
                        'filename': audio_path.name,
                        'challenge_id': challenge_id,
                        'confidence': confidence,
                        'score': score
                    })
                    self.processed_samples.add(challenge_id)
                    new_correct += 1
        
        # Print summary
        print(f"\nFound {new_correct} NEW correct predictions:")
        for label, samples in new_samples.items():
            print(f"  {label:12s}: {len(samples)} samples")
        
        return new_samples
    
    def add_samples_to_dataset(self, new_samples: Dict[str, list]) -> Tuple[int, int]:
        """
        Add new samples to combined dataset
        
        Args:
            new_samples: Dictionary mapping labels to sample info
            
        Returns:
            Tuple of (train_added, val_added)
        """
        print(f"\n{'='*80}")
        print("ADDING SAMPLES TO COMBINED DATASET")
        print(f"{'='*80}")
        
        # Create directories if they don't exist
        for label in ['background', 'drone', 'helicopter']:
            (self.combined_train / label).mkdir(parents=True, exist_ok=True)
            (self.combined_val / label).mkdir(parents=True, exist_ok=True)
        
        train_added = 0
        val_added = 0
        
        for label, samples in new_samples.items():
            if not samples:
                continue
            
            # Split into train/val
            split_idx = int(len(samples) * self.train_ratio)
            train_samples = samples[:split_idx]
            val_samples = samples[split_idx:]
            
            print(f"\n{label} ({len(samples)} samples):")
            print(f"  → Train: {len(train_samples)}, Val: {len(val_samples)}")
            
            # Copy train samples
            for sample in train_samples:
                src = sample['path']
                dst = self.combined_train / label / sample['filename']
                
                if not dst.exists():
                    shutil.copy2(src, dst)
                    train_added += 1
                    print(f"    ✓ Train: {sample['filename']} (conf: {sample['confidence']:.3f})")
            
            # Copy val samples
            for sample in val_samples:
                src = sample['path']
                dst = self.combined_val / label / sample['filename']
                
                if not dst.exists():
                    shutil.copy2(src, dst)
                    val_added += 1
                    print(f"    ✓ Val: {sample['filename']} (conf: {sample['confidence']:.3f})")
        
        print(f"\n📊 Total added: {train_added} train, {val_added} val")
        
        return train_added, val_added
    
    def count_dataset_samples(self) -> Dict[str, Dict[str, int]]:
        """Count samples in current combined dataset"""
        counts = {
            'train': {'background': 0, 'drone': 0, 'helicopter': 0},
            'val': {'background': 0, 'drone': 0, 'helicopter': 0}
        }
        
        for split in ['train', 'val']:
            split_dir = self.combined_dir / split
            for label in ['background', 'drone', 'helicopter']:
                label_dir = split_dir / label
                if label_dir.exists():
                    counts[split][label] = len(list(label_dir.glob('*.wav')))
        
        return counts
    
    def retrain_model(self, epochs: int = 20, batch_size: int = 32) -> bool:
        """
        Retrain model from last best checkpoint
        
        Args:
            epochs: Number of epochs to train
            batch_size: Batch size for training
            
        Returns:
            True if training successful
        """
        print(f"\n{'='*80}")
        print("RETRAINING MODEL FROM BEST CHECKPOINT")
        print(f"{'='*80}")
        
        try:
            # Load checkpoint to get model configuration
            if not self.best_model_path.exists():
                print(f"❌ Best model not found: {self.best_model_path}")
                return False
            
            checkpoint = torch.load(self.best_model_path, map_location='cpu', weights_only=False)
            
            model_type = checkpoint.get('model_type', 'crnn')
            num_classes = checkpoint.get('num_classes', 3)
            input_channels = checkpoint.get('input_channels', 3)
            n_mels = checkpoint.get('n_mels', 96)
            
            print(f"\nModel Configuration:")
            print(f"  Type: {model_type.upper()}")
            print(f"  Classes: {num_classes}")
            print(f"  Channels: {input_channels}")
            print(f"  N_mels: {n_mels}")
            
            # Count current dataset
            dataset_counts = self.count_dataset_samples()
            print(f"\nDataset:")
            print(f"  Train: {sum(dataset_counts['train'].values())} samples")
            for label, count in dataset_counts['train'].items():
                print(f"    - {label}: {count}")
            print(f"  Val: {sum(dataset_counts['val'].values())} samples")
            for label, count in dataset_counts['val'].items():
                print(f"    - {label}: {count}")
            
            # Get device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"\nDevice: {device}")
            
            # Create preprocessing
            preprocessor = AudioPreprocessor(
                sample_rate=16000,
                n_mels=n_mels,
                use_hpss=(input_channels == 3)
            )
            
            # Create augmentation (light for fine-tuning)
            augmentation = AugmentationPipeline(
                sample_rate=16000,
                use_time_pitch=False,  # Disable for fine-tuning
                use_noise=True,
                use_spec_augment=True,
                use_mixup=False  # Disable for fine-tuning
            )
            
            # Create data loaders
            print("\nCreating data loaders...")
            train_loader, val_loader, class_weights = create_dataloaders(
                train_dir=str(self.combined_train),
                val_dir=str(self.combined_val),
                preprocessor=preprocessor,
                augmentation_pipeline=augmentation,
                batch_size=batch_size,
                num_workers=4
            )
            
            # Create model
            model = create_model(
                model_type=model_type,
                num_classes=num_classes,
                input_channels=input_channels,
                n_mels=n_mels if model_type == 'crnn' else None
            )
            
            # Load weights from best checkpoint
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            print(f"\n✓ Loaded weights from: {self.best_model_path}")
            
            # Get previous best metrics
            prev_best_loss = checkpoint.get('val_loss', checkpoint.get('best_val_loss', float('inf')))
            prev_best_acc = checkpoint.get('val_acc', checkpoint.get('best_val_acc', 0.0))
            print(f"  Previous best: Val Loss = {prev_best_loss:.4f}, Val Acc = {prev_best_acc:.4f}")
            
            # Setup training
            criterion = CombinedLoss(
                num_classes=num_classes,
                focal_alpha=0.25,
                focal_gamma=2.0,
                label_smoothing=0.1
            )
            
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=0.0001,  # Lower LR for fine-tuning
                weight_decay=0.01
            )
            
            # Simple learning rate scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )
            
            early_stopping = EarlyStopping(patience=10, min_delta=0.001)
            metrics_tracker = MetricsTracker()
            
            # Train model
            print(f"\n🚀 Starting training for {epochs} epochs...")
            print(f"  Batch size: {batch_size}")
            print(f"  Learning rate: 0.0001")
            
            best_val_acc = prev_best_acc
            best_val_loss = prev_best_loss
            
            for epoch in range(epochs):
                # Train epoch
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
                for spectrograms, labels in pbar:
                    spectrograms = spectrograms.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    logits = model(spectrograms)
                    loss = criterion(logits, labels)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                    if labels.dim() > 1:
                        labels = labels.argmax(dim=1)
                    
                    predictions = logits.argmax(dim=1)
                    train_correct += (predictions == labels).sum().item()
                    train_total += labels.size(0)
                    
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{train_correct/train_total:.4f}'
                    })
                
                avg_train_loss = train_loss / len(train_loader)
                train_acc = train_correct / train_total
                
                # Validation
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for spectrograms, labels in val_loader:
                        spectrograms = spectrograms.to(device)
                        labels = labels.to(device)
                        
                        logits = model(spectrograms)
                        loss = criterion(logits, labels)
                        
                        val_loss += loss.item()
                        predictions = logits.argmax(dim=1)
                        val_correct += (predictions == labels).sum().item()
                        val_total += labels.size(0)
                
                avg_val_loss = val_loss / len(val_loader)
                val_acc = val_correct / val_total
                
                # Update scheduler
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                
                # Track metrics
                metrics_tracker.update(
                    train_loss=avg_train_loss,
                    train_acc=train_acc,
                    val_loss=avg_val_loss,
                    val_acc=val_acc,
                    val_macro_f1=val_acc,  # Simplified
                    lr=current_lr
                )
                
                # Print summary
                print(f"\nEpoch {epoch+1}/{epochs}:")
                print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
                print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.4f}")
                print(f"  LR: {current_lr:.6f}")
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_loss = avg_val_loss
                    
                    # Save checkpoint
                    best_checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'val_loss': avg_val_loss,
                        'best_val_acc': best_val_acc,
                        'best_val_loss': best_val_loss,
                        'model_type': model_type,
                        'num_classes': num_classes,
                        'input_channels': input_channels,
                        'n_mels': n_mels
                    }
                    
                    torch.save(best_checkpoint, self.best_model_path)
                    print(f"  ✓ Saved best model (Val Acc: {val_acc:.4f})")
                
                # Early stopping
                if early_stopping(val_acc):
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
            
            # Save final model (overwrites existing crnn_final.pt)
            final_checkpoint = torch.load(self.best_model_path, map_location='cpu', weights_only=False)
            torch.save(final_checkpoint, self.final_model_path)
            print(f"\n✓ Updated final model: {self.final_model_path}")
            print(f"  (Bot will automatically use this on next inference)")
            
            # Save training history
            metrics_tracker.save(self.history_path)
            
            # Print improvement
            print(f"\n📊 Training Results:")
            print(f"  Old: Val Loss = {prev_best_loss:.4f}, Val Acc = {prev_best_acc:.4f}")
            print(f"  New: Val Loss = {best_val_loss:.4f}, Val Acc = {best_val_acc:.4f}")
            
            if best_val_acc > prev_best_acc:
                print(f"  ✓ IMPROVED! Accuracy +{(best_val_acc - prev_best_acc)*100:.2f}%")
            else:
                print(f"  → No improvement in accuracy")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_single_cycle(self, epochs: int = 20, batch_size: int = 32) -> bool:
        """
        Run a single pipeline cycle
        
        Args:
            epochs: Number of epochs to train
            batch_size: Batch size for training
            
        Returns:
            True if training occurred
        """
        self.cycle_count += 1
        
        print(f"\n{'#'*80}")
        print(f"CYCLE #{self.cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*80}")
        
        # Step 1: Collect new correct samples
        new_samples = self.collect_new_correct_samples()
        
        total_new = sum(len(samples) for samples in new_samples.values())
        
        if total_new == 0:
            print("\n⏸️  No new correct samples to add. Skipping training.")
            self.save_processed_samples()
            return False
        
        # Step 2: Add to combined dataset
        train_added, val_added = self.add_samples_to_dataset(new_samples)
        self.total_samples_added += (train_added + val_added)
        
        # Save processed samples list
        self.save_processed_samples()
        
        # Step 3: Retrain model
        print(f"\n🔄 Starting retraining with {train_added + val_added} new samples...")
        success = self.retrain_model(epochs=epochs, batch_size=batch_size)
        
        if success:
            print(f"\n✅ Cycle #{self.cycle_count} completed successfully!")
        else:
            print(f"\n⚠️  Cycle #{self.cycle_count} completed with errors")
        
        return success
    
    def run_continuous(
        self,
        max_cycles: int = None,
        epochs_per_cycle: int = 20,
        batch_size: int = 32
    ):
        """
        Run continuous training pipeline
        
        Args:
            max_cycles: Maximum number of cycles (None = infinite)
            epochs_per_cycle: Epochs per training cycle
            batch_size: Batch size for training
        """
        print(f"\n{'='*80}")
        print("CONTINUOUS TRAINING PIPELINE")
        print(f"{'='*80}")
        print(f"Model directory: {self.model_dir}")
        print(f"Combined dataset: {self.combined_dir}")
        print(f"Cycle interval: {self.cycle_interval}s ({self.cycle_interval/60:.1f} min)")
        print(f"Epochs per cycle: {epochs_per_cycle}")
        print(f"Max cycles: {max_cycles if max_cycles else 'Infinite'}")
        print(f"{'='*80}\n")
        
        training_count = 0
        
        try:
            while True:
                # Check if we've reached max cycles
                if max_cycles and self.cycle_count >= max_cycles:
                    break
                
                # Run cycle
                trained = self.run_single_cycle(
                    epochs=epochs_per_cycle,
                    batch_size=batch_size
                )
                
                if trained:
                    training_count += 1
                
                # Print summary
                print(f"\n{'─'*80}")
                print(f"Pipeline Summary:")
                print(f"  Total cycles: {self.cycle_count}")
                print(f"  Training runs: {training_count}")
                print(f"  Samples added: {self.total_samples_added}")
                print(f"  Processed challenges: {len(self.processed_samples)}")
                print(f"{'─'*80}")
                
                # Wait for next cycle
                if max_cycles is None or self.cycle_count < max_cycles:
                    print(f"\n⏳ Waiting {self.cycle_interval/60:.1f} minutes until next cycle...")
                    print(f"   Next check at: {datetime.fromtimestamp(time.time() + self.cycle_interval).strftime('%H:%M:%S')}")
                    time.sleep(self.cycle_interval)
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Pipeline stopped by user")
        
        # Final summary
        print(f"\n{'='*80}")
        print("FINAL PIPELINE SUMMARY")
        print(f"{'='*80}")
        print(f"Total cycles run: {self.cycle_count}")
        print(f"Training runs: {training_count}")
        print(f"Total samples added: {self.total_samples_added}")
        print(f"Processed challenges: {len(self.processed_samples)}")
        
        # Show final dataset size
        final_counts = self.count_dataset_samples()
        print(f"\nFinal dataset size:")
        print(f"  Train: {sum(final_counts['train'].values())} samples")
        print(f"  Val: {sum(final_counts['val'].values())} samples")
        
        print(f"\n✓ Pipeline complete!")


def main():
    parser = argparse.ArgumentParser(description='Continuous Training Pipeline')
    
    parser.add_argument('--challenge-dir', type=str, default='challenge_results',
                       help='Directory containing challenge results')
    parser.add_argument('--combined-dir', type=str, default='data/combined_dataset',
                       help='Directory for combined training dataset')
    parser.add_argument('--model-dir', type=str, default='models/crnn_combined',
                       help='Directory containing the model to update')
    parser.add_argument('--original-train', type=str, default='data/edth_munich_dataset/data/train',
                       help='Original training data directory')
    parser.add_argument('--original-val', type=str, default='data/edth_munich_dataset/data/val',
                       help='Original validation data directory')
    parser.add_argument('--interval', type=int, default=1200,
                       help='Cycle interval in seconds (default: 1200 = 20 min)')
    parser.add_argument('--max-cycles', type=int, default=None,
                       help='Maximum number of cycles (default: infinite)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Epochs per training cycle (default: 20)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Ratio of new samples to add to training (default: 0.8)')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = ContinuousTrainingPipeline(
        challenge_results_dir=args.challenge_dir,
        combined_data_dir=args.combined_dir,
        model_dir=args.model_dir,
        original_train_dir=args.original_train,
        original_val_dir=args.original_val,
        train_ratio=args.train_ratio,
        cycle_interval=args.interval
    )
    
    # Run continuous pipeline
    pipeline.run_continuous(
        max_cycles=args.max_cycles,
        epochs_per_cycle=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
