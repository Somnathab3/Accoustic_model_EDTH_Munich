"""
Real-time Detection Performance Comparison: Baseline CRNN vs LIGO-Modified Matched Filter Bank

This script evaluates both models on the entire training and validation datasets
to compare detection performance metrics.

Usage:
    python compare_detection_performance.py --baseline models/crnn_combined/best_model.pt --enhanced models/matched_bank_comparison/enhanced_crnn.pt
"""

import torch
import torch.nn as nn
from pathlib import Path
import argparse
import json
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Import models
from src.adrone.models.acoustic_models import CRNNWithAttention
from src.models.enhanced_models_with_bank import create_enhanced_crnn

# Import data utilities
try:
    from src.adrone.data.acoustic_dataset import create_dataloaders
    from src.adrone.preprocessing.audio_transforms import AudioPreprocessor, AugmentationPipeline
    DATA_LOADER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import data utilities: {e}")
    DATA_LOADER_AVAILABLE = False


class EnhancedCRNN(nn.Module):
    """Modified CRNN to accept 9 input channels"""
    def __init__(
        self,
        num_classes: int = 3,
        input_channels: int = 9,
        n_mels: int = 96,
        dropout: float = 0.3
    ):
        super().__init__()
        
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
        
        self.gru = nn.GRU(
            input_size=128 * (n_mels // 8),
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        batch, channels, freq, time = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch, time, -1)
        
        x, _ = self.gru(x)
        x = x.mean(dim=1)
        
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


def evaluate_model(model, data_loader, device, class_names, model_name="Model"):
    """Comprehensive evaluation of a model"""
    model.eval()
    
    all_preds = []
    all_probs = []
    all_targets = []
    inference_times = []
    
    print(f"\n{'='*80}")
    print(f"Evaluating {model_name}")
    print(f"{'='*80}")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(data_loader, desc=f"Evaluating {model_name}")):
            data, target = data.to(device), target.to(device)
            
            # Measure inference time
            start_time = time.time()
            output = model(data)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Get predictions and probabilities
            probs = torch.softmax(output, dim=1)
            _, predicted = output.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0
    )
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        all_targets, all_preds, labels=range(len(class_names)), zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Inference statistics
    avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
    std_inference_time = np.std(inference_times) * 1000
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'macro_precision': precision * 100,
        'macro_recall': recall * 100,
        'macro_f1': f1 * 100,
        'per_class': {
            class_names[i]: {
                'precision': per_class_precision[i] * 100,
                'recall': per_class_recall[i] * 100,
                'f1': per_class_f1[i] * 100,
                'support': int(support[i])
            }
            for i in range(len(class_names))
        },
        'confusion_matrix': cm.tolist(),
        'inference_time_ms': {
            'mean': avg_inference_time,
            'std': std_inference_time,
            'min': np.min(inference_times) * 1000,
            'max': np.max(inference_times) * 1000
        },
        'predictions': all_preds.tolist(),
        'probabilities': all_probs.tolist(),
        'targets': all_targets.tolist()
    }
    
    # Print summary
    print(f"\n{model_name} Results:")
    print(f"  Overall Accuracy: {accuracy:.2f}%")
    print(f"  Macro Precision:  {precision*100:.2f}%")
    print(f"  Macro Recall:     {recall*100:.2f}%")
    print(f"  Macro F1:         {f1*100:.2f}%")
    print(f"\n  Per-Class Performance:")
    for class_name in class_names:
        class_metrics = results['per_class'][class_name]
        print(f"    {class_name:15s} - P: {class_metrics['precision']:5.2f}% | "
              f"R: {class_metrics['recall']:5.2f}% | F1: {class_metrics['f1']:5.2f}% | "
              f"N: {class_metrics['support']:4d}")
    
    print(f"\n  Inference Time: {avg_inference_time:.2f} ± {std_inference_time:.2f} ms")
    
    return results


def plot_comparison(baseline_results, enhanced_results, class_names, output_dir):
    """Create comprehensive comparison visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Overall Metrics Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Baseline CRNN vs LIGO-Modified Matched Filter Bank\nDetection Performance Comparison', 
                 fontsize=16, fontweight='bold')
    
    # Overall metrics bar chart
    metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    baseline_values = [baseline_results[m] for m in metrics]
    enhanced_values = [enhanced_results[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, baseline_values, width, label='Baseline CRNN', color='skyblue', edgecolor='black')
    axes[0, 0].bar(x + width/2, enhanced_values, width, label='LIGO-Modified', color='lightgreen', edgecolor='black')
    
    axes[0, 0].set_ylabel('Performance (%)', fontsize=12)
    axes[0, 0].set_title('Overall Performance Metrics', fontsize=13, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metric_labels)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 105])
    
    # Add value labels on bars
    for i, (b_val, e_val) in enumerate(zip(baseline_values, enhanced_values)):
        axes[0, 0].text(i - width/2, b_val + 1, f'{b_val:.1f}%', ha='center', va='bottom', fontsize=9)
        axes[0, 0].text(i + width/2, e_val + 1, f'{e_val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Show improvement
        improvement = e_val - b_val
        if abs(improvement) > 0.5:
            color = 'green' if improvement > 0 else 'red'
            axes[0, 0].text(i, max(b_val, e_val) + 4, f'{improvement:+.1f}%', 
                           ha='center', fontsize=10, color=color, fontweight='bold')
    
    # Per-class Precision comparison
    class_precisions_baseline = [baseline_results['per_class'][c]['precision'] for c in class_names]
    class_precisions_enhanced = [enhanced_results['per_class'][c]['precision'] for c in class_names]
    
    x_classes = np.arange(len(class_names))
    axes[0, 1].bar(x_classes - width/2, class_precisions_baseline, width, 
                   label='Baseline CRNN', color='skyblue', edgecolor='black')
    axes[0, 1].bar(x_classes + width/2, class_precisions_enhanced, width, 
                   label='LIGO-Modified', color='lightgreen', edgecolor='black')
    
    axes[0, 1].set_ylabel('Precision (%)', fontsize=12)
    axes[0, 1].set_title('Per-Class Precision', fontsize=13, fontweight='bold')
    axes[0, 1].set_xticks(x_classes)
    axes[0, 1].set_xticklabels(class_names, rotation=0)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim([0, 105])
    
    # Per-class Recall comparison
    class_recalls_baseline = [baseline_results['per_class'][c]['recall'] for c in class_names]
    class_recalls_enhanced = [enhanced_results['per_class'][c]['recall'] for c in class_names]
    
    axes[1, 0].bar(x_classes - width/2, class_recalls_baseline, width, 
                   label='Baseline CRNN', color='skyblue', edgecolor='black')
    axes[1, 0].bar(x_classes + width/2, class_recalls_enhanced, width, 
                   label='LIGO-Modified', color='lightgreen', edgecolor='black')
    
    axes[1, 0].set_ylabel('Recall (%)', fontsize=12)
    axes[1, 0].set_title('Per-Class Recall', fontsize=13, fontweight='bold')
    axes[1, 0].set_xticks(x_classes)
    axes[1, 0].set_xticklabels(class_names, rotation=0)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_ylim([0, 105])
    
    # Per-class F1 comparison
    class_f1_baseline = [baseline_results['per_class'][c]['f1'] for c in class_names]
    class_f1_enhanced = [enhanced_results['per_class'][c]['f1'] for c in class_names]
    
    axes[1, 1].bar(x_classes - width/2, class_f1_baseline, width, 
                   label='Baseline CRNN', color='skyblue', edgecolor='black')
    axes[1, 1].bar(x_classes + width/2, class_f1_enhanced, width, 
                   label='LIGO-Modified', color='lightgreen', edgecolor='black')
    
    axes[1, 1].set_ylabel('F1 Score (%)', fontsize=12)
    axes[1, 1].set_title('Per-Class F1 Score', fontsize=13, fontweight='bold')
    axes[1, 1].set_xticks(x_classes)
    axes[1, 1].set_xticklabels(class_names, rotation=0)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'detection_performance_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {output_dir / 'detection_performance_comparison.png'}")
    plt.close()
    
    # 2. Confusion Matrices Side-by-Side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold')
    
    cm_baseline = np.array(baseline_results['confusion_matrix'])
    cm_enhanced = np.array(enhanced_results['confusion_matrix'])
    
    # Normalize
    cm_baseline_norm = cm_baseline.astype('float') / cm_baseline.sum(axis=1)[:, np.newaxis] * 100
    cm_enhanced_norm = cm_enhanced.astype('float') / cm_enhanced.sum(axis=1)[:, np.newaxis] * 100
    
    # Baseline confusion matrix
    sns.heatmap(cm_baseline_norm, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, 
                ax=axes[0], cbar_kws={'label': 'Percentage (%)'})
    axes[0].set_title('Baseline CRNN', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=11)
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    
    # Enhanced confusion matrix
    sns.heatmap(cm_enhanced_norm, annot=True, fmt='.1f', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names, 
                ax=axes[1], cbar_kws={'label': 'Percentage (%)'})
    axes[1].set_title('LIGO-Modified Matched Bank', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=11)
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved confusion matrices: {output_dir / 'confusion_matrices_comparison.png'}")
    plt.close()
    
    # 3. Inference Time Comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    baseline_time = baseline_results['inference_time_ms']['mean']
    enhanced_time = enhanced_results['inference_time_ms']['mean']
    baseline_std = baseline_results['inference_time_ms']['std']
    enhanced_std = enhanced_results['inference_time_ms']['std']
    
    models = ['Baseline\nCRNN', 'LIGO-Modified\nMatched Bank']
    times = [baseline_time, enhanced_time]
    stds = [baseline_std, enhanced_std]
    colors = ['skyblue', 'lightgreen']
    
    bars = ax.bar(models, times, yerr=stds, capsize=10, color=colors, edgecolor='black', linewidth=2)
    ax.set_ylabel('Inference Time (ms)', fontsize=12)
    ax.set_title('Average Inference Time per Batch', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, time, std) in enumerate(zip(bars, times, stds)):
        ax.text(bar.get_x() + bar.get_width()/2, time + std + 0.5, 
               f'{time:.2f} ± {std:.2f} ms', 
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'inference_time_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved inference time plot: {output_dir / 'inference_time_comparison.png'}")
    plt.close()


def create_comparison_table(baseline_results, enhanced_results, class_names, output_dir):
    """Create detailed comparison table"""
    
    comparison_data = []
    
    # Overall metrics
    comparison_data.append({
        'Metric': 'Overall Accuracy',
        'Baseline': f"{baseline_results['accuracy']:.2f}%",
        'LIGO-Modified': f"{enhanced_results['accuracy']:.2f}%",
        'Improvement': f"{enhanced_results['accuracy'] - baseline_results['accuracy']:+.2f}%"
    })
    
    comparison_data.append({
        'Metric': 'Macro Precision',
        'Baseline': f"{baseline_results['macro_precision']:.2f}%",
        'LIGO-Modified': f"{enhanced_results['macro_precision']:.2f}%",
        'Improvement': f"{enhanced_results['macro_precision'] - baseline_results['macro_precision']:+.2f}%"
    })
    
    comparison_data.append({
        'Metric': 'Macro Recall',
        'Baseline': f"{baseline_results['macro_recall']:.2f}%",
        'LIGO-Modified': f"{enhanced_results['macro_recall']:.2f}%",
        'Improvement': f"{enhanced_results['macro_recall'] - baseline_results['macro_recall']:+.2f}%"
    })
    
    comparison_data.append({
        'Metric': 'Macro F1 Score',
        'Baseline': f"{baseline_results['macro_f1']:.2f}%",
        'LIGO-Modified': f"{enhanced_results['macro_f1']:.2f}%",
        'Improvement': f"{enhanced_results['macro_f1'] - baseline_results['macro_f1']:+.2f}%"
    })
    
    # Add separator
    comparison_data.append({
        'Metric': '─' * 40,
        'Baseline': '─' * 15,
        'LIGO-Modified': '─' * 15,
        'Improvement': '─' * 15
    })
    
    # Per-class metrics
    for class_name in class_names:
        baseline_class = baseline_results['per_class'][class_name]
        enhanced_class = enhanced_results['per_class'][class_name]
        
        comparison_data.append({
            'Metric': f'{class_name} - Precision',
            'Baseline': f"{baseline_class['precision']:.2f}%",
            'LIGO-Modified': f"{enhanced_class['precision']:.2f}%",
            'Improvement': f"{enhanced_class['precision'] - baseline_class['precision']:+.2f}%"
        })
        
        comparison_data.append({
            'Metric': f'{class_name} - Recall',
            'Baseline': f"{baseline_class['recall']:.2f}%",
            'LIGO-Modified': f"{enhanced_class['recall']:.2f}%",
            'Improvement': f"{enhanced_class['recall'] - baseline_class['recall']:+.2f}%"
        })
        
        comparison_data.append({
            'Metric': f'{class_name} - F1',
            'Baseline': f"{baseline_class['f1']:.2f}%",
            'LIGO-Modified': f"{enhanced_class['f1']:.2f}%",
            'Improvement': f"{enhanced_class['f1'] - baseline_class['f1']:+.2f}%"
        })
    
    # Add separator
    comparison_data.append({
        'Metric': '─' * 40,
        'Baseline': '─' * 15,
        'LIGO-Modified': '─' * 15,
        'Improvement': '─' * 15
    })
    
    # Inference time
    comparison_data.append({
        'Metric': 'Avg Inference Time',
        'Baseline': f"{baseline_results['inference_time_ms']['mean']:.2f} ms",
        'LIGO-Modified': f"{enhanced_results['inference_time_ms']['mean']:.2f} ms",
        'Improvement': f"{enhanced_results['inference_time_ms']['mean'] - baseline_results['inference_time_ms']['mean']:+.2f} ms"
    })
    
    df = pd.DataFrame(comparison_data)
    
    # Save as CSV
    df.to_csv(output_dir / 'detection_comparison_table.csv', index=False)
    print(f"\n✓ Saved comparison table: {output_dir / 'detection_comparison_table.csv'}")
    
    # Print table
    print("\n" + "="*100)
    print("DETAILED COMPARISON TABLE")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compare detection performance: Baseline CRNN vs LIGO-Modified Matched Filter Bank"
    )
    
    parser.add_argument("--baseline", type=str, 
                        default="models/crnn_combined/best_model.pt",
                        help="Path to baseline CRNN checkpoint")
    parser.add_argument("--enhanced", type=str, 
                        default="models/matched_bank_comparison/enhanced_crnn.pt",
                        help="Path to enhanced CRNN checkpoint")
    parser.add_argument("--data-dir", type=str, 
                        default="data/combined_dataset",
                        help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, 
                        default="detection_comparison_results",
                        help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--compression", type=int, default=6,
                        help="Matched bank compression factor")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print("\nLoading datasets...")
    data_dir = Path(args.data_dir)
    
    if not DATA_LOADER_AVAILABLE:
        print("ERROR: Data loader not available!")
        return
    
    # Create preprocessor (no augmentation for evaluation)
    preprocessor = AudioPreprocessor(
        sample_rate=16000,
        n_mels=96,
        use_hpss=True,
        n_fft=512,
        hop_length=160,
        f_min=20,
        f_max=8000
    )
    
    # Load train and val sets
    train_loader, val_loader, _, _ = create_dataloaders(
        train_dir=str(data_dir / "train"),
        val_dir=str(data_dir / "val"),
        preprocessor=preprocessor,
        augmentation_pipeline=None,  # No augmentation for evaluation
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_balanced_sampler=False  # Sequential evaluation
    )
    
    # Get class names
    train_dataset = train_loader.dataset
    class_names = train_dataset.classes if hasattr(train_dataset, 'classes') else ['drone', 'helicopter', 'background']
    num_classes = len(class_names)
    
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # =========================================================================
    # Load Baseline Model
    # =========================================================================
    
    print("\n" + "="*80)
    print("LOADING BASELINE CRNN")
    print("="*80)
    
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"ERROR: Baseline checkpoint not found: {baseline_path}")
        return
    
    baseline_model = CRNNWithAttention(
        num_classes=num_classes,
        input_channels=3,
        n_mels=96,
        dropout=0.3
    ).to(device)
    
    baseline_checkpoint = torch.load(baseline_path, map_location=device, weights_only=False)
    baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
    
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    print(f"✓ Loaded baseline model: {baseline_params:,} parameters")
    
    # =========================================================================
    # Load Enhanced Model
    # =========================================================================
    
    print("\n" + "="*80)
    print("LOADING LIGO-MODIFIED MATCHED FILTER BANK MODEL")
    print("="*80)
    
    enhanced_path = Path(args.enhanced)
    if not enhanced_path.exists():
        print(f"ERROR: Enhanced checkpoint not found: {enhanced_path}")
        return
    
    # Create enhanced backbone
    enhanced_backbone = EnhancedCRNN(
        num_classes=num_classes,
        input_channels=9,
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
    
    enhanced_checkpoint = torch.load(enhanced_path, map_location=device, weights_only=False)
    enhanced_model.load_state_dict(enhanced_checkpoint['model_state_dict'])
    
    enhanced_params = sum(p.numel() for p in enhanced_model.parameters())
    print(f"✓ Loaded enhanced model: {enhanced_params:,} parameters")
    print(f"  Parameter overhead: {enhanced_params - baseline_params:,} ({(enhanced_params-baseline_params)/baseline_params*100:.1f}%)")
    
    # =========================================================================
    # Evaluate on Training Set
    # =========================================================================
    
    print("\n" + "="*80)
    print("EVALUATING ON TRAINING SET")
    print("="*80)
    
    train_baseline_results = evaluate_model(
        baseline_model, train_loader, device, class_names, 
        model_name="Baseline CRNN (Train)"
    )
    
    train_enhanced_results = evaluate_model(
        enhanced_model, train_loader, device, class_names, 
        model_name="LIGO-Modified (Train)"
    )
    
    # =========================================================================
    # Evaluate on Validation Set
    # =========================================================================
    
    print("\n" + "="*80)
    print("EVALUATING ON VALIDATION SET")
    print("="*80)
    
    val_baseline_results = evaluate_model(
        baseline_model, val_loader, device, class_names, 
        model_name="Baseline CRNN (Val)"
    )
    
    val_enhanced_results = evaluate_model(
        enhanced_model, val_loader, device, class_names, 
        model_name="LIGO-Modified (Val)"
    )
    
    # =========================================================================
    # Generate Comparison Reports
    # =========================================================================
    
    print("\n" + "="*80)
    print("GENERATING COMPARISON REPORTS")
    print("="*80)
    
    # Save all results
    all_results = {
        'train': {
            'baseline': train_baseline_results,
            'enhanced': train_enhanced_results
        },
        'val': {
            'baseline': val_baseline_results,
            'enhanced': val_enhanced_results
        },
        'model_info': {
            'baseline_params': baseline_params,
            'enhanced_params': enhanced_params,
            'param_overhead': enhanced_params - baseline_params,
            'param_overhead_pct': (enhanced_params - baseline_params) / baseline_params * 100
        }
    }
    
    with open(output_dir / 'full_comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Saved full results: {output_dir / 'full_comparison_results.json'}")
    
    # Create visualizations for validation set
    plot_comparison(val_baseline_results, val_enhanced_results, class_names, output_dir)
    
    # Create comparison tables
    print("\n" + "="*80)
    print("VALIDATION SET COMPARISON")
    print("="*80)
    create_comparison_table(val_baseline_results, val_enhanced_results, class_names, output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    val_acc_improvement = val_enhanced_results['accuracy'] - val_baseline_results['accuracy']
    val_recall_improvement = val_enhanced_results['macro_recall'] - val_baseline_results['macro_recall']
    
    print(f"\nValidation Set Performance:")
    print(f"  Baseline Accuracy:     {val_baseline_results['accuracy']:.2f}%")
    print(f"  LIGO-Modified Accuracy: {val_enhanced_results['accuracy']:.2f}%")
    print(f"  Improvement:            {val_acc_improvement:+.2f}%")
    print(f"\n  Baseline Recall:        {val_baseline_results['macro_recall']:.2f}%")
    print(f"  LIGO-Modified Recall:   {val_enhanced_results['macro_recall']:.2f}%")
    print(f"  Improvement:            {val_recall_improvement:+.2f}%")
    
    print(f"\n✓ All results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
