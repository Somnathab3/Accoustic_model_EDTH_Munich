"""
Evaluation and comparison script for matched filter bank enhanced models.

Usage:
    python evaluate_matched_bank.py --baseline-model baseline.pt --enhanced-model enhanced.pt --test-data test/
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
import seaborn as sns

# Adjust imports as needed
import sys
sys.path.append(str(Path(__file__).parent))

from src.training.matched_bank_training import CurriculumSNRAugmentation


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # This assumes your checkpoint contains model architecture info
    # Adjust based on your actual checkpoint structure
    model_state = checkpoint.get('model_state_dict', checkpoint)
    
    # You'll need to reconstruct the model architecture here
    # For now, return a placeholder
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Available keys: {checkpoint.keys()}")
    
    return None, model_state


def evaluate_at_snr(
    model,
    dataloader,
    snr_db,
    device,
    curriculum=None
):
    """
    Evaluate model at specific SNR level.
    
    Args:
        model: Neural network model
        dataloader: Test data loader
        snr_db: SNR level in dB (None = no noise)
        device: Torch device
        curriculum: CurriculumSNRAugmentation instance
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc=f"Eval @ SNR={snr_db} dB"):
            data, target = data.to(device), target.to(device)
            
            # Add noise if specified
            if snr_db is not None and curriculum is not None:
                # Generate noise
                noise = torch.randn_like(data)
                # Add at specified SNR
                data = curriculum._add_noise_at_snr(
                    data, noise, torch.tensor([snr_db] * data.shape[0], device=device)
                )
            
            # Forward pass
            output = model(data)
            probs = torch.softmax(output, dim=1)
            
            _, predicted = output.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # Concatenate probabilities
    all_probs = np.concatenate(all_probs, axis=0)
    
    # Compute metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0
    )
    
    # Per-class metrics
    report = classification_report(
        all_targets, all_preds,
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    metrics = {
        'snr_db': snr_db,
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs
    }
    
    return metrics


def compute_snr_curve(
    model,
    dataloader,
    snr_levels,
    device,
    model_name="Model"
):
    """
    Compute performance across SNR levels.
    
    Args:
        model: Neural network model
        dataloader: Test data loader
        snr_levels: List of SNR levels to test
        device: Torch device
        model_name: Name for plotting
        
    Returns:
        results: Dictionary mapping SNR -> metrics
    """
    curriculum = CurriculumSNRAugmentation()
    results = {}
    
    print(f"\n{'='*80}")
    print(f"Computing SNR curve for {model_name}")
    print(f"{'='*80}")
    
    for snr in snr_levels:
        print(f"\nEvaluating at SNR = {snr} dB...")
        metrics = evaluate_at_snr(model, dataloader, snr, device, curriculum)
        results[snr] = metrics
        
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  Recall:   {metrics['recall']:.2f}%")
        print(f"  F1:       {metrics['f1']:.2f}%")
    
    return results


def plot_snr_comparison(
    baseline_results,
    enhanced_results,
    output_path
):
    """Plot SNR curve comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Baseline vs Enhanced Model: SNR Robustness Comparison", fontsize=16, fontweight='bold')
    
    snr_levels = sorted(baseline_results.keys())
    
    metrics_to_plot = [
        ('accuracy', 'Accuracy (%)', axes[0, 0]),
        ('recall', 'Recall (%)', axes[0, 1]),
        ('precision', 'Precision (%)', axes[1, 0]),
        ('f1', 'F1 Score (%)', axes[1, 1])
    ]
    
    for metric_key, ylabel, ax in metrics_to_plot:
        baseline_values = [baseline_results[snr][metric_key] for snr in snr_levels]
        enhanced_values = [enhanced_results[snr][metric_key] for snr in snr_levels]
        
        ax.plot(snr_levels, baseline_values, 'o-', label='Baseline', linewidth=2, markersize=8)
        ax.plot(snr_levels, enhanced_values, 's-', label='Enhanced (Matched Bank)', linewidth=2, markersize=8)
        
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{ylabel} vs SNR', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add improvement annotations
        for i, snr in enumerate(snr_levels):
            improvement = enhanced_values[i] - baseline_values[i]
            if improvement > 2:  # Show significant improvements
                mid_y = (baseline_values[i] + enhanced_values[i]) / 2
                ax.annotate(f'+{improvement:.1f}%', 
                           xy=(snr, mid_y),
                           fontsize=9, 
                           color='green',
                           fontweight='bold',
                           ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved SNR comparison plot to {output_path}")
    plt.close()


def plot_confusion_matrices(
    baseline_cm,
    enhanced_cm,
    class_names,
    snr_db,
    output_path
):
    """Plot confusion matrix comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Confusion Matrices at SNR = {snr_db} dB", fontsize=14, fontweight='bold')
    
    # Normalize confusion matrices
    baseline_cm_norm = baseline_cm.astype('float') / baseline_cm.sum(axis=1)[:, np.newaxis]
    enhanced_cm_norm = enhanced_cm.astype('float') / enhanced_cm.sum(axis=1)[:, np.newaxis]
    
    # Plot baseline
    sns.heatmap(baseline_cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Normalized Count'})
    axes[0].set_title('Baseline Model', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=11)
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    
    # Plot enhanced
    sns.heatmap(enhanced_cm_norm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Normalized Count'})
    axes[1].set_title('Enhanced Model (Matched Bank)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=11)
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved confusion matrix comparison to {output_path}")
    plt.close()


def plot_roc_curves(
    baseline_probs,
    enhanced_probs,
    targets,
    class_names,
    output_path
):
    """Plot ROC curves for each class."""
    from sklearn.preprocessing import label_binarize
    
    n_classes = len(class_names)
    
    # Binarize targets
    targets_bin = label_binarize(targets, classes=range(n_classes))
    
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5))
    if n_classes == 1:
        axes = [axes]
    
    fig.suptitle("ROC Curves: Baseline vs Enhanced", fontsize=14, fontweight='bold')
    
    for i, (cls_name, ax) in enumerate(zip(class_names, axes)):
        # Baseline ROC
        fpr_base, tpr_base, _ = roc_curve(targets_bin[:, i], baseline_probs[:, i])
        auc_base = auc(fpr_base, tpr_base)
        
        # Enhanced ROC
        fpr_enh, tpr_enh, _ = roc_curve(targets_bin[:, i], enhanced_probs[:, i])
        auc_enh = auc(fpr_enh, tpr_enh)
        
        ax.plot(fpr_base, tpr_base, 'b-', label=f'Baseline (AUC={auc_base:.3f})', linewidth=2)
        ax.plot(fpr_enh, tpr_enh, 'g-', label=f'Enhanced (AUC={auc_enh:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
        
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'{cls_name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Highlight improvement
        improvement = auc_enh - auc_base
        color = 'green' if improvement > 0 else 'red'
        ax.text(0.6, 0.2, f'Δ AUC: {improvement:+.3f}',
                fontsize=10, fontweight='bold', color=color,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved ROC curves to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate matched filter bank models")
    
    parser.add_argument("--baseline-model", type=str,
                        help="Path to baseline model checkpoint")
    parser.add_argument("--enhanced-model", type=str,
                        help="Path to enhanced model checkpoint")
    parser.add_argument("--test-data", type=str,
                        help="Path to test dataset")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results",
                        help="Output directory for results")
    parser.add_argument("--snr-levels", type=float, nargs="+",
                        default=[30, 20, 15, 10, 5, 0, -5],
                        help="SNR levels to test (dB)")
    parser.add_argument("--class-names", type=str, nargs="+",
                        default=["Drone", "Helicopter", "Background"],
                        help="Class names")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    print("\n" + "="*80)
    print("Loading models...")
    print("="*80)
    
    # PLACEHOLDER: Replace with actual model loading
    print("⚠ WARNING: Model loading is a placeholder!")
    print("Replace with your actual model loading code.")
    
    # baseline_model, _ = load_model(args.baseline_model, device)
    # enhanced_model, _ = load_model(args.enhanced_model, device)
    
    # Load test data
    print("\n" + "="*80)
    print("Loading test data...")
    print("="*80)
    
    # PLACEHOLDER: Replace with actual data loading
    print("⚠ WARNING: Data loading is a placeholder!")
    print("Replace with your actual data loading code.")
    
    # test_dataset = YourDataset(args.test_data, split="test")
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers
    # )
    
    # Evaluate at different SNR levels
    print("\n" + "="*80)
    print("Evaluating models across SNR levels...")
    print("="*80)
    
    # PLACEHOLDER: Skip actual evaluation
    print("⚠ Skipping actual evaluation - no models/data loaded.")
    print("\nTo complete this script:")
    print("1. Add your model loading code")
    print("2. Add your dataset loading code")
    print("3. Uncomment the evaluation sections below")
    
    # baseline_results = compute_snr_curve(
    #     baseline_model, test_loader, args.snr_levels, device, "Baseline"
    # )
    
    # enhanced_results = compute_snr_curve(
    #     enhanced_model, test_loader, args.snr_levels, device, "Enhanced"
    # )
    
    # Save results
    # results = {
    #     'baseline': baseline_results,
    #     'enhanced': enhanced_results,
    #     'config': vars(args)
    # }
    
    # with open(output_dir / "snr_evaluation_results.json", "w") as f:
    #     json.dump(results, f, indent=2, default=str)
    
    # Generate plots
    # print("\n" + "="*80)
    # print("Generating comparison plots...")
    # print("="*80)
    
    # # SNR curves
    # plot_snr_comparison(
    #     baseline_results,
    #     enhanced_results,
    #     output_dir / "snr_curve_comparison.png"
    # )
    
    # # Confusion matrices (at worst SNR)
    # worst_snr = min(args.snr_levels)
    # plot_confusion_matrices(
    #     np.array(baseline_results[worst_snr]['confusion_matrix']),
    #     np.array(enhanced_results[worst_snr]['confusion_matrix']),
    #     args.class_names,
    #     worst_snr,
    #     output_dir / f"confusion_matrices_snr_{worst_snr}db.png"
    # )
    
    # # ROC curves (at worst SNR)
    # plot_roc_curves(
    #     baseline_results[worst_snr]['probabilities'],
    #     enhanced_results[worst_snr]['probabilities'],
    #     baseline_results[worst_snr]['targets'],
    #     args.class_names,
    #     output_dir / f"roc_curves_snr_{worst_snr}db.png"
    # )
    
    print("\n" + "="*80)
    print("✓ Evaluation script template ready!")
    print("Add your model/data loading code and uncomment evaluation sections.")
    print("="*80)


if __name__ == "__main__":
    main()
