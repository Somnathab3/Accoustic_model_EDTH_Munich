"""
Comprehensive evaluation and calibration module
Implements macro-F1, per-class metrics, ECE, temperature scaling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 3
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics
    
    Args:
        predictions: Predicted class indices (batch,)
        targets: True class indices (batch,)
        num_classes: Number of classes
    
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Overall metrics
    accuracy = (preds_np == targets_np).mean()
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        targets_np,
        preds_np,
        average=None,
        zero_division=0
    )
    
    # Macro-averaged metrics (important for imbalanced classes)
    macro_f1 = f1_score(targets_np, preds_np, average='macro', zero_division=0)
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    
    # Confusion matrix
    cm = confusion_matrix(targets_np, preds_np, labels=list(range(num_classes)))
    
    # Balanced accuracy (average of per-class accuracies)
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    balanced_accuracy = per_class_acc.mean()
    
    metrics = {
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_accuracy),
        'macro_f1': float(macro_f1),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
    }
    
    # Per-class metrics
    class_names = ['background', 'drone', 'helicopter']
    for i, class_name in enumerate(class_names):
        metrics[f'{class_name}_f1'] = float(f1[i])
        metrics[f'{class_name}_precision'] = float(precision[i])
        metrics[f'{class_name}_recall'] = float(recall[i])
        metrics[f'{class_name}_support'] = int(support[i])
    
    return metrics, cm


def compute_roc_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 3
) -> Dict[str, float]:
    """
    Compute ROC-AUC and PR-AUC metrics
    
    Args:
        probabilities: Class probabilities (batch, num_classes)
        targets: True class indices (batch,)
        num_classes: Number of classes
    
    Returns:
        Dictionary of ROC/PR metrics
    """
    probs_np = probabilities.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # One-hot encode targets
    targets_onehot = np.eye(num_classes)[targets_np]
    
    metrics = {}
    
    # Per-class ROC-AUC and PR-AUC
    class_names = ['background', 'drone', 'helicopter']
    roc_aucs = []
    pr_aucs = []
    
    for i, class_name in enumerate(class_names):
        try:
            roc_auc = roc_auc_score(targets_onehot[:, i], probs_np[:, i])
            pr_auc = average_precision_score(targets_onehot[:, i], probs_np[:, i])
            
            metrics[f'{class_name}_roc_auc'] = float(roc_auc)
            metrics[f'{class_name}_pr_auc'] = float(pr_auc)
            
            roc_aucs.append(roc_auc)
            pr_aucs.append(pr_auc)
        except:
            # Handle case where class is not present in batch
            pass
    
    if roc_aucs:
        metrics['mean_roc_auc'] = float(np.mean(roc_aucs))
        metrics['mean_pr_auc'] = float(np.mean(pr_aucs))
    
    return metrics


def compute_calibration_error(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error (ECE)
    
    Measures the difference between predicted confidence and actual accuracy
    Lower ECE = better calibrated model
    
    Args:
        probabilities: Class probabilities (batch, num_classes)
        targets: True class indices (batch,)
        n_bins: Number of bins for calibration curve
    
    Returns:
        ece: Expected calibration error
        bin_accuracies: Accuracy in each bin
        bin_confidences: Average confidence in each bin
        bin_counts: Number of samples in each bin
    """
    probs_np = probabilities.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Get predicted class and confidence
    confidences = probs_np.max(axis=1)
    predictions = probs_np.argmax(axis=1)
    accuracies = (predictions == targets_np)
    
    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Compute bin statistics
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_accuracies[i] = accuracies[mask].mean()
            bin_confidences[i] = confidences[mask].mean()
            bin_counts[i] = mask.sum()
    
    # Compute ECE
    ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / len(confidences)
    
    return float(ece), bin_accuracies, bin_confidences, bin_counts


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for model calibration
    From "On Calibration of Modern Neural Networks" (Guo et al., 2017)
    
    Applies a single temperature parameter to logits before softmax
    """
    
    def __init__(self, init_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling"""
        return logits / self.temperature
    
    def fit(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100
    ):
        """
        Optimize temperature on validation set
        
        Args:
            logits: Model logits (batch, num_classes)
            targets: True labels (batch,)
            lr: Learning rate
            max_iter: Maximum iterations
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self(logits)
            loss = F.cross_entropy(scaled_logits, targets)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        print(f"Optimal temperature: {self.temperature.item():.4f}")


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    compute_roc: bool = True,
    compute_calibration: bool = True
) -> Dict[str, any]:
    """
    Comprehensive model evaluation
    
    Args:
        model: Model to evaluate
        dataloader: Validation dataloader
        device: Device to use
        compute_roc: Whether to compute ROC/PR metrics
        compute_calibration: Whether to compute calibration metrics
    
    Returns:
        Dictionary of metrics and diagnostics
    """
    model.eval()
    
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for spectrograms, targets in dataloader:
            spectrograms = spectrograms.to(device)
            targets = targets.to(device)
            
            logits = model(spectrograms)
            
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate all batches
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute probabilities and predictions
    all_probs = F.softmax(all_logits, dim=1)
    all_preds = all_logits.argmax(dim=1)
    
    # Core metrics
    metrics, confusion_mat = compute_metrics(all_preds, all_targets)
    
    # ROC metrics
    if compute_roc:
        roc_metrics = compute_roc_metrics(all_probs, all_targets)
        metrics.update(roc_metrics)
    
    # Calibration
    if compute_calibration:
        ece, bin_acc, bin_conf, bin_counts = compute_calibration_error(all_probs, all_targets)
        metrics['ece'] = ece
    
    # Add raw data for further analysis
    results = {
        'metrics': metrics,
        'confusion_matrix': confusion_mat,
        'logits': all_logits,
        'probabilities': all_probs,
        'predictions': all_preds,
        'targets': all_targets
    }
    
    if compute_calibration:
        results['calibration'] = {
            'ece': ece,
            'bin_accuracies': bin_acc,
            'bin_confidences': bin_conf,
            'bin_counts': bin_counts
        }
    
    return results


def print_evaluation_report(metrics: Dict[str, float], confusion_mat: np.ndarray):
    """Print formatted evaluation report"""
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    
    # Overall metrics
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"  Macro Precision:   {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:      {metrics['macro_recall']:.4f}")
    
    # Calibration
    if 'ece' in metrics:
        print(f"  ECE (calibration): {metrics['ece']:.4f}")
    
    # ROC metrics
    if 'mean_roc_auc' in metrics:
        print(f"  Mean ROC-AUC:      {metrics['mean_roc_auc']:.4f}")
        print(f"  Mean PR-AUC:       {metrics['mean_pr_auc']:.4f}")
    
    # Per-class metrics
    class_names = ['background', 'drone', 'helicopter']
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<12} {'F1':<8} {'Precision':<12} {'Recall':<8} {'Support':<8}")
    print("-" * 60)
    
    for class_name in class_names:
        f1 = metrics[f'{class_name}_f1']
        prec = metrics[f'{class_name}_precision']
        rec = metrics[f'{class_name}_recall']
        sup = metrics[f'{class_name}_support']
        print(f"{class_name:<12} {f1:<8.4f} {prec:<12.4f} {rec:<8.4f} {sup:<8}")
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    print(f"{'':>12} " + " ".join(f"{name:>12}" for name in class_names))
    for i, class_name in enumerate(class_names):
        row = " ".join(f"{confusion_mat[i, j]:>12}" for j in range(len(class_names)))
        print(f"{class_name:>12} {row}")
    
    print("="*60 + "\n")


if __name__ == '__main__':
    # Test evaluation functions
    batch_size = 100
    num_classes = 3
    
    # Generate test data
    logits = torch.randn(batch_size, num_classes)
    probs = F.softmax(logits, dim=1)
    preds = logits.argmax(dim=1)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test metrics computation
    print("Testing metrics computation...")
    metrics, cm = compute_metrics(preds, targets)
    print_evaluation_report(metrics, cm)
    
    # Test ROC metrics
    print("\nTesting ROC metrics...")
    roc_metrics = compute_roc_metrics(probs, targets)
    for key, value in roc_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test calibration
    print("\nTesting calibration...")
    ece, _, _, _ = compute_calibration_error(probs, targets)
    print(f"  ECE: {ece:.4f}")
    
    # Test temperature scaling
    print("\nTesting temperature scaling...")
    temp_scaler = TemperatureScaling()
    temp_scaler.fit(logits, targets)
