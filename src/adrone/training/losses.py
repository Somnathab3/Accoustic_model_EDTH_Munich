"""
Loss functions and training utilities for acoustic drone detection
Implements label smoothing, class-balanced loss, and focal loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy with label smoothing
    Prevents overconfident predictions and improves calibration
    From "Rethinking the Inception Architecture" (Szegedy et al., 2016)
    """
    
    def __init__(self, smoothing: float = 0.05, weight: Optional[torch.Tensor] = None):
        """
        Args:
            smoothing: Label smoothing factor (typically 0.05-0.1)
            weight: Class weights for imbalanced datasets
        """
        super().__init__()
        self.smoothing = smoothing
        self.register_buffer('weight', weight)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Logits of shape (batch, num_classes)
            target: Hard labels of shape (batch,) or soft labels of shape (batch, num_classes)
        """
        num_classes = pred.size(-1)
        
        # Convert hard labels to soft labels if needed
        if target.dim() == 1:
            # Create soft labels
            soft_target = torch.full_like(pred, self.smoothing / (num_classes - 1))
            soft_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        else:
            # Already soft labels (from mixup)
            soft_target = target
        
        # Compute log probabilities
        log_probs = F.log_softmax(pred, dim=-1)
        
        # Weighted cross entropy
        if self.weight is not None:
            # Apply class weights
            weight_expanded = self.weight.unsqueeze(0).expand_as(log_probs)
            loss = -(soft_target * log_probs * weight_expanded).sum(dim=-1)
        else:
            loss = -(soft_target * log_probs).sum(dim=-1)
        
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    From "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    Focuses training on hard examples by down-weighting easy examples
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Class weights (optional)
            gamma: Focusing parameter (0 = cross-entropy, 2 = strong focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.register_buffer('alpha', alpha)
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Logits of shape (batch, num_classes)
            target: Hard labels of shape (batch,)
        """
        # Compute probabilities
        probs = F.softmax(pred, dim=-1)
        
        # Get probability of true class
        target_probs = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        
        # Compute focal term: (1 - p)^gamma
        focal_weight = (1.0 - target_probs) ** self.gamma
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_weight = self.alpha[target]
            focal_loss = alpha_weight * focal_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on Effective Number of Samples
    From "Class-Balanced Loss Based on Effective Number of Samples" (Cui et al., 2019)
    
    Addresses class imbalance by re-weighting based on effective number of samples
    """
    
    def __init__(
        self,
        samples_per_class: torch.Tensor,
        num_classes: int,
        beta: float = 0.9999,
        smoothing: float = 0.05
    ):
        """
        Args:
            samples_per_class: Number of samples per class [class0_count, class1_count, ...]
            num_classes: Total number of classes
            beta: Hyperparameter for effective number (0.9-0.9999, higher = more aggressive reweighting)
            smoothing: Label smoothing factor
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        
        # Compute effective number of samples
        effective_num = 1.0 - torch.pow(beta, samples_per_class)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes  # Normalize
        
        self.register_buffer('weights', weights)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Logits of shape (batch, num_classes)
            target: Hard labels of shape (batch,)
        """
        # Apply label smoothing
        soft_target = torch.full_like(pred, self.smoothing / (self.num_classes - 1))
        soft_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        # Compute log probabilities
        log_probs = F.log_softmax(pred, dim=-1)
        
        # Apply class-balanced weights
        weight_expanded = self.weights.unsqueeze(0).expand_as(log_probs)
        loss = -(soft_target * log_probs * weight_expanded).sum(dim=-1)
        
        return loss.mean()


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss
    Combines hard target loss with soft target (teacher) loss
    From "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)
    """
    
    def __init__(
        self,
        temperature: float = 3.0,
        alpha: float = 0.5,
        base_criterion: Optional[nn.Module] = None
    ):
        """
        Args:
            temperature: Temperature for softening probabilities (2-5 typical)
            alpha: Weight for distillation loss (0-1, rest goes to hard target loss)
            base_criterion: Base loss for hard targets (default: CrossEntropyLoss)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.base_criterion = base_criterion if base_criterion else nn.CrossEntropyLoss()
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            student_logits: Student model predictions (batch, num_classes)
            teacher_logits: Teacher model predictions (batch, num_classes)
            target: Hard labels (batch,)
        
        Returns:
            Combined distillation loss
        """
        # Hard target loss
        hard_loss = self.base_criterion(student_logits, target)
        
        # Soft target loss (KL divergence between student and teacher)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL divergence * T^2 (temperature scaling)
        soft_loss = F.kl_div(
            soft_student,
            soft_teacher,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Combine losses
        return self.alpha * soft_loss + (1.0 - self.alpha) * hard_loss


class CombinedLoss(nn.Module):
    """
    Combination of label smoothing cross entropy and auxiliary losses
    Can be extended to include contrastive loss, etc.
    """
    
    def __init__(
        self,
        use_focal: bool = False,
        use_label_smoothing: bool = True,
        use_class_balanced: bool = False,
        class_weights: Optional[torch.Tensor] = None,
        samples_per_class: Optional[torch.Tensor] = None,
        num_classes: int = 3,
        smoothing: float = 0.05,
        focal_gamma: float = 2.0,
        cb_beta: float = 0.9999
    ):
        super().__init__()
        
        self.use_focal = use_focal
        self.use_label_smoothing = use_label_smoothing
        self.use_class_balanced = use_class_balanced
        
        if use_class_balanced and samples_per_class is not None:
            # Class-balanced loss (best for imbalanced data)
            self.loss_fn = ClassBalancedLoss(
                samples_per_class=samples_per_class,
                num_classes=num_classes,
                beta=cb_beta,
                smoothing=smoothing
            )
        elif use_focal:
            # Focal loss (good for hard examples)
            self.loss_fn = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        elif use_label_smoothing:
            # Label smoothing CE (default)
            self.loss_fn = LabelSmoothingCrossEntropy(smoothing=smoothing, weight=class_weights)
        else:
            # Standard CE
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss"""
        if self.use_focal or self.use_class_balanced:
            # Focal and CB loss only work with hard labels
            if target.dim() > 1:
                target = target.argmax(dim=1)
        return self.loss_fn(pred, target)


def cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01
):
    """
    Create a learning rate scheduler with warmup and cosine decay
    
    Args:
        optimizer: Optimizer instance
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum LR as ratio of initial LR
    """
    def lr_lambda(current_step: int):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)))
        
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = 'max'
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'max' for metrics to maximize (accuracy), 'min' for minimize (loss)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop
        
        Args:
            score: Current metric value
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        # Check for improvement
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class MetricsTracker:
    """Track training and validation metrics"""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_macro_f1': [],
            'lr': []
        }
    
    def update(self, **metrics):
        """Update metrics"""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def get_best_epoch(self, metric: str = 'val_macro_f1') -> int:
        """Get epoch with best metric"""
        if not self.history[metric]:
            return 0
        return int(torch.argmax(torch.tensor(self.history[metric])).item())
    
    def save(self, path: str):
        """Save metrics history to JSON"""
        import json
        
        # Convert to serializable format
        history_serializable = {
            k: [float(v) for v in values]
            for k, values in self.history.items()
        }
        
        with open(path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        print(f"Saved metrics to {path}")
    
    def plot(self, save_path: Optional[str] = None):
        """Plot training curves"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Loss
            axes[0, 0].plot(self.history['train_loss'], label='Train')
            axes[0, 0].plot(self.history['val_loss'], label='Val')
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Accuracy
            axes[0, 1].plot(self.history['train_acc'], label='Train')
            axes[0, 1].plot(self.history['val_acc'], label='Val')
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Macro F1
            axes[1, 0].plot(self.history['val_macro_f1'])
            axes[1, 0].set_title('Validation Macro F1')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].grid(True)
            
            # Learning rate
            axes[1, 1].plot(self.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Saved plot to {save_path}")
            else:
                plt.show()
            
            plt.close()
        
        except ImportError:
            print("Matplotlib not available, skipping plot")


if __name__ == '__main__':
    # Test loss functions
    batch_size = 8
    num_classes = 3
    
    # Test data
    pred = torch.randn(batch_size, num_classes)
    target = torch.randint(0, num_classes, (batch_size,))
    
    # Test label smoothing
    print("Testing Label Smoothing Cross Entropy...")
    loss_fn = LabelSmoothingCrossEntropy(smoothing=0.05)
    loss = loss_fn(pred, target)
    print(f"Loss: {loss.item():.4f}\n")
    
    # Test focal loss
    print("Testing Focal Loss...")
    loss_fn = FocalLoss(gamma=2.0)
    loss = loss_fn(pred, target)
    print(f"Loss: {loss.item():.4f}\n")
    
    # Test with soft labels (mixup)
    soft_target = F.one_hot(target, num_classes).float()
    soft_target = 0.7 * soft_target + 0.3 * soft_target.roll(1, dims=0)
    
    print("Testing with soft labels...")
    loss_fn = LabelSmoothingCrossEntropy(smoothing=0.05)
    loss = loss_fn(pred, soft_target)
    print(f"Loss: {loss.item():.4f}")
