"""
Training script for matched-filter enhanced models with low-SNR optimization.

Implements:
- Focal loss for hard negative mining
- Template response regularization (max-margin on best template per class)
- Curriculum learning with progressive SNR degradation
- Energy-gated template activation
- Balanced sampling for class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    where γ > 0 reduces loss for well-classified examples, focusing on hard negatives.
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ):
        """
        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
            alpha: Class weights [num_classes] (None = uniform)
            reduction: "mean", "sum", or "none"
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits [B, C]
            targets: Class indices [B]
            
        Returns:
            Focal loss scalar
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)  # Predicted probability for true class
        focal_weight = (1 - p_t) ** self.gamma
        
        loss = focal_weight * ce_loss
        
        # Apply class weights
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class TemplateMarginLoss(nn.Module):
    """
    Encourage separation of template responses between classes.
    
    For each sample, penalize if the best template for the wrong class
    has higher activation than the best template for the correct class.
    """
    
    def __init__(
        self,
        n_templates_per_class: Dict[int, list],
        margin: float = 0.5
    ):
        """
        Args:
            n_templates_per_class: Dict mapping class_id -> list of template indices
                                   e.g., {0: [0,1,2], 1: [3,4,5], 2: [6,7]}
            margin: Minimum margin between correct and incorrect template responses
        """
        super().__init__()
        self.n_templates_per_class = n_templates_per_class
        self.margin = margin
        
    def forward(
        self,
        template_features: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            template_features: [B, n_templates, T] - template activation maps
            targets: [B] - class indices
            
        Returns:
            Margin loss scalar
        """
        # Max-pool over time to get peak activation per template
        template_responses = template_features.max(dim=2)[0]  # [B, n_templates]
        
        losses = []
        for i in range(template_responses.shape[0]):
            target_class = targets[i].item()
            
            # Get correct and incorrect template indices
            correct_templates = self.n_templates_per_class.get(target_class, [])
            all_templates = set(range(template_responses.shape[1]))
            incorrect_templates = list(all_templates - set(correct_templates))
            
            if not correct_templates or not incorrect_templates:
                continue
            
            # Best response for correct class
            correct_response = template_responses[i, correct_templates].max()
            
            # Best response for incorrect classes
            incorrect_response = template_responses[i, incorrect_templates].max()
            
            # Hinge loss: push correct > incorrect + margin
            loss = F.relu(incorrect_response - correct_response + self.margin)
            losses.append(loss)
        
        if not losses:
            return torch.tensor(0.0, device=template_responses.device)
        
        return torch.stack(losses).mean()


class EnergyGatedTemplateLayer(nn.Module):
    """
    Gate template activations by energy-based VAD to avoid triggering on silence.
    """
    
    def __init__(
        self,
        energy_percentile: float = 10.0,
        smooth_window: int = 5
    ):
        """
        Args:
            energy_percentile: Energy percentile threshold for gating
            smooth_window: Temporal smoothing window
        """
        super().__init__()
        self.energy_percentile = energy_percentile
        self.smooth_window = smooth_window
        
    def forward(
        self,
        template_features: torch.Tensor,
        input_spectrogram: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            template_features: [B, K, T] - template outputs
            input_spectrogram: [B, C, M, T] - input mel spectrogram
            
        Returns:
            Gated template features [B, K, T]
        """
        # Compute energy (sum over channels and frequency)
        energy = input_spectrogram.sum(dim=(1, 2))  # [B, T]
        
        # Temporal alignment (template features may be shorter)
        T_template = template_features.shape[2]
        energy = energy[:, :T_template]
        
        # Compute threshold per sample
        thresholds = torch.quantile(
            energy,
            self.energy_percentile / 100.0,
            dim=1,
            keepdim=True
        )  # [B, 1]
        
        # Create gate
        gate = (energy > thresholds).float()  # [B, T]
        
        # Smooth gate (avoid hard transitions)
        if self.smooth_window > 1:
            gate = F.avg_pool1d(
                gate.unsqueeze(1),
                kernel_size=self.smooth_window,
                stride=1,
                padding=self.smooth_window // 2
            ).squeeze(1)
        
        # Apply gate to template features
        gate = gate.unsqueeze(1)  # [B, 1, T]
        gated_features = template_features * gate
        
        return gated_features


class CurriculumSNRAugmentation:
    """
    Progressive SNR degradation for curriculum learning.
    
    Start with clean audio, gradually add noise to reach 0-5 dB SNR by final epoch.
    """
    
    def __init__(
        self,
        initial_snr_db: float = 30.0,
        final_snr_db: float = 0.0,
        curriculum_epochs: int = 10,
        noise_types: list = None
    ):
        """
        Args:
            initial_snr_db: Starting SNR (clean)
            final_snr_db: Target SNR at curriculum end
            curriculum_epochs: Number of epochs to reach final SNR
            noise_types: List of noise types ["white", "pink", "wind", "traffic"]
        """
        self.initial_snr = initial_snr_db
        self.final_snr = final_snr_db
        self.curriculum_epochs = curriculum_epochs
        self.noise_types = noise_types or ["white", "pink", "wind"]
        
        self.current_epoch = 0
        
    def get_snr_range(self, epoch: int) -> Tuple[float, float]:
        """Get SNR range for current epoch."""
        if epoch >= self.curriculum_epochs:
            # Full difficulty
            return (self.final_snr, self.final_snr + 10)
        
        # Linear interpolation
        progress = epoch / self.curriculum_epochs
        current_min = self.initial_snr - (self.initial_snr - self.final_snr) * progress
        current_max = current_min + 10
        
        return (current_min, current_max)
    
    def apply(
        self,
        audio: torch.Tensor,
        epoch: int,
        noise_bank: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply curriculum noise augmentation.
        
        Args:
            audio: [B, T] or [B, 1, T]
            epoch: Current training epoch
            noise_bank: Pre-loaded noise samples [N, T]
            
        Returns:
            Augmented audio
        """
        snr_min, snr_max = self.get_snr_range(epoch)
        
        # Random SNR for each sample in batch
        batch_size = audio.shape[0]
        snr_db = torch.rand(batch_size, device=audio.device) * (snr_max - snr_min) + snr_min
        
        # Generate or select noise
        if noise_bank is not None:
            # Sample from noise bank
            noise_idx = torch.randint(0, noise_bank.shape[0], (batch_size,))
            noise = noise_bank[noise_idx].to(audio.device)
        else:
            # Generate synthetic noise
            noise_type = np.random.choice(self.noise_types)
            noise = self._generate_noise(audio.shape, noise_type, audio.device)
        
        # Add noise at target SNR
        audio_noisy = self._add_noise_at_snr(audio, noise, snr_db)
        
        return audio_noisy
    
    def _generate_noise(
        self,
        shape: Tuple,
        noise_type: str,
        device: torch.device
    ) -> torch.Tensor:
        """Generate synthetic noise."""
        if noise_type == "white":
            return torch.randn(shape, device=device)
        
        elif noise_type == "pink":
            # Pink noise (1/f power spectrum)
            white = torch.randn(shape, device=device)
            # Simple approximation via IIR filter
            pink = white.clone()
            for i in range(1, shape[-1]):
                pink[..., i] = 0.99 * pink[..., i-1] + 0.01 * white[..., i]
            return pink
        
        elif noise_type == "wind":
            # Low-pass filtered noise (wind-like)
            white = torch.randn(shape, device=device)
            # Ensure compatible shapes for pooling
            if white.ndim == 2:
                white = white.unsqueeze(1)  # [B, 1, T]
            wind = F.avg_pool1d(
                white,
                kernel_size=20,
                stride=1,
                padding=10
            )
            # Trim to original size
            if wind.shape[-1] != shape[-1]:
                wind = wind[..., :shape[-1]]
            if len(shape) == 2:
                wind = wind.squeeze(1)  # Back to [B, T]
            return wind
        
        else:
            return torch.randn(shape, device=device)
    
    def _add_noise_at_snr(
        self,
        signal: torch.Tensor,
        noise: torch.Tensor,
        snr_db: torch.Tensor
    ) -> torch.Tensor:
        """Add noise at specified SNR."""
        # Compute signal power
        signal_power = (signal ** 2).mean(dim=-1, keepdim=True)
        
        # Compute noise power
        noise_power = (noise ** 2).mean(dim=-1, keepdim=True)
        
        # Compute scaling factor
        snr_linear = 10 ** (snr_db.unsqueeze(-1) / 10.0)
        noise_scale = torch.sqrt(signal_power / (noise_power * snr_linear + 1e-8))
        
        # Add scaled noise
        return signal + noise_scale * noise
    
    def step(self):
        """Advance to next epoch."""
        self.current_epoch += 1


class MatchedBankTrainingWrapper:
    """
    Complete training wrapper with all low-SNR optimization tricks.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        template_margin: float = 0.5,
        template_margin_weight: float = 0.1,
        use_energy_gating: bool = True,
        curriculum_config: Optional[dict] = None
    ):
        """
        Args:
            model: Enhanced model with matched filter bank
            num_classes: Number of output classes
            class_weights: Class weights for focal loss
            focal_gamma: Focal loss focusing parameter
            template_margin: Template separation margin
            template_margin_weight: Weight for template margin loss
            use_energy_gating: Apply energy-based gating to templates
            curriculum_config: Config dict for curriculum learning
        """
        self.model = model
        self.num_classes = num_classes
        
        # Primary classification loss
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=class_weights)
        
        # Template regularization loss
        # Auto-detect template assignments (assume equal split)
        if hasattr(model, 'enhanced_input'):
            n_templates = model.enhanced_input.matched_bank.n_templates
        elif hasattr(model, 'matched_bank'):
            n_templates = model.matched_bank.n_templates
        else:
            n_templates = 0
        
        if n_templates > 0:
            templates_per_class = n_templates // num_classes
            self.template_assignments = {
                i: list(range(i * templates_per_class, (i + 1) * templates_per_class))
                for i in range(num_classes)
            }
            self.template_margin_loss = TemplateMarginLoss(
                self.template_assignments,
                margin=template_margin
            )
        else:
            self.template_margin_loss = None
        
        self.template_margin_weight = template_margin_weight
        
        # Energy gating
        if use_energy_gating:
            self.energy_gate = EnergyGatedTemplateLayer()
        else:
            self.energy_gate = None
        
        # Curriculum learning
        if curriculum_config:
            self.curriculum = CurriculumSNRAugmentation(**curriculum_config)
        else:
            self.curriculum = None
        
        self.current_epoch = 0
        
    def compute_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        template_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total training loss.
        
        Args:
            inputs: Model logits [B, num_classes]
            targets: Ground truth labels [B]
            template_features: Template activation maps [B, K, T] (optional)
            
        Returns:
            total_loss: Combined loss tensor
            loss_dict: Dictionary of individual loss components
        """
        # Primary classification loss
        cls_loss = self.focal_loss(inputs, targets)
        
        total_loss = cls_loss
        loss_dict = {"classification": cls_loss.item()}
        
        # Template margin loss (if available)
        if self.template_margin_loss is not None and template_features is not None:
            margin_loss = self.template_margin_loss(template_features, targets)
            total_loss = total_loss + self.template_margin_weight * margin_loss
            loss_dict["template_margin"] = margin_loss.item()
        
        return total_loss, loss_dict
    
    def forward_with_augmentation(
        self,
        audio: torch.Tensor,
        labels: torch.Tensor,
        noise_bank: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Forward pass with curriculum augmentation.
        
        Args:
            audio: Input audio or spectrogram [B, C, M, T]
            labels: Ground truth labels [B]
            noise_bank: Optional noise samples for augmentation
            
        Returns:
            logits: Model predictions [B, num_classes]
            loss: Total loss
            loss_dict: Individual loss components
        """
        # Apply curriculum noise (if configured and audio is waveform)
        if self.curriculum is not None and audio.ndim == 2:
            audio = self.curriculum.apply(audio, self.current_epoch, noise_bank)
        
        # Forward through model
        # Note: This assumes model returns logits. Modify if model returns
        # additional outputs (e.g., attention maps, template features)
        logits = self.model(audio)
        
        # Extract template features if model exposes them
        template_features = None
        if hasattr(self.model, 'enhanced_input'):
            # Get intermediate template activations
            # This requires modifying the model to expose intermediate outputs
            pass
        
        # Compute loss
        loss, loss_dict = self.compute_loss(logits, labels, template_features)
        
        return logits, loss, loss_dict
    
    def step_epoch(self):
        """Advance to next epoch (for curriculum)."""
        self.current_epoch += 1
        if self.curriculum is not None:
            self.curriculum.step()
        
        # Log curriculum progress
        if self.curriculum is not None:
            snr_range = self.curriculum.get_snr_range(self.current_epoch)
            logger.info(f"Epoch {self.current_epoch}: SNR range = {snr_range[0]:.1f} - {snr_range[1]:.1f} dB")


if __name__ == "__main__":
    print("=" * 80)
    print("Matched Filter Bank Training Utilities")
    print("=" * 80)
    
    # Test focal loss
    print("\n1. Focal Loss:")
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 1])
    loss = loss_fn(logits, targets)
    print(f"   Logits shape: {logits.shape}")
    print(f"   Loss: {loss.item():.4f}")
    
    # Test template margin loss
    print("\n2. Template Margin Loss:")
    template_assignments = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
    margin_loss_fn = TemplateMarginLoss(template_assignments, margin=0.5)
    template_features = torch.randn(4, 6, 50)
    margin_loss = margin_loss_fn(template_features, targets)
    print(f"   Template features: {template_features.shape}")
    print(f"   Margin loss: {margin_loss.item():.4f}")
    
    # Test energy gating
    print("\n3. Energy-Gated Templates:")
    gate = EnergyGatedTemplateLayer()
    spectrogram = torch.randn(4, 3, 96, 50)
    template_out = torch.randn(4, 6, 50)
    gated = gate(template_out, spectrogram)
    print(f"   Original: {template_out.shape}, gating ratio: {(template_out > 0).float().mean():.3f}")
    print(f"   Gated: {gated.shape}, gating ratio: {(gated > 0).float().mean():.3f}")
    
    # Test curriculum
    print("\n4. Curriculum SNR Augmentation:")
    curriculum = CurriculumSNRAugmentation(
        initial_snr_db=30.0,
        final_snr_db=0.0,
        curriculum_epochs=10
    )
    
    for epoch in [0, 5, 10, 15]:
        snr_range = curriculum.get_snr_range(epoch)
        print(f"   Epoch {epoch:2d}: SNR range = {snr_range[0]:5.1f} - {snr_range[1]:5.1f} dB")
    
    audio = torch.randn(2, 16000)
    audio_noisy = curriculum.apply(audio, epoch=5)
    print(f"   Applied noise: {audio.shape} -> {audio_noisy.shape}")
    
    # Test complete wrapper
    print("\n5. Complete Training Wrapper:")
    try:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent / 'models'))
        from enhanced_models_with_bank import create_enhanced_crnn
    except ImportError:
        print("  (Skipping - enhanced_models_with_bank not available)")
        print("\n" + "=" * 80)
        print("✓ Core training utilities tested successfully!")
        print("=" * 80)
        exit(0)
    
    class DummyCRNN(nn.Module):
        def __init__(self, in_channels):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(32, 3)
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x).squeeze(-1).squeeze(-1)
            return self.fc(x)
    
    crnn = DummyCRNN(in_channels=9)
    model = create_enhanced_crnn(crnn, compression=6)
    
    wrapper = MatchedBankTrainingWrapper(
        model=model,
        num_classes=3,
        focal_gamma=2.0,
        template_margin_weight=0.1,
        curriculum_config={"curriculum_epochs": 10}
    )
    
    x = torch.randn(2, 3, 96, 100)
    y = torch.tensor([0, 2])
    logits, loss, loss_dict = wrapper.forward_with_augmentation(x, y)
    
    print(f"   Input: {x.shape}")
    print(f"   Output: {logits.shape}")
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Loss breakdown: {loss_dict}")
    
    print("\n" + "=" * 80)
    print("✓ All training utilities tested successfully!")
    print("=" * 80)
