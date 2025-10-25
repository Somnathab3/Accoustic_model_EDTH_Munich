"""
Create complete training pipeline flowchart
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

fig, ax = plt.subplots(figsize=(16, 22), dpi=150)
ax.set_xlim(0, 10)
ax.set_ylim(0, 26)
ax.axis('off')

color_data = '#FFE5B4'
color_train = '#B4D7FF'
color_val = '#D4FFB4'
color_optimize = '#FFB4D4'
color_save = '#F0E68C'

def draw_box(ax, x, y, width, height, text, color, fontsize=9, bold=False):
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1",
        linewidth=2,
        edgecolor='black',
        facecolor=color,
        zorder=2
    )
    ax.add_patch(box)
    
    weight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', fontsize=fontsize,
            weight=weight, zorder=3)

def draw_arrow(ax, x1, y1, x2, y2, label='', style='->'):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style + ',head_width=0.4,head_length=0.8',
        linewidth=2,
        color='black',
        zorder=1
    )
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.5, mid_y, label, fontsize=8, style='italic',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white'))

def draw_decision(ax, x, y, size, text):
    """Draw a diamond for decision points"""
    diamond = mpatches.FancyBboxPatch(
        (x - size/2, y - size/2), size, size,
        boxstyle="round,pad=0.05",
        transform=ax.transData,
        linewidth=2,
        edgecolor='black',
        facecolor='#FFD700',
        zorder=2
    )
    # Rotate to make diamond
    t = ax.transData
    t2 = mpatches.transforms.Affine2D().rotate_deg_around(x, y, 45) + t
    diamond.set_transform(t2)
    ax.add_patch(diamond)
    
    ax.text(x, y, text, ha='center', va='center',
            fontsize=8, weight='bold', zorder=3)

# Title
ax.text(5, 25, 'Training Pipeline', ha='center', fontsize=18, weight='bold')
ax.text(5, 24.3, 'Complete Training Process for CRNN Model', ha='center', fontsize=12, style='italic')

y_pos = 23

# ========== DATA LOADING ==========
ax.text(5, y_pos + 0.3, 'PHASE 1: DATA PREPARATION', ha='center', fontsize=11, weight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor=color_data, edgecolor='black', linewidth=2))

y_pos -= 1.2
draw_box(ax, 1.5, y_pos, 7, 0.8,
         'Load Dataset\nTrain: 80% | Val: 10% | Test: 10%',
         color_data, fontsize=9)

draw_arrow(ax, 5, y_pos, 5, y_pos - 0.5)
y_pos -= 1.3

draw_box(ax, 1.5, y_pos, 7, 0.8,
         'Apply Preprocessing Pipeline\nAudio → 3-Channel Log-Mel Spectrogram',
         color_data, fontsize=9)

draw_arrow(ax, 5, y_pos, 5, y_pos - 0.5)
y_pos -= 1.3

draw_box(ax, 1.5, y_pos, 7, 1,
         'Data Augmentation (Training Only)\n' +
         'SpecAugment | Time-Pitch Shift\n' +
         'Background Noise | Mixup (α=0.2)',
         color_data, fontsize=9)

draw_arrow(ax, 5, y_pos, 5, y_pos - 0.5)
y_pos -= 1.8

# ========== TRAINING SETUP ==========
ax.text(5, y_pos + 0.3, 'PHASE 2: TRAINING SETUP', ha='center', fontsize=11, weight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor=color_train, edgecolor='black', linewidth=2))

y_pos -= 1.2
draw_box(ax, 1, y_pos, 8, 1.2,
         'Initialize Model: CRNNWithAttention\n' +
         'Parameters: ~1.8M | Device: CUDA/CPU\n' +
         'Loss: Cross-Entropy + Label Smoothing (0.1)\n' +
         'Optimizer: AdamW (lr=0.001, weight_decay=0.01)',
         color_train, fontsize=8)

draw_arrow(ax, 5, y_pos, 5, y_pos - 0.5)
y_pos -= 1.8

draw_box(ax, 1.5, y_pos, 7, 0.8,
         'Learning Rate Scheduler\nCosine Annealing with Warmup (5 epochs)',
         color_train, fontsize=9)

draw_arrow(ax, 5, y_pos, 5, y_pos - 0.5)
y_pos -= 1.5

# ========== TRAINING LOOP ==========
# Epoch counter
circle = Circle((5, y_pos), 0.4, color='#FFD700', ec='black', linewidth=2, zorder=2)
ax.add_patch(circle)
ax.text(5, y_pos, 'Epoch\nLoop', ha='center', va='center', fontsize=8, weight='bold', zorder=3)

draw_arrow(ax, 5, y_pos - 0.4, 5, y_pos - 1, 'Start epoch')
y_pos -= 1.5

draw_box(ax, 1.5, y_pos, 7, 0.7,
         'Load Training Batch (32 samples)',
         color_train, fontsize=9)

draw_arrow(ax, 5, y_pos, 5, y_pos - 0.3)
y_pos -= 1

draw_box(ax, 1.5, y_pos, 7, 0.7,
         'Forward Pass: logits = model(spectrograms)',
         color_train, fontsize=9)

draw_arrow(ax, 5, y_pos, 5, y_pos - 0.3)
y_pos -= 1

draw_box(ax, 1.5, y_pos, 7, 0.7,
         'Compute Loss: Cross-Entropy(logits, labels)',
         color_train, fontsize=9)

draw_arrow(ax, 5, y_pos, 5, y_pos - 0.3)
y_pos -= 1

draw_box(ax, 1.5, y_pos, 7, 0.7,
         'Backward Pass: loss.backward()',
         color_train, fontsize=9)

draw_arrow(ax, 5, y_pos, 5, y_pos - 0.3)
y_pos -= 1

draw_box(ax, 1.5, y_pos, 7, 0.7,
         'Gradient Clipping: clip_grad_norm(max=1.0)',
         color_train, fontsize=9)

draw_arrow(ax, 5, y_pos, 5, y_pos - 0.3)
y_pos -= 1

draw_box(ax, 1.5, y_pos, 7, 0.7,
         'Update Weights: optimizer.step()',
         color_optimize, fontsize=9)

draw_arrow(ax, 5, y_pos, 5, y_pos - 0.5)
y_pos -= 1.2

# Decision: More batches?
draw_decision(ax, 5, y_pos, 1, 'More\nbatches?')

# Loop back
draw_arrow(ax, 5.7, y_pos, 8.5, y_pos, 'Yes')
draw_arrow(ax, 8.5, y_pos, 8.5, y_pos + 7.5)
draw_arrow(ax, 8.5, y_pos + 7.5, 5.7, y_pos + 7.5)

draw_arrow(ax, 5, y_pos - 0.7, 5, y_pos - 1.2, 'No')
y_pos -= 1.7

# ========== VALIDATION ==========
ax.text(5, y_pos + 0.3, 'PHASE 3: VALIDATION', ha='center', fontsize=11, weight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor=color_val, edgecolor='black', linewidth=2))

y_pos -= 1.2
draw_box(ax, 1.5, y_pos, 7, 0.7,
         'Evaluate on Validation Set\nmodel.eval() | No gradients',
         color_val, fontsize=9)

draw_arrow(ax, 5, y_pos, 5, y_pos - 0.3)
y_pos -= 1

draw_box(ax, 1.5, y_pos, 7, 0.9,
         'Compute Metrics\n' +
         'Accuracy | Precision | Recall | F1-Score',
         color_val, fontsize=9)

draw_arrow(ax, 5, y_pos, 5, y_pos - 0.5)
y_pos -= 1.5

# Decision: Best model?
draw_decision(ax, 5, y_pos, 1.2, 'Best F1\nscore?')

draw_arrow(ax, 5.8, y_pos, 7.5, y_pos, 'Yes')
y_pos_save = y_pos
draw_box(ax, 7.5, y_pos - 0.4, 2, 0.8,
         'Save Model\nCheckpoint',
         color_save, fontsize=8)

draw_arrow(ax, 5, y_pos - 0.8, 5, y_pos - 1.5, 'No/Continue')
y_pos -= 2

# Decision: Early stopping?
draw_decision(ax, 5, y_pos, 1.2, 'No improve\n5 epochs?')

draw_arrow(ax, 5.8, y_pos, 7.5, y_pos, 'Yes → STOP')
draw_box(ax, 7.5, y_pos - 0.4, 2, 0.8,
         'Early\nStopping',
         '#FF6B6B', fontsize=8)

draw_arrow(ax, 5, y_pos - 0.8, 5, y_pos - 1.5, 'No')
y_pos -= 2

# Decision: Max epochs?
draw_decision(ax, 5, y_pos, 1.2, 'Epoch <\nmax?')

# Loop back to epoch
draw_arrow(ax, 4.2, y_pos, 1, y_pos, 'Yes')
draw_arrow(ax, 1, y_pos, 1, y_pos + 15)
draw_arrow(ax, 1, y_pos + 15, 4.3, y_pos + 15)

draw_arrow(ax, 5, y_pos - 0.8, 5, y_pos - 1.5, 'No → END')
y_pos -= 2

# ========== FINAL OUTPUT ==========
draw_box(ax, 2, y_pos, 6, 1.2,
         'TRAINING COMPLETE\n\n' +
         'Saved: best_model.pt | training_history.json\n' +
         'Final Metrics: Accuracy, F1, Confusion Matrix',
         color_save, fontsize=9, bold=True)

# Technical specs
y_pos -= 1.8
draw_box(ax, 0.3, y_pos, 9.4, 1.4,
         'Training Configuration:\n\n' +
         '• Batch Size: 32 | Epochs: 50 (with early stopping)\n' +
         '• Learning Rate: 0.001 → 0 (cosine decay) | Warmup: 5 epochs\n' +
         '• Optimizer: AdamW (β₁=0.9, β₂=0.999, ε=1e-8, weight_decay=0.01)\n' +
         '• Loss Function: CrossEntropy + Label Smoothing (0.1)\n' +
         '• Regularization: Dropout (0.3), BatchNorm, Gradient Clipping\n' +
         '• Early Stopping: Patience 5 epochs (monitor val_f1_score)\n' +
         '• Training Time: ~30-45 minutes on GPU | ~3-4 hours on CPU',
         '#E8F4F8', fontsize=8)

plt.tight_layout()
plt.savefig('visualizations/03_training_pipeline.jpg',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Training pipeline flowchart saved: visualizations/03_training_pipeline.jpg")
plt.close()
