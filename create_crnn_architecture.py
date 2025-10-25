"""
Create detailed CRNN architecture diagram for acoustic drone detection
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Set up the figure
fig, ax = plt.subplots(figsize=(18, 22), dpi=150)
ax.set_xlim(0, 12)
ax.set_ylim(0, 28)
ax.axis('off')

# Define colors
color_input = '#FFE5B4'
color_conv = '#87CEEB'
color_bn = '#98FB98'
color_activation = '#FFB6C1'
color_pool = '#DDA0DD'
color_attention = '#F0E68C'
color_rnn = '#FFA07A'
color_fc = '#FF6B6B'
color_output = '#90EE90'

def draw_box(ax, x, y, width, height, text, color, fontsize=9, bold=False):
    """Draw a rounded box with text"""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.08",
        linewidth=2,
        edgecolor='black',
        facecolor=color,
        zorder=2
    )
    ax.add_patch(box)
    
    weight = 'bold' if bold else 'normal'
    ax.text(
        x + width/2, y + height/2, text,
        ha='center', va='center',
        fontsize=fontsize,
        weight=weight,
        zorder=3
    )

def draw_arrow(ax, x1, y1, x2, y2, label='', style='->', width=2):
    """Draw an arrow"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style + ',head_width=0.4,head_length=0.8',
        linewidth=width,
        color='black',
        zorder=1
    )
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.5, mid_y, label, fontsize=8, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))

def draw_3d_block(ax, x, y, channels, height, width, label, color):
    """Draw a 3D representation of a tensor"""
    # Main rectangle
    rect = Rectangle((x, y), width, height, 
                     linewidth=2, edgecolor='black', 
                     facecolor=color, zorder=2)
    ax.add_patch(rect)
    
    # 3D effect
    offset = 0.15
    # Top face
    top_points = np.array([
        [x, y + height],
        [x + offset, y + height + offset],
        [x + width + offset, y + height + offset],
        [x + width, y + height]
    ])
    top = plt.Polygon(top_points, closed=True, 
                     linewidth=2, edgecolor='black',
                     facecolor=color, alpha=0.7, zorder=1)
    ax.add_patch(top)
    
    # Right face
    right_points = np.array([
        [x + width, y],
        [x + width + offset, y + offset],
        [x + width + offset, y + height + offset],
        [x + width, y + height]
    ])
    right = plt.Polygon(right_points, closed=True,
                       linewidth=2, edgecolor='black',
                       facecolor=color, alpha=0.5, zorder=1)
    ax.add_patch(right)
    
    # Label
    ax.text(x + width/2, y + height/2, label,
           ha='center', va='center', fontsize=8, weight='bold', zorder=3)

# Title
ax.text(6, 27, 'CRNN Architecture with Temporal-Frequency Attention',
        ha='center', fontsize=18, weight='bold')
ax.text(6, 26.3, 'Convolutional Recurrent Neural Network for Acoustic Drone Detection',
        ha='center', fontsize=12, style='italic')

# Model parameters box
draw_box(ax, 0.5, 24.8, 11, 1.2,
         'Model Parameters: ~1.8M | Input: (batch, 3, 96, 101) | ' +
         'Classes: 3 (drone, no-drone, background) | Training: AdamW, Cosine LR',
         '#F0F0F0', fontsize=9)

y_pos = 23.5

# ========== INPUT LAYER ==========
ax.text(6, y_pos + 0.3, 'INPUT LAYER', ha='center', fontsize=12, weight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE5B4', edgecolor='black', linewidth=2))

y_pos -= 1.2
draw_3d_block(ax, 4, y_pos, 3, 1.5, 4, 
              '3-Channel\nLog-Mel Spectrogram\n(3, 96, 101)',
              color_input)

ax.text(8.5, y_pos + 0.75, 
        'Channels:\n1. Total signal\n2. Harmonic (rotor)\n3. Percussive (motor)',
        fontsize=8, va='center')

draw_arrow(ax, 6, y_pos - 0.1, 6, y_pos - 1)

y_pos -= 1.5

# ========== CONVOLUTIONAL BLOCK 1 ==========
ax.text(1, y_pos + 1.3, 'CONV BLOCK 1', ha='center', fontsize=11, weight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#87CEEB', edgecolor='black', linewidth=2))

# Conv2d
draw_box(ax, 1, y_pos, 10, 0.7,
         'Conv2d(in=3, out=32, kernel=3×3, padding=1, stride=1)',
         color_conv, fontsize=9)

ax.text(11.3, y_pos + 0.35, 'Params: 3×32×3×3 = 864', fontsize=7, va='center')

draw_arrow(ax, 6, y_pos, 6, y_pos - 0.2)
y_pos -= 0.9

# BatchNorm
draw_box(ax, 1.5, y_pos, 9, 0.5,
         'BatchNorm2d(32) → Normalize activations',
         color_bn, fontsize=8)
ax.text(11.3, y_pos + 0.25, 'Params: 64', fontsize=7, va='center')

draw_arrow(ax, 6, y_pos, 6, y_pos - 0.2)
y_pos -= 0.7

# ReLU
draw_box(ax, 2, y_pos, 8, 0.5,
         'ReLU() → f(x) = max(0, x)',
         color_activation, fontsize=8)
ax.text(10.5, y_pos + 0.25, 'Non-linearity', fontsize=7, va='center')

draw_arrow(ax, 6, y_pos, 6, y_pos - 0.2)
y_pos -= 0.7

# MaxPool
draw_box(ax, 1.5, y_pos, 9, 0.5,
         'MaxPool2d(2×2) → Downsample by 2',
         color_pool, fontsize=8)

draw_3d_block(ax, 0.5, y_pos - 1.2, 32, 0.8, 2,
              'Output\n(32, 48, 50)',
              '#B0E0E6')

draw_arrow(ax, 6, y_pos, 6, y_pos - 1.5)
y_pos -= 2

# ========== CONVOLUTIONAL BLOCK 2 ==========
ax.text(1, y_pos + 1.3, 'CONV BLOCK 2', ha='center', fontsize=11, weight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#87CEEB', edgecolor='black', linewidth=2))

draw_box(ax, 1, y_pos, 10, 0.7,
         'Conv2d(in=32, out=64, kernel=3×3, padding=1)',
         color_conv, fontsize=9)
ax.text(11.3, y_pos + 0.35, 'Params: 18,432', fontsize=7, va='center')

draw_arrow(ax, 6, y_pos, 6, y_pos - 0.2)
y_pos -= 0.9

draw_box(ax, 1.5, y_pos, 9, 0.5,
         'BatchNorm2d(64) → ReLU() → MaxPool2d(2×2)',
         '#B0E0E6', fontsize=8)

draw_3d_block(ax, 0.5, y_pos - 1.2, 64, 0.8, 2,
              'Output\n(64, 24, 25)',
              '#B0E0E6')

draw_arrow(ax, 6, y_pos, 6, y_pos - 1.5)
y_pos -= 2

# ========== CONVOLUTIONAL BLOCK 3 ==========
ax.text(1, y_pos + 1.3, 'CONV BLOCK 3', ha='center', fontsize=11, weight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#87CEEB', edgecolor='black', linewidth=2))

draw_box(ax, 1, y_pos, 10, 0.7,
         'Conv2d(in=64, out=128, kernel=3×3, padding=1)',
         color_conv, fontsize=9)
ax.text(11.3, y_pos + 0.35, 'Params: 73,728', fontsize=7, va='center')

draw_arrow(ax, 6, y_pos, 6, y_pos - 0.2)
y_pos -= 0.9

draw_box(ax, 1.5, y_pos, 9, 0.5,
         'BatchNorm2d(128) → ReLU() → MaxPool2d(2×2)',
         '#B0E0E6', fontsize=8)

draw_3d_block(ax, 0.5, y_pos - 1.2, 128, 0.8, 2,
              'Output\n(128, 12, 12)',
              '#B0E0E6')

draw_arrow(ax, 6, y_pos, 6, y_pos - 1.5)
y_pos -= 2.2

# ========== ATTENTION MODULE ==========
ax.text(6, y_pos + 0.5, 'TEMPORAL-FREQUENCY ATTENTION', ha='center', fontsize=11, weight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0E68C', edgecolor='black', linewidth=2))

y_pos -= 1
draw_box(ax, 0.8, y_pos, 5, 1.8,
         'Temporal Attention Path\n\n' +
         '1. AdaptiveAvgPool(time→1)\n' +
         '2. FC(128→32)→ReLU\n' +
         '3. FC(32→128)→Sigmoid\n' +
         '4. Weights shape: (B,C,F,1)',
         color_attention, fontsize=8)

draw_box(ax, 6.2, y_pos, 5, 1.8,
         'Frequency Attention Path\n\n' +
         '1. AdaptiveAvgPool(freq→1)\n' +
         '2. FC(128→32)→ReLU\n' +
         '3. FC(32→128)→Sigmoid\n' +
         '4. Weights shape: (B,C,1,T)',
         color_attention, fontsize=8)

y_pos -= 2
draw_box(ax, 2, y_pos, 8, 0.7,
         'Element-wise Multiply: Output = Input × Temporal_Attn × Freq_Attn',
         '#FFD700', fontsize=9, bold=True)

ax.text(10.8, y_pos + 0.35,
        'Focuses on\nrotor harmonics',
        fontsize=7, va='center')

draw_arrow(ax, 6, y_pos, 6, y_pos - 0.5)
y_pos -= 1

# ========== RESHAPE FOR RNN ==========
draw_box(ax, 2, y_pos, 8, 0.7,
         'Reshape: (B, 128, 12, 12) → (B, 12, 128×12=1536)',
         '#E0E0E0', fontsize=9)

ax.text(10.8, y_pos + 0.35,
        'Prepare for\ntemporal modeling',
        fontsize=7, va='center')

draw_arrow(ax, 6, y_pos, 6, y_pos - 0.5)
y_pos -= 1.2

# ========== BIDIRECTIONAL GRU ==========
ax.text(6, y_pos + 0.5, 'RECURRENT LAYER', ha='center', fontsize=11, weight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFA07A', edgecolor='black', linewidth=2))

y_pos -= 1
draw_box(ax, 1.5, y_pos, 9, 1.5,
         'Bidirectional GRU\n\n' +
         'Input size: 1536 (128×12)\n' +
         'Hidden size: 128\n' +
         'Num layers: 2\n' +
         'Dropout: 0.3\n' +
         'Output: (B, time=12, 256)',
         color_rnn, fontsize=9, bold=True)

ax.text(11.3, y_pos + 0.75,
        'Forward: 128 units\n' +
        'Backward: 128 units\n' +
        'Total: 256 features\n\n' +
        'Params: ~1.5M\n' +
        '(largest component)',
        fontsize=7, va='center')

draw_arrow(ax, 6, y_pos, 6, y_pos - 0.5)
y_pos -= 2

# ========== TEMPORAL POOLING ==========
draw_box(ax, 2.5, y_pos, 7, 0.7,
         'Mean Pooling over time: (B, 12, 256) → (B, 256)',
         '#FFC0CB', fontsize=9)

draw_arrow(ax, 6, y_pos, 6, y_pos - 0.5)
y_pos -= 1.2

# ========== CLASSIFICATION HEAD ==========
ax.text(6, y_pos + 0.5, 'CLASSIFICATION HEAD', ha='center', fontsize=11, weight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FF6B6B', edgecolor='black', linewidth=2))

y_pos -= 0.8
draw_box(ax, 3, y_pos, 6, 0.6,
         'Dropout(p=0.3)',
         color_fc, fontsize=9)

draw_arrow(ax, 6, y_pos, 6, y_pos - 0.2)
y_pos -= 0.8

draw_box(ax, 2.5, y_pos, 7, 0.7,
         'Fully Connected: (256 → 3)',
         color_fc, fontsize=9, bold=True)

ax.text(10, y_pos + 0.35, 'Params: 771', fontsize=7, va='center')

draw_arrow(ax, 6, y_pos, 6, y_pos - 0.5)
y_pos -= 1

# ========== OUTPUT ==========
draw_box(ax, 3, y_pos, 6, 1,
         'OUTPUT LOGITS\n(batch, 3)\nClass scores before softmax',
         color_output, fontsize=10, bold=True)

draw_arrow(ax, 6, y_pos, 6, y_pos - 0.5)
y_pos -= 1.5

# ========== INFERENCE ==========
draw_box(ax, 1.5, y_pos, 9, 1.2,
         'INFERENCE: Softmax(logits) → Probabilities\n' +
         'ArgMax → Predicted Class\n' +
         'Output: {"class": "drone", "confidence": 0.89, "probabilities": [...]}',
         '#90EE90', fontsize=9, bold=True)

# Summary box
y_pos -= 2
draw_box(ax, 0.3, y_pos, 11.4, 1.5,
         'Architecture Summary:\n\n' +
         '• Total Parameters: ~1,847,427 (~1.8M) | Trainable: 100%\n' +
         '• Conv Layers: 3 blocks (32→64→128 filters) with BatchNorm + ReLU + MaxPool\n' +
         '• Attention: Temporal-Frequency attention module (learns important time-freq regions)\n' +
         '• RNN: 2-layer Bidirectional GRU with 128 hidden units (captures temporal patterns)\n' +
         '• Activation Functions: ReLU (conv), Tanh (GRU), Sigmoid (attention), Softmax (output)\n' +
         '• Regularization: Dropout (0.3), BatchNorm, Gradient Clipping (max_norm=1.0)\n' +
         '• Inference Time: ~15-20ms on GPU | ~50-80ms on CPU',
         '#E8F4F8', fontsize=8)

plt.tight_layout()
plt.savefig('visualizations/02_crnn_architecture.jpg',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ CRNN architecture diagram saved: visualizations/02_crnn_architecture.jpg")
plt.close()
