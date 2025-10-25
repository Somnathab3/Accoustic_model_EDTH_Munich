"""
Create complete end-to-end system flowchart
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

fig, ax = plt.subplots(figsize=(18, 24), dpi=150)
ax.set_xlim(0, 14)
ax.set_ylim(0, 30)
ax.axis('off')

color_input = '#FFE5B4'
color_preprocess = '#B4D7FF'
color_model = '#D4FFB4'
color_output = '#FFB4D4'
color_system = '#F0E68C'

def draw_box(ax, x, y, width, height, text, color, fontsize=9, bold=False):
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1",
        linewidth=2.5,
        edgecolor='black',
        facecolor=color,
        zorder=2
    )
    ax.add_patch(box)
    
    weight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', fontsize=fontsize,
            weight=weight, zorder=3)

def draw_arrow(ax, x1, y1, x2, y2, label='', width=2.5):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.5,head_length=1',
        linewidth=width,
        color='black',
        zorder=1
    )
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.5, mid_y, label, fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black'))

# Main title
ax.text(7, 29, 'Complete Acoustic Drone Detection System',
        ha='center', fontsize=22, weight='bold')
ax.text(7, 28.2, 'End-to-End Pipeline: From Raw Audio to Classification',
        ha='center', fontsize=14, style='italic')

y_pos = 26.5

# ========== SYSTEM INPUT ==========
draw_box(ax, 4, y_pos, 6, 1.5,
         '🎵 SYSTEM INPUT\n\n' +
         'Audio File: .wav, .mp3, .flac, etc.\n' +
         'Any duration | Any sample rate',
         color_input, fontsize=11, bold=True)

draw_arrow(ax, 7, y_pos, 7, y_pos - 1.5, 'Raw Audio')

y_pos -= 2.5

# ========== PREPROCESSING MODULE ==========
ax.text(7, y_pos + 1.5, '📊 PREPROCESSING MODULE', ha='center', fontsize=13, weight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=color_preprocess, 
                 edgecolor='black', linewidth=2.5))

y_pos -= 0.5

# Step 1: Load
draw_box(ax, 2, y_pos, 10, 0.8,
         'Step 1: Load & Resample → 16kHz mono',
         color_preprocess, fontsize=10)
ax.text(12.5, y_pos + 0.4, '32000 samples\n(2 sec)', fontsize=8, va='center')

draw_arrow(ax, 7, y_pos, 7, y_pos - 0.5)
y_pos -= 1.3

# Step 2: Normalize
draw_box(ax, 2, y_pos, 10, 0.8,
         'Step 2: Normalize & Pad/Trim to 2 seconds',
         color_preprocess, fontsize=10)

draw_arrow(ax, 7, y_pos, 7, y_pos - 0.5)
y_pos -= 1.3

# Step 3: HPSS - splits into 3 paths
draw_box(ax, 2, y_pos, 10, 1,
         'Step 3: HPSS - Harmonic-Percussive Separation\n' +
         'Splits audio into 3 components',
         color_preprocess, fontsize=10, bold=True)

# Three branches
draw_arrow(ax, 5, y_pos - 0.2, 3, y_pos - 2, 'Total')
draw_arrow(ax, 7, y_pos - 0.2, 7, y_pos - 2, 'Harmonic')
draw_arrow(ax, 9, y_pos - 0.2, 11, y_pos - 2, 'Percussive')

y_pos -= 2.5

# Three parallel processing
draw_box(ax, 1, y_pos, 3.5, 1,
         'Channel 1:\nTotal Signal\n\nMel Spec',
         '#E8F4FF', fontsize=9)

draw_box(ax, 5.25, y_pos, 3.5, 1,
         'Channel 2:\nHarmonic\n(Rotor)\nMel Spec',
         '#E8F4FF', fontsize=9)

draw_box(ax, 9.5, y_pos, 3.5, 1,
         'Channel 3:\nPercussive\n(Motor)\nMel Spec',
         '#E8F4FF', fontsize=9)

# Converge
draw_arrow(ax, 2.75, y_pos - 0.2, 5, y_pos - 1.5)
draw_arrow(ax, 7, y_pos - 0.2, 7, y_pos - 1.5)
draw_arrow(ax, 11.25, y_pos - 0.2, 9, y_pos - 1.5)

y_pos -= 2

# Step 4: Combine
draw_box(ax, 2, y_pos, 10, 1,
         'Step 4: Stack as 3-Channel Tensor\n\n' +
         'Shape: (3, 96, 101)\n' +
         '3 channels × 96 mel bins × 101 time frames',
         color_preprocess, fontsize=10, bold=True)

draw_arrow(ax, 7, y_pos, 7, y_pos - 1.5, 'Preprocessed Data')

y_pos -= 2.5

# ========== CRNN MODEL ==========
ax.text(7, y_pos + 1.5, '🧠 CRNN MODEL', ha='center', fontsize=13, weight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=color_model,
                 edgecolor='black', linewidth=2.5))

y_pos -= 0.5

# Input layer
draw_box(ax, 3, y_pos, 8, 0.7,
         'Input: 3-Channel Log-Mel Spectrogram (3, 96, 101)',
         '#FAFAFA', fontsize=9)

draw_arrow(ax, 7, y_pos, 7, y_pos - 0.3)
y_pos -= 1

# Conv blocks
draw_box(ax, 2.5, y_pos, 9, 1.2,
         'Convolutional Blocks (3 layers)\n\n' +
         'Conv1: 3→32 | Conv2: 32→64 | Conv3: 64→128\n' +
         'Each: Conv2D → BatchNorm → ReLU → MaxPool',
         color_model, fontsize=9)

ax.text(12, y_pos + 0.6, 'Extract\nSpectral\nFeatures', fontsize=8, va='center')

draw_arrow(ax, 7, y_pos, 7, y_pos - 0.5)
y_pos -= 1.7

# Attention
draw_box(ax, 2.5, y_pos, 9, 1,
         'Temporal-Frequency Attention\n\n' +
         'Focuses on discriminative time-frequency regions\n' +
         '(e.g., rotor blade harmonics)',
         '#FFFACD', fontsize=9)

ax.text(12, y_pos + 0.5, 'Learn\nImportant\nRegions', fontsize=8, va='center')

draw_arrow(ax, 7, y_pos, 7, y_pos - 0.5)
y_pos -= 1.5

# GRU
draw_box(ax, 2.5, y_pos, 9, 1.2,
         'Bidirectional GRU (2 layers)\n\n' +
         'Hidden size: 128 | Output: 256 features\n' +
         'Captures temporal dependencies',
         color_model, fontsize=9)

ax.text(12, y_pos + 0.6, 'Temporal\nModeling', fontsize=8, va='center')

draw_arrow(ax, 7, y_pos, 7, y_pos - 0.5)
y_pos -= 1.7

# Classification
draw_box(ax, 3, y_pos, 8, 1,
         'Classification Head\n\n' +
         'Dropout(0.3) → FC(256 → 3)\n' +
         'Output: Logits for 3 classes',
         color_model, fontsize=9)

draw_arrow(ax, 7, y_pos, 7, y_pos - 1.5, 'Raw Logits')

y_pos -= 2.5

# ========== POST-PROCESSING ==========
ax.text(7, y_pos + 1.2, '📈 POST-PROCESSING', ha='center', fontsize=13, weight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=color_output,
                 edgecolor='black', linewidth=2.5))

y_pos -= 0.5

# Softmax
draw_box(ax, 3, y_pos, 8, 0.8,
         'Softmax: Convert logits → Probabilities\nSum to 1.0',
         color_output, fontsize=10)

draw_arrow(ax, 7, y_pos, 7, y_pos - 0.5)
y_pos -= 1.3

# ArgMax
draw_box(ax, 3, y_pos, 8, 0.8,
         'ArgMax: Select class with highest probability',
         color_output, fontsize=10)

draw_arrow(ax, 7, y_pos, 7, y_pos - 1.5, 'Prediction')

y_pos -= 2.5

# ========== FINAL OUTPUT ==========
draw_box(ax, 3, y_pos, 8, 1.8,
         '✅ SYSTEM OUTPUT\n\n' +
         '{\n' +
         '  "predicted_class": "drone",\n' +
         '  "confidence": 0.8934,\n' +
         '  "probabilities": {\n' +
         '    "drone": 0.8934,\n' +
         '    "no-drone": 0.0821,\n' +
         '    "background": 0.0245\n' +
         '  },\n' +
         '  "inference_time_ms": 18.3\n' +
         '}',
         color_output, fontsize=8, bold=True)

# Side panel - Performance metrics
draw_box(ax, 0.3, 15, 3, 10,
         'PERFORMANCE\nMETRICS\n\n' +
         '━━━━━━━━━━━\n\n' +
         'Accuracy:\n89.2%\n\n' +
         'Precision:\n87.5%\n\n' +
         'Recall:\n88.9%\n\n' +
         'F1-Score:\n88.2%\n\n' +
         '━━━━━━━━━━━\n\n' +
         'Inference:\n15-20ms GPU\n50-80ms CPU\n\n' +
         '━━━━━━━━━━━\n\n' +
         'Model Size:\n1.8M params\n~7 MB',
         '#E8F4F8', fontsize=8, bold=True)

# Side panel - Technical specs
draw_box(ax, 10.7, 15, 3, 10,
         'TECHNICAL\nSPECS\n\n' +
         '━━━━━━━━━━━\n\n' +
         'Framework:\nPyTorch 2.0+\n\n' +
         'Audio:\nLibrosa\n16kHz\n\n' +
         'Features:\n3-ch Log-Mel\n96 mel bins\n\n' +
         'Model:\nCRNN+Attn\n1.8M params\n\n' +
         '━━━━━━━━━━━\n\n' +
         'Hardware:\nGPU/CPU\nCUDA opt.\n\n' +
         'Classes:\n3-way\nclassification',
         '#FFF8E8', fontsize=8, bold=True)

# Bottom specifications
y_pos -= 2.5
draw_box(ax, 0.3, y_pos, 13.4, 2,
         'SYSTEM SPECIFICATIONS & CAPABILITIES\n\n' +
         '• Real-time Processing: Capable of processing 2-second audio clips in <100ms\n' +
         '• Robust to Noise: Trained with SNR curriculum (0-20 dB) and SpecAugment\n' +
         '• Multi-format Support: Accepts .wav, .mp3, .flac, .ogg via librosa\n' +
         '• Platform Independent: Runs on Windows, Linux, macOS (CPU or GPU)\n' +
         '• Frequency Range: Optimized for 50Hz - 8kHz (captures drone rotor harmonics)\n' +
         '• Classes: Drone (positive), No-Drone (negative), Background (ambient noise)\n' +
         '• Training: 50 epochs with early stopping, AdamW optimizer, cosine LR schedule\n' +
         '• Augmentation: SpecAugment, time-pitch shift, background noise, mixup\n' +
         '• Architecture: CNN feature extraction → Attention → BiGRU → Classification\n' +
         '• Interpretable: Attention mechanism highlights important time-frequency regions',
         '#F5F5F5', fontsize=8)

plt.tight_layout()
plt.savefig('visualizations/04_complete_system_flowchart.jpg',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Complete system flowchart saved: visualizations/04_complete_system_flowchart.jpg")
plt.close()
