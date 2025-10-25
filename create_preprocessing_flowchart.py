"""
Create detailed preprocessing flowchart for acoustic drone detection system
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure with high DPI for presentation quality
fig, ax = plt.subplots(figsize=(16, 20), dpi=150)
ax.set_xlim(0, 10)
ax.set_ylim(0, 25)
ax.axis('off')

# Define colors
color_input = '#FFE5B4'      # Peach - Input
color_process = '#B4D7FF'    # Light Blue - Processing
color_feature = '#D4FFB4'    # Light Green - Features
color_output = '#FFB4D4'     # Pink - Output
color_arrow = '#333333'      # Dark gray

def draw_box(ax, x, y, width, height, text, color, fontsize=10, bold=False):
    """Draw a rounded box with text"""
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
    ax.text(
        x + width/2, y + height/2, text,
        ha='center', va='center',
        fontsize=fontsize,
        weight=weight,
        wrap=True,
        zorder=3
    )

def draw_arrow(ax, x1, y1, x2, y2, label=''):
    """Draw an arrow between two points"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.4,head_length=0.8',
        linewidth=2,
        color=color_arrow,
        zorder=1
    )
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=8, style='italic')

# Title
ax.text(5, 24, 'Audio Preprocessing Pipeline', 
        ha='center', va='center', fontsize=20, weight='bold')
ax.text(5, 23.3, 'Acoustic Drone Detection System', 
        ha='center', va='center', fontsize=14, style='italic')

# Step 1: Input Audio
y_pos = 21.5
draw_box(ax, 2, y_pos, 6, 1, 
         'INPUT: Raw Audio File\n.wav, .mp3, any format\nVariable length, any sample rate',
         color_input, fontsize=11, bold=True)

# Arrow to loading
draw_arrow(ax, 5, y_pos, 5, y_pos - 1)

# Step 2: Audio Loading
y_pos -= 1.5
draw_box(ax, 1.5, y_pos, 7, 0.8,
         'Step 1: Load Audio with Librosa\nlibrosa.load(path, sr=16000, mono=True)',
         color_process, fontsize=10)

# Technical details box
y_pos -= 0.9
draw_box(ax, 0.5, y_pos, 9, 0.6,
         'Output: Mono waveform @ 16kHz | Shape: (1, samples) | Duration: Variable',
         '#F0F0F0', fontsize=9)

# Arrow to resampling
draw_arrow(ax, 5, y_pos, 5, y_pos - 1)

# Step 3: Resampling and Normalization
y_pos -= 1.5
draw_box(ax, 1.5, y_pos, 7, 0.8,
         'Step 2: Resample to 16kHz & Normalize\nlibrosa.util.normalize(waveform)',
         color_process, fontsize=10)

y_pos -= 0.9
draw_box(ax, 0.5, y_pos, 9, 0.6,
         'Sample Rate: 44.1kHz → 16kHz | Normalize: [-1, 1] range | Amplitude normalization',
         '#F0F0F0', fontsize=9)

# Arrow to padding
draw_arrow(ax, 5, y_pos, 5, y_pos - 1)

# Step 4: Padding/Trimming
y_pos -= 1.5
draw_box(ax, 1.5, y_pos, 7, 0.8,
         'Step 3: Pad or Trim to Fixed Length\nTarget: 2.0 seconds (32,000 samples)',
         color_process, fontsize=10)

y_pos -= 0.9
draw_box(ax, 0.5, y_pos, 9, 0.6,
         'If shorter: Zero-pad | If longer: Random crop | Output: (1, 32000)',
         '#F0F0F0', fontsize=9)

# Arrow splits into HPSS
draw_arrow(ax, 5, y_pos, 5, y_pos - 1)

# Step 5: HPSS (Harmonic-Percussive Source Separation)
y_pos -= 1.5
draw_box(ax, 1.5, y_pos, 7, 1.2,
         'Step 4: HPSS - Harmonic-Percussive Separation\n' +
         'librosa.effects.hpss(waveform, margin=2.0)\n' +
         'Separates rotor harmonics from motor noise',
         color_process, fontsize=10, bold=True)

# HPSS splits into three paths
draw_arrow(ax, 3.5, y_pos - 0.1, 2, y_pos - 2.5, 'Total')
draw_arrow(ax, 5, y_pos - 0.1, 5, y_pos - 2.5, 'Harmonic')
draw_arrow(ax, 6.5, y_pos - 0.1, 8, y_pos - 2.5, 'Percussive')

# Three parallel paths
y_pos -= 3.5

# Path 1: Total signal
draw_box(ax, 0.5, y_pos, 3, 0.8,
         'Total Signal\nOriginal waveform\n(32000 samples)',
         color_feature, fontsize=9)

# Path 2: Harmonic component
draw_box(ax, 3.5, y_pos, 3, 0.8,
         'Harmonic Component\nRotor blade harmonics\n(32000 samples)',
         color_feature, fontsize=9)

# Path 3: Percussive component
draw_box(ax, 6.5, y_pos, 3, 0.8,
         'Percussive Component\nMotor noise, transients\n(32000 samples)',
         color_feature, fontsize=9)

# Arrows to mel spectrogram
draw_arrow(ax, 2, y_pos - 0.1, 2, y_pos - 1.5)
draw_arrow(ax, 5, y_pos - 0.1, 5, y_pos - 1.5)
draw_arrow(ax, 8, y_pos - 0.1, 8, y_pos - 1.5)

# Step 6: Mel Spectrogram Computation
y_pos -= 2.5

# Left path
draw_box(ax, 0.5, y_pos, 3, 1.2,
         'Mel Spectrogram\nn_fft=1024\nhop=320\nn_mels=96',
         color_process, fontsize=9)

# Middle path
draw_box(ax, 3.5, y_pos, 3, 1.2,
         'Mel Spectrogram\n(Harmonic)\nSame parameters',
         color_process, fontsize=9)

# Right path
draw_box(ax, 6.5, y_pos, 3, 1.2,
         'Mel Spectrogram\n(Percussive)\nSame parameters',
         color_process, fontsize=9)

# Technical details
y_pos -= 1.3
draw_box(ax, 0.5, y_pos, 9, 0.7,
         'STFT Parameters: n_fft=1024 (64ms), hop=320 (20ms), window=Hann | ' +
         'Frequency range: 50Hz - 8000Hz',
         '#F0F0F0', fontsize=8)

# Arrows to log conversion
draw_arrow(ax, 2, y_pos, 2, y_pos - 1.5)
draw_arrow(ax, 5, y_pos, 5, y_pos - 1.5)
draw_arrow(ax, 8, y_pos, 8, y_pos - 1.5)

# Step 7: Log-Mel Conversion
y_pos -= 2.5

draw_box(ax, 0.5, y_pos, 3, 1,
         'Log-Mel (dB)\n10*log10(S+ε)\nTop dB: 80',
         color_process, fontsize=9)

draw_box(ax, 3.5, y_pos, 3, 1,
         'Log-Mel (dB)\n(Harmonic)\nTop dB: 80',
         color_process, fontsize=9)

draw_box(ax, 6.5, y_pos, 3, 1,
         'Log-Mel (dB)\n(Percussive)\nTop dB: 80',
         color_process, fontsize=9)

# Output shapes
y_pos -= 1.1
draw_box(ax, 0.5, y_pos, 3, 0.6,
         'Shape: (1, 96, 101)',
         color_feature, fontsize=8)

draw_box(ax, 3.5, y_pos, 3, 0.6,
         'Shape: (1, 96, 101)',
         color_feature, fontsize=8)

draw_box(ax, 6.5, y_pos, 3, 0.6,
         'Shape: (1, 96, 101)',
         color_feature, fontsize=8)

# Arrows converge
draw_arrow(ax, 2, y_pos - 0.1, 3.5, y_pos - 1.5)
draw_arrow(ax, 5, y_pos - 0.1, 5, y_pos - 1.5)
draw_arrow(ax, 8, y_pos - 0.1, 6.5, y_pos - 1.5)

# Step 8: Stack as 3-channel tensor
y_pos -= 2.5
draw_box(ax, 2, y_pos, 6, 1.2,
         'Step 5: Stack as 3-Channel Tensor\n' +
         'torch.cat([total, harmonic, percussive], dim=0)\n' +
         'Similar to RGB image channels',
         color_process, fontsize=10, bold=True)

# Final output
y_pos -= 1.5
draw_arrow(ax, 5, y_pos + 1.2, 5, y_pos + 0.8)

draw_box(ax, 2, y_pos, 6, 1.2,
         'OUTPUT: 3-Channel Log-Mel Spectrogram\n' +
         'Shape: (3, 96, 101)\n' +
         '3 channels × 96 mel bins × 101 time frames',
         color_output, fontsize=11, bold=True)

# Technical specifications box at bottom
y_pos -= 1.8
draw_box(ax, 0.3, y_pos, 9.4, 1.5,
         'Technical Specifications:\n\n' +
         '• Sample Rate: 16,000 Hz (professional quality)\n' +
         '• Window Size: 2.0 seconds (32,000 samples)\n' +
         '• FFT Size: 1024 points (64ms window)\n' +
         '• Hop Length: 320 samples (20ms, 50 fps temporal resolution)\n' +
         '• Mel Bins: 96 (frequency resolution)\n' +
         '• Frequency Range: 50 Hz - 8,000 Hz (captures drone rotor harmonics)\n' +
         '• Time Frames: 101 frames (2 second window with 20ms hops)\n' +
         '• Data Type: 32-bit float, log-scale dB normalized',
         '#E8F4F8', fontsize=8)

plt.tight_layout()
plt.savefig('visualizations/01_preprocessing_flowchart.jpg', 
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Preprocessing flowchart saved: visualizations/01_preprocessing_flowchart.jpg")
plt.close()
