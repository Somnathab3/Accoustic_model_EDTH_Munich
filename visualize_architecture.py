""""""

SOTA Architecture VisualizerSOTA Architecture Visualizer

Creates detailed diagrams and architecture insights for CRNN, PANNs, and Transformer modelsCreates detailed diagrams and architecture insights for CRNN, PANNs, and Transformer models

""""""

import sysimport sys

import osimport os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))



import torchimport torch

import numpy as npimport numpy as np

import matplotlib.pyplot as pltimport matplotlib.pyplot as plt

import matplotlib.patches as mpatchesimport matplotlib.patches as mpatches

from pathlib import Pathfrom pathlib import Path

import jsonimport json

import argparseimport argparse



# Try to import graphviz for model visualization# Try to import graphviz for model visualization

try:try:

    import torchviz    import torchviz

    HAS_TORCHVIZ = True    HAS_TORCHVIZ = True

except:except:

    HAS_TORCHVIZ = False    HAS_TORCHVIZ = False



from adrone.models.acoustic_models import create_model, CRNNWithAttention, PANNsCNN14, AudioTransformerfrom adrone.models.acoustic_models import create_model, CRNNWithAttention, PANNsCNN14, AudioTransformer

from adrone.preprocessing import AudioPreprocessorfrom adrone.preprocessing import AudioPreprocessor





def visualize_crnn_architecture():def visualize_architecture_flow():

    """Create detailed flow diagram for CRNN architecture"""    """Create a detailed flow diagram of the architecture"""

    fig, ax = plt.subplots(figsize=(14, 16))    fig, ax = plt.subplots(figsize=(14, 16))

    ax.set_xlim(0, 10)    ax.set_xlim(0, 10)

    ax.set_ylim(0, 20)    ax.set_ylim(0, 20)

    ax.axis('off')    ax.axis('off')

        

    # Color scheme    # Color scheme

    color_input = '#E8F4F8'    color_input = '#E8F4F8'

    color_conv = '#81C784'    color_fft = '#B3E5FC'

    color_attn = '#9575CD'    color_cnn = '#81C784'

    color_rnn = '#FFB74D'    color_dnn = '#FFB74D'

    color_output = '#EF5350'    color_output = '#EF5350'

        

    # Title    # Title

    ax.text(5, 19.5, 'CRNN with Temporal-Frequency Attention',     ax.text(5, 19.5, 'FFT + CNN + DNN Architecture', 

            ha='center', va='top', fontsize=18, fontweight='bold')            ha='center', va='top', fontsize=18, fontweight='bold')

        

    # Stage 1: Input    # Stage 1: Input

    y_pos = 18    y_pos = 18

    rect = mpatches.FancyBboxPatch((1, y_pos-0.8), 8, 0.6,     rect = mpatches.FancyBboxPatch((1, y_pos-0.8), 8, 0.6, 

                                    boxstyle="round,pad=0.1",                                     boxstyle="round,pad=0.1", 

                                    edgecolor='black', facecolor=color_input, linewidth=2)                                    edgecolor='black', facecolor=color_input, linewidth=2)

    ax.add_patch(rect)    ax.add_patch(rect)

    ax.text(5, y_pos-0.5, 'Input: Mel Spectrogram (HPSS)',     ax.text(5, y_pos-0.5, 'Input: Raw Audio WAV File', 

            ha='center', va='center', fontsize=11, fontweight='bold')            ha='center', va='center', fontsize=11, fontweight='bold')

    ax.text(5, y_pos-1.1, 'Shape: (3, 96, 100) - [Total, Harmonic, Percussive]',     ax.text(5, y_pos-1.1, 'Shape: (32,000 samples) @ 16kHz, 2.0 seconds', 

            ha='center', va='top', fontsize=9, style='italic')            ha='center', va='top', fontsize=9, style='italic')

        

    # Arrow    # Arrow

    ax.annotate('', xy=(5, y_pos-1.3), xytext=(5, y_pos-1.8),    ax.annotate('', xy=(5, y_pos-1.3), xytext=(5, y_pos-1.8),

                arrowprops=dict(arrowstyle='->', lw=2, color='black'))                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

        

    # Stage 2: Convolutional Blocks    # Stage 2: FFT Preprocessing (creates mel spectrogram)

    y_pos = 16    y_pos = 16.5

    rect = mpatches.FancyBboxPatch((0.5, y_pos-2.5), 9, 2.3,     rect = mpatches.FancyBboxPatch((0.5, y_pos-1.8), 9, 1.5, 

                                    boxstyle="round,pad=0.15",                                     boxstyle="round,pad=0.15", 

                                    edgecolor='black', facecolor=color_conv, linewidth=2)                                    edgecolor='black', facecolor=color_fft, linewidth=2)

    ax.add_patch(rect)    ax.add_patch(rect)

    ax.text(5, y_pos-0.3, 'Stage 1: Convolutional Feature Extraction',     ax.text(5, y_pos-0.3, 'Stage 1: FFT Preprocessing (Shared)', 

            ha='center', va='top', fontsize=12, fontweight='bold')            ha='center', va='top', fontsize=12, fontweight='bold')

        

    conv_layers = [    # FFT preprocessing

        'Conv2d(3→32) + BatchNorm + ReLU + MaxPool(2) → (32, 48, 50)',    fft_features = [

        'Conv2d(32→64) + BatchNorm + ReLU + MaxPool(2) → (64, 24, 25)',        'STFT (Short-Time Fourier Transform)',

        'Conv2d(64→128) + BatchNorm + ReLU + MaxPool(2) → (128, 12, 12)',        '→ Mel Spectrogram (128 mel bands)',

    ]        'Output: (1, 128, 63) tensor'

    for i, layer in enumerate(conv_layers):    ]

        ax.text(5, y_pos-0.8-i*0.5, layer,     for i, feature in enumerate(fft_features):

                ha='center', va='top', fontsize=9)        ax.text(5, y_pos-0.7-i*0.3, feature, 

                    ha='center', va='top', fontsize=9)

    # Arrow    

    ax.annotate('', xy=(5, y_pos-2.7), xytext=(5, y_pos-3.2),    # Split arrow - goes to both FFT and CNN paths

                arrowprops=dict(arrowstyle='->', lw=2, color='black'))    ax.annotate('', xy=(2.5, y_pos-2.2), xytext=(4.5, y_pos-2.0),

                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Stage 3: Attention Mechanism    ax.annotate('', xy=(7.5, y_pos-2.2), xytext=(5.5, y_pos-2.0),

    y_pos = 12.5                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    rect = mpatches.FancyBboxPatch((0.5, y_pos-2.0), 9, 1.8,     ax.text(5, y_pos-2.4, 'PARALLEL PROCESSING', 

                                    boxstyle="round,pad=0.15",             ha='center', va='top', fontsize=9, fontweight='bold', style='italic')

                                    edgecolor='purple', facecolor=color_attn, linewidth=2)    

    ax.add_patch(rect)    # Stage 3a: FFT Statistical Features (LEFT PATH)

    ax.text(5, y_pos-0.3, 'Stage 2: Temporal-Frequency Attention',     y_pos = 13.5

            ha='center', va='top', fontsize=12, fontweight='bold')    rect = mpatches.FancyBboxPatch((0.2, y_pos-2.5), 4, 2.3, 

                                        boxstyle="round,pad=0.1", 

    attn_features = [                                    edgecolor='blue', facecolor='#E3F2FD', linewidth=2)

        '• Temporal Attention: Focus on key time frames (rotor events)',    ax.add_patch(rect)

        '• Frequency Attention: Focus on harmonic patterns',    ax.text(2.2, y_pos-0.3, 'Path A: FFT Features', 

        '• Multiplicative gating: Enhanced discriminative features',            ha='center', va='top', fontsize=11, fontweight='bold')

        'Output: (128, 12, 12) - attention-weighted'    

    ]    fft_path = [

    for i, feature in enumerate(attn_features):        'Statistical Analysis:',

        ax.text(5, y_pos-0.7-i*0.35, feature,         '• Mean, Std, Min, Max',

                ha='center', va='top', fontsize=9)        '• Spectral statistics',

            '→ 50 features',

    # Arrow        '→ FC(50→128→256)',

    ax.annotate('', xy=(5, y_pos-2.2), xytext=(5, y_pos-2.7),        'Output: (256,)'

                arrowprops=dict(arrowstyle='->', lw=2, color='black'))    ]

        for i, item in enumerate(fft_path):

    # Stage 4: Recurrent Processing        ax.text(2.2, y_pos-0.7-i*0.3, item, 

    y_pos = 9.5                ha='center', va='top', fontsize=8)

    rect = mpatches.FancyBboxPatch((0.5, y_pos-2.3), 9, 2.1,     

                                    boxstyle="round,pad=0.15",     # Arrow down from FFT path

                                    edgecolor='black', facecolor=color_rnn, linewidth=2)    ax.annotate('', xy=(2.2, y_pos-2.7), xytext=(2.2, y_pos-3.2),

    ax.add_patch(rect)                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))

    ax.text(5, y_pos-0.3, 'Stage 3: Bidirectional GRU (Temporal Modeling)',     

            ha='center', va='top', fontsize=12, fontweight='bold')    # Stage 3b: CNN Feature Learning (RIGHT PATH)

        rect = mpatches.FancyBboxPatch((5.8, y_pos-2.5), 4, 2.3, 

    rnn_features = [                                    boxstyle="round,pad=0.1", 

        'Reshape: (128, 12, 12) → (12, 128×12=1536) [time, features]',                                    edgecolor='green', facecolor='#E8F5E9', linewidth=2)

        'BiGRU(1536→128, 2 layers, dropout=0.3)',    ax.add_patch(rect)

        '→ Forward & Backward temporal context',    ax.text(7.8, y_pos-0.3, 'Path B: CNN Features', 

        'Output: (12, 256) - bidirectional features',            ha='center', va='top', fontsize=11, fontweight='bold')

        'Temporal Pooling: Mean over time → (256,)'    

    ]    cnn_path = [

    for i, feature in enumerate(rnn_features):        'Deep Learning:',

        ax.text(5, y_pos-0.7-i*0.35, feature,         '• Conv + ResBlocks',

                ha='center', va='top', fontsize=9)        '• Attention mechanisms',

            '→ Spatial patterns',

    # Arrow        '→ FC(256→512)',

    ax.annotate('', xy=(5, y_pos-2.5), xytext=(5, y_pos-3.0),        'Output: (512,)'

                arrowprops=dict(arrowstyle='->', lw=2, color='black'))    ]

        for i, item in enumerate(cnn_path):

    # Stage 5: Classification        ax.text(7.8, y_pos-0.7-i*0.3, item, 

    y_pos = 6                ha='center', va='top', fontsize=8)

    rect = mpatches.FancyBboxPatch((1, y_pos-1.5), 8, 1.3,     

                                    boxstyle="round,pad=0.1",     # Arrow down from CNN path

                                    edgecolor='black', facecolor=color_output, linewidth=2)    ax.annotate('', xy=(7.8, y_pos-2.7), xytext=(7.8, y_pos-3.2),

    ax.add_patch(rect)                arrowprops=dict(arrowstyle='->', lw=2, color='green'))

    ax.text(5, y_pos-0.3, 'Stage 4: Classification Head',     

            ha='center', va='top', fontsize=11, fontweight='bold')    # Merge arrows

    ax.text(5, y_pos-0.7, 'Dropout(0.3) + Linear(256→3)',     ax.annotate('', xy=(5, y_pos-3.7), xytext=(2.2, y_pos-3.5),

            ha='center', va='top', fontsize=9)                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))

    ax.text(5, y_pos-1.0, 'Output: Class logits (3,)',     ax.annotate('', xy=(5, y_pos-3.7), xytext=(7.8, y_pos-3.5),

            ha='center', va='top', fontsize=9, fontweight='bold')                arrowprops=dict(arrowstyle='->', lw=2, color='green'))

        

    # Key insights box    # Fusion point

    y_pos = 4    y_pos = 9

    rect = mpatches.FancyBboxPatch((0.2, y_pos-1.5), 9.6, 1.4,     rect = mpatches.FancyBboxPatch((3.5, y_pos-0.8), 3, 0.6, 

                                    boxstyle="round,pad=0.05",                                     boxstyle="round,pad=0.1", 

                                    edgecolor='blue', facecolor='lightyellow',                                     edgecolor='purple', facecolor='#F3E5F5', linewidth=2)

                                    linewidth=1, linestyle='--')    ax.add_patch(rect)

    ax.add_patch(rect)    ax.text(5, y_pos-0.5, 'FUSION: Concatenate', 

    ax.text(5, y_pos-0.2, '🎯 Key Advantages',             ha='center', va='center', fontsize=10, fontweight='bold')

            ha='center', va='top', fontsize=11, fontweight='bold')    ax.text(5, y_pos-1.1, '[FFT Features | CNN Features] = (768,)', 

    advantages = [            ha='center', va='top', fontsize=8, family='monospace')

        '✓ Lightweight: ~1-2M parameters, fast inference (~1-2ms)',    

        '✓ Attention focuses on rotor harmonics and temporal events',    # Arrow down

        '✓ BiGRU captures temporal context (before/after patterns)',    ax.annotate('', xy=(5, y_pos-1.3), xytext=(5, y_pos-1.8),

        '✓ Ideal for edge deployment and real-time detection'                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ]    

    for i, adv in enumerate(advantages):    # Stage 4: DNN Classification

        ax.text(5, y_pos-0.5-i*0.25, adv,     y_pos = 7

                ha='center', va='top', fontsize=8)    rect = mpatches.FancyBboxPatch((1, y_pos-2.3), 8, 2.1, 

                                        boxstyle="round,pad=0.15", 

    plt.tight_layout()                                    edgecolor='black', facecolor=color_dnn, linewidth=2)

    return fig    ax.add_patch(rect)

    ax.text(5, y_pos-0.3, 'Stage 2: DNN Classification (Fused)', 

            ha='center', va='top', fontsize=12, fontweight='bold')

def visualize_panns_architecture():    

    """Create detailed flow diagram for PANNs architecture"""    # DNN layers

    fig, ax = plt.subplots(figsize=(14, 16))    dnn_layers = [

    ax.set_xlim(0, 10)        'Dense(768→256) + BatchNorm + ReLU + Dropout(0.3)',

    ax.set_ylim(0, 20)        'Dense(256→128) + BatchNorm + ReLU + Dropout(0.3)',

    ax.axis('off')        'Dense(128→3) [Output Layer]',

            'Output: Logits (3 classes)'

    # Color scheme    ]

    color_input = '#E8F4F8'    for i, layer in enumerate(dnn_layers):

    color_conv = '#81C784'        ax.text(5, y_pos-0.7-i*0.38, layer, 

    color_pool = '#FFB74D'                ha='center', va='top', fontsize=9)

    color_output = '#EF5350'    

        # Arrow

    # Title    ax.annotate('', xy=(5, y_pos-2.5), xytext=(5, y_pos-3.0),

    ax.text(5, 19.5, 'PANNs-CNN14 Architecture',                 arrowprops=dict(arrowstyle='->', lw=2, color='black'))

            ha='center', va='top', fontsize=18, fontweight='bold')    

        # Stage 5: Output

    # Stage 1: Input    y_pos = 3.5

    y_pos = 18    rect = mpatches.FancyBboxPatch((1, y_pos-1.3), 8, 1.1, 

    rect = mpatches.FancyBboxPatch((1, y_pos-0.8), 8, 0.6,                                     boxstyle="round,pad=0.1", 

                                    boxstyle="round,pad=0.1",                                     edgecolor='black', facecolor=color_output, linewidth=2)

                                    edgecolor='black', facecolor=color_input, linewidth=2)    ax.add_patch(rect)

    ax.add_patch(rect)    ax.text(5, y_pos-0.3, 'Output: Softmax Probabilities', 

    ax.text(5, y_pos-0.5, 'Input: Mel Spectrogram (HPSS)',             ha='center', va='top', fontsize=11, fontweight='bold')

            ha='center', va='center', fontsize=11, fontweight='bold')    ax.text(5, y_pos-0.7, 'drone: 0.92 | bird: 0.05 | background: 0.03', 

    ax.text(5, y_pos-1.1, 'Shape: (3, 96, 100)',             ha='center', va='top', fontsize=9, family='monospace')

            ha='center', va='top', fontsize=9, style='italic')    ax.text(5, y_pos-1.0, 'Prediction: "drone" (Confidence: 92%)', 

                ha='center', va='top', fontsize=9, fontweight='bold')

    # Arrow    

    ax.annotate('', xy=(5, y_pos-1.3), xytext=(5, y_pos-1.8),    # Key insights box

                arrowprops=dict(arrowstyle='->', lw=2, color='black'))    y_pos = 0.8

        rect = mpatches.FancyBboxPatch((0.2, y_pos-0.7), 9.6, 0.6, 

    # Conv blocks                                    boxstyle="round,pad=0.05", 

    conv_blocks = [                                    edgecolor='blue', facecolor='lightyellow', 

        ('Conv Block 1', 16.5, '2× Conv(3→64) + BN + ReLU → AvgPool(2)', '(64, 48, 50)'),                                    linewidth=1, linestyle='--')

        ('Conv Block 2', 14.5, '2× Conv(64→128) + BN + ReLU → AvgPool(2)', '(128, 24, 25)'),    ax.add_patch(rect)

        ('Conv Block 3', 12.5, '2× Conv(128→256) + BN + ReLU → AvgPool(2)', '(256, 12, 12)'),    ax.text(5, y_pos-0.4, 

        ('Conv Block 4', 10.5, '2× Conv(256→512) + BN + ReLU → AvgPool(2)', '(512, 6, 6)'),            'Total Parameters: 1,526,755 | Training Time: ~2-3 hours | Inference: ~50ms',

    ]            ha='center', va='center', fontsize=8, style='italic')

        

    for i, (name, y, desc, shape) in enumerate(conv_blocks):    plt.tight_layout()

        rect = mpatches.FancyBboxPatch((0.8, y-1.5), 8.4, 1.3,     return fig

                                        boxstyle="round,pad=0.1", 

                                        edgecolor='black', facecolor=color_conv, linewidth=2)

        ax.add_patch(rect)def visualize_feature_dimensions():

        ax.text(5, y-0.3, name, ha='center', va='top', fontsize=11, fontweight='bold')    """Visualize how tensor dimensions change through the network"""

        ax.text(5, y-0.7, desc, ha='center', va='top', fontsize=9)    fig, ax = plt.subplots(figsize=(12, 8))

        ax.text(5, y-1.1, f'Output: {shape}', ha='center', va='top', fontsize=9, style='italic')    

            # Define stages and their shapes

        # Arrow    stages = [

        if i < len(conv_blocks) - 1:        ('Input Audio', (32000,), 'Raw waveform'),

            ax.annotate('', xy=(5, y-1.7), xytext=(5, y-2.2),        ('FFT Features', (1, 128, 63), 'Mel spectrogram'),

                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))        ('Conv1', (32, 64, 31), 'Initial features'),

            ('ResBlock1', (64, 32, 15), 'Low-level patterns'),

    # Global pooling        ('ResBlock2', (128, 16, 7), 'Mid-level patterns'),

    y_pos = 8.5        ('ResBlock3', (256, 8, 3), 'High-level patterns'),

    ax.annotate('', xy=(5, y_pos), xytext=(5, y_pos-0.5),        ('Global Pool', (256,), 'Spatial aggregate'),

                arrowprops=dict(arrowstyle='->', lw=2, color='black'))        ('FC Layer', (512,), 'Feature vector'),

            ('DNN Hidden1', (256,), 'Abstract features'),

    rect = mpatches.FancyBboxPatch((1.5, y_pos-1.5), 7, 1.0,         ('DNN Hidden2', (128,), 'Refined features'),

                                    boxstyle="round,pad=0.1",         ('Output', (3,), 'Class logits')

                                    edgecolor='purple', facecolor=color_pool, linewidth=2)    ]

    ax.add_patch(rect)    

    ax.text(5, y_pos-0.3, 'Global Average Pooling',     # Plot

            ha='center', va='top', fontsize=11, fontweight='bold')    x_positions = np.linspace(0, 10, len(stages))

    ax.text(5, y_pos-0.7, '(512, 6, 6) → (512,)',     colors = plt.cm.viridis(np.linspace(0, 1, len(stages)))

            ha='center', va='top', fontsize=9)    

        for i, ((name, shape, desc), x, color) in enumerate(zip(stages, x_positions, colors)):

    # Arrow        # Calculate "size" for visualization

    ax.annotate('', xy=(5, y_pos-1.7), xytext=(5, y_pos-2.2),        if len(shape) == 1:

                arrowprops=dict(arrowstyle='->', lw=2, color='black'))            size = shape[0]

                shape_text = f'{shape[0]}'

    # Classification head        elif len(shape) == 3:

    y_pos = 6            size = shape[0] * shape[1] * shape[2]

    rect = mpatches.FancyBboxPatch((1, y_pos-1.8), 8, 1.6,             shape_text = f'{shape[0]}×{shape[1]}×{shape[2]}'

                                    boxstyle="round,pad=0.1",         else:

                                    edgecolor='black', facecolor=color_output, linewidth=2)            size = np.prod(shape)

    ax.add_patch(rect)            shape_text = '×'.join(map(str, shape))

    ax.text(5, y_pos-0.3, 'Classification Head',         

            ha='center', va='top', fontsize=11, fontweight='bold')        # Logarithmic scale for better visualization

            height = np.log10(size + 1) * 0.5

    fc_layers = [        

        'Dropout(0.3) + Linear(512→512) + ReLU',        # Draw bar

        'Dropout(0.3) + Linear(512→3)',        ax.bar(x, height, width=0.6, color=color, alpha=0.7, edgecolor='black', linewidth=1.5)

        'Output: Class logits (3,)'        

    ]        # Add text

    for i, layer in enumerate(fc_layers):        ax.text(x, height + 0.1, name, ha='center', va='bottom', 

        ax.text(5, y_pos-0.7-i*0.35, layer,                 fontsize=9, fontweight='bold', rotation=45)

                ha='center', va='top', fontsize=9)        ax.text(x, height/2, shape_text, ha='center', va='center', 

                    fontsize=8, fontweight='bold')

    # Key insights box        ax.text(x, -0.3, desc, ha='center', va='top', 

    y_pos = 3.8                fontsize=7, style='italic', rotation=45)

    rect = mpatches.FancyBboxPatch((0.2, y_pos-1.5), 9.6, 1.4,         

                                    boxstyle="round,pad=0.05",         # Draw arrows between stages

                                    edgecolor='blue', facecolor='lightyellow',         if i < len(stages) - 1:

                                    linewidth=1, linestyle='--')            ax.annotate('', xy=(x_positions[i+1]-0.3, 0.1), xytext=(x+0.3, 0.1),

    ax.add_patch(rect)                       arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))

    ax.text(5, y_pos-0.2, '🎯 Key Advantages',     

            ha='center', va='top', fontsize=11, fontweight='bold')    ax.set_xlim(-0.5, 10.5)

    advantages = [    ax.set_ylim(-1.5, 5)

        '✓ Balanced: ~5-10M parameters, strong accuracy',    ax.set_ylabel('Log10(Tensor Size)', fontsize=11, fontweight='bold')

        '✓ Deep hierarchical features (4 conv blocks)',    ax.set_title('Tensor Dimension Flow Through FFT + CNN + DNN Pipeline', 

        '✓ Inspired by pre-trained audio models (PANNs)',                 fontsize=14, fontweight='bold', pad=20)

        '✓ Good for moderate-resource scenarios'    ax.set_xticks([])

    ]    ax.spines['top'].set_visible(False)

    for i, adv in enumerate(advantages):    ax.spines['right'].set_visible(False)

        ax.text(5, y_pos-0.5-i*0.25, adv,     ax.grid(axis='y', alpha=0.3, linestyle='--')

                ha='center', va='top', fontsize=8)    

        plt.tight_layout()

    plt.tight_layout()    return fig

    return fig



def analyze_model_complexity():

def visualize_transformer_architecture():    """Analyze and visualize model complexity"""

    """Create detailed flow diagram for Transformer architecture"""    model = FFTCNNDNNFusion(n_classes=3, in_channels=1, 

    fig, ax = plt.subplots(figsize=(14, 18))                            cnn_feature_dim=512, dnn_hidden_dims=[256, 128])

    ax.set_xlim(0, 10)    

    ax.set_ylim(0, 22)    # Count parameters per component

    ax.axis('off')    component_params = {}

        

    # Color scheme    # CNN components

    color_input = '#E8F4F8'    cnn_params = sum(p.numel() for p in model.cnn.parameters())

    color_patch = '#B3E5FC'    component_params['CNN Feature Extractor'] = cnn_params

    color_trans = '#9575CD'    

    color_output = '#EF5350'    # DNN components

        dnn_params = sum(p.numel() for p in model.dnn.parameters())

    # Title    component_params['DNN Classifier'] = dnn_params

    ax.text(5, 21.5, 'Audio Spectrogram Transformer (AST)',     

            ha='center', va='top', fontsize=18, fontweight='bold')    total_params = sum(component_params.values())

        

    # Stage 1: Input    # Create visualization

    y_pos = 20    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    rect = mpatches.FancyBboxPatch((1, y_pos-0.8), 8, 0.6,     

                                    boxstyle="round,pad=0.1",     # Pie chart

                                    edgecolor='black', facecolor=color_input, linewidth=2)    colors = ['#81C784', '#FFB74D']

    ax.add_patch(rect)    explode = (0.05, 0.05)

    ax.text(5, y_pos-0.5, 'Input: Mel Spectrogram (HPSS)',     wedges, texts, autotexts = ax1.pie(component_params.values(), 

            ha='center', va='center', fontsize=11, fontweight='bold')                                         labels=component_params.keys(),

    ax.text(5, y_pos-1.1, 'Shape: (3, 96, 100)',                                          autopct='%1.1f%%',

            ha='center', va='top', fontsize=9, style='italic')                                         colors=colors,

                                             explode=explode,

    # Arrow                                         startangle=90,

    ax.annotate('', xy=(5, y_pos-1.3), xytext=(5, y_pos-1.8),                                         textprops={'fontsize': 11})

                arrowprops=dict(arrowstyle='->', lw=2, color='black'))    ax1.set_title('Parameter Distribution', fontsize=13, fontweight='bold')

        

    # Patch embedding    # Make percentage text bold

    y_pos = 18    for autotext in autotexts:

    rect = mpatches.FancyBboxPatch((0.5, y_pos-1.8), 9, 1.6,         autotext.set_color('white')

                                    boxstyle="round,pad=0.15",         autotext.set_fontweight('bold')

                                    edgecolor='black', facecolor=color_patch, linewidth=2)    

    ax.add_patch(rect)    # Bar chart with details

    ax.text(5, y_pos-0.3, 'Patch Embedding (Vision Transformer Style)',     components = list(component_params.keys())

            ha='center', va='top', fontsize=12, fontweight='bold')    params = list(component_params.values())

        

    patch_features = [    bars = ax2.barh(components, params, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        'Conv2d(3→384, kernel=16, stride=16) - Split into patches',    ax2.set_xlabel('Number of Parameters', fontsize=11, fontweight='bold')

        'Flatten: (384, 6, 6) → (36 patches, 384 dims)',    ax2.set_title('Parameter Count by Component', fontsize=13, fontweight='bold')

        'Add CLS token: (1 + 36, 384)',    ax2.grid(axis='x', alpha=0.3, linestyle='--')

        'Add positional embeddings (learned)'    

    ]    # Add value labels on bars

    for i, feature in enumerate(patch_features):    for bar, param in zip(bars, params):

        ax.text(5, y_pos-0.7-i*0.27, feature,         width = bar.get_width()

                ha='center', va='top', fontsize=8.5)        ax2.text(width, bar.get_y() + bar.get_height()/2, 

                    f'{param:,}', ha='left', va='center', fontsize=10, fontweight='bold')

    # Arrow    

    ax.annotate('', xy=(5, y_pos-2.0), xytext=(5, y_pos-2.5),    # Add total

                arrowprops=dict(arrowstyle='->', lw=2, color='black'))    fig.suptitle(f'Model Complexity Analysis - Total Parameters: {total_params:,}', 

                     fontsize=15, fontweight='bold', y=0.98)

    # Transformer blocks    

    y_pos = 15    plt.tight_layout()

    rect = mpatches.FancyBboxPatch((0.5, y_pos-3.5), 9, 3.3,     return fig, component_params, total_params

                                    boxstyle="round,pad=0.15", 

                                    edgecolor='purple', facecolor=color_trans, linewidth=2)

    ax.add_patch(rect)def create_receptive_field_diagram():

    ax.text(5, y_pos-0.3, 'Transformer Encoder (12 layers)',     """Visualize receptive field growth through CNN layers"""

            ha='center', va='top', fontsize=12, fontweight='bold')    fig, ax = plt.subplots(figsize=(12, 6))

        

    trans_features = [    # Layer information: (name, receptive_field_size, feature_map_size)

        'Each layer contains:',    layers = [

        '  1. Multi-Head Self-Attention (6 heads)',        ('Input', 1, 128),

        '     → Learn relationships between all patches',        ('Conv1 3×3', 3, 64),

        '     → Capture long-range dependencies',        ('MaxPool 2×2', 6, 32),

        '  2. Feed-Forward Network (MLP)',        ('ResBlock1', 14, 16),

        '     → Expand: 384 → 1536 dims (4x)',        ('ResBlock2', 30, 8),

        '     → Contract: 1536 → 384 dims',        ('ResBlock3', 62, 4),

        '  3. Layer Normalization + Residual connections',        ('Global Pool', 128, 1)

        '',    ]

        'Output: (37, 384) - contextualized patch embeddings'    

    ]    x_pos = np.arange(len(layers))

    for i, feature in enumerate(trans_features):    receptive_fields = [l[1] for l in layers]

        ax.text(5, y_pos-0.7-i*0.28, feature,     feature_sizes = [l[2] for l in layers]

                ha='center', va='top', fontsize=8.5)    

        # Create dual-axis plot

    # Arrow    ax1 = ax

    ax.annotate('', xy=(5, y_pos-3.7), xytext=(5, y_pos-4.2),    ax2 = ax1.twinx()

                arrowprops=dict(arrowstyle='->', lw=2, color='black'))    

        # Plot receptive field

    # Classification    line1 = ax1.plot(x_pos, receptive_fields, 'o-', color='#2196F3', 

    y_pos = 10.5                     linewidth=2.5, markersize=10, label='Receptive Field')

    rect = mpatches.FancyBboxPatch((1, y_pos-1.5), 8, 1.3,     ax1.fill_between(x_pos, 0, receptive_fields, alpha=0.2, color='#2196F3')

                                    boxstyle="round,pad=0.1",     ax1.set_ylabel('Receptive Field Size (frequency bins)', 

                                    edgecolor='black', facecolor=color_output, linewidth=2)                   fontsize=11, fontweight='bold', color='#2196F3')

    ax.add_patch(rect)    ax1.tick_params(axis='y', labelcolor='#2196F3')

    ax.text(5, y_pos-0.3, 'Classification Head',     

            ha='center', va='top', fontsize=11, fontweight='bold')    # Plot feature map size

    ax.text(5, y_pos-0.7, 'Extract CLS token (first position) → (384,)',     line2 = ax2.plot(x_pos, feature_sizes, 's-', color='#F44336', 

            ha='center', va='top', fontsize=9)                     linewidth=2.5, markersize=10, label='Feature Map Size')

    ax.text(5, y_pos-1.0, 'Linear(384→3) → Class logits',     ax2.fill_between(x_pos, 0, feature_sizes, alpha=0.2, color='#F44336')

            ha='center', va='top', fontsize=9, fontweight='bold')    ax2.set_ylabel('Feature Map Height', fontsize=11, fontweight='bold', color='#F44336')

        ax2.tick_params(axis='y', labelcolor='#F44336')

    # Self-attention visualization    ax2.set_yscale('log')

    y_pos = 8.5    

    rect = mpatches.FancyBboxPatch((0.3, y_pos-2.5), 9.4, 2.3,     # X-axis

                                    boxstyle="round,pad=0.1",     ax1.set_xticks(x_pos)

                                    edgecolor='green', facecolor='#E8F5E9',     ax1.set_xticklabels([l[0] for l in layers], rotation=45, ha='right')

                                    linewidth=1.5, linestyle='--')    ax1.set_xlabel('Network Layer', fontsize=11, fontweight='bold')

    ax.add_patch(rect)    

    ax.text(5, y_pos-0.2, '💡 Self-Attention Mechanism',     # Title and grid

            ha='center', va='top', fontsize=11, fontweight='bold', color='green')    ax1.set_title('Receptive Field Growth & Feature Map Reduction Through CNN', 

                      fontsize=13, fontweight='bold', pad=20)

    attn_desc = [    ax1.grid(True, alpha=0.3, linestyle='--')

        'Each patch attends to all other patches:',    

        '• Query (Q), Key (K), Value (V) projections',    # Legend

        '• Attention = softmax(QK^T / √d) × V',    lines = line1 + line2

        '• Learns to focus on: rotor harmonics, temporal patterns, spatial correlations',    labels = [l.get_label() for l in lines]

        '• 6 attention heads capture different feature aspects simultaneously'    ax1.legend(lines, labels, loc='center left', fontsize=10)

    ]    

    for i, desc in enumerate(attn_desc):    plt.tight_layout()

        ax.text(5, y_pos-0.5-i*0.35, desc,     return fig

                ha='center', va='top', fontsize=8)

    

    # Key insights boxdef generate_architecture_report():

    y_pos = 5.5    """Generate detailed architecture analysis report"""

    rect = mpatches.FancyBboxPatch((0.2, y_pos-1.8), 9.6, 1.6,     model = FFTCNNDNNFusion(n_classes=3, in_channels=1)

                                    boxstyle="round,pad=0.05",     

                                    edgecolor='blue', facecolor='lightyellow',     report = []

                                    linewidth=1, linestyle='--')    report.append("="*80)

    ax.add_patch(rect)    report.append("FFT + CNN + DNN ARCHITECTURE ANALYSIS")

    ax.text(5, y_pos-0.2, '🎯 Key Advantages',     report.append("="*80)

            ha='center', va='top', fontsize=11, fontweight='bold')    report.append("")

    advantages = [    

        '✓ Best accuracy: ~20-30M parameters, state-of-the-art',    # Overall statistics

        '✓ Global context: Each patch sees all other patches (no locality bias)',    total_params = sum(p.numel() for p in model.parameters())

        '✓ Learns complex patterns: Long-range dependencies, harmonics',    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        '✓ Inspired by ViT/AST: Vision & Audio Transformers',    

        '✗ Requires GPU for practical use, ~10-20ms inference'    report.append("OVERALL MODEL STATISTICS")

    ]    report.append("-" * 80)

    for i, adv in enumerate(advantages):    report.append(f"Total Parameters:      {total_params:>15,}")

        ax.text(5, y_pos-0.5-i*0.25, adv,     report.append(f"Trainable Parameters:  {trainable_params:>15,}")

                ha='center', va='top', fontsize=8)    report.append(f"Model Size (approx):   {total_params * 4 / (1024**2):>12.2f} MB (float32)")

        report.append("")

    plt.tight_layout()    

    return fig    # Component breakdown

    report.append("COMPONENT BREAKDOWN")

    report.append("-" * 80)

def analyze_model_complexity():    

    """Analyze and compare all three models"""    # CNN

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))    cnn_params = sum(p.numel() for p in model.cnn.parameters())

        report.append(f"CNN Feature Extractor: {cnn_params:>15,} params ({cnn_params/total_params*100:.1f}%)")

    # Create models    report.append("  - Initial Conv Layer:     32 channels")

    models = {    report.append("  - ResBlock 1 + Attention: 64 channels")

        'CRNN': create_model('crnn', num_classes=3, input_channels=3, n_mels=96),    report.append("  - ResBlock 2 + Attention: 128 channels")

        'PANNs': create_model('panns', num_classes=3, input_channels=3),    report.append("  - ResBlock 3 + Attention: 256 channels")

        'Transformer': create_model('transformer', num_classes=3, input_channels=3, depth=12)    report.append("  - Feature Projection:     256 → 512 dims")

    }    report.append("")

        

    # Count parameters    # DNN

    params = {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()}    dnn_params = sum(p.numel() for p in model.dnn.parameters())

        report.append(f"DNN Classifier:        {dnn_params:>15,} params ({dnn_params/total_params*100:.1f}%)")

    # 1. Parameter comparison (bar chart)    report.append("  - Hidden Layer 1:         512 → 256 dims")

    names = list(params.keys())    report.append("  - Hidden Layer 2:         256 → 128 dims")

    counts = [params[n] / 1e6 for n in names]  # In millions    report.append("  - Output Layer:           128 → 3 dims")

    colors = ['#81C784', '#FFB74D', '#9575CD']    report.append("")

        

    bars = ax1.bar(names, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)    # Key features

    ax1.set_ylabel('Parameters (Millions)', fontsize=12, fontweight='bold')    report.append("KEY ARCHITECTURAL FEATURES")

    ax1.set_title('Model Size Comparison', fontsize=14, fontweight='bold')    report.append("-" * 80)

    ax1.grid(axis='y', alpha=0.3, linestyle='--')    report.append("✓ Residual Connections:    Better gradient flow, deeper training")

        report.append("✓ Channel Attention:       Focus on important frequency channels")

    for bar, count in zip(bars, counts):    report.append("✓ Batch Normalization:     Faster convergence, regularization")

        height = bar.get_height()    report.append("✓ Dropout (0.3, 0.2):      Prevent overfitting")

        ax1.text(bar.get_x() + bar.get_width()/2., height,    report.append("✓ Global Average Pooling:  Reduce parameters, prevent overfitting")

                f'{count:.2f}M',    report.append("✓ Multi-layer DNN:         Hierarchical decision making")

                ha='center', va='bottom', fontsize=11, fontweight='bold')    report.append("")

        

    # 2. Speed comparison (estimated)    # Performance characteristics

    inference_times = {    report.append("PERFORMANCE CHARACTERISTICS")

        'CRNN': 1.5,    report.append("-" * 80)

        'PANNs': 5.0,    report.append("Input Processing:")

        'Transformer': 15.0    report.append("  - Audio Duration:         2.0 seconds @ 16kHz")

    }    report.append("  - FFT Window:             2048 samples")

        report.append("  - Hop Length:             512 samples")

    names = list(inference_times.keys())    report.append("  - Mel Bands:              128")

    times = [inference_times[n] for n in names]    report.append("  - Input Shape:            (1, 128, 63)")

        report.append("")

    bars = ax2.barh(names, times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)    report.append("Inference Speed (CPU):      ~50-100ms per sample")

    ax2.set_xlabel('Inference Time (ms) - GPU', fontsize=12, fontweight='bold')    report.append("Inference Speed (GPU):      ~10-20ms per sample")

    ax2.set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')    report.append("Training Time (50 epochs):  ~2-3 hours on GPU")

    ax2.grid(axis='x', alpha=0.3, linestyle='--')    report.append("Memory Usage (training):    ~2-4 GB GPU")

    ax2.invert_yaxis()    report.append("")

        

    for bar, time in zip(bars, times):    # Comparison with alternatives

        width = bar.get_width()    report.append("COMPARISON WITH BASELINE")

        ax2.text(width, bar.get_y() + bar.get_height()/2.,    report.append("-" * 80)

                f' {time}ms',    report.append("FFT + CNN + DNN vs Simple CNN:")

                ha='left', va='center', fontsize=11, fontweight='bold')    report.append("  ✓ Better frequency representation (FFT preprocessing)")

        report.append("  ✓ Deeper feature learning (Residual blocks)")

    # 3. Complexity breakdown (pie charts)    report.append("  ✓ More sophisticated classification (Multi-layer DNN)")

    ax3.axis('off')    report.append("  ✓ Higher accuracy (~5-10% improvement expected)")

    ax3.text(0.5, 0.95, 'Architecture Characteristics',     report.append("  ✗ More parameters (1.5M vs ~500K)")

            ha='center', va='top', fontsize=14, fontweight='bold', transform=ax3.transAxes)    report.append("  ✗ Slightly slower inference (~2x)")

        report.append("")

    characteristics = {    

        'CRNN': ['Conv', 'Attention', 'RNN', 'FC'],    report.append("="*80)

        'PANNs': ['Conv Blocks', 'Global Pool', 'FC'],    

        'Transformer': ['Patches', 'Self-Attn', 'MLP', 'FC']    return "\n".join(report)

    }

    

    colors_pie = plt.cm.Set3(np.linspace(0, 1, 4))def main():

        """Generate all visualizations and reports"""

    for i, (name, components) in enumerate(characteristics.items()):    print("="*80)

        values = [1] * len(components)  # Equal weighting for visualization    print("Generating FFT + CNN + DNN Architecture Visualizations")

            print("="*80)

        # Create mini pie chart    

        circle = plt.Circle((0.2 + i*0.3, 0.5), 0.12, transform=ax3.transAxes,     output_dir = Path("architecture_visualizations")

                           fill=False, edgecolor='gray', linewidth=1)    output_dir.mkdir(exist_ok=True)

            

        ax_pie = fig.add_axes([0.08 + i*0.28, 0.05, 0.2, 0.2])    # 1. Architecture flow diagram

        ax_pie.pie(values, labels=components, colors=colors_pie[:len(components)],    print("\n1. Creating architecture flow diagram...")

                  textprops={'fontsize': 8}, startangle=90)    fig1 = visualize_architecture_flow()

        ax_pie.set_title(name, fontsize=11, fontweight='bold', pad=10)    fig1.savefig(output_dir / "architecture_flow.png", dpi=300, bbox_inches='tight')

        print(f"   ✓ Saved: {output_dir / 'architecture_flow.png'}")

    # 4. Use case recommendations    

    ax4.axis('off')    # 2. Feature dimensions

    ax4.text(0.5, 0.95, 'Model Selection Guide',     print("\n2. Creating feature dimension visualization...")

            ha='center', va='top', fontsize=14, fontweight='bold', transform=ax4.transAxes)    fig2 = visualize_feature_dimensions()

        fig2.savefig(output_dir / "feature_dimensions.png", dpi=300, bbox_inches='tight')

    recommendations = [    print(f"   ✓ Saved: {output_dir / 'feature_dimensions.png'}")

        ('CRNN with Attention',     

         '• Edge devices & embedded systems\n• Real-time detection required\n• Limited compute resources\n• Good baseline accuracy',    # 3. Model complexity

         '#81C784'),    print("\n3. Analyzing model complexity...")

        ('PANNs-CNN14',     fig3, comp_params, total = analyze_model_complexity()

         '• Balanced accuracy/speed tradeoff\n• Desktop/server applications\n• Moderate GPU resources\n• Strong overall performance',    fig3.savefig(output_dir / "model_complexity.png", dpi=300, bbox_inches='tight')

         '#FFB74D'),    print(f"   ✓ Saved: {output_dir / 'model_complexity.png'}")

        ('Audio Transformer',     print(f"   Total Parameters: {total:,}")

         '• Maximum accuracy required\n• GPU available\n• Complex acoustic scenes\n• Research & benchmarking',    

         '#9575CD')    # 4. Receptive field

    ]    print("\n4. Creating receptive field diagram...")

        fig4 = create_receptive_field_diagram()

    y_start = 0.75    fig4.savefig(output_dir / "receptive_field.png", dpi=300, bbox_inches='tight')

    for i, (name, desc, color) in enumerate(recommendations):    print(f"   ✓ Saved: {output_dir / 'receptive_field.png'}")

        y = y_start - i * 0.25    

            # 5. Architecture report

        # Box    print("\n5. Generating architecture analysis report...")

        rect = mpatches.FancyBboxPatch((0.05, y-0.18), 0.9, 0.15,     report = generate_architecture_report()

                                       transform=ax4.transAxes,    report_path = output_dir / "architecture_report.txt"

                                       boxstyle="round,pad=0.01",     with open(report_path, 'w', encoding='utf-8') as f:

                                       edgecolor='black', facecolor=color,         f.write(report)

                                       alpha=0.3, linewidth=1.5)    print(f"   ✓ Saved: {report_path}")

        ax4.add_patch(rect)    

            # Print report to console

        # Text    print("\n" + report)

        ax4.text(0.08, y-0.03, name, transform=ax4.transAxes,    

                fontsize=11, fontweight='bold', va='top')    print(f"\n{'='*80}")

        ax4.text(0.08, y-0.06, desc, transform=ax4.transAxes,    print("All visualizations generated successfully!")

                fontsize=8, va='top', linespacing=1.5)    print(f"Check the '{output_dir}' directory for all outputs")

        print("="*80)

    plt.tight_layout()    

    return fig, params    plt.show()





def generate_architecture_report(model_type='panns'):if __name__ == '__main__':

    """Generate detailed architecture analysis report"""    main()

    model = create_model(model_type, num_classes=3, input_channels=3, 
                        n_mels=96 if model_type == 'crnn' else None)
    
    report = []
    report.append("="*80)
    report.append(f"SOTA MODEL ARCHITECTURE ANALYSIS - {model_type.upper()}")
    report.append("="*80)
    report.append("")
    
    # Overall statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    report.append("OVERALL MODEL STATISTICS")
    report.append("-" * 80)
    report.append(f"Model Type:            {model_type.upper()}")
    report.append(f"Total Parameters:      {total_params:>15,}")
    report.append(f"Trainable Parameters:  {trainable_params:>15,}")
    report.append(f"Model Size (approx):   {total_params * 4 / (1024**2):>12.2f} MB (float32)")
    report.append("")
    
    # Model-specific details
    if model_type == 'crnn':
        report.append("CRNN WITH ATTENTION ARCHITECTURE")
        report.append("-" * 80)
        report.append("Components:")
        report.append("  1. Convolutional Blocks (3 layers):")
        report.append("     - Conv2d + BatchNorm + ReLU + MaxPool")
        report.append("     - Progressively increase channels: 32 → 64 → 128")
        report.append("  2. Temporal-Frequency Attention:")
        report.append("     - Learns to focus on discriminative regions")
        report.append("     - Separate temporal and frequency attention")
        report.append("  3. Bidirectional GRU (2 layers):")
        report.append("     - Captures temporal context")
        report.append("     - Hidden size: 128 (256 bidirectional)")
        report.append("  4. Classification Head:")
        report.append("     - Dropout + Linear layer")
        report.append("")
    
    elif model_type == 'panns':
        report.append("PANNS-CNN14 ARCHITECTURE")
        report.append("-" * 80)
        report.append("Components:")
        report.append("  1. Convolutional Blocks (4 blocks, 8 conv layers):")
        report.append("     - Block 1: 2× Conv(3→64) + AvgPool")
        report.append("     - Block 2: 2× Conv(64→128) + AvgPool")
        report.append("     - Block 3: 2× Conv(128→256) + AvgPool")
        report.append("     - Block 4: 2× Conv(256→512) + AvgPool")
        report.append("  2. Global Average Pooling:")
        report.append("     - Reduces spatial dimensions to single vector")
        report.append("  3. Classification Head:")
        report.append("     - FC(512→512) + ReLU + Dropout")
        report.append("     - FC(512→3)")
        report.append("")
    
    elif model_type == 'transformer':
        report.append("AUDIO SPECTROGRAM TRANSFORMER ARCHITECTURE")
        report.append("-" * 80)
        report.append("Components:")
        report.append("  1. Patch Embedding:")
        report.append("     - Conv2d projection (patch_size=16)")
        report.append("     - Splits spectrogram into patches")
        report.append("     - Adds CLS token and positional embeddings")
        report.append("  2. Transformer Encoder (12 layers):")
        report.append("     - Multi-head self-attention (6 heads)")
        report.append("     - Feed-forward MLP (expand 4x)")
        report.append("     - Layer normalization + residuals")
        report.append("  3. Classification Head:")
        report.append("     - Extract CLS token")
        report.append("     - Linear(384→3)")
        report.append("")
    
    # Key features
    report.append("KEY ARCHITECTURAL FEATURES")
    report.append("-" * 80)
    
    if model_type == 'crnn':
        report.append("✓ Lightweight & Fast:         ~1-2M params, 1-2ms inference")
        report.append("✓ Attention Mechanism:        Focuses on rotor harmonics")
        report.append("✓ Temporal Modeling:          BiGRU captures context")
        report.append("✓ Edge-Friendly:              Suitable for embedded systems")
    elif model_type == 'panns':
        report.append("✓ Balanced Performance:       ~5-10M params, 5-10ms inference")
        report.append("✓ Deep Hierarchy:             4 conv blocks, 8 layers")
        report.append("✓ Proven Architecture:        Based on PANNs (Kong et al., 2020)")
        report.append("✓ Good Accuracy:              Strong performance on audio tasks")
    elif model_type == 'transformer':
        report.append("✓ State-of-the-Art:           ~20-30M params, best accuracy")
        report.append("✓ Global Context:             Self-attention sees all patches")
        report.append("✓ No Inductive Bias:          Learns patterns from data")
        report.append("✓ Inspired by ViT/AST:        Vision & Audio Transformers")
    
    report.append("")
    
    # Performance characteristics
    report.append("PERFORMANCE CHARACTERISTICS")
    report.append("-" * 80)
    report.append("Input Processing:")
    report.append("  - Audio Duration:         2.0 seconds @ 16kHz")
    report.append("  - Mel Bands:              96")
    report.append("  - HPSS Channels:          3 (Total, Harmonic, Percussive)")
    report.append("  - Input Shape:            (3, 96, ~100)")
    report.append("")
    
    if model_type == 'crnn':
        report.append("Inference Speed (GPU):      ~1-2ms per sample")
        report.append("Inference Speed (CPU):      ~5-10ms per sample")
        report.append("Training Time (50 epochs):  ~30-60 minutes")
        report.append("Memory Usage (training):    ~1-2 GB GPU")
    elif model_type == 'panns':
        report.append("Inference Speed (GPU):      ~5-10ms per sample")
        report.append("Inference Speed (CPU):      ~20-50ms per sample")
        report.append("Training Time (50 epochs):  ~1-2 hours")
        report.append("Memory Usage (training):    ~2-4 GB GPU")
    elif model_type == 'transformer':
        report.append("Inference Speed (GPU):      ~10-20ms per sample")
        report.append("Inference Speed (CPU):      ~100-200ms per sample")
        report.append("Training Time (50 epochs):  ~2-4 hours")
        report.append("Memory Usage (training):    ~4-8 GB GPU")
    
    report.append("")
    report.append("="*80)
    
    return "\n".join(report)


def main():
    """Generate all visualizations and reports"""
    parser = argparse.ArgumentParser(description='Visualize SOTA model architectures')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'crnn', 'panns', 'transformer'],
                       help='Which model to visualize')
    args = parser.parse_args()
    
    print("="*80)
    print("Generating SOTA Architecture Visualizations")
    print("="*80)
    
    output_dir = Path("architecture_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    models_to_viz = ['crnn', 'panns', 'transformer'] if args.model == 'all' else [args.model]
    
    # Architecture diagrams
    for model_type in models_to_viz:
        print(f"\n📊 Creating {model_type.upper()} architecture diagram...")
        if model_type == 'crnn':
            fig = visualize_crnn_architecture()
        elif model_type == 'panns':
            fig = visualize_panns_architecture()
        else:
            fig = visualize_transformer_architecture()
        
        fig.savefig(output_dir / f"{model_type}_architecture.png", dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {output_dir / f'{model_type}_architecture.png'}")
        plt.close(fig)
    
    # Comparison analysis
    if args.model == 'all':
        print("\n📈 Creating model comparison analysis...")
        fig, params = analyze_model_complexity()
        fig.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {output_dir / 'model_comparison.png'}")
        print(f"\n   Parameter counts:")
        for name, count in params.items():
            print(f"     {name}: {count:,} ({count/1e6:.2f}M)")
        plt.close(fig)
    
    # Architecture reports
    for model_type in models_to_viz:
        print(f"\n📝 Generating {model_type.UPPER()} architecture report...")
        report = generate_architecture_report(model_type)
        report_path = output_dir / f"{model_type}_architecture_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"   ✓ Saved: {report_path}")
    
    print(f"\n{'='*80}")
    print("All visualizations generated successfully!")
    print(f"Check the '{output_dir}' directory for all outputs")
    print("="*80)


if __name__ == '__main__':
    main()
