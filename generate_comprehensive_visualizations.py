"""
Generate comprehensive visualizations for README documentation:
1. Preprocessing pipeline flowchart with sample transformations
2. CRNN architecture diagram with detailed parameters
3. Training process flowchart
4. Sample comparisons (drone vs helicopter vs background)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import seaborn as sns
from pathlib import Path
import librosa
import librosa.display
from advanced_preprocessing import AudioPreprocessor
import torch
from src.adrone.models.acoustic_models import CRNNWithAttention

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
OUTPUT_DIR = Path("visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_preprocessing_flowchart_with_examples():
    """
    Create detailed preprocessing flowchart with visual examples at each stage
    Shows: WAV -> Resampling -> Mel Spec -> MFCC -> Spectral -> 3-channel stack
    """
    print("Creating preprocessing flowchart with examples...")
    
    # Load sample audio files from each class
    train_dir = Path("data/edth_munich_dataset/data/train")
    sample_files = {
        'drone': list((train_dir / "drone").glob("*.wav"))[0],
        'helicopter': list((train_dir / "helicopter").glob("*.wav"))[0],
        'background': list((train_dir / "background").glob("*.wav"))[0]
    }
    
    # Create preprocessor
    preprocessor = AudioPreprocessor(
        sample_rate=22050,
        duration=3.0,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        n_mfcc=40
    )
    
    # Process one sample to show the pipeline
    sample_audio_path = str(sample_files['drone'])
    
    # Step-by-step processing
    print(f"  Processing sample: {sample_audio_path}")
    
    # 1. Load raw audio
    audio_raw, sr_original = librosa.load(sample_audio_path, sr=None, duration=3.0)
    
    # 2. Resample to 22050 Hz
    audio_resampled = librosa.resample(audio_raw, orig_sr=sr_original, target_sr=22050)
    audio_normalized = librosa.util.normalize(audio_resampled)
    
    # Pad/trim to fixed length
    n_samples = int(22050 * 3.0)
    if len(audio_normalized) < n_samples:
        audio_normalized = np.pad(audio_normalized, (0, n_samples - len(audio_normalized)))
    else:
        audio_normalized = audio_normalized[:n_samples]
    
    # 3. Extract features
    mel_spec = preprocessor.extract_mel_spectrogram(audio_normalized)
    mfcc = preprocessor.extract_mfcc(audio_normalized)
    spectral = preprocessor.extract_spectral_features(audio_normalized)
    
    # Create the comprehensive flowchart
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 6, hspace=0.4, wspace=0.3)
    
    # Title
    fig.suptitle('Audio Preprocessing Pipeline for Acoustic Drone Detection', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Row 1: Flowchart boxes
    ax_flow = fig.add_subplot(gs[0, :])
    ax_flow.axis('off')
    
    # Draw flowchart
    box_width = 0.13
    box_height = 0.6
    y_pos = 0.4
    
    boxes = [
        {'x': 0.02, 'label': 'Raw WAV\nFile', 'color': '#FF6B6B'},
        {'x': 0.17, 'label': 'Resample\n22.05kHz', 'color': '#4ECDC4'},
        {'x': 0.32, 'label': 'Normalize\nAmplitude', 'color': '#4ECDC4'},
        {'x': 0.47, 'label': 'Mel\nSpectrogram', 'color': '#95E1D3'},
        {'x': 0.62, 'label': 'MFCC +\nDeltas', 'color': '#95E1D3'},
        {'x': 0.77, 'label': 'Spectral\nFeatures', 'color': '#95E1D3'},
        {'x': 0.92, 'label': '3-Channel\nStack', 'color': '#F38181'}
    ]
    
    for i, box in enumerate(boxes):
        rect = FancyBboxPatch((box['x'], y_pos - box_height/2), box_width, box_height,
                              boxstyle="round,pad=0.05", 
                              edgecolor='black', facecolor=box['color'],
                              linewidth=2, transform=ax_flow.transAxes)
        ax_flow.add_patch(rect)
        ax_flow.text(box['x'] + box_width/2, y_pos, box['label'],
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    transform=ax_flow.transAxes)
        
        # Draw arrows
        if i < len(boxes) - 1:
            arrow = FancyArrowPatch((box['x'] + box_width, y_pos),
                                   (boxes[i+1]['x'], y_pos),
                                   arrowstyle='->', mutation_scale=30, linewidth=2,
                                   color='black', transform=ax_flow.transAxes)
            ax_flow.add_patch(arrow)
    
    # Row 2: Raw waveform and resampled
    ax1 = fig.add_subplot(gs[1, 0:2])
    times_raw = np.arange(len(audio_raw)) / sr_original
    ax1.plot(times_raw, audio_raw, linewidth=0.5, color='#FF6B6B')
    ax1.set_title(f'Step 1: Raw Audio\n(SR={sr_original}Hz, Length={len(audio_raw)} samples)', 
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[1, 2:4])
    times_resampled = np.arange(len(audio_normalized)) / 22050
    ax2.plot(times_resampled, audio_normalized, linewidth=0.5, color='#4ECDC4')
    ax2.set_title(f'Step 2: Resampled & Normalized\n(SR=22050Hz, Length={len(audio_normalized)} samples)', 
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)
    
    # Information box
    ax_info = fig.add_subplot(gs[1, 4:6])
    ax_info.axis('off')
    info_text = """
    Preprocessing Parameters:
    
    • Sample Rate: 22,050 Hz
    • Duration: 3.0 seconds
    • Samples: 66,150
    • n_fft: 2048
    • hop_length: 512
    • n_mels: 128
    • n_mfcc: 40
    • freq_min: 20 Hz
    • freq_max: 8000 Hz
    """
    ax_info.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
    
    # Row 3: Mel Spectrogram, MFCC, Spectral Features
    ax3 = fig.add_subplot(gs[2, 0:2])
    img1 = librosa.display.specshow(mel_spec, sr=22050, hop_length=512, 
                                     x_axis='time', y_axis='mel', cmap='viridis', ax=ax3)
    ax3.set_title(f'Step 3: Mel Spectrogram\nShape: {mel_spec.shape} (freq × time)', 
                  fontsize=11, fontweight='bold')
    plt.colorbar(img1, ax=ax3, format='%0.2f')
    
    ax4 = fig.add_subplot(gs[2, 2:4])
    img2 = librosa.display.specshow(mfcc[:40], sr=22050, hop_length=512,
                                     x_axis='time', cmap='coolwarm', ax=ax4)
    ax4.set_title(f'Step 4: MFCC (first 40)\nShape: {mfcc.shape} (coeffs × time)', 
                  fontsize=11, fontweight='bold')
    ax4.set_ylabel('MFCC Coefficients')
    plt.colorbar(img2, ax=ax4, format='%0.2f')
    
    ax5 = fig.add_subplot(gs[2, 4:6])
    img3 = librosa.display.specshow(spectral, sr=22050, hop_length=512,
                                     x_axis='time', cmap='plasma', ax=ax5)
    ax5.set_title(f'Step 5: Spectral Features\nShape: {spectral.shape} (features × time)', 
                  fontsize=11, fontweight='bold')
    ax5.set_ylabel('Feature Index')
    plt.colorbar(img3, ax=ax5, format='%0.2f')
    
    # Row 4: Combined 3-channel representation
    # Resize all to the same shape (128, width)
    target_height = 128
    target_width = mel_spec.shape[1]
    
    # Resize MFCC
    if mfcc.shape[0] != target_height or mfcc.shape[1] != target_width:
        mfcc_resized = librosa.util.fix_length(mfcc[:target_height], size=target_width, axis=1)
        if mfcc_resized.shape[0] < target_height:
            mfcc_resized = np.pad(mfcc_resized, ((0, target_height - mfcc_resized.shape[0]), (0, 0)))
    else:
        mfcc_resized = mfcc[:target_height]
    
    # Resize spectral
    if spectral.shape[0] != target_height or spectral.shape[1] != target_width:
        spectral_resized = librosa.util.fix_length(spectral, size=target_width, axis=1)
        if spectral_resized.shape[0] < target_height:
            spectral_resized = np.pad(spectral_resized, ((0, target_height - spectral_resized.shape[0]), (0, 0)))
        else:
            spectral_resized = spectral_resized[:target_height]
    else:
        spectral_resized = spectral[:target_height]
    
    combined = np.stack([mel_spec[:target_height], mfcc_resized, spectral_resized], axis=0)
    
    ax6 = fig.add_subplot(gs[3, 0:2])
    ax6.imshow(combined[0], aspect='auto', origin='lower', cmap='viridis')
    ax6.set_title('Channel 1: Mel Spectrogram\n(Frequency content)', 
                  fontsize=11, fontweight='bold')
    ax6.set_xlabel('Time Frames')
    ax6.set_ylabel('Frequency Bins')
    
    ax7 = fig.add_subplot(gs[3, 2:4])
    ax7.imshow(combined[1], aspect='auto', origin='lower', cmap='coolwarm')
    ax7.set_title('Channel 2: MFCC + Deltas\n(Timbral features)', 
                  fontsize=11, fontweight='bold')
    ax7.set_xlabel('Time Frames')
    ax7.set_ylabel('Feature Bins')
    
    ax8 = fig.add_subplot(gs[3, 4:6])
    ax8.imshow(combined[2], aspect='auto', origin='lower', cmap='plasma')
    ax8.set_title('Channel 3: Spectral Features\n(Spectral characteristics)', 
                  fontsize=11, fontweight='bold')
    ax8.set_xlabel('Time Frames')
    ax8.set_ylabel('Feature Bins')
    
    plt.savefig(OUTPUT_DIR / 'preprocessing_pipeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {OUTPUT_DIR / 'preprocessing_pipeline.png'}")


def create_crnn_architecture_diagram():
    """
    Create detailed CRNN architecture diagram with all layer specifications
    """
    print("Creating CRNN architecture diagram...")
    
    # Create model to extract actual parameters
    model = CRNNWithAttention(num_classes=3, input_channels=3, n_mels=128, dropout=0.3)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Title
    ax.text(5, 13.5, 'CRNN with Attention Architecture', 
            ha='center', fontsize=22, fontweight='bold')
    ax.text(5, 13, f'Total Parameters: {total_params:,} | Trainable: {trainable_params:,}', 
            ha='center', fontsize=12, style='italic')
    
    # Layer definitions with positions
    layers = [
        {
            'y': 12, 'name': 'Input', 'color': '#FFE5E5',
            'text': 'Input Tensor\n[Batch, 3, 128, 129]\n3 channels (Mel, MFCC, Spectral)\n128 frequency bins\n129 time frames'
        },
        {
            'y': 10.5, 'name': 'Conv Block 1', 'color': '#E3F2FD',
            'text': 'Conv2d(3→32, k=3, p=1)\nBatchNorm2d(32)\nReLU()\nMaxPool2d(2)\n→ [B, 32, 64, 64]\nParams: 896'
        },
        {
            'y': 9, 'name': 'Conv Block 2', 'color': '#E8F5E9',
            'text': 'Conv2d(32→64, k=3, p=1)\nBatchNorm2d(64)\nReLU()\nMaxPool2d(2)\n→ [B, 64, 32, 32]\nParams: 18,624'
        },
        {
            'y': 7.5, 'name': 'Conv Block 3', 'color': '#FFF3E0',
            'text': 'Conv2d(64→128, k=3, p=1)\nBatchNorm2d(128)\nReLU()\nMaxPool2d(2)\n→ [B, 128, 16, 16]\nParams: 73,984'
        },
        {
            'y': 6, 'name': 'TF-Attention', 'color': '#F3E5F5',
            'text': 'Temporal-Frequency Attention\nTemporal: AdaptiveAvgPool + FC\nFrequency: AdaptiveAvgPool + FC\nChannel attention weights\n→ [B, 128, 16, 16]\nParams: 16,512'
        },
        {
            'y': 4.5, 'name': 'Reshape', 'color': '#E0F2F1',
            'text': 'Reshape for RNN\nPermute: [B, 128, 16, 16] → [B, 16, 128×16]\n→ [B, 16, 2048]\ntime_steps=16, features=2048'
        },
        {
            'y': 3, 'name': 'BiGRU', 'color': '#FCE4EC',
            'text': 'Bidirectional GRU\ninput_size=2048\nhidden_size=128\nnum_layers=2\ndropout=0.3\n→ [B, 16, 256]\nParams: 3,947,520'
        },
        {
            'y': 1.5, 'name': 'Pooling', 'color': '#E1F5FE',
            'text': 'Temporal Pooling\nMean over time dimension\n→ [B, 256]\nGlobal representation'
        },
        {
            'y': 0.3, 'name': 'Output', 'color': '#C8E6C9',
            'text': 'Dropout(0.3)\nLinear(256→3)\nSoftmax\n→ [B, 3]\nParams: 771\nClasses: [background, drone, helicopter]'
        }
    ]
    
    # Draw layers
    for i, layer in enumerate(layers):
        # Draw box
        rect = FancyBboxPatch((1.5, layer['y'] - 0.5), 7, 1,
                              boxstyle="round,pad=0.05",
                              edgecolor='black', facecolor=layer['color'],
                              linewidth=2.5)
        ax.add_patch(rect)
        
        # Add text
        ax.text(5, layer['y'], layer['text'],
               ha='center', va='center', fontsize=9, family='monospace',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.5))
        
        # Draw arrows
        if i < len(layers) - 1:
            arrow = FancyArrowPatch((5, layer['y'] - 0.6), (5, layers[i+1]['y'] + 0.6),
                                   arrowstyle='->', mutation_scale=25, linewidth=3,
                                   color='darkblue')
            ax.add_patch(arrow)
    
    # Add side annotations
    # Activation functions
    ax.text(0.3, 10.5, 'Activation:\nReLU', ha='left', va='center', 
            fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    ax.text(0.3, 3, 'Activation:\ntanh (GRU)', ha='left', va='center',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Key features
    features_text = """
    Key Features:
    • Multi-scale CNN for feature extraction
    • Temporal-Frequency attention mechanism
    • Bidirectional GRU captures temporal dynamics
    • Batch normalization for stable training
    • Dropout for regularization
    • Efficient: ~4M parameters
    • Inference: ~10-20ms on GPU
    """
    ax.text(9.5, 7, features_text, ha='left', va='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.savefig(OUTPUT_DIR / 'crnn_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {OUTPUT_DIR / 'crnn_architecture.png'}")


def create_training_pipeline_flowchart():
    """
    Create training pipeline flowchart
    """
    print("Creating training pipeline flowchart...")
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis('off')
    
    # Title
    ax.text(5, 12.5, 'Training Pipeline for CRNN Model', 
            ha='center', fontsize=20, fontweight='bold')
    
    # Main pipeline
    steps = [
        {
            'y': 11.5, 'text': '1. Data Loading\n• Load WAV files from train/val directories\n• Class labels: background, drone, helicopter\n• Balanced sampling with WeightedRandomSampler',
            'color': '#FFE5E5'
        },
        {
            'y': 10, 'text': '2. Preprocessing\n• Resample to 22050 Hz\n• Extract Mel Spectrogram (128 bins)\n• Extract MFCC + Deltas (40 coeffs)\n• Extract Spectral Features',
            'color': '#E3F2FD'
        },
        {
            'y': 8.5, 'text': '3. Data Augmentation (Training Only)\n• SpecAugment (time & frequency masking)\n• Time shifting\n• Pitch shifting\n• Adding noise',
            'color': '#E8F5E9'
        },
        {
            'y': 7, 'text': '4. Batch Formation\n• Stack 3 channels [Mel, MFCC, Spectral]\n• Create batches (batch_size=32)\n• Move to GPU if available',
            'color': '#FFF3E0'
        },
        {
            'y': 5.5, 'text': '5. Forward Pass\n• Input through Conv blocks\n• Apply attention mechanism\n• BiGRU temporal modeling\n• Classification head',
            'color': '#F3E5F5'
        },
        {
            'y': 4, 'text': '6. Loss Computation\n• CrossEntropyLoss with class weights\n• Handles class imbalance\n• Compute gradients',
            'color': '#FCE4EC'
        },
        {
            'y': 2.5, 'text': '7. Optimization\n• AdamW optimizer (lr=1e-4)\n• Gradient clipping (max_norm=1.0)\n• CosineAnnealingLR scheduler\n• Weight decay=1e-4',
            'color': '#E1F5FE'
        },
        {
            'y': 1, 'text': '8. Validation & Checkpointing\n• Evaluate on validation set every epoch\n• Track: loss, accuracy, F1-score\n• Save best model based on F1-score\n• Early stopping if no improvement',
            'color': '#C8E6C9'
        }
    ]
    
    # Draw steps
    for i, step in enumerate(steps):
        rect = FancyBboxPatch((1, step['y'] - 0.6), 8, 1.2,
                              boxstyle="round,pad=0.05",
                              edgecolor='black', facecolor=step['color'],
                              linewidth=2)
        ax.add_patch(rect)
        
        ax.text(5, step['y'], step['text'],
               ha='center', va='center', fontsize=9, family='monospace')
        
        if i < len(steps) - 1:
            arrow = FancyArrowPatch((5, step['y'] - 0.7), (5, steps[i+1]['y'] + 0.7),
                                   arrowstyle='->', mutation_scale=25, linewidth=2.5,
                                   color='darkblue')
            ax.add_patch(arrow)
    
    # Training loop annotation
    loop_text = """
    Training Loop:
    • Epochs: 50-100
    • Batch size: 32
    • Learning rate: 1e-4
    • Early stopping: patience=10
    • GPU: CUDA enabled
    • Mixed precision: FP16
    """
    ax.text(0.5, 5.5, loop_text, ha='left', va='center', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
           rotation=0)
    
    plt.savefig(OUTPUT_DIR / 'training_pipeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {OUTPUT_DIR / 'training_pipeline.png'}")


def create_class_comparison_visualizations():
    """
    Create side-by-side comparison of drone, helicopter, and background samples
    showing their spectrograms and features
    """
    print("Creating class comparison visualizations...")
    
    # Load sample from each class
    train_dir = Path("data/edth_munich_dataset/data/train")
    sample_files = {
        'Drone': list((train_dir / "drone").glob("*.wav"))[0],
        'Helicopter': list((train_dir / "helicopter").glob("*.wav"))[0],
        'Background': list((train_dir / "background").glob("*.wav"))[0]
    }
    
    # Create preprocessor
    preprocessor = AudioPreprocessor(
        sample_rate=22050,
        duration=3.0,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        n_mfcc=40
    )
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle('Class Comparison: Drone vs Helicopter vs Background', 
                 fontsize=18, fontweight='bold')
    
    class_colors = {'Drone': '#FF6B6B', 'Helicopter': '#4ECDC4', 'Background': '#95E1D3'}
    
    for idx, (class_name, audio_path) in enumerate(sample_files.items()):
        print(f"  Processing {class_name}: {audio_path.name}")
        
        # Load and process
        audio = preprocessor.load_audio(str(audio_path))
        mel_spec = preprocessor.extract_mel_spectrogram(audio)
        mfcc = preprocessor.extract_mfcc(audio)
        spectral = preprocessor.extract_spectral_features(audio)
        
        # Waveform
        ax = axes[idx, 0]
        times = np.arange(len(audio)) / 22050
        ax.plot(times, audio, linewidth=0.5, color=class_colors[class_name])
        ax.set_title(f'{class_name}\nWaveform', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        
        # Mel Spectrogram
        ax = axes[idx, 1]
        img = librosa.display.specshow(mel_spec, sr=22050, hop_length=512,
                                       x_axis='time', y_axis='mel', 
                                       cmap='viridis', ax=ax)
        ax.set_title(f'{class_name}\nMel Spectrogram', fontweight='bold')
        plt.colorbar(img, ax=ax, format='%0.2f')
        
        # MFCC
        ax = axes[idx, 2]
        img = librosa.display.specshow(mfcc[:40], sr=22050, hop_length=512,
                                       x_axis='time', cmap='coolwarm', ax=ax)
        ax.set_title(f'{class_name}\nMFCC', fontweight='bold')
        ax.set_ylabel('MFCC Coeffs')
        plt.colorbar(img, ax=ax, format='%0.2f')
        
        # Spectral Features
        ax = axes[idx, 3]
        img = librosa.display.specshow(spectral, sr=22050, hop_length=512,
                                       x_axis='time', cmap='plasma', ax=ax)
        ax.set_title(f'{class_name}\nSpectral Features', fontweight='bold')
        plt.colorbar(img, ax=ax, format='%0.2f')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'class_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {OUTPUT_DIR / 'class_comparison.png'}")


def create_detailed_feature_analysis():
    """
    Create detailed analysis showing what each preprocessing step captures
    """
    print("Creating detailed feature analysis...")
    
    train_dir = Path("data/edth_munich_dataset/data/train")
    drone_sample = list((train_dir / "drone").glob("*.wav"))[0]
    
    preprocessor = AudioPreprocessor(sample_rate=22050, duration=3.0)
    audio = preprocessor.load_audio(str(drone_sample))
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    fig.suptitle('Detailed Feature Analysis - What Each Feature Captures', 
                 fontsize=16, fontweight='bold')
    
    # Waveform
    ax1 = fig.add_subplot(gs[0, :])
    times = np.arange(len(audio)) / 22050
    ax1.plot(times, audio, linewidth=0.8, color='darkblue')
    ax1.set_title('Raw Audio Waveform (Time Domain)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.95, 'Captures: Overall amplitude envelope, temporal structure',
            transform=ax1.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Mel Spectrogram
    mel_spec = preprocessor.extract_mel_spectrogram(audio)
    ax2 = fig.add_subplot(gs[1, 0])
    img = librosa.display.specshow(mel_spec, sr=22050, hop_length=512,
                                   x_axis='time', y_axis='mel', cmap='viridis', ax=ax2)
    ax2.set_title('Mel Spectrogram', fontsize=11, fontweight='bold')
    plt.colorbar(img, ax=ax2, format='%0.2f')
    ax2.text(0.02, 0.98, 'Captures:\n• Frequency content over time\n• Harmonic patterns\n• Rotor blade frequencies\n• Energy distribution',
            transform=ax2.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # MFCC
    mfcc = preprocessor.extract_mfcc(audio)
    ax3 = fig.add_subplot(gs[1, 1])
    img = librosa.display.specshow(mfcc[:40], sr=22050, hop_length=512,
                                   x_axis='time', cmap='coolwarm', ax=ax3)
    ax3.set_title('MFCC (Mel-Frequency Cepstral Coefficients)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('MFCC Coefficients')
    plt.colorbar(img, ax=ax3, format='%0.2f')
    ax3.text(0.02, 0.98, 'Captures:\n• Timbral texture\n• Spectral envelope\n• Sound source characteristics\n• Voice-like qualities',
            transform=ax3.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Spectral Features
    spectral = preprocessor.extract_spectral_features(audio)
    ax4 = fig.add_subplot(gs[1, 2])
    img = librosa.display.specshow(spectral, sr=22050, hop_length=512,
                                   x_axis='time', cmap='plasma', ax=ax4)
    ax4.set_title('Spectral Features', fontsize=11, fontweight='bold')
    plt.colorbar(img, ax=ax4, format='%0.2f')
    ax4.text(0.02, 0.98, 'Captures:\n• Spectral contrast\n• Spectral rolloff\n• Bandwidth\n• Frequency distribution shape',
            transform=ax4.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    # Combined representation
    ax5 = fig.add_subplot(gs[2, :])
    
    # Resize all features to same shape
    target_height = 128
    target_width = mel_spec.shape[1]
    
    mfcc_resized = librosa.util.fix_length(mfcc[:target_height], size=target_width, axis=1)
    if mfcc_resized.shape[0] < target_height:
        mfcc_resized = np.pad(mfcc_resized, ((0, target_height - mfcc_resized.shape[0]), (0, 0)))
    
    spectral_resized = librosa.util.fix_length(spectral, size=target_width, axis=1)
    if spectral_resized.shape[0] < target_height:
        spectral_resized = np.pad(spectral_resized, ((0, target_height - spectral_resized.shape[0]), (0, 0)))
    else:
        spectral_resized = spectral_resized[:target_height]
    
    combined = np.stack([mel_spec[:target_height], mfcc_resized, spectral_resized], axis=0)
    
    # Show all 3 channels side by side
    combined_vis = np.concatenate([combined[0], combined[1], combined[2]], axis=1)
    img = ax5.imshow(combined_vis, aspect='auto', origin='lower', cmap='viridis')
    ax5.set_title('3-Channel Combined Representation (Input to CRNN)', 
                  fontsize=12, fontweight='bold')
    ax5.set_xlabel('Time Frames (concatenated for visualization)')
    ax5.set_ylabel('Feature Bins')
    plt.colorbar(img, ax=ax5)
    
    # Add channel labels
    width = combined[0].shape[1]
    ax5.axvline(width, color='red', linewidth=2, linestyle='--')
    ax5.axvline(width*2, color='red', linewidth=2, linestyle='--')
    ax5.text(width/2, -10, 'Channel 1:\nMel Spec', ha='center', fontsize=9, fontweight='bold')
    ax5.text(width*1.5, -10, 'Channel 2:\nMFCC', ha='center', fontsize=9, fontweight='bold')
    ax5.text(width*2.5, -10, 'Channel 3:\nSpectral', ha='center', fontsize=9, fontweight='bold')
    
    plt.savefig(OUTPUT_DIR / 'feature_analysis_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {OUTPUT_DIR / 'feature_analysis_detailed.png'}")


def create_system_overview_diagram():
    """
    Create high-level system overview diagram
    """
    print("Creating system overview diagram...")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    ax.text(5, 11.5, 'Acoustic Drone Detection System - Complete Overview', 
            ha='center', fontsize=20, fontweight='bold')
    
    # Phase 1: Data Preparation
    rect1 = FancyBboxPatch((0.5, 9), 4, 2, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='#FFE5E5', linewidth=2)
    ax.add_patch(rect1)
    ax.text(2.5, 10.5, 'PHASE 1: DATA PREPARATION', ha='center', fontweight='bold', fontsize=12)
    ax.text(2.5, 10, '• Collect WAV files\n• Organize by class\n• Train/Val split (80/20)\n• Class distribution analysis',
           ha='center', va='center', fontsize=9, family='monospace')
    
    # Phase 2: Preprocessing
    rect2 = FancyBboxPatch((5.5, 9), 4, 2, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='#E3F2FD', linewidth=2)
    ax.add_patch(rect2)
    ax.text(7.5, 10.5, 'PHASE 2: PREPROCESSING', ha='center', fontweight='bold', fontsize=12)
    ax.text(7.5, 10, '• Resample: 22050 Hz\n• Extract Mel (128)\n• Extract MFCC (40)\n• Extract Spectral features',
           ha='center', va='center', fontsize=9, family='monospace')
    
    # Phase 3: Model Architecture
    rect3 = FancyBboxPatch((0.5, 6), 4, 2, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='#E8F5E9', linewidth=2)
    ax.add_patch(rect3)
    ax.text(2.5, 7.5, 'PHASE 3: CRNN MODEL', ha='center', fontweight='bold', fontsize=12)
    ax.text(2.5, 7, '• 3 Conv blocks (32→64→128)\n• TF-Attention mechanism\n• BiGRU (2 layers, 128 hidden)\n• Classification head',
           ha='center', va='center', fontsize=9, family='monospace')
    
    # Phase 4: Training
    rect4 = FancyBboxPatch((5.5, 6), 4, 2, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='#FFF3E0', linewidth=2)
    ax.add_patch(rect4)
    ax.text(7.5, 7.5, 'PHASE 4: TRAINING', ha='center', fontweight='bold', fontsize=12)
    ax.text(7.5, 7, '• AdamW optimizer\n• CrossEntropyLoss\n• Data augmentation\n• Early stopping',
           ha='center', va='center', fontsize=9, family='monospace')
    
    # Phase 5: Inference
    rect5 = FancyBboxPatch((3, 3), 4, 2, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='#F3E5F5', linewidth=2)
    ax.add_patch(rect5)
    ax.text(5, 4.5, 'PHASE 5: INFERENCE', ha='center', fontweight='bold', fontsize=12)
    ax.text(5, 4, '• Load audio file\n• Preprocess\n• Forward pass\n• Output: class + confidence',
           ha='center', va='center', fontsize=9, family='monospace')
    
    # Output
    rect6 = FancyBboxPatch((3, 0.5), 4, 1.5, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='#C8E6C9', linewidth=2)
    ax.add_patch(rect6)
    ax.text(5, 1.5, 'OUTPUT', ha='center', fontweight='bold', fontsize=12)
    ax.text(5, 1, 'Predicted Class:\nBackground / Drone / Helicopter\n+ Confidence Score',
           ha='center', va='center', fontsize=9, family='monospace')
    
    # Draw arrows
    arrows = [
        ((2.5, 9), (2.5, 8)),
        ((7.5, 9), (7.5, 8)),
        ((4.5, 7), (5.5, 7)),
        ((2.5, 6), (4.5, 5)),
        ((7.5, 6), (6.5, 5)),
        ((5, 3), (5, 2))
    ]
    
    for start, end in arrows:
        arrow = FancyArrowPatch(start, end, arrowstyle='->', 
                               mutation_scale=30, linewidth=3, color='darkblue')
        ax.add_patch(arrow)
    
    plt.savefig(OUTPUT_DIR / 'system_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {OUTPUT_DIR / 'system_overview.png'}")


def main():
    """Generate all visualizations"""
    print("\n" + "="*60)
    print("Generating Comprehensive Visualizations")
    print("="*60 + "\n")
    
    try:
        # 1. Preprocessing pipeline with examples
        create_preprocessing_flowchart_with_examples()
        
        # 2. CRNN architecture diagram
        create_crnn_architecture_diagram()
        
        # 3. Training pipeline
        create_training_pipeline_flowchart()
        
        # 4. Class comparison
        create_class_comparison_visualizations()
        
        # 5. Detailed feature analysis
        create_detailed_feature_analysis()
        
        # 6. System overview
        create_system_overview_diagram()
        
        print("\n" + "="*60)
        print("✓ All visualizations generated successfully!")
        print(f"✓ Saved to: {OUTPUT_DIR.absolute()}")
        print("="*60 + "\n")
        
        print("Generated files:")
        for img_file in sorted(OUTPUT_DIR.glob("*.png")):
            print(f"  - {img_file.name}")
        
    except Exception as e:
        print(f"\n❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
