"""
Visualize audio files with waveforms, spectrograms, and frequency analysis.
Usage: python scripts/visualize_audio.py --audio-dir data/raw/train --num-samples 5
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import librosa
import librosa.display
from pathlib import Path
import random

def plot_audio_analysis(audio_path, sr=16000, output_dir=None):
    """
    Create a comprehensive visualization of an audio file.
    Shows: waveform, spectrogram, mel-spectrogram, and frequency spectrum.
    """
    # Load audio
    y, sr = sf.read(audio_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    
    # Resample if needed
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Audio Analysis: {Path(audio_path).name}', fontsize=16, fontweight='bold')
    
    # 1. Waveform (Time Domain)
    ax1 = plt.subplot(3, 2, 1)
    time = np.linspace(0, len(y) / sr, len(y))
    ax1.plot(time, y, linewidth=0.5, alpha=0.8, color='blue')
    ax1.set_xlabel('Time (s)', fontsize=10)
    ax1.set_ylabel('Amplitude', fontsize=10)
    ax1.set_title('Waveform (Time Domain)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, time[-1]])
    
    # 2. Zoomed Waveform (first 0.1 seconds)
    ax2 = plt.subplot(3, 2, 2)
    zoom_samples = int(0.1 * sr)
    time_zoom = time[:zoom_samples]
    ax2.plot(time_zoom, y[:zoom_samples], linewidth=1, color='darkblue')
    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.set_ylabel('Amplitude', fontsize=10)
    ax2.set_title('Waveform (Zoomed - First 0.1s)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Spectrogram (STFT)
    ax3 = plt.subplot(3, 2, 3)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax3, cmap='viridis')
    ax3.set_title('Spectrogram (STFT)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency (Hz)', fontsize=10)
    ax3.set_xlabel('Time (s)', fontsize=10)
    plt.colorbar(img, ax=ax3, format='%+2.0f dB')
    
    # 4. Mel-Spectrogram
    ax4 = plt.subplot(3, 2, 4)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=1024, hop_length=320)
    S_db_mel = librosa.power_to_db(S, ref=np.max)
    img_mel = librosa.display.specshow(S_db_mel, sr=sr, x_axis='time', y_axis='mel', ax=ax4, cmap='magma')
    ax4.set_title('Mel-Spectrogram (64 mels)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Mel Frequency', fontsize=10)
    ax4.set_xlabel('Time (s)', fontsize=10)
    plt.colorbar(img_mel, ax=ax4, format='%+2.0f dB')
    
    # 5. Frequency Spectrum (FFT)
    ax5 = plt.subplot(3, 2, 5)
    fft = np.fft.fft(y)
    magnitude = np.abs(fft)
    frequency = np.linspace(0, sr, len(magnitude))
    # Only plot positive frequencies
    half_len = len(frequency) // 2
    ax5.plot(frequency[:half_len], magnitude[:half_len], linewidth=0.5, color='red')
    ax5.set_xlabel('Frequency (Hz)', fontsize=10)
    ax5.set_ylabel('Magnitude', fontsize=10)
    ax5.set_title('Frequency Spectrum (FFT)', fontsize=12, fontweight='bold')
    ax5.set_xlim([0, sr // 2])
    ax5.grid(True, alpha=0.3)
    
    # 6. Audio Statistics
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    
    # Calculate statistics
    duration = len(y) / sr
    rms = np.sqrt(np.mean(y**2))
    peak = np.max(np.abs(y))
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = np.mean(zcr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sc_mean = np.mean(spectral_centroid)
    
    stats_text = f"""
    Audio Statistics:
    ━━━━━━━━━━━━━━━━━━━━━━
    Duration: {duration:.2f} seconds
    Sample Rate: {sr} Hz
    Samples: {len(y):,}
    
    Amplitude:
    • RMS: {rms:.4f}
    • Peak: {peak:.4f}
    • Min: {np.min(y):.4f}
    • Max: {np.max(y):.4f}
    
    Frequency:
    • Spectral Centroid: {sc_mean:.2f} Hz
    • Zero Crossing Rate: {zcr_mean:.4f}
    
    File: {Path(audio_path).name}
    Label: {Path(audio_path).parent.name}
    """
    
    ax6.text(0.1, 0.5, stats_text, transform=ax6.transAxes, 
             fontsize=11, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save or show
    if output_dir:
        output_path = Path(output_dir) / f"{Path(audio_path).stem}_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_comparison(audio_files, sr=16000, output_path=None):
    """
    Plot multiple audio files side by side for comparison.
    """
    n_files = len(audio_files)
    fig, axes = plt.subplots(n_files, 3, figsize=(18, 4*n_files))
    
    if n_files == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Audio Comparison Analysis', fontsize=16, fontweight='bold')
    
    for idx, audio_path in enumerate(audio_files):
        # Load audio
        y, file_sr = sf.read(audio_path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if file_sr != sr:
            y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
        
        label = Path(audio_path).parent.name
        filename = Path(audio_path).stem
        
        # Waveform
        time = np.linspace(0, len(y) / sr, len(y))
        axes[idx, 0].plot(time, y, linewidth=0.5, alpha=0.8)
        axes[idx, 0].set_title(f'Waveform - Label: {label}', fontweight='bold')
        axes[idx, 0].set_xlabel('Time (s)')
        axes[idx, 0].set_ylabel('Amplitude')
        axes[idx, 0].grid(True, alpha=0.3)
        axes[idx, 0].text(0.02, 0.98, filename, transform=axes[idx, 0].transAxes,
                          fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', 
                          facecolor='white', alpha=0.8))
        
        # Mel-Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=1024, hop_length=320)
        S_db = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', 
                                       ax=axes[idx, 1], cmap='magma')
        axes[idx, 1].set_title(f'Mel-Spectrogram - Label: {label}', fontweight='bold')
        plt.colorbar(img, ax=axes[idx, 1], format='%+2.0f dB')
        
        # Frequency Spectrum
        fft = np.fft.fft(y)
        magnitude = np.abs(fft)
        frequency = np.linspace(0, sr, len(magnitude))
        half_len = len(frequency) // 2
        axes[idx, 2].plot(frequency[:half_len], magnitude[:half_len], linewidth=0.8)
        axes[idx, 2].set_title(f'Frequency Spectrum - Label: {label}', fontweight='bold')
        axes[idx, 2].set_xlabel('Frequency (Hz)')
        axes[idx, 2].set_ylabel('Magnitude')
        axes[idx, 2].set_xlim([0, sr // 2])
        axes[idx, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison: {output_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize audio files with waveforms and spectrograms')
    parser.add_argument('--audio-dir', type=str, default='data/raw/train',
                        help='Directory containing audio files')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to visualize per class')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                        help='Output directory for plots')
    parser.add_argument('--compare', action='store_true',
                        help='Create comparison plots between classes')
    parser.add_argument('--specific-files', nargs='+', type=str,
                        help='Specific audio files to visualize')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if args.specific_files:
        # Visualize specific files
        print(f"Visualizing {len(args.specific_files)} specific files...")
        for audio_file in args.specific_files:
            if Path(audio_file).exists():
                plot_audio_analysis(audio_file, output_dir=output_dir)
            else:
                print(f"Warning: File not found: {audio_file}")
    else:
        # Scan directory for audio files
        audio_dir = Path(args.audio_dir)
        
        # Collect samples by class
        samples_by_class = {}
        for class_dir in audio_dir.iterdir():
            if class_dir.is_dir():
                audio_files = list(class_dir.glob('*.wav'))
                if audio_files:
                    samples_by_class[class_dir.name] = random.sample(
                        audio_files, min(args.num_samples, len(audio_files))
                    )
        
        print(f"Found {len(samples_by_class)} classes")
        for class_name, files in samples_by_class.items():
            print(f"  Class '{class_name}': {len(files)} samples")
        
        # Create individual visualizations
        print("\nCreating individual visualizations...")
        for class_name, files in samples_by_class.items():
            class_output_dir = output_dir / class_name
            class_output_dir.mkdir(exist_ok=True, parents=True)
            
            for audio_file in files:
                plot_audio_analysis(audio_file, output_dir=class_output_dir)
        
        # Create comparison plots
        if args.compare and len(samples_by_class) > 1:
            print("\nCreating comparison plots...")
            comparison_files = []
            for class_name, files in samples_by_class.items():
                if files:
                    comparison_files.append(files[0])  # Take one sample from each class
            
            if comparison_files:
                plot_comparison(comparison_files, output_path=output_dir / 'class_comparison.png')
    
    print(f"\n✅ Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()
