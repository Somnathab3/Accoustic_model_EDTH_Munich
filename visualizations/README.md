# Audio Visualization Guide

## 📊 Generated Visualizations

The `visualize_audio.py` script creates comprehensive audio analysis plots showing:

### 1. **Waveform (Time Domain)**
- Shows amplitude over time
- Reveals patterns like drone propeller sounds vs background noise
- X-axis: Time (seconds)
- Y-axis: Amplitude (-1 to 1)

### 2. **Zoomed Waveform**
- First 0.1 seconds of audio in detail
- Shows fine-grained oscillation patterns
- Useful for identifying periodic signals (like propeller rotation)

### 3. **Spectrogram (STFT)**
- Short-Time Fourier Transform visualization
- Shows how frequency content changes over time
- Color intensity = energy/loudness at that frequency
- Drones typically show harmonic patterns from motors

### 4. **Mel-Spectrogram**
- Human perception-weighted frequency representation
- 64 mel bands (as used in model training)
- Shows what the neural network "sees" as input
- Warmer colors = higher energy

### 5. **Frequency Spectrum (FFT)**
- Overall frequency content of the audio
- Shows dominant frequencies
- Drones often have peaks at motor frequencies (typically 100-500 Hz)
- X-axis: Frequency (Hz, 0 to 8000 Hz)
- Y-axis: Magnitude

### 6. **Audio Statistics**
- Duration, sample rate, number of samples
- RMS (Root Mean Square) - overall loudness
- Peak amplitude
- Spectral centroid - "center of mass" of frequency spectrum
- Zero crossing rate - indicates noisiness

## 🎯 What to Look For

### Drone Audio (Class 1):
- **Harmonic patterns** in spectrogram (parallel lines)
- **Periodic waveforms** - regular oscillations
- **Higher energy** in 100-1000 Hz range (motor/propeller frequencies)
- **Stable frequency peaks** in FFT
- **Higher spectral centroid** (more high-frequency content)

### Non-Drone Audio (Class 0):
- **More random/irregular** patterns in spectrogram
- **Less periodic** waveforms
- **Broader frequency distribution**
- **More noise-like** characteristics
- **Variable energy distribution**

## 🚀 Usage Examples

### Visualize random samples from each class:
```powershell
python scripts/visualize_audio.py --num-samples 5 --compare
```

### Visualize specific audio files:
```powershell
python scripts/visualize_audio.py --specific-files data/raw/train/0/0000001.wav data/raw/train/1/0000001.wav
```

### Change output directory:
```powershell
python scripts/visualize_audio.py --output-dir my_plots --num-samples 10
```

### Just comparison plot (no individual files):
```powershell
python scripts/visualize_audio.py --num-samples 1 --compare
```

## 📁 Output Structure

```
visualizations/
├── class_comparison.png          # Side-by-side comparison of classes
├── 0/                             # Non-drone samples
│   ├── 0008342_analysis.png
│   ├── 0014678_analysis.png
│   └── ...
└── 1/                             # Drone samples
    ├── 0087223_analysis.png
    ├── 0150952_analysis.png
    └── ...
```

## 🔬 Technical Details

- **Sample Rate**: 16,000 Hz (16 kHz)
- **FFT Size**: 1024 samples
- **Hop Length**: 320 samples
- **Mel Bands**: 64 (same as model training)
- **Window**: Hamming window (default in librosa)

## 💡 Tips

1. **Compare multiple samples** from each class to see consistent patterns
2. **Look at the mel-spectrogram** - this is what the model uses for classification
3. **Check frequency spectrum peaks** - drones often have distinctive motor frequencies
4. **Use zoomed waveform** to see periodicity that might be hard to spot in full waveform

## 📊 Example Interpretations

### Good Drone Signal:
- Clear harmonic structure
- Stable frequency components
- Regular periodic waveform
- Energy concentrated in motor frequency range

### Good Non-Drone Signal:
- More random spectral pattern
- Irregular waveform
- Broader frequency distribution
- Lower overall energy or different frequency profile
