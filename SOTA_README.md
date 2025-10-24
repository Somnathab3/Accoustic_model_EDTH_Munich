# State-of-the-Art Acoustic Drone Detection 🎯

A clean, production-ready implementation of deep learning models for acoustic drone classification using best practices from peer-reviewed research.

## 🌟 Key Features

### Advanced Audio Preprocessing
- **Log-Mel Spectrograms** (96 mels, 16kHz sampling, 2.0s windows)
- **Harmonic-Percussive Source Separation** (HPSS) for emphasizing rotor harmonics
- **Comprehensive Augmentation Pipeline**:
  - SpecAugment (time/frequency masking)
  - Background noise mixing with SNR curriculum
  - Time stretching and pitch shifting
  - Mixup augmentation

### State-of-the-Art Models (3 Tiers)

#### 🥇 Tier S: Audio Transformer
- **Best accuracy** on complex/noisy scenes
- Patch-based attention mechanism
- Captures long-range dependencies
- ~20-30M parameters
- **Recommended for**: Maximum accuracy, GPU available

#### 🥈 Tier A: PANNs CNN14 (Default)
- **Balanced** accuracy and speed
- AudioSet-inspired architecture
- ~5-10M parameters
- **Recommended for**: Production deployment

#### 🥉 Tier B: CRNN with Attention
- **Edge-light** baseline
- Temporal-frequency attention
- ~1-2M parameters
- **Recommended for**: CPU-only inference, embedded systems

### Training Features
- **AdamW optimizer** with cosine LR decay and warmup
- **Label smoothing** for better calibration
- **Class-balanced loss** for imbalanced datasets
- **Focal loss** option for hard examples
- **SNR curriculum learning** (clean → noisy)
- **Macro-F1 monitoring** for minority classes
- **Early stopping** with patience
- **Comprehensive metrics**: Accuracy, Macro F1, ROC-AUC, PR-AUC, ECE

### Evaluation & Calibration
- Per-class F1, precision, recall
- ROC-AUC and PR-AUC for each class
- Confusion matrix
- **Expected Calibration Error (ECE)**
- **Temperature scaling** for calibration
- Balanced accuracy

### Inference
- Fast single-file classification
- Batch processing
- **Streaming mode** with:
  - Exponential smoothing
  - Hysteresis for stable alerts
  - Sliding window inference

## 📦 Installation

```bash
# Clone repository
git clone <your-repo-url>
cd acoustic-drone-detector

# Install dependencies
pip install torch torchaudio librosa scikit-learn numpy tqdm matplotlib requests
```

## 🚀 Quick Start

### 1. Train the Model

```bash
# Train PANNs CNN14 (recommended)
python train_sota_model.py \
  --train-dir data/edth_munich_dataset/data/train \
  --val-dir data/edth_munich_dataset/data/val \
  --model-type panns \
  --epochs 50 \
  --batch-size 32 \
  --output-dir models

# Quick test (2 epochs, small data)
python train_sota_model.py \
  --train-dir data/edth_munich_dataset/data/train \
  --val-dir data/edth_munich_dataset/data/val \
  --quick-test

# Train other models
python train_sota_model.py --model-type crnn ...      # CRNN baseline
python train_sota_model.py --model-type transformer ... # Audio Transformer
```

**Training Options:**
- `--model-type`: Choose `crnn`, `panns`, or `transformer`
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-4)
- `--use-focal-loss`: Use focal loss for class imbalance
- `--label-smoothing`: Label smoothing factor (default: 0.05)
- `--patience`: Early stopping patience (default: 10)
- `--quick-test`: Fast test with 2 epochs and small data

### 2. Inference

```bash
# Single file inference
python sota_inference.py models/panns_final.pt models/labels.json path/to/audio.wav

# Output:
# Prediction: drone
# Confidence: 0.9543
# 
# All probabilities:
#   background: 0.0234
#   drone: 0.9543
#   helicopter: 0.0223
```

### 3. Challenge Bot

```bash
# Run challenge bot with trained model
python sota_challenge_bot.py \
  --model models/panns_final.pt \
  --labels models/labels.json \
  --max-iterations 100 \
  --delay 1.0

# Options:
#   --model: Path to model checkpoint
#   --labels: Path to labels JSON
#   --max-iterations: Max challenges (default: infinite)
#   --delay: Seconds between challenges (default: 1.0)
```

## 📊 Expected Performance

Based on the methodology:

| Model | Accuracy | Macro F1 | Inference Time | Size |
|-------|----------|----------|----------------|------|
| CRNN-Attention | ~85-90% | ~0.82-0.88 | ~50ms (CPU) | ~2MB |
| PANNs CNN14 | ~90-95% | ~0.88-0.93 | ~80ms (CPU) | ~20MB |
| Audio Transformer | ~92-97% | ~0.90-0.95 | ~150ms (GPU) | ~80MB |

## 🏗️ Project Structure

```
acoustic-drone-detector/
├── src/adrone/
│   ├── preprocessing/
│   │   └── audio_transforms.py      # Preprocessing & augmentation
│   ├── models/
│   │   └── acoustic_models.py       # Model architectures
│   ├── data/
│   │   └── acoustic_dataset.py      # Dataset & dataloaders
│   ├── training/
│   │   └── losses.py                # Loss functions & utilities
│   ├── evaluation/
│   │   └── metrics.py               # Evaluation & calibration
│   └── serve/
│       └── challenge_handler.py     # API client & result storage
│
├── train_sota_model.py               # Training script
├── sota_inference.py                 # Inference module
├── sota_challenge_bot.py             # Challenge bot
│
├── models/                           # Trained models (created after training)
│   ├── panns_final.pt
│   ├── labels.json
│   ├── training_history.json
│   └── training_curves.png
│
└── data/
    └── edth_munich_dataset/
        └── data/
            ├── train/                # Training data
            │   ├── background/
            │   ├── drone/
            │   └── helicopter/
            └── val/                  # Validation data
                ├── background/
                ├── drone/
                └── helicopter/
```

## 🎓 Methodology & References

This implementation is based on peer-reviewed research:

### Preprocessing
- **SpecAugment**: Park et al., "SpecAugment: A Simple Data Augmentation Method for ASR" (2019)
- **HPSS**: Librosa implementation for harmonic-percussive separation
- **Mixup**: Zhang et al., "mixup: Beyond Empirical Risk Minimization" (2018)

### Models
- **CRNN-Attention**: Mu et al., "Temporal-Frequency Attention for ESC" (2021)
- **PANNs**: Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks" (2020)
- **Audio Transformer**: Gong et al., "AST: Audio Spectrogram Transformer" (2021)

### Training
- **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection" (2017)
- **Label Smoothing**: Szegedy et al., "Rethinking the Inception Architecture" (2016)
- **Cosine Schedule**: Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts" (2017)

### Calibration
- **Temperature Scaling**: Guo et al., "On Calibration of Modern Neural Networks" (2017)

### Domain-Specific
- Acoustic drone detection literature (Casabianca et al., Al-Emadi et al.)

## 🔧 Advanced Usage

### Custom Training

```python
from adrone.preprocessing import AudioPreprocessor, AugmentationPipeline
from adrone.data.acoustic_dataset import create_dataloaders
from adrone.models.acoustic_models import create_model

# Custom preprocessing
preprocessor = AudioPreprocessor(
    sample_rate=16000,
    n_mels=96,
    use_hpss=True
)

# Custom augmentation
augmentation = AugmentationPipeline(
    use_time_pitch=True,
    use_noise=True,
    use_spec_augment=True
)

# Create dataloaders
train_loader, val_loader, class_weights = create_dataloaders(
    train_dir='data/train',
    val_dir='data/val',
    preprocessor=preprocessor,
    augmentation_pipeline=augmentation,
    batch_size=32
)

# Create model
model = create_model(
    model_type='panns',
    num_classes=3,
    input_channels=3  # with HPSS
)
```

### Custom Inference

```python
from sota_inference import AcousticDroneClassifier

# Initialize classifier
classifier = AcousticDroneClassifier(
    model_path='models/panns_final.pt',
    labels_path='models/labels.json',
    device='cuda'
)

# Classify single file
prediction, confidence, all_probs = classifier.classify('audio.wav')

# Batch classification
results = classifier.classify_batch(['audio1.wav', 'audio2.wav'])
```

### Streaming Inference

```python
from sota_inference import AcousticDroneClassifier, StreamingClassifier

# Create base classifier
classifier = AcousticDroneClassifier(...)

# Create streaming classifier
streaming = StreamingClassifier(
    classifier=classifier,
    smoothing_tau=3.0,          # 3 second smoothing
    alert_threshold=0.9,        # Raise alert at 90% confidence
    clear_threshold=0.3,        # Clear alert below 30%
    min_consecutive=2           # Require 2 consecutive detections
)

# Process audio windows
for audio_window in audio_stream:
    result = streaming.update(audio_window)
    
    if result['alert_active']:
        print(f"ALERT: {result['smoothed_prediction']} detected!")
```

## 📈 Model Performance by Class

The models are designed to handle:

1. **Background** (ambient noise, traffic, wind)
   - Lower energy, intermittent activity
   - Trained to reject false positives

2. **Drone** (small multirotor UAVs)
   - High-frequency blade-pass fundamentals
   - Narrower spectral peaks
   - Electric motor noise signature

3. **Helicopter** (full-size rotorcraft)
   - Lower blade-pass frequency
   - Richer low-order harmonics
   - Stronger rotor whomp

The **HPSS** preprocessing helps separate these by emphasizing:
- **Harmonic**: Rotor blade-pass frequencies
- **Percussive**: Motor noise and transients

## 🐛 Troubleshooting

### Model predicts only one class
- **Cause**: Class imbalance or insufficient training
- **Solution**: 
  - Use `--use-class-weights` flag
  - Use `--use-focal-loss` flag
  - Train for more epochs
  - Check data distribution

### Low accuracy
- **Cause**: Insufficient data or overfitting
- **Solution**:
  - Enable all augmentations
  - Reduce model complexity (use `crnn` instead of `transformer`)
  - Collect more training data
  - Adjust `--label-smoothing`

### Out of memory during training
- **Cause**: Batch size too large for GPU
- **Solution**:
  - Reduce `--batch-size` (try 16 or 8)
  - Use smaller model (`crnn`)
  - Enable gradient checkpointing (requires code modification)

### Inference too slow
- **Cause**: Model too large or CPU inference
- **Solution**:
  - Use `crnn` model (~50ms/sample on CPU)
  - Enable GPU inference
  - Disable HPSS (`--use-hpss False`)

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- EDTH Munich Dataset creators
- Helsing AI for the challenge platform
- Open source audio processing libraries (librosa, torchaudio)
- Research community for published methodologies

## 📧 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Built with ❤️ using PyTorch and best practices from peer-reviewed research**
