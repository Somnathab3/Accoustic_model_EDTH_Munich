# SOTA Acoustic Drone Detector - Deployment Guide

## 🚀 State-of-the-Art Model for Kaggle/Cloud Deployment

This repository contains a production-ready SOTA (State-of-the-Art) acoustic drone detection model with advanced preprocessing, multiple architectures, and optimized inference.

## Quick Start for Kaggle

### 1. Clone Repository
```bash
!git clone https://github.com/Somnathab3/edth-acoustic-drone-detector.git
%cd edth-acoustic-drone-detector
```

### 2. Install Dependencies
```bash
!pip install -q -r requirements.txt
```

### 3. Download Model (Auto-detect)
```python
# Model auto-detection: uses best available
# - panns_final.pt (fully trained, best performance)
# - best_model.pt (training checkpoint, good performance)
```

### 4. Run Challenge Bot
```bash
# Fast mode (recommended)
!python sota_challenge_bot.py --delay 0.5 --max-iterations 1000

# Or let it run continuously
!python sota_challenge_bot.py
```

## 📊 Model Architecture

### Three-Tier System

#### 1. **CRNN with Attention** (~1.5M params)
- Fast inference (~50ms)
- Good for edge deployment
- 3 conv blocks + BiGRU + Temporal-Frequency attention

#### 2. **PANNs CNN14** (~5M params) ⭐ **Recommended**
- Balanced speed/accuracy
- AudioSet-inspired architecture
- 4 conv blocks with adaptive pooling

#### 3. **Audio Transformer** (~20M params)
- Highest accuracy
- Patch embedding + 12 transformer blocks
- Slower but most powerful

### Current Performance
- **Validation Accuracy**: 86.67%
- **Macro F1**: 0.8640
- **Classes**: background (100%), drone (63%), helicopter (97%)

## 🎯 Advanced Features

### Preprocessing Pipeline
- **Log-Mel Spectrograms**: 96 mels, 16kHz, 2.0s windows
- **HPSS**: Harmonic-Percussive Source Separation (3-channel input)
- **SpecAugment**: Time/frequency masking
- **SNR Curriculum**: Progressive noise degradation

### Data Augmentation
- Background noise mixing with SNR control
- Time/pitch augmentation (Doppler simulation)
- Mixup augmentation (α=0.2)
- Dynamic augmentation pipeline

### Training Features
- **AdamW optimizer** with cosine LR schedule
- **Label smoothing** (ε=0.05)
- **Class-balanced loss** with inverse frequency weighting
- **Early stopping** (patience=10, Macro-F1 metric)
- **Gradient clipping** (norm=1.0)

## 📁 Essential Files

### Core Scripts
```
├── sota_challenge_bot.py          # Main challenge bot ⭐
├── train_sota_model.py            # Training script
├── sota_inference.py              # Inference module
├── validate_model.py              # Model validation
├── analyze_results.py             # Results analyzer
└── quick_train.py                 # Quick training wrapper
```

### Source Code
```
src/adrone/
├── preprocessing/
│   └── audio_transforms.py        # Advanced preprocessing
├── models/
│   └── acoustic_models.py         # Three model architectures
├── data/
│   └── acoustic_dataset.py        # Dataset loader
├── training/
│   └── losses.py                  # Loss functions & schedulers
├── evaluation/
│   └── metrics.py                 # Evaluation metrics
└── serve/
    └── challenge_handler.py       # API client
```

### Model Files
```
models/
├── panns_final.pt                 # Fully trained (preferred)
├── best_model.pt                  # Training checkpoint
├── labels_current.json            # Class mappings ⭐
└── config.json                    # Training config
```

## 🎮 Usage Examples

### Example 1: Run Challenge Bot (Simple)
```bash
!python sota_challenge_bot.py
```
- Auto-detects best model
- Outputs to `challenge_results/results.csv`
- Smart timing with duplicate detection

### Example 2: Custom Configuration
```bash
!python sota_challenge_bot.py \
    --model models/panns_final.pt \
    --labels models/labels_current.json \
    --delay 0.5 \
    --max-iterations 1000 \
    --csv challenge_results/my_results.csv
```

### Example 3: Quick Inference
```python
from sota_inference import AcousticDroneClassifier

# Initialize
classifier = AcousticDroneClassifier(
    model_path='models/panns_final.pt',
    labels_path='models/labels_current.json'
)

# Classify
prediction, confidence, probs = classifier.classify('audio.wav')
print(f"{prediction}: {confidence:.2%}")
# Output: drone: 82.34%
```

### Example 4: Batch Processing
```python
import glob
from sota_inference import AcousticDroneClassifier

classifier = AcousticDroneClassifier()
audio_files = glob.glob('samples/*.wav')

for audio_file in audio_files:
    pred, conf, probs = classifier.classify(audio_file)
    print(f"{audio_file}: {pred} ({conf:.2%})")
```

### Example 5: Validate Model
```bash
!python validate_model.py \
    --model models/panns_final.pt \
    --labels models/labels_current.json \
    --val-dir data/edth_munich_dataset/data/val
```

## 🔧 Kaggle-Specific Setup

### Enable GPU
```python
# Settings → Accelerator → GPU T4 x2 (recommended)
```

### Enable Internet
```python
# Settings → Internet → On (required for API access)
```

### Install Dependencies
```bash
!pip install -q torch torchaudio librosa scikit-learn soundfile
```

### Check GPU
```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

## 📊 Challenge Bot Features

### Smart Timing Strategy
1. **Pre-Sync Mode**: Checks every 3s until first score received
2. **Synced Mode**: Waits 100s cycle after score synchronization
3. **Duplicate Detection**: Handles "already submitted" automatically
4. **Adaptive Backoff**: Exponential backoff on errors

### Output Files
```
challenge_results/
├── results.csv              # Main CSV (continuously updated) ⭐
├── results.jsonl            # Detailed JSONL
├── statistics.json          # Performance stats
└── audio_samples/           # Downloaded samples
    ├── correct/
    └── incorrect/
```

### CSV Format
```csv
iteration,timestamp,challenge_id,predicted,actual,correct,confidence,score_awarded,inference_time,total_time
1,2025-10-24 22:44:17,2b05de25...,drone,drone,True,0.8234,150,0.0747,0.1970
2,2025-10-24 22:44:18,41d23e80...,background,drone,False,0.7546,100,0.0725,0.1575
```

### Real-Time Monitoring
```
[1] ✓ Predicted: drone      | Actual: drone      | Conf: 0.823 | Score: +150 | Time: 0.17s
🎯 First score received! Now synced with server timing...
[2] ✗ Predicted: background | Actual: drone      | Conf: 0.755 | Score: +100 | Time: 0.16s
     [background:0.755 | drone:0.146 | helicopter:0.099]
[3] ✓ Predicted: helicopter | Actual: helicopter | Conf: 0.943 | Score: +150 | Time: 0.15s
```

## 📈 Performance Analysis

### View Results
```bash
!python analyze_results.py
```

Output:
```
📊 OVERALL PERFORMANCE
  Correct:  84/100 (84.0%)
  Wrong:    16/100 (16.0%)
  Total Score: 12,400
  Avg Score/Challenge: 124.0

🎯 PREDICTION DISTRIBUTION
  drone       : 45 (45.0%)
  background  : 30 (30.0%)
  helicopter  : 25 (25.0%)

📈 PER-CLASS ACCURACY
  Class        Correct  Total    Accuracy
  background    30       30       100.0%
  drone         25       40        62.5%
  helicopter    29       30        96.7%

⏱️  TIMING STATISTICS
  Inference Time: 0.074s (mean)
  Total Time: 0.197s (mean)
```

## 🚀 Overnight Kaggle Run

### Setup for Long Run
```bash
# In Kaggle notebook with GPU enabled
!git clone https://github.com/Somnathab3/edth-acoustic-drone-detector.git
%cd edth-acoustic-drone-detector
!pip install -q -r requirements.txt

# Run overnight (Kaggle limit: 9-12 hours)
!python sota_challenge_bot.py --delay 0.5 --max-iterations 10000
```

### Save Results to Kaggle Output
```python
import shutil

# Copy results to Kaggle output (persists after session)
!mkdir -p /kaggle/working/final_results
!cp -r challenge_results/* /kaggle/working/final_results/

# Download results
from IPython.display import FileLink
FileLink('/kaggle/working/final_results/results.csv')
```

## 🎓 Training Your Own Model

### Quick Training
```bash
!python quick_train.py
```

### Custom Training
```bash
!python train_sota_model.py \
    --train-dir data/edth_munich_dataset/data/train \
    --val-dir data/edth_munich_dataset/data/val \
    --model-type panns \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.0001
```

### Architecture Options
- `crnn` - Fast, lightweight (1.5M params)
- `panns` - Balanced (5M params) ⭐ **Recommended**
- `transformer` - Powerful (20M params)

## 🔑 API Configuration

- **URL**: https://edth.helsing.codes
- **Token**: Embedded in scripts (or set as environment variable)

Custom token:
```bash
export EDTH_API_TOKEN="your-token-here"
```

## 📦 Dependencies

Core (automatically installed):
```
torch>=2.0.0
torchaudio>=2.0.0
librosa>=0.10.0
numpy>=1.24.0
scikit-learn>=1.3.0
soundfile>=0.12.0
```

## 🐛 Troubleshooting

### GPU Not Available
```python
!nvidia-smi  # Check GPU
import torch
print(torch.cuda.is_available())  # Should be True
```

### Import Errors
```bash
!pip install --upgrade torch torchaudio
```

### Model Not Found
```bash
!ls -la models/
# Should see: panns_final.pt or best_model.pt
```

### Low Accuracy
```bash
# Validate model first
!python validate_model.py --model models/panns_final.pt --labels models/labels_current.json --val-dir data/val
```

## 📊 Expected Performance

### Good Session (1000 challenges)
- Runtime: ~30-60 minutes
- Accuracy: 80-85%
- Total Score: 120,000-130,000
- Avg time: 0.2s per challenge

### Model Validation
- Overall: 86.67% accuracy
- Background: 100% (perfect)
- Helicopter: 96.67% (excellent)
- Drone: 63.33% (needs improvement)

## 🎯 Quick Commands Reference

```bash
# Run challenge bot (default)
!python sota_challenge_bot.py

# Run 100 challenges fast
!python sota_challenge_bot.py --max-iterations 100 --delay 0.3

# Analyze results
!python analyze_results.py

# Validate model
!python validate_model.py --model models/panns_final.pt --labels models/labels_current.json --val-dir data/val

# Train new model
!python quick_train.py
```

## 📝 File Sizes

- Repository (without models): ~2MB
- PANNs model: ~20MB
- Transformer model: ~80MB
- Training data: Not included (download separately)

## 🔗 Useful Links

- Challenge: https://edth.helsing.codes
- Repository: https://github.com/Somnathab3/edth-acoustic-drone-detector
- Documentation: See `CHALLENGE_READY.md`, `SOTA_README.md`

## 💡 Pro Tips

1. **Start with validation** to know baseline performance
2. **Monitor CSV file** for patterns and improvements
3. **Use GPU** for 10-20x faster inference
4. **Enable internet** for API access
5. **Save results** to Kaggle output for persistence
6. **Run overnight** to maximize score
7. **Check timing** - adjust delay if seeing many duplicates

## 📧 Support

See documentation:
- `CHALLENGE_READY.md` - Quick start guide
- `SOTA_README.md` - Architecture details
- `TIMING_STRATEGY_EXPLAINED.md` - Timing strategy
- `QUICK_COMMANDS.md` - Command reference

---

**Last Updated**: October 2025  
**Model Version**: SOTA v1.0 (PANNs-based)  
**Status**: ✅ Production Ready
