# Quick Reference - Acoustic Drone Detector

## 🚀 Quick Start Commands

```powershell
# 1. Download dataset (one-time setup)
cd f:\EDTH\acoustic-drone-detector
python scripts/download_data.py

# 2. Train model
python -m src.adrone.train --config configs/train.yaml

# 3. Run inference on a single file
python infer.py path/to/audio.wav

# 4. Start API server
uvicorn src.adrone.serve.app:app --reload
```

## 📦 Project Status

| Component | Status | Notes |
|-----------|--------|-------|
| Dataset Download | ✅ Complete | 180,320 audio files processed |
| Training | 🏃 In Progress | Epoch 1/10 at 54% |
| Model | ⏳ Pending | Will be saved to `models/cnn_small.pt` |
| API Server | ⏳ Pending | Requires trained model |

## 📊 Dataset Info

- **Total Samples:** 180,320
- **Training Set:** 144,256 samples (80%)
- **Validation Set:** 36,064 samples (20%)
- **Classes:** 2 (0=non-drone, 1=drone)
- **Sample Rate:** 16,000 Hz
- **Window Duration:** 2.0 seconds

## ⚙️ Model Configuration

```yaml
Architecture: Small CNN
Input: Log-mel spectrogram (64 mels)
Batch Size: 32
Learning Rate: 0.001
Epochs: 10
Optimizer: Adam
```

## 🐛 Issues Fixed

1. ✅ **torchcodec/FFmpeg dependency error**
   - Solution: Bypassed automatic audio decoding in datasets library
   
2. ✅ **Module import errors**
   - Solution: Run as module with `python -m src.adrone.train`
   
3. ✅ **Path not found errors**  
   - Solution: Read audio from bytes field instead of file paths

## 📖 Full Documentation

See `RUNNING.md` for detailed instructions and troubleshooting.
