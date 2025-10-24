# FFT-CNN-DNN Acoustic Drone Detector - Deployment Guide

## Repository for Kaggle/Remote Deployment

This repository contains the essential files to run the FFT-CNN-DNN acoustic drone detection model remotely on Kaggle or other cloud platforms.

## 🚀 Quick Start for Kaggle

### 1. Clone the Repository

```bash
!git clone https://github.com/Somnathab3/Accoustic_model_EDTH_Munich.git
%cd Accoustic_model_EDTH_Munich
```

### 2. Install Dependencies

```bash
!pip install -r requirements.txt
```

### 3. Test the Installation

```bash
!python test_fft_cnn_dnn.py
```

### 4. Run the Challenge Bot (Overnight Training)

```bash
!python challenge_bot_fft_cnn_dnn.py --max-iterations 1000 --delay 1.0
```

## 📁 Essential Files for Deployment

### Core Scripts
- `challenge_bot_fft_cnn_dnn.py` - Main challenge bot with FFT-CNN-DNN architecture
- `train_fft_cnn_dnn_quick.py` - Training script for the model
- `test_fft_cnn_dnn.py` - System test script
- `infer.py` - Inference wrapper script

### Source Code (`src/adrone/`)
```
src/adrone/
├── __init__.py
├── config.py
├── infer.py
├── data/
│   ├── __init__.py
│   └── dataset.py
├── features/
│   ├── __init__.py
│   ├── fft_processor.py
│   └── melspec.py
├── models/
│   ├── __init__.py
│   ├── fft_cnn_dnn.py
│   ├── cnn_improved.py
│   └── cnn_small.py
├── serve/
│   ├── __init__.py
│   └── challenge_handler.py
└── utils/
    ├── __init__.py
    └── audio_io.py
```

### Model Files (`models/`)
- `cnn_edth_3class_improved.pt` - Trained model checkpoint (~50MB)
- `labels_edth_3class_improved.json` - Class labels mapping
- `training_history_improved.json` - Training metrics (optional)

### Configuration
- `requirements.txt` - Python dependencies
- `README.md` - Main documentation
- `FFT_CNN_DNN_README.md` - Architecture documentation

## 🔧 Kaggle-Specific Setup

### Option A: Using Kaggle Notebook

1. Create a new Kaggle notebook
2. Enable GPU acceleration (Settings → Accelerator → GPU T4 x2)
3. Enable internet access (Settings → Internet → On)
4. Run the quick start commands above

### Option B: Using Kaggle Datasets

If the model file is too large for GitHub:

1. Upload model files as a Kaggle Dataset:
   - Go to https://www.kaggle.com/datasets
   - Click "New Dataset"
   - Upload `cnn_edth_3class_improved.pt` and `labels_edth_3class_improved.json`

2. Add the dataset to your notebook:
   ```python
   import shutil
   
   # Copy model from Kaggle dataset to working directory
   !mkdir -p models
   !cp /kaggle/input/your-dataset-name/cnn_edth_3class_improved.pt models/
   !cp /kaggle/input/your-dataset-name/labels_edth_3class_improved.json models/
   ```

### Running Overnight on Kaggle

```python
# In a Kaggle notebook cell
!python challenge_bot_fft_cnn_dnn.py \
    --max-iterations 10000 \
    --delay 1.0 \
    --model models/cnn_edth_3class_improved.pt \
    --labels models/labels_edth_3class_improved.json \
    --storage-dir /kaggle/working/challenge_results
```

**Note**: Kaggle notebooks have a 9-hour runtime limit for GPU and 12-hour for CPU.

## 📊 Model Architecture

### FFT + CNN + DNN Pipeline

1. **FFT Feature Extraction**
   - Mel spectrograms (128 mel bands)
   - MFCC coefficients
   - Spectral statistics (centroid, rolloff, bandwidth)
   - Power distribution analysis

2. **CNN Feature Learning**
   - Multi-scale convolutional blocks
   - Residual connections
   - Channel attention mechanisms
   - Batch normalization and dropout

3. **DNN Classification**
   - Dense layers: [512 → 256 → 128]
   - ReLU activation with dropout
   - Final softmax for 3-class output

### Model Performance

- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~92%
- **Classes**: drone, bird, background
- **Input**: 2-second audio clips at 16kHz

## 🎯 Challenge Bot Features

- **Automatic Result Storage**: Saves all predictions and audio samples
- **Adaptive Learning**: Tracks performance and provides recommendations
- **Performance Analytics**: Monitors accuracy, score, and class distributions
- **Continuous Operation**: Can run indefinitely with configurable delays

## 📦 Dependencies

Core dependencies (automatically installed):
- PyTorch >= 2.3.0
- librosa >= 0.10.2
- numpy >= 1.26.4
- scikit-learn >= 1.5.2
- soundfile >= 0.12.1
- requests >= 2.31.0
- tqdm >= 4.66.5

## 🔑 API Configuration

The challenge bot connects to the EDTH challenge API:
- **API URL**: https://edth.helsing.codes
- **API Token**: (embedded in script, or set as environment variable)

To use your own token:
```python
export EDTH_API_TOKEN="your-token-here"
```

## 📈 Monitoring Results

Results are stored in `challenge_results/`:
- `results.jsonl` - Detailed results for each challenge
- `statistics.json` - Overall performance statistics
- `audio_samples/` - Saved audio files for analysis

View statistics:
```bash
!cat challenge_results/statistics.json
```

## 🐛 Troubleshooting

### GPU Not Detected
```bash
!python check_gpu.py  # If available
```

### Import Errors
Make sure you're in the repository directory:
```bash
%cd Accoustic_model_EDTH_Munich
!python -c "import sys; print(sys.path)"
```

### Model Not Found
```bash
!ls -la models/
```

## 📝 File Size Considerations

- Total repository size (without model): ~500KB
- Model file (`cnn_edth_3class_improved.pt`): ~50MB
- GitHub limit: 100MB per file, 1GB per repository

If the model exceeds GitHub limits, use:
1. **Git LFS** (Large File Storage)
2. **Kaggle Dataset** (recommended for Kaggle deployment)
3. **External hosting** (Google Drive, Dropbox, etc.)

## 🔗 Useful Links

- [Original Challenge](https://edth.helsing.codes)
- [Architecture Documentation](FFT_CNN_DNN_README.md)
- [Project Status](PROJECT_STATUS.md)

## 💡 Tips for Long-Running Sessions

1. **Save checkpoints periodically** to avoid losing progress
2. **Monitor memory usage** to prevent out-of-memory errors
3. **Use adaptive delays** if hitting rate limits
4. **Check Kaggle runtime limits** and plan accordingly
5. **Export results** to Kaggle output directory for persistence

## 📧 Support

For issues or questions:
- Check `FFT_CNN_DNN_README.md` for architecture details
- Review test scripts for system validation
- Examine challenge_results for debugging

---

**Last Updated**: October 2025
**Model Version**: v1.2 (FFT-CNN-DNN Improved)
