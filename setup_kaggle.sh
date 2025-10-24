#!/bin/bash
# Kaggle Setup Script for FFT-CNN-DNN Acoustic Drone Detector
# Run this script after cloning the repository on Kaggle

echo "========================================"
echo "FFT-CNN-DNN Setup Script for Kaggle"
echo "========================================"

# Check Python version
echo ""
echo "Checking Python version..."
python --version

# Check if GPU is available
echo ""
echo "Checking GPU availability..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Install requirements
echo ""
echo "Installing requirements..."
pip install -q -r requirements.txt

# Verify installations
echo ""
echo "Verifying installations..."
python -c "
import torch
import librosa
import numpy
import sklearn
import soundfile
import requests
import tqdm
print('✓ All core packages installed successfully')
"

# Check for model files
echo ""
echo "Checking for model files..."
if [ -f "models/cnn_edth_3class_improved.pt" ]; then
    echo "✓ Model file found: models/cnn_edth_3class_improved.pt"
    MODEL_SIZE=$(du -h models/cnn_edth_3class_improved.pt | cut -f1)
    echo "  Size: $MODEL_SIZE"
else
    echo "⚠ Model file NOT found: models/cnn_edth_3class_improved.pt"
    echo "  Please upload the model file or add it as a Kaggle dataset"
fi

if [ -f "models/labels_edth_3class_improved.json" ]; then
    echo "✓ Labels file found: models/labels_edth_3class_improved.json"
else
    echo "⚠ Labels file NOT found: models/labels_edth_3class_improved.json"
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p challenge_results
mkdir -p challenge_results/audio_samples
mkdir -p models
echo "✓ Directories created"

# Test the system
echo ""
echo "Running system tests..."
python test_fft_cnn_dnn.py

# Final message
echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To run the challenge bot:"
echo "  python challenge_bot_fft_cnn_dnn.py --max-iterations 1000 --delay 1.0"
echo ""
echo "To train the model:"
echo "  python train_fft_cnn_dnn_quick.py"
echo ""
echo "To test inference:"
echo "  python infer.py <path_to_audio.wav>"
echo ""
