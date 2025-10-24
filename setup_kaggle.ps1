# Kaggle Setup Script for FFT-CNN-DNN Acoustic Drone Detector
# PowerShell version for Windows
# Run this script after cloning the repository on Kaggle

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "FFT-CNN-DNN Setup Script for Kaggle" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check Python version
Write-Host "`nChecking Python version..." -ForegroundColor Yellow
python --version

# Check if GPU is available
Write-Host "`nChecking GPU availability..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Install requirements
Write-Host "`nInstalling requirements..." -ForegroundColor Yellow
pip install -q -r requirements.txt

# Verify installations
Write-Host "`nVerifying installations..." -ForegroundColor Yellow
python -c @"
import torch
import librosa
import numpy
import sklearn
import soundfile
import requests
import tqdm
print('✓ All core packages installed successfully')
"@

# Check for model files
Write-Host "`nChecking for model files..." -ForegroundColor Yellow
if (Test-Path "models/cnn_edth_3class_improved.pt") {
    Write-Host "✓ Model file found: models/cnn_edth_3class_improved.pt" -ForegroundColor Green
    $ModelSize = (Get-Item "models/cnn_edth_3class_improved.pt").Length / 1MB
    Write-Host "  Size: $([math]::Round($ModelSize, 2)) MB" -ForegroundColor Green
} else {
    Write-Host "⚠ Model file NOT found: models/cnn_edth_3class_improved.pt" -ForegroundColor Red
    Write-Host "  Please upload the model file or add it as a Kaggle dataset" -ForegroundColor Red
}

if (Test-Path "models/labels_edth_3class_improved.json") {
    Write-Host "✓ Labels file found: models/labels_edth_3class_improved.json" -ForegroundColor Green
} else {
    Write-Host "⚠ Labels file NOT found: models/labels_edth_3class_improved.json" -ForegroundColor Red
}

# Create necessary directories
Write-Host "`nCreating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "challenge_results" | Out-Null
New-Item -ItemType Directory -Force -Path "challenge_results/audio_samples" | Out-Null
New-Item -ItemType Directory -Force -Path "models" | Out-Null
Write-Host "✓ Directories created" -ForegroundColor Green

# Test the system
Write-Host "`nRunning system tests..." -ForegroundColor Yellow
python test_fft_cnn_dnn.py

# Final message
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nTo run the challenge bot:"
Write-Host "  python challenge_bot_fft_cnn_dnn.py --max-iterations 1000 --delay 1.0" -ForegroundColor White
Write-Host "`nTo train the model:"
Write-Host "  python train_fft_cnn_dnn_quick.py" -ForegroundColor White
Write-Host "`nTo test inference:"
Write-Host "  python infer.py <path_to_audio.wav>" -ForegroundColor White
Write-Host ""
