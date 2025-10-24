# Kaggle Setup Script for SOTA Acoustic Drone Detector
# Usage: .\setup_kaggle_sota.ps1

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "SOTA Acoustic Drone Detector Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check Python version
Write-Host "`nChecking Python version..." -ForegroundColor Yellow
python --version

# Check GPU availability
Write-Host "`nChecking GPU..." -ForegroundColor Yellow
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else `"CPU`"}')"

# Install dependencies
Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
pip install -q -r requirements_kaggle.txt

# Verify installation
Write-Host "`nVerifying installation..." -ForegroundColor Yellow
python -c "import torch; import torchaudio; import librosa; print('✓ All packages installed successfully')"

# Check for model files
Write-Host "`nChecking for model files..." -ForegroundColor Yellow
if (Test-Path "models/panns_final.pt") {
    Write-Host "✓ Found: models/panns_final.pt (fully trained)" -ForegroundColor Green
} elseif (Test-Path "models/best_model.pt") {
    Write-Host "✓ Found: models/best_model.pt (training checkpoint)" -ForegroundColor Green
} else {
    Write-Host "⚠️  No model found. Please download or train a model first." -ForegroundColor Yellow
}

if (Test-Path "models/labels_current.json") {
    Write-Host "✓ Found: models/labels_current.json" -ForegroundColor Green
} elseif (Test-Path "models/labels.json") {
    Write-Host "✓ Found: models/labels.json" -ForegroundColor Green
} else {
    Write-Host "⚠️  No labels file found." -ForegroundColor Yellow
}

# Create results directory
Write-Host "`nCreating results directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "challenge_results" | Out-Null
Write-Host "✓ Created: challenge_results/" -ForegroundColor Green

# Summary
Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Quick Start Commands:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Run challenge bot:" -ForegroundColor White
Write-Host "   python sota_challenge_bot.py" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Run 100 challenges:" -ForegroundColor White
Write-Host "   python sota_challenge_bot.py --max-iterations 100 --delay 0.5" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Validate model:" -ForegroundColor White
Write-Host "   python validate_model.py --model models/panns_final.pt --labels models/labels_current.json --val-dir data/val" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Analyze results:" -ForegroundColor White
Write-Host "   python analyze_results.py" -ForegroundColor Gray
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
