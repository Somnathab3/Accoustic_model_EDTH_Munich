#!/bin/bash
# Kaggle Setup Script for SOTA Acoustic Drone Detector
# Usage: bash setup_kaggle_sota.sh

echo "=========================================="
echo "SOTA Acoustic Drone Detector Setup"
echo "=========================================="

# Check Python version
echo "Checking Python version..."
python --version

# Check GPU availability
echo ""
echo "Checking GPU..."
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q -r requirements_kaggle.txt

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; import torchaudio; import librosa; print('✓ All packages installed successfully')"

# Check for model files
echo ""
echo "Checking for model files..."
if [ -f "models/panns_final.pt" ]; then
    echo "✓ Found: models/panns_final.pt (fully trained)"
elif [ -f "models/best_model.pt" ]; then
    echo "✓ Found: models/best_model.pt (training checkpoint)"
else
    echo "⚠️  No model found. Please download or train a model first."
fi

if [ -f "models/labels_current.json" ]; then
    echo "✓ Found: models/labels_current.json"
elif [ -f "models/labels.json" ]; then
    echo "✓ Found: models/labels.json"
else
    echo "⚠️  No labels file found."
fi

# Create results directory
echo ""
echo "Creating results directory..."
mkdir -p challenge_results
echo "✓ Created: challenge_results/"

# Summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Quick Start Commands:"
echo ""
echo "1. Run challenge bot:"
echo "   python sota_challenge_bot.py"
echo ""
echo "2. Run 100 challenges:"
echo "   python sota_challenge_bot.py --max-iterations 100 --delay 0.5"
echo ""
echo "3. Validate model:"
echo "   python validate_model.py --model models/panns_final.pt --labels models/labels_current.json --val-dir data/val"
echo ""
echo "4. Analyze results:"
echo "   python analyze_results.py"
echo ""
echo "=========================================="
