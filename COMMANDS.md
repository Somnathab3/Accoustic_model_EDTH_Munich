# Quick Command Reference

## Training Commands (Use These!)

### Option 1: Simple Quick Start (Recommended)
```powershell
# Quick test (2 epochs, small data)
python quick_train.py --quick-test

# Full training with PANNs (recommended)
python quick_train.py --model panns --epochs 50

# Full training with CRNN (lightweight)
python quick_train.py --model crnn --epochs 60

# Full training with Transformer (best accuracy)
python quick_train.py --model transformer --epochs 40
```

### Option 2: Direct Training Script
```powershell
# Use CORRECT paths (relative from project root):
python train_sota_model.py --train-dir data/edth_munich_dataset/data/train --val-dir data/edth_munich_dataset/data/val

# Quick test
python train_sota_model.py --train-dir data/edth_munich_dataset/data/train --val-dir data/edth_munich_dataset/data/val --quick-test

# Full training with custom settings
python train_sota_model.py --train-dir data/edth_munich_dataset/data/train --val-dir data/edth_munich_dataset/data/val --model-type panns --epochs 50 --batch-size 32
```

## Inference Commands

### Validate Model Performance
```powershell
# Validate trained model on validation set
python validate_model.py --model models/panns/panns_final.pt --labels models/panns/labels.json --val-dir data/edth_munich_dataset/data/val

# Or validate the best checkpoint
python validate_model.py --model models/panns/best_model.pt --labels models/panns/labels.json --val-dir data/edth_munich_dataset/data/val

# Hide error details
python validate_model.py --model models/panns/panns_final.pt --labels models/panns/labels.json --val-dir data/edth_munich_dataset/data/val --no-show-errors
```

### Single File
```powershell
python sota_inference.py models/panns/panns_final.pt models/panns/labels.json path/to/audio.wav
```

### Challenge Bot
```powershell
# Run with trained model
python sota_challenge_bot.py --model models/panns/panns_final.pt --labels models/panns/labels.json --max-iterations 100

# Run indefinitely
python sota_challenge_bot.py --model models/panns/panns_final.pt --labels models/panns/labels.json
```

## Expected Training Time

| Model | Epochs | Time (CPU) | Time (GPU) |
|-------|--------|------------|------------|
| CRNN | 60 | ~4-6 hours | ~30-45 min |
| PANNs | 50 | ~6-8 hours | ~45-60 min |
| Transformer | 40 | ~8-12 hours | ~60-90 min |

*Times assume ~5000 samples per class in training set*

## Model Selection Guide

### Use **CRNN** if:
- You need fast training and inference
- CPU-only deployment
- Edge devices (Raspberry Pi, etc.)
- Memory constraints

### Use **PANNs** if:
- You want best balance of accuracy and speed
- Production deployment
- GPU available but not required
- **This is the recommended default**

### Use **Transformer** if:
- You need maximum accuracy
- Complex/noisy environments
- GPU available
- Inference speed is not critical

## Troubleshooting

### Path Issues
❌ **WRONG**: `--train-dir acoustic-drone-detector\data\edth_munich_dataset\data\train`
✅ **CORRECT**: `--train-dir data/edth_munich_dataset/data/train`

Always use paths relative to the project root (where train_sota_model.py is located).

### Out of Memory
```powershell
# Reduce batch size
python train_sota_model.py ... --batch-size 16

# Use smaller model
python train_sota_model.py ... --model-type crnn
```

### Training Too Slow
```powershell
# Use GPU
python train_sota_model.py ... --device cuda

# Reduce epochs for testing
python train_sota_model.py ... --epochs 10
```

### Model Not Learning
```powershell
# Increase learning rate
python train_sota_model.py ... --lr 3e-4

# Use focal loss for imbalanced data
python train_sota_model.py ... --use-focal-loss --focal-gamma 2.0

# Disable HPSS if causing issues
python train_sota_model.py ... --no-use-hpss
```

## After Training

Your trained model will be in `models/<model_type>/`:
- `<model_type>_final.pt` - Model weights for deployment
- `best_model.pt` - Best checkpoint (includes optimizer state)
- `labels.json` - Class mapping
- `training_history.json` - Training metrics
- `training_curves.png` - Loss/accuracy plots
- `config.json` - Training configuration

## Next Steps

1. **Train**: `python quick_train.py --quick-test` (test) or `python quick_train.py --model panns` (full)
2. **Test**: `python sota_inference.py models/panns/panns_final.pt models/panns/labels.json <audio.wav>`
3. **Deploy**: `python sota_challenge_bot.py --model models/panns/panns_final.pt --labels models/panns/labels.json`
