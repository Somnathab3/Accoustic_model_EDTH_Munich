# Model Improvement Implementation Summary

## Overview
Implemented comprehensive improvements to fix underperforming models (PANNS, Transformer, SNN) based on CRNN's strong performance (Val F1 ≈ 0.972).

## Baseline Performance (BEFORE)
| Model | Val F1 | Drone Recall | Status |
|-------|--------|--------------|--------|
| CRNN | 0.972 | ~0.95 | ✓ BEST (Teacher) |
| PANNS | 0.860 | 0.651 | ⚠️ Low drone recall |
| Transformer | 0.647 | 0.408 | ❌ Underfit |
| SNN | 0.598 | 0.214 | ❌ Severe underfit |

## Improvements Implemented

### 1. Loss Functions (src/adrone/training/losses.py)
- ✅ **Focal Loss** with γ=2.0 for hard examples (drones)
- ✅ **Class-Balanced Loss** using effective number of samples
- ✅ **Knowledge Distillation Loss** (T=3-4, α=0.5) from CRNN teacher
- ✅ Updated CombinedLoss to support all options

### 2. Data Augmentation (src/adrone/preprocessing/audio_transforms.py)
- ✅ **Enhanced pitch shift** (±3-5%) for RPM variations
- ✅ **Narrowband notch filters** for microphone characteristics
- ✅ **Band-limiting filters** (80-150 Hz low, 6-8 kHz high)
- ✅ **SNR curriculum** (10+ dB → 0-5 dB over epochs)
- ✅ Kept existing: SpecAugment, mixup, time shift

### 3. Balanced Sampling (src/adrone/data/acoustic_dataset.py)
- ✅ **WeightedRandomSampler** for equal class representation per batch
- ✅ **get_samples_per_class()** method for class-balanced loss
- ✅ Updated create_dataloaders() with use_balanced_sampler flag

### 4. Training Infrastructure (train_improved_models.py)
- ✅ **Knowledge Distillation** with CRNN as teacher
- ✅ **Per-class recall tracking** (critical for drone monitoring)
- ✅ **Dual checkpoint saving**: best F1 + best drone recall
- ✅ **train_epoch_with_kd()** for KD training loop
- ✅ Support for all loss types (focal, class-balanced, label smoothing)

### 5. Model-Specific Configs

#### PANNS (configs/train_panns_improved.yaml)
- Loss: Focal (γ=2.0)
- Epochs: 40
- Progressive unfreezing: freeze→top 1/3→all
- KD: T=3.0, α=0.5
- Balanced sampling: ✓
- Target: F1 > 0.92, drone recall > 0.85

#### Transformer (configs/train_transformer_improved.yaml)
- Loss: Focal (γ=2.0)
- Epochs: 50
- Weight decay: 0.05 (high regularization)
- KD: T=4.0, α=0.5 (CRITICAL for small data)
- Balanced sampling: ✓
- Target: F1 > 0.90, drone recall > 0.80

#### SNN (configs/train_snn.yaml - updated)
- Loss: Focal (γ=2.0)
- Epochs: 50
- Hybrid architecture: conv front-end + LIF
- Timesteps: 4 → 10 (increased)
- LR: 0.0001 → 0.002 (higher for SNN)
- Batch size: 32 → 16 (memory)
- KD: T=3.0, α=0.5 (CRITICAL)
- Balanced sampling: ✓
- Target: F1 > 0.85, drone recall > 0.75

## Training Scripts

### Individual Models
```powershell
# PANNS (~2-3 hours)
.\train_panns_improved.ps1

# Transformer (~3-4 hours)
.\train_transformer_improved.ps1

# SNN (~3-4 hours)
.\train_snn_improved.ps1
```

### All Models (Sequential)
```powershell
# Total: ~8-11 hours
.\train_all_improved.ps1
```

## Key Innovations

### 1. Focal Loss for Minority Class
**Problem**: Conservative predictions on "drone" class
**Solution**: Focal loss (γ=2) down-weights easy examples, focuses on hard drone samples

### 2. Knowledge Distillation from CRNN
**Problem**: Small dataset, models underfit (especially Transformer/SNN)
**Solution**: Transfer CRNN's discrimination via soft targets (T=3-4)

### 3. Balanced Sampling
**Problem**: Class imbalance leads to background bias
**Solution**: WeightedRandomSampler ensures equal representation per batch

### 4. Rotor-Specific Augmentations
**Problem**: Overfitting to training acoustics
**Solution**: 
- Pitch shift (±5%) simulates RPM variations
- Notch/band-limit simulates different microphones
- SNR curriculum hardens against noise

### 5. Dual Checkpoint Strategy
**Problem**: Best F1 ≠ best drone recall
**Solution**: Save both checkpoints, use best drone recall for production if needed

## Expected Improvements

| Model | Current F1 | Current Drone Recall | Target F1 | Target Drone Recall |
|-------|-----------|---------------------|-----------|-------------------|
| PANNS | 0.860 | 0.651 | >0.92 | >0.85 |
| Transformer | 0.647 | 0.408 | >0.90 | >0.80 |
| SNN | 0.598 | 0.214 | >0.85 | >0.75 |

## Usage

### Training
```powershell
# Train all models
.\train_all_improved.ps1

# Or individual
.\train_panns_improved.ps1
```

### Evaluation
```python
# Compare all models
python compare_models.py

# Use best model in challenge
python sota_challenge_bot.py --model models/panns_improved/panns_final.pt
```

### Output Structure
```
models/
  panns_improved/
    best_model.pt              # Best macro F1
    best_drone_recall.pt       # Best drone recall
    panns_final.pt             # Final model
    labels.json
    training_history.json
    training_curves.png
  
  transformer_improved/
    [same structure]
  
  snn_improved/
    [same structure]
```

## Technical Details

### Loss Computation Order
1. **Class-Balanced** (if enabled): Effective number weighting
2. **Focal** (if enabled): (1-p)^γ * CE, γ=2.0
3. **Label Smoothing** (always): ε=0.05
4. **KD** (if enabled): α*KL(student||teacher) + (1-α)*hard_loss

### Sampling Strategy
- Training: WeightedRandomSampler with class weights
- Validation: Sequential (no sampling)
- Weights: Inverse frequency, normalized

### Augmentation Pipeline
```
Audio → Pitch Shift → Notch Filter → Band Limit → Noise (curriculum) 
     ↓
Spectrogram → SpecAugment (time/freq masks)
```

### KD Temperature Selection
- PANNS: T=3.0 (moderate softening)
- Transformer: T=4.0 (more softening for stability)
- SNN: T=3.0 (balance between soft/hard)

## Files Modified/Created

### Modified
- `src/adrone/training/losses.py` - Added ClassBalancedLoss, DistillationLoss
- `src/adrone/training/__init__.py` - Updated exports
- `src/adrone/preprocessing/audio_transforms.py` - Added NarrowbandNotchFilter, BandLimitingFilter
- `src/adrone/data/acoustic_dataset.py` - Added balanced sampling, get_samples_per_class()
- `configs/train_snn.yaml` - Updated with hybrid SNN settings

### Created
- `train_improved_models.py` - New training script with KD
- `configs/train_panns_improved.yaml` - PANNS config
- `configs/train_transformer_improved.yaml` - Transformer config
- `train_panns_improved.ps1` - PANNS training script
- `train_transformer_improved.ps1` - Transformer training script
- `train_snn_improved.ps1` - SNN training script
- `train_all_improved.ps1` - Master training script
- `IMPROVEMENTS_SUMMARY.md` - This file

## Next Steps

1. **Train models**:
   ```powershell
   .\train_all_improved.ps1
   ```

2. **Compare results**:
   ```python
   python compare_models.py
   ```

3. **Test best model**:
   ```python
   python sota_challenge_bot.py --model models/panns_improved/panns_final.pt
   ```

4. **If drone recall still low**:
   - Use `best_drone_recall.pt` instead of `best_model.pt`
   - Increase focal gamma to 3.0
   - Adjust balanced sampling weights

5. **Monitor during training**:
   - Watch "DRONE RECALL" metric in training logs
   - Early stop if drone recall plateaus
   - Adjust hyperparameters based on per-class metrics

## Troubleshooting

### If OOM (Out of Memory)
- Reduce batch size (SNN: 16→8, others: 32→16)
- Reduce num_workers (4→2)
- Disable mixed_precision for SNN

### If drone recall not improving
- Increase focal_gamma (2.0→3.0)
- Check class distribution in data
- Verify balanced_sampling is active
- Increase KD alpha (0.5→0.7) for more teacher influence

### If validation loss increasing but train loss decreasing
- Increase weight_decay
- Reduce learning rate
- More augmentation (increase p values)
- Stronger label smoothing (0.05→0.1)

## References
- Focal Loss: Lin et al., 2017
- Knowledge Distillation: Hinton et al., 2015
- Class-Balanced Loss: Cui et al., 2019
- SpecAugment: Park et al., 2019
