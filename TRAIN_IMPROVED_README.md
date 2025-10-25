# Quick Start: Training Improved Models

## 🚀 Quick Train (All Models)

Simply run:
```powershell
.\train_all_improved.ps1
```

This will train all three models sequentially (8-11 hours total).

## 📋 What's Been Improved

All three underperforming models now have:

### ✅ Core Fixes
1. **Focal Loss (γ=2.0)** - Up-weights drone class mistakes
2. **Balanced Sampling** - Equal bg/drone/heli per batch
3. **Knowledge Distillation** - Learn from CRNN teacher (F1=0.972)
4. **Enhanced Augmentations** - Pitch shift, notch filters, band-limiting

### ✅ Model-Specific
- **PANNS**: Progressive unfreezing, attentive pooling
- **Transformer**: High regularization, longer training (50 epochs)
- **SNN**: Hybrid architecture, 10 timesteps, higher LR (0.002)

## 🎯 Expected Results

| Model | Before F1 | Before Drone Recall | Target F1 | Target Drone Recall |
|-------|-----------|-------------------|-----------|-------------------|
| PANNS | 0.860 | 0.651 | **>0.92** | **>0.85** |
| Transformer | 0.647 | 0.408 | **>0.90** | **>0.80** |
| SNN | 0.598 | 0.214 | **>0.85** | **>0.75** |

## 📁 Output Structure

After training:
```
models/
  panns_improved/
    best_model.pt          ← Use this (best F1)
    best_drone_recall.pt   ← Or this (best drone recall)
    panns_final.pt
    labels.json
    training_history.json
  
  transformer_improved/
    [same files]
  
  snn_improved/
    [same files]
```

## 🔧 Individual Training

If you prefer to train models one at a time:

### PANNS (~2-3 hours)
```powershell
.\train_panns_improved.ps1
```

### Transformer (~3-4 hours)
```powershell
.\train_transformer_improved.ps1
```

### SNN (~3-4 hours)
```powershell
.\train_snn_improved.ps1
```

## 📊 Monitor Training

Watch for these key metrics in the output:
- **DRONE RECALL** - Most important! Should increase steadily
- **Val Macro F1** - Overall performance
- **Per-class Recall** - All three classes

Good training looks like:
```
Epoch 10/50:
  Train Loss: 0.3245 | Train Acc: 0.8923
  Val Loss:   0.2876 | Val Acc:   0.9012
  Val Macro F1: 0.8956
  Per-class Recall: {'background': 0.9234, 'drone': 0.8456, 'helicopter': 0.9178}
  ⚠️  DRONE RECALL: 0.8456  ← Should be >0.80
```

## 🧪 After Training

### 1. Compare Models
```python
python compare_models.py
```

### 2. Test Best Model
```python
python sota_challenge_bot.py --model models/panns_improved/panns_final.pt
```

### 3. If Drone Recall Still Low
Use the best_drone_recall checkpoint instead:
```python
python sota_challenge_bot.py --model models/panns_improved/best_drone_recall.pt
```

## ⚙️ Customization

Edit the training scripts if needed:

### Increase Focal Loss Strength
In `train_*_improved.ps1`, change:
```powershell
--focal-gamma 2.0  # → 3.0 for more aggressive
```

### Reduce Batch Size (if OOM)
```powershell
--batch-size 32  # → 16 for PANNS/Transformer
--batch-size 16  # → 8 for SNN
```

### Adjust KD Temperature
```powershell
--kd-temperature 3.0  # → 4.0 for softer targets (more exploration)
```

## 🐛 Troubleshooting

### Out of Memory
```powershell
# Reduce batch size and workers
--batch-size 16 --num-workers 2
```

### Drone Recall Not Improving
1. Check if balanced_sampling is active (should see "Using balanced sampler" in logs)
2. Increase focal_gamma to 3.0
3. Verify teacher model exists: `models/crnn_combined/crnn_final.pt`

### Training Too Slow
- Reduce num_workers if disk I/O is slow
- Use --epochs 30 for faster iteration
- Start with PANNS (fastest) to validate setup

## 📚 Documentation

- **IMPROVEMENTS_SUMMARY.md** - Full technical details
- **configs/train_*_improved.yaml** - Model configurations
- **train_improved_models.py** - Training script source

## ✨ Key Features

### Focal Loss
Automatically focuses on hard examples (drones) by down-weighting easy ones (background).

### Knowledge Distillation
Student models learn from CRNN's soft predictions, capturing nuanced decision boundaries.

### Balanced Sampling
Every batch has equal representation of background/drone/helicopter, preventing background bias.

### Enhanced Augmentations
- **Pitch shift (±5%)**: Simulates RPM variations
- **Notch filters**: Different microphone characteristics
- **Band-limiting**: 100-7500 Hz typical sensor response

## 🎓 What Makes This Work

1. **CRNN as Teacher** - Transfer knowledge from best model (F1=0.972)
2. **Focal Loss** - Fix conservative drone predictions
3. **Balanced Data** - Combat class imbalance
4. **Rotor Augmentations** - Generalize to different flight conditions

The combination of these four is the key to improving drone recall!

## ⏱️ Time Estimates

- **PANNS**: 2-3 hours (40 epochs, batch 32)
- **Transformer**: 3-4 hours (50 epochs, batch 32)
- **SNN**: 3-4 hours (50 epochs, batch 16)
- **Total**: 8-11 hours for all three

Start the training before bed or during work hours!

---

**Ready to train?**
```powershell
.\train_all_improved.ps1
```

Good luck! 🚁🎯
