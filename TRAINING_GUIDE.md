# Matched Filter Bank Training - Complete Guide

## 🎯 What's Happening

You're now training **TWO models in parallel**:

1. **Baseline CRNN** (standard model from `models/crnn_combined`)
   - Input: 3 channels (HPSS: full, harmonic, percussive)
   - Architecture: CRNNWithAttention (1.69M parameters)

2. **Enhanced CRNN with Matched Filter Bank** (NEW!)
   - Input: 3 channels → Matched Bank → 9 channels (3 + 6 compressed)
   - Architecture: EnhancedCRNN + Template Bank (~1.9M parameters)
   - Added: Physics-inspired templates for low-SNR detection

## 📊 Training Status

**Currently running**: `train_and_compare_matched_bank.py`

### What You'll See

```
Epoch 1/30
├── Baseline CRNN Training
│   ├── Train Loss, Train Acc
│   ├── Val Loss, Val Acc
│   └── Best model saved
│
├── Enhanced CRNN Training  
│   ├── Curriculum SNR: 30 dB → 0 dB over epochs
│   ├── Template margin loss tracking
│   ├── Train Loss, Train Acc
│   ├── Val Loss, Val Acc
│   └── Best model saved
│
└── ... (repeats for 30 epochs)
```

After training:
- SNR robustness evaluation (30, 20, 15, 10, 5, 0 dB)
- Comparison plots
- Final summary

## 📁 Output Files

All results saved to: `models/matched_bank_comparison/`

### During Training
- `config.json` - Training configuration
- `baseline_history.json` - Baseline training curves
- `enhanced_history.json` - Enhanced training curves
- `baseline_crnn.pt` - Best baseline checkpoint
- `enhanced_crnn.pt` - Best enhanced checkpoint

### After Training
- `training_comparison.png` - Side-by-side training curves
- `snr_comparison.png` - Accuracy/recall/F1 vs SNR
- `snr_results.json` - Detailed SNR evaluation metrics
- `summary.json` - Final performance summary

## 🔍 Monitor Progress

### Option 1: Live monitoring script
```bash
python monitor_training.py
```
This will check progress every 30 seconds and display:
- Current epoch
- Latest train/val loss and accuracy
- Auto-generated progress plots

### Option 2: Check logs manually
```bash
# In PowerShell
Get-Content models\matched_bank_comparison\baseline_history.json
Get-Content models\matched_bank_comparison\enhanced_history.json
```

### Option 3: View intermediate plots
```
models/matched_bank_comparison/baseline_progress.png
models/matched_bank_comparison/enhanced_progress.png
```

## ⏱️ Expected Training Time

With your dataset:
- **Train**: 893 samples (27 batches)
- **Val**: 306 samples (10 batches)
- **Epochs**: 30

Estimated time per epoch:
- Baseline: ~2-3 minutes
- Enhanced: ~3-4 minutes (extra processing for matched bank)

**Total**: ~2-3 hours for complete comparison

## 📈 What to Expect

### Baseline CRNN (from previous training)
- Val Acc: ~85-90% (clean data)
- Struggles at low SNR (<10 dB)

### Enhanced CRNN (with Matched Filter Bank)
- Val Acc: **Similar or slightly better** on clean data
- **Significantly better** at low SNR (0-5 dB)
- Expected improvement: +20-30% recall at 0 dB SNR

### Key Metrics to Watch

1. **Validation Accuracy**: Should be similar for both models on clean data
2. **SNR Robustness**: Enhanced model should shine at low SNR
3. **Template Loss**: Watch `loss_breakdown` in enhanced training
4. **Convergence**: Enhanced may take longer to converge initially

## 🎓 Understanding the Results

### Training Curves (`training_comparison.png`)
- **Similar curves**: Both models work on clean data
- **Faster convergence**: One model learns better
- **Lower val loss**: Better generalization

### SNR Curves (`snr_comparison.png`)
- **Flat line at high SNR**: Both models perform well
- **Divergence at low SNR**: Enhanced model maintains performance
- **+X% annotations**: Improvement at each SNR level

### Expected Pattern
```
Accuracy (%)
100 ├────────────────────────  Both models (clean)
    │                      ╲
 80 │                       ╲
    │                        ╲─── Baseline drops
 60 │                         ╲
    │                          ╲
 40 │                           ╲
    │                            ╲
    │                             ═══ Enhanced stays high!
  0 └────────────────────────────────
   30    20    15    10     5     0  SNR (dB)
```

## 🔧 Troubleshooting

### Training stops early
- Check GPU memory (reduce batch size if needed)
- Check disk space (checkpoints can be large)

### Poor performance
- Check if data loaded correctly: "Class distribution" should show balanced classes
- Verify HPSS is enabled: `use_hpss=True`

### Enhanced model not improving
- This is normal initially! Templates need ~5-10 epochs to adapt
- Check curriculum is enabled: `use_curriculum=True`
- Watch template margin loss: should decrease over time

## 🚀 After Training

### 1. Check Summary
```bash
cat models/matched_bank_comparison/summary.json
```

### 2. Load Models for Inference
```python
import torch
from src.models.enhanced_models_with_bank import create_enhanced_crnn
from train_and_compare_matched_bank import EnhancedCRNN

# Load enhanced model
enhanced_backbone = EnhancedCRNN(num_classes=3, input_channels=9)
enhanced_model = create_enhanced_crnn(enhanced_backbone, compression=6)

checkpoint = torch.load('models/matched_bank_comparison/enhanced_crnn.pt')
enhanced_model.load_state_dict(checkpoint['model_state_dict'])
enhanced_model.eval()

# Use for inference
output = enhanced_model(spectrogram)
```

### 3. Compare on Your Test Set
```python
# Evaluate both models
baseline_acc = evaluate(baseline_model, your_test_data)
enhanced_acc = evaluate(enhanced_model, your_test_data)

print(f"Improvement: {enhanced_acc - baseline_acc:+.2f}%")
```

## 📊 Expected Results Summary

Based on matched filtering theory and similar implementations:

| Metric | Baseline | Enhanced | Expected Gain |
|--------|----------|----------|---------------|
| **Clean Val Acc** | 85-90% | 85-92% | 0-5% |
| **Recall @ 0 dB** | 40-50% | **70-80%** | **+30-40%** |
| **FPR @ 95% TPR** | 8-10% | **3-5%** | **-50%** |
| **Parameters** | 1.69M | 1.9M | +12% |
| **Inference Time** | 10ms | 11ms | +10% |

## 🎉 Next Steps

1. ✅ **Wait for training to complete** (~2-3 hours)
2. ✅ **Check results**: `models/matched_bank_comparison/summary.json`
3. ✅ **Visualize**: Open `training_comparison.png` and `snr_comparison.png`
4. ✅ **Deploy**: Use enhanced model if improvement > 10%
5. ✅ **Iterate**: Tune compression, templates, curriculum if needed

## 📞 Questions?

- **Training too slow?** Reduce `--epochs` to 10 for quick test
- **Need more improvement?** Try `--compression 3` (less compression)
- **Want faster inference?** Try `--compression 12` (more compression)

---

**Current Status**: Training in progress... ⏳

Check `monitor_training.py` for live updates!
