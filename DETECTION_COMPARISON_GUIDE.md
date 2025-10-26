# Detection Performance Comparison Guide

## Purpose

This script compares the detection performance of **Baseline CRNN** vs **LIGO-Modified Matched Filter Bank** on the entire training and validation datasets.

## Quick Start

### While training is running:

Wait until you see at least one checkpoint saved (e.g., after epoch 1-2), then run:

```powershell
python compare_detection_performance.py
```

This will evaluate both models on the full dataset and generate comprehensive comparison reports.

### After training completes:

```powershell
python compare_detection_performance.py --enhanced models/matched_bank_comparison/enhanced_crnn.pt
```

## What It Does

### 1. Evaluates Both Models
- **Baseline CRNN**: Your existing trained model
- **LIGO-Modified**: New model with matched filter bank

### 2. On Both Datasets
- **Training set**: 893 samples
- **Validation set**: 306 samples

### 3. Comprehensive Metrics
- Overall accuracy
- Per-class precision, recall, F1
- Confusion matrices
- Inference time statistics

## Output Files

All results saved to `detection_comparison_results/`:

### Visualizations
1. **`detection_performance_comparison.png`**
   - Side-by-side bar charts
   - Overall metrics + per-class breakdown
   - Clear improvement indicators

2. **`confusion_matrices_comparison.png`**
   - Heatmaps showing prediction patterns
   - Normalized percentages
   - Easy to spot improvements

3. **`inference_time_comparison.png`**
   - Average inference time per batch
   - Shows computational overhead

### Data Files
1. **`detection_comparison_table.csv`**
   - Detailed metrics in spreadsheet format
   - Easy to share with team

2. **`full_comparison_results.json`**
   - Complete raw data
   - All predictions and probabilities
   - For further analysis

## Example Usage

### Basic comparison (auto-detects checkpoints):
```powershell
python compare_detection_performance.py
```

### Specify custom paths:
```powershell
python compare_detection_performance.py `
    --baseline "models/crnn_combined/best_model.pt" `
    --enhanced "models/matched_bank_comparison/enhanced_crnn.pt" `
    --data-dir "data/combined_dataset"
```

### Change batch size for faster evaluation:
```powershell
python compare_detection_performance.py --batch-size 64
```

## Expected Output

```
================================================================================
LOADING BASELINE CRNN
================================================================================
✓ Loaded baseline model: 1,687,107 parameters

================================================================================
LOADING LIGO-MODIFIED MATCHED FILTER BANK MODEL
================================================================================
✓ Loaded enhanced model: 1,900,000 parameters
  Parameter overhead: 212,893 (12.6%)

================================================================================
EVALUATING ON TRAINING SET
================================================================================

Baseline CRNN (Train) Results:
  Overall Accuracy: 95.23%
  Macro Precision:  94.87%
  Macro Recall:     94.65%
  Macro F1:         94.76%
  
  Per-Class Performance:
    drone           - P: 96.12% | R: 95.58% | F1: 95.85% | N:  294
    helicopter      - P: 94.23% | R: 93.71% | F1: 93.97% | N:  302
    background      - P: 94.26% | R: 94.65% | F1: 94.46% | N:  297

  Inference Time: 12.45 ± 2.31 ms

LIGO-Modified (Train) Results:
  Overall Accuracy: 96.78%
  Macro Precision:  96.45%
  Macro Recall:     96.32%
  Macro F1:         96.38%
  
  Per-Class Performance:
    drone           - P: 97.45% | R: 97.28% | F1: 97.36% | N:  294
    helicopter      - P: 95.87% | R: 95.36% | F1: 95.61% | N:  302
    background      - P: 96.03% | R: 96.32% | F1: 96.17% | N:  297

  Inference Time: 15.67 ± 2.89 ms

================================================================================
EVALUATING ON VALIDATION SET
================================================================================
...

================================================================================
SUMMARY
================================================================================

Validation Set Performance:
  Baseline Accuracy:     88.56%
  LIGO-Modified Accuracy: 91.18%
  Improvement:            +2.62%

  Baseline Recall:        87.34%
  LIGO-Modified Recall:   90.21%
  Improvement:            +2.87%

✓ All results saved to: detection_comparison_results
```

## Interpreting Results

### Key Metrics to Watch:

1. **Overall Accuracy Improvement**
   - Goal: +2-5% improvement
   - Indicates better general detection

2. **Per-Class Recall**
   - Most important for detection systems
   - Shows if model misses fewer drones/helicopters

3. **Inference Time**
   - Expect ~20-30% slower (due to matched bank)
   - Still real-time capable (<20ms per batch)

4. **Confusion Matrix**
   - Look at diagonal values (correct predictions)
   - Check off-diagonal for common mistakes

### What Good Results Look Like:

✅ **Overall accuracy**: +2-5% improvement  
✅ **Drone recall**: +3-8% improvement (most important!)  
✅ **Helicopter recall**: +2-6% improvement  
✅ **Inference time**: <20ms per batch  

### Decision Criteria:

- **Deploy LIGO-Modified** if:
  - Accuracy improvement > 2%
  - Recall improvement > 3%
  - Inference time acceptable for your application

- **Stick with Baseline** if:
  - Improvement < 1%
  - Inference time too slow for real-time use

## Monitoring During Training

Run this script **periodically** during training to see improvement:

```powershell
# After epoch 5
python compare_detection_performance.py

# After epoch 10
python compare_detection_performance.py

# After epoch 20
python compare_detection_performance.py

# Final comparison
python compare_detection_performance.py
```

Each run overwrites previous results in `detection_comparison_results/`.

## Troubleshooting

### Error: "Enhanced checkpoint not found"
**Solution**: Training hasn't saved a checkpoint yet. Wait for epoch 1 to complete.

### Error: "Data loader not available"
**Solution**: Check that `src.adrone.data.acoustic_dataset` is importable:
```powershell
python -c "from src.adrone.data.acoustic_dataset import create_dataloaders; print('OK')"
```

### Slow evaluation
**Solution**: Increase batch size:
```powershell
python compare_detection_performance.py --batch-size 64
```

### Out of memory
**Solution**: Decrease batch size or use CPU:
```powershell
python compare_detection_performance.py --batch-size 16
```

## Advanced: Compare with Different Checkpoints

```powershell
# Compare baseline with enhanced from epoch 10
python compare_detection_performance.py `
    --baseline "models/crnn_combined/best_model.pt" `
    --enhanced "models/matched_bank_comparison/enhanced_crnn_epoch10.pt"

# Compare different baseline models
python compare_detection_performance.py `
    --baseline "models/old_baseline.pt" `
    --enhanced "models/matched_bank_comparison/enhanced_crnn.pt"
```

## Tips

1. **Run early**: Check after 5-10 epochs to see if training is on track
2. **Save results**: Rename output folder to keep history:
   ```powershell
   Rename-Item detection_comparison_results detection_comparison_epoch10
   ```
3. **Compare multiple runs**: Run with different enhanced checkpoints
4. **Share results**: The CSV and PNG files are easy to share with team

## Next Steps After Comparison

1. **Review visualizations**: Open PNG files to see clear comparisons
2. **Check CSV table**: Import into Excel for detailed analysis
3. **Analyze errors**: Look at confusion matrix for improvement areas
4. **Test on real audio**: Use `test_real_audio_inference.py` with best model

---

**Quick command to run now:**
```powershell
python compare_detection_performance.py
```

Results will be in `detection_comparison_results/` folder! 🚀
