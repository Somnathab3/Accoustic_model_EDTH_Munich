# Evaluation Report: Drone-Detection-Dataset

## Dataset Information
- **Location**: `F:\EDTH\acoustic-drone-detector\data\Drone-detection-dataset`
- **Total Samples**: 90 audio files
- **Classes**: 3 (background, drone, helicopter)
- **Distribution**: 30 samples per class (perfectly balanced)
- **Audio Format**: WAV, 44.1kHz sample rate, 10 seconds duration
- **Nomenclature**: ✅ Correct - Files named as `CLASSNAME_###.wav` where CLASSNAME ∈ {BACKGROUND, DRONE, HELICOPTER}

## Label Verification
✅ **Labels are correctly extracted**: 
- Filenames follow pattern: `BACKGROUND_001.wav`, `DRONE_001.wav`, `HELICOPTER_001.wav`
- Label extraction logic: `filename.split('_')[0].lower()` works correctly
- All 90 files have valid labels

## Model Evaluation Results

### Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | Drone Precision | Drone Recall | Drone F1 |
|-------|----------|-----------|--------|----------|-----------------|--------------|----------|
| **CRNN (Combined)** | 33.33% | 11.11% | 33.33% | 16.67% | 0.00% | 0.00% | 0.00% |
| **CRNN (Hard Mining)** | 33.33% | 11.11% | 33.33% | 16.67% | 0.00% | 0.00% | 0.00% |
| **PANNS (Combined)** | 33.33% | 11.11% | 33.33% | 16.67% | 0.00% | 0.00% | 0.00% |

### Critical Findings

⚠️ **SEVERE GENERALIZATION FAILURE**

All three models exhibit identical catastrophic failure patterns:

1. **100% Helicopter Prediction Bias**
   - All 90 samples (background, drone, AND helicopter) are predicted as "helicopter"
   - Models have collapsed to a single-class predictor
   
2. **Low Confidence Scores**
   - CRNN models: ~47-48% confidence (barely above random chance)
   - PANNS model: ~46-47% confidence
   - Very narrow confidence range (std: 0.004-0.008)
   - Models are uncertain but consistently wrong

3. **Per-Class Analysis**

   **Background Class (30 samples)**:
   - Correct predictions: 0
   - All misclassified as helicopter
   - Average confidence: 47.4%
   
   **Drone Class (30 samples)**:
   - Correct predictions: 0  
   - All misclassified as helicopter
   - Average confidence: 48.4%
   
   **Helicopter Class (30 samples)**:
   - Correct predictions: 30 (by chance)
   - Average confidence: 47.9%

4. **Confusion Matrix** (identical for all models):
   ```
                Predicted
                BG  Drone  Heli
   True BG       0    0    30
   True Drone    0    0    30
   True Heli     0    0    30
   ```

## Root Cause Analysis

### Why Are Models Failing?

1. **Dataset Distribution Mismatch**
   - Training data likely has different acoustic characteristics
   - Different recording environments, microphones, or noise conditions
   - Different drone/helicopter types than training set

2. **Domain Shift**
   - This Drone-detection-dataset may be from a different source
   - Different preprocessing or recording methodology
   - Models haven't learned robust, transferable features

3. **Model Over-fitting**
   - Models may have memorized training set patterns
   - Lack of generalization to new acoustic environments
   - Feature extractors not capturing invariant characteristics

4. **Class Imbalance in Training** (hypothesis)
   - Models may have been biased toward helicopter class during training
   - Or helicopter class may have been most challenging, leading to conservative predictions

## Recommendations

### Immediate Actions

1. **Inspect Audio Samples**
   - Listen to samples from this dataset vs training set
   - Compare spectrograms visually
   - Check for systematic differences in audio quality

2. **Feature Analysis**
   - Extract and compare feature distributions (mel-spectrograms)
   - Check for preprocessing mismatches
   - Verify HPSS (Harmonic-Percussive Separation) is working correctly

3. **Re-train with Mixed Data**
   - Incorporate samples from Drone-detection-dataset into training
   - Use data augmentation to increase robustness
   - Implement domain adaptation techniques

4. **Model Debugging**
   - Check activation maps to see what features models attend to
   - Verify models aren't broken (load original validation results)
   - Test on original validation set to confirm models still work

### Long-term Improvements

1. **Robust Training Strategy**
   - Mix multiple drone datasets during training
   - Use stronger augmentation (noise, reverb, time-stretch)
   - Implement domain adversarial training

2. **Ensemble Methods**
   - Combine models trained on different datasets
   - Use uncertainty-aware predictions

3. **Transfer Learning**
   - Start from pre-trained audio models (AudioSet, ESC-50)
   - Fine-tune on drone-specific data

4. **Evaluation Protocol**
   - Always test on held-out datasets from different sources
   - Report cross-dataset generalization metrics
   - Build robustness benchmarks

## Conclusion

**The models' performance on the Drone-detection-dataset is unacceptable for deployment:**

- ❌ 33.3% accuracy = random guessing
- ❌ 0% drone detection recall (missed all drones)
- ❌ 0% background rejection (falsely alarmed on all background)
- ❌ Severe prediction bias toward helicopter class
- ❌ Low confidence indicates model uncertainty

**The models need significant retraining or architectural changes before they can reliably detect drones in real-world scenarios represented by this dataset.**

The consistent failure across CRNN and PANNS architectures suggests a fundamental dataset distribution problem rather than an architecture issue.

---

**Generated**: October 26, 2025
**Evaluation Script**: `evaluate_drone_detection_dataset.py`
**Results Location**: `evaluation_results/drone_detection_dataset_detailed.json`
