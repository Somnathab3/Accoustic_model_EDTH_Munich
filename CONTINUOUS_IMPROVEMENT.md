# Continuous Improvement System

This system enables **dynamic model improvement** while the challenge bot runs continuously.

## 🎯 Overview

The system consists of two parallel processes:

1. **Challenge Bot** (`sota_challenge_bot.py`) - Submits predictions continuously
2. **Training Pipeline** (`continuous_training_pipeline.py`) - Retrains model every 20 minutes with new correct samples

## 🚀 Quick Start

### Option 1: Automated Start (Recommended)
```powershell
.\start_continuous_improvement.ps1
```

This will open two windows:
- Window 1: Challenge bot
- Window 2: Training pipeline

### Option 2: Manual Start

**Terminal 1 - Challenge Bot:**
```bash
python sota_challenge_bot.py --delay 0.5
```

**Terminal 2 - Training Pipeline:**
```bash
python continuous_training_pipeline.py --interval 1200 --epochs 20
```

## 📊 How It Works

```
┌─────────────────────┐
│   Challenge Bot     │
│  (Continuously)     │
└──────────┬──────────┘
           │
           │ Submits predictions
           │ Saves correct samples
           ▼
┌─────────────────────┐
│ challenge_results/  │
│  - results.csv      │
│  - results.jsonl    │
│  - audio_samples/   │
└──────────┬──────────┘
           │
           │ Every 20 minutes
           ▼
┌─────────────────────┐
│ Training Pipeline   │
│  1. Collect correct │
│  2. Add to dataset  │
│  3. Retrain model   │
│  4. Update model    │
└──────────┬──────────┘
           │
           │ Overwrites model
           ▼
┌─────────────────────┐
│ models/crnn_combined│
│  - crnn_final.pt    │ ◄─── Bot uses this!
│  - best_model.pt    │
└─────────────────────┘
```

## 🔄 Pipeline Cycle

Every 20 minutes, the pipeline:

1. **Collects** new correct predictions from `challenge_results/results.jsonl`
   - Only samples with `score_awarded > 0`
   - Tracks processed samples to avoid duplicates

2. **Adds** correct samples to `data/combined_dataset/`
   - 80% to training set
   - 20% to validation set

3. **Retrains** the model
   - Loads from `best_model.pt` (last best checkpoint)
   - Trains for 20 epochs with new data
   - Uses lower learning rate (0.0001) for fine-tuning

4. **Updates** the model **IN-PLACE**
   - Overwrites `crnn_final.pt`
   - Bot automatically uses updated model on next inference
   - **No renaming, no downtime!**

## 📁 File Structure

```
acoustic-drone-detector/
├── sota_challenge_bot.py              # Challenge bot
├── continuous_training_pipeline.py    # Training pipeline
├── start_continuous_improvement.ps1   # Quick start script
│
├── challenge_results/
│   ├── results.csv                    # Challenge results
│   ├── results.jsonl                  # Detailed results
│   ├── processed_samples.json         # Tracked processed samples
│   └── audio_samples/                 # Audio files
│
├── data/
│   └── combined_dataset/              # Growing training dataset
│       ├── train/                     # Training samples
│       │   ├── background/
│       │   ├── drone/
│       │   └── helicopter/
│       └── val/                       # Validation samples
│           ├── background/
│           ├── drone/
│           └── helicopter/
│
└── models/
    └── crnn_combined/                 # Model directory
        ├── best_model.pt              # Best checkpoint (for retraining)
        ├── crnn_final.pt              # Active model (bot uses this)
        ├── labels.json                # Class labels
        └── training_history.json      # Training metrics
```

## ⚙️ Configuration

### Training Pipeline Options

```bash
python continuous_training_pipeline.py \
  --interval 1200 \           # Cycle interval (seconds)
  --epochs 20 \               # Epochs per cycle
  --batch-size 32 \           # Batch size
  --max-cycles 10 \           # Max cycles (optional)
  --train-ratio 0.8           # Train/val split ratio
```

### Challenge Bot Options

```bash
python sota_challenge_bot.py \
  --delay 0.5 \               # Delay between challenges
  --max-iterations 100        # Max iterations (optional)
```

## 📈 Monitoring Progress

### Bot Progress
```bash
# View results
cat challenge_results/results.csv

# Analyze performance
python analyze_results.py
```

### Pipeline Progress
```bash
# View training history
cat models/crnn_combined/training_history.json

# View processed samples
cat challenge_results/processed_samples.json
```

### Real-time Monitoring
Watch both terminal windows to see:
- Bot: Prediction accuracy, scores
- Pipeline: New samples collected, training progress

## 🎯 Expected Behavior

### First Cycle (20 minutes)
- Bot collects ~20-30 correct predictions
- Pipeline adds them to dataset
- Model trains for 20 epochs (~5-10 minutes)
- Accuracy improvement: +0.5-2%

### After Several Cycles
- Dataset grows continuously
- Model becomes more robust
- Accuracy improves incrementally
- Performance stabilizes

## 🔍 Key Features

### 1. **No Downtime**
- Model updated in-place
- Bot continues running
- No file renaming required

### 2. **Duplicate Prevention**
- Tracks processed challenge IDs
- Saves to `processed_samples.json`
- Never processes same sample twice

### 3. **Smart Fine-tuning**
- Loads from best checkpoint
- Lower learning rate (0.0001)
- Preserves learned features

### 4. **Automatic Recovery**
- Handles errors gracefully
- Continues even if cycle fails
- Logs all operations

## 🛠️ Troubleshooting

### Pipeline Not Training
**Check:**
- Are correct samples being collected?
- View: `challenge_results/processed_samples.json`
- If `total_processed` not increasing, no new correct predictions

### Model Not Improving
**Check:**
- Dataset imbalance (view counts in pipeline output)
- Training metrics in `training_history.json`
- May need more epochs or different hyperparameters

### Bot Using Old Model
**Check:**
- Is `crnn_final.pt` being updated? (check timestamp)
- Bot loads model once at startup
- Restart bot to load updated model immediately

## 📊 Performance Expectations

### Baseline (CRNN trained on original data)
- Validation Accuracy: ~60-70%
- Challenge Accuracy: ~5% (unknown ground truth)

### After Continuous Training (24 hours)
- ~100-200 new samples added
- Improved robustness on challenge data
- Better confidence calibration

### Long-term (1 week)
- ~500-1000 new samples
- Significantly improved performance
- Model specializes to challenge distribution

## 🎓 Advanced Usage

### Custom Cycle Timing
```bash
# 10-minute cycles (aggressive)
python continuous_training_pipeline.py --interval 600

# 1-hour cycles (conservative)
python continuous_training_pipeline.py --interval 3600
```

### More Training Per Cycle
```bash
# 50 epochs per cycle
python continuous_training_pipeline.py --epochs 50

# Larger batch size (needs more GPU memory)
python continuous_training_pipeline.py --batch-size 64
```

### Limited Cycles (Testing)
```bash
# Run only 3 cycles
python continuous_training_pipeline.py --max-cycles 3
```

## 🔄 Comparison with PANNs

To compare CRNN vs PANNs performance:

```bash
# Compare on failed PANNs samples
python compare_crnn_vs_panns.py

# View comparison
cat challenge_results/crnn_vs_panns_comparison.csv
```

## 📝 Notes

- **First cycle takes longest** (initial dataset copy)
- **Model size stays constant** (~6.7 MB for CRNN)
- **Pipeline is CPU-friendly** (inference uses GPU)
- **Safe to stop/restart** (tracks progress in JSON files)

## 🚦 Status Indicators

### Bot Status
- `✓` - Correct prediction
- `✗` - Incorrect prediction
- `⏸️` - Waiting for new challenge

### Pipeline Status
- `📦` - Collecting samples
- `🔄` - Retraining model
- `✅` - Cycle complete
- `⏸️` - No new samples (skipped)

## 🎯 Success Metrics

Track these to measure improvement:
1. **Challenge accuracy** (in bot output)
2. **Average confidence** (in results.csv)
3. **Total score** (in statistics.json)
4. **Validation accuracy** (in training_history.json)

---

## 🚀 Ready to Start?

```powershell
# Start the system
.\start_continuous_improvement.ps1

# Let it run for 24 hours
# Check back to see improved performance!
```

**Good luck with your continuous improvement! 🎯🚁**
