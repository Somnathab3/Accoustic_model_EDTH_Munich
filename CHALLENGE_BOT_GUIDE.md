# Challenge Bot Quick Guide

## 🚀 Quick Start

The challenge bot now **automatically** uses the best available model:

```bash
# Simple - just run it!
python sota_challenge_bot.py

# With options
python sota_challenge_bot.py --max-iterations 100 --delay 0.5
```

## 🎯 Key Features

### 1. **Smart Model Detection**
- ✅ Automatically uses `panns_final.pt` when training completes
- ⚡ Falls back to `best_model.pt` while training is in progress
- 📊 Clear indication of which model is being used

### 2. **Speed Optimized**
- 🔥 Model pre-warming for faster first inference
- ⚡ Immediate submission after classification (no delays)
- 🎯 Default 0.5s delay between challenges (adjustable)
- 📈 Typical speed: 0.15-0.20s per challenge after warmup

### 3. **Comprehensive Logging**
- 📄 **CSV File**: `challenge_results/results.csv` - continuously updated
- 📋 **JSONL File**: `challenge_results/results.jsonl` - detailed results
- 📊 **Statistics**: `challenge_results/statistics.json` - performance metrics
- 🖼️ **Audio Samples**: Stored in `challenge_results/audio_samples/`

## 📊 CSV Output Format

```csv
iteration,timestamp,challenge_id,predicted,actual,correct,confidence,score_awarded,inference_time,total_time
1,2025-10-24 22:44:17,2b05de25...,background,drone,False,0.7546,100,2.9931,3.4136
2,2025-10-24 22:44:17,2b05de25...,drone,drone,True,0.8234,150,0.0747,0.1970
```

**Columns:**
- `iteration`: Challenge number
- `timestamp`: When the challenge was processed
- `challenge_id`: Unique ID from API
- `predicted`: Model's prediction
- `actual`: True label (from API response)
- `correct`: True/False
- `confidence`: Model confidence (0-1)
- `score_awarded`: Points earned
- `inference_time`: Time for model inference only
- `total_time`: Total time including download/submit

## 🎮 Usage Examples

### Run 100 challenges with fast speed
```bash
python sota_challenge_bot.py --max-iterations 100 --delay 0.5
```

### Run continuously (until stopped with Ctrl+C)
```bash
python sota_challenge_bot.py --delay 0.5
```

### Use specific model (override auto-detection)
```bash
python sota_challenge_bot.py --model models/best_model.pt --labels models/labels_current.json
```

### Specify custom CSV path
```bash
python sota_challenge_bot.py --csv results/my_results.csv
```

## 📈 Real-time Monitoring

The bot prints live updates:

```
[1] ✓ Predicted: drone        | Actual: drone        | Conf: 0.823 | Score: +150 | Time: 0.17s
[2] ✗ Predicted: background   | Actual: drone        | Conf: 0.755 | Score: +100 | Time: 0.15s
     [background:0.755 | drone:0.146 | helicopter:0.099]
[3] ✓ Predicted: helicopter   | Actual: helicopter   | Conf: 0.943 | Score: +150 | Time: 0.16s
```

**Legend:**
- ✓ = Correct prediction
- ✗ = Wrong prediction (with probability breakdown)
- Conf = Model confidence for predicted class
- Score = Points awarded by API
- Time = Total time for this challenge

## 📊 Statistics (Every 10 iterations)

```
────────────────────────────────────────────────────────────
Attempts: 50 | Correct: 42 | Wrong: 8
Accuracy: 84.0% | Score: 6300 | Avg Time: 0.181s
────────────────────────────────────────────────────────────
```

## 🔄 Training Workflow

### While Training (Current State)
```bash
# Training is running in one terminal
python train_sota_model.py ...

# Run bot with best checkpoint in another terminal
python sota_challenge_bot.py --max-iterations 50
# Uses: best_model.pt (updated after each epoch)
```

### After Training Completes
```bash
# Bot automatically switches to final model
python sota_challenge_bot.py
# Uses: panns_final.pt (best trained model)
```

## 📁 Output Files

All results are saved in `challenge_results/`:

```
challenge_results/
├── results.csv              # CSV with all results (continuously updated)
├── results.jsonl            # Detailed JSONL (one line per challenge)
├── statistics.json          # Performance statistics
└── audio_samples/           # Downloaded audio files
    ├── correct/            # Correctly classified samples
    └── incorrect/          # Misclassified samples
```

## 🎯 Current Model Performance

**Validation Results (Epoch 9):**
- Overall Accuracy: **86.67%**
- Macro F1: **0.8640**

**Per-Class:**
- Background: 100% (60/60)
- Drone: 63.33% (38/60) ← Main weakness
- Helicopter: 96.67% (58/60)

**Known Issue:** Model tends to classify quiet drones as background. This should improve as training continues!

## 💡 Tips for Best Results

1. **Let training complete**: The model at epoch 9 is still improving
2. **Monitor the CSV**: Check `results.csv` to track performance trends
3. **Adjust delay**: Use `--delay 0.3` for maximum speed (if API allows)
4. **Run in bursts**: Do 50-100 challenges, analyze results, wait for better checkpoint
5. **Check probabilities**: When wrong, the probability breakdown shows model uncertainty

## 🚨 Troubleshooting

### "No model found"
- Make sure training has started and created `models/best_model.pt`
- Or training completed and created `models/panns_final.pt`

### "No labels file found"
- The bot looks for `models/labels_current.json` (created manually)
- Or `models/labels.json` (created by training at completion)

### Low accuracy
- Model is still training (wait for more epochs)
- Check validation results: `python validate_model.py --model models/best_model.pt --labels models/labels_current.json --val-dir data/edth_munich_dataset/data/val`

### Slow inference
- First challenge is always slower (model loading)
- Should be ~0.15-0.20s after warmup
- Check GPU is being used (should show `Device: cuda`)

## 📞 Quick Commands

```bash
# Run 100 challenges fast
python sota_challenge_bot.py --max-iterations 100 --delay 0.5

# Check current results
head -20 challenge_results/results.csv

# Count correct predictions
python -c "import csv; f=open('challenge_results/results.csv'); r=csv.DictReader(f); print(sum(1 for row in r if row['correct']=='True'))"

# Check model validation
python validate_model.py --model models/best_model.pt --labels models/labels_current.json --val-dir data/edth_munich_dataset/data/val
```

## 🎉 What's New

### Speed Optimizations
- ✅ Model pre-warming (faster first inference)
- ✅ Immediate submission after classification
- ✅ Reduced default delay (0.5s instead of 1.0s)
- ✅ Optimized CSV append (no file re-reading)

### Smart Features
- ✅ Auto-detection of best available model
- ✅ Automatic fallback while training
- ✅ CSV continuous updates
- ✅ Clear progress indicators

### Better Output
- ✅ Formatted predictions with alignment
- ✅ Probability breakdown on errors
- ✅ Clear model status messages
- ✅ Comprehensive CSV logging
