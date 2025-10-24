# 🚀 QUICK COMMAND REFERENCE

## Start Challenge Bot
```bash
# Default (recommended)
python sota_challenge_bot.py

# With custom settings
python sota_challenge_bot.py --delay 0.5 --max-iterations 100
```

## Analyze Results
```bash
python analyze_results.py
```

## Check Model Performance
```bash
python validate_model.py --model models/best_model.pt --labels models/labels_current.json --val-dir data/edth_munich_dataset/data/val
```

## View Recent Results
```bash
# PowerShell
Get-Content challenge_results/results.csv -Tail 20

# Live monitoring (updates every 5 seconds)
while($true) { cls; Get-Content challenge_results/results.csv -Tail 20; Start-Sleep 5 }
```

## Key Parameters
- `--delay 0.5` - Time between challenges (seconds)
- `--max-iterations 100` - Stop after N challenges
- `--model models/best_model.pt` - Use specific model
- `--csv results/my_run.csv` - Custom CSV path

## Output Files
- `challenge_results/results.csv` - Main results
- `challenge_results/results.jsonl` - Detailed logs
- `challenge_results/statistics.json` - Stats

## What to Watch For
- ✅ **Score: +150** = Correct prediction
- ⚠️ **Score: +100** = Wrong but attempted
- ⚠️ **Score: +0** = Unknown/error
- 🎯 **"Score received!"** = Synced with server
- ⏸️ **"Same challenge detected"** = Waiting for new challenge
- ⏳ **"Waiting 100s"** = Auto-sync in progress

## Timing Strategy
- **Normal**: 0.5s delay between challenges
- **Duplicate**: Auto-wait 100s for new challenge
- **Errors**: Exponential backoff (2-30s)

## Stop Bot
Press `Ctrl+C` - Results are saved automatically!
