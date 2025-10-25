# Continuous Improvement System - Quick Start
# Run this script to start BOTH the challenge bot and the continuous training pipeline

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*79) -ForegroundColor Cyan
Write-Host "CONTINUOUS IMPROVEMENT SYSTEM FOR ACOUSTIC DRONE DETECTION" -ForegroundColor Cyan
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*79) -ForegroundColor Cyan
Write-Host ""

Write-Host "This script will start TWO processes:" -ForegroundColor Yellow
Write-Host "  1. Challenge Bot (submits predictions)" -ForegroundColor Green
Write-Host "  2. Training Pipeline (improves model every 20 min)" -ForegroundColor Green
Write-Host ""

Write-Host "IMPORTANT: Keep BOTH windows open!" -ForegroundColor Red
Write-Host "  - Bot window: Submits predictions continuously" -ForegroundColor White
Write-Host "  - Pipeline window: Retrains model every 20 minutes" -ForegroundColor White
Write-Host ""

# Check if model exists
$model_path = "models\crnn_combined\crnn_final.pt"
if (-not (Test-Path $model_path)) {
    Write-Host "ERROR: Model not found at $model_path" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please train the model first:" -ForegroundColor Yellow
    Write-Host "python train_sota_model.py --train-dir data/edth_munich_dataset/data/train --val-dir data/edth_munich_dataset/data/val" -ForegroundColor White
    exit 1
}

Write-Host "Model found: $model_path" -ForegroundColor Green
Write-Host ""

# Prompt for confirmation
Write-Host "Press Enter to start both processes (Ctrl+C to cancel)..." -ForegroundColor Yellow
$null = Read-Host

Write-Host ""
Write-Host "Starting Challenge Bot in new window..." -ForegroundColor Cyan

# Start challenge bot in new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; python sota_challenge_bot.py --delay 0.5"

Write-Host "Challenge Bot started!" -ForegroundColor Green
Start-Sleep -Seconds 2

Write-Host ""
Write-Host "Starting Continuous Training Pipeline in new window..." -ForegroundColor Cyan

# Start training pipeline in new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; python continuous_training_pipeline.py --interval 1200 --epochs 20 --batch-size 32"

Write-Host "Training Pipeline started!" -ForegroundColor Green
Write-Host ""

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*79) -ForegroundColor Cyan
Write-Host "SYSTEM RUNNING!" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*79) -ForegroundColor Cyan
Write-Host ""

Write-Host "Two windows have been opened:" -ForegroundColor Yellow
Write-Host "  Window 1: Challenge Bot (submitting predictions)" -ForegroundColor White
Write-Host "  Window 2: Training Pipeline (retraining every 20 min)" -ForegroundColor White
Write-Host ""

Write-Host "How it works:" -ForegroundColor Cyan
Write-Host "  1. Bot submits predictions to challenges" -ForegroundColor White
Write-Host "  2. Correct predictions are saved to challenge_results/" -ForegroundColor White
Write-Host "  3. Every 20 minutes, pipeline:" -ForegroundColor White
Write-Host "     - Collects new correct samples" -ForegroundColor Gray
Write-Host "     - Adds them to combined dataset" -ForegroundColor Gray
Write-Host "     - Retrains model from last checkpoint" -ForegroundColor Gray
Write-Host "     - Updates crnn_final.pt (bot uses it immediately)" -ForegroundColor Gray
Write-Host ""

Write-Host "To stop the system:" -ForegroundColor Yellow
Write-Host "  - Close both terminal windows, or" -ForegroundColor White
Write-Host "  - Press Ctrl+C in each window" -ForegroundColor White
Write-Host ""

Write-Host "Monitor progress:" -ForegroundColor Cyan
Write-Host "  - Bot: challenge_results/results.csv" -ForegroundColor White
Write-Host "  - Pipeline: models/crnn_combined/training_history.json" -ForegroundColor White
Write-Host ""

Write-Host "Press Enter to close this window..." -ForegroundColor Yellow
$null = Read-Host
