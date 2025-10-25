# Train Improved SNN Model
# Fixes severe underperformance (Val F1 0.598, drone recall 0.214)

Write-Host "="*70 -ForegroundColor Cyan
Write-Host "TRAINING IMPROVED SNN MODEL" -ForegroundColor Cyan
Write-Host "="*70 -ForegroundColor Cyan
Write-Host ""

$TRAIN_DIR = "F:\EDTH\acoustic-drone-detector\data\combined_dataset\train"
$VAL_DIR = "F:\EDTH\acoustic-drone-detector\data\combined_dataset\val"
$OUTPUT_DIR = "models\snn_improved"
$TEACHER_MODEL = "models\crnn_combined\crnn_final.pt"

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Model: SNN (Hybrid Spiking Neural Network)" -ForegroundColor White
Write-Host "  Train dir: $TRAIN_DIR" -ForegroundColor White
Write-Host "  Val dir: $VAL_DIR" -ForegroundColor White
Write-Host "  Output: $OUTPUT_DIR" -ForegroundColor White
Write-Host "  Teacher: $TEACHER_MODEL" -ForegroundColor White
Write-Host ""
Write-Host "Improvements:" -ForegroundColor Yellow
Write-Host "  [+] Hybrid architecture (conv front-end + SNN)" -ForegroundColor Green
Write-Host "  [+] Increased timesteps (4 to 10)" -ForegroundColor Green
Write-Host "  [+] Higher learning rate (0.0001 to 0.002)" -ForegroundColor Green
Write-Host "  [+] Focal loss (gamma=2.0) for drone class" -ForegroundColor Green
Write-Host "  [+] Balanced sampling" -ForegroundColor Green
Write-Host "  [+] Knowledge distillation from CRNN (CRITICAL)" -ForegroundColor Green
Write-Host "  [+] Rate coding (more stable)" -ForegroundColor Green
Write-Host ""

Write-Host "Starting training..." -ForegroundColor Green
Write-Host ""

python train_improved_models.py `
    --model-type snn `
    --train-dir "$TRAIN_DIR" `
    --val-dir "$VAL_DIR" `
    --output-dir "$OUTPUT_DIR" `
    --use-hpss `
    --snn-timesteps 10 `
    --spike-slope 35.0 `
    --epochs 50 `
    --batch-size 16 `
    --lr 0.002 `
    --weight-decay 0.01 `
    --warmup-ratio 0.1 `
    --use-focal-loss `
    --focal-gamma 2.0 `
    --label-smoothing 0.05 `
    --use-class-weights `
    --balanced-sampling `
    --use-kd `
    --teacher-model "$TEACHER_MODEL" `
    --kd-temperature 3.0 `
    --kd-alpha 0.5 `
    --patience 15 `
    --num-workers 4

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "="*70 -ForegroundColor Green
    Write-Host "SNN TRAINING COMPLETE!" -ForegroundColor Green
    Write-Host "="*70 -ForegroundColor Green
    Write-Host ""
    Write-Host "Saved models:" -ForegroundColor Yellow
    Write-Host "  Best F1: $OUTPUT_DIR\best_model.pt" -ForegroundColor White
    Write-Host "  Best drone recall: $OUTPUT_DIR\best_drone_recall.pt" -ForegroundColor White
    Write-Host "  Final: $OUTPUT_DIR\snn_final.pt" -ForegroundColor White
    Write-Host ""
    Write-Host "Expected improvements:" -ForegroundColor Yellow
    Write-Host "  Current: Val F1 approx 0.598, drone recall approx 0.214" -ForegroundColor Red
    Write-Host "  Target:  Val F1 > 0.85,  drone recall > 0.75" -ForegroundColor Green
    Write-Host ""
    Write-Host "Note: SNN is most challenging - any improvement is significant!" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "Training failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
