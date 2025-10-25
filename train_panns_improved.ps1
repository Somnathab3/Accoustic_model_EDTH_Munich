# Train Improved PANNS Model
# Fixes low drone recall (0.651) with focal loss, KD, and progressive unfreezing

Write-Host "="*70 -ForegroundColor Cyan
Write-Host "TRAINING IMPROVED PANNS MODEL" -ForegroundColor Cyan
Write-Host "="*70 -ForegroundColor Cyan
Write-Host ""

$TRAIN_DIR = "F:\EDTH\acoustic-drone-detector\data\combined_dataset\train"
$VAL_DIR = "F:\EDTH\acoustic-drone-detector\data\combined_dataset\val"
$OUTPUT_DIR = "models\panns_improved"
$TEACHER_MODEL = "models\crnn_combined\crnn_final.pt"

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Model: PANNS (PANNs-CNN14 style)" -ForegroundColor White
Write-Host "  Train dir: $TRAIN_DIR" -ForegroundColor White
Write-Host "  Val dir: $VAL_DIR" -ForegroundColor White
Write-Host "  Output: $OUTPUT_DIR" -ForegroundColor White
Write-Host "  Teacher: $TEACHER_MODEL" -ForegroundColor White
Write-Host ""
Write-Host "Improvements:" -ForegroundColor Yellow
Write-Host "  [+] Focal loss (gamma=2.0) for drone class" -ForegroundColor Green
Write-Host "  [+] Balanced sampling (equal bg/drone/heli)" -ForegroundColor Green
Write-Host "  [+] Knowledge distillation from CRNN" -ForegroundColor Green
Write-Host "  [+] Enhanced augmentations (pitch, notch, band-limit)" -ForegroundColor Green
Write-Host "  [+] Class weights for imbalanced data" -ForegroundColor Green
Write-Host ""

Write-Host "Starting training..." -ForegroundColor Green
Write-Host ""

python train_improved_models.py `
    --model-type panns `
    --train-dir "$TRAIN_DIR" `
    --val-dir "$VAL_DIR" `
    --output-dir "$OUTPUT_DIR" `
    --use-hpss `
    --epochs 40 `
    --batch-size 32 `
    --lr 0.0001 `
    --weight-decay 0.0001 `
    --warmup-ratio 0.05 `
    --use-focal-loss `
    --focal-gamma 2.0 `
    --label-smoothing 0.05 `
    --use-class-weights `
    --balanced-sampling `
    --use-kd `
    --teacher-model "$TEACHER_MODEL" `
    --kd-temperature 3.0 `
    --kd-alpha 0.5 `
    --patience 10 `
    --num-workers 4

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "="*70 -ForegroundColor Green
    Write-Host "PANNS TRAINING COMPLETE!" -ForegroundColor Green
    Write-Host "="*70 -ForegroundColor Green
    Write-Host ""
    Write-Host "Saved models:" -ForegroundColor Yellow
    Write-Host "  Best F1: $OUTPUT_DIR\best_model.pt" -ForegroundColor White
    Write-Host "  Best drone recall: $OUTPUT_DIR\best_drone_recall.pt" -ForegroundColor White
    Write-Host "  Final: $OUTPUT_DIR\panns_final.pt" -ForegroundColor White
    Write-Host ""
    Write-Host "Expected improvements:" -ForegroundColor Yellow
    Write-Host "  Current: Val F1 approx 0.860, drone recall approx 0.651" -ForegroundColor Red
    Write-Host "  Target:  Val F1 > 0.92,  drone recall > 0.85" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Training failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
