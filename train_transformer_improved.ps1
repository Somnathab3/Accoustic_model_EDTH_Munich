# Train Improved Transformer Model
# Fixes undertraining (Val F1 0.647, drone recall 0.408)

Write-Host "="*70 -ForegroundColor Cyan
Write-Host "TRAINING IMPROVED TRANSFORMER MODEL" -ForegroundColor Cyan
Write-Host "="*70 -ForegroundColor Cyan
Write-Host ""

$TRAIN_DIR = "F:\EDTH\acoustic-drone-detector\data\combined_dataset\train"
$VAL_DIR = "F:\EDTH\acoustic-drone-detector\data\combined_dataset\val"
$OUTPUT_DIR = "models\transformer_improved"
$TEACHER_MODEL = "models\crnn_combined\crnn_final.pt"

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Model: Transformer (AST-style)" -ForegroundColor White
Write-Host "  Train dir: $TRAIN_DIR" -ForegroundColor White
Write-Host "  Val dir: $VAL_DIR" -ForegroundColor White
Write-Host "  Output: $OUTPUT_DIR" -ForegroundColor White
Write-Host "  Teacher: $TEACHER_MODEL" -ForegroundColor White
Write-Host ""
Write-Host "Improvements:" -ForegroundColor Yellow
Write-Host "  [+] Focal loss (gamma=2.0) for drone class" -ForegroundColor Green
Write-Host "  [+] Balanced sampling (equal representation)" -ForegroundColor Green
Write-Host "  [+] Knowledge distillation from CRNN (CRITICAL)" -ForegroundColor Green
Write-Host "  [+] Higher weight decay (0.05) for regularization" -ForegroundColor Green
Write-Host "  [+] Enhanced augmentations + SpecAugment" -ForegroundColor Green
Write-Host ""

Write-Host "Starting training..." -ForegroundColor Green
Write-Host ""

python train_improved_models.py `
    --model-type transformer `
    --train-dir "$TRAIN_DIR" `
    --val-dir "$VAL_DIR" `
    --output-dir "$OUTPUT_DIR" `
    --use-hpss `
    --epochs 50 `
    --batch-size 32 `
    --lr 0.0001 `
    --weight-decay 0.05 `
    --warmup-ratio 0.1 `
    --use-focal-loss `
    --focal-gamma 2.0 `
    --label-smoothing 0.05 `
    --use-class-weights `
    --balanced-sampling `
    --use-kd `
    --teacher-model "$TEACHER_MODEL" `
    --kd-temperature 4.0 `
    --kd-alpha 0.5 `
    --patience 15 `
    --num-workers 4

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "="*70 -ForegroundColor Green
    Write-Host "TRANSFORMER TRAINING COMPLETE!" -ForegroundColor Green
    Write-Host "="*70 -ForegroundColor Green
    Write-Host ""
    Write-Host "Saved models:" -ForegroundColor Yellow
    Write-Host "  Best F1: $OUTPUT_DIR\best_model.pt" -ForegroundColor White
    Write-Host "  Best drone recall: $OUTPUT_DIR\best_drone_recall.pt" -ForegroundColor White
    Write-Host "  Final: $OUTPUT_DIR\transformer_final.pt" -ForegroundColor White
    Write-Host ""
    Write-Host "Expected improvements:" -ForegroundColor Yellow
    Write-Host "  Current: Val F1 approx 0.647, drone recall approx 0.408" -ForegroundColor Red
    Write-Host "  Target:  Val F1 > 0.90,  drone recall > 0.80" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Training failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
