# Train All Improved Models Sequentially
# PANNS → Transformer → SNN

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  TRAINING ALL IMPROVED MODELS (PANNS, Transformer, SNN)           ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$START_TIME = Get-Date

Write-Host "Training Plan:" -ForegroundColor Yellow
Write-Host "  1. PANNS (40 epochs, approx 2-3 hours)" -ForegroundColor White
Write-Host "  2. Transformer (50 epochs, approx 3-4 hours)" -ForegroundColor White
Write-Host "  3. SNN (50 epochs, approx 3-4 hours)" -ForegroundColor White
Write-Host "  Total estimated time: 8-11 hours" -ForegroundColor White
Write-Host ""

$RESULTS = @()

# ============================================================================
# TRAIN PANNS
# ============================================================================
Write-Host ""
Write-Host "┌────────────────────────────────────────────────────────────────────┐" -ForegroundColor Green
Write-Host "│  STEP 1/3: TRAINING PANNS                                          │" -ForegroundColor Green
Write-Host "└────────────────────────────────────────────────────────────────────┘" -ForegroundColor Green
Write-Host ""

$PANNS_START = Get-Date
& .\train_panns_improved.ps1
$PANNS_END = Get-Date
$PANNS_DURATION = $PANNS_END - $PANNS_START

if ($LASTEXITCODE -eq 0) {
    $RESULTS += @{
        Model = "PANNS"
        Status = "[OK] SUCCESS"
        Duration = $PANNS_DURATION
    }
    Write-Host "[OK] PANNS completed in $($PANNS_DURATION.ToString('hh\:mm\:ss'))" -ForegroundColor Green
} else {
    $RESULTS += @{
        Model = "PANNS"
        Status = "[X] FAILED"
        Duration = $PANNS_DURATION
    }
    Write-Host "[X] PANNS failed after $($PANNS_DURATION.ToString('hh\:mm\:ss'))" -ForegroundColor Red
}

# ============================================================================
# TRAIN TRANSFORMER
# ============================================================================
Write-Host ""
Write-Host "┌────────────────────────────────────────────────────────────────────┐" -ForegroundColor Green
Write-Host "│  STEP 2/3: TRAINING TRANSFORMER                                    │" -ForegroundColor Green
Write-Host "└────────────────────────────────────────────────────────────────────┘" -ForegroundColor Green
Write-Host ""

$TRANSFORMER_START = Get-Date
& .\train_transformer_improved.ps1
$TRANSFORMER_END = Get-Date
$TRANSFORMER_DURATION = $TRANSFORMER_END - $TRANSFORMER_START

if ($LASTEXITCODE -eq 0) {
    $RESULTS += @{
        Model = "Transformer"
        Status = "[OK] SUCCESS"
        Duration = $TRANSFORMER_DURATION
    }
    Write-Host "[OK] Transformer completed in $($TRANSFORMER_DURATION.ToString('hh\:mm\:ss'))" -ForegroundColor Green
} else {
    $RESULTS += @{
        Model = "Transformer"
        Status = "[X] FAILED"
        Duration = $TRANSFORMER_DURATION
    }
    Write-Host "[X] Transformer failed after $($TRANSFORMER_DURATION.ToString('hh\:mm\:ss'))" -ForegroundColor Red
}

# ============================================================================
# TRAIN SNN
# ============================================================================
Write-Host ""
Write-Host "┌────────────────────────────────────────────────────────────────────┐" -ForegroundColor Green
Write-Host "│  STEP 3/3: TRAINING SNN                                            │" -ForegroundColor Green
Write-Host "└────────────────────────────────────────────────────────────────────┘" -ForegroundColor Green
Write-Host ""

$SNN_START = Get-Date
& .\train_snn_improved.ps1
$SNN_END = Get-Date
$SNN_DURATION = $SNN_END - $SNN_START

if ($LASTEXITCODE -eq 0) {
    $RESULTS += @{
        Model = "SNN"
        Status = "[OK] SUCCESS"
        Duration = $SNN_DURATION
    }
    Write-Host "[OK] SNN completed in $($SNN_DURATION.ToString('hh\:mm\:ss'))" -ForegroundColor Green
} else {
    $RESULTS += @{
        Model = "SNN"
        Status = "[X] FAILED"
        Duration = $SNN_DURATION
    }
    Write-Host "[X] SNN failed after $($SNN_DURATION.ToString('hh\:mm\:ss'))" -ForegroundColor Red
}

# ============================================================================
# FINAL SUMMARY
# ============================================================================
$END_TIME = Get-Date
$TOTAL_DURATION = $END_TIME - $START_TIME

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  TRAINING COMPLETE - SUMMARY                                       ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

foreach ($result in $RESULTS) {
    $color = if ($result.Status -like "*SUCCESS*") { "Green" } else { "Red" }
    Write-Host "  $($result.Model.PadRight(15)) $($result.Status.PadRight(10)) $($result.Duration.ToString('hh\:mm\:ss'))" -ForegroundColor $color
}

Write-Host ""
Write-Host "Total training time: $($TOTAL_DURATION.ToString('hh\:mm\:ss'))" -ForegroundColor Yellow
Write-Host ""

# Count successes
$SUCCESS_COUNT = ($RESULTS | Where-Object { $_.Status -like "*SUCCESS*" }).Count
$TOTAL_COUNT = $RESULTS.Count

Write-Host "Results: $SUCCESS_COUNT / $TOTAL_COUNT models trained successfully" -ForegroundColor $(if ($SUCCESS_COUNT -eq $TOTAL_COUNT) { "Green" } else { "Yellow" })
Write-Host ""

if ($SUCCESS_COUNT -eq $TOTAL_COUNT) {
    Write-Host "[SUCCESS] All models trained successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Compare results: python compare_models.py" -ForegroundColor White
    Write-Host "  2. Test on challenge: python sota_challenge_bot.py --model models/panns_improved/panns_final.pt" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "[WARNING] Some models failed to train" -ForegroundColor Yellow
    Write-Host "Check logs in models/*/training_history.json for details" -ForegroundColor White
    Write-Host ""
}

Write-Host "Model locations:" -ForegroundColor Yellow
Write-Host "  PANNS:       models\panns_improved\panns_final.pt" -ForegroundColor White
Write-Host "  Transformer: models\transformer_improved\transformer_final.pt" -ForegroundColor White
Write-Host "  SNN:         models\snn_improved\snn_final.pt" -ForegroundColor White
Write-Host ""
