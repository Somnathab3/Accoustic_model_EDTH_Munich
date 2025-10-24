# GitHub Upload Script for SOTA Acoustic Drone Detector
# PowerShell version

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "SOTA Model GitHub Upload Script" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Rename main README
Write-Host "Step 1: Preparing README..." -ForegroundColor Yellow
if (Test-Path "README_DEPLOYMENT.md") {
    Move-Item -Path "README_DEPLOYMENT.md" -Destination "README.md" -Force
    Write-Host "✓ README.md prepared" -ForegroundColor Green
} else {
    Write-Host "⚠️  README_DEPLOYMENT.md not found, using existing README.md" -ForegroundColor Yellow
}

# Step 2: Check if Git is initialized
Write-Host ""
Write-Host "Step 2: Checking Git status..." -ForegroundColor Yellow
if (-not (Test-Path ".git")) {
    Write-Host "Initializing Git repository..." -ForegroundColor Gray
    git init
    Write-Host "✓ Git initialized" -ForegroundColor Green
} else {
    Write-Host "✓ Git already initialized" -ForegroundColor Green
}

# Step 3: Setup Git LFS for large model files
Write-Host ""
Write-Host "Step 3: Setting up Git LFS..." -ForegroundColor Yellow
git lfs install
git lfs track "models/*.pt"
git lfs track "models/*.pth"
git add .gitattributes
Write-Host "✓ Git LFS configured for model files" -ForegroundColor Green

# Step 4: Check remote
Write-Host ""
Write-Host "Step 4: Checking remote repository..." -ForegroundColor Yellow
$remotes = git remote
if ($remotes -contains "origin") {
    Write-Host "✓ Remote 'origin' already exists" -ForegroundColor Green
    git remote -v
} else {
    Write-Host "Adding remote repository..." -ForegroundColor Gray
    $repoUrl = Read-Host "Enter your GitHub repository URL"
    git remote add origin $repoUrl
    Write-Host "✓ Remote added" -ForegroundColor Green
}

# Step 5: Check for essential files
Write-Host ""
Write-Host "Step 5: Verifying essential files..." -ForegroundColor Yellow
$essentialFiles = @(
    "sota_challenge_bot.py",
    "sota_inference.py",
    "train_sota_model.py",
    "models/panns_final.pt",
    "models/labels_current.json",
    "requirements_kaggle.txt"
)

$allPresent = $true
foreach ($file in $essentialFiles) {
    if (Test-Path $file) {
        Write-Host "✓ $file" -ForegroundColor Green
    } else {
        Write-Host "✗ Missing: $file" -ForegroundColor Red
        $allPresent = $false
    }
}

if (-not $allPresent) {
    Write-Host ""
    Write-Host "⚠️  Some essential files are missing!" -ForegroundColor Red
    Write-Host "Please ensure all files are present before uploading." -ForegroundColor Yellow
    exit 1
}

# Step 6: Stage all files
Write-Host ""
Write-Host "Step 6: Staging files..." -ForegroundColor Yellow
git add .
Write-Host "✓ Files staged" -ForegroundColor Green

# Step 7: Show status
Write-Host ""
Write-Host "Step 7: Git status:" -ForegroundColor Yellow
$status = git status --short
$status | Select-Object -First 20
$totalFiles = ($status | Measure-Object).Count
Write-Host "... ($totalFiles files total)" -ForegroundColor Gray

# Step 8: Commit
Write-Host ""
Write-Host "Step 8: Creating commit..." -ForegroundColor Yellow
$commitMsg = Read-Host "Enter commit message (or press Enter for default)"
if ([string]::IsNullOrWhiteSpace($commitMsg)) {
    $commitMsg = "Add SOTA acoustic drone detector with PANNs model, smart timing, and comprehensive documentation"
}
git commit -m $commitMsg
Write-Host "✓ Commit created" -ForegroundColor Green

# Step 9: Push
Write-Host ""
Write-Host "Step 9: Ready to push to GitHub" -ForegroundColor Yellow
$pushNow = Read-Host "Push to GitHub now? (y/n)"
if ($pushNow -eq "y" -or $pushNow -eq "Y") {
    Write-Host "Pushing to GitHub..." -ForegroundColor Gray
    git branch -M main
    git push -u origin main
    Write-Host "✓ Pushed to GitHub!" -ForegroundColor Green
} else {
    Write-Host "Skipping push. You can push later with: git push -u origin main" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Upload Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Go to your GitHub repository" -ForegroundColor White
Write-Host "2. Verify files are uploaded" -ForegroundColor White
Write-Host "3. Test on Kaggle:" -ForegroundColor White
Write-Host "   !git clone YOUR_REPO_URL" -ForegroundColor Gray
Write-Host "   %cd REPO_NAME" -ForegroundColor Gray
Write-Host "   !pip install -q -r requirements_kaggle.txt" -ForegroundColor Gray
Write-Host "   !python sota_challenge_bot.py --max-iterations 10" -ForegroundColor Gray
Write-Host ""
