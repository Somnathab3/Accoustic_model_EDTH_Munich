#!/bin/bash
# GitHub Upload Script for SOTA Acoustic Drone Detector
# This script prepares and uploads everything to GitHub

echo "=========================================="
echo "SOTA Model GitHub Upload Script"
echo "=========================================="
echo ""

# Step 1: Rename main README
echo "Step 1: Preparing README..."
if [ -f "README_DEPLOYMENT.md" ]; then
    mv -f README_DEPLOYMENT.md README.md
    echo "✓ README.md prepared"
else
    echo "⚠️  README_DEPLOYMENT.md not found, using existing README.md"
fi

# Step 2: Check if Git is initialized
echo ""
echo "Step 2: Checking Git status..."
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
    echo "✓ Git initialized"
else
    echo "✓ Git already initialized"
fi

# Step 3: Setup Git LFS for large model files
echo ""
echo "Step 3: Setting up Git LFS..."
git lfs install
git lfs track "models/*.pt"
git lfs track "models/*.pth"
git add .gitattributes
echo "✓ Git LFS configured for model files"

# Step 4: Check remote
echo ""
echo "Step 4: Checking remote repository..."
if git remote | grep -q "origin"; then
    echo "✓ Remote 'origin' already exists"
    git remote -v
else
    echo "Adding remote repository..."
    read -p "Enter your GitHub repository URL: " REPO_URL
    git remote add origin "$REPO_URL"
    echo "✓ Remote added"
fi

# Step 5: Check for essential files
echo ""
echo "Step 5: Verifying essential files..."
essential_files=(
    "sota_challenge_bot.py"
    "sota_inference.py"
    "train_sota_model.py"
    "models/panns_final.pt"
    "models/labels_current.json"
    "requirements_kaggle.txt"
)

all_present=true
for file in "${essential_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ Missing: $file"
        all_present=false
    fi
done

if [ "$all_present" = false ]; then
    echo ""
    echo "⚠️  Some essential files are missing!"
    echo "Please ensure all files are present before uploading."
    exit 1
fi

# Step 6: Stage all files
echo ""
echo "Step 6: Staging files..."
git add .
echo "✓ Files staged"

# Step 7: Show status
echo ""
echo "Step 7: Git status:"
git status --short | head -20
total_files=$(git status --short | wc -l)
echo "... ($total_files files total)"

# Step 8: Commit
echo ""
echo "Step 8: Creating commit..."
read -p "Enter commit message (or press Enter for default): " COMMIT_MSG
if [ -z "$COMMIT_MSG" ]; then
    COMMIT_MSG="Add SOTA acoustic drone detector with PANNs model, smart timing, and comprehensive documentation"
fi
git commit -m "$COMMIT_MSG"
echo "✓ Commit created"

# Step 9: Push
echo ""
echo "Step 9: Ready to push to GitHub"
read -p "Push to GitHub now? (y/n): " PUSH_NOW
if [ "$PUSH_NOW" = "y" ] || [ "$PUSH_NOW" = "Y" ]; then
    echo "Pushing to GitHub..."
    git branch -M main
    git push -u origin main
    echo "✓ Pushed to GitHub!"
else
    echo "Skipping push. You can push later with: git push -u origin main"
fi

echo ""
echo "=========================================="
echo "Upload Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Go to your GitHub repository"
echo "2. Verify files are uploaded"
echo "3. Test on Kaggle:"
echo "   !git clone YOUR_REPO_URL"
echo "   %cd REPO_NAME"
echo "   !pip install -q -r requirements_kaggle.txt"
echo "   !python sota_challenge_bot.py --max-iterations 10"
echo ""
