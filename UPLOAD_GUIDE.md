# COMPLETE GITHUB UPLOAD GUIDE
# Step-by-Step Instructions for Uploading FFT-CNN-DNN Model to GitHub
# Repository: https://github.com/Somnathab3/Accoustic_model_EDTH_Munich

## 📋 QUICK START SUMMARY

This guide helps you upload the essential FFT-CNN-DNN acoustic drone detector files to GitHub for Kaggle deployment.

### What You'll Upload:
- ✅ Core Python scripts (5 files)
- ✅ Source code directory with modules (35+ files)
- ✅ Documentation files (4 files)
- ✅ Setup scripts (2 files)
- ✅ Configuration files (3 files)
- ✅ Model files (~50MB - via Git LFS or Kaggle Dataset)

### Total Time: ~15-30 minutes

---

## 🚀 METHOD 1: RECOMMENDED - Using Git Command Line

### Step 1: Prepare Your Local Repository

```powershell
# Open PowerShell and navigate to your project
cd f:\EDTH\acoustic-drone-detector

# Verify you're in the right directory
Get-ChildItem | Select-Object Name
```

### Step 2: Initialize Git (If Not Already Done)

```powershell
# Initialize git repository
git init

# Set your identity (if not already set)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Add the remote repository
git remote add origin https://github.com/Somnathab3/Accoustic_model_EDTH_Munich.git

# Verify remote
git remote -v
```

### Step 3: Setup Git LFS for Large Model Files

```powershell
# Install Git LFS (if not already installed)
# Download from: https://git-lfs.github.com/

# Initialize Git LFS
git lfs install

# Track large model files
git lfs track "models/*.pt"
git lfs track "models/*.pth"

# Add .gitattributes
git add .gitattributes
```

### Step 4: Stage Essential Files

```powershell
# Add core scripts
git add challenge_bot_fft_cnn_dnn.py
git add train_fft_cnn_dnn_quick.py
git add test_fft_cnn_dnn.py
git add simple_infer.py
git add infer.py

# Add setup scripts
git add setup_kaggle.sh
git add setup_kaggle.ps1

# Add requirements
git add requirements.txt
git add requirements_minimal.txt

# Add documentation
git add DEPLOYMENT_README.md
git add FFT_CNN_DNN_README.md
git add MODEL_SETUP.md
git add GITHUB_UPLOAD_CHECKLIST.md
git add UPLOAD_GUIDE.md

# Add configuration
git add .gitignore

# Add entire source directory
git add src/

# Add model files (will use Git LFS)
git add models/cnn_edth_3class_improved.pt
git add models/labels_edth_3class_improved.json
git add models/training_history_improved.json

# Check what will be committed
git status
```

### Step 5: Commit and Push

```powershell
# Commit all changes
git commit -m "Initial commit: FFT-CNN-DNN deployment package

- Core challenge bot and training scripts
- Complete source code with modules
- Kaggle deployment setup scripts
- Comprehensive documentation
- Trained model with Git LFS"

# Push to GitHub (first time)
git branch -M main
git push -u origin main

# Enter your GitHub credentials when prompted
```

### Step 6: Verify Upload

```powershell
# Clone in a new directory to test
cd f:\EDTH\test
git clone https://github.com/Somnathab3/Accoustic_model_EDTH_Munich.git
cd Accoustic_model_EDTH_Munich

# Check files
Get-ChildItem -Recurse | Select-Object FullName

# Check model file
Get-Item models/cnn_edth_3class_improved.pt
```

---

## 🖥️ METHOD 2: Using GitHub Desktop (User-Friendly)

### Step 1: Install GitHub Desktop
- Download from: https://desktop.github.com/
- Install and sign in with your GitHub account

### Step 2: Add Repository

1. Open GitHub Desktop
2. File → Add Local Repository
3. Browse to: `f:\EDTH\acoustic-drone-detector`
4. Click "Add Repository"

### Step 3: Review Changes

1. Click on "Changes" tab (left sidebar)
2. You'll see all files that can be committed
3. **Important**: Uncheck these directories/files:
   - `__pycache__/`
   - `challenge_results/`
   - `test_results/`
   - `data/raw/`
   - `data/processed/`
   - `notebooks/`
   - `scripts/` (except essential ones)

### Step 4: Make First Commit

1. In the "Summary" field, type: "Initial commit: FFT-CNN-DNN deployment"
2. In the "Description" field, add details
3. Click "Commit to main"

### Step 5: Publish to GitHub

1. Click "Publish repository" button
2. Repository name: `Accoustic_model_EDTH_Munich`
3. Uncheck "Keep this code private" (or keep checked if you want it private)
4. Click "Publish repository"

### Step 6: Push Changes

1. Click "Push origin" button in the top bar
2. Wait for upload to complete

---

## 🌐 METHOD 3: GitHub Web Interface (Simplest for Small Files)

### Step 1: Create Repository (If Not Exists)

1. Go to https://github.com/Somnathab3
2. Click "New" repository button
3. Repository name: `Accoustic_model_EDTH_Munich`
4. Add description: "FFT-CNN-DNN acoustic drone detection for EDTH challenge"
5. Choose Public or Private
6. Don't initialize with README (we have our own)
7. Click "Create repository"

### Step 2: Upload Files

**Note**: This method has file size limits (25MB per file via web interface)

1. Click "uploading an existing file"
2. Drag and drop files/folders from your local directory
3. Add commit message
4. Click "Commit changes"

**For model files > 25MB**, use Method 1 (Git LFS) or upload to Kaggle Dataset instead

---

## 📦 HANDLING LARGE MODEL FILES

### Option A: Git LFS (Best for GitHub)

Already covered in Method 1, Step 3

### Option B: Kaggle Dataset (Best for Kaggle Deployment)

1. **Upload to Kaggle**:
   - Go to https://www.kaggle.com/datasets
   - Click "New Dataset"
   - Upload `cnn_edth_3class_improved.pt` and `labels_edth_3class_improved.json`
   - Title: "EDTH FFT-CNN-DNN Drone Model"
   - Make it public
   - Click "Create"

2. **Update GitHub README**:
   - Add instructions to download from Kaggle Dataset
   - Include your dataset URL

3. **Update code to support both**:
   ```python
   # In challenge_bot_fft_cnn_dnn.py or similar
   import os
   
   # Try local path first, then Kaggle dataset path
   MODEL_PATHS = [
       "models/cnn_edth_3class_improved.pt",
       "/kaggle/input/edth-fft-cnn-dnn-model/cnn_edth_3class_improved.pt"
   ]
   
   for path in MODEL_PATHS:
       if os.path.exists(path):
           MODEL_PATH = path
           break
   ```

---

## ✅ POST-UPLOAD CHECKLIST

### Verify on GitHub Website

- [ ] Go to https://github.com/Somnathab3/Accoustic_model_EDTH_Munich
- [ ] README displays correctly (rename DEPLOYMENT_README.md to README.md)
- [ ] All directories visible: `src/`, `models/`
- [ ] Core scripts present
- [ ] Model files show LFS badge (if using LFS)

### Update Repository Settings

1. **About Section** (right sidebar):
   - Click gear icon
   - Description: "FFT-CNN-DNN acoustic drone detection - Kaggle deployable"
   - Website: (optional)
   - Topics: `drone-detection`, `audio-classification`, `pytorch`, `kaggle`, `deep-learning`
   - Save changes

2. **README File**:
   - Rename `DEPLOYMENT_README.md` to `README.md` on GitHub
   - This becomes the main page

3. **License** (optional):
   - Click "Add file" → "Create new file"
   - Filename: `LICENSE`
   - Click "Choose a license template"
   - Select MIT or Apache 2.0
   - Commit

### Test the Deployment

```powershell
# In a new directory
cd f:\EDTH\test
git clone https://github.com/Somnathab3/Accoustic_model_EDTH_Munich.git
cd Accoustic_model_EDTH_Munich

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_fft_cnn_dnn.py

# Test inference (if model present)
python simple_infer.py test_audio.wav
```

---

## 🎯 KAGGLE DEPLOYMENT TEST

### Step 1: Create Kaggle Notebook

1. Go to https://www.kaggle.com/
2. Create new notebook
3. Settings:
   - Accelerator: GPU T4 x2
   - Internet: ON
   - Environment: Python 3.10+

### Step 2: Clone and Setup

```python
# Cell 1: Clone repository
!git clone https://github.com/Somnathab3/Accoustic_model_EDTH_Munich.git
%cd Accoustic_model_EDTH_Munich
!ls -la

# Cell 2: Install dependencies
!pip install -q -r requirements.txt

# Cell 3: Setup (if you have model via Kaggle Dataset)
import shutil
import os

# Add your Kaggle dataset if model file is there
# dataset_path = '/kaggle/input/your-model-dataset'
# os.makedirs('models', exist_ok=True)
# shutil.copy(f'{dataset_path}/cnn_edth_3class_improved.pt', 'models/')
# shutil.copy(f'{dataset_path}/labels_edth_3class_improved.json', 'models/')

# Cell 4: Test system
!python test_fft_cnn_dnn.py

# Cell 5: Run challenge bot
!python challenge_bot_fft_cnn_dnn.py --max-iterations 100 --delay 1.0
```

### Step 3: Verify Output

- Check that challenge bot runs without errors
- Results should be saved in `challenge_results/`
- Monitor performance in the output

---

## 🔧 TROUBLESHOOTING

### Issue: "Permission denied" when pushing

**Solution**: Setup GitHub authentication
```powershell
# Use GitHub CLI
gh auth login

# Or setup SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"
# Add to GitHub: Settings → SSH Keys → New SSH key
```

### Issue: "File is too large" (>100MB)

**Solution**: Use Git LFS or Kaggle Dataset
```powershell
git lfs install
git lfs track "models/*.pt"
git add .gitattributes
git add models/
git commit -m "Add model with LFS"
git push
```

### Issue: "Repository not found"

**Solution**: Check repository URL
```powershell
git remote -v
# If wrong, update:
git remote set-url origin https://github.com/Somnathab3/Accoustic_model_EDTH_Munich.git
```

### Issue: Model file not uploading via Git LFS

**Solution**: Check LFS status
```powershell
git lfs ls-files
# Should show your .pt files
```

---

## 📊 EXPECTED RESULTS

### Repository Size
- Without model: ~500KB
- With model (LFS): ~50MB (counted against LFS quota)

### File Count
- Core scripts: 5 files
- Source code: 35+ files
- Documentation: 5 files
- Total: ~45-50 files

### GitHub LFS Limits
- Free tier: 1GB storage, 1GB bandwidth/month
- Your model: ~50MB (well within limits)

---

## 🎉 SUCCESS CRITERIA

You've successfully uploaded when:

- ✅ Repository visible at https://github.com/Somnathab3/Accoustic_model_EDTH_Munich
- ✅ README displays correctly with instructions
- ✅ All core scripts present and viewable
- ✅ Source code directory structure intact
- ✅ Model files uploaded (via LFS or documented as Kaggle Dataset)
- ✅ Clone and run test successful on a new machine/Kaggle
- ✅ Challenge bot can run overnight on Kaggle

---

## 📞 NEED HELP?

### Resources
- Git Documentation: https://git-scm.com/doc
- Git LFS: https://git-lfs.github.com/
- GitHub Desktop: https://desktop.github.com/
- Kaggle Docs: https://www.kaggle.com/docs

### Common Commands Reference

```powershell
# Check status
git status

# See what changed
git diff

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Update from GitHub
git pull origin main

# Create new branch
git checkout -b feature-name

# Switch branches
git checkout main

# View commit history
git log --oneline
```

---

**Last Updated**: October 24, 2025
**Estimated Time**: 15-30 minutes
**Difficulty**: Moderate (with this guide: Easy!)

Good luck with your deployment! 🚀🎯
