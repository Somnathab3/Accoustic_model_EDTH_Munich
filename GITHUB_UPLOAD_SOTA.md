# 🚀 GitHub Upload Guide - SOTA Model

## Quick Upload Checklist

### ✅ Essential Files to Upload

#### Core Scripts (7 files)
- [ ] `sota_challenge_bot.py` - Main challenge bot ⭐
- [ ] `sota_inference.py` - Inference module
- [ ] `train_sota_model.py` - Training script
- [ ] `validate_model.py` - Validation script
- [ ] `analyze_results.py` - Results analyzer
- [ ] `quick_train.py` - Quick training wrapper
- [ ] `kaggle_quickstart.py` - Kaggle quick start

#### Source Code (Complete src/ directory)
- [ ] `src/adrone/__init__.py`
- [ ] `src/adrone/preprocessing/audio_transforms.py`
- [ ] `src/adrone/models/acoustic_models.py`
- [ ] `src/adrone/data/acoustic_dataset.py`
- [ ] `src/adrone/training/losses.py`
- [ ] `src/adrone/evaluation/metrics.py`
- [ ] `src/adrone/serve/challenge_handler.py`
- [ ] All `__init__.py` files in subdirectories

#### Configuration Files (4 files)
- [ ] `requirements.txt` - Full dependencies
- [ ] `requirements_kaggle.txt` - Kaggle-specific
- [ ] `.gitignore` - Exclude unnecessary files
- [ ] `.gitattributes` or `.gitattributes_sota` - Git LFS config

#### Setup Scripts (2 files)
- [ ] `setup_kaggle_sota.sh` - Bash setup
- [ ] `setup_kaggle_sota.ps1` - PowerShell setup

#### Documentation (6+ files)
- [ ] `README_DEPLOYMENT.md` - Main README (rename to README.md)
- [ ] `CHALLENGE_READY.md` - Quick start guide
- [ ] `QUICK_COMMANDS.md` - Command reference
- [ ] `TIMING_STRATEGY_EXPLAINED.md` - Timing strategy
- [ ] `CHALLENGE_BOT_GUIDE.md` - Bot guide
- [ ] `SMART_TIMING_GUIDE.md` - Smart timing

#### Model Files (3 files) ⚠️ Large
- [ ] `models/panns_final.pt` - Fully trained model (~20MB)
- [ ] `models/labels_current.json` - Class labels ⭐ Essential
- [ ] `models/config.json` - Training config

### ❌ Files to Exclude

- ❌ `data/` directories (too large)
- ❌ `challenge_results/` (user-generated)
- ❌ `test_results/` (user-generated)
- ❌ `__pycache__/` (Python cache)
- ❌ `.vscode/`, `.idea/` (IDE configs)
- ❌ Old/deprecated scripts
- ❌ Audio sample files (*.wav, *.mp3)

---

## 📋 Step-by-Step Upload Guide

### Method 1: GitHub Desktop (Easiest)

#### 1. Install GitHub Desktop
Download from: https://desktop.github.com/

#### 2. Create Repository
- Open GitHub Desktop
- File → New Repository
- Name: `edth-acoustic-drone-detector`
- Local Path: `F:\EDTH\acoustic-drone-detector`
- Click "Create Repository"

#### 3. Add Files
GitHub Desktop will automatically detect all files. Review:
- Ensure `.gitignore` is working (no `__pycache__`, etc.)
- Check that model files are included

#### 4. Commit
- Summary: "Initial commit: SOTA acoustic drone detector"
- Description: "Production-ready model with PANNs architecture, smart timing, and comprehensive documentation"
- Click "Commit to main"

#### 5. Publish to GitHub
- Click "Publish repository"
- Uncheck "Keep this code private" (if you want it public)
- Click "Publish repository"

### Method 2: Git Command Line

#### 1. Navigate to Directory
```powershell
cd F:\EDTH\acoustic-drone-detector
```

#### 2. Initialize Git
```powershell
git init
```

#### 3. Add Remote
```powershell
git remote add origin https://github.com/Somnathab3/edth-acoustic-drone-detector.git
```

#### 4. Setup Git LFS (for large model files)
```powershell
# Install Git LFS first: https://git-lfs.github.com/
git lfs install
git lfs track "models/*.pt"
git add .gitattributes
```

#### 5. Add Files
```powershell
git add .
```

#### 6. Commit
```powershell
git commit -m "Initial commit: SOTA acoustic drone detector"
```

#### 7. Push
```powershell
git branch -M main
git push -u origin main
```

### Method 3: Web Upload (For Small Files Only)

⚠️ **Not recommended** for this project (files too large)

But if model is hosted elsewhere:
1. Go to https://github.com/new
2. Create repository
3. Click "uploading an existing file"
4. Drag and drop files (max 100MB total per operation)

---

## 🔧 Handling Large Model Files

### Option A: Git LFS (Recommended for GitHub)

```powershell
# Install Git LFS
# Download from: https://git-lfs.github.com/

# Setup
git lfs install
git lfs track "models/*.pt"
git add .gitattributes

# Add and commit
git add models/panns_final.pt
git commit -m "Add trained model"
git push
```

**Pros:**
- Files on GitHub
- Easy cloning
- Version control for models

**Cons:**
- GitHub LFS quota (1GB free, then paid)
- Slower push/pull

### Option B: Kaggle Dataset (Recommended for Kaggle)

1. **Create Kaggle Dataset**
   - Go to https://www.kaggle.com/datasets
   - Click "New Dataset"
   - Upload `models/panns_final.pt` and `models/labels_current.json`
   - Title: "EDTH SOTA Acoustic Drone Model"
   - Make public

2. **Update Code**
   ```python
   # In Kaggle notebook, before running
   !cp /kaggle/input/edth-sota-model/panns_final.pt models/
   !cp /kaggle/input/edth-sota-model/labels_current.json models/
   ```

3. **Upload Code Only to GitHub**
   - Exclude model files
   - Update README with Kaggle Dataset link

**Pros:**
- Free unlimited storage
- Fast on Kaggle
- No Git LFS quota

**Cons:**
- Two-step setup (clone + dataset)
- Model separate from code

### Option C: External Hosting

Upload to:
- Google Drive
- Dropbox
- Hugging Face Model Hub
- AWS S3

Then add download script:
```python
# download_model.py
import requests
import os

def download_model():
    url = "https://your-url/panns_final.pt"
    os.makedirs('models', exist_ok=True)
    
    print("Downloading model...")
    response = requests.get(url, stream=True)
    with open('models/panns_final.pt', 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("✓ Model downloaded")

if __name__ == '__main__':
    download_model()
```

---

## 📝 Pre-Upload Checklist

### 1. Clean Up
```powershell
# Remove unnecessary files
Remove-Item -Recurse -Force challenge_results\audio_samples\*
Remove-Item -Recurse -Force __pycache__\
Remove-Item -Recurse -Force test_results\
```

### 2. Test Locally
```powershell
# Test in clean directory
cd F:\EDTH\test_deployment
git clone F:\EDTH\acoustic-drone-detector
cd acoustic-drone-detector

# Test installation
pip install -r requirements_kaggle.txt

# Test scripts
python -c "from sota_inference import AcousticDroneClassifier; print('✓ Imports OK')"
```

### 3. Update README
```powershell
# Rename deployment README to main README
Move-Item README_DEPLOYMENT.md README.md -Force
```

### 4. Check File Sizes
```powershell
# Check model size
Get-Item models\panns_final.pt | Select-Object Name, @{Name="SizeMB";Expression={[math]::Round($_.Length / 1MB, 2)}}
```

### 5. Verify .gitignore
```powershell
# Test git status (should not show data/ or results/)
git status
```

---

## 🎯 Recommended Approach

### For Your Use Case: **Git LFS + GitHub**

1. **Why:**
   - Model is ~20MB (within reasonable size)
   - One-step clone for users
   - Easy version control

2. **Steps:**
```powershell
cd F:\EDTH\acoustic-drone-detector

# Rename main README
Move-Item README_DEPLOYMENT.md README.md -Force

# Setup Git LFS
git lfs install
git lfs track "models/*.pt"

# Initialize and add
git init
git remote add origin https://github.com/Somnathab3/edth-acoustic-drone-detector.git
git add .
git commit -m "Initial commit: SOTA acoustic drone detector with PANNs model"

# Push
git branch -M main
git push -u origin main
```

3. **Result:**
   - All files on GitHub
   - One-command clone on Kaggle
   - Ready to use immediately

---

## 🚀 After Upload - Kaggle Test

### 1. Create Kaggle Notebook
```python
# Cell 1: Clone repository
!git clone https://github.com/Somnathab3/edth-acoustic-drone-detector.git
%cd edth-acoustic-drone-detector
```

### 2. Setup
```python
# Cell 2: Install dependencies
!pip install -q -r requirements_kaggle.txt
```

### 3. Verify
```python
# Cell 3: Test
!python -c "from sota_inference import AcousticDroneClassifier; print('✓ Ready!')"
```

### 4. Run
```python
# Cell 4: Run challenge bot
!python sota_challenge_bot.py --max-iterations 100 --delay 0.5
```

---

## 📊 Repository Structure (What Users Will See)

```
edth-acoustic-drone-detector/
├── README.md                       ⭐ Main documentation
├── requirements_kaggle.txt         ⭐ Install first
├── sota_challenge_bot.py           ⭐ Run this
├── setup_kaggle_sota.sh           ⭐ Or run this
│
├── Core Scripts/
│   ├── sota_inference.py
│   ├── train_sota_model.py
│   ├── validate_model.py
│   ├── analyze_results.py
│   └── kaggle_quickstart.py
│
├── Source Code/
│   └── src/adrone/
│       ├── preprocessing/
│       ├── models/
│       ├── data/
│       ├── training/
│       ├── evaluation/
│       └── serve/
│
├── Models/
│   ├── panns_final.pt            ⭐ Main model
│   ├── labels_current.json       ⭐ Class labels
│   └── config.json
│
└── Documentation/
    ├── CHALLENGE_READY.md
    ├── QUICK_COMMANDS.md
    ├── TIMING_STRATEGY_EXPLAINED.md
    └── (more guides)
```

---

## 💡 Pro Tips

1. **Test Before Push**: Always test in clean directory
2. **Commit Messages**: Be descriptive
3. **Branch Protection**: Consider protecting main branch
4. **Release Tags**: Tag versions (v1.0, v1.1, etc.)
5. **GitHub Actions**: Add CI/CD later for auto-testing

---

## ✅ Final Checklist

Before pushing:
- [ ] All essential files present
- [ ] No sensitive data (API tokens, passwords)
- [ ] .gitignore working correctly
- [ ] Model files handled (LFS or external)
- [ ] README.md is comprehensive
- [ ] Local test successful
- [ ] File sizes reasonable (<100MB per file)

After pushing:
- [ ] Repository visible on GitHub
- [ ] README displays correctly
- [ ] Files downloadable
- [ ] Kaggle clone test successful
- [ ] Challenge bot runs successfully

---

## 🆘 Troubleshooting

### "File too large"
- Use Git LFS
- Or use Kaggle Dataset
- Or split into chunks

### "Permission denied"
- Check GitHub credentials
- Use HTTPS instead of SSH
- Or use GitHub Desktop

### ".gitignore not working"
```powershell
git rm -r --cached .
git add .
git commit -m "Fix gitignore"
```

### "Model not found after clone"
- Check if Git LFS is installed
- Run `git lfs pull`
- Or add download script

---

## 📧 Support

Issues? Check:
1. GitHub Documentation: https://docs.github.com
2. Git LFS Guide: https://git-lfs.github.com/
3. Kaggle Docs: https://www.kaggle.com/docs

---

**Ready to upload? Follow the steps above and you're set!** 🚀
