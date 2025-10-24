# 🚀 QUICK REFERENCE - GitHub Upload & Kaggle Deployment

## ⚡ FASTEST PATH TO DEPLOYMENT

### 1️⃣ Upload to GitHub (5 minutes)

```powershell
cd f:\EDTH\acoustic-drone-detector
git init
git remote add origin https://github.com/Somnathab3/Accoustic_model_EDTH_Munich.git
git add .
git commit -m "Initial deployment package"
git push -u origin main
```

### 2️⃣ Deploy on Kaggle (2 minutes)

```python
# Kaggle notebook
!git clone https://github.com/Somnathab3/Accoustic_model_EDTH_Munich.git
%cd Accoustic_model_EDTH_Munich
!pip install -q -r requirements.txt
!python challenge_bot_fft_cnn_dnn.py --max-iterations 10000 --delay 1.0
```

---

## 📁 KEY FILES

| File | Purpose | Size |
|------|---------|------|
| `challenge_bot_fft_cnn_dnn.py` | Main bot | ~15KB |
| `simple_infer.py` | Quick inference | ~10KB |
| `test_fft_cnn_dnn.py` | System test | ~8KB |
| `requirements.txt` | Dependencies | ~1KB |
| `models/cnn_edth_3class_improved.pt` | Model weights | ~50MB |
| `models/labels_edth_3class_improved.json` | Class labels | ~1KB |

---

## 🔧 ESSENTIAL COMMANDS

### Git Commands
```powershell
git status                  # Check status
git add .                   # Add all files
git commit -m "message"     # Commit changes
git push                    # Push to GitHub
git pull                    # Pull from GitHub
```

### Kaggle Setup
```bash
pip install -r requirements.txt   # Install dependencies
python test_fft_cnn_dnn.py        # Test system
python simple_infer.py audio.wav  # Run inference
```

### Challenge Bot
```bash
# Short test (10 iterations)
python challenge_bot_fft_cnn_dnn.py --max-iterations 10 --delay 1.0

# Overnight run
python challenge_bot_fft_cnn_dnn.py --max-iterations 10000 --delay 1.0
```

---

## 🎯 DOCUMENTATION MAP

| Document | Use Case |
|----------|----------|
| `DEPLOYMENT_SUMMARY.md` | Overview & quick start |
| `UPLOAD_GUIDE.md` | Step-by-step upload |
| `MODEL_SETUP.md` | Model file handling |
| `GITHUB_UPLOAD_CHECKLIST.md` | Complete file list |
| `FFT_CNN_DNN_README.md` | Architecture details |

---

## ⚠️ IMPORTANT NOTES

### Model File Options:
1. **Git LFS** (if < 100MB) - Stays on GitHub
2. **Kaggle Dataset** (recommended) - Separate from code

### Kaggle Limits:
- **GPU Runtime**: 9 hours max
- **CPU Runtime**: 12 hours max
- **Internet**: Must be enabled
- **GPU**: T4 x2 recommended

### File Exclusions:
❌ Don't upload:
- `__pycache__/`
- `challenge_results/`
- `data/raw/`
- Development files

---

## 🔗 LINKS

- **GitHub Repo**: https://github.com/Somnathab3/Accoustic_model_EDTH_Munich
- **Git LFS**: https://git-lfs.github.com/
- **Kaggle Docs**: https://www.kaggle.com/docs
- **Challenge**: https://edth.helsing.codes

---

## 🆘 QUICK TROUBLESHOOTING

| Problem | Solution |
|---------|----------|
| "Repository not found" | Check remote URL with `git remote -v` |
| "File too large" | Use Git LFS or Kaggle Dataset |
| "Permission denied" | Setup SSH or use `gh auth login` |
| "Module not found" | Run `pip install -r requirements.txt` |
| "Model not found" | Check path or setup Kaggle Dataset |

---

## 📞 SUPPORT

Need help? Check these in order:
1. `UPLOAD_GUIDE.md` - Detailed instructions
2. `DEPLOYMENT_SUMMARY.md` - Complete overview
3. `MODEL_SETUP.md` - Model file issues

---

**Quick Start**: Read `UPLOAD_GUIDE.md` → Upload → Deploy on Kaggle → Run overnight! 🌙
