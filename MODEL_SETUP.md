# Model Files Setup Guide

## Overview

The trained FFT-CNN-DNN model files are required to run inference and the challenge bot. These files are typically too large to be directly included in the GitHub repository.

## Required Model Files

1. **Model Checkpoint**: `cnn_edth_3class_improved.pt` (~50MB)
   - Contains the trained neural network weights
   - Located in `models/` directory

2. **Labels File**: `labels_edth_3class_improved.json` (~1KB)
   - Contains the class label mappings
   - Located in `models/` directory

## Option 1: Using Git LFS (Recommended for GitHub)

If your model file is under 100MB, use Git Large File Storage:

### Setup Git LFS

```bash
# Install Git LFS (one-time setup)
git lfs install

# Track the model file
git lfs track "models/*.pt"
git lfs track "models/*.pth"

# Add .gitattributes
git add .gitattributes

# Add and commit model files
git add models/cnn_edth_3class_improved.pt
git add models/labels_edth_3class_improved.json
git commit -m "Add trained model files"
git push
```

### Clone with LFS

```bash
git lfs install
git clone https://github.com/Somnathab3/Accoustic_model_EDTH_Munich.git
cd Accoustic_model_EDTH_Munich
git lfs pull
```

## Option 2: Kaggle Dataset (Recommended for Kaggle Deployment)

Upload model files as a Kaggle dataset for easy access in notebooks:

### Creating the Dataset

1. Go to https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload files:
   - `cnn_edth_3class_improved.pt`
   - `labels_edth_3class_improved.json`
   - (Optional) `training_history_improved.json`
4. Set title: "EDTH FFT-CNN-DNN Drone Detection Model"
5. Add description and tags
6. Click "Create"

### Using in Kaggle Notebook

```python
# In your Kaggle notebook
import shutil
import os

# Create models directory
os.makedirs('models', exist_ok=True)

# Copy from Kaggle dataset (replace 'your-username' and 'dataset-name')
dataset_path = '/kaggle/input/edth-fft-cnn-dnn-model'  # Your dataset path

# Copy model files
shutil.copy(
    f'{dataset_path}/cnn_edth_3class_improved.pt',
    'models/cnn_edth_3class_improved.pt'
)
shutil.copy(
    f'{dataset_path}/labels_edth_3class_improved.json',
    'models/labels_edth_3class_improved.json'
)

print("✓ Model files copied successfully")
```

## Option 3: Direct Download (For External Hosting)

If hosting on Google Drive, Dropbox, or similar:

### Upload to Cloud Storage

1. Upload `cnn_edth_3class_improved.pt` to your cloud storage
2. Get a direct download link
3. Create a download script

### Download Script Example

```python
# download_models.py
import requests
import os
from pathlib import Path

MODEL_URLS = {
    'cnn_edth_3class_improved.pt': 'YOUR_DIRECT_DOWNLOAD_URL',
    'labels_edth_3class_improved.json': 'YOUR_LABELS_URL'
}

def download_file(url, destination):
    """Download file from URL"""
    print(f"Downloading {destination}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"✓ Downloaded {destination}")

def main():
    # Create models directory
    Path('models').mkdir(exist_ok=True)
    
    # Download model files
    for filename, url in MODEL_URLS.items():
        destination = f'models/{filename}'
        if not os.path.exists(destination):
            download_file(url, destination)
        else:
            print(f"✓ {destination} already exists")

if __name__ == '__main__':
    main()
```

Run on Kaggle:
```bash
python download_models.py
```

## Option 4: Manual Upload (Simple but Manual)

### On Local Machine

1. Locate model files in your local project:
   ```
   f:\EDTH\acoustic-drone-detector\models\cnn_edth_3class_improved.pt
   f:\EDTH\acoustic-drone-detector\models\labels_edth_3class_improved.json
   ```

2. Upload to GitHub (if < 100MB):
   - Navigate to repository on GitHub
   - Click "Add file" → "Upload files"
   - Drag and drop model files
   - Commit changes

### On Kaggle

1. In Kaggle notebook, click "Add Data"
2. Go to "Upload" tab
3. Upload model files directly
4. Access via `/kaggle/input/` path

## Verifying Model Files

After setting up, verify the files are accessible:

```python
import os
import torch
import json

# Check file existence
model_path = 'models/cnn_edth_3class_improved.pt'
labels_path = 'models/labels_edth_3class_improved.json'

print("Checking model files...")
print(f"Model exists: {os.path.exists(model_path)}")
print(f"Labels exist: {os.path.exists(labels_path)}")

if os.path.exists(model_path):
    # Check model size
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")
    
    # Try loading
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print("✓ Model loads successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")

if os.path.exists(labels_path):
    # Check labels
    try:
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        print(f"✓ Labels loaded: {list(labels.keys())}")
    except Exception as e:
        print(f"✗ Error loading labels: {e}")
```

## Model File Locations

After setup, your directory structure should look like:

```
Accoustic_model_EDTH_Munich/
├── models/
│   ├── cnn_edth_3class_improved.pt      ← Required
│   ├── labels_edth_3class_improved.json  ← Required
│   └── training_history_improved.json    ← Optional
├── src/
├── challenge_bot_fft_cnn_dnn.py
├── simple_infer.py
└── requirements.txt
```

## Troubleshooting

### "Model file not found"

```python
# Check current directory
import os
print("Current directory:", os.getcwd())
print("Files in models/:", os.listdir('models') if os.path.exists('models') else 'models/ not found')
```

### "Out of memory when loading model"

```python
# Load model to CPU first
checkpoint = torch.load(model_path, map_location='cpu')
```

### "Corrupted model file"

Re-download or re-upload the model file. Verify the file size matches the original.

## Quick Reference

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| Git LFS | GitHub repo | Version control | Setup required, GitHub LFS limits |
| Kaggle Dataset | Kaggle notebooks | Easy sharing, persistent | Kaggle-specific |
| Direct Download | Any platform | Universal | Requires external hosting |
| Manual Upload | Quick testing | Simple | Not automated |

## Support

If you encounter issues:
1. Check file permissions
2. Verify file sizes match
3. Test with `simple_infer.py` script
4. Check Kaggle/GitHub logs for errors

---

**Note**: Always keep a backup of your trained model files!
