# Running the Acoustic Drone Detector

## ✅ All Working Commands (Windows PowerShell)

### 1. Download Dataset
```powershell
cd f:\EDTH\acoustic-drone-detector
python scripts/download_data.py
```
**Status:** ✅ Working - Successfully processed 180,320 audio files

### 2. Prepare Dataset (Optional - for mel-spec caching)
```powershell
python scripts/prepare_dataset.py --in-csv data/raw/metadata_train.csv
```
**Status:** ✅ Working

### 3. Train Model
```powershell
python -m src.adrone.train --config configs/train.yaml
```
**Status:** ✅ Working - Training in progress

### 4. Run Inference (after training)
```powershell
python train.py  # Uses wrapper script
# Or directly as module:
python -m src.adrone.infer <audio_file.wav>
```

### 5. Start API Server (after training)
```powershell
uvicorn src.adrone.serve.app:app --host 0.0.0.0 --port 8000 --reload
```

## 📁 Generated Files

After running download_data.py:
- ✅ `data/raw/metadata_train.csv` - Training metadata (144,256 samples)
- ✅ `data/raw/metadata_val.csv` - Validation metadata (36,064 samples)
- ✅ `data/raw/train/0/` - Non-drone audio files (label 0)
- ✅ `data/raw/train/1/` - Drone audio files (label 1)
- ✅ `data/processed/labels.json` - Label mapping

After training:
- `models/cnn_small.pt` - Trained model weights
- `models/labels.json` - Label mapping for inference

## 🔧 Key Fixes Applied

1. **Fixed torchcodec/FFmpeg error** in download_data.py:
   - Bypassed automatic audio decoding
   - Directly accessed Apache Arrow table
   - Read audio from bytes field

2. **Fixed import errors** in all scripts:
   - Added `sys.path` manipulation for scripts
   - Use `python -m src.adrone.train` for module execution
   - Created wrapper scripts for convenience

3. **Updated Makefile** for correct module paths

## 💡 Tips

- Always run scripts from the project root: `f:\EDTH\acoustic-drone-detector`
- Training uses GPU if available, otherwise CPU
- Default config uses batch_size=32, adjust if needed for your system
- Model trains for 10 epochs by default (see configs/train.yaml)
