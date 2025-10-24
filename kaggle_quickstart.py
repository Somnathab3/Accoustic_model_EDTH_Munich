"""
Kaggle Quick Start Example
Copy this into a Kaggle notebook cell and run
"""

# 1. Clone repository
!git clone https://github.com/Somnathab3/edth-acoustic-drone-detector.git
%cd edth-acoustic-drone-detector

# 2. Install dependencies
!pip install -q torch torchaudio librosa scikit-learn soundfile tqdm requests

# 3. Check GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")

# 4. Quick inference test
from sota_inference import AcousticDroneClassifier

# Initialize classifier (auto-detects best model)
classifier = AcousticDroneClassifier(
    model_path='models/panns_final.pt',
    labels_path='models/labels_current.json'
)

print("✓ Model loaded successfully!")
print(f"Device: {classifier.device}")
print(f"Classes: {classifier.class_names}")

# 5. Run challenge bot
print("\n" + "="*60)
print("Starting Challenge Bot...")
print("="*60 + "\n")

!python sota_challenge_bot.py --max-iterations 100 --delay 0.5

# 6. Analyze results
print("\n" + "="*60)
print("Analyzing Results...")
print("="*60 + "\n")

!python analyze_results.py

# 7. Save results to Kaggle output
import shutil
!mkdir -p /kaggle/working/challenge_results
!cp -r challenge_results/* /kaggle/working/challenge_results/

print("\n✓ Results saved to /kaggle/working/challenge_results/")
print("✓ Download results.csv from Output tab")
