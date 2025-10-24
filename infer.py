#!/usr/bin/env python
"""
Wrapper script to run inference from the project root.
Usage: python infer.py <audio_file.wav>
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adrone.infer import InferenceModel
import argparse

def main():
    ap = argparse.ArgumentParser(description="Run inference on an audio file")
    ap.add_argument("audio_file", type=str, help="Path to audio file")
    ap.add_argument("--model", type=str, default="models/cnn_small.pt", help="Path to model checkpoint")
    ap.add_argument("--labels", type=str, default="models/labels.json", help="Path to labels JSON")
    args = ap.parse_args()
    
    print(f"Loading model from {args.model}...")
    model = InferenceModel(model_path=args.model, labels_path=args.labels)
    
    print(f"Running inference on {args.audio_file}...")
    result = model.predict_path(args.audio_file)
    
    print("\nPrediction:")
    for label, prob in result.items():
        print(f"  {label}: {prob:.4f}")

if __name__ == "__main__":
    main()
