import argparse, csv, json, os, sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adrone.utils.audio_io import load_wav
from src.adrone.features.melspec import melspec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", required=True)
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n-mels", type=int, default=64)
    ap.add_argument("--n-fft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=320)
    ap.add_argument("--win-sec", type=float, default=2.0)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    # This script is optional; cache mel-specs if desired
    # (kept minimal to avoid ballooning repo size)
    print("Optional: implement mel caching here if needed.")

if __name__ == "__main__":
    main()
