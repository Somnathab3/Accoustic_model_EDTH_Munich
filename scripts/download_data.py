import argparse, os, csv, json
from pathlib import Path
from typing import Optional
from datasets import load_dataset, ClassLabel
import soundfile as sf
from tqdm import tqdm
import random
import librosa
import numpy as np
import io

CAND_LABELS = ["label", "labels", "category", "class", "target"]

def infer_label_col(features) -> Optional[str]:
    for k in CAND_LABELS:
        if k in features: return k
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="data")
    ap.add_argument("--dataset", type=str, default="geronimobasso/drone-audio-detection-samples")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--target-sr", type=int, default=16000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "processed").mkdir(parents=True, exist_ok=True)

    # Load dataset WITHOUT automatic audio decoding to avoid torchcodec issues
    print("Loading dataset (this may take a moment)...")
    ds_dict = load_dataset(args.dataset)
    
    # We'll export a single combined "train" CSV and auto-split into train/val here.
    split_name = next(iter(ds_dict.keys()))  # use first split if only one
    ds = ds_dict.get("train", ds_dict[split_name])
    
    label_col = infer_label_col(ds.features)
    if label_col is None:
        raise RuntimeError(f"Could not infer label column from features: {list(ds.features.keys())}")

    label_names = None
    if isinstance(ds.features[label_col], ClassLabel):
        label_names = ds.features[label_col].names

    rows = []
    raw_root = out_dir / "raw" / "train"
    raw_root.mkdir(parents=True, exist_ok=True)

    # Access underlying arrow table directly to avoid audio decoding issues
    print("Processing audio files...")
    table = ds.data
    audio_col_idx = ds.column_names.index("audio")
    label_col_idx = ds.column_names.index(label_col)
    
    for i in tqdm(range(len(table)), desc="Exporting WAVs"):
        try:
            # Get raw audio data from arrow table
            audio_data = table.column(audio_col_idx)[i].as_py()
            label_val = table.column(label_col_idx)[i].as_py()
            
            # Handle audio data - prioritize bytes over path
            y = None
            sr = None
            
            if isinstance(audio_data, dict):
                # Try bytes first (most reliable for HuggingFace datasets)
                if "bytes" in audio_data and audio_data["bytes"]:
                    y, sr = sf.read(io.BytesIO(audio_data["bytes"]), always_2d=False)
                # Try array if already decoded
                elif "array" in audio_data and "sampling_rate" in audio_data:
                    y = np.array(audio_data["array"])
                    sr = audio_data["sampling_rate"]
                # Try path (check if absolute or construct relative)
                elif "path" in audio_data and audio_data["path"]:
                    audio_path = audio_data["path"]
                    # Check if absolute path exists
                    if os.path.isabs(audio_path) and os.path.exists(audio_path):
                        y, sr = sf.read(audio_path, always_2d=False)
                    # If not absolute, try relative to cache or skip
                    elif not os.path.isabs(audio_path):
                        # HuggingFace datasets cache - bytes should be available
                        if "bytes" not in audio_data or not audio_data["bytes"]:
                            print(f"\nSkipping sample {i}: path not found {audio_path}")
                        continue
                    else:
                        print(f"\nSkipping sample {i}: path not found {audio_path}")
                        continue
                else:
                    print(f"\nSkipping sample {i}: no valid audio data")
                    continue
            elif isinstance(audio_data, str):
                # Direct path string
                if os.path.isabs(audio_data) and os.path.exists(audio_data):
                    y, sr = sf.read(audio_data, always_2d=False)
                else:
                    print(f"\nSkipping sample {i}: path not found {audio_data}")
                    continue
            else:
                print(f"\nSkipping sample {i}: unexpected audio type {type(audio_data)}")
                continue
            
            if y is None or sr is None:
                print(f"\nSkipping sample {i}: failed to load audio")
                continue
            
            # Convert to mono if stereo
            if y.ndim > 1:
                y = y.mean(axis=1)
                
            # Resample if needed
            if sr != args.target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=args.target_sr)
                sr = args.target_sr
                
            # Get label
            label = label_names[label_val] if (label_names and isinstance(label_val, int)) else str(label_val)
            safe_label = "unknown" if not label or label.lower() in ("none","nan") else label

            # Save audio file
            cls_dir = raw_root / safe_label
            cls_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{i:07d}.wav"
            fpath = cls_dir / fname
            sf.write(fpath, y, sr)
            rows.append({"split":"train","path":str(fpath), "label":safe_label, "sr":sr, "duration_s":len(y)/sr})
            
        except Exception as e:
            print(f"\nError processing sample {i}: {e}")
            continue
    
    if not rows:
        raise RuntimeError("No audio files were successfully processed!")
    
    print(f"\nSuccessfully processed {len(rows)} audio files")

    # stratified split
    random.seed(42)
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    train_rows, val_rows = [], []
    for lab, lst in by_label.items():
        random.shuffle(lst)
        n_val = max(1, int(len(lst)*args.val_ratio))
        val_rows.extend(lst[:n_val]); train_rows.extend(lst[n_val:])

    def write_csv(name, data):
        mpath = out_dir / "raw" / f"metadata_{name}.csv"
        with open(mpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["split","path","label","sr","duration_s"])
            w.writeheader(); w.writerows(data)

    write_csv("train", train_rows)
    write_csv("val", val_rows)

    # write labels.json
    labels = sorted(set(r["label"] for r in rows))
    with open(out_dir / "processed" / "labels.json", "w") as f:
        json.dump({"labels": labels}, f, indent=2)

if __name__ == "__main__":
    main()
