"""
Transfer correctly predicted audio samples to training/validation dataset.
Only samples with non-zero scores are considered correct.
AUTOMATED VERSION - No user confirmation required.
"""
import json
import pandas as pd
import shutil
from pathlib import Path
import os

# Define paths
CHALLENGE_RESULTS_DIR = Path("F:/EDTH/acoustic-drone-detector/challenge_results")
AUDIO_SAMPLES_DIR = CHALLENGE_RESULTS_DIR / "audio_samples"
RESULTS_CSV = CHALLENGE_RESULTS_DIR / "results.csv"
RESULTS_JSONL = CHALLENGE_RESULTS_DIR / "results.jsonl"

# Define target directories for training data
DATA_DIR = Path("F:/EDTH/acoustic-drone-detector/data")
TRAIN_DIR = DATA_DIR / "edth_prepared" / "train"
VAL_DIR = DATA_DIR / "edth_prepared" / "val"

def load_results():
    """Load results from CSV and JSONL files."""
    print("Loading results from CSV and JSONL files...")
    
    # Load CSV
    df_csv = pd.read_csv(RESULTS_CSV)
    print(f"Loaded {len(df_csv)} entries from CSV")
    
    # Load JSONL
    jsonl_data = []
    with open(RESULTS_JSONL, 'r') as f:
        for line in f:
            if line.strip():
                jsonl_data.append(json.loads(line))
    df_jsonl = pd.DataFrame(jsonl_data)
    print(f"Loaded {len(df_jsonl)} entries from JSONL")
    
    return df_csv, df_jsonl

def analyze_correct_predictions(df_csv, df_jsonl):
    """Analyze which predictions were correct based on non-zero scores."""
    print("\n" + "="*80)
    print("ANALYZING CORRECT PREDICTIONS")
    print("="*80)
    
    # From CSV: filter entries with non-zero scores
    correct_csv = df_csv[df_csv['score_awarded'] > 0].copy()
    print(f"\nFrom CSV: Found {len(correct_csv)} entries with non-zero scores")
    
    # From JSONL: filter entries with non-zero scores
    correct_jsonl = df_jsonl[df_jsonl['score_awarded'] > 0].copy()
    print(f"From JSONL: Found {len(correct_jsonl)} entries with non-zero scores")
    
    # Print score statistics
    if len(correct_csv) > 0:
        print(f"\nCSV Score Statistics:")
        print(f"  Total score from correct predictions: {correct_csv['score_awarded'].sum()}")
        print(f"  Average score per correct prediction: {correct_csv['score_awarded'].mean():.2f}")
        print(f"  Score range: {correct_csv['score_awarded'].min()} - {correct_csv['score_awarded'].max()}")
    
    if len(correct_jsonl) > 0:
        print(f"\nJSONL Score Statistics:")
        print(f"  Total score from correct predictions: {correct_jsonl['score_awarded'].sum()}")
        print(f"  Average score per correct prediction: {correct_jsonl['score_awarded'].mean():.2f}")
        print(f"  Score range: {correct_jsonl['score_awarded'].min()} - {correct_jsonl['score_awarded'].max()}")
    
    # Show prediction distribution for correct predictions
    if len(correct_csv) > 0:
        print(f"\nCSV - Prediction distribution (correct only):")
        print(correct_csv['predicted'].value_counts())
    
    if len(correct_jsonl) > 0:
        print(f"\nJSONL - Prediction distribution (correct only):")
        print(correct_jsonl['prediction'].value_counts())
    
    return correct_csv, correct_jsonl

def get_audio_files_to_transfer(correct_csv, correct_jsonl):
    """Get list of audio files to transfer with their predicted labels."""
    print("\n" + "="*80)
    print("IDENTIFYING AUDIO FILES TO TRANSFER")
    print("="*80)
    
    files_to_transfer = {}
    
    # From JSONL (has audio_file field)
    for _, row in correct_jsonl.iterrows():
        audio_file = row.get('audio_file')
        prediction = row.get('prediction')
        confidence = row.get('confidence')
        score = row.get('score_awarded')
        
        if audio_file and prediction:
            # Convert to Path object
            audio_path = Path(audio_file)
            if audio_path.exists():
                filename = audio_path.name
                if filename not in files_to_transfer:
                    files_to_transfer[filename] = {
                        'path': audio_path,
                        'label': prediction,
                        'confidence': confidence,
                        'score': score,
                        'challenge_id': row.get('challenge_id')
                    }
    
    print(f"\nFound {len(files_to_transfer)} unique audio files to transfer")
    
    # Show breakdown by label
    label_counts = {}
    for file_info in files_to_transfer.values():
        label = file_info['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\nBreakdown by predicted label:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} files")
    
    return files_to_transfer

def transfer_audio_files(files_to_transfer, train_ratio=0.8):
    """Transfer audio files to training and validation directories."""
    print("\n" + "="*80)
    print("TRANSFERRING AUDIO FILES")
    print("="*80)
    
    # Create target directories if they don't exist
    for label in ['background', 'drone', 'helicopter']:
        (TRAIN_DIR / label).mkdir(parents=True, exist_ok=True)
        (VAL_DIR / label).mkdir(parents=True, exist_ok=True)
    
    # Sort files by label for consistent splitting
    files_by_label = {}
    for filename, info in files_to_transfer.items():
        label = info['label']
        if label not in files_by_label:
            files_by_label[label] = []
        files_by_label[label].append((filename, info))
    
    transfer_summary = {
        'train': {'background': 0, 'drone': 0, 'helicopter': 0},
        'val': {'background': 0, 'drone': 0, 'helicopter': 0}
    }
    
    # Transfer files
    for label, files in files_by_label.items():
        print(f"\nProcessing {len(files)} files for label '{label}'...")
        
        # Split into train and val
        split_idx = int(len(files) * train_ratio)
        train_files = files[:split_idx]
        val_files = files[split_idx:]
        
        print(f"  Train: {len(train_files)}, Val: {len(val_files)}")
        
        # Transfer train files
        for filename, info in train_files:
            src = info['path']
            dst = TRAIN_DIR / label / filename
            
            if not dst.exists():
                shutil.copy2(src, dst)
                transfer_summary['train'][label] += 1
                print(f"  ✓ Copied to train/{label}: {filename} (conf: {info['confidence']:.4f}, score: {info['score']})")
            else:
                print(f"  ⊙ Already exists in train/{label}: {filename}")
        
        # Transfer val files
        for filename, info in val_files:
            src = info['path']
            dst = VAL_DIR / label / filename
            
            if not dst.exists():
                shutil.copy2(src, dst)
                transfer_summary['val'][label] += 1
                print(f"  ✓ Copied to val/{label}: {filename} (conf: {info['confidence']:.4f}, score: {info['score']})")
            else:
                print(f"  ⊙ Already exists in val/{label}: {filename}")
    
    return transfer_summary

def print_transfer_summary(transfer_summary):
    """Print summary of transferred files."""
    print("\n" + "="*80)
    print("TRANSFER SUMMARY")
    print("="*80)
    
    print("\nFiles transferred to TRAINING set:")
    for label, count in sorted(transfer_summary['train'].items()):
        print(f"  {label}: {count} files")
    print(f"  Total: {sum(transfer_summary['train'].values())} files")
    
    print("\nFiles transferred to VALIDATION set:")
    for label, count in sorted(transfer_summary['val'].items()):
        print(f"  {label}: {count} files")
    print(f"  Total: {sum(transfer_summary['val'].values())} files")
    
    print(f"\nGRAND TOTAL: {sum(transfer_summary['train'].values()) + sum(transfer_summary['val'].values())} files transferred")

def main():
    """Main execution function."""
    print("="*80)
    print("TRANSFER CORRECT CHALLENGE SAMPLES TO TRAINING/VALIDATION DATASET")
    print("="*80)
    print("\nCriteria: Only samples with non-zero scores are considered correct")
    print(f"Source: {AUDIO_SAMPLES_DIR}")
    print(f"Destination: {DATA_DIR / 'edth_prepared'}")
    print("\n⚠ AUTOMATED MODE - Proceeding without confirmation")
    
    # Load results
    df_csv, df_jsonl = load_results()
    
    # Analyze correct predictions
    correct_csv, correct_jsonl = analyze_correct_predictions(df_csv, df_jsonl)
    
    # Get audio files to transfer
    files_to_transfer = get_audio_files_to_transfer(correct_csv, correct_jsonl)
    
    if len(files_to_transfer) == 0:
        print("\n⚠ No files to transfer. Exiting.")
        return
    
    # Transfer files (no confirmation needed)
    print(f"\n📦 Transferring {len(files_to_transfer)} files...")
    transfer_summary = transfer_audio_files(files_to_transfer, train_ratio=0.8)
    
    # Print summary
    print_transfer_summary(transfer_summary)
    
    print("\n" + "="*80)
    print("TRANSFER COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the transferred files in the train and val directories")
    print("2. Retrain your model with the augmented dataset")
    print("3. Evaluate model performance on the test set")

if __name__ == "__main__":
    main()
