"""
Add challenging but correct samples from challenge results to training dataset.
Criteria: correct_inferred=True AND confidence < 0.75 (75%)
These are samples the model got right but with low confidence - good for hard negative mining.
"""
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm

# Paths
RESULTS_CSV = Path("F:/EDTH/acoustic-drone-detector/challenge_results/results.csv")
CHALLENGE_AUDIO_DIR = Path("F:/EDTH/acoustic-drone-detector/challenge_results/audio_samples")
COMBINED_TRAIN = Path("F:/EDTH/acoustic-drone-detector/data/combined_dataset/train")

def add_challenging_samples():
    """Add low-confidence but correct samples to training dataset"""
    
    print("="*80)
    print("ADDING CHALLENGING SAMPLES TO TRAINING DATASET")
    print("Criteria: correct_inferred=True AND confidence < 75%")
    print("="*80)
    
    # Load results
    print(f"\n📊 Loading results from: {RESULTS_CSV}")
    df = pd.read_csv(RESULTS_CSV)
    
    print(f"Total challenge attempts: {len(df)}")
    
    # Filter for correct predictions
    correct_df = df[df['correct_inferred'] == True].copy()
    print(f"Correct predictions: {len(correct_df)}")
    
    # Filter for low confidence (< 75%)
    challenging_df = correct_df[correct_df['confidence'] < 0.8].copy()
    print(f"Correct but low confidence (< 75%): {len(challenging_df)}")
    
    if len(challenging_df) == 0:
        print("\n✓ No challenging samples found. All correct predictions have high confidence!")
        return
    
    # Show distribution
    print("\n📊 Challenging samples by class:")
    class_counts = challenging_df['predicted'].value_counts()
    for class_name, count in class_counts.items():
        print(f"  {class_name:12s}: {count:3d} samples")
    
    # Show confidence distribution
    print(f"\n📊 Confidence statistics:")
    print(f"  Mean: {challenging_df['confidence'].mean():.4f}")
    print(f"  Min:  {challenging_df['confidence'].min():.4f}")
    print(f"  Max:  {challenging_df['confidence'].max():.4f}")
    
    # Copy samples to training dataset
    print(f"\n📦 Copying challenging samples to training dataset...")
    
    stats = {
        'added': 0,
        'already_exists': 0,
        'not_found': 0,
        'by_class': {'background': 0, 'drone': 0, 'helicopter': 0}
    }
    
    for idx, row in tqdm(list(challenging_df.iterrows()), desc="Processing samples"):
        challenge_id = row['challenge_id']
        predicted_class = row['predicted']
        confidence = row['confidence']
        
        # Look for audio file with pattern: {challenge_id}_{class}_{timestamp}.wav
        # There may be multiple attempts, we'll take the first match
        audio_files = list(CHALLENGE_AUDIO_DIR.glob(f"{challenge_id}_{predicted_class}_*.wav"))
        
        if not audio_files:
            # Try without class name
            audio_files = list(CHALLENGE_AUDIO_DIR.glob(f"{challenge_id}_*.wav"))
        
        if not audio_files:
            stats['not_found'] += 1
            tqdm.write(f"  ⚠️  Audio not found for {challenge_id} ({predicted_class}, conf={confidence:.4f})")
            continue
        
        src_file = audio_files[0]  # Use first match
        
        # Destination path
        dst_dir = COMBINED_TRAIN / predicted_class
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        # Use a descriptive filename
        dst_file = dst_dir / f"challenge_{challenge_id}_conf{int(confidence*100)}.wav"
        
        if dst_file.exists():
            stats['already_exists'] += 1
            tqdm.write(f"  ⊙  Already exists: {dst_file.name}")
        else:
            shutil.copy2(src_file, dst_file)
            stats['added'] += 1
            stats['by_class'][predicted_class] += 1
            tqdm.write(f"  ✓  Added: {predicted_class}/{dst_file.name} (conf={confidence:.4f})")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n✓ Samples added:        {stats['added']}")
    print(f"⊙ Already existed:      {stats['already_exists']}")
    print(f"⚠️  Audio not found:     {stats['not_found']}")
    
    if stats['added'] > 0:
        print(f"\n📊 Added by class:")
        for class_name, count in sorted(stats['by_class'].items()):
            if count > 0:
                print(f"  {class_name:12s}: {count:3d} samples")
    
    print(f"\n✓ Training dataset updated: {COMBINED_TRAIN}")
    
    # Show top challenging samples
    if len(challenging_df) > 0:
        print(f"\n🎯 Most challenging samples (lowest confidence):")
        top_challenging = challenging_df.nsmallest(10, 'confidence')[['challenge_id', 'predicted', 'confidence']]
        for idx, row in top_challenging.iterrows():
            print(f"  {row['predicted']:12s} - confidence: {row['confidence']:.4f} - ID: {row['challenge_id']}")
    
    print("\n" + "="*80)
    print("✓ Process complete!")
    print("="*80)
    
    if stats['added'] > 0:
        print("\n📝 Next step - Retrain model with updated dataset:")
        print("\npython train_sota_model.py \\")
        print("  --train-dir data/combined_dataset/train \\")
        print("  --val-dir data/combined_dataset/val \\")
        print("  --model-type crnn \\")
        print("  --use-hpss \\")
        print("  --epochs 50 \\")
        print("  --batch-size 32 \\")
        print("  --use-focal-loss \\")
        print("  --focal-gamma 2.0 \\")
        print("  --use-class-weights \\")
        print("  --output-dir models/crnn_hardmining")

if __name__ == '__main__':
    add_challenging_samples()
