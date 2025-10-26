"""
Validate Enhanced Model Against Baseline's Correct Predictions

This script cross-checks the enhanced LIGO-modified model against the last 200
correct predictions from the baseline CRNN model to ensure backward compatibility.

Goal: Verify that the enhanced model maintains or improves accuracy on samples
      that the baseline model predicted correctly.
"""

import os
import sys
import pandas as pd
import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# Import our inference modules
from enhanced_inference import EnhancedAcousticClassifier


def load_results_csv(csv_path):
    """Load and parse the results CSV file."""
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} results from {csv_path}")
    print(f"[INFO] Columns: {df.columns.tolist()}")
    return df


def get_last_n_correct(df, n=200):
    """Extract the last N correct predictions from results."""
    # Filter for correct predictions (where success=True and correct_inferred=True or api_message='Correct!')
    correct_mask = (df['api_message'].str.contains('Correct', case=False, na=False)) | \
                   (df['correct_inferred'] == True)
    
    correct_df = df[correct_mask].copy()
    print(f"[INFO] Found {len(correct_df)} correct predictions in total")
    
    # Get last N
    last_n = correct_df.tail(n)
    print(f"[INFO] Selected last {len(last_n)} correct predictions for validation")
    
    return last_n


def get_last_n_samples(df, n=200):
    """Extract the last N samples (both correct and incorrect) from results."""
    last_n = df.tail(n).copy()
    
    # Separate correct and incorrect
    correct_mask = (last_n['api_message'].str.contains('Correct', case=False, na=False)) | \
                   (last_n['correct_inferred'] == True)
    
    correct_df = last_n[correct_mask]
    incorrect_df = last_n[~correct_mask]
    
    print(f"[INFO] Selected last {len(last_n)} samples for validation")
    print(f"  - Correct: {len(correct_df)} ({len(correct_df)/len(last_n)*100:.1f}%)")
    print(f"  - Incorrect: {len(incorrect_df)} ({len(incorrect_df)/len(last_n)*100:.1f}%)")
    
    return last_n, correct_df, incorrect_df


def load_audio_for_challenge(challenge_id, audio_dir):
    """Load audio file for a given challenge ID."""
    audio_dir = Path(audio_dir)
    
    # Try direct match first
    audio_path = audio_dir / f"{challenge_id}.wav"
    if audio_path.exists():
        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))
            return waveform, sample_rate
        except Exception as e:
            print(f"[ERROR] Failed to load {audio_path}: {e}")
            return None, None
    
    # Try pattern match (challenge_id_class_timestamp.wav)
    matching_files = list(audio_dir.glob(f"{challenge_id}_*.wav"))
    if matching_files:
        audio_path = matching_files[0]  # Take first match
        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))
            return waveform, sample_rate
        except Exception as e:
            print(f"[ERROR] Failed to load {audio_path}: {e}")
            return None, None
    
    return None, None


def validate_predictions(baseline_df, audio_dir, baseline_model_path, enhanced_model_path, labels_path, mode='correct'):
    """
    Validate enhanced model predictions against baseline predictions.
    
    Args:
        baseline_df: DataFrame containing baseline's predictions
        audio_dir: Directory containing audio samples
        baseline_model_path: Path to baseline CRNN model
        enhanced_model_path: Path to enhanced model with matched filter bank
        labels_path: Path to labels JSON file
        mode: 'correct' for correct predictions only, 'incorrect' for wrong ones, 'all' for both
    
    Returns:
        Dictionary with validation results
    """
    print("\n" + "="*80)
    if mode == 'correct':
        print("VALIDATION: Enhanced Model vs Baseline CORRECT Predictions")
    elif mode == 'incorrect':
        print("VALIDATION: Enhanced Model vs Baseline INCORRECT Predictions")
    else:
        print("VALIDATION: Enhanced Model vs Baseline ALL Predictions")
    print("="*80)
    
    # Load labels
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)
        label_to_idx = labels_data.get('label_to_idx', {})
        idx_to_label = {v: k for k, v in label_to_idx.items()}
    
    print(f"\n[LABELS] {label_to_idx}")
    
    # Initialize models
    print(f"\n[BASELINE] Loading baseline model: {baseline_model_path}")
    baseline_classifier = EnhancedAcousticClassifier(
        model_path=baseline_model_path,
        labels_path=labels_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"[ENHANCED] Loading enhanced model: {enhanced_model_path}")
    enhanced_classifier = EnhancedAcousticClassifier(
        model_path=enhanced_model_path,
        labels_path=labels_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Validation tracking
    results = {
        'mode': mode,
        'total_samples': 0,
        'audio_found': 0,
        'baseline_still_correct': 0,
        'enhanced_correct': 0,
        'both_correct': 0,
        'enhanced_wrong': 0,
        'baseline_changed': 0,
        'enhanced_improved': 0,  # Enhanced correct when baseline was wrong
        'enhanced_regressed': 0,  # Enhanced wrong when baseline was correct
        'both_wrong': 0,
        'detailed_results': [],
        'by_class': {}
    }
    
    # Process each sample
    print(f"\n[PROCESSING] Validating {len(baseline_df)} samples...")
    
    for idx, row in tqdm(baseline_df.iterrows(), total=len(baseline_df), desc="Validating"):
        challenge_id = row['challenge_id']
        baseline_pred = row['predicted']
        baseline_was_correct = ('Correct' in str(row.get('api_message', '')) or row.get('correct_inferred', False))
        
        # Initialize class tracking
        if baseline_pred not in results['by_class']:
            results['by_class'][baseline_pred] = {
                'total': 0,
                'audio_found': 0,
                'baseline_still_correct': 0,
                'enhanced_correct': 0,
                'both_correct': 0,
                'enhanced_wrong': 0,
                'enhanced_improved': 0,
                'both_wrong': 0
            }
        
        results['by_class'][baseline_pred]['total'] += 1
        results['total_samples'] += 1
        
        # Load audio
        waveform, sample_rate = load_audio_for_challenge(challenge_id, audio_dir)
        
        if waveform is None:
            continue
        
        results['audio_found'] += 1
        results['by_class'][baseline_pred]['audio_found'] += 1
        
        try:
            # Get baseline prediction (re-inference to verify)
            baseline_current_pred, baseline_current_conf, _ = baseline_classifier.classify_from_tensor(waveform)
            
            # Get enhanced prediction
            enhanced_pred, enhanced_conf, _ = enhanced_classifier.classify_from_tensor(waveform)
            
            # Check if baseline still predicts the same
            baseline_matches_original = (baseline_current_pred == baseline_pred)
            
            # Check if enhanced matches baseline
            enhanced_matches_baseline = (enhanced_pred == baseline_pred)
            
            # Update results
            if baseline_matches_original:
                results['baseline_still_correct'] += 1
                results['by_class'][baseline_pred]['baseline_still_correct'] += 1
            else:
                results['baseline_changed'] += 1
            
            if enhanced_matches_baseline:
                results['enhanced_correct'] += 1
                results['by_class'][baseline_pred]['enhanced_correct'] += 1
            else:
                results['enhanced_wrong'] += 1
                results['by_class'][baseline_pred]['enhanced_wrong'] += 1
            
            if baseline_matches_original and enhanced_matches_baseline:
                results['both_correct'] += 1
                results['by_class'][baseline_pred]['both_correct'] += 1
            
            # Track improvements and regressions
            if mode == 'incorrect' or mode == 'all':
                # For wrong predictions, check if enhanced model improved
                if not baseline_was_correct and enhanced_matches_baseline:
                    # This doesn't make sense - if baseline was wrong, enhanced matching it is also wrong
                    pass
                elif not baseline_was_correct and not enhanced_matches_baseline:
                    # Both were wrong, but enhanced tried something different
                    results['both_wrong'] += 1
                    results['by_class'][baseline_pred]['both_wrong'] += 1
            
            if mode == 'all':
                # Check if enhanced improved or regressed
                if baseline_was_correct and enhanced_matches_baseline:
                    results['enhanced_correct'] += 1
                    results['by_class'][baseline_pred]['enhanced_correct'] += 1
                elif baseline_was_correct and not enhanced_matches_baseline:
                    results['enhanced_regressed'] += 1
                elif not baseline_was_correct and enhanced_matches_baseline:
                    # Enhanced predicted same as wrong baseline
                    results['both_wrong'] += 1
                    results['by_class'][baseline_pred]['both_wrong'] += 1
            
            # Store detailed result
            results['detailed_results'].append({
                'challenge_id': challenge_id,
                'baseline_original': baseline_pred,
                'baseline_was_correct': baseline_was_correct,
                'baseline_reinference': baseline_current_pred,
                'baseline_conf': float(baseline_current_conf),
                'enhanced_pred': enhanced_pred,
                'enhanced_conf': float(enhanced_conf),
                'baseline_matches': baseline_matches_original,
                'enhanced_matches': enhanced_matches_baseline,
                'agreement': baseline_current_pred == enhanced_pred
            })
            
        except Exception as e:
            print(f"\n[ERROR] Failed to process {challenge_id}: {e}")
            continue
    
    return results


def print_validation_summary(results):
    """Print comprehensive validation summary."""
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    mode = results.get('mode', 'correct')
    total = results['total_samples']
    found = results['audio_found']
    
    print(f"\n[DATASET]")
    if mode == 'correct':
        print(f"  Total baseline correct samples: {total}")
    elif mode == 'incorrect':
        print(f"  Total baseline incorrect samples: {total}")
    else:
        print(f"  Total samples (all): {total}")
    print(f"  Audio files found: {found} ({found/total*100:.1f}%)")
    
    if found == 0:
        print("\n[ERROR] No audio files found! Cannot validate.")
        return
    
    baseline_still = results['baseline_still_correct']
    enhanced_correct = results['enhanced_correct']
    both_correct = results['both_correct']
    enhanced_wrong = results['enhanced_wrong']
    baseline_changed = results['baseline_changed']
    enhanced_improved = results.get('enhanced_improved', 0)
    enhanced_regressed = results.get('enhanced_regressed', 0)
    both_wrong = results.get('both_wrong', 0)
    
    print(f"\n[BASELINE RE-INFERENCE]")
    print(f"  Still predicts same: {baseline_still}/{found} ({baseline_still/found*100:.1f}%)")
    print(f"  Changed prediction: {baseline_changed}/{found} ({baseline_changed/found*100:.1f}%)")
    
    print(f"\n[ENHANCED MODEL VALIDATION]")
    print(f"  Matches baseline original: {enhanced_correct}/{found} ({enhanced_correct/found*100:.1f}%)")
    print(f"  Differs from baseline: {enhanced_wrong}/{found} ({enhanced_wrong/found*100:.1f}%)")
    
    print(f"\n[AGREEMENT]")
    print(f"  Both correct (vs original): {both_correct}/{found} ({both_correct/found*100:.1f}%)")
    
    if mode == 'incorrect' or mode == 'all':
        print(f"\n[IMPROVEMENTS & REGRESSIONS]")
        if enhanced_improved > 0:
            print(f"  Enhanced improved: {enhanced_improved}/{found} ({enhanced_improved/found*100:.1f}%)")
            print(f"    (Enhanced correct where baseline was wrong)")
        if enhanced_regressed > 0:
            print(f"  Enhanced regressed: {enhanced_regressed}/{found} ({enhanced_regressed/found*100:.1f}%)")
            print(f"    (Enhanced wrong where baseline was correct)")
        if both_wrong > 0:
            print(f"  Both wrong: {both_wrong}/{found} ({both_wrong/found*100:.1f}%)")
            print(f"    (Both models predicted incorrectly)")
    
    # Per-class breakdown
    print(f"\n[PER-CLASS BREAKDOWN]")
    for class_name, stats in results['by_class'].items():
        if stats['audio_found'] == 0:
            continue
        print(f"\n  {class_name.upper()}:")
        print(f"    Total samples: {stats['total']}")
        print(f"    Audio found: {stats['audio_found']}")
        print(f"    Baseline still correct: {stats['baseline_still_correct']}/{stats['audio_found']} " +
              f"({stats['baseline_still_correct']/stats['audio_found']*100:.1f}%)")
        print(f"    Enhanced matches: {stats['enhanced_correct']}/{stats['audio_found']} " +
              f"({stats['enhanced_correct']/stats['audio_found']*100:.1f}%)")
        print(f"    Both correct: {stats['both_correct']}/{stats['audio_found']} " +
              f"({stats['both_correct']/stats['audio_found']*100:.1f}%)")
        if mode == 'incorrect' or mode == 'all':
            if stats.get('enhanced_improved', 0) > 0:
                print(f"    Enhanced improved: {stats['enhanced_improved']}/{stats['audio_found']} " +
                      f"({stats['enhanced_improved']/stats['audio_found']*100:.1f}%)")
            if stats.get('both_wrong', 0) > 0:
                print(f"    Both wrong: {stats['both_wrong']}/{stats['audio_found']} " +
                      f"({stats['both_wrong']/stats['audio_found']*100:.1f}%)")
    
    # Calculate key metrics
    if found > 0:
        compatibility_rate = (enhanced_correct / found) * 100
        improvement_rate = ((enhanced_correct - baseline_still) / found) * 100 if baseline_still < enhanced_correct else 0
        
        print(f"\n[KEY METRICS]")
        print(f"  Backward Compatibility: {compatibility_rate:.2f}%")
        print(f"    (Enhanced model matches baseline's original predictions)")
        
        if improvement_rate > 0:
            print(f"  Improvement on changed cases: +{improvement_rate:.2f}%")
            print(f"    (Enhanced correct when baseline re-inference changed)")
        
        # Verdict
        print(f"\n[VERDICT]")
        if compatibility_rate >= 95.0:
            print(f"  [OK] EXCELLENT - Enhanced model maintains 95%+ compatibility")
        elif compatibility_rate >= 90.0:
            print(f"  [OK] GOOD - Enhanced model maintains 90%+ compatibility")
        elif compatibility_rate >= 85.0:
            print(f"  [WARN] ACCEPTABLE - Enhanced model at 85%+ compatibility")
        else:
            print(f"  [ERROR] POOR - Enhanced model <85% compatibility - Review required!")


def save_detailed_results(results, output_dir):
    """Save detailed validation results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed CSV
    detailed_df = pd.DataFrame(results['detailed_results'])
    csv_path = output_dir / f"validation_detailed_{timestamp}.csv"
    detailed_df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] Detailed results: {csv_path}")
    
    # Save summary JSON
    summary = {
        'timestamp': timestamp,
        'total_samples': results['total_samples'],
        'audio_found': results['audio_found'],
        'baseline_still_correct': results['baseline_still_correct'],
        'enhanced_correct': results['enhanced_correct'],
        'both_correct': results['both_correct'],
        'enhanced_wrong': results['enhanced_wrong'],
        'baseline_changed': results['baseline_changed'],
        'by_class': results['by_class'],
        'compatibility_rate': (results['enhanced_correct'] / results['audio_found'] * 100) if results['audio_found'] > 0 else 0
    }
    
    json_path = output_dir / f"validation_summary_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] Summary JSON: {json_path}")
    
    # Save disagreements (where enhanced differs from baseline)
    disagreements = [r for r in results['detailed_results'] if not r['enhanced_matches']]
    if disagreements:
        disagree_df = pd.DataFrame(disagreements)
        disagree_path = output_dir / f"validation_disagreements_{timestamp}.csv"
        disagree_df.to_csv(disagree_path, index=False)
        print(f"[SAVED] Disagreements ({len(disagreements)} samples): {disagree_path}")
    
    # Save improvements (for incorrect mode - where enhanced got it right but baseline didn't)
    mode = results.get('mode', 'correct')
    if mode == 'incorrect' or mode == 'all':
        # In incorrect mode, "enhanced_matches=False" means enhanced tried different prediction
        improvements = [r for r in results['detailed_results'] 
                       if not r['baseline_was_correct'] and not r['enhanced_matches']]
        if improvements:
            improve_df = pd.DataFrame(improvements)
            improve_path = output_dir / f"validation_enhanced_different_{timestamp}.csv"
            improve_df.to_csv(improve_path, index=False)
            print(f"[SAVED] Enhanced different predictions ({len(improvements)} samples): {improve_path}")


def main():
    """Main validation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Enhanced Model vs Baseline Correct Predictions')
    parser.add_argument('--results-csv', type=str, 
                        default='F:/EDTH/acoustic-drone-detector/challenge_results/results.csv',
                        help='Path to baseline results CSV')
    parser.add_argument('--audio-dir', type=str,
                        default='F:/EDTH/acoustic-drone-detector/challenge_results/audio_samples',
                        help='Directory containing audio samples')
    parser.add_argument('--baseline-model', type=str,
                        default='models/crnn_combined/best_model.pt',
                        help='Path to baseline CRNN model')
    parser.add_argument('--enhanced-model', type=str,
                        default='models/matched_bank_comparison/enhanced_crnn.pt',
                        help='Path to enhanced model with matched filter bank')
    parser.add_argument('--labels', type=str,
                        default='models/crnn_combined/labels.json',
                        help='Path to labels JSON')
    parser.add_argument('--n-samples', type=int, default=200,
                        help='Number of last samples to validate')
    parser.add_argument('--mode', type=str, default='all', choices=['correct', 'incorrect', 'all'],
                        help='Validation mode: correct (only correct predictions), incorrect (only wrong), all (both)')
    parser.add_argument('--output-dir', type=str,
                        default='validation_results',
                        help='Output directory for validation results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ENHANCED MODEL BACKWARD COMPATIBILITY VALIDATION")
    print("="*80)
    print(f"\n[CONFIG]")
    print(f"  Results CSV: {args.results_csv}")
    print(f"  Audio Directory: {args.audio_dir}")
    print(f"  Baseline Model: {args.baseline_model}")
    print(f"  Enhanced Model: {args.enhanced_model}")
    print(f"  Labels: {args.labels}")
    print(f"  Samples to validate: {args.n_samples}")
    print(f"  Mode: {args.mode}")
    print(f"  Output Directory: {args.output_dir}")
    
    # Load results
    df = load_results_csv(args.results_csv)
    
    # Get samples based on mode
    if args.mode == 'correct':
        # Get last N correct predictions only
        validation_df = get_last_n_correct(df, n=args.n_samples)
        if len(validation_df) == 0:
            print("\n[ERROR] No correct predictions found in results CSV!")
            return
    elif args.mode == 'incorrect':
        # Get last N samples and filter for incorrect only
        all_samples, correct_samples, incorrect_samples = get_last_n_samples(df, n=args.n_samples)
        validation_df = incorrect_samples
        if len(validation_df) == 0:
            print("\n[ERROR] No incorrect predictions found in last N samples!")
            return
    else:  # all
        # Get all last N samples (correct + incorrect)
        all_samples, correct_samples, incorrect_samples = get_last_n_samples(df, n=args.n_samples)
        validation_df = all_samples
        if len(validation_df) == 0:
            print("\n[ERROR] No samples found in results CSV!")
            return
    
    # Validate predictions
    results = validate_predictions(
        baseline_df=validation_df,
        audio_dir=args.audio_dir,
        baseline_model_path=args.baseline_model,
        enhanced_model_path=args.enhanced_model,
        labels_path=args.labels,
        mode=args.mode
    )
    
    # Print summary
    print_validation_summary(results)
    
    # Save detailed results
    save_detailed_results(results, args.output_dir)
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
