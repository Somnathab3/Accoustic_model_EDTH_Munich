"""
Compare Old vs New SOTA Model Predictions
Specifically checks if the new model fails on the same samples where the old model failed
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import torch.nn.functional as F
import pandas as pd
import json
from tqdm import tqdm
import argparse

from adrone.preprocessing import AudioPreprocessor
from adrone.models.acoustic_models import create_model


def load_model(model_path: str, device: torch.device):
    """Load a trained model"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model_type = checkpoint.get('model_type', 'panns')
    input_channels = checkpoint.get('input_channels', 3)
    num_classes = checkpoint.get('num_classes', 3)
    n_mels = checkpoint.get('n_mels', 96)
    
    print(f"Loading {model_type.upper()} model...")
    print(f"  Channels: {input_channels}, Classes: {num_classes}")
    
    model = create_model(
        model_type=model_type,
        num_classes=num_classes,
        input_channels=input_channels,
        n_mels=n_mels if model_type == 'crnn' else None
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,} ({params/1e6:.2f}M)")
    
    return model, checkpoint


def predict_audio(model, preprocessor, audio_path: str, device: torch.device, idx_to_class: dict):
    """Make prediction on audio file"""
    try:
        waveform = preprocessor.load_audio(audio_path)
        spectrogram = preprocessor(waveform)
        spectrogram = spectrogram.unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(spectrogram)
            probabilities = F.softmax(logits, dim=1)
            predicted_idx = probabilities.argmax(dim=1).item()
            confidence = probabilities[0, predicted_idx].item()
        
        predicted_label = idx_to_class[predicted_idx]
        return predicted_label, confidence, probabilities.cpu().numpy()[0]
    
    except Exception as e:
        print(f"Error predicting {audio_path}: {e}")
        return None, 0.0, None


def main():
    parser = argparse.ArgumentParser(description='Compare old vs new model predictions')
    parser.add_argument('--old-model', type=str, default='models/panns_combined/panns_final.pt',
                       help='Path to old model checkpoint')
    parser.add_argument('--new-model', type=str, default='models/best_model.pt',
                       help='Path to new SOTA model checkpoint')
    parser.add_argument('--audio-dir', type=str, default='challenge_results/audio_samples',
                       help='Directory containing audio samples')
    parser.add_argument('--failed-csv', type=str, default='challenge_results/failed_samples.csv',
                       help='CSV file with failed samples from old model')
    parser.add_argument('--labels', type=str, default='models/panns_combined/labels.json',
                       help='Labels JSON file')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"OLD vs NEW MODEL COMPARISON")
    print(f"{'='*80}")
    print(f"Device: {device}")
    
    # Load labels
    with open(args.labels, 'r') as f:
        label_data = json.load(f)
        if 'class_to_idx' in label_data:
            idx_to_class = {int(k): v for k, v in label_data['idx_to_class'].items()}
        else:
            labels_list = label_data['labels']
            idx_to_class = {i: label for i, label in enumerate(labels_list)}
    
    print(f"\nLabels: {idx_to_class}")
    
    # Load old model
    print(f"\n{'='*80}")
    print("LOADING OLD MODEL")
    print(f"{'='*80}")
    old_model, old_checkpoint = load_model(args.old_model, device)
    
    # Load new model
    print(f"\n{'='*80}")
    print("LOADING NEW SOTA MODEL")
    print(f"{'='*80}")
    new_model, new_checkpoint = load_model(args.new_model, device)
    
    # Setup preprocessor
    preprocessor = AudioPreprocessor(
        sample_rate=16000,
        n_mels=96,
        use_hpss=True
    )
    
    # Load failed samples
    failed_df = pd.read_csv(args.failed_csv)
    print(f"\n{'='*80}")
    print(f"TESTING ON {len(failed_df)} FAILED SAMPLES")
    print(f"{'='*80}")
    
    audio_dir = Path(args.audio_dir)
    
    # Results tracking
    results = []
    comparison_stats = {
        'both_correct': 0,
        'both_wrong': 0,
        'old_correct_new_wrong': 0,
        'old_wrong_new_correct': 0,
        'same_prediction': 0,
        'different_prediction': 0
    }
    
    for idx, row in tqdm(failed_df.iterrows(), total=len(failed_df), desc="Testing"):
        challenge_id = row['challenge_id']
        old_prediction = row['predicted']
        old_confidence = row['confidence']
        
        # Find audio file
        audio_files = list(audio_dir.glob(f"{challenge_id}*.wav"))
        if not audio_files:
            continue
        
        audio_path = str(audio_files[0])
        
        # Get new model prediction
        new_prediction, new_confidence, new_probs = predict_audio(
            new_model, preprocessor, audio_path, device, idx_to_class
        )
        
        if new_prediction is None:
            continue
        
        # Compare predictions
        same_pred = (old_prediction == new_prediction)
        if same_pred:
            comparison_stats['same_prediction'] += 1
        else:
            comparison_stats['different_prediction'] += 1
        
        # We don't know the ground truth, but we can analyze patterns
        results.append({
            'challenge_id': challenge_id,
            'audio_file': Path(audio_path).name,
            'old_prediction': old_prediction,
            'old_confidence': old_confidence,
            'new_prediction': new_prediction,
            'new_confidence': new_confidence,
            'same_prediction': same_pred,
            'confidence_change': new_confidence - old_confidence
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Analysis
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"\nTotal samples analyzed: {len(results_df)}")
    print(f"\nSame predictions: {comparison_stats['same_prediction']} ({comparison_stats['same_prediction']/len(results_df)*100:.1f}%)")
    print(f"Different predictions: {comparison_stats['different_prediction']} ({comparison_stats['different_prediction']/len(results_df)*100:.1f}%)")
    
    print(f"\n{'='*80}")
    print("NEW MODEL PREDICTION BREAKDOWN")
    print(f"{'='*80}")
    print(results_df['new_prediction'].value_counts())
    
    print(f"\n{'='*80}")
    print("CONFIDENCE COMPARISON")
    print(f"{'='*80}")
    print(f"\nOld Model Confidence:")
    print(f"  Mean: {results_df['old_confidence'].mean():.4f}")
    print(f"  Std:  {results_df['old_confidence'].std():.4f}")
    print(f"\nNew Model Confidence:")
    print(f"  Mean: {results_df['new_confidence'].mean():.4f}")
    print(f"  Std:  {results_df['new_confidence'].std():.4f}")
    print(f"\nConfidence Change (New - Old):")
    print(f"  Mean: {results_df['confidence_change'].mean():.4f}")
    print(f"  Median: {results_df['confidence_change'].median():.4f}")
    
    # Samples where predictions changed
    changed = results_df[~results_df['same_prediction']]
    print(f"\n{'='*80}")
    print(f"SAMPLES WHERE PREDICTIONS CHANGED ({len(changed)})")
    print(f"{'='*80}")
    if len(changed) > 0:
        print("\nTop 10 examples (sorted by confidence change):")
        print(changed.nlargest(10, 'confidence_change')[['audio_file', 'old_prediction', 'old_confidence', 
                                                          'new_prediction', 'new_confidence', 'confidence_change']])
        
        print("\n\nPrediction transitions:")
        for old_pred in idx_to_class.values():
            for new_pred in idx_to_class.values():
                if old_pred != new_pred:
                    count = len(changed[(changed['old_prediction'] == old_pred) & 
                                       (changed['new_prediction'] == new_pred)])
                    if count > 0:
                        print(f"  {old_pred} → {new_pred}: {count}")
    
    # Samples where predictions stayed the same
    same = results_df[results_df['same_prediction']]
    print(f"\n{'='*80}")
    print(f"SAMPLES WHERE PREDICTIONS STAYED THE SAME ({len(same)})")
    print(f"{'='*80}")
    if len(same) > 0:
        print("\nBreakdown by prediction:")
        print(same['new_prediction'].value_counts())
        
        print("\n\nConfidence changes for same predictions:")
        for pred in idx_to_class.values():
            pred_same = same[same['new_prediction'] == pred]
            if len(pred_same) > 0:
                avg_change = pred_same['confidence_change'].mean()
                print(f"  {pred}: Δconf = {avg_change:+.4f} (avg)")
    
    # Save results
    output_file = 'challenge_results/model_comparison.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n{'='*80}")
    print(f"✓ Saved detailed results to: {output_file}")
    
    # Save summary statistics
    summary = {
        'total_samples': len(results_df),
        'same_predictions': int(comparison_stats['same_prediction']),
        'different_predictions': int(comparison_stats['different_prediction']),
        'same_percentage': float(comparison_stats['same_prediction'] / len(results_df) * 100),
        'old_model_confidence': {
            'mean': float(results_df['old_confidence'].mean()),
            'std': float(results_df['old_confidence'].std())
        },
        'new_model_confidence': {
            'mean': float(results_df['new_confidence'].mean()),
            'std': float(results_df['new_confidence'].std())
        },
        'confidence_change': {
            'mean': float(results_df['confidence_change'].mean()),
            'median': float(results_df['confidence_change'].median())
        },
        'new_model_predictions': results_df['new_prediction'].value_counts().to_dict()
    }
    
    summary_file = 'challenge_results/model_comparison_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary statistics to: {summary_file}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
