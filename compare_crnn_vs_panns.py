"""
Compare CRNN vs PANNs Model Predictions
Analyzes if the new CRNN model performs better on previously failed samples
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
    print(f"  Channels: {input_channels}, Classes: {num_classes}, N_mels: {n_mels}")
    
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
    parser = argparse.ArgumentParser(description='Compare CRNN vs PANNs model predictions')
    parser.add_argument('--panns-model', type=str, default='models/panns_combined/panns_final.pt',
                       help='Path to PANNs model checkpoint')
    parser.add_argument('--crnn-model', type=str, default='models/crnn_combined/crnn_final.pt',
                       help='Path to CRNN model checkpoint')
    parser.add_argument('--audio-dir', type=str, default='challenge_results/audio_samples',
                       help='Directory containing audio samples')
    parser.add_argument('--failed-csv', type=str, default='challenge_results/failed_samples.csv',
                       help='CSV file with failed samples from PANNs model')
    parser.add_argument('--labels', type=str, default='models/crnn_combined/labels.json',
                       help='Labels JSON file')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"CRNN vs PANNs MODEL COMPARISON")
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
    
    # Load PANNs model
    print(f"\n{'='*80}")
    print("LOADING PANNs MODEL (BASELINE)")
    print(f"{'='*80}")
    panns_model, panns_checkpoint = load_model(args.panns_model, device)
    
    # Load CRNN model
    print(f"\n{'='*80}")
    print("LOADING CRNN MODEL (NEW)")
    print(f"{'='*80}")
    crnn_model, crnn_checkpoint = load_model(args.crnn_model, device)
    
    # Setup preprocessor
    preprocessor = AudioPreprocessor(
        sample_rate=16000,
        n_mels=96,
        use_hpss=True
    )
    
    # Load failed samples from PANNs
    failed_df = pd.read_csv(args.failed_csv)
    print(f"\n{'='*80}")
    print(f"TESTING ON {len(failed_df)} FAILED SAMPLES FROM PANNs")
    print(f"{'='*80}")
    
    audio_dir = Path(args.audio_dir)
    
    # Results tracking
    results = []
    comparison_stats = {
        'same_prediction': 0,
        'different_prediction': 0,
        'crnn_higher_confidence': 0,
        'panns_higher_confidence': 0
    }
    
    for idx, row in tqdm(failed_df.iterrows(), total=len(failed_df), desc="Testing"):
        challenge_id = row['challenge_id']
        panns_prediction = row['predicted']
        panns_confidence = row['confidence']
        
        # Find audio file
        audio_files = list(audio_dir.glob(f"{challenge_id}*.wav"))
        if not audio_files:
            continue
        
        audio_path = str(audio_files[0])
        
        # Get CRNN model prediction
        crnn_prediction, crnn_confidence, crnn_probs = predict_audio(
            crnn_model, preprocessor, audio_path, device, idx_to_class
        )
        
        if crnn_prediction is None:
            continue
        
        # Compare predictions
        same_pred = (panns_prediction == crnn_prediction)
        if same_pred:
            comparison_stats['same_prediction'] += 1
        else:
            comparison_stats['different_prediction'] += 1
        
        # Compare confidence
        if crnn_confidence > panns_confidence:
            comparison_stats['crnn_higher_confidence'] += 1
        else:
            comparison_stats['panns_higher_confidence'] += 1
        
        results.append({
            'challenge_id': challenge_id,
            'audio_file': Path(audio_path).name,
            'panns_prediction': panns_prediction,
            'panns_confidence': panns_confidence,
            'crnn_prediction': crnn_prediction,
            'crnn_confidence': crnn_confidence,
            'same_prediction': same_pred,
            'confidence_change': crnn_confidence - panns_confidence
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
    print(f"\nCRNN higher confidence: {comparison_stats['crnn_higher_confidence']} ({comparison_stats['crnn_higher_confidence']/len(results_df)*100:.1f}%)")
    print(f"PANNs higher confidence: {comparison_stats['panns_higher_confidence']} ({comparison_stats['panns_higher_confidence']/len(results_df)*100:.1f}%)")
    
    print(f"\n{'='*80}")
    print("CRNN MODEL PREDICTION BREAKDOWN")
    print(f"{'='*80}")
    print(results_df['crnn_prediction'].value_counts())
    
    print(f"\n{'='*80}")
    print("CONFIDENCE COMPARISON")
    print(f"{'='*80}")
    print(f"\nPANNs Model Confidence:")
    print(f"  Mean: {results_df['panns_confidence'].mean():.4f}")
    print(f"  Std:  {results_df['panns_confidence'].std():.4f}")
    print(f"\nCRNN Model Confidence:")
    print(f"  Mean: {results_df['crnn_confidence'].mean():.4f}")
    print(f"  Std:  {results_df['crnn_confidence'].std():.4f}")
    print(f"\nConfidence Change (CRNN - PANNs):")
    print(f"  Mean: {results_df['confidence_change'].mean():.4f}")
    print(f"  Median: {results_df['confidence_change'].median():.4f}")
    
    # Samples where predictions changed
    changed = results_df[~results_df['same_prediction']]
    print(f"\n{'='*80}")
    print(f"SAMPLES WHERE PREDICTIONS CHANGED ({len(changed)})")
    print(f"{'='*80}")
    if len(changed) > 0:
        print("\nTop 10 examples (sorted by confidence change):")
        print(changed.nlargest(10, 'confidence_change')[['audio_file', 'panns_prediction', 'panns_confidence', 
                                                          'crnn_prediction', 'crnn_confidence', 'confidence_change']])
        
        print("\n\nPrediction transitions:")
        for panns_pred in idx_to_class.values():
            for crnn_pred in idx_to_class.values():
                if panns_pred != crnn_pred:
                    count = len(changed[(changed['panns_prediction'] == panns_pred) & 
                                       (changed['crnn_prediction'] == crnn_pred)])
                    if count > 0:
                        print(f"  {panns_pred} → {crnn_pred}: {count}")
    
    # Samples where predictions stayed the same
    same = results_df[results_df['same_prediction']]
    print(f"\n{'='*80}")
    print(f"SAMPLES WHERE PREDICTIONS STAYED THE SAME ({len(same)})")
    print(f"{'='*80}")
    if len(same) > 0:
        print("\nBreakdown by prediction:")
        print(same['crnn_prediction'].value_counts())
        
        print("\n\nConfidence changes for same predictions:")
        for pred in idx_to_class.values():
            pred_same = same[same['crnn_prediction'] == pred]
            if len(pred_same) > 0:
                avg_change = pred_same['confidence_change'].mean()
                print(f"  {pred}: Δconf = {avg_change:+.4f} (avg)")
    
    # Save results
    output_file = 'challenge_results/crnn_vs_panns_comparison.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n{'='*80}")
    print(f"✓ Saved detailed results to: {output_file}")
    
    # Save summary statistics
    summary = {
        'total_samples': len(results_df),
        'same_predictions': int(comparison_stats['same_prediction']),
        'different_predictions': int(comparison_stats['different_prediction']),
        'same_percentage': float(comparison_stats['same_prediction'] / len(results_df) * 100),
        'crnn_higher_confidence': int(comparison_stats['crnn_higher_confidence']),
        'panns_higher_confidence': int(comparison_stats['panns_higher_confidence']),
        'panns_model_confidence': {
            'mean': float(results_df['panns_confidence'].mean()),
            'std': float(results_df['panns_confidence'].std())
        },
        'crnn_model_confidence': {
            'mean': float(results_df['crnn_confidence'].mean()),
            'std': float(results_df['crnn_confidence'].std())
        },
        'confidence_change': {
            'mean': float(results_df['confidence_change'].mean()),
            'median': float(results_df['confidence_change'].median())
        },
        'crnn_model_predictions': results_df['crnn_prediction'].value_counts().to_dict(),
        'panns_model_predictions': results_df['panns_prediction'].value_counts().to_dict()
    }
    
    summary_file = 'challenge_results/crnn_vs_panns_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary statistics to: {summary_file}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
