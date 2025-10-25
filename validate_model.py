"""
Validation script for SOTA models
Checks model performance on validation set with comprehensive metrics
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from adrone.preprocessing import AudioPreprocessor
from adrone.models.acoustic_models import create_model
from adrone.evaluation import evaluate_model, print_evaluation_report

def load_model_and_config(model_path: str):
    """Load model checkpoint and extract configuration"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n🔧 Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract model configuration
    model_type = checkpoint.get('model_type', 'panns')
    input_channels = checkpoint.get('input_channels', 3)
    num_classes = checkpoint.get('num_classes', 3)
    n_mels = checkpoint.get('n_mels', 96)
    
    print(f"   Model type: {model_type}")
    print(f"   Input channels: {input_channels}")
    print(f"   Num classes: {num_classes}")
    print(f"   N mels: {n_mels}")
    
    # Create model
    model = create_model(
        model_type=model_type,
        num_classes=num_classes,
        input_channels=input_channels,
        n_mels=n_mels if model_type == 'crnn' else None
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    config = {
        'model_type': model_type,
        'input_channels': input_channels,
        'num_classes': num_classes,
        'n_mels': n_mels,
        'device': device
    }
    
    if 'best_epoch' in checkpoint:
        config['best_epoch'] = checkpoint['best_epoch']
        print(f"   Best epoch: {checkpoint['best_epoch']}")
    if 'best_macro_f1' in checkpoint:
        config['best_macro_f1'] = checkpoint['best_macro_f1']
        print(f"   Best macro F1: {checkpoint['best_macro_f1']:.4f}")
    
    return model, config

def validate_on_directory(
    model,
    val_dir: str,
    preprocessor: AudioPreprocessor,
    labels_path: str,
    device: torch.device,
    show_errors: bool = True
):
    """
    Validate model on a directory of audio files
    
    Args:
        model: Trained model
        val_dir: Path to validation directory with class subdirectories
        preprocessor: AudioPreprocessor instance
        labels_path: Path to labels JSON file
        device: Device to use
        show_errors: Whether to print misclassified samples
    """
    # Load labels
    with open(labels_path, 'r') as f:
        label_data = json.load(f)
        
        # Handle both labels.json and config.json formats
        if 'class_to_idx' in label_data:
            class_to_idx = label_data['class_to_idx']
            idx_to_class = {int(k): v for k, v in label_data['idx_to_class'].items()}
        elif 'labels' in label_data:
            # Old format with just labels list
            labels_list = label_data['labels']
            class_to_idx = {label: i for i, label in enumerate(labels_list)}
            idx_to_class = {i: label for i, label in enumerate(labels_list)}
        else:
            print(f"❌ ERROR: Invalid labels file format!")
            print(f"Expected 'class_to_idx' and 'idx_to_class' or 'labels' in {labels_path}")
            print(f"Found keys: {list(label_data.keys())}")
            return None
    
    labels = list(class_to_idx.keys())
    
    # Collect all audio files
    val_path = Path(val_dir)
    all_files = []
    
    for class_name in labels:
        class_dir = val_path / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist, skipping")
            continue
        
        audio_files = list(class_dir.glob('*.wav'))
        for audio_file in audio_files:
            all_files.append((str(audio_file), class_name))
    
    if not all_files:
        print("❌ No audio files found!")
        return
    
    print(f"\nValidating on {len(all_files)} samples...")
    
    # Validate
    y_true = []
    y_pred = []
    y_probs = []
    errors = []
    
    predictions_by_class = {label: {'correct': 0, 'total': 0} for label in labels}
    
    model.eval()
    
    with torch.no_grad():
        for audio_path, true_label in tqdm(all_files, desc="Validating"):
            try:
                # Preprocess
                waveform = preprocessor.load_audio(audio_path)
                spectrogram = preprocessor(waveform)
                spectrogram = spectrogram.unsqueeze(0).to(device)
                
                # Predict
                logits = model(spectrogram)
                probabilities = F.softmax(logits, dim=1)
                predicted_idx = probabilities.argmax(dim=1).item()
                confidence = probabilities[0, predicted_idx].item()
                
                predicted_label = idx_to_class[predicted_idx]
                
                y_true.append(true_label)
                y_pred.append(predicted_label)
                y_probs.append(probabilities.cpu().numpy())
                
                # Track per-class accuracy
                predictions_by_class[true_label]['total'] += 1
                if predicted_label == true_label:
                    predictions_by_class[true_label]['correct'] += 1
                else:
                    errors.append({
                        'file': Path(audio_path).name,
                        'true': true_label,
                        'pred': predicted_label,
                        'conf': confidence
                    })
            
            except Exception as e:
                print(f"\nError processing {audio_path}: {e}")
    
    # Results
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    
    # Per-class accuracy
    print("\nPer-Class Performance:")
    print(f"{'Class':<12} {'Correct':<8} {'Total':<8} {'Accuracy':<10} {'F1-Score':<10}")
    print("-" * 80)
    
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    
    for i, label in enumerate(labels):
        correct = predictions_by_class[label]['correct']
        total = predictions_by_class[label]['total']
        accuracy = 100 * correct / total if total > 0 else 0
        print(f"{label:<12} {correct:<8} {total:<8} {accuracy:<10.2f}% {f1[i]:<10.4f}")
    
    # Overall metrics
    overall_accuracy = 100 * sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"\n{'Overall Accuracy:':<20} {overall_accuracy:.2f}%")
    print(f"{'Macro F1:':<20} {macro_f1:.4f}")
    print(f"{'Weighted F1:':<20} {weighted_f1:.4f}")
    
    # Confusion matrix
    print("\n" + "="*80)
    print("CONFUSION MATRIX")
    print("="*80)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    print(f"\n{'':>12}", end="")
    for label in labels:
        print(f"{label:>12}", end="")
    print()
    print("-" * (12 + 12 * len(labels)))
    
    for i, label in enumerate(labels):
        print(f"{label:>12}", end="")
        for j in range(len(labels)):
            print(f"{cm[i][j]:>12}", end="")
        print()
    
    # Classification report
    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*80)
    print()
    print(classification_report(y_true, y_pred, labels=labels, target_names=labels, digits=4))
    
    # Show errors
    if show_errors and errors:
        print("\n" + "="*80)
        print(f"MISCLASSIFIED SAMPLES (showing first 20 of {len(errors)})")
        print("="*80)
        print(f"\n{'File':<30} {'True':<12} {'Predicted':<12} {'Confidence':<12}")
        print("-" * 80)
        
        for error in errors[:20]:
            print(f"{error['file']:<30} {error['true']:<12} {error['pred']:<12} {error['conf']:<12.4f}")
    
    return {
        'accuracy': overall_accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'confusion_matrix': cm,
        'per_class': predictions_by_class,
        'errors': errors
    }


def main():
    parser = argparse.ArgumentParser(description='Validate SOTA acoustic drone detection model')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint (e.g., models/panns/panns_final.pt)')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to labels JSON file')
    parser.add_argument('--val-dir', type=str, required=True,
                        help='Path to validation directory')
    parser.add_argument('--use-hpss', action='store_true', default=True,
                        help='Use HPSS preprocessing (should match training)')
    parser.add_argument('--show-errors', action='store_true', default=True,
                        help='Show misclassified samples')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("SOTA MODEL VALIDATION")
    print("="*80)
    
    # Check paths
    model_path = Path(args.model)
    labels_path = Path(args.labels)
    val_dir = Path(args.val_dir).resolve()
    
    if not model_path.exists():
        print(f"\n❌ ERROR: Model not found: {model_path}")
        sys.exit(1)
    
    if not labels_path.exists():
        print(f"\n❌ ERROR: Labels file not found: {labels_path}")
        sys.exit(1)
    
    if not val_dir.exists():
        print(f"\n❌ ERROR: Validation directory not found: {val_dir}")
        sys.exit(1)
    
    print(f"\n✓ Model: {model_path}")
    print(f"✓ Labels: {labels_path}")
    print(f"✓ Validation dir: {val_dir}")
    
    # Load model and config
    print("\nLoading model...")
    model, config = load_model_and_config(str(model_path))
    
    print(f"  Model type: {config['model_type']}")
    print(f"  Input channels: {config['input_channels']}")
    print(f"  Device: {config['device']}")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params / 1e6:.2f}M")
    
    if 'best_epoch' in config:
        print(f"  Best epoch: {config['best_epoch']}")
    if 'best_macro_f1' in config:
        print(f"  Training Macro F1: {config['best_macro_f1']:.4f}")
    
    # Create preprocessor (matching training)
    print("\nInitializing preprocessor...")
    preprocessor = AudioPreprocessor(
        sample_rate=16000,
        n_fft=1024,
        hop_length=320,
        n_mels=config.get('n_mels', 96),
        window_duration=2.0,
        use_hpss=args.use_hpss
    )
    
    # Validate
    results = validate_on_directory(
        model=model,
        val_dir=str(val_dir),
        preprocessor=preprocessor,
        labels_path=str(labels_path),
        device=config['device'],
        show_errors=args.show_errors
    )
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"\nFinal Accuracy: {results['accuracy']:.2f}%")
    print(f"Final Macro F1: {results['macro_f1']:.4f}")
    print(f"Total misclassified: {len(results['errors'])}")
    
    # Save results
    output_dir = model_path.parent
    results_file = output_dir / 'validation_results.json'
    
    results_to_save = {
        'model': str(model_path),
        'validation_dir': str(val_dir),
        'accuracy': float(results['accuracy']),
        'macro_f1': float(results['macro_f1']),
        'weighted_f1': float(results['weighted_f1']),
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'per_class': results['per_class'],
        'num_errors': len(results['errors']),
        'errors': results['errors'][:50]  # Save first 50 errors
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")


if __name__ == "__main__":
    main()
