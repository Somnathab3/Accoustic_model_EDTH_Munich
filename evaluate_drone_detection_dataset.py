"""
Evaluate CRNN and PANNS models on the Drone-detection-dataset
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import model architectures
import sys
sys.path.append(str(Path(__file__).parent / 'src'))
from adrone.models.acoustic_models import CRNNWithAttention, PANNsCNN14


def extract_features(audio_path, use_hpss=True, sr=16000, n_mels=96, duration=5.0):
    """Extract mel-spectrogram features from audio (3-channel: total, harmonic, percussive)"""
    try:
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr, duration=duration)
        
        # Pad or truncate to fixed length
        target_length = int(sr * duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]
        
        if use_hpss:
            # Harmonic-Percussive Source Separation
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Mel spectrograms for all components
            mel_total = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512
            )
            mel_harmonic = librosa.feature.melspectrogram(
                y=y_harmonic, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512
            )
            mel_percussive = librosa.feature.melspectrogram(
                y=y_percussive, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512
            )
            
            # Convert to log scale
            mel_total_db = librosa.power_to_db(mel_total, ref=np.max)
            mel_harmonic_db = librosa.power_to_db(mel_harmonic, ref=np.max)
            mel_percussive_db = librosa.power_to_db(mel_percussive, ref=np.max)
            
            # Stack all three channels
            features = np.stack([mel_total_db, mel_harmonic_db, mel_percussive_db], axis=0)
        else:
            # Single channel mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            features = mel_db[np.newaxis, :]
        
        # Normalize
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        return torch.FloatTensor(features)
    
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def load_model(model_path, model_type='crnn', num_classes=3, input_channels=3, n_mels=96, device='cpu'):
    """Load trained model"""
    if model_type == 'crnn':
        model = CRNNWithAttention(
            num_classes=num_classes, 
            input_channels=input_channels,
            n_mels=n_mels,
            dropout=0.3
        )
    elif model_type == 'panns':
        model = PANNsCNN14(
            num_classes=num_classes, 
            input_channels=input_channels,
            dropout=0.3
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def evaluate_model(model, data_dir, label_mapping, use_hpss=True, n_mels=96, device='cpu'):
    """Evaluate model on dataset"""
    data_dir = Path(data_dir)
    
    # Get all audio files
    audio_files = list(data_dir.glob('*.wav'))
    
    # Parse labels from filenames
    all_preds = []
    all_labels = []
    all_probs = []
    file_results = []
    
    print(f"Evaluating on {len(audio_files)} files...")
    
    with torch.no_grad():
        for audio_file in tqdm(audio_files):
            # Extract true label from filename
            filename = audio_file.stem
            true_label_str = filename.split('_')[0].lower()
            
            if true_label_str not in label_mapping:
                print(f"Warning: Unknown label '{true_label_str}' in {filename}")
                continue
            
            true_label = label_mapping[true_label_str]
            
            # Extract features
            features = extract_features(audio_file, use_hpss=use_hpss, n_mels=n_mels)
            if features is None:
                continue
            
            # Predict
            features = features.unsqueeze(0).to(device)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            pred_label = torch.argmax(outputs, dim=1).item()
            
            all_preds.append(pred_label)
            all_labels.append(true_label)
            all_probs.append(probs.cpu().numpy()[0])
            
            file_results.append({
                'filename': filename,
                'true_label': true_label_str,
                'pred_label': list(label_mapping.keys())[pred_label],
                'confidence': float(probs[0, pred_label].item()),
                'correct': pred_label == true_label
            })
    
    return all_preds, all_labels, all_probs, file_results


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = Path('F:/EDTH/acoustic-drone-detector/data/Drone-detection-dataset')
    
    # Label mapping
    label_mapping = {
        'background': 0,
        'drone': 1,
        'helicopter': 2
    }
    idx_to_label = {v: k for k, v in label_mapping.items()}
    
    # Model paths
    models_to_evaluate = {
        'CRNN (Combined)': {
            'path': 'models/crnn_combined/best_model.pt',
            'type': 'crnn',
            'use_hpss': True,
            'input_channels': 3,
            'n_mels': 96
        },
        'CRNN (Hard Mining)': {
            'path': 'models/crnn_hardmining/best_model.pt',
            'type': 'crnn',
            'use_hpss': True,
            'input_channels': 3,
            'n_mels': 96
        },
        'PANNS (Combined)': {
            'path': 'models/panns_combined/best_model.pt',
            'type': 'panns',
            'use_hpss': True,
            'input_channels': 3,
            'n_mels': 96
        },
    }
    
    results_summary = []
    detailed_results = {}
    
    for model_name, model_config in models_to_evaluate.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print('='*60)
        
        model_path = Path(model_config['path'])
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            continue
        
        # Load model
        model = load_model(
            model_path,
            model_type=model_config['type'],
            num_classes=3,
            input_channels=model_config['input_channels'],
            n_mels=model_config['n_mels'],
            device=device
        )
        
        # Evaluate
        preds, labels, probs, file_results = evaluate_model(
            model, data_dir, label_mapping, 
            use_hpss=model_config['use_hpss'],
            n_mels=model_config['n_mels'],
            device=device
        )
        
        # Calculate metrics
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        
        # Print results
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        print(f"\nPer-Class Metrics:")
        for i, class_name in enumerate(['background', 'drone', 'helicopter']):
            print(f"  {class_name.capitalize()}:")
            print(f"    Precision: {precision_per_class[i]:.4f}")
            print(f"    Recall:    {recall_per_class[i]:.4f}")
            print(f"    F1-Score:  {f1_per_class[i]:.4f}")
            print(f"    Support:   {support_per_class[i]}")
        
        print(f"\nConfusion Matrix:")
        print("           Pred: BG   Drone  Heli")
        for i, class_name in enumerate(['Background', 'Drone', 'Helicopter']):
            print(f"True {class_name:10s}: {cm[i][0]:4d}  {cm[i][1]:5d}  {cm[i][2]:4d}")
        
        # Store results
        results_summary.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Drone_Precision': precision_per_class[1],
            'Drone_Recall': recall_per_class[1],
            'Drone_F1': f1_per_class[1],
            'Helicopter_Precision': precision_per_class[2],
            'Helicopter_Recall': recall_per_class[2],
            'Helicopter_F1': f1_per_class[2],
        })
        
        detailed_results[model_name] = {
            'overall': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            },
            'per_class': {
                class_name: {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1_score': float(f1_per_class[i]),
                    'support': int(support_per_class[i])
                }
                for i, class_name in enumerate(['background', 'drone', 'helicopter'])
            },
            'confusion_matrix': cm.tolist(),
            'file_results': file_results
        }
    
    # Save results
    output_dir = Path('evaluation_results')
    output_dir.mkdir(exist_ok=True)
    
    # Summary table
    df_summary = pd.DataFrame(results_summary)
    df_summary.to_csv(output_dir / 'drone_detection_dataset_evaluation.csv', index=False)
    
    # Detailed results
    with open(output_dir / 'drone_detection_dataset_detailed.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(df_summary.to_string(index=False))
    
    print(f"\nResults saved to: {output_dir}")
    print("  - drone_detection_dataset_evaluation.csv")
    print("  - drone_detection_dataset_detailed.json")


if __name__ == '__main__':
    main()
