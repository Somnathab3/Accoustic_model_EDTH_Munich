"""
Simple validation on original dataset
"""
import torch
import librosa
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent / 'src'))
from adrone.models.acoustic_models import CRNNWithAttention, PANNsCNN14


def extract_features(audio_path, sr=16000, n_mels=96, duration=5.0):
    """Extract 3-channel features"""
    y, _ = librosa.load(audio_path, sr=sr, duration=duration)
    target_length = int(sr * duration)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
    
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    mel_total = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512)
    mel_harmonic = librosa.feature.melspectrogram(y=y_harmonic, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512)
    mel_percussive = librosa.feature.melspectrogram(y=y_percussive, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512)
    
    mel_total_db = librosa.power_to_db(mel_total, ref=np.max)
    mel_harmonic_db = librosa.power_to_db(mel_harmonic, ref=np.max)
    mel_percussive_db = librosa.power_to_db(mel_percussive, ref=np.max)
    
    features = np.stack([mel_total_db, mel_harmonic_db, mel_percussive_db], axis=0)
    features = (features - features.mean()) / (features.std() + 1e-8)
    
    return torch.FloatTensor(features)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load CRNN model
    print("="*70)
    print("EVALUATING CRNN MODEL ON ORIGINAL VALIDATION SET")
    print("="*70)
    
    model = CRNNWithAttention(num_classes=3, input_channels=3, n_mels=96, dropout=0.3)
    checkpoint = torch.load('models/crnn_combined/best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load labels
    with open('models/crnn_combined/labels.json', 'r') as f:
        label_data = json.load(f)
    class_to_idx = label_data['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Evaluate on validation set
    val_dir = Path('data/combined_dataset/val')
    
    all_preds = []
    all_labels = []
    
    print("\nProcessing validation set...")
    
    with torch.no_grad():
        for class_name, class_idx in class_to_idx.items():
            class_dir = val_dir / class_name
            audio_files = list(class_dir.glob('*.wav'))
            print(f"  {class_name}: {len(audio_files)} files")
            
            for audio_file in tqdm(audio_files, desc=f"  {class_name}", leave=False):
                features = extract_features(audio_file, n_mels=96)
                features = features.unsqueeze(0).to(device)
                
                logits = model(features)
                pred_idx = torch.argmax(logits, dim=1).item()
                
                all_preds.append(pred_idx)
                all_labels.append(class_idx)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print results
    print(f"\n{'='*70}")
    print("RESULTS ON ORIGINAL VALIDATION SET")
    print('='*70)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print(f"\nPer-Class Metrics:")
    for i, class_name in enumerate(['background', 'drone', 'helicopter']):
        print(f"  {class_name.upper()}:")
        print(f"    Precision: {precision_per_class[i]:.4f}")
        print(f"    Recall:    {recall_per_class[i]:.4f}")
        print(f"    F1-Score:  {f1_per_class[i]:.4f}")
        print(f"    Support:   {support_per_class[i]}")
    
    print(f"\nConfusion Matrix:")
    print("                Predicted")
    print("              BG    Drone  Heli")
    for i, class_name in enumerate(['Background', 'Drone', 'Helicopter']):
        print(f"True {class_name:10s}: {cm[i][0]:4d}  {cm[i][1]:5d}  {cm[i][2]:4d}")
    
    print(f"\nPer-Class Accuracy:")
    for i, class_name in enumerate(['background', 'drone', 'helicopter']):
        class_acc = cm[i][i] / support_per_class[i]
        print(f"  {class_name.capitalize():11s}: {cm[i][i]:3d}/{support_per_class[i]:3d} = {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
