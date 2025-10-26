"""
Test model on real audio samples from the dataset
"""
import torch
import librosa
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent / 'src'))
from adrone.models.acoustic_models import CRNNWithAttention
import json

def extract_features(audio_path, sr=16000, n_mels=96, duration=5.0):
    """Extract 3-channel features (total, harmonic, percussive)"""
    # Load audio
    y, _ = librosa.load(audio_path, sr=sr, duration=duration)
    
    # Pad or truncate
    target_length = int(sr * duration)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
    
    # HPSS
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Mel spectrograms
    mel_total = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512)
    mel_harmonic = librosa.feature.melspectrogram(y=y_harmonic, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512)
    mel_percussive = librosa.feature.melspectrogram(y=y_percussive, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512)
    
    # Convert to dB
    mel_total_db = librosa.power_to_db(mel_total, ref=np.max)
    mel_harmonic_db = librosa.power_to_db(mel_harmonic, ref=np.max)
    mel_percussive_db = librosa.power_to_db(mel_percussive, ref=np.max)
    
    # Stack
    features = np.stack([mel_total_db, mel_harmonic_db, mel_percussive_db], axis=0)
    
    # Normalize
    features = (features - features.mean()) / (features.std() + 1e-8)
    
    return torch.FloatTensor(features)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load labels
    with open('models/crnn_combined/labels.json', 'r') as f:
        label_data = json.load(f)
    idx_to_class = {int(k): v for k, v in label_data['idx_to_class'].items()}
    
    # Load model
    model = CRNNWithAttention(num_classes=3, input_channels=3, n_mels=96, dropout=0.3)
    checkpoint = torch.load('models/crnn_combined/best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Test on samples from each class
    data_dir = Path('F:/EDTH/acoustic-drone-detector/data/Drone-detection-dataset')
    
    test_samples = [
        data_dir / 'BACKGROUND_001.wav',
        data_dir / 'BACKGROUND_002.wav',
        data_dir / 'DRONE_001.wav',
        data_dir / 'DRONE_002.wav',
        data_dir / 'HELICOPTER_001.wav',
        data_dir / 'HELICOPTER_002.wav',
    ]
    
    print("="*80)
    print("TESTING MODEL ON REAL AUDIO SAMPLES")
    print("="*80)
    
    for audio_path in test_samples:
        if not audio_path.exists():
            print(f"File not found: {audio_path}")
            continue
        
        # Extract features
        features = extract_features(audio_path, n_mels=96)
        features = features.unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            logits = model(features)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            
            true_label = audio_path.stem.split('_')[0].lower()
            pred_label = idx_to_class[pred_idx]
            
            print(f"\n{audio_path.name}")
            print(f"  True: {true_label:11s} | Predicted: {pred_label:11s} | {'✓' if pred_label == true_label else '✗'}")
            print(f"  Logits: [{logits[0, 0].item():6.3f}, {logits[0, 1].item():6.3f}, {logits[0, 2].item():6.3f}]")
            print(f"  Probs:  background={probs[0, 0].item():.3f}, drone={probs[0, 1].item():.3f}, helicopter={probs[0, 2].item():.3f}")
            print(f"  Feature stats: mean={features.mean().item():.3f}, std={features.std().item():.3f}, min={features.min().item():.3f}, max={features.max().item():.3f}")
    
    # Check model parameters statistics
    print(f"\n{'='*80}")
    print("MODEL PARAMETER STATISTICS")
    print("="*80)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Check final layer weights
    fc_weight = model.fc.weight.data.cpu().numpy()
    fc_bias = model.fc.bias.data.cpu().numpy()
    
    print(f"\nFinal layer (fc) weights shape: {fc_weight.shape}")
    print(f"Final layer bias: {fc_bias}")
    print(f"\nWeight norms by class:")
    for i in range(3):
        weight_norm = np.linalg.norm(fc_weight[i])
        print(f"  {idx_to_class[i]:11s}: {weight_norm:.4f}")

if __name__ == '__main__':
    main()
