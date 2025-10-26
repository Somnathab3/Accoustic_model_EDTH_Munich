"""
Test CRNN model with CORRECT preprocessing (matching challenge bot)
"""
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

from adrone.models.acoustic_models import CRNNWithAttention
from adrone.preprocessing import AudioPreprocessor
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import json

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load model
    print("="*70)
    print("TESTING WITH CORRECT PREPROCESSING")
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
    
    # Initialize preprocessor (SAME AS CHALLENGE BOT!)
    preprocessor = AudioPreprocessor(
        sample_rate=16000,
        n_fft=1024,
        hop_length=320,
        n_mels=96,
        window_duration=2.0,  # 2 seconds like challenge bot, NOT 5!
        use_hpss=True  # Creates 3 channels: total, harmonic, percussive
    )
    
    print("\nPreprocessor settings:")
    print(f"  Sample rate: {preprocessor.sample_rate}")
    print(f"  N_FFT: {preprocessor.n_fft}")
    print(f"  Hop length: {preprocessor.hop_length}")
    print(f"  N_mels: {preprocessor.n_mels}")
    print(f"  Window duration: {preprocessor.window_duration}s")
    print(f"  HPSS: {preprocessor.use_hpss}")
    print(f"  Target length: {preprocessor.target_length} samples")
    
    # Test on original validation set
    val_dir = Path('data/combined_dataset/val')
    
    print("\n" + "="*70)
    print("TESTING ON ORIGINAL VALIDATION SET")
    print("="*70)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for class_name, class_idx in class_to_idx.items():
            class_dir = val_dir / class_name
            audio_files = list(class_dir.glob('*.wav'))
            print(f"\n{class_name}: {len(audio_files)} files")
            
            for audio_file in tqdm(audio_files, desc=f"  {class_name}", leave=False):
                # Use AudioPreprocessor (same as challenge bot!)
                waveform = preprocessor.load_audio(str(audio_file))
                spectrogram = preprocessor(waveform)
                
                # Add batch dimension
                spectrogram = spectrogram.unsqueeze(0).to(device)
                
                # Inference
                logits = model(spectrogram)
                pred_idx = torch.argmax(logits, dim=1).item()
                
                all_preds.append(pred_idx)
                all_labels.append(class_idx)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\n{'='*70}")
    print("RESULTS WITH CORRECT PREPROCESSING")
    print('='*70)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print(f"\nConfusion Matrix:")
    print("                Predicted")
    print("              BG    Drone  Heli")
    for i, class_name in enumerate(['Background', 'Drone', 'Helicopter']):
        print(f"True {class_name:10s}: {cm[i][0]:4d}  {cm[i][1]:5d}  {cm[i][2]:4d}")
    
    print(f"\nPer-Class Accuracy:")
    for i, class_name in enumerate(['background', 'drone', 'helicopter']):
        class_total = sum(cm[i])
        class_acc = cm[i][i] / class_total if class_total > 0 else 0
        print(f"  {class_name.capitalize():11s}: {cm[i][i]:3d}/{class_total:3d} = {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    # Now test on Drone-detection-dataset
    print("\n\n" + "="*70)
    print("TESTING ON DRONE-DETECTION-DATASET")
    print("="*70)
    
    data_dir = Path('F:/EDTH/acoustic-drone-detector/data/Drone-detection-dataset')
    
    all_preds2 = []
    all_labels2 = []
    
    with torch.no_grad():
        for class_name, class_idx in class_to_idx.items():
            # Files are named like "DRONE_001.wav", "BACKGROUND_001.wav", etc.
            audio_files = list(data_dir.glob(f'{class_name.upper()}_*.wav'))
            print(f"\n{class_name}: {len(audio_files)} files")
            
            for audio_file in tqdm(audio_files, desc=f"  {class_name}", leave=False):
                # Use AudioPreprocessor
                waveform = preprocessor.load_audio(str(audio_file))
                spectrogram = preprocessor(waveform)
                
                # Add batch dimension
                spectrogram = spectrogram.unsqueeze(0).to(device)
                
                # Inference
                logits = model(spectrogram)
                pred_idx = torch.argmax(logits, dim=1).item()
                
                all_preds2.append(pred_idx)
                all_labels2.append(class_idx)
    
    # Calculate metrics
    accuracy2 = accuracy_score(all_labels2, all_preds2)
    cm2 = confusion_matrix(all_labels2, all_preds2)
    
    print(f"\n{'='*70}")
    print("RESULTS ON DRONE-DETECTION-DATASET (WITH CORRECT PREPROCESSING)")
    print('='*70)
    print(f"\nOverall Accuracy: {accuracy2:.4f} ({accuracy2*100:.2f}%)")
    
    print(f"\nConfusion Matrix:")
    print("                Predicted")
    print("              BG    Drone  Heli")
    for i, class_name in enumerate(['Background', 'Drone', 'Helicopter']):
        print(f"True {class_name:10s}: {cm2[i][0]:4d}  {cm2[i][1]:5d}  {cm2[i][2]:4d}")
    
    print(f"\nPer-Class Accuracy:")
    for i, class_name in enumerate(['background', 'drone', 'helicopter']):
        class_total = sum(cm2[i])
        class_acc = cm2[i][i] / class_total if class_total > 0 else 0
        print(f"  {class_name.capitalize():11s}: {cm2[i][i]:3d}/{class_total:3d} = {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
