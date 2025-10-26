"""
Check labels and audio characteristics in the Drone-detection-dataset
"""
import librosa
import numpy as np
from pathlib import Path
import pandas as pd

def check_audio_info(audio_path):
    """Get basic info about audio file"""
    try:
        y, sr = librosa.load(audio_path, sr=None, duration=1)
        duration = librosa.get_duration(path=audio_path)
        return {
            'sample_rate': sr,
            'duration': duration,
            'samples': len(y),
            'has_audio': np.abs(y).max() > 0.001
        }
    except Exception as e:
        return {'error': str(e)}

def main():
    data_dir = Path('F:/EDTH/acoustic-drone-detector/data/Drone-detection-dataset')
    
    # Get all files
    audio_files = sorted(list(data_dir.glob('*.wav')))
    
    print(f"Total files found: {len(audio_files)}\n")
    
    # Check labels
    label_counts = {}
    samples_info = []
    
    for audio_file in audio_files[:10]:  # Check first 10 of each type
        filename = audio_file.stem
        label = filename.split('_')[0]
        
        # Get audio info
        info = check_audio_info(audio_file)
        
        samples_info.append({
            'filename': filename,
            'label_extracted': label,
            'label_lower': label.lower(),
            **info
        })
        
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("="*60)
    print("LABEL EXTRACTION CHECK")
    print("="*60)
    df = pd.DataFrame(samples_info)
    print(df.to_string(index=False))
    
    print(f"\n{'='*60}")
    print("LABEL COUNTS (first 10)")
    print('='*60)
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    # Check all labels
    all_labels = {}
    for audio_file in audio_files:
        label = audio_file.stem.split('_')[0].lower()
        all_labels[label] = all_labels.get(label, 0) + 1
    
    print(f"\n{'='*60}")
    print("ALL LABEL COUNTS")
    print('='*60)
    for label, count in sorted(all_labels.items()):
        print(f"  {label}: {count}")
    
    # Check some sample durations from each class
    print(f"\n{'='*60}")
    print("SAMPLE AUDIO CHARACTERISTICS")
    print('='*60)
    
    for label_prefix in ['BACKGROUND', 'DRONE', 'HELICOPTER']:
        files = list(data_dir.glob(f'{label_prefix}_*.wav'))[:3]
        print(f"\n{label_prefix}:")
        for f in files:
            info = check_audio_info(f)
            print(f"  {f.name}: {info}")

if __name__ == '__main__':
    main()
