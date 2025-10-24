#!/usr/bin/env python
"""
Simple Inference Script for FFT-CNN-DNN Model
Optimized for deployment on Kaggle and remote platforms
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import librosa
import numpy as np
import json
import argparse
from pathlib import Path

from adrone.models.fft_cnn_dnn import FFTCNNDNNFusion
from adrone.features.fft_processor import FFTProcessor


class SimpleInference:
    """Simplified inference class for deployment"""
    
    def __init__(
        self,
        model_path: str = "models/cnn_edth_3class_improved.pt",
        labels_path: str = "models/labels_edth_3class_improved.json",
        device: str = None
    ):
        """
        Initialize inference model
        
        Args:
            model_path: Path to model checkpoint
            labels_path: Path to labels JSON
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load labels
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)
        self.idx_to_label = {v: k for k, v in self.labels.items()}
        
        # Initialize FFT processor
        self.fft_processor = FFTProcessor(
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            sample_rate=16000
        )
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        self.model = FFTCNNDNNFusion(
            n_classes=len(self.labels),
            in_channels=1,
            cnn_feature_dim=512,
            dnn_hidden_dims=[256, 128]
        )
        
        # Load weights
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Classes: {list(self.labels.keys())}")
    
    def predict(self, audio_path: str, top_k: int = 3) -> dict:
        """
        Run inference on audio file
        
        Args:
            audio_path: Path to audio file
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, duration=2.0)
        
        # Pad or trim to 2 seconds
        target_length = 16000 * 2
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        # Extract features
        features = self.fft_processor.extract_features_for_model(audio)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
        features_tensor = features_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(self.labels)))
        
        results = {
            'predictions': {},
            'top_prediction': None,
            'confidence': None,
            'all_probabilities': {}
        }
        
        # All probabilities
        for idx, prob in enumerate(probabilities.cpu().numpy()):
            label = self.idx_to_label[idx]
            results['all_probabilities'][label] = float(prob)
        
        # Top predictions
        for idx, (prob, class_idx) in enumerate(zip(top_probs.cpu().numpy(), top_indices.cpu().numpy())):
            label = self.idx_to_label[int(class_idx)]
            results['predictions'][label] = float(prob)
            
            if idx == 0:
                results['top_prediction'] = label
                results['confidence'] = float(prob)
        
        return results
    
    def predict_batch(self, audio_paths: list) -> list:
        """
        Run inference on multiple audio files
        
        Args:
            audio_paths: List of paths to audio files
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path)
                result['file'] = audio_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                results.append({'file': audio_path, 'error': str(e)})
        
        return results


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Run FFT-CNN-DNN inference on audio files'
    )
    parser.add_argument(
        'audio_files',
        nargs='+',
        help='Path(s) to audio file(s)'
    )
    parser.add_argument(
        '--model',
        default='models/cnn_edth_3class_improved.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--labels',
        default='models/labels_edth_3class_improved.json',
        help='Path to labels JSON'
    )
    parser.add_argument(
        '--device',
        default=None,
        help='Device to use (cuda/cpu, default: auto)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of top predictions to show'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    args = parser.parse_args()
    
    # Initialize model
    print("Initializing model...")
    model = SimpleInference(
        model_path=args.model,
        labels_path=args.labels,
        device=args.device
    )
    print()
    
    # Run inference
    if len(args.audio_files) == 1:
        # Single file
        audio_path = args.audio_files[0]
        print(f"Processing: {audio_path}")
        print("-" * 60)
        
        result = model.predict(audio_path, top_k=args.top_k)
        
        if args.json:
            import json
            print(json.dumps(result, indent=2))
        else:
            print(f"\n✓ Top Prediction: {result['top_prediction']}")
            print(f"  Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
            print(f"\nTop {args.top_k} Predictions:")
            for label, prob in result['predictions'].items():
                print(f"  {label:15s}: {prob:.4f} ({prob*100:.2f}%)")
            print(f"\nAll Probabilities:")
            for label, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {label:15s}: {prob:.4f}")
    
    else:
        # Multiple files
        print(f"Processing {len(args.audio_files)} files...")
        print("-" * 60)
        
        results = model.predict_batch(args.audio_files)
        
        if args.json:
            import json
            print(json.dumps(results, indent=2))
        else:
            for result in results:
                if 'error' in result:
                    print(f"\n✗ {result['file']}: ERROR - {result['error']}")
                else:
                    print(f"\n✓ {result['file']}")
                    print(f"  Prediction: {result['top_prediction']} ({result['confidence']*100:.2f}%)")


if __name__ == '__main__':
    main()
