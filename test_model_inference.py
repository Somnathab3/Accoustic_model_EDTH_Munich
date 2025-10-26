"""
Debug script to check model inference and label mapping
"""
import torch
import json
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent / 'src'))
from adrone.models.acoustic_models import CRNNWithAttention, PANNsCNN14

def test_model_output():
    """Test what the model actually outputs"""
    
    # Load labels
    labels_path = Path('models/crnn_combined/labels.json')
    with open(labels_path, 'r') as f:
        label_data = json.load(f)
    
    print("="*60)
    print("LABEL MAPPING CHECK")
    print("="*60)
    print(f"\nFrom labels.json:")
    print(f"  class_to_idx: {label_data['class_to_idx']}")
    print(f"  idx_to_class: {label_data['idx_to_class']}")
    
    # Check what list(keys()) gives
    label_mapping = label_data['class_to_idx']
    print(f"\nlist(label_mapping.keys()): {list(label_mapping.keys())}")
    print(f"list(label_mapping.keys())[0]: {list(label_mapping.keys())[0]}")
    print(f"list(label_mapping.keys())[1]: {list(label_mapping.keys())[1]}")
    print(f"list(label_mapping.keys())[2]: {list(label_mapping.keys())[2]}")
    
    # Create idx_to_class properly
    idx_to_class_correct = {v: k for k, v in label_mapping.items()}
    print(f"\nCorrect idx_to_class: {idx_to_class_correct}")
    print(f"idx_to_class_correct[0]: {idx_to_class_correct[0]}")
    print(f"idx_to_class_correct[1]: {idx_to_class_correct[1]}")
    print(f"idx_to_class_correct[2]: {idx_to_class_correct[2]}")
    
    # Load model
    print(f"\n{'='*60}")
    print("MODEL OUTPUT TEST")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = Path('models/crnn_combined/best_model.pt')
    
    # Load model
    model = CRNNWithAttention(num_classes=3, input_channels=3, n_mels=96, dropout=0.3)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create dummy input (batch_size=1, channels=3, n_mels=96, time=126)
    dummy_input = torch.randn(1, 3, 96, 126).to(device)
    
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        
        print(f"\nDummy input shape: {dummy_input.shape}")
        print(f"Logits: {logits[0].cpu().numpy()}")
        print(f"Probabilities: {probs[0].cpu().numpy()}")
        print(f"Predicted index: {pred_idx}")
        print(f"Predicted class (using idx_to_class): {idx_to_class_correct[pred_idx]}")
        print(f"Predicted class (using list(keys)): {list(label_mapping.keys())[pred_idx]}")
        
        # Show probability for each class
        print(f"\nProbabilities by class:")
        for idx in range(3):
            print(f"  idx {idx} ({idx_to_class_correct[idx]}): {probs[0, idx].item():.4f}")
    
    # Test with BACKGROUND-like features (all zeros - silence)
    print(f"\n{'='*60}")
    print("TESTING WITH SILENCE (should predict background)")
    print("="*60)
    
    zero_input = torch.zeros(1, 3, 96, 126).to(device)
    with torch.no_grad():
        logits = model(zero_input)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        
        print(f"Logits: {logits[0].cpu().numpy()}")
        print(f"Probabilities: {probs[0].cpu().numpy()}")
        print(f"Predicted index: {pred_idx}")
        print(f"Predicted class: {idx_to_class_correct[pred_idx]}")
        
        for idx in range(3):
            print(f"  {idx_to_class_correct[idx]}: {probs[0, idx].item():.4f}")
    
    # Test with high-energy input (should predict drone or helicopter)
    print(f"\n{'='*60}")
    print("TESTING WITH HIGH ENERGY (should predict drone/helicopter)")
    print("="*60)
    
    high_energy = torch.ones(1, 3, 96, 126).to(device) * 5.0  # High values
    with torch.no_grad():
        logits = model(high_energy)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        
        print(f"Logits: {logits[0].cpu().numpy()}")
        print(f"Probabilities: {probs[0].cpu().numpy()}")
        print(f"Predicted index: {pred_idx}")
        print(f"Predicted class: {idx_to_class_correct[pred_idx]}")
        
        for idx in range(3):
            print(f"  {idx_to_class_correct[idx]}: {probs[0, idx].item():.4f}")

if __name__ == '__main__':
    test_model_output()
