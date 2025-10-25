"""
Quick test to see if bot can load CRNN model correctly
"""
import torch
from pathlib import Path

model_path = Path('models/crnn_combined/crnn_final.pt')

if not model_path.exists():
    print(f"Model not found: {model_path}")
    exit(1)

print(f"Loading checkpoint from: {model_path}")
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

print("\nCheckpoint keys:")
for key in checkpoint.keys():
    print(f"  - {key}")

print(f"\nModel type: {checkpoint.get('model_type', 'NOT FOUND')}")
print(f"Num classes: {checkpoint.get('num_classes', 'NOT FOUND')}")
print(f"Input channels: {checkpoint.get('input_channels', 'NOT FOUND')}")
print(f"N_mels: {checkpoint.get('n_mels', 'NOT FOUND')}")

print("\nFirst 10 model state_dict keys:")
state_dict_keys = list(checkpoint['model_state_dict'].keys())
for i, key in enumerate(state_dict_keys[:10]):
    print(f"  {i+1}. {key}")

print(f"\n... (total {len(state_dict_keys)} keys)")
