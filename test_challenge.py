"""
Simple test script to try one challenge manually
Useful for testing the API and model before running the full bot
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import requests
import torch
import librosa
import numpy as np
import json
from pathlib import Path
from adrone.models.cnn_small import CNNSmall

# Configuration
API_BASE_URL = "https://edth.helsing.codes"
API_TOKEN = "9726345a-34ed-4995-94d9-ecc239b47c1d"
MODEL_PATH = "models/cnn_edth_3class.pt"
LABELS_PATH = "models/labels_edth_3class.json"

# Audio parameters (from train_edth.yaml)
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 320
MAX_DURATION = 2.0


def preprocess_audio(audio_path):
    """Preprocess audio to match training format"""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=MAX_DURATION)
    
    target_length = int(SAMPLE_RATE * MAX_DURATION)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    else:
        y = y[:target_length]
    
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
    
    return torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0)


def main():
    print("🤖 Testing Challenge Bot\n")
    
    # Load model
    print("Loading model...")
    with open(LABELS_PATH, 'r') as f:
        labels = json.load(f)['labels']
    
    model = CNNSmall(n_classes=len(labels))
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=False))
    model.eval()
    print(f"Classes: {labels}\n")
    
    # Get challenge
    print("Fetching challenge...")
    response = requests.get(f"{API_BASE_URL}/api/challenge")
    challenge = response.json()
    print(f"Challenge ID: {challenge['challenge_id']}")
    print(f"Audio URL: {challenge['wav_url']}")
    print(f"Time left: {challenge['time_until_next_rotation_ms']}ms\n")
    
    # Download audio
    print("Downloading audio...")
    audio_file = "test_challenge.wav"
    audio_url = f"{API_BASE_URL}{challenge['wav_url']}"
    audio_data = requests.get(audio_url).content
    with open(audio_file, 'wb') as f:
        f.write(audio_data)
    print(f"Saved to {audio_file}\n")
    
    # Classify
    print("Classifying...")
    audio_tensor = preprocess_audio(audio_file)
    with torch.no_grad():
        outputs = model(audio_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
    
    prediction = labels[pred_idx]
    print(f"Prediction: {prediction.upper()}")
    print(f"Confidence: {confidence:.2%}")
    print("\nAll probabilities:")
    for label, prob in zip(labels, probs[0].tolist()):
        print(f"  {label:12s}: {prob:.2%}")
    
    # Ask user to confirm
    print(f"\n📤 Ready to submit '{prediction}'")
    user_input = input("Submit? (y/n or enter different classification): ").strip().lower()
    
    if user_input and user_input != 'y':
        if user_input == 'n':
            print("Submission cancelled.")
            return
        else:
            prediction = user_input
            print(f"Using manual classification: {prediction}")
    
    # Submit
    print(f"\nSubmitting classification: {prediction}")
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "challenge_id": challenge['challenge_id'],
        "classification": prediction
    }
    
    response = requests.post(
        f"{API_BASE_URL}/api/challenge",
        headers=headers,
        json=payload
    )
    result = response.json()
    
    print("\n" + "="*60)
    if result.get('success'):
        print("✅ CORRECT!")
        print(f"Score awarded: {result.get('score_awarded', 0)}")
        print(f"Total score: {result.get('total_score', 0)}")
    else:
        print("❌ WRONG!")
        print(f"Message: {result.get('message', 'No message')}")
    print("="*60)


if __name__ == "__main__":
    main()
