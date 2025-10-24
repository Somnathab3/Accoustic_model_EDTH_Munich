"""
Automated Challenge Bot for Drone Acoustic Classification
Uses the trained cnn_edth_3class model to participate in real-time challenges
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import requests
import torch
import librosa
import numpy as np
import json
import time
from pathlib import Path
import tempfile
from datetime import datetime
from adrone.models.cnn_small import CNNSmall

# Configuration
API_BASE_URL = "https://edth.helsing.codes"
API_TOKEN = "9726345a-34ed-4995-94d9-ecc239b47c1d"
MODEL_PATH = "models/cnn_edth_3class.pt"
LABELS_PATH = "models/labels_edth_3class.json"

# Audio processing parameters (from train_edth.yaml config)
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 320
MAX_DURATION = 2.0  # seconds


class ChallengeBotAPIError(Exception):
    """Custom exception for API errors"""
    pass


def load_model_and_labels():
    """Load the trained model and label mapping"""
    print(f"Loading model from {MODEL_PATH}...")
    
    # Load labels first to know number of classes
    with open(LABELS_PATH, 'r') as f:
        labels_data = json.load(f)
        labels = labels_data['labels']
    
    # Create model architecture and load state dict
    model = CNNSmall(n_classes=len(labels))
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=False))
    model.eval()
    
    print(f"Model loaded successfully! Classes: {labels}")
    return model, labels


def preprocess_audio(audio_path):
    """
    Preprocess audio file to match training format
    Returns mel spectrogram as tensor
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=MAX_DURATION)
    
    # Pad or truncate to fixed length
    target_length = int(SAMPLE_RATE * MAX_DURATION)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    else:
        y = y[:target_length]
    
    # Create mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    
    # Convert to log scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
    
    # Convert to tensor and add batch and channel dimensions
    tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0)
    
    return tensor


def get_current_challenge():
    """Fetch the current challenge from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/challenge", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise ChallengeBotAPIError(f"Failed to fetch challenge: {e}")


def download_audio(wav_url, output_path):
    """Download the audio file from the challenge"""
    try:
        full_url = f"{API_BASE_URL}{wav_url}"
        response = requests.get(full_url, timeout=10)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        return True
    except requests.exceptions.RequestException as e:
        raise ChallengeBotAPIError(f"Failed to download audio: {e}")


def classify_audio(model, audio_path, labels):
    """Classify the audio file using the trained model"""
    try:
        # Preprocess
        audio_tensor = preprocess_audio(audio_path)
        
        # Inference
        with torch.no_grad():
            outputs = model(audio_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_idx].item()
        
        prediction = labels[predicted_idx]
        return prediction, confidence, probabilities[0].tolist()
    
    except Exception as e:
        raise ChallengeBotAPIError(f"Classification failed: {e}")


def submit_classification(challenge_id, classification):
    """Submit the classification to the API"""
    try:
        headers = {
            "Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "challenge_id": challenge_id,
            "classification": classification
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/challenge",
            headers=headers,
            json=payload,
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        raise ChallengeBotAPIError(f"Failed to submit classification: {e}")


def run_challenge_bot(max_iterations=None, delay_between_challenges=1.0):
    """
    Main bot loop
    
    Args:
        max_iterations: Maximum number of challenges to attempt (None for infinite)
        delay_between_challenges: Seconds to wait between challenges
    """
    print("=" * 80)
    print("🤖 DRONE ACOUSTIC CHALLENGE BOT")
    print("=" * 80)
    
    # Load model
    try:
        model, labels = load_model_and_labels()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    print(f"\n🎯 Target API: {API_BASE_URL}")
    print(f"🔑 Using API Token: {API_TOKEN[:8]}...")
    print(f"📊 Ready to classify: {', '.join(labels)}")
    print("\n" + "=" * 80)
    
    iteration = 0
    total_score = 0
    correct_count = 0
    wrong_count = 0
    
    while max_iterations is None or iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*80}")
        print(f"🔄 Challenge #{iteration} - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*80}")
        
        try:
            # Step 1: Get current challenge
            print("📥 Fetching current challenge...")
            challenge = get_current_challenge()
            challenge_id = challenge['challenge_id']
            wav_url = challenge['wav_url']
            time_left = challenge.get('time_until_next_rotation_ms', 0)
            
            print(f"   Challenge ID: {challenge_id}")
            print(f"   Audio URL: {wav_url}")
            print(f"   Time until rotation: {time_left}ms")
            
            # Step 2: Download audio
            print("\n🎵 Downloading audio...")
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            download_audio(wav_url, tmp_path)
            print(f"   Saved to: {tmp_path}")
            
            # Step 3: Classify
            print("\n🔍 Classifying audio...")
            start_time = time.time()
            prediction, confidence, all_probs = classify_audio(model, tmp_path, labels)
            inference_time = time.time() - start_time
            
            print(f"   Prediction: {prediction.upper()}")
            print(f"   Confidence: {confidence:.2%}")
            print(f"   All probabilities:")
            for label, prob in zip(labels, all_probs):
                print(f"      {label:12s}: {prob:.2%}")
            print(f"   Inference time: {inference_time:.3f}s")
            
            # Step 4: Submit
            print(f"\n📤 Submitting classification: {prediction}")
            result = submit_classification(challenge_id, prediction)
            
            # Display result
            print("\n" + "="*80)
            if result.get('success'):
                print("✅ CORRECT!")
                correct_count += 1
                score_awarded = result.get('score_awarded', 0)
                total_score = result.get('total_score', total_score + score_awarded)
                print(f"   Score awarded: {score_awarded}")
                print(f"   Total score: {total_score}")
            else:
                print("❌ WRONG!")
                wrong_count += 1
                print(f"   Message: {result.get('message', 'No message')}")
            
            print(f"\n📊 Session Stats:")
            print(f"   Correct: {correct_count}")
            print(f"   Wrong: {wrong_count}")
            print(f"   Accuracy: {correct_count/(correct_count+wrong_count)*100:.1f}%")
            print(f"   Total Score: {total_score}")
            print("="*80)
            
            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)
            
            # Wait before next challenge
            if max_iterations is None or iteration < max_iterations:
                print(f"\n⏳ Waiting {delay_between_challenges}s before next challenge...")
                time.sleep(delay_between_challenges)
        
        except ChallengeBotAPIError as e:
            print(f"\n❌ API Error: {e}")
            print(f"   Retrying in {delay_between_challenges}s...")
            time.sleep(delay_between_challenges)
        
        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped by user")
            break
        
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print(f"   Retrying in {delay_between_challenges}s...")
            time.sleep(delay_between_challenges)
    
    print("\n" + "="*80)
    print("🏁 Challenge Bot Session Complete!")
    print(f"   Total Challenges: {correct_count + wrong_count}")
    print(f"   Correct: {correct_count}")
    print(f"   Wrong: {wrong_count}")
    if correct_count + wrong_count > 0:
        print(f"   Accuracy: {correct_count/(correct_count+wrong_count)*100:.1f}%")
    print(f"   Final Score: {total_score}")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated Drone Acoustic Challenge Bot")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum number of challenges to attempt (default: infinite)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between challenges in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help=f"Path to model file (default: {MODEL_PATH})"
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=LABELS_PATH,
        help=f"Path to labels file (default: {LABELS_PATH})"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=API_BASE_URL,
        help=f"API base URL (default: {API_BASE_URL})"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=API_TOKEN,
        help="API authentication token"
    )
    
    args = parser.parse_args()
    
    # Update globals with command line arguments
    MODEL_PATH = args.model
    LABELS_PATH = args.labels
    API_BASE_URL = args.api_url
    API_TOKEN = args.token
    
    # Run the bot
    run_challenge_bot(
        max_iterations=args.max_iterations,
        delay_between_challenges=args.delay
    )
