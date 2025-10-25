"""
Test API Response to see all available fields
This will help us understand what score-related data is available
"""
import requests
import json
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_BASE_URL = os.getenv('API_BASE_URL', 'https://edth.helsing.codes')
API_TOKEN = os.getenv('API_TOKEN')

if API_TOKEN is None:
    raise ValueError(
        "API_TOKEN not found! Please set it in .env file.\n"
        "Create a .env file with: API_TOKEN=your_token_here"
    )

def test_api_responses():
    """Test API responses to see all available fields"""
    
    session = requests.Session()
    session.headers.update({
        'Authorization': f'Bearer {API_TOKEN}',
        'Content-Type': 'application/json'
    })
    
    print("="*60)
    print("TESTING API RESPONSES")
    print("="*60)
    
    # 1. Test GET challenge
    print("\n1. GET /api/challenge")
    print("-" * 40)
    try:
        response = session.get(f"{API_BASE_URL}/api/challenge", timeout=15)
        response.raise_for_status()
        challenge_data = response.json()
        
        print("Status Code:", response.status_code)
        print("Response JSON:")
        print(json.dumps(challenge_data, indent=2))
        
        # Extract info
        challenge_id = challenge_data.get('challenge_id')
        wav_url = challenge_data.get('wav_url')
        time_until_next = challenge_data.get('time_until_next_rotation_ms')
        
        print(f"\nExtracted fields:")
        print(f"  challenge_id: {challenge_id}")
        print(f"  wav_url: {wav_url}")
        print(f"  time_until_next_rotation_ms: {time_until_next}")
        
        # Check for any score-related fields
        score_fields = [k for k in challenge_data.keys() if 'score' in k.lower()]
        if score_fields:
            print(f"  Score-related fields found: {score_fields}")
        else:
            print("  No score-related fields in GET response")
        
        # 2. Download audio
        print(f"\n2. Downloading audio from: {wav_url}")
        print("-" * 40)
        full_url = f"{API_BASE_URL}{wav_url}"
        audio_response = session.get(full_url, timeout=20)
        audio_response.raise_for_status()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(audio_response.content)
        
        print(f"✓ Audio downloaded: {len(audio_response.content)} bytes")
        
        # 3. Submit a classification (test with "background" as it's common)
        print(f"\n3. POST /api/challenge (submitting classification)")
        print("-" * 40)
        
        payload = {
            'challenge_id': challenge_id,
            'classification': 'background'  # Test classification
        }
        
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        submit_response = session.post(
            f"{API_BASE_URL}/api/challenge",
            json=payload,
            timeout=30
        )
        submit_response.raise_for_status()
        submit_data = submit_response.json()
        
        print(f"\nStatus Code: {submit_response.status_code}")
        print("Response JSON:")
        print(json.dumps(submit_data, indent=2))
        
        # Analyze all fields
        print(f"\n✓ All Response Fields:")
        for key, value in submit_data.items():
            print(f"  {key}: {value} (type: {type(value).__name__})")
        
        # Check specifically for score fields
        score_fields = [k for k in submit_data.keys() if 'score' in k.lower()]
        print(f"\n✓ Score-related fields: {score_fields if score_fields else 'NONE FOUND'}")
        
        # Check for common response fields
        common_fields = ['correct', 'actual_classification', 'score_awarded', 
                        'points', 'total_score', 'current_score', 'reward']
        print(f"\n✓ Common field check:")
        for field in common_fields:
            if field in submit_data:
                print(f"  ✓ {field}: {submit_data[field]}")
            else:
                print(f"  ✗ {field}: NOT FOUND")
        
        # Clean up
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        print("\n" + "="*60)
        print("TEST COMPLETE")
        print("="*60)
        
        return submit_data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request Error: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response Text: {e.response.text}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    result = test_api_responses()
    
    if result:
        print(f"\n📊 SUMMARY:")
        print(f"  Total fields in response: {len(result)}")
        print(f"  Field names: {list(result.keys())}")
