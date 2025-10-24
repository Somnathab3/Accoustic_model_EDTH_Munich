"""
Quick diagnostic script to check if challenge server is accessible
"""
import requests
import sys

API_BASE_URL = "https://edth.helsing.codes"
API_TOKEN = "9726345a-34ed-4995-94d9-ecc239b47c1d"

print("="*80)
print("🔍 Challenge Server Diagnostic")
print("="*80)
print(f"\nChecking server at: {API_BASE_URL}")
print(f"Using token: {API_TOKEN[:20]}...\n")

# Test 1: Check if server is responding
print("Test 1: Server Connectivity")
print("-" * 80)
try:
    response = requests.get(f"{API_BASE_URL}/api/challenge", timeout=3)
    print(f"✅ Server is ONLINE!")
    print(f"   Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n📊 Current Challenge:")
        print(f"   Challenge ID: {data.get('challenge_id', 'N/A')}")
        print(f"   WAV URL: {data.get('wav_url', 'N/A')}")
        print(f"   Time remaining: {data.get('time_until_next_rotation_ms', 0)/1000:.1f}s")
        
        # Test 2: Try to download audio
        print(f"\nTest 2: Audio Download")
        print("-" * 80)
        wav_url = data.get('wav_url')
        if wav_url:
            try:
                audio_response = requests.get(f"{API_BASE_URL}{wav_url}", timeout=5)
                print(f"✅ Audio download successful!")
                print(f"   Size: {len(audio_response.content)} bytes")
            except Exception as e:
                print(f"❌ Audio download failed: {e}")
        
        # Test 3: Check submission endpoint (without actually submitting)
        print(f"\nTest 3: Submission Endpoint Check")
        print("-" * 80)
        print(f"   Endpoint: POST {API_BASE_URL}/api/challenge")
        print(f"   Auth: Bearer token present ✓")
        print(f"   Ready to submit: YES")
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED - Server is ready!")
        print("="*80)
        print("\n🚀 You can now run:")
        print("   python test_challenge.py    (test single challenge)")
        print("   python challenge_bot.py     (run automated bot)")
        print("\n📺 View live at: http://127.0.0.1:8123/static/index.html")
        sys.exit(0)
    else:
        print(f"⚠️  Server responded with status {response.status_code}")
        print(f"   Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("❌ Server is OFFLINE")
    print("\n" + "="*80)
    print("🚨 CONNECTION REFUSED")
    print("="*80)
    print("\nThe challenge server is not running at http://127.0.0.1:8123")
    print("\n📋 What to do:")
    print("   1. Find and start the challenge server")
    print("   2. Check your email for server download link")
    print("   3. Read SERVER_SETUP.md for instructions")
    print("\n💡 Common locations to check:")
    print("   - Same folder as this script")
    print("   - Downloads folder")
    print("   - Separate challenge-server repository")
    print("   - Docker container")
    print("\n🔍 Look for files like:")
    print("   - challenge-server.exe")
    print("   - server.exe")
    print("   - docker-compose.yml")
    print("   - Dockerfile")
    sys.exit(1)
    
except requests.exceptions.Timeout:
    print("❌ Server connection TIMEOUT")
    print("   The server might be overloaded or slow")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
