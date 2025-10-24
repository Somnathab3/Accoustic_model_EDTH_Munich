# Challenge Server Setup Guide

## 🚨 IMPORTANT: Server Not Running

Your challenge bot is configured correctly, but the challenge server at `http://127.0.0.1:8123` is not responding.

## ✅ Your Configuration (Already Set)
- **Username**: Ethos
- **Email**: somnathab3@gmail.com
- **Token**: `9726345a-34ed-4995-94d9-ecc239b47c1d` ✓
- **API URL**: `http://127.0.0.1:8123` ✓
- **Model**: `cnn_edth_3class.pt` (3 classes) ✓

## 🔍 What You Need

The challenge server should have been provided separately. Look for:

1. **Executable file** (e.g., `challenge-server.exe`, `server.exe`)
2. **Docker container** instructions
3. **Separate repository** with server code
4. **Download link** in the challenge email

## 🚀 How to Start the Server

### Option 1: If you have an executable
```powershell
# Navigate to where the server is located
cd path\to\server

# Run the server
.\challenge-server.exe
```

### Option 2: If you have Docker
```powershell
docker run -p 8123:8123 challenge-server
```

### Option 3: If you have Python server
```powershell
cd path\to\server
python server.py
# or
uvicorn main:app --host 127.0.0.1 --port 8123
```

## ✅ Verify Server is Running

Run this PowerShell command:
```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8123/api/challenge" | Select-Object StatusCode, Content
```

Or open in browser:
- API: http://127.0.0.1:8123/api/challenge
- Viewer: http://127.0.0.1:8123/static/index.html

## 🤖 Once Server is Running

Your bot is already configured! Just run:
```powershell
# Run the automated bot
python challenge_bot.py

# Or test single challenge first
python test_challenge.py
```

## 📊 Expected Server Response

When working correctly, you should see:
```json
{
  "wav_url": "/wavs/1e3cc7a3-b1af-4048-b143-bd8a7d264e2d.wav",
  "time_until_next_rotation_ms": 62000,
  "challenge_id": "5875f5cd-d860-48ff-acfb-06f6044a2067"
}
```

## 🆘 Troubleshooting

### Error: "Connection Refused" or "Unable to connect"
→ Server is not running. Start the server first.

### Error: "401 Unauthorized"
→ Token is wrong (but yours is already correct!)

### Error: "Already submitted"
→ You can only submit once per challenge. Wait for next rotation (100s).

## 📧 Need Help?

Check your challenge registration email for:
- Server download link
- Setup instructions
- Discord/Slack channel for support

## 🎯 Current Challenge Info (from your email)

- **Challenge ID**: `5875f5cd-d860-48ff-acfb-06f6044a2067`
- **WAV File**: `1e3cc7a3-b1af-4048-b143-bd8a7d264e2d.wav`
- **Time**: Rotates every 100 seconds
- **Classes**: `background`, `drone`, `helicopter`

---

**Your bot is ready to go - just start the server!** 🚀
