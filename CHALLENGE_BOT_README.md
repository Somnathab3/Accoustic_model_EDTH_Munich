# Challenge Bot - Automated Acoustic Classification

Automated bot to participate in the real-time drone acoustic classification challenge using the trained `cnn_edth_3class` model.

## 🚀 Quick Start

### Test Single Challenge (Manual)
```powershell
python test_challenge.py
```

This will:
1. Fetch one challenge
2. Download and classify the audio
3. Show predictions with confidence scores
4. Ask for confirmation before submitting

### Run Automated Bot
```powershell
# Run indefinitely
python challenge_bot.py

# Run for 10 challenges
python challenge_bot.py --max-iterations 10

# Custom delay between challenges (2 seconds)
python challenge_bot.py --delay 2.0
```

## 📋 Requirements

Install required packages:
```powershell
pip install requests torch librosa numpy
```

## ⚙️ Configuration

The scripts use these default settings:

| Setting | Value |
|---------|-------|
| API URL | `http://127.0.0.1:8123` |
| API Token | `9726345a-34ed-4995-94d9-ecc239b47c1d` |
| Model Path | `models/cnn_edth_3class.pt` |
| Labels Path | `models/labels_edth_3class.json` |

### Custom Configuration

```powershell
python challenge_bot.py `
  --api-url "http://different-host:8123" `
  --token "your-token-here" `
  --model "path/to/model.pt" `
  --labels "path/to/labels.json"
```

## 🎯 How It Works

1. **Fetch Challenge**: GET `/api/challenge` to get current audio URL and challenge ID
2. **Download Audio**: Retrieve WAV file from server
3. **Preprocess**: Convert to mel spectrogram (128 mels, 16kHz, 5s max)
4. **Classify**: Run through CNN model to predict: `background`, `drone`, or `helicopter`
5. **Submit**: POST classification with challenge ID and auth token
6. **Score**: Receive feedback and points

## 🏆 Bot Features

- ✅ Automatic challenge fetching and submission
- ✅ Real-time classification using trained model
- ✅ Confidence scores for predictions
- ✅ Session statistics (accuracy, total score)
- ✅ Error handling and retry logic
- ✅ Configurable iteration limit and delay
- ✅ Clean terminal output with emojis

## 📊 Output Example

```
🤖 DRONE ACOUSTIC CHALLENGE BOT
================================================================================
Loading model from models/cnn_edth_3class.pt...
Loading labels from models/labels_edth_3class.json...
Model loaded successfully! Classes: ['background', 'drone', 'helicopter']

🎯 Target API: http://127.0.0.1:8123
🔑 Using API Token: 9726345a...
📊 Ready to classify: background, drone, helicopter

================================================================================
🔄 Challenge #1 - 19:30:45
================================================================================
📥 Fetching current challenge...
   Challenge ID: 123e4567-e89b-12d3-a456-426614174000
   Audio URL: /wavs/550e8400-e29b-41d4-a716-446655440000.wav
   Time until rotation: 7500ms

🎵 Downloading audio...
   Saved to: C:\Users\...\tmp_xyz.wav

🔍 Classifying audio...
   Prediction: DRONE
   Confidence: 94.23%
   All probabilities:
      background  : 2.45%
      drone       : 94.23%
      helicopter  : 3.32%
   Inference time: 0.234s

📤 Submitting classification: drone

================================================================================
✅ CORRECT!
   Score awarded: 150
   Total score: 150

📊 Session Stats:
   Correct: 1
   Wrong: 0
   Accuracy: 100.0%
   Total Score: 150
================================================================================
```

## 🎮 Command Line Options

### challenge_bot.py

| Option | Description | Default |
|--------|-------------|---------|
| `--max-iterations` | Max challenges to attempt | Infinite |
| `--delay` | Seconds between challenges | 1.0 |
| `--model` | Path to model file | `models/cnn_edth_3class.pt` |
| `--labels` | Path to labels file | `models/labels_edth_3class.json` |
| `--api-url` | API base URL | `http://127.0.0.1:8123` |
| `--token` | API auth token | Default token |

### Examples

```powershell
# Run 5 challenges with 3-second delay
python challenge_bot.py --max-iterations 5 --delay 3

# Use different API endpoint
python challenge_bot.py --api-url "http://192.168.1.100:8123"

# Use custom model
python challenge_bot.py --model "models/my_custom_model.pt"
```

## 🔧 Troubleshooting

### Connection Refused
- Ensure the challenge server is running at `http://127.0.0.1:8123`
- Check if you can access `http://127.0.0.1:8123/static/index.html`

### Wrong Predictions
- Verify model and labels are correctly loaded
- Check if audio preprocessing matches training config
- Review audio quality and duration

### API Errors
- Confirm API token is valid
- Check challenge hasn't expired (time_until_next_rotation)
- Ensure only one submission per challenge

## 📈 Model Information

**Model**: `cnn_edth_3class.pt`
- Architecture: CNN for acoustic classification
- Input: Mel spectrogram (128 mels)
- Classes: 3 (background, drone, helicopter)
- Trained: October 24, 2025

**Audio Processing**:
- Sample Rate: 16 kHz
- Max Duration: 5 seconds
- FFT Size: 2048
- Hop Length: 512
- Mel Bins: 128

## 🎯 Tips for Maximum Score

1. **Speed Matters**: Submit quickly for speed bonus (up to 100 points)
2. **Accuracy First**: Only submit if confidence > 70%
3. **Monitor Live**: Watch at `http://127.0.0.1:8123/static/index.html`
4. **Optimize Delay**: Balance between speed and server load
5. **Check Stats**: Monitor accuracy to tune confidence thresholds

## 🔗 API Endpoints

- `GET /api/challenge` - Get current challenge
- `POST /api/challenge` - Submit classification
- `GET /wavs/{uuid}.wav` - Download audio file
- `WS /ws` - WebSocket for live updates
- `GET /static/index.html` - Live viewer

## 📝 License

Part of the EDTH Munich Drone Acoustics project.
