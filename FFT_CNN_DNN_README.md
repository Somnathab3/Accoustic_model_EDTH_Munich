# FFT + CNN + DNN Architecture for Drone Acoustic Classification

## Overview

This enhanced architecture uses a three-stage pipeline for improved drone acoustic classification:

1. **FFT Feature Extraction**: Extract frequency domain features from audio
2. **CNN Feature Learning**: Deep convolutional network learns high-level audio patterns
3. **DNN Classification**: Dense neural network makes final classification decision

## Architecture Benefits

- **Better Feature Representation**: FFT captures frequency domain characteristics essential for acoustic analysis
- **Hierarchical Learning**: CNN extracts spatial-temporal patterns, DNN refines for classification
- **Improved Accuracy**: Multi-stage processing allows for more nuanced decision-making
- **Adaptive Learning**: System tracks performance and stores results for continuous improvement

## Key Components

### 1. FFT Processor (`src/adrone/features/fft_processor.py`)

Extracts multiple frequency-domain features:
- Mel spectrograms
- MFCC (Mel-frequency cepstral coefficients)
- Spectral statistics (centroid, rolloff, bandwidth)
- Power distribution across frequency bands

```python
from adrone.features.fft_processor import FFTProcessor

fft_processor = FFTProcessor(
    n_fft=2048,
    hop_length=512,
    n_mels=128,
    sample_rate=16000
)

# Extract features from audio
audio_features = fft_processor.extract_features_for_model(audio_waveform)
```

### 2. FFT + CNN + DNN Model (`src/adrone/models/fft_cnn_dnn.py`)

Complete fusion model with:
- Residual connections for better gradient flow
- Channel attention mechanisms
- Batch normalization and dropout for regularization
- Flexible DNN classifier head

```python
from adrone.models.fft_cnn_dnn import FFTCNNDNNFusion

model = FFTCNNDNNFusion(
    n_classes=3,
    in_channels=1,
    cnn_feature_dim=512,
    dnn_hidden_dims=[256, 128]
)
```

### 3. Challenge Handler (`src/adrone/serve/challenge_handler.py`)

Manages challenge submissions and result storage:
- **ChallengeResultStorage**: Stores audio samples, predictions, and performance metrics
- **ChallengeAPIClient**: Handles API communication
- **AdaptiveLearningTracker**: Tracks performance patterns and provides recommendations

## Usage

### Running the Enhanced Challenge Bot

The new challenge bot uses the complete FFT + CNN + DNN pipeline:

```bash
python challenge_bot_fft_cnn_dnn.py --max-iterations 100 --delay 1.0
```

**Options:**
- `--max-iterations`: Number of challenges to attempt (default: infinite)
- `--delay`: Seconds between challenges (default: 1.0)
- `--model`: Path to model checkpoint
- `--labels`: Path to labels JSON
- `--storage-dir`: Directory for storing results (default: `challenge_results`)

### Features

1. **Automatic Result Storage**
   - All audio samples are saved with metadata
   - Predictions, confidences, and scores are logged
   - Performance statistics are continuously updated

2. **Adaptive Learning**
   - Tracks per-class performance
   - Identifies patterns in correct/incorrect predictions
   - Provides recommendations for improvement

3. **Performance Analytics**
   - Overall accuracy and score tracking
   - Recent performance trends
   - Class-wise prediction distribution
   - Confusion matrix (when actual labels are known)

### Accessing Stored Results

Results are stored in the `challenge_results/` directory:

```
challenge_results/
├── audio_samples/           # Stored audio files
│   └── <challenge_id>_<prediction>_<timestamp>.wav
├── results.jsonl           # Line-delimited JSON of all results
└── statistics.json         # Aggregated statistics
```

**View statistics:**
```python
from adrone.serve.challenge_handler import ChallengeResultStorage

storage = ChallengeResultStorage('challenge_results')
stats = storage.get_statistics()
print(f"Accuracy: {stats['accuracy']*100:.1f}%")
print(f"Total Score: {stats['total_score']}")
```

**Export data for retraining:**
```python
storage.export_for_training('data/challenge_samples')
# Creates directory structure:
# data/challenge_samples/
# ├── drone/
# ├── bird/
# └── background/
```

### Training the FFT + CNN + DNN Model

```bash
python scripts/train_fft_cnn_dnn.py \
    --train-dir data/edth_prepared/train \
    --val-dir data/edth_prepared/val \
    --output-dir models \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --model-type fusion
```

**Model Types:**
- `fusion`: FFTCNNDNNFusion (recommended)
- `multiscale`: MultiScaleCNNDNN (processes multiple temporal scales)

## Challenge API Integration

### Scoring System

- **Base Score**: 100 points for correct answer
- **Speed Bonus**: Up to 100 extra points for fast response
- **Penalty**: 0 points for wrong answer
- **Limit**: One submission per challenge

### API Endpoints

**Get Current Challenge:**
```bash
curl http://127.0.0.1:8123/api/challenge
```

**Submit Classification:**
```bash
curl -X POST http://127.0.0.1:8123/api/challenge \
  -H "Authorization: Bearer 9726345a-34ed-4995-94d9-ecc239b47c1d" \
  -H "Content-Type: application/json" \
  -d '{
    "challenge_id": "123e4567-e89b-12d3-a456-426614174000",
    "classification": "drone"
  }'
```

**Response:**
```json
{
  "success": true,
  "message": "Correct!",
  "score_awarded": 150,
  "total_score": 450
}
```

## Performance Improvement Strategy

### Current Issue

If showing 6 correct with score of 201:
- Average score per correct: ~33.5 points
- Expected: 100+ points per correct
- **Diagnosis**: Low scores indicate either:
  - Slow response times (missing speed bonus)
  - Some predictions might be getting 0 points (wrong)

### Solutions

1. **Improve Model Accuracy**
   ```bash
   # Train with more data
   python scripts/train_fft_cnn_dnn.py --epochs 100 --batch-size 64
   ```

2. **Analyze Stored Results**
   ```python
   storage = ChallengeResultStorage('challenge_results')
   analysis = storage.analyze_performance()
   print(analysis)
   ```

3. **Optimize Inference Speed**
   - Use GPU if available
   - Reduce FFT feature extraction time
   - Cache model in memory

4. **Fine-tune on Challenge Data**
   ```bash
   # Export challenge samples
   python -c "
   from adrone.serve.challenge_handler import ChallengeResultStorage
   storage = ChallengeResultStorage('challenge_results')
   storage.export_for_training('data/challenge_finetune')
   "
   
   # Retrain on challenge distribution
   python scripts/train_fft_cnn_dnn.py \
       --train-dir data/challenge_finetune \
       --val-dir data/edth_prepared/val \
       --epochs 20 \
       --lr 0.0001
   ```

## Dependencies

Add to `requirements.txt`:
```
torch>=1.9.0
librosa>=0.9.0
numpy>=1.21.0
scipy>=1.7.0
opencv-python>=4.5.0  # For FFT processor
requests>=2.26.0
tqdm>=4.62.0
scikit-learn>=0.24.0
```

Install:
```bash
pip install -r requirements.txt
```

## Model Files

After training, you'll have:
- `models/best_model.pt` - Best model checkpoint
- `models/best_model_metadata.json` - Model metadata
- `models/classification_report_best.txt` - Performance report
- `models/training_history.json` - Training curves

## Next Steps

1. **Run the enhanced challenge bot**:
   ```bash
   python challenge_bot_fft_cnn_dnn.py --max-iterations 20
   ```

2. **Monitor performance**:
   - Check `challenge_results/statistics.json`
   - Review stored audio samples
   - Analyze patterns in predictions

3. **Iterate and improve**:
   - Use stored samples for fine-tuning
   - Adjust model architecture based on performance
   - Optimize inference pipeline for speed bonus

## Troubleshooting

### Low Scores Despite High Accuracy

**Issue**: Getting correct predictions but low scores

**Solutions**:
- Check response time - implement caching and optimization
- Verify FFT parameters match training configuration
- Ensure model is using GPU for faster inference

### Import Errors

**Issue**: Cannot import `adrone` modules

**Solution**:
```bash
# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or use the bot which handles this automatically
python challenge_bot_fft_cnn_dnn.py
```

### Model Not Found

**Issue**: Model checkpoint doesn't exist

**Solution**:
```bash
# Train a new model first
python scripts/train_fft_cnn_dnn.py \
    --train-dir data/edth_prepared/train \
    --val-dir data/edth_prepared/val \
    --output-dir models
```

## Architecture Diagram

```
                    Input Audio (WAV)
                          ↓
              [FFT Preprocessing Layer]
                Mel Spectrogram
              (1, 128, 63) tensor
                          ↓
        ┌─────────────────┴─────────────────┐
        ↓                                   ↓
[FFT Feature Extractor]          [CNN Feature Extractor]
  Statistical Analysis             Deep Learning
  • Mean, Std, Max, Min            • Conv2D Layers
  • Spectral Features              • Residual Blocks
  • Temporal Patterns              • Channel Attention
        ↓                                   ↓
  FFT Features (256-dim)           CNN Features (512-dim)
        ↓                                   ↓
        └─────────────────┬─────────────────┘
                          ↓
                [Feature Fusion]
              Concatenate Features
                (768-dim vector)
                          ↓
                [DNN Classifier]
              • Dense(768→256)
              • Dense(256→128)
              • Dense(128→3)
                          ↓
                Class Probabilities
                          ↓
              Prediction + Confidence
```

### Key Innovation: Parallel Dual-Path Processing

**Why Parallel Architecture?**

1. **FFT Path (Handcrafted Features)**:
   - Extracts proven acoustic features
   - Captures statistical properties
   - Fast and interpretable
   - Good for frequency-domain patterns

2. **CNN Path (Learned Features)**:
   - Learns optimal representations from data
   - Captures complex spatial-temporal patterns
   - Attention mechanisms focus on important regions
   - Good for subtle acoustic signatures

3. **Fusion (Best of Both Worlds)**:
   - Combines complementary information
   - FFT provides domain knowledge
   - CNN provides data-driven insights
   - DNN makes informed final decision

### Information Flow

```
Step 1: Audio → FFT → Mel Spectrogram (shared input)
        ↓
Step 2a: Mel Spec → FFT Extractor → Statistical Features (256-dim)
Step 2b: Mel Spec → CNN Extractor → Learned Features (512-dim)
        ↓
Step 3: [FFT Features | CNN Features] → Concatenated (768-dim)
        ↓
Step 4: Fused Features → DNN → Classification → Output
```

## Detailed Architecture Insights

### Why This Architecture Works

#### 1. **Parallel Processing = Complementary Features**

Traditional approaches use either handcrafted features (FFT) OR learned features (CNN). Our architecture uses BOTH:

- **FFT Path**: Fast, interpretable, proven acoustic features
  - Computes statistical summaries of frequency content
  - Mean, std, min, max across time and frequency
  - Captures overall spectral characteristics
  - ~256 dimensional feature vector

- **CNN Path**: Deep learning discovers optimal patterns
  - Learns hierarchical representations automatically  
  - Residual connections enable very deep networks
  - Attention mechanisms focus on discriminative regions
  - ~512 dimensional feature vector

#### 2. **Feature Fusion = Better Decisions**

By concatenating both feature types, the DNN classifier has access to:
- Domain knowledge (from FFT statistics)
- Data-driven patterns (from CNN learning)
- Complementary information that neither path alone can provide

```python
# Fusion happens here:
fft_features = self.fft_extractor(mel_spec)  # (B, 256)
cnn_features = self.cnn(mel_spec)            # (B, 512)
fused = torch.cat([fft_features, cnn_features], dim=1)  # (B, 768)
```

#### 3. **Hierarchical Representation Learning**

**FFT Extractor** (Statistical Path):
```
Mel Spectrogram (1, 128, 63)
    ↓ Statistical Analysis
50 Features (mean, std, max, min, etc.)
    ↓ FC(50→128)
128-dim intermediate
    ↓ FC(128→256)
256-dim FFT Features
```

**CNN Extractor** (Learning Path):
```
Mel Spectrogram (1, 128, 63)
    ↓ Conv2D + MaxPool
(32, 64, 31) - Basic patterns
    ↓ ResBlock1 + Attention
(64, 32, 15) - Low-level features
    ↓ ResBlock2 + Attention
(128, 16, 7) - Mid-level features
    ↓ ResBlock3 + Attention
(256, 8, 3) - High-level features
    ↓ Global Pooling + FC
512-dim CNN Features
```

**DNN Classifier** (Fusion Path):
```
Fused Features (768)
    ↓ FC(768→256) + BatchNorm + Dropout
256-dim abstract features
    ↓ FC(256→128) + BatchNorm + Dropout
128-dim refined features
    ↓ FC(128→3)
3 class logits
```

### Tensor Shape Transformations

| Stage | Operation | Input Shape | Output Shape | Parameters |
|-------|-----------|-------------|--------------|------------|
| Input | Audio WAV | (32000,) | → | 0 |
| FFT | Mel Spec | (32000,) | (1, 128, 63) | 0 |
| **Path A** | **FFT Extractor** | | | |
| └─ Stats | Extract | (1, 128, 63) | (50,) | 0 |
| └─ FC1 | Dense | (50,) | (128,) | 6,528 |
| └─ FC2 | Dense | (128,) | (256,) | 33,024 |
| **Path B** | **CNN Extractor** | | | |
| └─ Conv1 | Conv+Pool | (1, 128, 63) | (32, 64, 31) | 320 |
| └─ ResBlock1 | +Attention | (32, 64, 31) | (64, 32, 15) | ~75K |
| └─ ResBlock2 | +Attention | (64, 32, 15) | (128, 16, 7) | ~300K |
| └─ ResBlock3 | +Attention | (128, 16, 7) | (256, 8, 3) | ~1.2M |
| └─ Pool+FC | Project | (256,) | (512,) | 131,584 |
| **Fusion** | Concatenate | (256,)+(512,) | (768,) | 0 |
| **DNN** | **Classifier** | | | |
| └─ Dense1 | +BN+Drop | (768,) | (256,) | 196,864 |
| └─ Dense2 | +BN+Drop | (256,) | (128,) | 32,896 |
| └─ Output | Dense | (128,) | (3,) | 387 |
| **Total** | | | | **~2.0M params** |

### Receptive Field Analysis

The CNN path builds increasingly large receptive fields:

| Layer | Receptive Field | Feature Map Size | What It Sees |
|-------|----------------|------------------|--------------|
| Input | 1 bin | 128 bins | Single frequency |
| Conv1 | 3 bins | 64 bins | Local frequency patterns |
| ResBlock1 | ~14 bins | 32 bins | Harmonic structures |
| ResBlock2 | ~30 bins | 16 bins | Spectral envelopes |
| ResBlock3 | ~62 bins | 8 bins | Broad frequency patterns |
| Global Pool | 128 bins | 1 bin | Full spectrum |

**Insight**: By the final layer, each feature "sees" the entire frequency range, enabling detection of complex acoustic signatures that span multiple octaves.

### Attention Mechanism Visualization

Channel attention learns to focus on important frequency bands:

```
Example for Drone Detection:
High Attention:  ████████ (200-400 Hz) - Propeller fundamental
Medium Attention: ███ (800-1600 Hz) - Harmonics
Low Attention:    █ (4000+ Hz) - Background noise
```

The network automatically learns which frequency bands matter most for each class!

## Contact & Support

For issues or questions about the FFT + CNN + DNN architecture, please refer to:
- Main README: `README.md`
- Project status: `PROJECT_STATUS.md`
- Challenge bot documentation: `CHALLENGE_BOT_README.md`
