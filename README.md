# 🎯 SOTA Acoustic Drone Detector - Complete Guide

## 🚀 State-of-the-Art Model for Real-Time Drone Detection

This repository contains a production-ready SOTA (State-of-the-Art) acoustic drone detection system with advanced preprocessing, multiple architectures, and optimized inference pipeline.

### ⚡ **Ultra-Fast Inference**: 0.2-0.3s latency | **High Accuracy**: 86.67% | **GPU-Accelerated**

---

## 📋 Table of Contents

1. [System Architecture & Process Flow](#system-architecture--process-flow)
2. [SOTA Process Step-by-Step](#sota-process-step-by-step)
3. [Quick Start](#quick-start)
4. [Model Architecture Details](#model-architecture-details)
5. [Challenge Bot Operation](#challenge-bot-operation)
6. [Training Pipeline](#training-pipeline)
7. [Performance & Results](#performance--results)

---

---

## 📊 System Architecture & Process Flow

### 🎯 Overall System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ACOUSTIC DRONE DETECTION SYSTEM                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
            ┌───────▼────────┐              ┌──────▼──────┐
            │  TRAINING      │              │  INFERENCE  │
            │  PIPELINE      │              │  PIPELINE   │
            └───────┬────────┘              └──────┬──────┘
                    │                               │
        ┌───────────┴──────────┐         ┌─────────┴─────────┐
        │                      │         │                   │
   ┌────▼─────┐         ┌─────▼────┐   ┌▼──────────┐  ┌────▼─────┐
   │   DATA   │         │  MODEL   │   │ REAL-TIME │  │ CHALLENGE│
   │PREPARATION│         │ TRAINING │   │CLASSIFIER │  │   BOT    │
   └──────────┘         └──────────┘   └───────────┘  └──────────┘
```

### 🔄 End-to-End Process Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                          TRAINING PHASE                              │
└─────────────────────────────────────────────────────────────────────┘

    1. RAW AUDIO               2. PREPROCESSING           3. AUGMENTATION
┌─────────────────┐        ┌──────────────────┐     ┌─────────────────┐
│ .wav files      │        │ • Resample 16kHz │     │ • SpecAugment   │
│ (44.1kHz)       │───────▶│ • HPSS (3-chan)  │────▶│ • Noise mixing  │
│ Variable length │        │ • Log-Mel (96)   │     │ • Time/Pitch    │
└─────────────────┘        │ • 2s windows     │     │ • Mixup (α=0.2) │
                           └──────────────────┘     └─────────────────┘
                                                              │
    4. MODEL TRAINING      5. VALIDATION              6. CHECKPOINT
┌─────────────────┐        ┌──────────────────┐     ┌─────────────────┐
│ • AdamW optim   │        │ • Macro F1 metric│     │ • best_model.pt │
│ • Cosine LR     │◀───────│ • Early stopping │────▶│ • panns_final.pt│
│ • Class balance │        │ • Confusion mat  │     │ • labels.json   │
│ • Grad clipping │        └──────────────────┘     └─────────────────┘
└─────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                          INFERENCE PHASE                             │
└─────────────────────────────────────────────────────────────────────┘

    1. INPUT                   2. PREPROCESS           3. MODEL
┌─────────────────┐        ┌──────────────────┐     ┌─────────────────┐
│ Audio file      │        │ Load & resample  │     │ PANNs CNN14     │
│ (any format)    │───────▶│ HPSS transform   │────▶│ (GPU inference) │
│ (any length)    │        │ Log-Mel spectrg  │     │ ~0.05s          │
└─────────────────┘        │ Normalize        │     └────────┬────────┘
                           └──────────────────┘              │
    4. POST-PROCESS        5. OUTPUT                         │
┌─────────────────┐        ┌──────────────────┐             │
│ Softmax         │        │ Class: "drone"   │             │
│ Argmax          │◀───────│ Conf: 0.8234     │◀────────────┘
│ Threshold       │        │ All probs: {...} │
└─────────────────┘        └──────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                        CHALLENGE BOT FLOW                            │
└─────────────────────────────────────────────────────────────────────┘

    START                                                          END
      │                                                             │
      ▼                                                             │
┌─────────────┐     ┌──────────────┐     ┌───────────────┐       │
│ Initialize  │────▶│ Warm-up GPU  │────▶│ Start timing  │       │
│ - Model     │     │ - Dummy pass │     │ - Reset state │       │
│ - API       │     │ - Verify GPU │     └───────┬───────┘       │
└─────────────┘     └──────────────┘             │               │
                                                  │               │
                    ┌─────────────────────────────┘               │
                    │                                             │
                    ▼                                             │
    ┌───────────────────────────────────────────┐                │
    │         MAIN CHALLENGE LOOP               │                │
    │  ┌─────────────────────────────────────┐  │                │
    │  │ 1. Fetch Challenge (API)            │  │                │
    │  │    └─ GET /api/challenge            │  │                │
    │  │                                     │  │                │
    │  │ 2. Check Duplicate                  │  │                │
    │  │    └─ Compare challenge_id          │  │                │
    │  │       ├─ Same? → Wait & retry       │  │                │
    │  │       └─ New? → Continue            │  │                │
    │  │                                     │  │                │
    │  │ 3. Download Audio                   │  │                │
    │  │    └─ GET wav_url → temp file       │  │                │
    │  │                                     │  │                │
    │  │ 4. Classify (FAST!)                 │  │                │
    │  │    ├─ Preprocess                    │  │                │
    │  │    ├─ GPU inference                 │  │                │
    │  │    └─ Get prediction                │  │                │
    │  │                                     │  │                │
    │  │ 5. ⚡ SUBMIT IMMEDIATELY!           │  │                │
    │  │    └─ POST /api/challenge           │  │                │
    │  │       (NO DELAYS BEFORE THIS!)      │  │                │
    │  │                                     │  │                │
    │  │ 6. Process Result                   │  │                │
    │  │    ├─ Calculate timing              │  │                │
    │  │    ├─ Write CSV                     │  │                │
    │  │    ├─ Store JSON                    │  │                │
    │  │    └─ Print results                 │  │                │
    │  │                                     │  │                │
    │  │ 7. Timing Strategy                  │  │                │
    │  │    ├─ Score 100-120? → Re-sync      │  │                │
    │  │    ├─ Pre-sync mode: check 1s       │  │                │
    │  │    └─ Synced: wait 98s + rapid poll │  │                │
    │  └─────────────────────────────────────┘  │                │
    │                   │                        │                │
    │                   └────────────────────────┤                │
    └────────────────────────────────────────────┘                │
                                                                  │
              Max iterations reached OR Ctrl+C                    │
                                  │                               │
                                  ▼                               │
                    ┌──────────────────────────┐                 │
                    │ Final Summary & Analysis │────────────────▶┘
                    │ - Accuracy               │
                    │ - Total score            │
                    │ - Per-class performance  │
                    └──────────────────────────┘
```

---

## 🎓 SOTA Process Step-by-Step

### Phase 1: Data Preparation & Preprocessing

```
┌──────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PIPELINE                         │
└──────────────────────────────────────────────────────────────────┘

INPUT: Raw Audio (44.1 kHz, stereo/mono, variable length)
   │
   ▼
┌─────────────────────────────────────┐
│ STEP 1: Load & Resample             │
│ ────────────────────────────────    │
│ • Load with librosa                 │
│ • Convert to mono (if stereo)       │
│ • Resample to 16 kHz                │
│ • Purpose: Standardize input        │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ STEP 2: HPSS (Optional)             │
│ ────────────────────────────────    │
│ • Harmonic-Percussive Separation    │
│ • Creates 3 channels:               │
│   1. Harmonic (tonal sounds)        │
│   2. Percussive (transients)        │
│   3. Full spectrum                  │
│ • Purpose: Separate drone/heli      │
│   harmonics from background         │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ STEP 3: Windowing                   │
│ ────────────────────────────────    │
│ • Segment into 2.0s windows         │
│ • Stride: 50% overlap               │
│ • Padding: If < 2s, pad with zeros  │
│ • Purpose: Fixed-size inputs        │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ STEP 4: Log-Mel Spectrogram         │
│ ────────────────────────────────    │
│ Parameters:                         │
│ • n_fft: 1024                       │
│ • hop_length: 320                   │
│ • n_mels: 96                        │
│ • Output: (3, 96, 126) tensor       │
│   ├─ 3: channels (harmonic/perc)   │
│   ├─ 96: mel frequency bins         │
│   └─ 126: time frames               │
│ • Purpose: Time-frequency repr.     │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ STEP 5: Normalization               │
│ ────────────────────────────────    │
│ • Mean: 0.0                         │
│ • Std: 1.0                          │
│ • Per-channel normalization         │
│ • Purpose: Stable training          │
└────────────┬────────────────────────┘
             │
             ▼
OUTPUT: Tensor (3, 96, 126) ready for model
```

### Phase 2: Data Augmentation (Training Only)

```
┌──────────────────────────────────────────────────────────────────┐
│                    AUGMENTATION PIPELINE                          │
└──────────────────────────────────────────────────────────────────┘

INPUT: Spectrogram (3, 96, 126)
   │
   ├──────────────────┬──────────────────┬──────────────────┐
   │                  │                  │                  │
   ▼                  ▼                  ▼                  ▼
┌─────────┐    ┌──────────┐      ┌──────────┐      ┌──────────┐
│SpecAug  │    │  Noise   │      │Time/Pitch│      │  Mixup   │
│(80%)    │    │ Mixing   │      │ Shift    │      │ (α=0.2)  │
│         │    │ (SNR)    │      │ (±10%)   │      │          │
├─────────┤    ├──────────┤      ├──────────┤      ├──────────┤
│Time mask│    │Add       │      │Simulate  │      │Mix two   │
│2 bands  │    │background│      │Doppler   │      │samples   │
│27 frames│    │noise at  │      │effect    │      │λ blend   │
│         │    │SNR 5-20dB│      │          │      │          │
│Freq mask│    │          │      │          │      │          │
│2 bands  │    │          │      │          │      │          │
│8 mels   │    │          │      │          │      │          │
└────┬────┘    └─────┬────┘      └─────┬────┘      └─────┬────┘
     │               │                  │                  │
     └───────────────┴──────────────────┴──────────────────┘
                              │
                              ▼
              Augmented Spectrogram (3, 96, 126)
```

### Phase 3: Model Architecture (PANNs CNN14)

```
┌──────────────────────────────────────────────────────────────────┐
│                      PANNs CNN14 ARCHITECTURE                     │
└──────────────────────────────────────────────────────────────────┘

INPUT: (Batch, 3, 96, 126)
   │
   ▼
┌─────────────────────────────────────┐
│ Conv Block 1                        │
│ ─────────────────────────────────   │
│ Conv2d(3→64, 3x3) + BN + ReLU       │
│ Conv2d(64→64, 3x3) + BN + ReLU      │
│ MaxPool2d(2x2) → (64, 48, 63)       │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ Conv Block 2                        │
│ ─────────────────────────────────   │
│ Conv2d(64→128, 3x3) + BN + ReLU     │
│ Conv2d(128→128, 3x3) + BN + ReLU    │
│ MaxPool2d(2x2) → (128, 24, 31)      │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ Conv Block 3                        │
│ ─────────────────────────────────   │
│ Conv2d(128→256, 3x3) + BN + ReLU    │
│ Conv2d(256→256, 3x3) + BN + ReLU    │
│ MaxPool2d(2x2) → (256, 12, 15)      │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ Conv Block 4                        │
│ ─────────────────────────────────   │
│ Conv2d(256→512, 3x3) + BN + ReLU    │
│ Conv2d(512→512, 3x3) + BN + ReLU    │
│ MaxPool2d(2x2) → (512, 6, 7)        │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ Adaptive Pooling                    │
│ ─────────────────────────────────   │
│ AdaptiveAvgPool2d(1x1)              │
│ AdaptiveMaxPool2d(1x1)              │
│ Concatenate → (1024,)               │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ Classifier Head                     │
│ ─────────────────────────────────   │
│ Linear(1024→512) + ReLU + Dropout   │
│ Linear(512→3) [background/drone/    │
│                helicopter]          │
└────────────┬────────────────────────┘
             │
             ▼
OUTPUT: Logits (Batch, 3)
```

### Phase 4: Training Strategy

```
┌──────────────────────────────────────────────────────────────────┐
│                        TRAINING LOOP                              │
└──────────────────────────────────────────────────────────────────┘

INITIALIZATION
├─ Model: PANNs CNN14 (~5M parameters)
├─ Optimizer: AdamW (lr=0.0001, weight_decay=0.01)
├─ Scheduler: CosineAnnealingLR (T_max=50)
├─ Loss: CrossEntropy + Label Smoothing (ε=0.05)
└─ Class Weights: [1.0, 2.0, 1.5] (balance classes)

FOR each epoch (1 to 50):
  │
  ├─ TRAINING PHASE
  │  └─ For each batch:
  │     ├─ Forward pass
  │     ├─ Calculate loss
  │     ├─ Backward pass
  │     ├─ Gradient clipping (max_norm=1.0)
  │     └─ Optimizer step
  │
  ├─ VALIDATION PHASE
  │  └─ For each val batch:
  │     ├─ Forward pass (no grad)
  │     ├─ Calculate metrics
  │     └─ Accumulate results
  │
  ├─ METRICS CALCULATION
  │  ├─ Accuracy
  │  ├─ Macro F1 Score (main metric)
  │  ├─ Per-class precision/recall
  │  └─ Confusion matrix
  │
  ├─ CHECKPOINT SAVING
  │  └─ If val_f1 > best_f1:
  │     ├─ Save best_model.pt
  │     ├─ Update best_f1
  │     └─ Reset patience counter
  │
  └─ EARLY STOPPING CHECK
     └─ If no improvement for 10 epochs:
        └─ STOP TRAINING

FINAL
└─ Save panns_final.pt (best checkpoint)
```

### Phase 5: Inference Pipeline (Challenge Bot)

```
┌──────────────────────────────────────────────────────────────────┐
│                    REAL-TIME INFERENCE FLOW                       │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────┐
│ 1. INITIALIZATION (Once)            │
│ ─────────────────────────────────   │
│ • Load model checkpoint             │
│ • Move to GPU (if available)        │
│ • Set model.eval() mode             │
│ • Warm-up GPU with dummy tensor     │
│ • Initialize API client             │
│ Time: ~2-3 seconds                  │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 2. FETCH CHALLENGE (Per request)    │
│ ─────────────────────────────────   │
│ • GET /api/challenge                │
│ • Extract: challenge_id, wav_url    │
│ • Check for duplicates              │
│ Time: ~0.02s (network)              │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 3. DOWNLOAD AUDIO                   │
│ ─────────────────────────────────   │
│ • GET wav_url → temp file           │
│ • Save to disk                      │
│ Time: ~0.05s (network)              │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 4. PREPROCESS (Fast)                │
│ ─────────────────────────────────   │
│ • Load audio                        │
│ • Resample to 16kHz                 │
│ • HPSS (3 channels)                 │
│ • Log-Mel spectrogram               │
│ • Normalize                         │
│ Time: ~0.03s (CPU)                  │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 5. GPU INFERENCE (Ultra-fast!)      │
│ ─────────────────────────────────   │
│ • Move tensor to GPU                │
│ • Forward pass (no gradient)        │
│ • Softmax probabilities             │
│ • Argmax for prediction             │
│ Time: ~0.05s (GPU)                  │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 6. ⚡ SUBMIT RESULT (CRITICAL!)     │
│ ─────────────────────────────────   │
│ • POST /api/challenge               │
│ • Payload: {challenge_id, class}    │
│ • NO DELAYS BEFORE THIS!            │
│ Time: ~0.02s (network)              │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 7. POST-PROCESSING (After submit)   │
│ ─────────────────────────────────   │
│ • Calculate timing                  │
│ • Write to CSV                      │
│ • Store JSONL                       │
│ • Print results                     │
│ • Update statistics                 │
│ Time: ~0.01s (doesn't affect score) │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 8. TIMING STRATEGY                  │
│ ─────────────────────────────────   │
│ • Score 100-120? → Reset to 1s      │
│ • Pre-sync: Check every 1s          │
│ • Synced: Wait 98s + rapid poll     │
└─────────────────────────────────────┘

TOTAL TIME: ~0.20-0.30s per challenge
```

### Phase 6: Scoring & Timing Strategy

```
┌──────────────────────────────────────────────────────────────────┐
│                       SCORING MECHANISM                           │
└──────────────────────────────────────────────────────────────────┘

SCORE CALCULATION:
├─ Base Score (Correctness):
│  ├─ Correct prediction: 100 points
│  └─ Wrong prediction: 100 points (participation)
│  ├─ Correct prediction: 100 points
│  └─ Wrong prediction: 100 points (participation)
│
├─ Speed Bonus (Response Time):
│  ├─ < 0.5s: +50 points (EXCELLENT)
│  ├─ 0.5-1.0s: +20-40 points (GOOD)
│  ├─ 1.0-2.0s: +10-20 points (OK)
│  └─ > 2.0s: 0 bonus (SLOW)
│
└─ TOTAL SCORE: Base + Speed Bonus
   ├─ Perfect (fast): 150 points
   ├─ Perfect (slow): 100 points
   └─ Wrong: 100 points

TIMING STRATEGY:
┌──────────────────────────────────────┐
│ MODE 1: PRE-SYNC (Initial)           │
│ ──────────────────────────────────   │
│ • Check every 1 second               │
│ • Wait for first score > 0           │
│ • Learn challenge timing             │
│ • Exit: When synced                  │
└──────────────────────────────────────┘
         │
         │ First score received
         ▼
┌──────────────────────────────────────┐
│ MODE 2: SYNCED (Optimized)           │
│ ──────────────────────────────────   │
│ • Wait 98 seconds after last score   │
│ • Enter rapid polling (0.1s)         │
│ • Poll for 2 seconds window          │
│ • Catch new challenge early!         │
│ • Exit: Score 100-120 detected       │
└──────────────────────────────────────┘
         │
         │ Score 100-120 (mistimed)
         ▼
┌──────────────────────────────────────┐
│ MODE 3: RE-SYNC (Recovery)           │
│ ──────────────────────────────────   │
│ • Reset to MODE 1                    │
│ • Check every 1 second again         │
│ • Re-learn timing                    │
│ • Exit: Good score received          │
└──────────────────────────────────────┘
```

---

## ⚡ Quick Start

### Option 1: Kaggle (Recommended)

**Step 1: Setup Environment**
```bash
# Clone repository
!git clone https://github.com/Somnathab3/edth-acoustic-drone-detector.git
%cd edth-acoustic-drone-detector

# Install dependencies
!pip install -q -r requirements.txt

# Enable GPU (Settings → Accelerator → GPU T4 x2)
# Enable Internet (Settings → Internet → On)
# Enable GPU (Settings → Accelerator → GPU T4 x2)
# Enable Internet (Settings → Internet → On)
```

**Step 2: Verify GPU**
```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
# Expected: GPU Available: True, Device: Tesla T4
```

**Step 3: Run Challenge Bot**
```bash
# Auto-detect model and start (recommended)
!python sota_challenge_bot.py --delay 0.5

# Or run overnight (max iterations)
!python sota_challenge_bot.py --max-iterations 10000 --delay 0.5
```

**Step 4: Monitor Results**
```python
# View real-time results
!tail -f challenge_results/results.csv

# Analyze performance
!python analyze_results.py
```

### Option 2: Local Machine

**Prerequisites:**
- Python 3.8+
- CUDA-capable GPU (recommended)
- 4GB+ RAM

**Installation:**
```bash
git clone https://github.com/Somnathab3/edth-acoustic-drone-detector.git
cd edth-acoustic-drone-detector
pip install -r requirements.txt
```

**Run:**
```bash
# Verify GPU
python check_gpu.py

# Run challenge bot
python sota_challenge_bot.py --delay 0.5

# Analyze results
python analyze_results.py
```

---

---

## 🏗️ Model Architecture Details

### Architecture Comparison

### Architecture Comparison

```
┌─────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ Architecture│  Parameters  │ Inference    │  Accuracy    │   Use Case   │
│             │              │  Time (GPU)  │              │              │
├─────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ CRNN        │   ~1.5M      │   ~30ms      │    82-84%    │ Edge/Mobile  │
│ + Attention │              │   (FAST)     │   (Good)     │  Real-time   │
├─────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ PANNs CNN14 │   ~5M        │   ~50ms      │    86-87%    │ Server/Cloud │
│ ⭐ DEFAULT  │  (Balanced)  │  (OPTIMAL)   │  (Excellent) │ ⭐ RECOMMENDED│
├─────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Audio       │   ~20M       │   ~150ms     │    88-90%    │ Offline      │
│ Transformer │   (Large)    │   (SLOW)     │  (Best)      │  Analysis    │
└─────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

### Detailed Architecture Breakdown

#### 1. **CRNN with Attention** (Fast & Lightweight)
- Fast inference (~50ms)
- Good for edge deployment
- 3 conv blocks + BiGRU + Temporal-Frequency attention

#### 2. **PANNs CNN14** (~5M params) ⭐ **Recommended**
- Balanced speed/accuracy
- AudioSet-inspired architecture
- 4 conv blocks with adaptive pooling

#### 3. **Audio Transformer** (~20M params)
- Highest accuracy
- Patch embedding + 12 transformer blocks
- Slower but most powerful

### Current Performance
- **Validation Accuracy**: 86.67%
- **Macro F1**: 0.8640
- **Classes**: background (100%), drone (63%), helicopter (97%)

## 🎯 Advanced Features

### Preprocessing Pipeline
- **Log-Mel Spectrograms**: 96 mels, 16kHz, 2.0s windows
- **HPSS**: Harmonic-Percussive Source Separation (3-channel input)
- **SpecAugment**: Time/frequency masking
- **SNR Curriculum**: Progressive noise degradation

### Data Augmentation
- Background noise mixing with SNR control
- Time/pitch augmentation (Doppler simulation)
- Mixup augmentation (α=0.2)
- Dynamic augmentation pipeline

### Training Features
- **AdamW optimizer** with cosine LR schedule
- **Label smoothing** (ε=0.05)
- **Class-balanced loss** with inverse frequency weighting
- **Early stopping** (patience=10, Macro-F1 metric)
- **Gradient clipping** (norm=1.0)

## 📁 Essential Files

### Core Scripts
```
├── sota_challenge_bot.py          # Main challenge bot ⭐
├── train_sota_model.py            # Training script
├── sota_inference.py              # Inference module
├── validate_model.py              # Model validation
├── analyze_results.py             # Results analyzer
└── quick_train.py                 # Quick training wrapper
```

### Source Code
```
src/adrone/
├── preprocessing/
│   └── audio_transforms.py        # Advanced preprocessing
├── models/
│   └── acoustic_models.py         # Three model architectures
├── data/
│   └── acoustic_dataset.py        # Dataset loader
├── training/
│   └── losses.py                  # Loss functions & schedulers
├── evaluation/
│   └── metrics.py                 # Evaluation metrics
└── serve/
    └── challenge_handler.py       # API client
```

### Model Files
```
models/
├── panns_final.pt                 # Fully trained (preferred)
├── best_model.pt                  # Training checkpoint
├── labels_current.json            # Class mappings ⭐
└── config.json                    # Training config
```

## 🎮 Usage Examples

### Example 1: Run Challenge Bot (Simple)
```bash
!python sota_challenge_bot.py
```
- Auto-detects best model
- Outputs to `challenge_results/results.csv`
- Smart timing with duplicate detection

### Example 2: Custom Configuration
```bash
!python sota_challenge_bot.py \
    --model models/panns_final.pt \
    --labels models/labels_current.json \
    --delay 0.5 \
    --max-iterations 1000 \
    --csv challenge_results/my_results.csv
```

### Example 3: Quick Inference
```python
from sota_inference import AcousticDroneClassifier

# Initialize
classifier = AcousticDroneClassifier(
    model_path='models/panns_final.pt',
    labels_path='models/labels_current.json'
)

# Classify
prediction, confidence, probs = classifier.classify('audio.wav')
print(f"{prediction}: {confidence:.2%}")
# Output: drone: 82.34%
```

### Example 4: Batch Processing
```python
import glob
from sota_inference import AcousticDroneClassifier

classifier = AcousticDroneClassifier()
audio_files = glob.glob('samples/*.wav')

for audio_file in audio_files:
    pred, conf, probs = classifier.classify(audio_file)
    print(f"{audio_file}: {pred} ({conf:.2%})")
```

### Example 5: Validate Model
```bash
!python validate_model.py \
    --model models/panns_final.pt \
    --labels models/labels_current.json \
    --val-dir data/edth_munich_dataset/data/val
```

## 🔧 Kaggle-Specific Setup

### Enable GPU
```python
# Settings → Accelerator → GPU T4 x2 (recommended)
```

### Enable Internet
```python
# Settings → Internet → On (required for API access)
```

### Install Dependencies
```bash
!pip install -q torch torchaudio librosa scikit-learn soundfile
```

### Check GPU
```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

## 📊 Challenge Bot Features

### Smart Timing Strategy
1. **Pre-Sync Mode**: Checks every 3s until first score received
2. **Synced Mode**: Waits 100s cycle after score synchronization
3. **Duplicate Detection**: Handles "already submitted" automatically
4. **Adaptive Backoff**: Exponential backoff on errors

### Output Files
```
challenge_results/
├── results.csv              # Main CSV (continuously updated) ⭐
├── results.jsonl            # Detailed JSONL
├── statistics.json          # Performance stats
└── audio_samples/           # Downloaded samples
    ├── correct/
    └── incorrect/
```

### CSV Format
```csv
iteration,timestamp,challenge_id,predicted,actual,correct,confidence,score_awarded,inference_time,total_time
1,2025-10-24 22:44:17,2b05de25...,drone,drone,True,0.8234,150,0.0747,0.1970
2,2025-10-24 22:44:18,41d23e80...,background,drone,False,0.7546,100,0.0725,0.1575
```

### Real-Time Monitoring
```
[1] ✓ Predicted: drone      | Actual: drone      | Conf: 0.823 | Score: +150 | Time: 0.17s
🎯 First score received! Now synced with server timing...
[2] ✗ Predicted: background | Actual: drone      | Conf: 0.755 | Score: +100 | Time: 0.16s
     [background:0.755 | drone:0.146 | helicopter:0.099]
[3] ✓ Predicted: helicopter | Actual: helicopter | Conf: 0.943 | Score: +150 | Time: 0.15s
```

## 📈 Performance Analysis

### View Results
```bash
!python analyze_results.py
```

Output:
```
📊 OVERALL PERFORMANCE
  Correct:  84/100 (84.0%)
  Wrong:    16/100 (16.0%)
  Total Score: 12,400
  Avg Score/Challenge: 124.0

🎯 PREDICTION DISTRIBUTION
  drone       : 45 (45.0%)
  background  : 30 (30.0%)
  helicopter  : 25 (25.0%)

📈 PER-CLASS ACCURACY
  Class        Correct  Total    Accuracy
  background    30       30       100.0%
  drone         25       40        62.5%
  helicopter    29       30        96.7%

⏱️  TIMING STATISTICS
  Inference Time: 0.074s (mean)
  Total Time: 0.197s (mean)
```

## 🚀 Overnight Kaggle Run

### Setup for Long Run
```bash
# In Kaggle notebook with GPU enabled
!git clone https://github.com/Somnathab3/edth-acoustic-drone-detector.git
%cd edth-acoustic-drone-detector
!pip install -q -r requirements.txt

# Run overnight (Kaggle limit: 9-12 hours)
!python sota_challenge_bot.py --delay 0.5 --max-iterations 10000
```

### Save Results to Kaggle Output
```python
import shutil

# Copy results to Kaggle output (persists after session)
!mkdir -p /kaggle/working/final_results
!cp -r challenge_results/* /kaggle/working/final_results/

# Download results
from IPython.display import FileLink
FileLink('/kaggle/working/final_results/results.csv')
```

## 🎓 Training Your Own Model

### Quick Training
```bash
!python quick_train.py
```

### Custom Training
```bash
!python train_sota_model.py \
    --train-dir data/edth_munich_dataset/data/train \
    --val-dir data/edth_munich_dataset/data/val \
    --model-type panns \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.0001
```

### Architecture Options
- `crnn` - Fast, lightweight (1.5M params)
- `panns` - Balanced (5M params) ⭐ **Recommended**
- `transformer` - Powerful (20M params)

## 🔑 API Configuration

- **URL**: https://edth.helsing.codes
- **Token**: Embedded in scripts (or set as environment variable)

Custom token:
```bash
export EDTH_API_TOKEN="your-token-here"
```

## 📦 Dependencies

Core (automatically installed):
```
torch>=2.0.0
torchaudio>=2.0.0
librosa>=0.10.0
numpy>=1.24.0
scikit-learn>=1.3.0
soundfile>=0.12.0
```

## 🐛 Troubleshooting

### GPU Not Available
```python
!nvidia-smi  # Check GPU
import torch
print(torch.cuda.is_available())  # Should be True
```

### Import Errors
```bash
!pip install --upgrade torch torchaudio
```

### Model Not Found
```bash
!ls -la models/
# Should see: panns_final.pt or best_model.pt
```

### Low Accuracy
```bash
# Validate model first
!python validate_model.py --model models/panns_final.pt --labels models/labels_current.json --val-dir data/val
```

## 📊 Expected Performance

### Good Session (1000 challenges)
- Runtime: ~30-60 minutes
- Accuracy: 80-85%
- Total Score: 120,000-130,000
- Avg time: 0.2s per challenge

### Model Validation
- Overall: 86.67% accuracy
- Background: 100% (perfect)
- Helicopter: 96.67% (excellent)
- Drone: 63.33% (needs improvement)

## 🎯 Quick Commands Reference

```bash
# Run challenge bot (default)
!python sota_challenge_bot.py

# Run 100 challenges fast
!python sota_challenge_bot.py --max-iterations 100 --delay 0.3

# Analyze results
!python analyze_results.py

# Validate model
!python validate_model.py --model models/panns_final.pt --labels models/labels_current.json --val-dir data/val

# Train new model
!python quick_train.py
```

## 📝 File Sizes

- Repository (without models): ~2MB
- PANNs model: ~20MB
- Transformer model: ~80MB
- Training data: Not included (download separately)

## 🔗 Useful Links

- Challenge: https://edth.helsing.codes
- Repository: https://github.com/Somnathab3/edth-acoustic-drone-detector
- Documentation: See `CHALLENGE_READY.md`, `SOTA_README.md`

## 💡 Pro Tips

1. **Start with validation** to know baseline performance
2. **Monitor CSV file** for patterns and improvements
3. **Use GPU** for 10-20x faster inference
4. **Enable internet** for API access
5. **Save results** to Kaggle output for persistence
6. **Run overnight** to maximize score
7. **Check timing** - adjust delay if seeing many duplicates

## 📧 Support

See documentation:
- `CHALLENGE_READY.md` - Quick start guide
- `SOTA_README.md` - Architecture details
- `TIMING_STRATEGY_EXPLAINED.md` - Timing strategy
- `QUICK_COMMANDS.md` - Command reference

---

**Last Updated**: October 2025  
**Model Version**: SOTA v1.0 (PANNs-based)  
**Status**: ✅ Production Ready
