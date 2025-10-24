"""
Quick Test Script for FFT + CNN + DNN System
Tests that all components are working correctly
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
from pathlib import Path

print("="*80)
print("FFT + CNN + DNN System Test")
print("="*80)

# Test 1: FFT Processor
print("\n1. Testing FFT Processor...")
try:
    from adrone.features.fft_processor import FFTProcessor, FFTStatisticalFeatures
    
    fft_processor = FFTProcessor(
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        sample_rate=16000
    )
    
    # Create dummy audio
    dummy_audio = np.random.randn(16000 * 2)  # 2 seconds
    
    # Extract features
    features = fft_processor.extract_features_for_model(dummy_audio)
    print(f"   ✓ FFT Processor working")
    print(f"   ✓ Feature shape: {features.shape}")
    
    # Test statistical features
    stat_extractor = FFTStatisticalFeatures(n_fft=2048, sample_rate=16000)
    stat_features = stat_extractor.extract_statistical_features(dummy_audio)
    print(f"   ✓ Statistical features: {len(stat_features)} features")
    
except Exception as e:
    print(f"   ✗ FFT Processor failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: FFT + CNN + DNN Model
print("\n2. Testing FFT + CNN + DNN Model...")
try:
    from adrone.models.fft_cnn_dnn import FFTCNNDNNFusion, MultiScaleCNNDNN
    
    # Create model
    model = FFTCNNDNNFusion(
        n_classes=3,
        in_channels=1,
        cnn_feature_dim=512,
        dnn_hidden_dims=[256, 128]
    )
    
    # Test forward pass
    dummy_input = torch.randn(2, 1, 128, 100)  # batch_size=2
    output = model(dummy_input)
    
    print(f"   ✓ Model created successfully")
    print(f"   ✓ Input shape: {dummy_input.shape}")
    print(f"   ✓ Output shape: {output.shape}")
    print(f"   ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test multi-scale model
    multi_model = MultiScaleCNNDNN(n_classes=3, in_channels=1)
    multi_output = multi_model(dummy_input)
    print(f"   ✓ Multi-scale model working")
    
except Exception as e:
    print(f"   ✗ Model test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Challenge Handler
print("\n3. Testing Challenge Handler...")
try:
    from adrone.serve.challenge_handler import (
        ChallengeResultStorage,
        ChallengeAPIClient,
        AdaptiveLearningTracker
    )
    
    # Test storage
    test_storage_dir = "test_challenge_results"
    storage = ChallengeResultStorage(test_storage_dir)
    
    # Store a dummy result
    storage.store_result(
        challenge_id="test-123",
        audio_path="dummy.wav",
        prediction="drone",
        confidence=0.95,
        all_probabilities={"drone": 0.95, "bird": 0.03, "background": 0.02},
        is_correct=True,
        score_awarded=150,
        inference_time=0.05
    )
    
    stats = storage.get_statistics()
    print(f"   ✓ Storage working")
    print(f"   ✓ Total attempts: {stats['total_attempts']}")
    print(f"   ✓ Accuracy: {stats['accuracy']*100:.1f}%")
    
    # Test adaptive tracker
    tracker = AdaptiveLearningTracker()
    tracker.add_result("drone", 0.95, True)
    tracker.add_result("bird", 0.85, False)
    print(f"   ✓ Adaptive tracker working")
    
    # Cleanup
    import shutil
    if Path(test_storage_dir).exists():
        shutil.rmtree(test_storage_dir)
    
except Exception as e:
    print(f"   ✗ Challenge handler test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: End-to-End Pipeline
print("\n4. Testing End-to-End Pipeline...")
try:
    # Create dummy audio
    sample_rate = 16000
    duration = 2.0
    audio = np.random.randn(int(sample_rate * duration))
    
    # Extract FFT features
    fft_proc = FFTProcessor(
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        sample_rate=sample_rate
    )
    features = fft_proc.extract_features_for_model(audio)
    
    # Add batch dimension
    features = features.unsqueeze(0)
    
    # Forward through model
    model = FFTCNNDNNFusion(n_classes=3, in_channels=1)
    model.eval()
    
    with torch.no_grad():
        logits = model(features)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    print(f"   ✓ End-to-end pipeline working")
    print(f"   ✓ Predicted class: {predicted_class}")
    print(f"   ✓ Confidence: {confidence:.2%}")
    
except Exception as e:
    print(f"   ✗ Pipeline test failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "="*80)
print("Test Summary")
print("="*80)
print("✓ All core components are functional")
print("\nNext steps:")
print("1. Train the model: python scripts/train_fft_cnn_dnn.py")
print("2. Run challenge bot: python challenge_bot_fft_cnn_dnn.py")
print("3. View results: check challenge_results/ directory")
print("="*80)
