"""
Detailed analysis of model predictions on Drone-detection-dataset
"""
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Read the detailed results
with open('evaluation_results/drone_detection_dataset_detailed.json', 'r') as f:
    results = json.load(f)

# Analyze predictions for each model
for model_name, model_results in results.items():
    print(f"\n{'='*80}")
    print(f"ANALYSIS: {model_name}")
    print('='*80)
    
    file_results = model_results['file_results']
    df = pd.DataFrame(file_results)
    
    # Group by true label and show predictions
    print("\nPrediction Distribution by True Label:")
    print("="*80)
    
    for true_label in ['background', 'drone', 'helicopter']:
        subset = df[df['true_label'] == true_label]
        pred_counts = subset['pred_label'].value_counts()
        print(f"\n{true_label.upper()} ({len(subset)} samples):")
        for pred, count in pred_counts.items():
            pct = (count / len(subset)) * 100
            print(f"  Predicted as {pred:12s}: {count:2d} ({pct:5.1f}%)")
        
        # Average confidence
        avg_conf = subset['confidence'].mean()
        print(f"  Average confidence: {avg_conf:.4f}")
    
    # Show some examples with lowest confidence
    print(f"\n{'='*80}")
    print("LOWEST CONFIDENCE PREDICTIONS (10 samples):")
    print('='*80)
    lowest_conf = df.nsmallest(10, 'confidence')[['filename', 'true_label', 'pred_label', 'confidence', 'correct']]
    print(lowest_conf.to_string(index=False))
    
    # Check if any samples were correctly classified
    correct = df[df['correct'] == True]
    print(f"\n{'='*80}")
    print(f"CORRECTLY CLASSIFIED: {len(correct)} / {len(df)} ({len(correct)/len(df)*100:.1f}%)")
    print('='*80)
    
    if len(correct) > 0:
        print("\nExamples of correct predictions:")
        print(correct.head(10)[['filename', 'true_label', 'confidence']].to_string(index=False))
    else:
        print("No samples were correctly classified!")
    
    # Confidence distribution by prediction
    print(f"\n{'='*80}")
    print("CONFIDENCE STATISTICS BY PREDICTED CLASS:")
    print('='*80)
    for pred_label in ['background', 'drone', 'helicopter']:
        subset = df[df['pred_label'] == pred_label]
        if len(subset) > 0:
            print(f"\n{pred_label.upper()} predictions ({len(subset)} samples):")
            print(f"  Mean confidence: {subset['confidence'].mean():.4f}")
            print(f"  Std confidence:  {subset['confidence'].std():.4f}")
            print(f"  Min confidence:  {subset['confidence'].min():.4f}")
            print(f"  Max confidence:  {subset['confidence'].max():.4f}")
