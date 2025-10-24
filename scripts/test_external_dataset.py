"""
Test model robustness on external DroneAudioDataset (Al-Emadi et al.)
This script evaluates cross-domain performance on indoor drone recordings.

Usage: python scripts/test_external_dataset.py --test-dir data/test_dataset/Binary_Drone_Audio
"""
import argparse
import sys
from pathlib import Path
import random
import json
from datetime import datetime
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adrone.infer import InferenceModel

def calculate_metrics(true_positives, false_positives, true_negatives, false_negatives):
    """Calculate precision, recall, F1-score, and accuracy"""
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives) if (true_positives + true_negatives + false_positives + false_negatives) > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def main():
    parser = argparse.ArgumentParser(description='Test model on external DroneAudioDataset')
    parser.add_argument('--test-dir', type=str, default='data/test_dataset/Binary_Drone_Audio',
                        help='Directory containing yes_drone and unknown folders')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to test per class (None = all)')
    parser.add_argument('--model', type=str, default='models/cnn_small.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--labels', type=str, default='models/labels.json',
                        help='Path to labels JSON')
    parser.add_argument('--output', type=str, default='test_results',
                        help='Output directory for results')
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                        help='Confidence threshold for binary classification')
    args = parser.parse_args()
    
    print("="*80)
    print("🎯 EXTERNAL DATASET TESTING - DroneAudioDataset (Al-Emadi et al.)")
    print("="*80)
    print(f"📁 Test Directory: {args.test_dir}")
    print(f"🤖 Model: {args.model}")
    print(f"🎚️  Confidence Threshold: {args.confidence_threshold}")
    print("="*80)
    
    # Load model
    print(f"\n🚀 Loading model...")
    model = InferenceModel(model_path=args.model, labels_path=args.labels)
    print(f"✅ Model loaded successfully!")
    
    test_dir = Path(args.test_dir)
    
    # Expected folders: yes_drone (class 1) and unknown (class 0)
    yes_drone_dir = test_dir / "yes_drone"
    unknown_dir = test_dir / "unknown"
    
    if not yes_drone_dir.exists() or not unknown_dir.exists():
        print(f"❌ Error: Expected 'yes_drone' and 'unknown' folders in {test_dir}")
        return
    
    # Confusion matrix counters
    true_positives = 0  # Correctly predicted as drone
    false_positives = 0  # Predicted as drone but was unknown
    true_negatives = 0  # Correctly predicted as unknown
    false_negatives = 0  # Predicted as unknown but was drone
    
    results = {
        'yes_drone': {'samples': [], 'correct': 0, 'total': 0},
        'unknown': {'samples': [], 'correct': 0, 'total': 0}
    }
    
    errors_by_type = defaultdict(list)
    
    # Test yes_drone samples (True class: 1 - drone)
    print(f"\n{'='*80}")
    print(f"🚁 Testing DRONE samples (yes_drone folder)")
    print(f"{'='*80}")
    
    yes_drone_files = list(yes_drone_dir.glob('*.wav'))
    if args.num_samples:
        yes_drone_files = random.sample(yes_drone_files, min(args.num_samples, len(yes_drone_files)))
    
    print(f"📊 Total samples to test: {len(yes_drone_files)}")
    
    for i, audio_file in enumerate(yes_drone_files, 1):
        try:
            prediction = model.predict_path(str(audio_file))
            
            # Get prediction for class "1" (drone)
            drone_confidence = prediction.get('1', 0.0)
            predicted_class = '1' if drone_confidence >= args.confidence_threshold else '0'
            
            is_correct = predicted_class == '1'
            
            if is_correct:
                true_positives += 1
                results['yes_drone']['correct'] += 1
                status = "✅"
            else:
                false_negatives += 1
                status = "❌"
                # Categorize error by file type
                if 'bebop' in audio_file.name.lower():
                    errors_by_type['bebop_missed'].append(audio_file.name)
                elif 'membo' in audio_file.name.lower():
                    errors_by_type['membo_missed'].append(audio_file.name)
                elif 'mixed' in audio_file.name.lower():
                    errors_by_type['mixed_missed'].append(audio_file.name)
            
            results['yes_drone']['samples'].append({
                'file': audio_file.name,
                'true_label': '1',
                'predicted': predicted_class,
                'drone_confidence': drone_confidence,
                'correct': is_correct
            })
            results['yes_drone']['total'] += 1
            
            if i % 100 == 0 or i == len(yes_drone_files):
                print(f"  Progress: {i}/{len(yes_drone_files)} - Current Accuracy: {true_positives}/{i} = {true_positives/i:.2%}")
        
        except Exception as e:
            print(f"❌ Error processing {audio_file.name}: {e}")
    
    # Test unknown samples (True class: 0 - not drone)
    print(f"\n{'='*80}")
    print(f"🔇 Testing NON-DRONE samples (unknown folder)")
    print(f"{'='*80}")
    
    unknown_files = list(unknown_dir.glob('*.wav'))
    if args.num_samples:
        unknown_files = random.sample(unknown_files, min(args.num_samples, len(unknown_files)))
    
    print(f"📊 Total samples to test: {len(unknown_files)}")
    
    for i, audio_file in enumerate(unknown_files, 1):
        try:
            prediction = model.predict_path(str(audio_file))
            
            # Get prediction for class "1" (drone)
            drone_confidence = prediction.get('1', 0.0)
            predicted_class = '1' if drone_confidence >= args.confidence_threshold else '0'
            
            is_correct = predicted_class == '0'
            
            if is_correct:
                true_negatives += 1
                results['unknown']['correct'] += 1
                status = "✅"
            else:
                false_positives += 1
                status = "❌"
                # Categorize false positive by file type
                if 'silence' in audio_file.name.lower():
                    errors_by_type['silence_false_positive'].append(audio_file.name)
                elif 'noise' in audio_file.name.lower():
                    errors_by_type['noise_false_positive'].append(audio_file.name)
                elif audio_file.name.startswith(('3-', '4-', '5-')):
                    errors_by_type['environmental_false_positive'].append(audio_file.name)
                else:
                    errors_by_type['other_false_positive'].append(audio_file.name)
            
            results['unknown']['samples'].append({
                'file': audio_file.name,
                'true_label': '0',
                'predicted': predicted_class,
                'drone_confidence': drone_confidence,
                'correct': is_correct
            })
            results['unknown']['total'] += 1
            
            if i % 100 == 0 or i == len(unknown_files):
                print(f"  Progress: {i}/{len(unknown_files)} - Current Accuracy: {true_negatives}/{i} = {true_negatives/i:.2%}")
        
        except Exception as e:
            print(f"❌ Error processing {audio_file.name}: {e}")
    
    # Calculate metrics
    metrics = calculate_metrics(true_positives, false_positives, true_negatives, false_negatives)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"📊 TEST RESULTS - DroneAudioDataset")
    print(f"{'='*80}")
    
    print(f"\n📈 Confusion Matrix:")
    print(f"                    Predicted Drone    Predicted Unknown")
    print(f"  Actual Drone      {true_positives:^15}    {false_negatives:^17}")
    print(f"  Actual Unknown    {false_positives:^15}    {true_negatives:^17}")
    
    print(f"\n🎯 Performance Metrics:")
    print(f"  Overall Accuracy:  {metrics['accuracy']:.2%}")
    print(f"  Precision:         {metrics['precision']:.2%} (of predicted drones, how many were correct)")
    print(f"  Recall (TPR):      {metrics['recall']:.2%} (of actual drones, how many were detected)")
    print(f"  F1-Score:          {metrics['f1_score']:.2%}")
    
    # Calculate specificity
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    print(f"  Specificity (TNR): {specificity:.2%} (of actual unknowns, how many were correctly rejected)")
    
    # Per-class breakdown
    print(f"\n📊 Per-Class Results:")
    yes_drone_acc = results['yes_drone']['correct'] / results['yes_drone']['total'] if results['yes_drone']['total'] > 0 else 0
    unknown_acc = results['unknown']['correct'] / results['unknown']['total'] if results['unknown']['total'] > 0 else 0
    
    print(f"  Drone (yes_drone):   {results['yes_drone']['correct']}/{results['yes_drone']['total']} = {yes_drone_acc:.2%}")
    print(f"  Unknown:             {results['unknown']['correct']}/{results['unknown']['total']} = {unknown_acc:.2%}")
    
    # Error analysis
    if errors_by_type:
        print(f"\n❌ Error Analysis:")
        for error_type, files in sorted(errors_by_type.items()):
            print(f"  {error_type}: {len(files)} errors")
            if len(files) <= 5:
                for f in files:
                    print(f"    - {f}")
            else:
                for f in files[:3]:
                    print(f"    - {f}")
                print(f"    ... and {len(files) - 3} more")
    
    # Save detailed results
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON report
    report = {
        'timestamp': timestamp,
        'model': args.model,
        'test_dataset': str(test_dir),
        'confidence_threshold': args.confidence_threshold,
        'confusion_matrix': {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        },
        'metrics': metrics,
        'specificity': specificity,
        'per_class': {
            'yes_drone': {
                'accuracy': yes_drone_acc,
                'total': results['yes_drone']['total'],
                'correct': results['yes_drone']['correct']
            },
            'unknown': {
                'accuracy': unknown_acc,
                'total': results['unknown']['total'],
                'correct': results['unknown']['correct']
            }
        },
        'error_analysis': {k: len(v) for k, v in errors_by_type.items()},
        'detailed_results': results
    }
    
    report_file = output_dir / f"external_test_report_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: {report_file}")
    
    # Save summary text report
    summary_file = output_dir / f"external_test_summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXTERNAL DATASET TEST RESULTS - DroneAudioDataset (Al-Emadi et al.)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Test Dataset: {test_dir}\n")
        f.write(f"Confidence Threshold: {args.confidence_threshold}\n\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write(f"                    Predicted Drone    Predicted Unknown\n")
        f.write(f"  Actual Drone      {true_positives:^15}    {false_negatives:^17}\n")
        f.write(f"  Actual Unknown    {false_positives:^15}    {true_negatives:^17}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"  Overall Accuracy:  {metrics['accuracy']:.2%}\n")
        f.write(f"  Precision:         {metrics['precision']:.2%}\n")
        f.write(f"  Recall (TPR):      {metrics['recall']:.2%}\n")
        f.write(f"  Specificity (TNR): {specificity:.2%}\n")
        f.write(f"  F1-Score:          {metrics['f1_score']:.2%}\n\n")
        
        f.write("PER-CLASS RESULTS:\n")
        f.write(f"  Drone (yes_drone):   {results['yes_drone']['correct']}/{results['yes_drone']['total']} = {yes_drone_acc:.2%}\n")
        f.write(f"  Unknown:             {results['unknown']['correct']}/{results['unknown']['total']} = {unknown_acc:.2%}\n")
    
    print(f"📄 Summary report saved to: {summary_file}")
    
    # Print recommendation
    print(f"\n{'='*80}")
    print("💡 INTERPRETATION:")
    print(f"{'='*80}")
    
    if metrics['accuracy'] >= 0.90:
        print("✅ Excellent cross-domain performance! Model generalizes well.")
    elif metrics['accuracy'] >= 0.80:
        print("✅ Good cross-domain performance. Model is reasonably robust.")
    elif metrics['accuracy'] >= 0.70:
        print("⚠️  Moderate performance. Consider domain adaptation or retraining.")
    else:
        print("❌ Poor cross-domain performance. Significant domain shift detected.")
    
    if metrics['recall'] < 0.80:
        print(f"⚠️  Low recall ({metrics['recall']:.2%}) - Missing many actual drones (high false negatives)")
    
    if metrics['precision'] < 0.80:
        print(f"⚠️  Low precision ({metrics['precision']:.2%}) - Many false alarms (high false positives)")
    
    if specificity < 0.80:
        print(f"⚠️  Low specificity ({specificity:.2%}) - Incorrectly classifying non-drones as drones")
    
    print(f"\n{'='*80}")
    print("✅ Testing complete!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
