"""
Batch inference script to test multiple audio files.
Usage: python scripts/batch_infer.py --num-samples 10
"""
import argparse
import sys
from pathlib import Path
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adrone.infer import InferenceModel

def main():
    parser = argparse.ArgumentParser(description='Run batch inference on multiple audio files')
    parser.add_argument('--audio-dir', type=str, default='data/raw/train',
                        help='Directory containing audio files organized by class')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to test per class')
    parser.add_argument('--model', type=str, default='models/cnn_small.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--labels', type=str, default='models/labels.json',
                        help='Path to labels JSON')
    args = parser.parse_args()
    
    print(f"🚀 Loading model from {args.model}...")
    model = InferenceModel(model_path=args.model, labels_path=args.labels)
    
    audio_dir = Path(args.audio_dir)
    
    # Collect samples by class
    results = {}
    total_correct = 0
    total_tested = 0
    
    print(f"\n📊 Testing {args.num_samples} samples per class...\n")
    
    for class_dir in sorted(audio_dir.iterdir()):
        if class_dir.is_dir():
            class_name = class_dir.name
            audio_files = list(class_dir.glob('*.wav'))
            
            if not audio_files:
                print(f"⚠️  No audio files found in class '{class_name}'")
                continue
            
            # Sample random files
            test_files = random.sample(audio_files, min(args.num_samples, len(audio_files)))
            
            print(f"{'='*70}")
            print(f"Class: {class_name} ({len(test_files)} samples)")
            print(f"{'='*70}")
            
            correct = 0
            class_results = []
            
            for audio_file in test_files:
                try:
                    prediction = model.predict_path(str(audio_file))
                    predicted_class = max(prediction, key=prediction.get)
                    confidence = prediction[predicted_class]
                    
                    is_correct = predicted_class == class_name
                    if is_correct:
                        correct += 1
                    
                    status = "✅" if is_correct else "❌"
                    
                    print(f"{status} {audio_file.name[:30]:30} → Class: {predicted_class} (conf: {confidence:.4f})")
                    
                    class_results.append({
                        'file': audio_file.name,
                        'true_label': class_name,
                        'predicted': predicted_class,
                        'confidence': confidence,
                        'correct': is_correct
                    })
                    
                except Exception as e:
                    print(f"❌ Error processing {audio_file.name}: {e}")
            
            accuracy = correct / len(test_files) if test_files else 0
            print(f"\n📈 Class '{class_name}' Accuracy: {correct}/{len(test_files)} = {accuracy:.2%}\n")
            
            results[class_name] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': len(test_files),
                'samples': class_results
            }
            
            total_correct += correct
            total_tested += len(test_files)
    
    # Overall summary
    print(f"\n{'='*70}")
    print(f"📊 OVERALL RESULTS")
    print(f"{'='*70}")
    
    for class_name, class_data in results.items():
        print(f"Class '{class_name}': {class_data['correct']}/{class_data['total']} = {class_data['accuracy']:.2%}")
    
    overall_accuracy = total_correct / total_tested if total_tested else 0
    print(f"\n🎯 Overall Accuracy: {total_correct}/{total_tested} = {overall_accuracy:.2%}")
    
    # Show misclassified samples
    misclassified = []
    for class_name, class_data in results.items():
        for sample in class_data['samples']:
            if not sample['correct']:
                misclassified.append(sample)
    
    if misclassified:
        print(f"\n❌ Misclassified Samples ({len(misclassified)}):")
        print(f"{'='*70}")
        for sample in misclassified:
            print(f"  {sample['file']:30} True: {sample['true_label']} → Pred: {sample['predicted']} (conf: {sample['confidence']:.4f})")
    else:
        print(f"\n✅ All samples classified correctly!")

if __name__ == "__main__":
    main()
