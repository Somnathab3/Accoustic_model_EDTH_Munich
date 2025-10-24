"""
Quick CSV Results Analyzer
Analyzes challenge_results/results.csv and shows performance metrics
"""
import csv
import sys
from pathlib import Path
from collections import Counter, defaultdict

def analyze_csv(csv_path='challenge_results/results.csv'):
    """Analyze CSV results and print statistics"""
    
    if not Path(csv_path).exists():
        print(f"❌ CSV file not found: {csv_path}")
        print("\nRun the challenge bot first:")
        print("  python sota_challenge_bot.py --max-iterations 10")
        return
    
    # Read all results
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        results = list(reader)
    
    if not results:
        print("❌ No results in CSV file yet")
        return
    
    print("\n" + "="*70)
    print(f"CHALLENGE BOT RESULTS ANALYSIS")
    print("="*70)
    print(f"Total Challenges: {len(results)}")
    print(f"CSV File: {csv_path}")
    print("="*70 + "\n")
    
    # Overall stats
    correct = sum(1 for r in results if r['correct'] == 'True')
    total = len(results)
    accuracy = correct / total * 100 if total > 0 else 0
    total_score = sum(int(r['score_awarded']) for r in results)
    
    print(f"📊 OVERALL PERFORMANCE")
    print(f"  Correct:  {correct}/{total} ({accuracy:.1f}%)")
    print(f"  Wrong:    {total - correct}/{total} ({100-accuracy:.1f}%)")
    print(f"  Total Score: {total_score:,}")
    print(f"  Avg Score/Challenge: {total_score/total:.1f}")
    print()
    
    # Prediction distribution
    predictions = Counter(r['predicted'] for r in results)
    actuals = Counter(r['actual'] for r in results if r['actual'] != 'unknown')
    
    print(f"🎯 PREDICTION DISTRIBUTION")
    for label, count in predictions.most_common():
        pct = count / total * 100
        print(f"  {label:12s}: {count:3d} ({pct:5.1f}%)")
    print()
    
    if actuals:
        print(f"📌 ACTUAL DISTRIBUTION")
        total_actual = sum(actuals.values())
        for label, count in actuals.most_common():
            pct = count / total_actual * 100
            print(f"  {label:12s}: {count:3d} ({pct:5.1f}%)")
        print()
    
    # Per-class accuracy
    class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in results:
        if r['actual'] != 'unknown':
            label = r['actual']
            class_stats[label]['total'] += 1
            if r['correct'] == 'True':
                class_stats[label]['correct'] += 1
    
    if class_stats:
        print(f"📈 PER-CLASS ACCURACY")
        print(f"  {'Class':<12s}  Correct  Total    Accuracy")
        print(f"  {'-'*12}  -------  -----    --------")
        for label in sorted(class_stats.keys()):
            stats = class_stats[label]
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"  {label:12s}  {stats['correct']:4d}     {stats['total']:4d}     {acc:5.1f}%")
        print()
    
    # Confusion matrix
    if class_stats:
        print(f"🔀 CONFUSION MATRIX")
        all_labels = sorted(set(predictions.keys()) | set(actuals.keys()))
        
        # Header
        print(f"  {'Actual \\ Pred':<12s}", end='  ')
        for label in all_labels:
            print(f"{label[:10]:>10s}", end='  ')
        print()
        print(f"  {'-'*12}", end='  ')
        for _ in all_labels:
            print(f"{'-'*10}", end='  ')
        print()
        
        # Matrix
        confusion = defaultdict(lambda: defaultdict(int))
        for r in results:
            if r['actual'] != 'unknown':
                confusion[r['actual']][r['predicted']] += 1
        
        for actual in all_labels:
            if actual in actuals:
                print(f"  {actual:12s}", end='  ')
                for predicted in all_labels:
                    count = confusion[actual][predicted]
                    print(f"{count:10d}", end='  ')
                print()
        print()
    
    # Timing stats
    inference_times = [float(r['inference_time']) for r in results]
    total_times = [float(r['total_time']) for r in results]
    
    print(f"⏱️  TIMING STATISTICS")
    print(f"  Inference Time (model only):")
    print(f"    Mean:   {sum(inference_times)/len(inference_times):.3f}s")
    print(f"    Min:    {min(inference_times):.3f}s")
    print(f"    Max:    {max(inference_times):.3f}s")
    print(f"  Total Time (download + inference + submit):")
    print(f"    Mean:   {sum(total_times)/len(total_times):.3f}s")
    print(f"    Min:    {min(total_times):.3f}s")
    print(f"    Max:    {max(total_times):.3f}s")
    print()
    
    # Confidence stats
    confidences = [float(r['confidence']) for r in results]
    correct_conf = [float(r['confidence']) for r in results if r['correct'] == 'True']
    wrong_conf = [float(r['confidence']) for r in results if r['correct'] == 'False']
    
    print(f"🎲 CONFIDENCE STATISTICS")
    print(f"  Overall:")
    print(f"    Mean:   {sum(confidences)/len(confidences):.3f}")
    print(f"    Min:    {min(confidences):.3f}")
    print(f"    Max:    {max(confidences):.3f}")
    if correct_conf:
        print(f"  Correct Predictions:")
        print(f"    Mean:   {sum(correct_conf)/len(correct_conf):.3f}")
    if wrong_conf:
        print(f"  Wrong Predictions:")
        print(f"    Mean:   {sum(wrong_conf)/len(wrong_conf):.3f}")
    print()
    
    # Recent performance (last 10)
    if len(results) >= 10:
        recent = results[-10:]
        recent_correct = sum(1 for r in recent if r['correct'] == 'True')
        recent_acc = recent_correct / 10 * 100
        recent_score = sum(int(r['score_awarded']) for r in recent)
        
        print(f"🔥 RECENT PERFORMANCE (Last 10)")
        print(f"  Accuracy: {recent_correct}/10 ({recent_acc:.1f}%)")
        print(f"  Score: {recent_score}")
        print()
    
    # Score per class (when correct)
    class_scores = defaultdict(list)
    for r in results:
        if r['correct'] == 'True' and r['actual'] != 'unknown':
            class_scores[r['actual']].append(int(r['score_awarded']))
    
    if class_scores:
        print(f"💰 AVERAGE SCORE PER CLASS (when correct)")
        for label in sorted(class_scores.keys()):
            scores = class_scores[label]
            avg_score = sum(scores) / len(scores)
            print(f"  {label:12s}: {avg_score:.1f} (n={len(scores)})")
        print()
    
    print("="*70)
    print(f"✓ Analysis complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'challenge_results/results.csv'
    analyze_csv(csv_path)
