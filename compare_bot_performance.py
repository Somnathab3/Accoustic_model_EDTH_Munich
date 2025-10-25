"""
Compare performance between original and optimized challenge bot
"""
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_csv(csv_path, name):
    """Analyze a results CSV file"""
    if not Path(csv_path).exists():
        print(f"❌ {name}: File not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    
    if len(df) == 0:
        print(f"❌ {name}: No data")
        return None
    
    # Filter successful submissions (score > 0)
    successful = df[df['score_awarded'] > 0]
    
    if len(successful) == 0:
        print(f"❌ {name}: No successful submissions")
        return None
    
    # Calculate statistics
    stats = {
        'name': name,
        'total_submissions': len(df),
        'successful_submissions': len(successful),
        'accuracy': len(successful) / len(df) * 100,
        'avg_score': successful['score_awarded'].mean(),
        'max_score': successful['score_awarded'].max(),
        'total_score': df['total_score'].max() if len(df) > 0 else 0,
        'avg_total_time': successful['total_time'].mean(),
        'avg_inference_time': successful['inference_time'].mean(),
        'std_total_time': successful['total_time'].std(),
        'min_total_time': successful['total_time'].min(),
        'max_total_time': successful['total_time'].max(),
    }
    
    # Check optimal window (99.5-100.2s)
    in_window = successful[(successful['total_time'] >= 99.5) & (successful['total_time'] <= 100.2)]
    stats['in_optimal_window'] = len(in_window)
    stats['window_percentage'] = len(in_window) / len(successful) * 100 if len(successful) > 0 else 0
    stats['avg_score_in_window'] = in_window['score_awarded'].mean() if len(in_window) > 0 else 0
    
    # High scores (>=190)
    high_scores = successful[successful['score_awarded'] >= 190]
    stats['high_scores_count'] = len(high_scores)
    stats['high_scores_percentage'] = len(high_scores) / len(successful) * 100 if len(successful) > 0 else 0
    stats['avg_time_high_scores'] = high_scores['total_time'].mean() if len(high_scores) > 0 else 0
    
    return stats

def compare_results():
    """Compare original and optimized results"""
    print("="*80)
    print("CHALLENGE BOT PERFORMANCE COMPARISON")
    print("="*80)
    
    original_path = "challenge_results/results.csv"
    optimized_path = "challenge_results/results_optimized.csv"
    
    # Analyze both
    original = analyze_csv(original_path, "ORIGINAL")
    optimized = analyze_csv(optimized_path, "OPTIMIZED")
    
    if original is None and optimized is None:
        print("\n❌ No data to compare. Please run both bots first.")
        return
    
    print("\n" + "="*80)
    print("DETAILED COMPARISON")
    print("="*80)
    
    # Print side-by-side comparison
    metrics = [
        ('Total Submissions', 'total_submissions', ''),
        ('Successful Submissions', 'successful_submissions', ''),
        ('Accuracy', 'accuracy', '%'),
        ('Total Score', 'total_score', ''),
        ('Average Score/Submission', 'avg_score', ''),
        ('Max Score', 'max_score', ''),
        ('', '', ''),
        ('Average Total Time', 'avg_total_time', 's'),
        ('Std Dev Total Time', 'std_total_time', 's'),
        ('Min Total Time', 'min_total_time', 's'),
        ('Max Total Time', 'max_total_time', 's'),
        ('Average Inference Time', 'avg_inference_time', 's'),
        ('', '', ''),
        ('In Optimal Window (99.5-100.2s)', 'in_optimal_window', ''),
        ('Window Hit Rate', 'window_percentage', '%'),
        ('Avg Score in Window', 'avg_score_in_window', ''),
        ('', '', ''),
        ('High Scores (>=190)', 'high_scores_count', ''),
        ('High Score Rate', 'high_scores_percentage', '%'),
        ('Avg Time for High Scores', 'avg_time_high_scores', 's'),
    ]
    
    print(f"\n{'Metric':<35} {'Original':<20} {'Optimized':<20} {'Improvement':<15}")
    print("-"*90)
    
    for metric_name, key, unit in metrics:
        if not metric_name:
            print()
            continue
        
        if original and key in original:
            orig_val = original[key]
        else:
            orig_val = None
        
        if optimized and key in optimized:
            opt_val = optimized[key]
        else:
            opt_val = None
        
        # Format values
        if orig_val is not None:
            if isinstance(orig_val, float):
                orig_str = f"{orig_val:.2f}{unit}"
            else:
                orig_str = f"{orig_val}{unit}"
        else:
            orig_str = "N/A"
        
        if opt_val is not None:
            if isinstance(opt_val, float):
                opt_str = f"{opt_val:.2f}{unit}"
            else:
                opt_str = f"{opt_val}{unit}"
        else:
            opt_str = "N/A"
        
        # Calculate improvement
        if orig_val is not None and opt_val is not None and isinstance(orig_val, (int, float)) and isinstance(opt_val, (int, float)):
            if key in ['avg_total_time', 'std_total_time', 'avg_inference_time']:
                # Lower is better
                improvement = ((orig_val - opt_val) / orig_val * 100) if orig_val != 0 else 0
                if improvement > 0:
                    impr_str = f"↓ {improvement:.1f}%"
                elif improvement < 0:
                    impr_str = f"↑ {-improvement:.1f}%"
                else:
                    impr_str = "="
            else:
                # Higher is better
                improvement = ((opt_val - orig_val) / orig_val * 100) if orig_val != 0 else 0
                if improvement > 0:
                    impr_str = f"↑ {improvement:.1f}%"
                elif improvement < 0:
                    impr_str = f"↓ {-improvement:.1f}%"
                else:
                    impr_str = "="
        else:
            impr_str = "-"
        
        print(f"{metric_name:<35} {orig_str:<20} {opt_str:<20} {impr_str:<15}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    if optimized and original:
        if optimized['window_percentage'] > original['window_percentage']:
            improvement = optimized['window_percentage'] - original['window_percentage']
            print(f"✓ Optimal window hit rate improved by {improvement:.1f} percentage points")
        
        if optimized['avg_score'] > original['avg_score']:
            improvement = optimized['avg_score'] - original['avg_score']
            print(f"✓ Average score improved by {improvement:.1f} points per submission")
        
        if optimized['high_scores_percentage'] > original['high_scores_percentage']:
            improvement = optimized['high_scores_percentage'] - original['high_scores_percentage']
            print(f"✓ High score rate (>=190) improved by {improvement:.1f} percentage points")
        
        if optimized['std_total_time'] < original['std_total_time']:
            print(f"✓ Timing consistency improved (lower variance)")
        
        # Check if optimized is more consistent around 100s
        opt_close_to_100 = abs(100 - optimized['avg_total_time'])
        orig_close_to_100 = abs(100 - original['avg_total_time'])
        
        if opt_close_to_100 < orig_close_to_100:
            print(f"✓ Optimized bot is closer to ideal 100s timing window")
        
        print(f"\n🎯 RECOMMENDATION:")
        if optimized['avg_score'] > original['avg_score'] * 1.05:  # 5% improvement
            print(f"   USE OPTIMIZED BOT - Significant improvement detected!")
        elif optimized['window_percentage'] > original['window_percentage'] * 1.1:  # 10% improvement
            print(f"   USE OPTIMIZED BOT - Better timing control!")
        else:
            print(f"   Continue testing both versions to gather more data")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    compare_results()
