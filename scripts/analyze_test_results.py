"""
Analyze and visualize test results from external dataset testing.
Usage: python scripts/analyze_test_results.py --report test_results/external_test_report_*.json
"""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def plot_confusion_matrix(cm_data, output_dir):
    """Plot confusion matrix"""
    matrix = [
        [cm_data['true_positives'], cm_data['false_negatives']],
        [cm_data['false_positives'], cm_data['true_negatives']]
    ]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Drone', 'Predicted Unknown'],
                yticklabels=['Actual Drone', 'Actual Unknown'])
    plt.title('Confusion Matrix - External Dataset Test')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    
    output_file = output_dir / 'confusion_matrix.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Confusion matrix saved to: {output_file}")

def plot_metrics(metrics, output_dir):
    """Plot performance metrics"""
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score']
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metric_names, metric_values, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2%}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylim([0, 1.1])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Metrics - External Dataset', fontsize=14, fontweight='bold')
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.7, label='80% threshold')
    ax.legend()
    plt.tight_layout()
    
    output_file = output_dir / 'performance_metrics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Performance metrics plot saved to: {output_file}")

def plot_error_distribution(error_analysis, output_dir):
    """Plot error type distribution"""
    if not error_analysis:
        return
    
    error_types = list(error_analysis.keys())
    error_counts = list(error_analysis.values())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(error_types, error_counts, color='#e74c3c')
    
    # Add value labels
    for bar, count in zip(bars, error_counts):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' {count}',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Number of Errors', fontsize=12)
    ax.set_title('Error Distribution by Type', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / 'error_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Error distribution plot saved to: {output_file}")

def analyze_confidence_distribution(results, output_dir):
    """Analyze confidence score distributions"""
    drone_correct_conf = []
    drone_wrong_conf = []
    unknown_correct_conf = []
    unknown_wrong_conf = []
    
    for sample in results['yes_drone']['samples']:
        if sample['correct']:
            drone_correct_conf.append(sample['drone_confidence'])
        else:
            drone_wrong_conf.append(sample['drone_confidence'])
    
    for sample in results['unknown']['samples']:
        if sample['correct']:
            unknown_correct_conf.append(sample['drone_confidence'])
        else:
            unknown_wrong_conf.append(sample['drone_confidence'])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Drone samples - Correct predictions
    axes[0, 0].hist(drone_correct_conf, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(x=0.5, color='red', linestyle='--', label='Threshold')
    axes[0, 0].set_title(f'Drone Samples - Correct Predictions (n={len(drone_correct_conf)})')
    axes[0, 0].set_xlabel('Drone Confidence')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # Drone samples - Incorrect predictions
    if drone_wrong_conf:
        axes[0, 1].hist(drone_wrong_conf, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=0.5, color='red', linestyle='--', label='Threshold')
        axes[0, 1].set_title(f'Drone Samples - Incorrect Predictions (n={len(drone_wrong_conf)})')
        axes[0, 1].set_xlabel('Drone Confidence')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, 'No Incorrect Predictions', 
                       ha='center', va='center', fontsize=14, color='green')
        axes[0, 1].set_title('Drone Samples - Incorrect Predictions (n=0)')
    
    # Unknown samples - Correct predictions
    axes[1, 0].hist(unknown_correct_conf, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0.5, color='red', linestyle='--', label='Threshold')
    axes[1, 0].set_title(f'Unknown Samples - Correct Predictions (n={len(unknown_correct_conf)})')
    axes[1, 0].set_xlabel('Drone Confidence')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Unknown samples - Incorrect predictions
    if unknown_wrong_conf:
        axes[1, 1].hist(unknown_wrong_conf, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=0.5, color='red', linestyle='--', label='Threshold')
        axes[1, 1].set_title(f'Unknown Samples - Incorrect Predictions (n={len(unknown_wrong_conf)})')
        axes[1, 1].set_xlabel('Drone Confidence')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'No Incorrect Predictions', 
                       ha='center', va='center', fontsize=14, color='green')
        axes[1, 1].set_title('Unknown Samples - Incorrect Predictions (n=0)')
    
    plt.suptitle('Confidence Score Distributions', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = output_dir / 'confidence_distributions.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Confidence distributions saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze external dataset test results')
    parser.add_argument('--report', type=str, required=True,
                        help='Path to JSON report file')
    parser.add_argument('--output-dir', type=str, default='test_results/visualizations',
                        help='Output directory for plots')
    args = parser.parse_args()
    
    print("="*80)
    print("📊 TEST RESULTS ANALYSIS")
    print("="*80)
    
    # Load report
    report_path = Path(args.report)
    if not report_path.exists():
        print(f"❌ Report file not found: {report_path}")
        return
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Loading report: {report_path}")
    print(f"📅 Test timestamp: {report['timestamp']}")
    print(f"🤖 Model tested: {report['model']}")
    
    # Print summary
    metrics = report['metrics']
    cm = report['confusion_matrix']
    
    print(f"\n{'='*80}")
    print("📊 SUMMARY")
    print(f"{'='*80}")
    print(f"  Overall Accuracy:  {metrics['accuracy']:.2%}")
    print(f"  Precision:         {metrics['precision']:.2%}")
    print(f"  Recall:            {metrics['recall']:.2%}")
    print(f"  F1-Score:          {metrics['f1_score']:.2%}")
    print(f"  Specificity:       {report.get('specificity', 0):.2%}")
    
    print(f"\n  True Positives:    {cm['true_positives']}")
    print(f"  False Positives:   {cm['false_positives']}")
    print(f"  True Negatives:    {cm['true_negatives']}")
    print(f"  False Negatives:   {cm['false_negatives']}")
    
    # Generate plots
    print(f"\n{'='*80}")
    print("📈 GENERATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    plot_confusion_matrix(cm, output_dir)
    plot_metrics(metrics, output_dir)
    
    if 'error_analysis' in report and report['error_analysis']:
        plot_error_distribution(report['error_analysis'], output_dir)
    
    if 'detailed_results' in report:
        analyze_confidence_distribution(report['detailed_results'], output_dir)
    
    print(f"\n✅ Analysis complete! Visualizations saved to: {output_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
