"""
Quick script to monitor training progress and visualize intermediate results.

Usage:
    python monitor_training.py
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import time

def plot_progress(history_file, title, output_file):
    """Plot training progress from history file"""
    if not history_file.exists():
        return False
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    if not history['train_loss']:
        return False
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=100)
    plt.close()
    
    return True

def main():
    output_dir = Path("models/matched_bank_comparison")
    
    print("Monitoring training progress...")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            print("\n" + "="*60)
            print(f"Time: {time.strftime('%H:%M:%S')}")
            print("="*60)
            
            # Check baseline
            baseline_history = output_dir / "baseline_history.json"
            if baseline_history.exists():
                with open(baseline_history, 'r') as f:
                    hist = json.load(f)
                if hist['train_loss']:
                    latest_epoch = len(hist['train_loss'])
                    print(f"\nBaseline CRNN - Epoch {latest_epoch}/30")
                    print(f"  Train Loss: {hist['train_loss'][-1]:.4f}")
                    print(f"  Train Acc:  {hist['train_acc'][-1]:.2f}%")
                    print(f"  Val Loss:   {hist['val_loss'][-1]:.4f}")
                    print(f"  Val Acc:    {hist['val_acc'][-1]:.2f}%")
                    
                    # Plot
                    plot_progress(
                        baseline_history,
                        "Baseline CRNN Training Progress",
                        output_dir / "baseline_progress.png"
                    )
            else:
                print("\nBaseline training not started yet...")
            
            # Check enhanced
            enhanced_history = output_dir / "enhanced_history.json"
            if enhanced_history.exists():
                with open(enhanced_history, 'r') as f:
                    hist = json.load(f)
                if hist['train_loss']:
                    latest_epoch = len(hist['train_loss'])
                    print(f"\nEnhanced CRNN (Matched Bank) - Epoch {latest_epoch}/30")
                    print(f"  Train Loss: {hist['train_loss'][-1]:.4f}")
                    print(f"  Train Acc:  {hist['train_acc'][-1]:.2f}%")
                    print(f"  Val Loss:   {hist['val_loss'][-1]:.4f}")
                    print(f"  Val Acc:    {hist['val_acc'][-1]:.2f}%")
                    
                    # Plot
                    plot_progress(
                        enhanced_history,
                        "Enhanced CRNN (Matched Bank) Training Progress",
                        output_dir / "enhanced_progress.png"
                    )
            else:
                print("\nEnhanced training not started yet...")
            
            # Check if training is complete
            summary_file = output_dir / "summary.json"
            if summary_file.exists():
                print("\n" + "="*60)
                print("TRAINING COMPLETE!")
                print("="*60)
                
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                
                print(f"\nFinal Results:")
                print(f"  Baseline Val Acc:  {summary['baseline']['best_val_acc']:.2f}%")
                print(f"  Enhanced Val Acc:  {summary['enhanced']['best_val_acc']:.2f}%")
                print(f"  Improvement:       {summary['improvement']['val_acc']:+.2f}%")
                print(f"  Param Overhead:    +{summary['enhanced']['param_overhead_pct']:.1f}%")
                
                break
            
            # Wait before next check
            time.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    main()
