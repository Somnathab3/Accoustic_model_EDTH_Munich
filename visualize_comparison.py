"""
Visualize Old vs New Model Comparison
Creates plots showing prediction changes and confidence analysis
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Load data
df = pd.read_csv('challenge_results/model_comparison.csv')
summary = json.load(open('challenge_results/model_comparison_summary.json'))

fig = plt.figure(figsize=(18, 12))

# 1. Prediction Agreement
ax1 = plt.subplot(2, 3, 1)
agreement_data = [
    summary['same_predictions'],
    summary['different_predictions']
]
colors = ['#4CAF50', '#FF9800']
wedges, texts, autotexts = ax1.pie(agreement_data, 
                                     labels=['Same Prediction', 'Different Prediction'],
                                     autopct='%1.1f%%',
                                     colors=colors,
                                     explode=(0.05, 0.05),
                                     startangle=90,
                                     textprops={'fontsize': 11, 'fontweight': 'bold'})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax1.set_title('Old vs New Model\nPrediction Agreement', fontsize=13, fontweight='bold', pad=15)

# 2. New Model Prediction Distribution
ax2 = plt.subplot(2, 3, 2)
new_preds = df['new_prediction'].value_counts()
bars = ax2.bar(new_preds.index, new_preds.values, 
               color=['#2196F3', '#F44336', '#9C27B0'], alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
ax2.set_title('New Model Predictions\n(on Old Model Failures)', fontsize=13, fontweight='bold', pad=15)
ax2.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# 3. Confidence Comparison
ax3 = plt.subplot(2, 3, 3)
data_to_plot = [df['old_confidence'], df['new_confidence']]
bp = ax3.boxplot(data_to_plot, 
                 labels=['Old Model', 'New Model'],
                 patch_artist=True,
                 widths=0.6)
colors_box = ['#FFB74D', '#81C784']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax3.set_ylabel('Confidence', fontsize=11, fontweight='bold')
ax3.set_title('Confidence Distribution\nComparison', fontsize=13, fontweight='bold', pad=15)
ax3.grid(axis='y', alpha=0.3)

# Add mean lines
means = [df['old_confidence'].mean(), df['new_confidence'].mean()]
for i, mean in enumerate(means):
    ax3.plot([i+0.7, i+1.3], [mean, mean], 'r--', linewidth=2, label='Mean' if i == 0 else '')
ax3.legend(fontsize=9)

# 4. Confidence Change Distribution
ax4 = plt.subplot(2, 3, 4)
conf_change = df['confidence_change']
n, bins, patches = ax4.hist(conf_change, bins=30, color='#9575CD', alpha=0.7, edgecolor='black')
ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='No Change')
ax4.axvline(conf_change.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {conf_change.mean():.3f}')
ax4.set_xlabel('Confidence Change (New - Old)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
ax4.set_title('Confidence Change Distribution', fontsize=13, fontweight='bold', pad=15)
ax4.legend(fontsize=9)
ax4.grid(axis='y', alpha=0.3)

# 5. Prediction Transitions (Sankey-style)
ax5 = plt.subplot(2, 3, 5)
changed = df[~df['same_prediction']]

# Count transitions
transitions = {}
for _, row in changed.iterrows():
    key = f"{row['old_prediction']} → {row['new_prediction']}"
    transitions[key] = transitions.get(key, 0) + 1

if transitions:
    sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
    labels, counts = zip(*sorted_transitions[:10])  # Top 10
    
    y_pos = np.arange(len(labels))
    bars = ax5.barh(y_pos, counts, color='#FF5722', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(labels, fontsize=9)
    ax5.set_xlabel('Count', fontsize=11, fontweight='bold')
    ax5.set_title('Prediction Transitions\n(Changed Predictions)', fontsize=13, fontweight='bold', pad=15)
    ax5.invert_yaxis()
    ax5.grid(axis='x', alpha=0.3)
    
    for bar, count in zip(bars, counts):
        width = bar.get_width()
        ax5.text(width, bar.get_y() + bar.get_height()/2.,
                f' {int(count)}',
                ha='left', va='center', fontsize=10, fontweight='bold')

# 6. Confidence by Prediction Type
ax6 = plt.subplot(2, 3, 6)
same_df = df[df['same_prediction']]
changed_df = df[~df['same_prediction']]

x_labels = ['Same\nPrediction', 'Changed\nPrediction']
old_conf = [same_df['old_confidence'].mean(), changed_df['old_confidence'].mean()]
new_conf = [same_df['new_confidence'].mean(), changed_df['new_confidence'].mean()]

x = np.arange(len(x_labels))
width = 0.35

bars1 = ax6.bar(x - width/2, old_conf, width, label='Old Model', 
                color='#FFB74D', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax6.bar(x + width/2, new_conf, width, label='New Model',
                color='#81C784', alpha=0.7, edgecolor='black', linewidth=1.5)

ax6.set_ylabel('Average Confidence', fontsize=11, fontweight='bold')
ax6.set_title('Confidence by Agreement Type', fontsize=13, fontweight='bold', pad=15)
ax6.set_xticks(x)
ax6.set_xticklabels(x_labels, fontsize=10)
ax6.legend(fontsize=10)
ax6.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Overall title
fig.suptitle('Old vs New SOTA Model Comparison Analysis\n(Testing on Old Model Failures)', 
             fontsize=16, fontweight='bold', y=0.98)

# Add text summary
summary_text = f"""
Summary Statistics:
• Total Samples: {summary['total_samples']}
• Agreement Rate: {summary['same_percentage']:.1f}%
• Old Model Avg Confidence: {summary['old_model_confidence']['mean']:.3f} ± {summary['old_model_confidence']['std']:.3f}
• New Model Avg Confidence: {summary['new_model_confidence']['mean']:.3f} ± {summary['new_model_confidence']['std']:.3f}
• Avg Confidence Change: {summary['confidence_change']['mean']:.3f}
"""

fig.text(0.02, 0.02, summary_text, fontsize=10, family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout(rect=[0, 0.05, 1, 0.96])
plt.savefig('challenge_results/model_comparison_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization to: challenge_results/model_comparison_analysis.png")
plt.show()

# Additional detailed analysis
print("\n" + "="*80)
print("DETAILED ANALYSIS")
print("="*80)

print("\n1. PREDICTION CHANGES WHERE NEW MODEL IS MORE CONFIDENT:")
print("-" * 80)
increased_conf = df[df['confidence_change'] > 0].sort_values('confidence_change', ascending=False)
if len(increased_conf) > 0:
    print(f"   Count: {len(increased_conf)}")
    print("\n   Top 5:")
    for idx, row in increased_conf.head(5).iterrows():
        print(f"   • {row['old_prediction']} → {row['new_prediction']}: "
              f"conf {row['old_confidence']:.3f} → {row['new_confidence']:.3f} "
              f"(Δ{row['confidence_change']:+.3f})")

print("\n2. PREDICTION CHANGES WHERE NEW MODEL IS LESS CONFIDENT:")
print("-" * 80)
decreased_conf = df[df['confidence_change'] < 0].sort_values('confidence_change')
if len(decreased_conf) > 0:
    print(f"   Count: {len(decreased_conf)}")
    print("\n   Top 5:")
    for idx, row in decreased_conf.head(5).iterrows():
        print(f"   • {row['old_prediction']} → {row['new_prediction']}: "
              f"conf {row['old_confidence']:.3f} → {row['new_confidence']:.3f} "
              f"(Δ{row['confidence_change']:+.3f})")

print("\n3. SAME PREDICTIONS WITH BIGGEST CONFIDENCE DROP:")
print("-" * 80)
same_decreased = df[df['same_prediction'] & (df['confidence_change'] < 0)].sort_values('confidence_change')
if len(same_decreased) > 0:
    print(f"   Count: {len(same_decreased)}")
    print("\n   Top 5:")
    for idx, row in same_decreased.head(5).iterrows():
        print(f"   • {row['new_prediction']}: "
              f"conf {row['old_confidence']:.3f} → {row['new_confidence']:.3f} "
              f"(Δ{row['confidence_change']:+.3f})")

print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)

# Calculate key insights
same_rate = summary['same_percentage']
avg_conf_drop = summary['confidence_change']['mean']

if same_rate > 60:
    print("⚠️  HIGH AGREEMENT: New model makes similar predictions to old model (>{:.0f}%)".format(same_rate))
    print("    → This suggests the new model may be learning similar patterns/biases")
elif same_rate > 40:
    print("📊 MODERATE AGREEMENT: New model shows some different behavior ({:.0f}%)".format(same_rate))
    print("    → Partial improvement, but still overlaps with old model errors")
else:
    print("✨ LOW AGREEMENT: New model makes very different predictions (<{:.0f}%)".format(same_rate))
    print("    → Significant change in model behavior")

if avg_conf_drop < -0.15:
    print("\n⚠️  LOWER CONFIDENCE: New model is less confident on average ({:.3f})".format(avg_conf_drop))
    print("    → This could indicate uncertainty, which may be good for ambiguous samples")
elif avg_conf_drop > 0.05:
    print("\n✓ HIGHER CONFIDENCE: New model is more confident on average (+{:.3f})".format(avg_conf_drop))
    print("    → Better feature learning, but verify if justified")
else:
    print("\n➡️  SIMILAR CONFIDENCE: Confidence levels are comparable")

# Check if new model just changed from drone to background
drone_to_bg = len(changed[(changed['old_prediction'] == 'drone') & (changed['new_prediction'] == 'background')])
total_changed = len(changed)
if total_changed > 0 and drone_to_bg / total_changed > 0.5:
    print(f"\n⚠️  WARNING: {drone_to_bg}/{total_changed} changes are 'drone → background'")
    print("    → New model may be over-predicting background class")

print("\n" + "="*80)
