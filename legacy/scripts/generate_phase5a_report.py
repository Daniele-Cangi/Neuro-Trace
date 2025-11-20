"""
Generate PHASE5A_LAYER_SWEEP.md Report

Reads phase5a_layer_sweep_results.json and generates a markdown report
with vulnerability rankings and analysis.
"""

import json
from pathlib import Path

# Load results
results_path = Path("phase5a_layer_sweep_results.json")
if not results_path.exists():
    print("ERROR: phase5a_layer_sweep_results.json not found!")
    print("Run layer_vulnerability_sweep.py first.")
    exit(1)

with open(results_path) as f:
    data = json.load(f)

layer_results = data["layer_results"]
baseline = data["baseline"]
config = data["config"]

# Sort by test accuracy change (most vulnerable first)
sorted_results = sorted(layer_results, key=lambda x: x['test']['accuracy_change'])

# Generate markdown
md = []

md.append("# Phase 5A: Layer Vulnerability Sweep")
md.append("")
md.append(f"**Date**: {data['timestamp'][:10]}")
md.append(f"**Total Time**: {data['total_time_seconds']:.1f}s ({data['total_time_seconds']/60:.1f} min)")
md.append("")

md.append("## Overview")
md.append("")
md.append("Trained adversarial steering vectors for all 12 layers using Phase 4B method.")
md.append("Each layer was optimized independently to minimize IOI logit difference on borderline examples.")
md.append("")

md.append("## Configuration")
md.append("")
md.append(f"- **Training examples**: {config['dataset_size']}")
md.append(f"- **Test examples**: {config['test_size']}")
md.append(f"- **Borderline threshold**: {config['borderline_threshold']}")
md.append(f"- **Epochs per layer**: {config['num_epochs']}")
md.append(f"- **Learning rate**: {config['learning_rate']}")
md.append(f"- **L2 regularization**: λ={config['lambda_reg']}")
md.append("")

md.append("## Baseline Performance")
md.append("")
md.append(f"- **Train**: {baseline['train']['logit_diff']:.3f} ({baseline['train']['accuracy']*100:.1f}% accuracy)")
md.append(f"- **Test**: {baseline['test']['logit_diff']:.3f} ({baseline['test']['accuracy']*100:.1f}% accuracy)")
md.append("")

md.append("## Layer Vulnerability Results")
md.append("")
md.append("### Summary Table")
md.append("")
md.append("| Rank | Layer | ||δ|| | Train Δacc | Test Δacc | Train Δdiff | Test Δdiff |")
md.append("|------|-------|--------|------------|-----------|-------------|------------|")

for rank, result in enumerate(sorted_results, 1):
    layer = result['layer']
    delta_norm = result['delta_norm']
    train_acc_change = result['train']['accuracy_change']
    test_acc_change = result['test']['accuracy_change']
    train_effect = result['train']['effect']
    test_effect = result['test']['effect']

    md.append(f"| {rank:2d} | {layer:2d} | {delta_norm:.2f} | {train_acc_change:+.1f}% | {test_acc_change:+.1f}% | {train_effect:+.3f} | {test_effect:+.3f} |")

md.append("")

md.append("### Detailed Results by Layer")
md.append("")

for result in sorted(layer_results, key=lambda x: x['layer']):
    layer = result['layer']
    md.append(f"#### Layer {layer}")
    md.append("")
    md.append(f"- **Delta norm**: {result['delta_norm']:.3f}")
    md.append(f"- **Training time**: {result['training_time_seconds']:.1f}s")
    md.append("")

    md.append("**Train Set**:")
    md.append(f"- Baseline: {result['train']['baseline_logit_diff']:.3f} ({result['train']['baseline_accuracy']*100:.1f}%)")
    md.append(f"- Steered: {result['train']['steered_logit_diff']:.3f} ({result['train']['steered_accuracy']*100:.1f}%)")
    md.append(f"- Effect: {result['train']['effect']:+.3f}")
    md.append(f"- Accuracy change: {result['train']['accuracy_change']:+.1f}%")
    md.append("")

    md.append("**Test Set**:")
    md.append(f"- Baseline: {result['test']['baseline_logit_diff']:.3f} ({result['test']['baseline_accuracy']*100:.1f}%)")
    md.append(f"- Steered: {result['test']['steered_logit_diff']:.3f} ({result['test']['steered_accuracy']*100:.1f}%)")
    md.append(f"- Effect: {result['test']['effect']:+.3f}")
    md.append(f"- Accuracy change: {result['test']['accuracy_change']:+.1f}%")
    md.append("")

md.append("## Analysis")
md.append("")

# Find most/least vulnerable
most_vulnerable = sorted_results[0]
least_vulnerable = sorted_results[-1]

md.append(f"### Most Vulnerable Layer: **Layer {most_vulnerable['layer']}**")
md.append("")
md.append(f"- Test accuracy drop: **{most_vulnerable['test']['accuracy_change']:+.1f}%**")
md.append(f"- Test effect: **{most_vulnerable['test']['effect']:+.3f}**")
md.append(f"- Delta norm: {most_vulnerable['delta_norm']:.3f}")
md.append("")

md.append(f"### Least Vulnerable Layer: **Layer {least_vulnerable['layer']}**")
md.append("")
md.append(f"- Test accuracy drop: **{least_vulnerable['test']['accuracy_change']:+.1f}%**")
md.append(f"- Test effect: **{least_vulnerable['test']['effect']:+.3f}**")
md.append(f"- Delta norm: {least_vulnerable['delta_norm']:.3f}")
md.append("")

# Calculate statistics
test_acc_changes = [r['test']['accuracy_change'] for r in layer_results]
import numpy as np
mean_acc_change = np.mean(test_acc_changes)
std_acc_change = np.std(test_acc_changes)

md.append("### Statistics")
md.append("")
md.append(f"- Mean test accuracy change: {mean_acc_change:.1f}% ± {std_acc_change:.1f}%")
md.append(f"- Range: [{min(test_acc_changes):.1f}%, {max(test_acc_changes):.1f}%]")
md.append(f"- Vulnerability spread: {max(test_acc_changes) - min(test_acc_changes):.1f}%")
md.append("")

# Layer groups
early_layers = [r for r in layer_results if r['layer'] < 4]
mid_layers = [r for r in layer_results if 4 <= r['layer'] < 8]
late_layers = [r for r in layer_results if r['layer'] >= 8]

early_mean = np.mean([r['test']['accuracy_change'] for r in early_layers])
mid_mean = np.mean([r['test']['accuracy_change'] for r in mid_layers])
late_mean = np.mean([r['test']['accuracy_change'] for r in late_layers])

md.append("### Layer Groups")
md.append("")
md.append(f"- **Early layers (0-3)**: {early_mean:.1f}% average accuracy drop")
md.append(f"- **Mid layers (4-7)**: {mid_mean:.1f}% average accuracy drop")
md.append(f"- **Late layers (8-11)**: {late_mean:.1f}% average accuracy drop")
md.append("")

md.append("## Key Findings")
md.append("")

# Determine pattern
if late_mean < early_mean and late_mean < mid_mean:
    md.append("- **Late layers (8-11) are MOST vulnerable** to adversarial steering")
elif early_mean < mid_mean and early_mean < late_mean:
    md.append("- **Early layers (0-3) are MOST vulnerable** to adversarial steering")
else:
    md.append("- **Mid layers (4-7) are MOST vulnerable** to adversarial steering")

md.append("")

# Check if vulnerability correlates with layer depth
from scipy.stats import pearsonr
layers_num = [r['layer'] for r in layer_results]
test_changes = [r['test']['accuracy_change'] for r in layer_results]
corr, p_value = pearsonr(layers_num, test_changes)

md.append(f"- Correlation between layer depth and vulnerability: r={corr:.3f} (p={p_value:.3f})")

if abs(corr) > 0.5 and p_value < 0.05:
    if corr < 0:
        md.append("  - **Significant negative correlation**: Deeper layers are MORE vulnerable")
    else:
        md.append("  - **Significant positive correlation**: Shallower layers are MORE vulnerable")
else:
    md.append("  - No significant linear correlation with layer depth")

md.append("")

md.append("## Implications")
md.append("")
md.append("1. **Layer-specific vulnerability** varies significantly across the network")
md.append(f"2. **Vulnerability range**: {max(test_acc_changes) - min(test_acc_changes):.1f}% spread between most/least vulnerable layers")
md.append("3. **Strategic targeting**: Adversarial attacks should focus on most vulnerable layers")
md.append("4. **Defensive priorities**: Protect most vulnerable layers with countermeasures")
md.append("")

md.append("## Checkpoints")
md.append("")
md.append("Adversarial deltas saved for all layers:")
md.append("")
for i in range(12):
    md.append(f"- `checkpoints/adversarial_delta_layer{i}.pt`")
md.append("")

md.append("---")
md.append("")
md.append("**Next Steps**: Multi-layer delta combinations, defensive steering, layer-wise analysis")

# Write report
output_path = Path("PHASE5A_LAYER_SWEEP.md")
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md))

print(f"Report generated: {output_path}")
print()
print("Summary:")
print(f"  Most vulnerable: Layer {most_vulnerable['layer']} ({most_vulnerable['test']['accuracy_change']:+.1f}%)")
print(f"  Least vulnerable: Layer {least_vulnerable['layer']} ({least_vulnerable['test']['accuracy_change']:+.1f}%)")
print(f"  Vulnerability range: {max(test_acc_changes) - min(test_acc_changes):.1f}%")
