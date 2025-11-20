"""
Generate PHASE5B_SPARSE_SAE_VIRUS.md Report

Reads phase5b_sparse_sae_virus_results.json and generates a markdown report
with sparsity-performance tradeoff analysis.
"""

import json
import numpy as np
from pathlib import Path

# Load results
results_path = Path("phase5b_sparse_sae_virus_results.json")
if not results_path.exists():
    print("ERROR: phase5b_sparse_sae_virus_results.json not found!")
    print("Run learn_sparse_sae_virus.py first.")
    exit(1)

with open(results_path) as f:
    data = json.load(f)

results = data["results"]
baseline = data["baseline"]
config = data["config"]

# Generate markdown
md = []

md.append("# Phase 5B: Sparse SAE Virus")
md.append("")
md.append(f"**Date**: {data['timestamp'][:10]}")
md.append(f"**Total Time**: {data['total_time_seconds']:.1f}s ({data['total_time_seconds']/60:.1f} min)")
md.append("")

md.append("## Overview")
md.append("")
md.append("Learned adversarial steering vectors **directly in SAE feature space** (α coefficients)")
md.append("with L1 regularization to enforce sparsity. This tests whether sparse, interpretable")
md.append("feature combinations can achieve the same adversarial power as dense residual vectors.")
md.append("")

md.append("## Configuration")
md.append("")
md.append(f"- **Target layer**: {config['target_layer']}")
md.append(f"- **Training examples**: {config['dataset_size']}")
md.append(f"- **Test examples**: {config['test_size']}")
md.append(f"- **Borderline threshold**: {config['borderline_threshold']}")
md.append(f"- **Epochs**: {config['num_epochs']}")
md.append(f"- **Learning rate**: {config['learning_rate']}")
md.append(f"- **L2 regularization**: λ₂={config['lambda_l2']}")
md.append(f"- **L1 values tested**: {config['lambda_l1_values']}")
md.append("")

md.append("## Baseline Performance")
md.append("")
md.append(f"- **Train**: {baseline['train']['logit_diff']:.3f} ({baseline['train']['accuracy']*100:.1f}% accuracy)")
md.append(f"- **Test**: {baseline['test']['logit_diff']:.3f} ({baseline['test']['accuracy']*100:.1f}% accuracy)")
md.append("")

md.append("## Sparsity-Performance Tradeoff")
md.append("")
md.append("### Summary Table")
md.append("")
md.append("| λ₁ | #Active | % Active | ‖δ‖ | ‖α‖₁ | Test Δacc | Test Δdiff |")
md.append("|---------|---------|----------|--------|--------|-----------|------------|")

for result in results:
    lambda_l1 = result['lambda_l1']
    num_active = result['num_active']

    # Get n_features from first result's alpha shape or infer from data
    # For now, assume 6144 features (dict_mult=8 * 768)
    n_features = 6144
    pct_active = (num_active / n_features) * 100 if n_features > 0 else 0

    delta_norm = result['delta_norm']
    alpha_l1 = result['alpha_l1']
    test_acc_change = result['test']['accuracy_change']
    test_effect = result['test']['effect']

    md.append(f"| {lambda_l1:.0e} | {num_active:>7d} | {pct_active:>7.1f}% | {delta_norm:>6.3f} | {alpha_l1:>6.1f} | {test_acc_change:>+9.1f}% | {test_effect:>+10.3f} |")

md.append("")

md.append("### Detailed Results")
md.append("")

for result in results:
    lambda_l1 = result['lambda_l1']
    md.append(f"#### λ₁ = {lambda_l1:.0e}")
    md.append("")
    md.append(f"- **Active features**: {result['num_active']} ({result['num_active']/6144*100:.1f}%)")
    md.append(f"- **Delta norm**: {result['delta_norm']:.3f}")
    md.append(f"- **Alpha L1**: {result['alpha_l1']:.1f}")
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

# Find best performance (no sparsity constraint)
best_performance = min(results, key=lambda x: x['test']['accuracy_change'])

md.append(f"### Best Overall Performance: λ₁={best_performance['lambda_l1']:.0e}")
md.append("")
md.append(f"- **Test accuracy drop**: {best_performance['test']['accuracy_change']:+.1f}%")
md.append(f"- **Test effect**: {best_performance['test']['effect']:+.3f}")
md.append(f"- **Active features**: {best_performance['num_active']} ({best_performance['num_active']/6144*100:.1f}%)")
md.append(f"- **Delta norm**: {best_performance['delta_norm']:.3f}")
md.append("")

# Find sparsest strong result
strong_results = [r for r in results if r['test']['accuracy_change'] <= -40]
if strong_results:
    sparsest_strong = min(strong_results, key=lambda x: x['num_active'])
    md.append(f"### Sparsest Strong Attack: λ₁={sparsest_strong['lambda_l1']:.0e}")
    md.append("")
    md.append(f"- **Test accuracy drop**: {sparsest_strong['test']['accuracy_change']:+.1f}%")
    md.append(f"- **Active features**: {sparsest_strong['num_active']} ({sparsest_strong['num_active']/6144*100:.1f}%)")
    md.append(f"- **Delta norm**: {sparsest_strong['delta_norm']:.3f}")
    md.append("")
else:
    md.append("### Sparsest Strong Attack")
    md.append("")
    md.append("**None found**: No λ₁ setting achieved ≤-40% accuracy drop")
    md.append("")

# Check for <100 features with strong effect
ultra_sparse = [r for r in results if r['num_active'] < 100 and r['test']['accuracy_change'] <= -40]
if ultra_sparse:
    md.append(f"### ✅ Ultra-Sparse Attack Found")
    md.append("")
    md.append(f"**{len(ultra_sparse)} setting(s) achieved strong attack (<100 features, ≤-40% drop)**:")
    md.append("")
    for r in ultra_sparse:
        md.append(f"- λ₁={r['lambda_l1']:.0e}: **{r['num_active']} features** → **{r['test']['accuracy_change']:+.1f}%** accuracy drop")
    md.append("")
else:
    md.append("### ❌ Ultra-Sparse Attack Not Achieved")
    md.append("")
    md.append("**No setting achieved ≤-40% drop with <100 features**")
    md.append("")
    # Find closest attempt
    sparse_results = [r for r in results if r['num_active'] < 100]
    if sparse_results:
        closest = min(sparse_results, key=lambda x: x['test']['accuracy_change'])
        md.append(f"Closest attempt: {closest['num_active']} features → {closest['test']['accuracy_change']:+.1f}% drop")
    md.append("")

# Sparsity statistics
test_drops = [r['test']['accuracy_change'] for r in results]
num_actives = [r['num_active'] for r in results]

md.append("### Sparsity Statistics")
md.append("")
md.append(f"- **Accuracy drop range**: [{min(test_drops):.1f}%, {max(test_drops):.1f}%]")
md.append(f"- **Active features range**: [{min(num_actives)}, {max(num_actives)}]")
md.append("")

# Correlation analysis
from scipy.stats import pearsonr
if len(num_actives) > 2:
    corr, p_value = pearsonr(num_actives, test_drops)
    md.append(f"- **Correlation** (# features vs accuracy drop): r={corr:.3f} (p={p_value:.3f})")
    if p_value < 0.05:
        if corr > 0:
            md.append("  - Significant positive correlation: More features → Less attack power")
        else:
            md.append("  - Significant negative correlation: More features → More attack power")
    md.append("")

md.append("## Key Findings")
md.append("")

# Determine main finding
if ultra_sparse:
    md.append(f"1. **✅ Sparse adversarial control is possible**: Achieved {ultra_sparse[0]['test']['accuracy_change']:+.1f}% drop with only {ultra_sparse[0]['num_active']} features")
    md.append(f"2. **Interpretability breakthrough**: Ultra-sparse attacks enable feature-level analysis")
elif strong_results:
    md.append(f"1. **⚠️ Strong attacks require moderate sparsity**: Best result uses {sparsest_strong['num_active']} features ({sparsest_strong['num_active']/6144*100:.1f}%)")
    md.append(f"2. **Gap between sparse and dense**: Ultra-sparse (<100 features) insufficient for strong control")
else:
    md.append(f"1. **❌ Sparse attacks are weak**: Even with {best_performance['num_active']} features, only {best_performance['test']['accuracy_change']:+.1f}% drop achieved")
    md.append(f"2. **Dense control required**: Confirms Phase 4B finding that IOI control is fundamentally dense")

md.append("")

md.append("## Comparison with Phase 4B")
md.append("")

md.append("| Method | Space | Features/Dims | Effect | Accuracy Δ |")
md.append("|--------|-------|---------------|--------|------------|")
md.append("| Phase 4B | Residual δ | 768 dims | -4.180 | -60.4% |")
md.append("| Phase 4B-B | Top-200 SAE | 200 features | -0.389 | -1.4% |")
md.append(f"| **Phase 5B (best)** | **SAE α (L1)** | **{best_performance['num_active']} features** | **{best_performance['test']['effect']:+.3f}** | **{best_performance['test']['accuracy_change']:+.1f}%** |")

md.append("")

if best_performance['test']['accuracy_change'] < -50:
    md.append("**Phase 5B achieves comparable performance to Phase 4B** by directly optimizing in SAE space with sparsity constraints.")
elif best_performance['test']['accuracy_change'] < -30:
    md.append("**Phase 5B achieves moderate success** but falls short of Phase 4B's full adversarial power.")
else:
    md.append("**Phase 5B validates Phase 4B findings**: Dense residual control is more effective than sparse SAE features.")

md.append("")

md.append("## Implications")
md.append("")

if ultra_sparse:
    md.append(f"1. **Sparse interpretability is viable**: {ultra_sparse[0]['num_active']} features sufficient for strong adversarial control")
    md.append("2. **Feature-level mechanistic understanding**: Can now analyze specific features driving attack")
    md.append("3. **Defensive priorities**: Protect these specific features to prevent attacks")
else:
    md.append("1. **Sparsity-performance tradeoff exists**: Stronger attacks require more features")
    md.append("2. **Interpretability limit**: Cannot achieve Phase 4B power with ultra-sparse features")
    md.append("3. **Dense mechanisms dominate**: IOI control fundamentally requires distributed features")

md.append("")

md.append("## Checkpoints")
md.append("")
md.append("Sparse SAE virus checkpoints saved:")
md.append("")
for result in results:
    lambda_l1 = result['lambda_l1']
    md.append(f"- `checkpoints/sparse_sae_virus_layer{config['target_layer']}_l1{lambda_l1:.0e}.pt`")
md.append("")

md.append("---")
md.append("")
md.append("**Next Steps**: Analyze top features in best sparse model, multi-layer sparse attacks, defensive steering")

# Write report
output_path = Path("PHASE5B_SPARSE_SAE_VIRUS.md")
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md))

print(f"Report generated: {output_path}")
print()
print("Summary:")
print(f"  Best performance: λ₁={best_performance['lambda_l1']:.0e}, {best_performance['num_active']} features, {best_performance['test']['accuracy_change']:+.1f}%")
if strong_results:
    print(f"  Sparsest strong: λ₁={sparsest_strong['lambda_l1']:.0e}, {sparsest_strong['num_active']} features, {sparsest_strong['test']['accuracy_change']:+.1f}%")
if ultra_sparse:
    print(f"  ✅ Ultra-sparse attack achieved: {ultra_sparse[0]['num_active']} features")
else:
    print(f"  ❌ Ultra-sparse attack not achieved (<100 features, ≤-40% drop)")
print()
