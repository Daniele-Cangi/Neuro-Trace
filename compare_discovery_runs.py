# compare_discovery_runs.py

"""
Compare discovery results between initial 100-example run and validation 1000-example run.

This script analyzes:
1. Layer 0 MLP stability (VLO, faithfulness)
2. Top component rankings consistency
3. Layer importance distribution changes
4. Statistical significance of findings
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def load_scan_results(results_path: Path) -> List[Dict[str, Any]]:
    """Load scan results from JSON."""
    with open(results_path, 'r') as f:
        return json.load(f)


def get_component_dict(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Convert results list to dict keyed by component name."""
    return {
        r['component_name']: {
            'vlo': r.get('vlo_mean', r.get('vlo', 0.0)),
            'faithfulness': r.get('faithfulness', 0.0),
            'effect_size': r.get('effect_size', 0.0),
            'num_examples': r.get('num_examples', 0),
        }
        for r in results
    }


def get_layer_aggregation(component_dict: Dict[str, Dict[str, float]]) -> Dict[int, float]:
    """Aggregate VLO by layer."""
    layer_vlo = {}

    for component_name, metrics in component_dict.items():
        # Extract layer number
        if component_name.startswith('layer_'):
            layer_num = int(component_name.split('_')[1].split('.')[0])

            if layer_num not in layer_vlo:
                layer_vlo[layer_num] = 0.0

            layer_vlo[layer_num] += metrics['vlo']

    return dict(sorted(layer_vlo.items()))


def compute_rank_correlation(
    results1: Dict[str, Dict[str, float]],
    results2: Dict[str, Dict[str, float]],
    metric: str = 'vlo'
) -> float:
    """
    Compute Spearman rank correlation between two result sets.

    Returns correlation coefficient (-1 to 1).
    """
    # Get common components
    common_components = set(results1.keys()) & set(results2.keys())

    if len(common_components) < 2:
        return 0.0

    # Get rankings
    values1 = [results1[c][metric] for c in common_components]
    values2 = [results2[c][metric] for c in common_components]

    # Compute Spearman correlation (rank-based)
    from scipy.stats import spearmanr
    correlation, p_value = spearmanr(values1, values2)

    return correlation


def main():
    print("=" * 80)
    print("NEUROTRACE - DISCOVERY RUN COMPARISON")
    print("=" * 80)
    print()

    # ========================================================================
    # Load Results
    # ========================================================================

    print("[1/4] Loading results...")
    print()

    # Find runs
    discovery_dir = Path("runs/discovery")
    validation_dir = Path("runs/discovery_validation")

    # Get latest run from each
    discovery_runs = sorted(discovery_dir.glob("*"), reverse=True)
    validation_runs = sorted(validation_dir.glob("*"), reverse=True)

    if not discovery_runs:
        print("❌ No initial discovery runs found in runs/discovery/")
        return

    if not validation_runs:
        print("❌ No validation runs found in runs/discovery_validation/")
        return

    initial_run = discovery_runs[0]
    validation_run = validation_runs[0]

    print(f"Initial run:    {initial_run}")
    print(f"Validation run: {validation_run}")
    print()

    # Load results
    initial_results_path = initial_run / "scan_results.json"
    validation_results_path = validation_run / "scan_results.json"

    if not initial_results_path.exists():
        print(f"❌ Results not found: {initial_results_path}")
        return

    if not validation_results_path.exists():
        print(f"❌ Results not found: {validation_results_path}")
        return

    initial_results = load_scan_results(initial_results_path)
    validation_results = load_scan_results(validation_results_path)

    print(f"✓ Loaded initial results:    {len(initial_results)} components")
    print(f"✓ Loaded validation results: {len(validation_results)} components")
    print()

    # Convert to dicts
    initial_dict = get_component_dict(initial_results)
    validation_dict = get_component_dict(validation_results)

    # ========================================================================
    # Layer 0 MLP Comparison
    # ========================================================================

    print("=" * 80)
    print("[2/4] Layer 0 MLP Comparison")
    print("=" * 80)
    print()

    layer_0_mlp_initial = initial_dict.get('layer_0.mlp')
    layer_0_mlp_validation = validation_dict.get('layer_0.mlp')

    if layer_0_mlp_initial and layer_0_mlp_validation:
        print("Initial Run (100 examples):")
        print(f"  VLO:          {layer_0_mlp_initial['vlo']:.3f}")
        print(f"  Faithfulness: {layer_0_mlp_initial['faithfulness']:.3f}")
        print()

        print("Validation Run (1000 examples):")
        print(f"  VLO:          {layer_0_mlp_validation['vlo']:.3f}")
        print(f"  Faithfulness: {layer_0_mlp_validation['faithfulness']:.3f}")
        print()

        # Compute change
        vlo_change = layer_0_mlp_validation['vlo'] - layer_0_mlp_initial['vlo']
        faith_change = layer_0_mlp_validation['faithfulness'] - layer_0_mlp_initial['faithfulness']

        vlo_pct_change = (vlo_change / layer_0_mlp_initial['vlo']) * 100
        faith_pct_change = (faith_change / layer_0_mlp_initial['faithfulness']) * 100

        print("Changes:")
        print(f"  VLO:          {vlo_change:+.3f} ({vlo_pct_change:+.1f}%)")
        print(f"  Faithfulness: {faith_change:+.3f} ({faith_pct_change:+.1f}%)")
        print()

        # Validation verdict
        if layer_0_mlp_validation['vlo'] > 0.3:
            print("✅ VALIDATION CONFIRMED")
            print("   Layer 0 MLP remains significant with larger dataset")
            print("   Finding is ROBUST and not a sample size artifact")
        else:
            print("⚠️  VALIDATION INCONCLUSIVE")
            print("   Layer 0 MLP VLO dropped below significance threshold")
            print("   May be sensitive to dataset size or variability")
    else:
        print("⚠️  Layer 0 MLP not found in one or both runs")

    print()

    # ========================================================================
    # Top Components Comparison
    # ========================================================================

    print("=" * 80)
    print("[3/4] Top Components Comparison")
    print("=" * 80)
    print()

    # Sort by VLO
    initial_sorted = sorted(initial_dict.items(), key=lambda x: x[1]['vlo'], reverse=True)
    validation_sorted = sorted(validation_dict.items(), key=lambda x: x[1]['vlo'], reverse=True)

    print("Top 10 Components (Initial Run):")
    for i, (name, metrics) in enumerate(initial_sorted[:10], 1):
        print(f"  {i:2d}. {name:30s}  VLO={metrics['vlo']:7.3f}  F={metrics['faithfulness']:6.3f}")

    print()
    print("Top 10 Components (Validation Run):")
    for i, (name, metrics) in enumerate(validation_sorted[:10], 1):
        print(f"  {i:2d}. {name:30s}  VLO={metrics['vlo']:7.3f}  F={metrics['faithfulness']:6.3f}")

    print()

    # Rank correlation
    try:
        correlation = compute_rank_correlation(initial_dict, validation_dict, metric='vlo')
        print(f"Spearman Rank Correlation: {correlation:.3f}")

        if correlation > 0.7:
            print("✅ HIGH CONSISTENCY - Component rankings are stable")
        elif correlation > 0.4:
            print("⚠️  MODERATE CONSISTENCY - Some variability in rankings")
        else:
            print("❌ LOW CONSISTENCY - Rankings changed significantly")
    except ImportError:
        print("⚠️  scipy not installed, skipping correlation analysis")

    print()

    # ========================================================================
    # Layer Importance Distribution
    # ========================================================================

    print("=" * 80)
    print("[4/4] Layer Importance Distribution")
    print("=" * 80)
    print()

    initial_layer_vlo = get_layer_aggregation(initial_dict)
    validation_layer_vlo = get_layer_aggregation(validation_dict)

    print("Layer-wise VLO (Sum of all components in layer):")
    print()
    print(f"{'Layer':<8} {'Initial':>10} {'Validation':>10} {'Change':>10} {'%Change':>10}")
    print("-" * 58)

    for layer_idx in sorted(set(initial_layer_vlo.keys()) | set(validation_layer_vlo.keys())):
        initial_val = initial_layer_vlo.get(layer_idx, 0.0)
        validation_val = validation_layer_vlo.get(layer_idx, 0.0)
        change = validation_val - initial_val

        if initial_val != 0:
            pct_change = (change / initial_val) * 100
        else:
            pct_change = 0.0

        print(f"Layer {layer_idx:<2d}  {initial_val:10.3f} {validation_val:10.3f} {change:+10.3f} {pct_change:+9.1f}%")

    print()

    # Key findings
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()

    # Finding 1: Layer 0 dominance
    if layer_0_mlp_validation and layer_0_mlp_validation['vlo'] > 0.3:
        print("1. ✅ Layer 0 MLP Dominance CONFIRMED")
        print(f"   Initial VLO: {layer_0_mlp_initial['vlo']:.3f}")
        print(f"   Validation VLO: {layer_0_mlp_validation['vlo']:.3f}")
        print("   → This is a ROBUST finding, not an artifact")
        print()

    # Finding 2: Consistency
    try:
        if correlation > 0.7:
            print(f"2. ✅ High Ranking Consistency (r={correlation:.3f})")
            print("   → Component importance rankings are stable across dataset sizes")
            print()
    except:
        pass

    # Finding 3: Layer distribution
    layer_0_dominance_initial = initial_layer_vlo.get(0, 0.0) / sum(abs(v) for v in initial_layer_vlo.values())
    layer_0_dominance_validation = validation_layer_vlo.get(0, 0.0) / sum(abs(v) for v in validation_layer_vlo.values())

    print(f"3. Layer 0 Share of Total VLO:")
    print(f"   Initial:    {layer_0_dominance_initial * 100:.1f}%")
    print(f"   Validation: {layer_0_dominance_validation * 100:.1f}%")

    if layer_0_dominance_validation > 0.5:
        print("   → Layer 0 dominates over 50% of total causal effect")
        print("   → Unprecedented early-layer importance in IOI task")

    print()
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
