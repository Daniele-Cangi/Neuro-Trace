# complete_validation_analysis.py

"""
Complete the validation analysis from existing scan results.

This script loads the completed scan from checkpoints and completes:
- Results saving
- Component interaction matrix
- Circuit extraction
- Visualizations
- Comparison analysis
"""

import sys
import json
import torch
from pathlib import Path
from dataclasses import asdict

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from neurotrace.discovery import ComponentInteractionMatrix
from neurotrace.datasets import IOIDatasetGenerator
from neurotrace.causal import CircuitExtractor, InterventionType
from neurotrace.causal.vlo_tester import VLOResult
from neurotrace.control import CircuitRegistry
from neurotrace.visualization import MetricsPlotter, CircuitGraphVisualizer


def main():
    print("=" * 80)
    print("NEUROTRACE - COMPLETE VALIDATION ANALYSIS")
    print("=" * 80)
    print()

    # Use the completed validation run
    latest_run = Path("runs/discovery_validation/20251116_120236")

    if not latest_run.exists():
        print(f"❌ Run not found: {latest_run}")
        return

    print(f"Loading results from: {latest_run}")
    print()

    # Load checkpoint 150 (final)
    checkpoint_path = latest_run / "checkpoints" / "scan_checkpoint_150.json"

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        for cp in (latest_run / "checkpoints").glob("*.json"):
            print(f"  - {cp.name}")
        return

    print(f"✓ Loading checkpoint: {checkpoint_path.name}")

    with open(checkpoint_path, 'r') as f:
        results = json.load(f)

    print(f"✓ Loaded {len(results)} scan results")
    print()

    # Filter significant results
    significant_results = [
        r for r in results
        if r['vlo'] > 0.3 and r['faithfulness'] > 0.2
    ]

    print(f"✓ Found {len(significant_results)} significant components (VLO > 0.3)")
    print()

    # ========================================================================
    # Save Full Results
    # ========================================================================

    print("[1/5] Saving results...")

    results_path = latest_run / "scan_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved to {results_path}")
    print()

    # ========================================================================
    # Build Component Interaction Matrix
    # ========================================================================

    print("[2/5] Building component interaction matrix...")

    # For matrix we need ScanResult-like objects, use dicts with attribute access
    class DictObj:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    result_objs = [DictObj(r) for r in results]

    matrix = ComponentInteractionMatrix()
    matrix.build_from_scan_results(result_objs)

    matrix_path = latest_run / "interaction_matrix.json"
    matrix.save(str(matrix_path))

    print(f"✓ Saved to {matrix_path}")

    # Print layer importance
    layer_importance = matrix.get_layer_importance()
    print("\n=== Layer Importance ===")
    for layer_idx in sorted(layer_importance.keys()):
        imp = layer_importance[layer_idx]
        bar = "█" * int(abs(imp) * 5)
        sign = "+" if imp >= 0 else "-"
        print(f"Layer {layer_idx:2d}: {sign}{abs(imp):7.3f}  {bar}")

    print()

    # ========================================================================
    # Extract Circuit
    # ========================================================================

    print("[3/5] Extracting circuit...")

    # Load dataset for examples
    dataset_path = latest_run / "ioi_dataset.json"
    with open(dataset_path, 'r') as f:
        dataset_data = json.load(f)

    # Extract examples list (could be dict with 'examples' key or direct list)
    if isinstance(dataset_data, dict) and 'examples' in dataset_data:
        dataset_json = dataset_data['examples']
    elif isinstance(dataset_data, list):
        dataset_json = dataset_data
    else:
        dataset_json = []

    # Convert to VLOResults
    vlo_results = [
        VLOResult(
            clean_logit_diff=r['clean_logit_diff'],
            intervened_logit_diff=r['intervened_logit_diff'],
            vlo=r['vlo'],
            faithfulness=r['faithfulness'],
            effect_size=r['effect_size'],
            intervention_type=InterventionType.ZERO_ABLATION,
            component_name=r['component_name'],
            num_examples=r['num_examples'],
        )
        for r in significant_results
    ]

    extractor = CircuitExtractor(min_vlo=0.3, min_faithfulness=0.2)
    circuit = extractor.extract_from_vlo_results(
        vlo_results=vlo_results,
        circuit_id="gpt2_ioi_validation_1000",
        model_name="gpt2",
        task_tag="ioi",
        human_label="GPT-2 IOI Circuit (Validation with 1000 examples)",
        description=f"Validation run with 1000 IOI examples - {len(significant_results)} significant components found",
        examples=[ex['text'] for ex in dataset_json[:5]],
    )

    print(f"✓ Circuit extracted: {len(circuit.components)} components")
    print()

    # Save to registry
    registry = CircuitRegistry(db_path=str(latest_run / "circuits.db"))
    registry.upsert(circuit)
    print(f"✓ Saved to registry")
    print()

    # ========================================================================
    # Generate Visualizations
    # ========================================================================

    print("[4/5] Generating visualizations...")

    viz_dir = latest_run / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    plotter = MetricsPlotter(template="plotly_dark")

    # VLO results
    vlo_viz_path = viz_dir / "vlo_results.html"
    all_vlo_results = [VLOResult(
        clean_logit_diff=r['clean_logit_diff'],
        intervened_logit_diff=r['intervened_logit_diff'],
        vlo=r['vlo'],
        faithfulness=r['faithfulness'],
        effect_size=r['effect_size'],
        intervention_type=InterventionType.ZERO_ABLATION,
        component_name=r['component_name'],
        num_examples=r['num_examples'],
    ) for r in results]

    plotter.plot_vlo_results(
        vlo_results=all_vlo_results,
        output_path=str(vlo_viz_path),
        sort_by="vlo",
    )
    print(f"✓ VLO results: {vlo_viz_path}")

    # VLO distribution
    dist_viz_path = viz_dir / "vlo_distribution.html"
    plotter.plot_vlo_distribution(
        vlo_results=all_vlo_results,
        output_path=str(dist_viz_path),
    )
    print(f"✓ VLO distribution: {dist_viz_path}")

    # Circuit graph
    try:
        visualizer = CircuitGraphVisualizer()
        graph_path = viz_dir / "circuit_graph.html"
        visualizer.visualize_circuit(
            circuit=circuit,
            output_path=str(graph_path),
            layout="hierarchical",
            node_color_by="vlo",
        )
        print(f"✓ Circuit graph: {graph_path}")
    except ImportError:
        print("⚠️  Pyvis not installed, skipping circuit graph")

    print()

    # ========================================================================
    # Print Top Components
    # ========================================================================

    print("[5/5] Top 10 components by VLO:")
    print()

    sorted_results = sorted(results, key=lambda r: r['vlo'], reverse=True)
    for i, r in enumerate(sorted_results[:10], 1):
        print(f"  {i:2d}. {r['component_name']:30s}  VLO={r['vlo']:7.3f}  F={r['faithfulness']:6.3f}")

    print()
    print("=" * 80)
    print("✅ VALIDATION ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print(f"Output directory: {latest_run}")
    print(f"Total components: {len(results)}")
    print(f"Significant components: {len(significant_results)}")
    print()

    registry.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
