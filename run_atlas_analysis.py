# run_atlas_analysis.py
"""
Complete Atlas Analysis Pipeline

Executes the full research workflow for which the Neural Atlas was designed:
1. Load all 12 trained SAEs
2. Extract cross-layer feature patterns
3. Discover multi-layer circuits
4. Generate interactive visualizations
5. Save comprehensive results

Uses ONLY existing neurotrace modules.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from neurotrace.control import EnhancedSAEFeatureStore, CircuitRegistry
from neurotrace.visualization.circuit_graph import CircuitGraphVisualizer
from neurotrace.visualization.activation_explorer import ActivationExplorer, DimReductionMethod

print("=" * 80)
print("NEURAL ATLAS - COMPLETE ANALYSIS PIPELINE")
print("=" * 80)
print()
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print()

# ============================================================================
# STEP 1: LOAD ATLAS
# ============================================================================
print("=" * 80)
print("STEP 1: LOAD 12-LAYER NEURAL ATLAS")
print("=" * 80)
print()

feature_store = EnhancedSAEFeatureStore()
atlas_dir = Path("checkpoints/all_layers_sae")

if not atlas_dir.exists():
    print("ERROR: Atlas not found!")
    print(f"Expected: {atlas_dir}")
    print("Run: python train_atlas_simple.py --layers all")
    sys.exit(1)

loaded_layers = []
for layer_idx in range(12):
    checkpoint_path = atlas_dir / f"layer_{layer_idx}" / "final.pt"
    if checkpoint_path.exists():
        feature_store.load_sae(str(checkpoint_path), layer=layer_idx, device=device)
        loaded_layers.append(layer_idx)

print(f"Loaded SAEs: {len(loaded_layers)}/12 layers")
print(f"Total features: {len(loaded_layers) * 3072}")
print()

if len(loaded_layers) < 12:
    print("WARNING: Not all layers available")
    print(f"Missing: {[i for i in range(12) if i not in loaded_layers]}")
    print()

# ============================================================================
# STEP 2: CROSS-LAYER FEATURE ANALYSIS
# ============================================================================
print("=" * 80)
print("STEP 2: CROSS-LAYER FEATURE ANALYSIS")
print("=" * 80)
print()

# Load activations
activations_dir = Path("runs/deep_ioi_capture/20251116_171258/activations")
if not activations_dir.exists():
    print("WARNING: No activation data found, skipping feature analysis")
    layer_activations = {}
else:
    batch_files = list(activations_dir.glob("batch_*.pt"))
    if not batch_files:
        print("WARNING: No batch files found, skipping feature analysis")
        layer_activations = {}
    else:
        batch_file = batch_files[0]
        batch_data = torch.load(batch_file, map_location=device)
        num_samples = min(100, len(next(iter(batch_data.values()))))

        print(f"Using {num_samples} samples from: {batch_file.name}")
        print()

        layer_activations = {}
        for layer_idx in loaded_layers:
            layer_name = f"layer_{layer_idx}.mlp"
            if layer_name not in batch_data:
                continue

            sae = feature_store.saes.get(layer_idx)
            if sae is None:
                continue

            # Get activations
            acts = batch_data[layer_name][:num_samples]

            # Forward through SAE
            with torch.no_grad():
                output = sae.forward(acts)
                codes = output['codes']

            # Store statistics
            mean_activation = codes.mean(dim=0).cpu()
            max_activation = codes.max(dim=0)[0].cpu()

            layer_activations[layer_idx] = {
                'mean': mean_activation,
                'max': max_activation,
                'sparsity': (codes > 0.1).float().mean().item(),
            }

            # Find top features
            top_k = 10
            top_values, top_indices = torch.topk(mean_activation, k=top_k)

            print(f"Layer {layer_idx:2d}:")
            print(f"  Top feature activation: {top_values[0].item():.2f}")
            print(f"  Sparsity: {layer_activations[layer_idx]['sparsity']*100:.1f}% active")
            print(f"  Top 5 features: {top_indices[:5].tolist()}")
            print()

# ============================================================================
# STEP 3: CIRCUIT DISCOVERY & REGISTRY
# ============================================================================
print("=" * 80)
print("STEP 3: CIRCUIT DISCOVERY & REGISTRY")
print("=" * 80)
print()

circuits_dir = Path("circuits")
circuits_dir.mkdir(exist_ok=True)

registry_path = circuits_dir / "atlas_circuits.db"
if registry_path.exists():
    registry = CircuitRegistry(db_path=str(registry_path))
    circuits = registry.list()

    print(f"Circuit Registry: {registry_path}")
    print(f"Total circuits: {len(circuits)}")
    print()

    for circuit in circuits:
        print(f"Circuit: {circuit.circuit_id}")
        print(f"  Task: {circuit.semantics.task_tag}")
        print(f"  Layers: {sorted([c.layer for c in circuit.components])}")
        print(f"  Components: {len(circuit.components)}")
        print(f"  VLO: {circuit.causal_metrics.vlo_mean:.3f}")
        print(f"  SAE features: {sum(len(v) for v in circuit.features.sae_indices.values())}")
        print()
else:
    print("No circuit registry found")
    print("Circuits can be discovered and saved using:")
    print("  - neurotrace.causal.VLOTester for component testing")
    print("  - neurotrace.causal.CircuitExtractor for circuit building")
    circuits = []
    print()

# ============================================================================
# STEP 4: INTERACTIVE VISUALIZATIONS
# ============================================================================
print("=" * 80)
print("STEP 4: GENERATE INTERACTIVE VISUALIZATIONS")
print("=" * 80)
print()

viz_dir = Path("visualizations")
viz_dir.mkdir(exist_ok=True)

# Check if visualization dependencies are available
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    print("WARNING: pyvis not installed, skipping circuit graphs")
    print("Install with: pip install pyvis")
    print()

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("WARNING: plotly not installed, skipping 3D plots")
    print("Install with: pip install plotly")
    print()

# Generate circuit visualizations
if PYVIS_AVAILABLE and circuits:
    viz = CircuitGraphVisualizer()

    for circuit in circuits:
        output_path = viz_dir / f"{circuit.circuit_id}_graph.html"
        try:
            viz.visualize_circuit(
                circuit,
                output_path=output_path,
                layout="hierarchical"
            )
            print(f"[OK] Circuit graph: {output_path.name}")
        except Exception as e:
            print(f"[ERROR] Failed to visualize {circuit.circuit_id}: {e}")
    print()

# Generate activation visualizations
if PLOTLY_AVAILABLE and layer_activations:
    explorer = ActivationExplorer()

    # Collect layer-level statistics for visualization
    # Each layer = 1 point with aggregated features
    layer_features = []
    layer_names = []

    for layer_idx in sorted(layer_activations.keys()):
        # Use mean activation vector as layer representation
        mean_act = layer_activations[layer_idx]['mean'].numpy()  # [3072]
        layer_features.append(mean_act)
        layer_names.append(f"Layer {layer_idx}")

    if len(layer_features) >= 3:  # Need at least 3 points for 3D
        features_array = np.vstack(layer_features)  # [12, 3072]

        # PCA visualization
        try:
            output_path = viz_dir / "layer_features_pca_3d.html"
            fig = explorer.plot_activations_3d(
                features_array,
                labels=layer_names,
                method=DimReductionMethod.PCA,
                title="Layer Feature Representations (PCA 3D)",
                output_path=str(output_path),
                show=False
            )
            print(f"[OK] 3D PCA plot: {output_path.name}")
        except Exception as e:
            print(f"[ERROR] PCA visualization failed: {e}")
    else:
        print(f"[SKIP] Need at least 3 layers for 3D visualization (have {len(layer_features)})")
    print()

# ============================================================================
# STEP 5: GENERATE ANALYSIS REPORT
# ============================================================================
print("=" * 80)
print("STEP 5: GENERATE ANALYSIS REPORT")
print("=" * 80)
print()

report = {
    "timestamp": datetime.now().isoformat(),
    "atlas": {
        "layers_loaded": len(loaded_layers),
        "total_features": len(loaded_layers) * 3072,
        "device": device,
    },
    "feature_analysis": {
        "samples_analyzed": num_samples if layer_activations else 0,
        "layers_analyzed": len(layer_activations),
        "layer_stats": {
            f"layer_{idx}": {
                "mean_activation": float(stats['mean'].mean().item()),
                "max_activation": float(stats['max'].max().item()),
                "sparsity_pct": float(stats['sparsity'] * 100),
            }
            for idx, stats in layer_activations.items()
        }
    },
    "circuits": {
        "total_discovered": len(circuits),
        "circuits": [
            {
                "id": c.circuit_id,
                "task": c.semantics.task_tag,
                "layers": sorted([comp.layer for comp in c.components]),
                "vlo": float(c.causal_metrics.vlo_mean),
            }
            for c in circuits
        ]
    },
    "visualizations": {
        "circuit_graphs": len([f for f in viz_dir.glob("*_graph.html")]) if viz_dir.exists() else 0,
        "activation_plots": len([f for f in viz_dir.glob("activations_*.html")]) if viz_dir.exists() else 0,
    }
}

report_path = Path("atlas_analysis_report.json")
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"Analysis report: {report_path}")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()
print("Results:")
print(f"  - Atlas layers loaded: {len(loaded_layers)}/12")
print(f"  - Features analyzed: {len(layer_activations)} layers")
print(f"  - Circuits discovered: {len(circuits)}")
print(f"  - Visualizations: {viz_dir}/" if viz_dir.exists() else "  - Visualizations: none")
print(f"  - Report: {report_path}")
print()
print("Next steps:")
print("  - View circuit graphs in visualizations/*.html")
print("  - Explore 3D activation plots")
print("  - Use circuits for downstream tasks (steering, ablation)")
print()
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
