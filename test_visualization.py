"""
Test per Visualization Module.

Verifica:
1. CircuitGraphVisualizer: grafi interattivi con Pyvis
2. MetricsPlotter: plot metriche training e VLO
3. ActivationExplorer: PCA/t-SNE/UMAP 2D/3D
4. SAEFeatureAnalyzer: analisi feature SAE
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
import numpy as np

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from neurotrace.visualization import (
    CircuitGraphVisualizer,
    MetricsPlotter,
    TrainingMetricsPlot,
    VLOMetricsPlot,
    ActivationExplorer,
    DimReductionMethod,
    SAEFeatureAnalyzer,
)
from neurotrace.control.circuit_registry import (
    CircuitRecord,
    CircuitComponent,
    CircuitCausalMetrics,
    CircuitSemantics,
)
from neurotrace.causal.vlo_tester import VLOResult, InterventionType
from neurotrace.state_indexer.sae_feature_extractor import LayerSparseAutoencoder


# ============================================================================
# Test Circuit Graph Visualizer
# ============================================================================


def test_circuit_graph_visualizer():
    """Test Pyvis circuit graph visualization."""
    print("\n" + "=" * 70)
    print("TEST 1: Circuit Graph Visualizer")
    print("=" * 70)

    # Check if pyvis available
    try:
        from pyvis.network import Network
    except ImportError:
        print("⚠️  pyvis not installed, skipping test")
        print("   Install with: pip install pyvis")
        return

    # Create mock circuit
    components = [
        CircuitComponent(
            layer_idx=7,
            component_type="attention_head",
            component_idx=9,
            component_name="layer_7.attention_head.9",
            vlo=1.2,
            faithfulness=0.6,
        ),
        CircuitComponent(
            layer_idx=9,
            component_type="attention_head",
            component_idx=9,
            component_name="layer_9.attention_head.9",
            vlo=2.5,
            faithfulness=0.85,
        ),
        CircuitComponent(
            layer_idx=10,
            component_type="mlp",
            component_idx=0,
            component_name="layer_10.mlp.0",
            vlo=1.8,
            faithfulness=0.72,
        ),
    ]

    circuit = CircuitRecord(
        circuit_id="test_ioi_circuit",
        model_name="gpt2",
        components=components,
        causal_metrics=CircuitCausalMetrics(
            vlo_mean=1.83,
            vlo_std=0.53,
            faithfulness=0.72,
            num_examples=10,
        ),
        semantics=CircuitSemantics(
            task_tag="ioi",
            human_label="IOI Test Circuit",
            description="Test circuit for visualization",
            example_prompts=["When Alice and Bob went to the store, Alice gave"],
        ),
    )

    print(f"✓ Created mock circuit: {circuit.circuit_id}")
    print(f"  Components: {len(circuit.components)}")

    # Create visualizer
    visualizer = CircuitGraphVisualizer(
        height="750px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
    )
    print(f"✓ Created CircuitGraphVisualizer")

    # Visualize single circuit (hierarchical)
    output_dir = Path(tempfile.mkdtemp(prefix="test_viz_"))
    output_path_hier = output_dir / "circuit_hierarchical.html"

    visualizer.visualize_circuit(
        circuit=circuit,
        output_path=output_path_hier,
        layout="hierarchical",
        node_color_by="vlo",
    )
    print(f"✓ Generated hierarchical graph: {output_path_hier}")
    assert output_path_hier.exists()

    # Visualize single circuit (physics)
    output_path_phys = output_dir / "circuit_physics.html"
    visualizer.visualize_circuit(
        circuit=circuit,
        output_path=output_path_phys,
        layout="physics",
        node_color_by="faithfulness",
    )
    print(f"✓ Generated physics graph: {output_path_phys}")
    assert output_path_phys.exists()

    # Visualize multi-circuits
    circuit2 = CircuitRecord(
        circuit_id="test_circuit_2",
        model_name="gpt2",
        components=[
            components[0],  # Shared component
            components[2],  # Shared component
        ],
        causal_metrics=CircuitCausalMetrics(
            vlo_mean=1.5,
            vlo_std=0.3,
            faithfulness=0.66,
            num_examples=10,
        ),
        semantics=CircuitSemantics(
            task_tag="test",
            human_label="Test Circuit 2",
        ),
    )

    output_path_multi = output_dir / "circuit_multi.html"
    visualizer.visualize_multi_circuits(
        circuits=[circuit, circuit2],
        output_path=output_path_multi,
        merge_mode="union",
    )
    print(f"✓ Generated multi-circuit graph: {output_path_multi}")
    assert output_path_multi.exists()

    # Cleanup
    import shutil
    shutil.rmtree(output_dir, ignore_errors=True)

    print("\n✅ Circuit Graph Visualizer tests PASSED")


# ============================================================================
# Test Metrics Plotter
# ============================================================================


def test_metrics_plotter():
    """Test Plotly metrics plotting."""
    print("\n" + "=" * 70)
    print("TEST 2: Metrics Plotter")
    print("=" * 70)

    # Check if plotly available
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("⚠️  plotly not installed, skipping test")
        print("   Install with: pip install plotly")
        return

    plotter = MetricsPlotter(template="plotly_dark", width=1200, height=600)
    print(f"✓ Created MetricsPlotter")

    output_dir = Path(tempfile.mkdtemp(prefix="test_metrics_"))

    # 1. Training metrics
    steps = list(range(0, 1000, 10))
    total_loss = [1.0 * np.exp(-i / 300) + 0.01 for i in steps]
    mse_loss = [0.7 * np.exp(-i / 300) + 0.005 for i in steps]
    l1_loss = [0.3 * np.exp(-i / 300) + 0.005 for i in steps]
    sparsity = [20 + 30 * (1 - np.exp(-i / 200)) for i in steps]
    lr = [3e-4 * (0.1 + 0.9 * np.cos(np.pi * i / 1000)) for i in steps]

    training_metrics = TrainingMetricsPlot(
        steps=steps,
        total_loss=total_loss,
        mse_loss=mse_loss,
        l1_loss=l1_loss,
        sparsity=sparsity,
        learning_rate=lr,
    )

    output_path_training = output_dir / "training_metrics.html"
    plotter.plot_training_metrics(training_metrics, output_path=output_path_training)
    print(f"✓ Generated training metrics plot: {output_path_training}")
    assert output_path_training.exists()

    # 2. VLO results
    vlo_results = [
        VLOResult(
            clean_logit_diff=2.5,
            intervened_logit_diff=1.0,
            vlo=1.5,
            faithfulness=0.6,
            effect_size=1.5,
            intervention_type=InterventionType.ZERO_ABLATION,
            component_name=f"layer_{i}.attention_head.{j}",
            num_examples=10,
        )
        for i, j in [(7, 9), (9, 9), (10, 0)]
    ]

    output_path_vlo = output_dir / "vlo_results.html"
    plotter.plot_vlo_results(vlo_results, output_path=output_path_vlo, sort_by="vlo")
    print(f"✓ Generated VLO results plot: {output_path_vlo}")
    assert output_path_vlo.exists()

    # 3. VLO distribution
    output_path_vlo_dist = output_dir / "vlo_distribution.html"
    plotter.plot_vlo_distribution(vlo_results, output_path=output_path_vlo_dist)
    print(f"✓ Generated VLO distribution plot: {output_path_vlo_dist}")
    assert output_path_vlo_dist.exists()

    # 4. Circuit comparison
    circuits = [
        CircuitRecord(
            circuit_id=f"circuit_{i}",
            model_name="gpt2",
            components=[],
            causal_metrics=CircuitCausalMetrics(
                vlo_mean=1.0 + i * 0.5,
                vlo_std=0.2,
                faithfulness=0.5 + i * 0.1,
            ),
            semantics=CircuitSemantics(task_tag="test", human_label=f"Circuit {i}"),
        )
        for i in range(3)
    ]
    circuits[0].components = vlo_results[:2]
    circuits[1].components = vlo_results[1:]
    circuits[2].components = vlo_results

    output_path_comparison = output_dir / "circuit_comparison.html"
    plotter.plot_circuit_comparison(circuits, output_path=output_path_comparison)
    print(f"✓ Generated circuit comparison plot: {output_path_comparison}")
    assert output_path_comparison.exists()

    # Cleanup
    import shutil
    shutil.rmtree(output_dir, ignore_errors=True)

    print("\n✅ Metrics Plotter tests PASSED")


# ============================================================================
# Test Activation Explorer
# ============================================================================


def test_activation_explorer():
    """Test PCA/t-SNE/UMAP visualization."""
    print("\n" + "=" * 70)
    print("TEST 3: Activation Explorer")
    print("=" * 70)

    # Check if plotly available
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("⚠️  plotly not installed, skipping test")
        return

    explorer = ActivationExplorer(template="plotly_dark", width=1000, height=800)
    print(f"✓ Created ActivationExplorer")

    # Create mock activations with known structure (3 clusters)
    torch.manual_seed(42)
    n_samples_per_cluster = 50
    n_clusters = 3
    dim = 768

    activations_list = []
    labels_list = []
    for cluster_id in range(n_clusters):
        center = torch.randn(dim) * 5
        cluster_activations = center + torch.randn(n_samples_per_cluster, dim) * 0.5
        activations_list.append(cluster_activations)
        labels_list.extend([f"Cluster {cluster_id}"] * n_samples_per_cluster)

    activations = torch.cat(activations_list, dim=0)
    print(f"✓ Created mock activations: {tuple(activations.shape)} ({n_clusters} clusters)")

    output_dir = Path(tempfile.mkdtemp(prefix="test_activation_"))

    # 1. PCA 2D
    output_path_pca2d = output_dir / "pca_2d.html"
    explorer.plot_activations_2d(
        activations,
        labels=labels_list,
        method=DimReductionMethod.PCA,
        output_path=output_path_pca2d,
    )
    print(f"✓ Generated PCA 2D plot: {output_path_pca2d}")
    assert output_path_pca2d.exists()

    # 2. PCA 3D
    output_path_pca3d = output_dir / "pca_3d.html"
    explorer.plot_activations_3d(
        activations,
        labels=labels_list,
        method=DimReductionMethod.PCA,
        output_path=output_path_pca3d,
    )
    print(f"✓ Generated PCA 3D plot: {output_path_pca3d}")
    assert output_path_pca3d.exists()

    # 3. t-SNE 2D
    output_path_tsne2d = output_dir / "tsne_2d.html"
    explorer.plot_activations_2d(
        activations,
        labels=labels_list,
        method=DimReductionMethod.TSNE,
        output_path=output_path_tsne2d,
        perplexity=20,  # Lower for small dataset
        max_iter=500,  # Faster for testing
    )
    print(f"✓ Generated t-SNE 2D plot: {output_path_tsne2d}")
    assert output_path_tsne2d.exists()

    # 4. Variance explained
    output_path_variance = output_dir / "variance_explained.html"
    explorer.plot_variance_explained(
        activations,
        max_components=50,
        output_path=output_path_variance,
    )
    print(f"✓ Generated variance explained plot: {output_path_variance}")
    assert output_path_variance.exists()

    # 5. Activation heatmap
    output_path_heatmap = output_dir / "activation_heatmap.html"
    explorer.plot_activation_heatmap(
        activations[:50, :100],  # Subsample for visibility
        output_path=output_path_heatmap,
    )
    print(f"✓ Generated activation heatmap: {output_path_heatmap}")
    assert output_path_heatmap.exists()

    # Cleanup
    import shutil
    shutil.rmtree(output_dir, ignore_errors=True)

    print("\n✅ Activation Explorer tests PASSED")


# ============================================================================
# Test SAE Feature Analyzer
# ============================================================================


def test_sae_feature_analyzer():
    """Test SAE feature visualization."""
    print("\n" + "=" * 70)
    print("TEST 4: SAE Feature Analyzer")
    print("=" * 70)

    # Check if plotly available
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("⚠️  plotly not installed, skipping test")
        return

    # Create SAE
    input_dim = 768
    dict_size = input_dim * 4
    sae = LayerSparseAutoencoder(
        input_dim=input_dim,
        dict_size=dict_size,
        sparsity_lambda=1e-3,
    )
    print(f"✓ Created SAE: {input_dim} → {dict_size}")

    # Create analyzer
    analyzer = SAEFeatureAnalyzer(sae, template="plotly_dark", width=1200, height=800)
    print(f"✓ Created SAEFeatureAnalyzer")

    # Create mock inputs
    torch.manual_seed(42)
    num_samples = 100
    inputs = torch.randn(num_samples, input_dim)
    print(f"✓ Created mock inputs: {tuple(inputs.shape)}")

    output_dir = Path(tempfile.mkdtemp(prefix="test_sae_viz_"))

    # 1. Reconstruction quality
    output_path_reconstruction = output_dir / "reconstruction_quality.html"
    analyzer.plot_reconstruction_quality(
        inputs,
        num_samples=5,
        output_path=output_path_reconstruction,
    )
    print(f"✓ Generated reconstruction quality plot: {output_path_reconstruction}")
    assert output_path_reconstruction.exists()

    # 2. Reconstruction errors
    output_path_errors = output_dir / "reconstruction_errors.html"
    analyzer.plot_reconstruction_errors(
        inputs,
        output_path=output_path_errors,
    )
    print(f"✓ Generated reconstruction errors plot: {output_path_errors}")
    assert output_path_errors.exists()

    # 3. Top features
    output_path_top_features = output_dir / "top_features.html"
    analyzer.plot_top_features(
        inputs,
        top_k=20,
        output_path=output_path_top_features,
    )
    print(f"✓ Generated top features plot: {output_path_top_features}")
    assert output_path_top_features.exists()

    # 4. Feature activation heatmap
    output_path_heatmap = output_dir / "feature_heatmap.html"
    analyzer.plot_feature_activation_heatmap(
        inputs,
        max_samples=50,
        max_features=100,
        output_path=output_path_heatmap,
    )
    print(f"✓ Generated feature heatmap: {output_path_heatmap}")
    assert output_path_heatmap.exists()

    # 5. Feature frequency
    output_path_frequency = output_dir / "feature_frequency.html"
    analyzer.plot_feature_frequency(
        inputs,
        threshold=0.01,
        output_path=output_path_frequency,
    )
    print(f"✓ Generated feature frequency plot: {output_path_frequency}")
    assert output_path_frequency.exists()

    # 6. Get top activating examples for specific feature
    feature_idx = 42
    top_examples, top_activations = analyzer.get_top_activating_examples(
        inputs,
        feature_idx=feature_idx,
        top_k=10,
    )
    print(f"✓ Got top activating examples for feature {feature_idx}:")
    print(f"  Top 3 activations: {top_activations[:3].tolist()}")
    assert top_examples.shape == (10, input_dim)
    assert top_activations.shape == (10,)

    # Cleanup
    import shutil
    shutil.rmtree(output_dir, ignore_errors=True)

    print("\n✅ SAE Feature Analyzer tests PASSED")


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("\n" + "=" * 70)
    print("NEUROTRACE VISUALIZATION - INTEGRATION TESTS")
    print("=" * 70)

    try:
        test_circuit_graph_visualizer()
        test_metrics_plotter()
        test_activation_explorer()
        test_sae_feature_analyzer()

        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED")
        print("=" * 70)
        print("\nVisualization Module is ready!")
        print("\nYou can now:")
        print("1. Visualize circuits as interactive graphs (Pyvis)")
        print("2. Plot training and VLO metrics (Plotly)")
        print("3. Explore activations with PCA/t-SNE/UMAP (2D/3D)")
        print("4. Analyze SAE features and reconstruction quality")
        print("\nAll outputs are HTML files that can be opened in a browser.")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
