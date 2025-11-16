"""
NEUROTRACE - COMPLETE NEURAL CARTOGRAPHY
Automated Circuit Discovery Runner

Questo script esegue:
1. Genera IOI dataset (1000+ esempi)
2. Scansiona TUTTI i 288 componenti di GPT-2
3. Salva circuiti scoperti in CircuitRegistry
4. Genera visualizzazioni complete
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from neurotrace.datasets import IOIDatasetGenerator
from neurotrace.discovery import ExhaustiveCircuitScanner, ScanConfig, ComponentInteractionMatrix
from neurotrace.causal import CircuitExtractor
from neurotrace.control import CircuitRegistry
from neurotrace.visualization import MetricsPlotter, CircuitGraphVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    print("=" * 80)
    print("NEUROTRACE - COMPLETE NEURAL CARTOGRAPHY")
    print("Automated Circuit Discovery on GPT-2")
    print("=" * 80)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    output_dir = Path("runs/discovery") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # ========================================================================
    # STEP 1: Generate IOI Dataset
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Generating IOI Dataset")
    print("=" * 80)

    generator = IOIDatasetGenerator(seed=42)
    ioi_examples = generator.generate(num_examples=100, ensure_diversity=True)  # 100 for speed

    dataset_path = output_dir / "ioi_dataset.json"
    generator.save_to_json(ioi_examples, dataset_path)
    print(f"✓ Generated {len(ioi_examples)} IOI examples")
    print(f"✓ Saved to: {dataset_path}")

    # ========================================================================
    # STEP 2: Prepare Inputs for VLO Testing
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Preparing Inputs")
    print("=" * 80)

    # Load model
    print("Loading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print("✓ Model loaded")

    # Tokenize examples
    texts = [ex.text for ex in ioi_examples]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Target positions (last token)
    target_positions = torch.tensor([input_ids.shape[1] - 1] * len(ioi_examples))

    # Correct/incorrect token IDs
    correct_token_ids = torch.tensor([
        tokenizer.encode(" " + ex.correct_answer)[0] for ex in ioi_examples
    ]).to(device)

    incorrect_token_ids = torch.tensor([
        tokenizer.encode(" " + ex.incorrect_answer)[0] for ex in ioi_examples
    ]).to(device)

    print(f"✓ Tokenized {len(ioi_examples)} examples")
    print(f"  Input shape: {tuple(input_ids.shape)}")

    # ========================================================================
    # STEP 3: Exhaustive Circuit Scan
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Exhaustive Circuit Scan (288 components)")
    print("=" * 80)

    config = ScanConfig(
        num_layers=12,
        num_heads=12,
        scan_attention_heads=True,
        scan_mlps=True,
        scan_full_layers=False,  # Skip for speed (already covered)
        min_vlo_threshold=0.3,
        min_faithfulness_threshold=0.2,
        device=device,
        checkpoint_dir=str(output_dir / "checkpoints"),
        save_every_n_components=50,
        verbose=True,
    )

    scanner = ExhaustiveCircuitScanner(model, tokenizer, config)

    # Run scan
    significant_results = scanner.scan_all_components(
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_positions=target_positions,
        correct_token_ids=correct_token_ids,
        incorrect_token_ids=incorrect_token_ids,
        task_name="IOI",
    )

    # Save results
    results_path = output_dir / "scan_results.json"
    scanner.save_results(results_path)
    print(f"\n✓ Saved scan results: {results_path}")

    # Get top components
    top_components = scanner.get_top_components(top_k=20, sort_by="vlo")
    print(f"\n=== Top 20 Components (by VLO) ===")
    for i, result in enumerate(top_components, 1):
        print(f"{i:2d}. {result.component_name:30s} VLO={result.vlo:.3f} F={result.faithfulness:.3f}")

    # ========================================================================
    # STEP 4: Build Component Interaction Matrix
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Building Component Interaction Matrix")
    print("=" * 80)

    matrix = ComponentInteractionMatrix()
    matrix.build_from_scan_results(scanner.results)

    matrix_path = output_dir / "interaction_matrix.json"
    matrix.save(matrix_path)
    print(f"✓ Saved interaction matrix: {matrix_path}")

    # Layer importance
    layer_importance = matrix.get_layer_importance()
    print(f"\n=== Layer Importance ===")
    for layer_idx in sorted(layer_importance.keys()):
        print(f"Layer {layer_idx:2d}: {layer_importance[layer_idx]:.3f}")

    # ========================================================================
    # STEP 5: Extract Circuit and Save to Registry
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Extracting Circuit and Saving to Registry")
    print("=" * 80)

    # Convert ScanResults to VLOResults for extractor
    from neurotrace.causal.vlo_tester import VLOResult
    from neurotrace.causal import InterventionType

    vlo_results = [
        VLOResult(
            clean_logit_diff=r.clean_logit_diff,
            intervened_logit_diff=r.intervened_logit_diff,
            vlo=r.vlo,
            faithfulness=r.faithfulness,
            effect_size=r.effect_size,
            intervention_type=InterventionType.ZERO_ABLATION,
            component_name=r.component_name,
            num_examples=r.num_examples,
        )
        for r in significant_results
    ]

    # Extract circuit
    extractor = CircuitExtractor(min_vlo=0.3, min_faithfulness=0.2)
    circuit = extractor.extract_from_vlo_results(
        vlo_results=vlo_results,
        circuit_id="gpt2_ioi_discovered",
        model_name="gpt2",
        task_tag="ioi",
        human_label="GPT-2 IOI Circuit (Auto-Discovered)",
        description=f"Automatically discovered circuit from {len(ioi_examples)} IOI examples",
        examples=[ex.text for ex in ioi_examples[:5]],
    )

    print(f"✓ Extracted circuit: {len(circuit.components)} components")
    print(f"  VLO mean: {circuit.causal_metrics.vlo_mean:.3f}")
    print(f"  Faithfulness: {circuit.causal_metrics.faithfulness:.3f}")

    # Save to registry
    registry_path = output_dir / "circuits.db"
    registry = CircuitRegistry(str(registry_path))
    registry.upsert(circuit)
    print(f"✓ Saved circuit to registry: {registry_path}")

    # ========================================================================
    # STEP 6: Generate Visualizations
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: Generating Visualizations")
    print("=" * 80)

    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    # 1. VLO results plot
    plotter = MetricsPlotter(template="plotly_dark")
    vlo_plot_path = viz_dir / "vlo_results.html"
    plotter.plot_vlo_results(
        vlo_results,
        output_path=vlo_plot_path,
        sort_by="vlo",
    )
    print(f"✓ VLO results plot: {vlo_plot_path}")

    # 2. VLO distribution
    vlo_dist_path = viz_dir / "vlo_distribution.html"
    plotter.plot_vlo_distribution(
        vlo_results,
        output_path=vlo_dist_path,
    )
    print(f"✓ VLO distribution plot: {vlo_dist_path}")

    # 3. Circuit graph (if pyvis available)
    try:
        visualizer = CircuitGraphVisualizer()
        circuit_graph_path = viz_dir / "circuit_graph.html"
        visualizer.visualize_circuit(
            circuit,
            output_path=circuit_graph_path,
            layout="hierarchical",
            node_color_by="vlo",
        )
        print(f"✓ Circuit graph: {circuit_graph_path}")
    except ImportError:
        print("⚠️  Pyvis not installed, skipping circuit graph")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("🎉 DISCOVERY COMPLETE!")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nDiscovered Components:")
    print(f"  Total scanned:    {len(scanner.results)}")
    print(f"  Significant:      {len(significant_results)}")
    print(f"  In final circuit: {len(circuit.components)}")
    print(f"\nCircuit Metrics:")
    print(f"  VLO mean:        {circuit.causal_metrics.vlo_mean:.3f}")
    print(f"  Faithfulness:    {circuit.causal_metrics.faithfulness:.3f}")
    print(f"\nFiles Generated:")
    print(f"  - {dataset_path}")
    print(f"  - {results_path}")
    print(f"  - {matrix_path}")
    print(f"  - {registry_path}")
    print(f"  - {viz_dir}/*.html")
    print("\n" + "=" * 80)

    registry.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Discovery interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Discovery failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
