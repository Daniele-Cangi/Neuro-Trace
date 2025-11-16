# run_discovery_validation.py

"""
Validation run for Layer 0 MLP discovery with 1000 IOI examples.

This script validates the surprising finding from the initial 100-example discovery
that Layer 0 MLP is the dominant component (VLO=1.874) in the IOI task.
"""

import sys
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from transformers import AutoModelForCausalLM, AutoTokenizer

from neurotrace.discovery import (
    ExhaustiveCircuitScanner,
    ScanConfig,
    ComponentInteractionMatrix,
)
from neurotrace.datasets import IOIDatasetGenerator, IOIExample
from neurotrace.causal import CircuitExtractor, InterventionType
from neurotrace.causal.vlo_tester import VLOResult
from neurotrace.control import CircuitRegistry
from neurotrace.visualization import MetricsPlotter, CircuitGraphVisualizer


def prepare_ioi_inputs(
    examples: List[IOIExample],
    tokenizer,
    device: str = "cuda",
) -> tuple:
    """
    Tokenize IOI examples and prepare inputs for scanner.

    Returns:
        (input_ids, attention_mask, target_positions, correct_ids, incorrect_ids)
    """
    texts = [ex.text for ex in examples]

    # Tokenize
    encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    # Target positions (last position for each example)
    target_positions = (attention_mask.sum(dim=1) - 1).tolist()

    # Correct and incorrect token IDs
    correct_token_ids = [
        tokenizer.encode(" " + ex.correct_answer, add_special_tokens=False)[0]
        for ex in examples
    ]
    incorrect_token_ids = [
        tokenizer.encode(" " + ex.incorrect_answer, add_special_tokens=False)[0]
        for ex in examples
    ]

    return (
        input_ids,
        attention_mask,
        target_positions,
        correct_token_ids,
        incorrect_token_ids,
    )


def main():
    """Run validation discovery with 1000 IOI examples."""

    print("=" * 80)
    print("NEUROTRACE - VALIDATION DISCOVERY RUN")
    print("=" * 80)
    print()
    print("Purpose: Validate Layer 0 MLP dominance finding with 1000 IOI examples")
    print("Previous run: 100 examples, Layer 0 MLP VLO=1.874")
    print()

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_examples = 1000  # 10x larger than initial run (batch processing enabled for 6GB VRAM)
    seed = 42

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"runs/discovery_validation/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print()

    # ========================================================================
    # STEP 1: Generate IOI Dataset (1000 examples)
    # ========================================================================

    print("[1/6] Generating IOI dataset...")
    print(f"      - Examples: {num_examples}")
    print(f"      - Diversity: Enabled (round-robin templates)")
    print()

    generator = IOIDatasetGenerator(seed=seed)
    ioi_examples = generator.generate(
        num_examples=num_examples,
        ensure_diversity=True,
    )

    print(f"✓ Generated {len(ioi_examples)} IOI examples")

    # Save dataset
    dataset_path = output_dir / "ioi_dataset.json"
    generator.save_to_json(ioi_examples, dataset_path)

    print(f"✓ Saved to {dataset_path}")
    print()

    # ========================================================================
    # STEP 2: Load Model and Tokenizer
    # ========================================================================

    print("[2/6] Loading GPT-2 model...")

    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print(f"✓ Model loaded: GPT-2 (124M parameters)")
    print(f"✓ Device: {device}")
    print()

    # ========================================================================
    # STEP 3: Prepare Inputs
    # ========================================================================

    print("[3/6] Tokenizing examples...")

    input_ids, attention_mask, target_positions, correct_ids, incorrect_ids = (
        prepare_ioi_inputs(ioi_examples, tokenizer, device)
    )

    print(f"✓ Input shape: {input_ids.shape}")
    print(f"✓ Target positions: {len(target_positions)}")
    print()

    # ========================================================================
    # STEP 4: Run Exhaustive Scan
    # ========================================================================

    print("[4/6] Running exhaustive circuit scan...")
    print(f"      - Total components: 156 (144 heads + 12 MLPs)")
    print(f"      - Examples: {num_examples}")
    print(f"      - VLO threshold: 0.3")
    print(f"      - Faithfulness threshold: 0.2")
    print(f"      - Bootstrap samples: 0 (disabled due to GPU memory constraints)")
    print()

    config = ScanConfig(
        num_layers=12,
        num_heads=12,
        scan_attention_heads=True,
        scan_mlps=True,
        scan_full_layers=False,  # Skip for now
        min_vlo_threshold=0.3,
        min_faithfulness_threshold=0.2,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        save_every_n_components=50,
        num_bootstrap_samples=0,  # Disabled due to GPU memory (6GB insufficient for 500 examples + bootstrap)
    )

    scanner = ExhaustiveCircuitScanner(model, tokenizer, config)

    import time
    start_time = time.time()

    significant_results = scanner.scan_all_components(
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_positions=target_positions,
        correct_token_ids=correct_ids,
        incorrect_token_ids=incorrect_ids,
        task_name="ioi_validation_1000",
    )

    elapsed_time = time.time() - start_time

    print()
    print(f"✓ Scan complete in {elapsed_time:.1f} seconds")
    print(f"✓ Scanned: {len(scanner.results)} components")
    print(f"✓ Significant: {len(significant_results)} components (VLO > 0.3)")
    print()

    # Save scan results
    from dataclasses import asdict
    results_path = output_dir / "scan_results.json"
    with open(results_path, "w") as f:
        json.dump(
            [asdict(r) for r in scanner.results],
            f,
            indent=2,
            default=str,  # Handle InterventionType enum
        )

    print(f"✓ Results saved to {results_path}")
    print()

    # ========================================================================
    # STEP 5: Build Component Interaction Matrix
    # ========================================================================

    print("[5/6] Building component interaction matrix...")

    matrix = ComponentInteractionMatrix()
    matrix.build_from_scan_results(scanner.results)

    # Get layer importance
    layer_importance = matrix.get_layer_importance()

    print("✓ Layer Importance (VLO sum):")
    for layer_idx in sorted(layer_importance.keys()):
        importance = layer_importance[layer_idx]
        bar = "█" * int(abs(importance) * 10)
        sign = "+" if importance >= 0 else "-"
        print(f"   Layer {layer_idx:2d}: {sign}{abs(importance):7.3f}  {bar}")

    print()

    # Save matrix
    matrix_path = output_dir / "interaction_matrix.json"
    matrix.save(str(matrix_path))
    print(f"✓ Matrix saved to {matrix_path}")
    print()

    # ========================================================================
    # STEP 6: Extract Circuit and Visualize
    # ========================================================================

    print("[6/6] Extracting circuit and generating visualizations...")

    # Convert ScanResults to VLOResults for extractor
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
    extractor = CircuitExtractor(
        min_vlo=0.3,
        min_faithfulness=0.2,
    )

    circuit = extractor.extract_from_vlo_results(
        vlo_results=vlo_results,
        circuit_id="gpt2_ioi_validation_1000",
        model_name="gpt2",
        task_tag="ioi",
        human_label="GPT-2 IOI Circuit (Validation with 1000 examples)",
        description=f"Validation run with {num_examples} IOI examples to confirm Layer 0 MLP discovery",
        examples=[ex.text for ex in ioi_examples[:5]],
    )

    print(f"✓ Circuit extracted: {len(circuit.components)} components")
    print()

    # Save to registry
    registry = CircuitRegistry(db_path=str(output_dir / "circuits.db"))
    registry.upsert(circuit)
    print(f"✓ Circuit saved to registry")
    print()

    # Generate visualizations
    plotter = MetricsPlotter(template="plotly_dark")

    # VLO results
    vlo_viz_path = viz_dir / "vlo_results.html"
    plotter.plot_vlo_results(
        vlo_results=scanner.results,
        output_path=str(vlo_viz_path),
        sort_by="vlo",
    )
    print(f"✓ VLO results visualization: {vlo_viz_path}")

    # VLO distribution
    dist_viz_path = viz_dir / "vlo_distribution.html"
    plotter.plot_vlo_distribution(
        vlo_results=scanner.results,
        output_path=str(dist_viz_path),
        threshold=0.3,
    )
    print(f"✓ VLO distribution visualization: {dist_viz_path}")

    # Circuit graph
    graph_viz = CircuitGraphVisualizer()
    graph_path = viz_dir / "circuit_graph.html"
    graph_viz.visualize_circuit(
        circuit=circuit,
        output_path=str(graph_path),
        layout="hierarchical",
        node_color_by="vlo",
    )
    print(f"✓ Circuit graph visualization: {graph_path}")
    print()

    # ========================================================================
    # Summary
    # ========================================================================

    print("=" * 80)
    print("VALIDATION DISCOVERY COMPLETE")
    print("=" * 80)
    print()
    print(f"Dataset: {num_examples} IOI examples")
    print(f"Components scanned: {len(scanner.results)}")
    print(f"Significant components: {len(significant_results)}")
    print(f"Time elapsed: {elapsed_time:.1f} seconds")
    print()

    # Print top 10 components
    print("Top 10 Components by VLO:")
    sorted_results = sorted(scanner.results, key=lambda r: r.vlo, reverse=True)
    for i, result in enumerate(sorted_results[:10], 1):
        print(f"  {i:2d}. {result.component_name:20s}  VLO={result.vlo:7.3f}  F={result.faithfulness:6.3f}")

    print()
    print(f"Output directory: {output_dir}")
    print()

    # Comparison with initial run
    print("=" * 80)
    print("COMPARISON WITH INITIAL 100-EXAMPLE RUN")
    print("=" * 80)
    print()
    print("Initial run (100 examples):")
    print("  - Layer 0 MLP: VLO=1.874, Faithfulness=4.433")
    print()
    print(f"Validation run ({num_examples} examples):")

    # Find layer 0 MLP
    layer_0_mlp = next((r for r in scanner.results if r.component_name == "layer_0.mlp"), None)
    if layer_0_mlp:
        print(f"  - Layer 0 MLP: VLO={layer_0_mlp.vlo:.3f}, Faithfulness={layer_0_mlp.faithfulness:.3f}")

        if layer_0_mlp.vlo > 0.3:
            print()
            print("✅ VALIDATION CONFIRMED: Layer 0 MLP remains significant with larger dataset")
        else:
            print()
            print("⚠️  VALIDATION FAILED: Layer 0 MLP not significant with larger dataset")
            print("    Likely an artifact of small sample size in initial run")
    else:
        print("  - Layer 0 MLP: Not found in results")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
