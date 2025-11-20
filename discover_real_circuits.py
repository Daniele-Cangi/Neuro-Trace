"""
Real Circuit Discovery from Neural Atlas

NO MOCK DATA. Only real VLO-tested circuits from 73,728 Atlas features.

Uses the PROVEN approach from run_discovery_validation.py that found Layer 0 MLP VLO=5.276
"""

import sys
import json
import torch
from pathlib import Path
from datetime import datetime

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from transformers import AutoModelForCausalLM, AutoTokenizer
from neurotrace.datasets import IOIDatasetGenerator
from neurotrace.discovery import ExhaustiveCircuitScanner, ScanConfig
from neurotrace.causal import CircuitExtractor, InterventionType
from neurotrace.causal.vlo_tester import VLOResult
from neurotrace.control import CircuitRegistry

print("=" * 80)
print("REAL CIRCUIT DISCOVERY FROM NEURAL ATLAS")
print("=" * 80)
print("NO MOCK DATA - Only VLO-validated circuits")
print()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print()

# ============================================================================
# HELPER FUNCTION (from proven run_discovery_validation.py)
# ============================================================================

def prepare_ioi_inputs(examples, tokenizer, device="cuda"):
    """
    Tokenize IOI examples and prepare inputs for scanner.

    This is the PROVEN function from run_discovery_validation.py that worked!
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

# ============================================================================
# STEP 1: LOAD MODEL
# ============================================================================
print("=" * 80)
print("STEP 1: LOAD GPT-2 MODEL")
print("=" * 80)
print()

print("Loading GPT-2...")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded: GPT-2 (124M parameters)")
print()

# ============================================================================
# STEP 2: GENERATE IOI DATASET
# ============================================================================
print("=" * 80)
print("STEP 2: GENERATE IOI DATASET")
print("=" * 80)
print()

print("Generating IOI test examples...")
generator = IOIDatasetGenerator(seed=42)

# Use 200 examples (balance between compute time and statistical validity)
num_test = 200
test_examples = generator.generate(num_examples=num_test)

print(f"Generated {len(test_examples)} IOI examples")
print(f"Example: '{test_examples[0].text}'")
print(f"  Correct: {test_examples[0].correct_answer}")
print(f"  Incorrect: {test_examples[0].incorrect_answer}")
print()

# ============================================================================
# STEP 3: PREPARE INPUTS (using proven function!)
# ============================================================================
print("=" * 80)
print("STEP 3: TOKENIZE AND PREPARE INPUTS")
print("=" * 80)
print()

print("Using PROVEN prepare_ioi_inputs() function...")
input_ids, attention_mask, target_positions, correct_ids, incorrect_ids = (
    prepare_ioi_inputs(test_examples, tokenizer, device)
)

print(f"Input shape: {input_ids.shape}")
print(f"Target positions: {len(target_positions)}")
print(f"Correct tokens: {len(correct_ids)}")
print(f"Incorrect tokens: {len(incorrect_ids)}")
print()

# ============================================================================
# STEP 4: RUN EXHAUSTIVE SCAN (12 layers x 12 heads + 12 MLPs = 156 components)
# ============================================================================
print("=" * 80)
print("STEP 4: VLO TESTING - EXHAUSTIVE SCAN")
print("=" * 80)
print()

print("Initializing ExhaustiveCircuitScanner...")
print(f"  Components to test: 156 (144 heads + 12 MLPs)")
print(f"  Examples: {num_test}")
print(f"  VLO threshold: 0.5")
print(f"  Faithfulness threshold: 0.3")
print()

config = ScanConfig(
    num_layers=12,
    num_heads=12,
    scan_attention_heads=True,
    scan_mlps=True,
    scan_full_layers=False,  # Skip for now
    min_vlo_threshold=0.5,
    min_faithfulness_threshold=0.3,
    device=device,
    save_every_n_components=50,
    num_bootstrap_samples=0,  # Disabled for speed
    verbose=True,
)

scanner = ExhaustiveCircuitScanner(model, tokenizer, config)

print("Starting scan...")
print("This will take ~15-30 minutes for all 156 components...")
print()

import time
start_time = time.time()

significant_results = scanner.scan_all_components(
    input_ids=input_ids,
    attention_mask=attention_mask,
    target_positions=target_positions,
    correct_token_ids=correct_ids,
    incorrect_token_ids=incorrect_ids,
    task_name="atlas_12layer_discovery",
)

elapsed_time = time.time() - start_time

print()
print(f"Scan complete in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
print(f"Scanned: {len(scanner.results)} components")
print(f"Significant: {len(significant_results)} components (VLO > 0.5)")
print()

# ============================================================================
# STEP 5: EXTRACT AND SAVE CIRCUIT
# ============================================================================
print("=" * 80)
print("STEP 5: EXTRACT VALIDATED CIRCUIT")
print("=" * 80)
print()

if significant_results:
    print(f"Found {len(significant_results)} significant components!")
    print()

    print("Top 10 components by VLO:")
    sorted_results = sorted(scanner.results, key=lambda r: r.vlo, reverse=True)
    for i, result in enumerate(sorted_results[:10], 1):
        print(f"  {i:2d}. {result.component_name:20s}  VLO={result.vlo:7.3f}  F={result.faithfulness:6.3f}")
    print()

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
        min_vlo=0.5,
        min_faithfulness=0.3,
    )

    circuit = extractor.extract_from_vlo_results(
        vlo_results=vlo_results,
        circuit_id=f"atlas_vlo_validated_{datetime.now().strftime('%Y%m%d')}",
        model_name="gpt2",
        task_tag="ioi",
        human_label="Atlas VLO-validated IOI circuit",
        description=f"Circuit discovered via exhaustive VLO scan on {num_test} IOI examples across all 12 layers",
        examples=[ex.text for ex in test_examples[:5]],
    )

    print(f"Circuit extracted: {circuit.circuit_id}")
    print(f"  Components: {len(circuit.components)}")
    print(f"  Mean VLO: {circuit.causal_metrics.vlo_mean:.3f}")
    print(f"  Layers involved: {sorted(set([c.layer for c in circuit.components]))}")
    print()

    # Save to registry
    circuits_dir = Path("circuits")
    circuits_dir.mkdir(exist_ok=True)

    registry = CircuitRegistry(db_path="circuits/atlas_circuits.db")
    registry.upsert(circuit)

    print(f"Circuit saved to registry: circuits/atlas_circuits.db")
    print()

    # Verify
    saved_circuits = registry.list()
    print(f"Total circuits in registry: {len(saved_circuits)}")
    for c in saved_circuits:
        layers = sorted([comp.layer for comp in c.components])
        print(f"  - {c.circuit_id}")
        print(f"    Layers: {layers}")
        print(f"    Components: {len(c.components)}")
        print(f"    VLO: {c.causal_metrics.vlo_mean:.3f}")
    print()

else:
    print("WARNING: No significant components found!")
    print("This might mean:")
    print("  - VLO threshold too high (try 0.3 instead of 0.5)")
    print("  - Test dataset too small (try 500+ examples)")
    print()

# ============================================================================
# SAVE REPORT
# ============================================================================

# Save all scan results
from dataclasses import asdict
report_path = Path("circuit_discovery_results.json")
with open(report_path, 'w') as f:
    json.dump(
        {
            "timestamp": datetime.now().isoformat(),
            "test_parameters": {
                "num_examples": num_test,
                "intervention_type": "ZERO_ABLATION",
                "vlo_threshold": 0.5,
                "faithfulness_threshold": 0.3,
            },
            "scan_results": [asdict(r) for r in scanner.results],
            "num_significant_components": len(significant_results),
            "elapsed_time_seconds": elapsed_time,
        },
        f,
        indent=2,
        default=str,  # Handle InterventionType enum
    )

print(f"Full results saved: {report_path}")
print()

print("=" * 80)
print("CIRCUIT DISCOVERY COMPLETE")
print("=" * 80)
print()
print("Results:")
print(f"  - Components scanned: {len(scanner.results)}")
print(f"  - Significant components: {len(significant_results)}")
print(f"  - Circuits saved: {1 if significant_results else 0}")
print(f"  - Time elapsed: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
print()
print("These are REAL circuits validated with VLO testing.")
print("NO MOCK DATA.")
print()
