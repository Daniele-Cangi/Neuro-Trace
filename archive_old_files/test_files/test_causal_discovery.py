"""
Test per Causal Discovery Pipeline (Phase 3-6).

Verifica:
1. Geometric Analysis: LID, spectral features
2. VLO Tester: logit difference, interventions
3. Circuit Extractor: VLO → CircuitRecord
4. Integration end-to-end
"""

from __future__ import annotations

import sys
import tempfile

import torch

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from neurotrace.analysis import (
    compute_lid,
    compute_spectral_features,
    ActivationGeometry,
)
from neurotrace.causal import (
    VLOTester,
    InterventionType,
    CircuitExtractor,
    extract_circuit_from_components,
)
from neurotrace.control import CircuitRegistry


# ============================================================================
# Test Geometric Analysis
# ============================================================================


def test_geometric_analysis():
    """Test LID and spectral features computation."""
    print("\n" + "=" * 70)
    print("TEST 1: Geometric Analysis")
    print("=" * 70)

    # Create mock activations with known structure
    torch.manual_seed(42)

    # Low-dimensional manifold: project 768-dim to 10-dim
    N, D_manifold, D_ambient = 100, 10, 768
    latent = torch.randn(N, D_manifold)
    projection = torch.randn(D_manifold, D_ambient) / D_ambient ** 0.5
    activations = latent @ projection  # [100, 768]

    print(f"✓ Created mock activations: {tuple(activations.shape)}")
    print(f"  True manifold dim: {D_manifold}, Ambient dim: {D_ambient}")

    # 1. Test LID
    lid_mean, lid_std = compute_lid(activations, k=20, method="mle")
    print(f"✓ LID computed: {lid_mean:.2f} ± {lid_std:.2f}")
    assert 5 < lid_mean < 30, f"LID should detect low-dim structure, got {lid_mean}"

    # 2. Test spectral features
    spectral = compute_spectral_features(activations, top_k=50)
    print(f"✓ Spectral features:")
    print(f"  Spectral entropy: {spectral['spectral_entropy']:.3f}")
    print(f"  Participation ratio: {spectral['participation_ratio']:.1f}")
    print(f"  Effective rank: {spectral['effective_rank']:.1f}")
    print(f"  Explained variance (top-50): {spectral['explained_variance_ratio']:.3f}")

    assert 0 <= spectral['spectral_entropy'] <= 1
    assert spectral['effective_rank'] < D_ambient

    # 3. Test ActivationGeometry (integrated)
    analyzer = ActivationGeometry(lid_k=20, spectral_top_k=50)
    features = analyzer.analyze(activations)

    print(f"\n✓ GeometricFeatures:")
    print(f"  LID: {features.lid:.2f} ± {features.lid_std:.2f}")
    print(f"  Spectral entropy: {features.spectral_entropy:.3f}")
    print(f"  Effective rank: {features.effective_rank:.1f}")
    print(f"  Samples: {features.num_samples}, Ambient dim: {features.ambient_dim}")

    assert features.num_samples == N
    assert features.ambient_dim == D_ambient

    print("\n✅ Geometric Analysis tests PASSED")


# ============================================================================
# Test VLO Tester
# ============================================================================


def test_vlo_tester():
    """Test VLO computation and interventions."""
    print("\n" + "=" * 70)
    print("TEST 2: VLO Tester")
    print("=" * 70)

    # Load GPT-2 for real testing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("✓ Loaded GPT-2")

    # Create simple IOI-like examples
    # "When John and Mary went to the store, John gave a drink to"
    # Correct: "Mary", Incorrect: "John"
    examples = [
        "When John and Mary went to the store, John gave a drink to",
        "Alice and Bob were at the park. Alice handed the ball to",
    ]

    inputs = tokenizer(examples, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Target positions: last token
    target_positions = torch.tensor([input_ids.shape[1] - 1] * len(examples))

    # Correct/incorrect tokens (manually set for this test)
    # For real IOI, we'd parse from examples
    mary_id = tokenizer.encode(" Mary")[0]
    john_id = tokenizer.encode(" John")[0]
    bob_id = tokenizer.encode(" Bob")[0]
    alice_id = tokenizer.encode(" Alice")[0]

    correct_token_ids = torch.tensor([mary_id, bob_id])
    incorrect_token_ids = torch.tensor([john_id, alice_id])

    print(f"✓ Created {len(examples)} IOI-like examples")

    # Test VLO on layer 9 (known IOI layer)
    tester = VLOTester(model, tokenizer, device=device)

    result = tester.test_component(
        layer_idx=9,
        component_type="attention_head",
        component_idx=None,  # Full attention layer
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_positions=target_positions,
        correct_token_ids=correct_token_ids,
        incorrect_token_ids=incorrect_token_ids,
        intervention_type=InterventionType.ZERO_ABLATION,
    )

    print(f"\n✓ VLO Result for layer_9.attention_head:")
    print(f"  Clean logit diff: {result.clean_logit_diff:.3f}")
    print(f"  Intervened logit diff: {result.intervened_logit_diff:.3f}")
    print(f"  VLO: {result.vlo:.3f}")
    print(f"  Faithfulness: {result.faithfulness:.3f}")

    # Validate structure
    assert isinstance(result.vlo, float)
    assert isinstance(result.faithfulness, float)
    assert result.num_examples == len(examples)

    # Test multiple components
    components = [
        (7, "attention_head", None),
        (9, "attention_head", None),
        (10, "mlp", None),
    ]

    results = tester.test_circuit(
        components=components,
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_positions=target_positions,
        correct_token_ids=correct_token_ids,
        incorrect_token_ids=incorrect_token_ids,
    )

    print(f"\n✓ Tested {len(results)} components:")
    for r in results:
        print(f"  {r.component_name}: VLO={r.vlo:.3f}, Faithfulness={r.faithfulness:.3f}")

    assert len(results) == len(components)

    print("\n✅ VLO Tester tests PASSED")


# ============================================================================
# Test Circuit Extractor
# ============================================================================


def test_circuit_extractor():
    """Test CircuitRecord extraction from VLO results."""
    print("\n" + "=" * 70)
    print("TEST 3: Circuit Extractor")
    print("=" * 70)

    # Create mock VLO results
    from neurotrace.causal.vlo_tester import VLOResult

    vlo_results = [
        VLOResult(
            clean_logit_diff=2.5,
            intervened_logit_diff=1.0,
            vlo=1.5,
            faithfulness=0.6,
            effect_size=1.5,
            intervention_type=InterventionType.ZERO_ABLATION,
            component_name="layer_9.attention_head.9",
            num_examples=10,
        ),
        VLOResult(
            clean_logit_diff=2.5,
            intervened_logit_diff=0.5,
            vlo=2.0,
            faithfulness=0.8,
            effect_size=2.0,
            intervention_type=InterventionType.ZERO_ABLATION,
            component_name="layer_10.attention_head.0",
            num_examples=10,
        ),
        VLOResult(
            clean_logit_diff=2.5,
            intervened_logit_diff=2.4,
            vlo=0.1,  # Low VLO - should be filtered
            faithfulness=0.04,
            effect_size=0.1,
            intervention_type=InterventionType.ZERO_ABLATION,
            component_name="layer_3.mlp.0",
            num_examples=10,
        ),
    ]

    print(f"✓ Created {len(vlo_results)} mock VLO results")

    # Extract circuit
    extractor = CircuitExtractor(min_vlo=0.5, min_faithfulness=0.3)

    circuit_record = extractor.extract_from_vlo_results(
        vlo_results=vlo_results,
        circuit_id="test_ioi_circuit",
        model_name="gpt2",
        task_tag="ioi",
        human_label="IOI Name Mover",
        description="Circuit for Indirect Object Identification",
        examples=["John gave Mary the book, John gave it to"],
        sae_features={"layer_9": [42, 103], "layer_10": [7, 15]},
    )

    print(f"\n✓ Extracted CircuitRecord:")
    print(f"  Circuit ID: {circuit_record.circuit_id}")
    print(f"  Model: {circuit_record.model_name}")
    print(f"  Task: {circuit_record.semantics.task_tag}")
    print(f"  Components: {len(circuit_record.components)}")
    print(f"  VLO mean: {circuit_record.causal_metrics.vlo_mean:.3f}")
    print(f"  Faithfulness: {circuit_record.causal_metrics.faithfulness:.3f}")

    # Validate filtering (should keep 2/3 components)
    assert len(circuit_record.components) == 2, f"Expected 2 components, got {len(circuit_record.components)}"
    assert circuit_record.causal_metrics.vlo_mean > 1.0

    # Test utility function
    manual_circuit = extract_circuit_from_components(
        components=[(9, "attention_head", 9), (10, "attention_head", 0)],
        circuit_id="manual_ioi",
        model_name="gpt2",
        task_tag="ioi",
        vlo_mean=1.75,
        faithfulness=0.7,
        human_label="Manual IOI",
    )

    print(f"\n✓ Created manual circuit:")
    print(f"  Components: {len(manual_circuit.components)}")
    print(f"  VLO: {manual_circuit.causal_metrics.vlo_mean:.3f}")

    assert len(manual_circuit.components) == 2

    print("\n✅ Circuit Extractor tests PASSED")


# ============================================================================
# Test Integration with CircuitRegistry
# ============================================================================


def test_registry_integration():
    """Test saving extracted circuits to registry."""
    print("\n" + "=" * 70)
    print("TEST 4: Registry Integration")
    print("=" * 70)

    db_path = "test_causal_circuits.db"
    try:
        # Cleanup
        import os
        for ext in ["", "-shm", "-wal"]:
            if os.path.exists(db_path + ext):
                try:
                    os.remove(db_path + ext)
                except:
                    pass

        registry = CircuitRegistry(db_path)

        # Create circuit
        circuit = extract_circuit_from_components(
            components=[
                (9, "attention_head", 9),
                (10, "attention_head", 0),
                (10, "mlp", 0),
            ],
            circuit_id="ioi_complete_circuit",
            model_name="gpt2",
            task_tag="ioi",
            vlo_mean=1.85,
            faithfulness=0.82,
            human_label="IOI Complete",
            description="Full IOI circuit with name mover heads",
            examples=["When John and Mary went to the store, John gave a drink to Mary"],
            sae_indices={"layer_9": [42, 103, 200], "layer_10": [7, 15, 88]},
        )

        # Save to registry
        registry.upsert(circuit)
        print(f"✓ Saved circuit to registry: {circuit.circuit_id}")

        # Retrieve
        retrieved = registry.get("ioi_complete_circuit")
        assert retrieved is not None
        assert retrieved.circuit_id == "ioi_complete_circuit"
        assert len(retrieved.components) == 3
        print(f"✓ Retrieved circuit: {retrieved.semantics.human_label}")

        # Query by metrics
        high_vlo_circuits = registry.list(task_tag="ioi", min_vlo=1.5)
        assert len(high_vlo_circuits) == 1
        print(f"✓ Query found {len(high_vlo_circuits)} circuit(s) with VLO > 1.5")

        # Close
        registry.close()

    finally:
        import time
        time.sleep(0.1)
        for ext in ["", "-shm", "-wal"]:
            if os.path.exists(db_path + ext):
                try:
                    os.remove(db_path + ext)
                except:
                    pass

    print("\n✅ Registry Integration tests PASSED")


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
    print("NEUROTRACE CAUSAL DISCOVERY - INTEGRATION TESTS")
    print("=" * 70)

    try:
        test_geometric_analysis()
        test_vlo_tester()
        test_circuit_extractor()
        test_registry_integration()

        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED")
        print("=" * 70)
        print("\nCausal Discovery Pipeline is ready!")
        print("\nYou can now:")
        print("1. Analyze geometric features of activations")
        print("2. Test causal importance with VLO")
        print("3. Extract and register circuits")
        print("4. Use circuits in Control Plane for steering")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
