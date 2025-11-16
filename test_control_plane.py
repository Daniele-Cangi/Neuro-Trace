"""
Test di integrazione per NeuroTrace Control Plane.

Verifica:
1. CircuitRegistry: CRUD operations
2. SteeringBuilder: costruzione steering vectors da SAE
3. CircuitController: attivazione circuiti + generazione
4. Integration end-to-end: registry → steering → generation

Nota: usa circuiti mock per non dipendere da SAE addestrati.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

from neurotrace.config import NeuroTraceConfig
from neurotrace.models.wrapper import TargetModelWrapper
from neurotrace.control import (
    CircuitComponent,
    CircuitCausalMetrics,
    CircuitSemantics,
    CircuitFeatures,
    CircuitRecord,
    CircuitRegistry,
    SteeringBuilder,
    CircuitController,
    FeatureStore,
)


# ============================================================================
# Mock FeatureStore per testing senza SAE reali
# ============================================================================


class MockFeatureStore(FeatureStore):
    """
    FeatureStore di test che ritorna direzioni random normalizzate.
    """

    def __init__(self, hidden_dim: int = 768, device: torch.device | None = None):
        self.hidden_dim = hidden_dim
        self.device = device or torch.device("cpu")

    def get_sae_directions(
        self,
        model_name: str,
        layer: int,
        feature_indices: list[int],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        device = device or self.device
        n_features = len(feature_indices)

        # Genera direzioni random normalizzate (simula SAE decoder rows)
        directions = torch.randn(n_features, self.hidden_dim, device=device)
        directions = directions / torch.norm(directions, dim=1, keepdim=True)

        return directions


# ============================================================================
# Test Functions
# ============================================================================


def test_circuit_registry():
    """Test CRUD operations su CircuitRegistry."""
    print("\n" + "=" * 70)
    print("TEST 1: CircuitRegistry CRUD")
    print("=" * 70)

    # Use current directory instead of temp to avoid Windows WAL file locks
    db_path = "test_circuits_temp.db"
    try:
        # Cleanup if exists
        for ext in ["", "-shm", "-wal"]:
            if os.path.exists(db_path + ext):
                try:
                    os.remove(db_path + ext)
                except:
                    pass

        registry = CircuitRegistry(db_path)

        # 1. Crea un circuito di test
        circuit = CircuitRecord(
            circuit_id="test_ioi_001",
            model_name="gpt2",
            model_revision="main",
            components=[
                CircuitComponent(layer=9, component_type="attention_head", index=9),
                CircuitComponent(layer=10, component_type="attention_head", index=0),
            ],
            features=CircuitFeatures(
                sae_indices={"layer_9": [42, 103], "layer_10": [7, 15]},
                geometric={"lid": 2.3, "spectral_entropy": 0.41},
            ),
            causal_metrics=CircuitCausalMetrics(
                vlo_mean=1.72, vlo_std=0.15, faithfulness=0.83
            ),
            semantics=CircuitSemantics(
                task_tag="ioi",
                human_label="name_mover_core",
                description="Circuit for Indirect Object Identification",
                examples=["John told Mary that Peter helped her because..."],
            ),
        )

        # 2. Upsert
        registry.upsert(circuit)
        print(f"✓ Inserted circuit: {circuit.circuit_id}")

        # 3. Get
        retrieved = registry.get("test_ioi_001")
        assert retrieved is not None
        assert retrieved.circuit_id == "test_ioi_001"
        assert retrieved.semantics.task_tag == "ioi"
        assert len(retrieved.components) == 2
        print(f"✓ Retrieved circuit: {retrieved.circuit_id}")

        # 4. List with filters
        circuits = registry.list(task_tag="ioi", min_vlo=1.5)
        assert len(circuits) == 1
        assert circuits[0].circuit_id == "test_ioi_001"
        print(f"✓ List query returned {len(circuits)} circuit(s)")

        # 5. Insert secondo circuito
        circuit2 = CircuitRecord(
            circuit_id="test_count_001",
            model_name="gpt2",
            components=[CircuitComponent(layer=8, component_type="mlp", index=0)],
            features=CircuitFeatures(sae_indices={"layer_8": [99]}),
            causal_metrics=CircuitCausalMetrics(vlo_mean=0.5),
            semantics=CircuitSemantics(task_tag="counting"),
        )
        registry.upsert(circuit2)
        print(f"✓ Inserted second circuit: {circuit2.circuit_id}")

        # 6. List all
        all_circuits = registry.list()
        assert len(all_circuits) == 2
        print(f"✓ Total circuits in registry: {len(all_circuits)}")

        # 7. Delete
        registry.delete("test_count_001")
        remaining = registry.list()
        assert len(remaining) == 1
        print(f"✓ Deleted circuit, remaining: {len(remaining)}")

        # 8. Close registry before cleanup (Windows WAL fix)
        registry.close()
        print("✓ Registry closed")

    finally:
        # Cleanup
        import time
        time.sleep(0.1)  # Give Windows time to release file handles
        for ext in ["", "-shm", "-wal"]:
            if os.path.exists(db_path + ext):
                try:
                    os.remove(db_path + ext)
                except:
                    pass

    print("\n✅ CircuitRegistry tests PASSED")


def test_steering_builder():
    """Test costruzione steering vectors."""
    print("\n" + "=" * 70)
    print("TEST 2: SteeringBuilder")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Mock FeatureStore
    feature_store = MockFeatureStore(hidden_dim=768, device=device)

    # SteeringBuilder
    builder = SteeringBuilder(
        feature_store=feature_store,
        default_alpha=0.7,
        alpha_bounds=(-2.0, 2.0),
        device=device,
    )

    # Circuito di test
    circuit = CircuitRecord(
        circuit_id="test_steering_001",
        model_name="gpt2",
        components=[
            CircuitComponent(layer=5, component_type="attention_head", index=3),
            CircuitComponent(layer=7, component_type="mlp", index=0),
        ],
        features=CircuitFeatures(
            sae_indices={
                "layer_5": [10, 20, 30],  # 3 features
                "layer_7": [5, 15],  # 2 features
            }
        ),
        causal_metrics=CircuitCausalMetrics(vlo_mean=1.5),
        semantics=CircuitSemantics(task_tag="test"),
    )

    # Build steering spec
    spec = builder.build_from_circuit(circuit)

    print(f"✓ Built SteeringSpec for circuit: {spec.circuit_id}")
    print(f"  Active layers: {spec.active_layers()}")
    assert len(spec.active_layers()) == 2
    assert 5 in spec.layer_vectors
    assert 7 in spec.layer_vectors

    # Verifica layer 5
    lv5 = spec.layer_vectors[5]
    assert lv5.layer == 5
    assert lv5.direction.shape == (768,)
    assert abs(torch.norm(lv5.direction).item() - 1.0) < 1e-5  # normalized
    print(f"  Layer 5: direction shape={tuple(lv5.direction.shape)}, alpha={lv5.default_alpha}")

    # Verifica layer 7
    lv7 = spec.layer_vectors[7]
    assert lv7.layer == 7
    assert lv7.direction.shape == (768,)
    print(f"  Layer 7: direction shape={tuple(lv7.direction.shape)}, alpha={lv7.default_alpha}")

    print("\n✅ SteeringBuilder tests PASSED")


def test_controller_integration():
    """Test integrazione completa con TargetModelWrapper."""
    print("\n" + "=" * 70)
    print("TEST 3: CircuitController Integration")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = "fp16" if device.type == "cuda" else "fp32"
    print(f"Device: {device}, Precision: {precision}")

    # 1. Setup modello
    cfg = NeuroTraceConfig(
        model_name_or_path="gpt2",
        device=str(device),
        precision=precision,
    )
    wrapper = TargetModelWrapper(cfg)
    print(f"✓ Loaded model: {cfg.model_name_or_path}")

    # 2. Setup registry
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_controller.db")
        registry = CircuitRegistry(db_path)

        # 3. Crea circuito di test (layer 6 e 9 di GPT-2)
        circuit = CircuitRecord(
            circuit_id="test_control_001",
            model_name="gpt2",
            components=[
                CircuitComponent(layer=6, component_type="attention_head", index=5),
                CircuitComponent(layer=9, component_type="attention_head", index=9),
            ],
            features=CircuitFeatures(
                sae_indices={
                    "layer_6": [100, 200],
                    "layer_9": [50, 150],
                }
            ),
            causal_metrics=CircuitCausalMetrics(vlo_mean=1.8, faithfulness=0.85),
            semantics=CircuitSemantics(
                task_tag="test_control",
                human_label="test_circuit",
                description="Test circuit for control plane validation",
            ),
        )
        registry.upsert(circuit)
        print(f"✓ Registered circuit: {circuit.circuit_id}")

        # 4. Setup controller
        feature_store = MockFeatureStore(hidden_dim=768, device=device)
        steering_builder = SteeringBuilder(feature_store=feature_store, device=device)
        controller = CircuitController(
            model_wrapper=wrapper,
            registry=registry,
            steering_builder=steering_builder,
            residual_position="post_mlp",
        )
        print("✓ Initialized CircuitController")

        # 5. List circuits
        circuits = controller.list_circuits(task_tag="test_control")
        assert len(circuits) == 1
        print(f"✓ Found {len(circuits)} circuit(s) with task_tag='test_control'")

        # 6. Enable circuit
        controller.enable_circuit("test_control_001", global_alpha=0.5)
        print("✓ Enabled circuit with alpha=0.5")

        # 7. Verifica active circuits
        summary = controller.active_circuits_summary()
        assert summary["count"] == 1
        assert "test_control_001" in [c["circuit_id"] for c in summary["circuits"]]
        print(f"✓ Active circuits: {summary['count']}")
        print(f"  Circuit: {summary['circuits'][0]['circuit_id']}")
        print(f"  Layers: {summary['circuits'][0]['layers']}")
        print(f"  Alphas: {summary['circuits'][0]['alpha_per_layer']}")

        # 8. Generazione con steering (baseline)
        prompt = "The capital of France is"
        print(f"\n🔹 Baseline generation (no steering):")
        controller.clear_all()
        output_baseline = controller.generate(prompt, max_new_tokens=10, temperature=0.0)
        print(f"  Input:  {prompt}")
        print(f"  Output: {output_baseline}")

        # 9. Generazione con steering attivo
        print(f"\n🔹 Steered generation (alpha=0.5):")
        controller.enable_circuit("test_control_001", global_alpha=0.5)
        output_steered = controller.generate(prompt, max_new_tokens=10, temperature=0.0)
        print(f"  Input:  {prompt}")
        print(f"  Output: {output_steered}")

        # 10. Verifica control trace
        trace = controller.last_trace()
        assert trace is not None
        assert trace.prompt == prompt
        assert "test_control_001" in trace.active_circuits
        print(f"\n✓ Control trace captured:")
        print(f"  Prompt: {trace.prompt[:50]}...")
        print(f"  Active circuits: {trace.active_circuits}")
        print(f"  Layer alphas: {trace.layer_alphas}")

        # 11. Disable circuit
        controller.disable_circuit("test_control_001")
        summary_after = controller.active_circuits_summary()
        assert summary_after["count"] == 0
        print(f"\n✓ Disabled circuit, active count: {summary_after['count']}")

    print("\n✅ CircuitController integration tests PASSED")


def test_multi_circuit_composition():
    """Test composizione multipli circuiti simultanei."""
    print("\n" + "=" * 70)
    print("TEST 4: Multi-Circuit Composition")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = "fp16" if device.type == "cuda" else "fp32"

    cfg = NeuroTraceConfig(
        model_name_or_path="gpt2",
        device=str(device),
        precision=precision,
    )
    wrapper = TargetModelWrapper(cfg)
    print(f"✓ Loaded model: {cfg.model_name_or_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_multi.db")
        registry = CircuitRegistry(db_path)

        # Crea 2 circuiti che operano su layer diversi
        circuit1 = CircuitRecord(
            circuit_id="circuit_layer3",
            model_name="gpt2",
            components=[CircuitComponent(layer=3, component_type="attention_head", index=0)],
            features=CircuitFeatures(sae_indices={"layer_3": [10, 20]}),
            causal_metrics=CircuitCausalMetrics(vlo_mean=1.0),
            semantics=CircuitSemantics(task_tag="test_multi"),
        )

        circuit2 = CircuitRecord(
            circuit_id="circuit_layer7",
            model_name="gpt2",
            components=[CircuitComponent(layer=7, component_type="mlp", index=0)],
            features=CircuitFeatures(sae_indices={"layer_7": [30, 40]}),
            causal_metrics=CircuitCausalMetrics(vlo_mean=1.2),
            semantics=CircuitSemantics(task_tag="test_multi"),
        )

        registry.upsert(circuit1)
        registry.upsert(circuit2)
        print(f"✓ Registered 2 circuits")

        # Setup controller
        feature_store = MockFeatureStore(hidden_dim=768, device=device)
        steering_builder = SteeringBuilder(feature_store=feature_store, device=device)
        controller = CircuitController(
            model_wrapper=wrapper,
            registry=registry,
            steering_builder=steering_builder,
        )

        # Enable entrambi
        controller.enable_circuit("circuit_layer3", global_alpha=0.3)
        controller.enable_circuit("circuit_layer7", global_alpha=0.6)
        print("✓ Enabled 2 circuits with different alphas")

        summary = controller.active_circuits_summary()
        assert summary["count"] == 2
        print(f"✓ Active circuits: {summary['count']}")

        for c in summary["circuits"]:
            print(f"  - {c['circuit_id']}: layers={c['layers']}, alphas={c['alpha_per_layer']}")

        # Genera con entrambi attivi
        output = controller.generate("Once upon a time", max_new_tokens=15, temperature=0.0)
        print(f"\n✓ Generated with 2 active circuits:")
        print(f"  {output[:100]}...")

        # Trace
        trace = controller.last_trace()
        assert len(trace.active_circuits) == 2
        print(f"✓ Trace captured {len(trace.active_circuits)} active circuits")

        controller.clear_all()
        assert controller.active_circuits_summary()["count"] == 0
        print("✓ Cleared all circuits")

    print("\n✅ Multi-circuit composition tests PASSED")


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    import logging
    import sys

    # Fix Windows console encoding issues
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("\n" + "=" * 70)
    print("NEUROTRACE CONTROL PLANE - INTEGRATION TESTS")
    print("=" * 70)

    try:
        test_circuit_registry()
        test_steering_builder()
        test_controller_integration()
        test_multi_circuit_composition()

        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED")
        print("=" * 70)
        print("\nControl Plane is ready for use!")
        print("\nNext steps:")
        print("1. Train SAE models (see SAEFeatureStore)")
        print("2. Run causal discovery to populate CircuitRegistry")
        print("3. Use CircuitController for active steering")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
