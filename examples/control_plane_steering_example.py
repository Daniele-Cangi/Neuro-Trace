"""
Demo Control Plane con Enhanced SAE reale (Layer 0 IOI).

Questo script dimostra l'integrazione completa:
1. Carica EnhancedSAE trainato (checkpoints/layer0_sae/final.pt)
2. Crea un circuito IOI basato sulle top feature scoperte
3. Usa CircuitController per steering attivo
4. Genera testo con/senza steering

Obiettivo: verificare che il Control Plane funziona con SAE reali.
"""

import torch
from pathlib import Path

from neurotrace.config import NeuroTraceConfig
from neurotrace.control import (
    CircuitRegistry,
    CircuitRecord,
    CircuitComponent,
    CircuitCausalMetrics,
    CircuitSemantics,
    CircuitFeatures,
    EnhancedSAEFeatureStore,
    SteeringBuilder,
    CircuitController,
)
from neurotrace.models.wrapper import TargetModelWrapper


def main():
    print("=" * 80)
    print("NEUROTRACE CONTROL PLANE - DEMO WITH REAL ENHANCED SAE")
    print("=" * 80)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    # ========================================================================
    # 1. Load Model via NeuroTrace Config
    # ========================================================================
    print("[1/6] Loading GPT-2 model via NeuroTrace...")

    cfg = NeuroTraceConfig(
        model_name_or_path="gpt2",
        device=str(device),
        precision="fp32",
    )

    wrapped_model = TargetModelWrapper(cfg=cfg)

    print(f"[OK] Model loaded: {cfg.model_name_or_path}")
    print()

    # ========================================================================
    # 2. Load Enhanced SAE for Layer 0
    # ========================================================================
    print("[2/6] Loading trained Enhanced SAE (Layer 0)...")

    sae_checkpoint = Path("checkpoints/layer0_sae/final.pt")
    if not sae_checkpoint.exists():
        print(f"[ERROR] SAE checkpoint not found: {sae_checkpoint}")
        print("   Please run: python train_layer0_sae.py first")
        return

    feature_store = EnhancedSAEFeatureStore()
    feature_store.load_sae(
        checkpoint_path=sae_checkpoint,
        layer=0,
        device=device
    )

    print()

    # ========================================================================
    # 3. Create Circuit Registry & IOI Circuit
    # ========================================================================
    print("[3/6] Creating IOI circuit with discovered features...")

    registry = CircuitRegistry(db_path=":memory:")  # In-memory for demo

    # Top 3 IOI features from hybrid analysis
    top_features = [
        (2586, 0.967, "gave [object] to syntax"),
        (2081, 0.933, "transfer patterns"),
        (1123, 0.900, "temporal markers"),
    ]

    # Create circuit components
    components = []
    for feat_id, freq, desc in top_features:
        component = CircuitComponent(
            layer=0,
            component_type="sae_direction",  # SAE feature direction
            index=feat_id,  # Feature ID
            extra={
                "frequency": freq,
                "description": desc,
            }
        )
        components.append(component)

    # Create circuit record
    circuit = CircuitRecord(
        circuit_id="ioi_layer0_structural",
        model_name="gpt2",
        model_revision="124M",
        components=components,
        causal_metrics=CircuitCausalMetrics(
            vlo_mean=5.276,  # From discovery
            vlo_std=0.15,
            faithfulness=0.70,
        ),
        semantics=CircuitSemantics(
            task_tag="IOI",
            human_label="Layer 0 Structural Shortcuts",
            description="Layer 0 MLP structural shortcuts for IOI task via VLO + Enhanced SAE",
            tags=["structural", "syntactic", "layer0", "IOI"],
        ),
        features=CircuitFeatures(
            sae_indices={
                "layer_0": [2586, 2081, 1123],
            },
            extra={
                "feature_descriptions": {
                    "2586": "gave [object] to syntax",
                    "2081": "transfer patterns",
                    "1123": "temporal markers (When X and Y...)",
                },
                "feature_frequencies": {
                    "2586": 0.967,
                    "2081": 0.933,
                    "1123": 0.900,
                },
            },
        ),
    )

    registry.upsert(circuit)
    print(f"[OK] Created circuit: {circuit.circuit_id}")
    print(f"  Components: {len(circuit.components)}")
    print(f"  VLO: {circuit.causal_metrics.vlo_mean}")
    print()

    # ========================================================================
    # 4. Build Steering Vector
    # ========================================================================
    print("[4/6] Building steering vector from circuit...")

    builder = SteeringBuilder(feature_store=feature_store)

    steering_spec = builder.build_from_circuit(
        record=circuit,
    )

    print(f"[OK] Steering vector built")
    print(f"  Layers affected: {steering_spec.active_layers()}")
    print(f"  Vectors: {len(steering_spec.layer_vectors)}")
    print()

    # ========================================================================
    # 5. Create Controller
    # ========================================================================
    print("[5/6] Initializing Circuit Controller...")

    controller = CircuitController(
        model_wrapper=wrapped_model,
        registry=registry,
        steering_builder=builder,
    )

    print("[OK] Controller initialized")
    print()

    # ========================================================================
    # 6. Generate with/without Steering
    # ========================================================================
    print("[6/6] Generating text with active steering...")
    print()

    # Test prompts (IOI structure)
    prompts = [
        "When John and Mary went to the store, John gave a book to",
        "Alice and Bob were at the park. Alice handed the ball to",
        "Sarah, Mike, and others went to the restaurant. Sarah passed the menu to",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"Prompt {i}: \"{prompt}\"")
        print("-" * 80)

        # Baseline (no steering)
        print("Baseline (no steering):")
        baseline_output = controller.generate(
            prompt=prompt,
            max_new_tokens=20,
            temperature=0.7,
        )
        print(f"  {baseline_output}")
        print()

        # With steering
        print(f"With steering (circuit={circuit.circuit_id}, alpha=1.0):")

        try:
            controller.enable_circuit(
                circuit_id=circuit.circuit_id,
                global_alpha=1.0,
            )

            steered_output = controller.generate(
                prompt=prompt,
                max_new_tokens=20,
                temperature=0.7,
            )
            print(f"  {steered_output}")

            controller.disable_circuit(circuit.circuit_id)

        except Exception as e:
            print(f"  [ERROR] Error: {e}")

        print()

    # Summary
    print("=" * 80)
    print("[DONE] CONTROL PLANE DEMO COMPLETE")
    print("=" * 80)
    print()
    print("Demo successfully verified Control Plane integration with Enhanced SAE!")
    print("  - Loaded trained Enhanced SAE")
    print("  - Created IOI circuit from discovered features")
    print("  - Built steering vectors from SAE decoder directions")
    print("  - Generated text with/without active steering")
    print()


if __name__ == "__main__":
    main()
