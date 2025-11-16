# cli/neuro_control_run.py

"""
Script CLI per usare il Control Plane su un modello già strumentato.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

from neurotrace.models.wrapper import TargetModelWrapper
from neurotrace.control import (
    CircuitRegistry,
    SteeringBuilder,
    CircuitController,
    SAEFeatureStore,
)
from neurotrace.state_indexer.sae_feature_extractor import SAEFeatureExtractor
from neurotrace.config import NeuroTraceConfig, SAEConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NeuroTrace Control Plane – run generation with circuit steering"
    )
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--registry_db", type=str, required=True)
    p.add_argument(
        "--circuit_ids",
        type=str,
        nargs="+",
        help="One or more circuit IDs to activate",
        required=True,
    )
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument(
        "--prompt",
        type=str,
        required=False,
        help="Prompt text (if omitted, read from stdin)",
    )
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--precision", type=str, default="auto", choices=["fp32", "fp16", "bf16", "auto"])
    p.add_argument("--use_mock_sae", action="store_true", help="Use mock SAE for testing (no trained SAE required)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Precision setup
    if args.precision == "auto":
        precision = "fp16" if device.type == "cuda" else "fp32"
    else:
        precision = args.precision

    print(f"[INFO] Device: {device}, Precision: {precision}", file=sys.stderr)

    # 1) Carica modello
    cfg = NeuroTraceConfig(
        model_name_or_path=args.model_name_or_path,
        device=str(device),
        precision=precision,
    )
    model_wrapper = TargetModelWrapper(cfg)
    print(f"[INFO] Loaded model: {args.model_name_or_path}", file=sys.stderr)

    # 2) Registry
    registry = CircuitRegistry(db_path=args.registry_db)
    print(f"[INFO] Connected to registry: {args.registry_db}", file=sys.stderr)

    # 3) FeatureStore + SteeringBuilder
    if args.use_mock_sae:
        print("[WARN] Using MockFeatureStore - steering vectors will be random!", file=sys.stderr)
        # Import mock from test
        from test_control_plane import MockFeatureStore
        feature_store = MockFeatureStore(hidden_dim=768, device=device)
    else:
        # Real SAE integration
        print("[INFO] Initializing SAEFeatureExtractor...", file=sys.stderr)
        sae_config = SAEConfig()
        sae_extractor = SAEFeatureExtractor(
            cfg=cfg,
            sae_cfg=sae_config,
        )
        # Note: SAE devono essere già addestrati/caricati
        # Se non esistono, verrà creata auto-init (pesi random)
        feature_store = SAEFeatureStore(sae_extractor)
        print(f"[INFO] SAEFeatureStore initialized with {len(sae_extractor.saes)} SAE layers", file=sys.stderr)

    steering_builder = SteeringBuilder(feature_store=feature_store, device=device)

    # 4) Controller
    controller = CircuitController(
        model_wrapper=model_wrapper,
        registry=registry,
        steering_builder=steering_builder,
    )
    print("[INFO] CircuitController initialized", file=sys.stderr)

    # 5) Prompt
    prompt: str
    if args.prompt:
        prompt = args.prompt
    else:
        print("Enter prompt (Ctrl+D to end):", file=sys.stderr)
        prompt = sys.stdin.read().strip()

    print(f"\n[PROMPT] {prompt}\n", file=sys.stderr)

    # 6) Attiva circuiti
    for cid in args.circuit_ids:
        try:
            controller.enable_circuit(cid, global_alpha=args.alpha)
            print(f"[INFO] Enabled circuit: {cid} (alpha={args.alpha})", file=sys.stderr)
        except ValueError as e:
            print(f"[ERROR] Failed to enable circuit '{cid}': {e}", file=sys.stderr)
            return 1

    # 7) Genera
    try:
        output = controller.generate(
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,  # greedy for deterministic steering
        )
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1

    # Output su stdout (separato da logging)
    print(output)

    # 8) Summary su stderr
    summary = controller.active_circuits_summary()
    trace = controller.last_trace()

    print("\n" + "=" * 70, file=sys.stderr)
    print("[CONTROL SUMMARY]", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"Active circuits: {summary['count']}", file=sys.stderr)
    for circ in summary["circuits"]:
        print(f"  • {circ['circuit_id']}", file=sys.stderr)
        print(f"    Task: {circ['task_tag']}", file=sys.stderr)
        print(f"    Label: {circ['label']}", file=sys.stderr)
        print(f"    Layers: {circ['layers']}", file=sys.stderr)
        print(f"    Alphas: {circ['alpha_per_layer']}", file=sys.stderr)

    if trace:
        print(f"\nGenerated {len(output.split())} words", file=sys.stderr)
        print(f"Active during generation: {trace.active_circuits}", file=sys.stderr)

    print("=" * 70 + "\n", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
