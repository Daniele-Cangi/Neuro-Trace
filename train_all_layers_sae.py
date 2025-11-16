# train_all_layers_sae.py

"""
Train Enhanced SAE on ALL 12 layers for complete neural cartography.

This script trains separate SAEs for each layer (0-11), enabling:
- Cross-layer feature comparison
- Complete 1:1 neural mapping
- Understanding information flow through network

Usage:
    python train_all_layers_sae.py \
        --activations_dir runs/deep_ioi_capture/.../activations \
        --epochs 10
"""

import sys
import json
import torch
import argparse
from pathlib import Path
from datetime import datetime

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from torch.utils.data import DataLoader

from neurotrace.training import (
    EnhancedSAE,
    create_enhanced_sae,
    EnhancedSAETrainer,
    EnhancedTrainingConfig,
    LayerActivationDataset,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Enhanced SAE on all 12 layers"
    )
    parser.add_argument(
        "--activations_dir",
        type=str,
        required=True,
        help="Path to deep capture activations directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs per layer",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/all_layers_sae",
        help="Output directory for trained SAEs",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layers to train (comma-separated, e.g., '0,6,9' or 'all')",
    )
    return parser.parse_args()


def train_layer_sae(
    layer_idx: int,
    activations_dir: Path,
    config: EnhancedTrainingConfig,
    output_dir: Path,
) -> dict:
    """Train Enhanced SAE for a single layer."""

    print("\n" + "=" * 80)
    print(f"TRAINING SAE FOR LAYER {layer_idx}")
    print("=" * 80)
    print()

    layer_name = f"layer_{layer_idx}.mlp"

    # Load dataset
    print(f"[1/3] Loading activations for {layer_name}...")
    try:
        dataset = LayerActivationDataset(
            activations_dir=str(activations_dir),
            layer_name=layer_name,
            flatten_sequences=True,
        )
        print(f"✓ Loaded {len(dataset)} activation samples")
    except Exception as e:
        print(f"❌ Failed to load dataset for {layer_name}: {e}")
        return None

    # Create SAE
    print(f"\n[2/3] Creating Enhanced SAE...")
    sae = create_enhanced_sae(
        input_dim=config.input_dim,
        dict_mult=config.dict_mult,
        k_sparse=config.k_sparse,
        sparsity_lambda=config.sparsity_lambda,
        use_jumprelu=config.use_jumprelu,
        normalize_decoder=True,
    )
    print(f"✓ SAE created: {sae.dict_size} features")

    # Create trainer
    trainer = EnhancedSAETrainer(sae, config)

    # Train
    print(f"\n[3/3] Training SAE for Layer {layer_idx}...")
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
    )

    try:
        trainer.train(dataloader)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted")
        return None

    # Save checkpoint
    layer_output_dir = output_dir / f"layer_{layer_idx}"
    layer_output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = layer_output_dir / "final.pt"
    trainer.save_checkpoint("final", save_dir=str(layer_output_dir))

    # Get metrics
    summary = trainer.get_metrics_summary()

    print(f"\n✓ Layer {layer_idx} SAE training complete:")
    print(f"  MSE: {summary['final_mse_loss']:.4f}")
    print(f"  Dead features: {summary['final_dead_fraction']:.1%}")
    print(f"  Checkpoint: {checkpoint_path}")

    return {
        'layer': layer_idx,
        'checkpoint_path': str(checkpoint_path),
        'metrics': summary,
    }


def main():
    args = parse_args()

    print("=" * 80)
    print("NEUROTRACE - ALL LAYERS SAE TRAINING")
    print("=" * 80)
    print()
    print("Goal: Complete neural cartography (1:1 mapping of all 12 layers)")
    print()

    device = args.device if torch.cuda.is_available() else "cpu"
    activations_dir = Path(args.activations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which layers to train
    if args.layers == "all":
        layers_to_train = list(range(12))
    else:
        layers_to_train = [int(x) for x in args.layers.split(',')]

    print(f"Configuration:")
    print(f"  Activations: {activations_dir}")
    print(f"  Layers: {layers_to_train}")
    print(f"  Epochs per layer: {args.epochs}")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")
    print()

    # Estimate time
    est_time_per_layer = args.epochs * 5  # ~5 min per epoch
    est_total_time = len(layers_to_train) * est_time_per_layer

    print(f"📊 Estimated total time: ~{est_total_time} minutes ({est_total_time/60:.1f} hours)")
    print()

    input("Press ENTER to start training (or Ctrl+C to cancel)...")
    print()

    # Training configuration
    config = EnhancedTrainingConfig(
        input_dim=768,
        dict_mult=4,
        k_sparse=64,
        use_jumprelu=False,
        sparsity_lambda=1e-3,
        ghost_grad_weight=0.1,
        learning_rate=3e-4,
        weight_decay=1e-5,
        grad_clip=1.0,
        num_epochs=args.epochs,
        warmup_steps=500,
        use_cosine_schedule=True,
        batch_size=args.batch_size,
        device=device,
        checkpoint_dir=str(output_dir),
        save_every_n_epochs=5,
        log_every_n_batches=100,
        evaluate_every_n_epochs=1,
    )

    # Train each layer
    results = []

    for layer_idx in layers_to_train:
        result = train_layer_sae(
            layer_idx=layer_idx,
            activations_dir=activations_dir,
            config=config,
            output_dir=output_dir,
        )

        if result is not None:
            results.append(result)

    # Save summary
    summary_path = output_dir / "training_summary.json"
    summary = {
        'timestamp': datetime.now().isoformat(),
        'layers_trained': layers_to_train,
        'num_epochs': args.epochs,
        'results': results,
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Final summary
    print("\n" + "=" * 80)
    print("✅ ALL LAYERS SAE TRAINING COMPLETE")
    print("=" * 80)
    print()

    print(f"Trained SAEs for {len(results)} layers:")
    for result in results:
        print(f"\n  Layer {result['layer']}:")
        print(f"    MSE: {result['metrics']['final_mse_loss']:.4f}")
        print(f"    Dead features: {result['metrics']['final_dead_fraction']:.1%}")
        print(f"    Checkpoint: {result['checkpoint_path']}")

    print()
    print(f"Summary saved to: {summary_path}")
    print()

    print("Next Steps:")
    print("  1. Cross-layer feature comparison")
    print("  2. Information flow analysis (how features transform across layers)")
    print("  3. Complete neural cartography visualization")
    print()

    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
