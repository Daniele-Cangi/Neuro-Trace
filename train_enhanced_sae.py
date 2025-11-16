# train_enhanced_sae.py

"""
Train SOTA Enhanced SAE on captured Layer 0 MLP activations.

This script uses the state-of-the-art SAE implementation with:
- Decoder normalization
- Ghost gradients
- Top-K activation
- Pre-bias correction

Usage:
    python train_enhanced_sae.py
"""

import sys
import torch
from pathlib import Path

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


def main():
    print("=" * 80)
    print("NEUROTRACE - ENHANCED SAE TRAINING (SOTA)")
    print("=" * 80)
    print()
    print("Features:")
    print("  ✅ Decoder weight normalization (Anthropic 2023)")
    print("  ✅ Ghost gradients for dead features (Anthropic 2023)")
    print("  ✅ Top-K activation (Gao et al. 2024)")
    print("  ✅ Pre-bias correction (Anthropic 2024)")
    print("  ✅ Advanced learning rate scheduling")
    print()

    # ========================================================================
    # Configuration
    # ========================================================================

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use Phase 1 activations
    activations_dir = Path("runs/phase1_ioi_activations/activations")

    if not activations_dir.exists():
        print(f"❌ Activations not found: {activations_dir}")
        print("Please run Phase 1 capture first (run_phase1.bat)")
        return

    print(f"Activations directory: {activations_dir}")
    print(f"Device: {device}")
    print()

    # ========================================================================
    # Load Dataset
    # ========================================================================

    print("[1/4] Loading activation dataset...")

    # Note: Phase 1 saves compressed activations, but we can work with them
    # For now, we'll estimate hidden_dim from first batch
    first_batch_path = list(activations_dir.glob("batch_*.pt"))[0]
    first_batch = torch.load(first_batch_path)

    # Phase 1 saves compressed, so we need to check the format
    if 'compressed_activations' in first_batch:
        # Compressed format - get projection dim
        comp_data = first_batch['compressed_activations']
        first_layer = list(comp_data.keys())[0]
        proj_dim = comp_data[first_layer]['proj_dim']
        print(f"  ⚠️  Using compressed activations (dim={proj_dim})")
        print(f"  Note: For best results, recapture with no compression")
        input_dim = proj_dim
    else:
        # Direct activations
        input_dim = 768  # GPT-2 default
        print(f"  ✓ Using full activations (dim={input_dim})")

    # For this demo, we'll use Layer 0 which should be "layer_0.block"
    layer_name = "layer_0.block"

    try:
        dataset = LayerActivationDataset(
            activations_dir=str(activations_dir),
            layer_name=layer_name,
            flatten_sequences=True,
        )
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        print(f"   Available layers in first batch: {list(comp_data.keys())}")
        return

    print(f"✓ Loaded dataset for layer: {layer_name}")
    print(f"  Hidden dim: {input_dim}")
    print()

    # ========================================================================
    # Create Enhanced SAE
    # ========================================================================

    print("[2/4] Creating Enhanced SAE...")

    dict_mult = 4  # Dictionary size = 4x input dim
    k_sparse = max(32, (input_dim * dict_mult) // 32)  # ~3% sparsity

    sae = create_enhanced_sae(
        input_dim=input_dim,
        dict_mult=dict_mult,
        k_sparse=k_sparse,
        sparsity_lambda=1e-3,
        use_jumprelu=False,  # Start with top-k, can enable later
        ghost_threshold=1e-5,
        normalize_decoder=True,
    )

    print(f"✓ Enhanced SAE created:")
    print(f"  Input dim: {input_dim}")
    print(f"  Dictionary size: {sae.dict_size}")
    print(f"  Sparsity (k): {k_sparse}")
    print(f"  Parameters: {sum(p.numel() for p in sae.parameters()):,}")
    print()

    # ========================================================================
    # Create Trainer
    # ========================================================================

    print("[3/4] Setting up trainer...")

    config = EnhancedTrainingConfig(
        input_dim=input_dim,
        dict_mult=dict_mult,
        k_sparse=k_sparse,
        use_jumprelu=False,
        # Loss weights
        sparsity_lambda=1e-3,
        ghost_grad_weight=0.1,
        # Optimization
        learning_rate=3e-4,
        weight_decay=1e-5,
        grad_clip=1.0,
        # Training
        num_epochs=10,
        warmup_steps=500,
        use_cosine_schedule=True,
        batch_size=256,
        # Device
        device=device,
        # Checkpointing
        checkpoint_dir="checkpoints/enhanced_sae",
        save_every_n_epochs=2,
        # Logging
        log_every_n_batches=50,
        evaluate_every_n_epochs=1,
    )

    trainer = EnhancedSAETrainer(sae, config)

    print(f"✓ Trainer configured:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Warmup steps: {config.warmup_steps}")
    print()

    # ========================================================================
    # Train
    # ========================================================================

    print("[4/4] Training Enhanced SAE...")
    print("=" * 80)
    print()

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
    )

    try:
        trainer.train(dataloader)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint("interrupted")
        print("✓ Checkpoint saved")
        return

    # ========================================================================
    # Summary
    # ========================================================================

    print()
    print("=" * 80)
    print("✅ ENHANCED SAE TRAINING COMPLETE")
    print("=" * 80)
    print()

    summary = trainer.get_metrics_summary()
    print("Final Metrics:")
    print(f"  Total loss: {summary['final_total_loss']:.4f}")
    print(f"  MSE loss: {summary['final_mse_loss']:.4f}")
    print(f"  L1 loss: {summary['final_l1_loss']:.4f}")
    print(f"  Ghost loss: {summary['final_ghost_loss']:.4f}")
    print(f"  L0 sparsity: {summary['final_l0_sparsity']:.1f}")
    print(f"  Dead features: {summary['final_dead_fraction']:.1%}")
    print(f"  Total steps: {summary['total_steps']}")
    print()

    # Feature statistics
    stats = sae.get_feature_statistics()
    print("Feature Quality:")
    print(f"  Total features: {sae.dict_size}")
    print(f"  Active features: {sae.dict_size - stats['num_dead']}")
    print(f"  Dead features: {stats['num_dead']} ({stats['num_dead']/sae.dict_size:.1%})")
    print()

    if stats['num_dead'] / sae.dict_size < 0.05:
        print("✅ Excellent! <5% dead features (SOTA quality)")
    elif stats['num_dead'] / sae.dict_size < 0.15:
        print("✅ Good! <15% dead features (acceptable)")
    else:
        print("⚠️  High dead feature rate - consider tuning hyperparameters")

    print()
    print(f"Checkpoints saved to: {config.checkpoint_dir}")
    print()

    # Next steps
    print("Next Steps:")
    print("  1. Analyze feature monosemanticity")
    print("  2. Visualize top activating examples per feature")
    print("  3. Integrate into Control Plane for steering")
    print("  4. Compare with basic SAE")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
