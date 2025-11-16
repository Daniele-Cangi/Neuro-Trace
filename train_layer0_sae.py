# train_layer0_sae.py

"""
Train Enhanced SAE on Layer 0 MLP using deep capture data.

This script trains a publication-quality SAE on 100K IOI examples.

Usage:
    python train_layer0_sae.py
"""

import sys
import torch
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from torch.utils.data import Dataset, DataLoader

from neurotrace.training import (
    create_enhanced_sae,
    EnhancedSAETrainer,
    EnhancedTrainingConfig,
)


class DeepCaptureDataset(Dataset):
    """Dataset for deep capture activations."""

    def __init__(self, activations_dir: str, layer_name: str = "layer_0.mlp"):
        self.activations_dir = Path(activations_dir)
        self.layer_name = layer_name

        # Find all batch files
        self.batch_files = sorted(self.activations_dir.glob("batch_*.pt"))

        if len(self.batch_files) == 0:
            raise ValueError(f"No batch files found in {activations_dir}")

        # Load all activations into memory (since we have enough RAM)
        print(f"Loading activations from {len(self.batch_files)} batches...")
        all_activations = []

        for i, batch_file in enumerate(self.batch_files):
            if i % 100 == 0:
                print(f"  Loading batch {i+1}/{len(self.batch_files)}...")

            batch_data = torch.load(batch_file)

            if self.layer_name not in batch_data:
                raise KeyError(f"Layer {self.layer_name} not found in {batch_file}")

            # batch_data[layer_name] is [batch, seq, 768]
            # Flatten to [batch*seq, 768]
            layer_acts = batch_data[self.layer_name]
            if len(layer_acts.shape) == 3:
                layer_acts = layer_acts.reshape(-1, layer_acts.shape[-1])

            all_activations.append(layer_acts)

        # Concatenate all
        self.activations = torch.cat(all_activations, dim=0)

        print(f"✓ Loaded {self.activations.shape[0]:,} activation vectors")
        print(f"  Shape: {self.activations.shape}")
        print(f"  Memory: {self.activations.element_size() * self.activations.nelement() / (1024**3):.2f} GB")

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return self.activations[idx]


def main():
    print("=" * 80)
    print("NEUROTRACE - ENHANCED SAE TRAINING (Layer 0 MLP)")
    print("=" * 80)
    print()
    print("Dataset: 100K IOI examples (deep capture)")
    print("Architecture: SOTA Enhanced SAE")
    print("Features:")
    print("  ✅ Decoder normalization (Anthropic 2023)")
    print("  ✅ Ghost gradients (resurrect dead features)")
    print("  ✅ Top-K activation (exact sparsity)")
    print("  ✅ Pre-bias correction")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use deep capture data
    activations_dir = Path("runs/deep_ioi_capture/20251116_171258/activations")

    if not activations_dir.exists():
        print(f"❌ Activations not found: {activations_dir}")
        print("Please run: run_deep_capture.bat")
        return

    print(f"Configuration:")
    print(f"  Activations: {activations_dir}")
    print(f"  Device: {device}")
    print()

    # ========================================================================
    # Load Dataset
    # ========================================================================

    print("[1/3] Loading deep capture dataset...")

    dataset = DeepCaptureDataset(
        activations_dir=str(activations_dir),
        layer_name="layer_0.mlp",
    )

    input_dim = dataset.activations.shape[-1]
    print()

    # ========================================================================
    # Create Enhanced SAE
    # ========================================================================

    print("[2/3] Creating Enhanced SAE...")

    dict_mult = 4
    k_sparse = 64

    sae = create_enhanced_sae(
        input_dim=input_dim,
        dict_mult=dict_mult,
        k_sparse=k_sparse,
        sparsity_lambda=1e-3,
        use_jumprelu=False,
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

    print("[3/3] Training Enhanced SAE...")

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
        warmup_steps=1000,
        use_cosine_schedule=True,
        batch_size=512,  # Large batch for 100K dataset
        # Device
        device=device,
        # Checkpointing
        checkpoint_dir="checkpoints/layer0_sae",
        save_every_n_epochs=2,
        # Logging
        log_every_n_batches=100,
        evaluate_every_n_epochs=1,
    )

    trainer = EnhancedSAETrainer(sae, config)

    print(f"Training configuration:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Warmup steps: {config.warmup_steps}")
    print()

    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print()

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True if device == "cuda" else False,
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
        print("✅ Good! <15% dead features (publication-ready)")
    else:
        print("⚠️  Consider training longer or tuning hyperparameters")

    print()
    print(f"Checkpoints saved to: {config.checkpoint_dir}")
    print()

    print("Next Steps:")
    print("  1. python setup_saelens.py (install SAELens)")
    print("  2. python hybrid_sae_analysis.py --enhanced_sae_path checkpoints/layer0_sae/final.pt")
    print("  3. Compare Layer 0 vs Layer 9 features")
    print()

    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
