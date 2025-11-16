# cli/train_sae.py

"""
CLI per training Sparse Autoencoders (SAE) da attivazioni Phase 1.

Usage:
    python cli/train_sae.py \
        --activations_dir runs/phase1_capture/activations \
        --layer_name layer_9.block \
        --output_dir checkpoints/sae \
        --model_name gpt2 \
        --epochs 10 \
        --batch_size 256

"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from neurotrace.state_indexer.sae_feature_extractor import LayerSparseAutoencoder
from neurotrace.training import (
    LayerActivationDataset,
    SAETrainer,
    TrainingConfig,
    SAECheckpoint,
    CheckpointMetadata,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Sparse Autoencoder on captured activations"
    )

    # Data
    p.add_argument("--activations_dir", type=str, required=True,
                   help="Directory with batch_*.pt activation files")
    p.add_argument("--layer_name", type=str, required=True,
                   help="Layer name to train SAE for (e.g., 'layer_9.block')")
    p.add_argument("--model_name", type=str, default="gpt2",
                   help="Model name for metadata")

    # Output
    p.add_argument("--output_dir", type=str, default="checkpoints/sae",
                   help="Output directory for checkpoints")

    # SAE architecture
    p.add_argument("--dict_mult", type=int, default=4,
                   help="Dictionary size multiplier (dict_size = dict_mult * input_dim)")
    p.add_argument("--sparsity_lambda", type=float, default=1e-3,
                   help="L1 sparsity penalty weight")

    # Training
    p.add_argument("--epochs", type=int, default=10,
                   help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=256,
                   help="Batch size for training")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-5,
                   help="Weight decay")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Gradient clipping value")

    # Scheduling
    p.add_argument("--no_cosine_schedule", action="store_true",
                   help="Disable cosine LR scheduling")
    p.add_argument("--min_lr_factor", type=float, default=0.1,
                   help="Min LR factor for cosine schedule")

    # Checkpointing
    p.add_argument("--save_every_n_batches", type=int, default=None,
                   help="Save checkpoint every N batches (None = disabled)")
    p.add_argument("--save_every_n_epochs", type=int, default=1,
                   help="Save checkpoint every N epochs")

    # Logging
    p.add_argument("--log_every_n_batches", type=int, default=100,
                   help="Log metrics every N batches")

    # Device
    p.add_argument("--device", type=str, default="auto",
                   help="Device (auto/cuda/cpu)")

    # Data limits (for debugging)
    p.add_argument("--max_batches", type=int, default=None,
                   help="Limit number of batches (for debugging)")

    # Resume
    p.add_argument("--resume_from", type=str, default=None,
                   help="Resume training from checkpoint name")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(args.output_dir) / "training.log", mode="a"),
        ],
    )
    logger = logging.getLogger(__name__)

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger.info("=" * 70)
    logger.info("SAE TRAINING - NEUROTRACE")
    logger.info("=" * 70)
    logger.info(f"Activations: {args.activations_dir}")
    logger.info(f"Layer: {args.layer_name}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 70)

    # 1. Determine input_dim from data
    try:
        from neurotrace.training.activation_dataset import ActivationDataset
        input_dim = ActivationDataset.estimate_hidden_dim(args.activations_dir)
        logger.info(f"✓ Detected input_dim: {input_dim}")
    except Exception as e:
        logger.error(f"Failed to determine input_dim: {e}")
        return 1

    # 2. Create dataset
    logger.info(f"Loading dataset for layer: {args.layer_name}")
    dataset = LayerActivationDataset(
        activations_dir=args.activations_dir,
        layer_name=args.layer_name,
        flatten_sequences=True,
        device=device,
        max_batches=args.max_batches,
    )

    # 3. Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,  # IterableDataset doesn't support multi-worker
    )
    logger.info(f"✓ DataLoader created: {len(dataset)} batch files")

    # 4. Create or load SAE
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint_manager = SAECheckpoint(args.output_dir)
        sae, metadata = checkpoint_manager.load(args.resume_from, device=device)
        logger.info(f"✓ Resumed SAE: {metadata.layer_name}")
    else:
        dict_size = args.dict_mult * input_dim
        logger.info(f"Creating new SAE: {input_dim} → {dict_size}")
        sae = LayerSparseAutoencoder(
            input_dim=input_dim,
            dict_size=dict_size,
            sparsity_lambda=args.sparsity_lambda,
        )

    # 5. Create training config
    training_config = TrainingConfig(
        input_dim=input_dim,
        dict_mult=args.dict_mult,
        sparsity_lambda=args.sparsity_lambda,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        grad_clip=args.grad_clip,
        use_cosine_schedule=not args.no_cosine_schedule,
        min_lr_factor=args.min_lr_factor,
        device=device,
        checkpoint_dir=args.output_dir,
        save_every_n_batches=args.save_every_n_batches,
        save_every_n_epochs=args.save_every_n_epochs,
        log_every_n_batches=args.log_every_n_batches,
    )

    # 6. Create trainer
    trainer = SAETrainer(sae, training_config)

    # 7. Train
    try:
        logger.info("\nStarting training...\n")
        trainer.train(dataloader, num_epochs=args.epochs)
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user!")
        trainer._save_checkpoint("interrupted")
    except Exception as e:
        logger.error(f"\nTraining failed: {e}", exc_info=True)
        return 1

    # 8. Save final checkpoint with metadata
    logger.info("\nSaving final checkpoint with metadata...")
    summary = trainer.get_metrics_summary()

    metadata = CheckpointMetadata(
        layer_name=args.layer_name,
        model_name=args.model_name,
        input_dim=input_dim,
        dict_size=sae.dict_size,
        sparsity_lambda=args.sparsity_lambda,
        training_steps=summary.get("total_steps", 0),
        training_epochs=args.epochs,
        final_loss=summary.get("final_total_loss", 0.0),
        final_sparsity=summary.get("final_sparsity", 0.0),
        created_at=datetime.utcnow().isoformat(),
        notes=f"Trained on {args.activations_dir}",
    )

    checkpoint_manager = SAECheckpoint(args.output_dir)
    final_path = checkpoint_manager.save(
        sae=sae,
        metadata=metadata,
        name=f"{args.layer_name}_final",
        optimizer_state=trainer.optimizer.state_dict(),
    )

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Final checkpoint: {final_path}")
    logger.info(f"Training summary:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
