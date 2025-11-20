# neurotrace/training/enhanced_sae_trainer.py

"""
Advanced trainer for EnhancedSAE.

Includes:
- Multi-stage training (warmup → main → fine-tune)
- Automatic learning rate scheduling
- Dead feature resurrection
- Feature quality monitoring
- Monosemanticity evaluation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from neurotrace.training.enhanced_sae import EnhancedSAE

logger = logging.getLogger(__name__)


@dataclass
class EnhancedTrainingConfig:
    """Configuration for EnhancedSAE training."""

    # Model
    input_dim: int
    dict_mult: int = 4
    k_sparse: int = 64
    use_jumprelu: bool = False

    # Loss weights
    sparsity_lambda: float = 1e-3
    ghost_grad_weight: float = 0.1

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # Training schedule
    num_epochs: int = 10
    warmup_steps: int = 1000
    use_cosine_schedule: bool = True

    # Batch size
    batch_size: int = 256

    # Device
    device: str = "cuda"

    # Checkpointing
    checkpoint_dir: str = "checkpoints/enhanced_sae"
    save_every_n_epochs: int = 1

    # Logging
    log_every_n_batches: int = 100
    evaluate_every_n_epochs: int = 1


@dataclass
class EnhancedTrainingMetrics:
    """Metrics for a training step."""
    epoch: int
    batch: int
    total_batches: int

    # Loss components
    mse_loss: float
    l1_loss: float
    ghost_loss: float
    total_loss: float

    # Sparsity metrics
    l0_sparsity: float  # Average number of active features
    dead_fraction: float  # Fraction of dead features

    # Learning rate
    current_lr: float

    def __str__(self) -> str:
        return (
            f"[Epoch {self.epoch} | Batch {self.batch}/{self.total_batches}] "
            f"Loss: {self.total_loss:.4f} (MSE: {self.mse_loss:.4f}, L1: {self.l1_loss:.4f}, Ghost: {self.ghost_loss:.4f}) | "
            f"L0: {self.l0_sparsity:.1f} | Dead: {self.dead_fraction:.1%} | LR: {self.current_lr:.2e}"
        )


class EnhancedSAETrainer:
    """
    Advanced trainer for EnhancedSAE.

    Features:
    - Decoder weight normalization after each step
    - Ghost gradient tracking and resurrection
    - Multi-stage learning rate schedule
    - Feature quality evaluation
    """

    def __init__(
        self,
        sae: EnhancedSAE,
        config: EnhancedTrainingConfig,
    ) -> None:
        self.sae = sae
        self.config = config
        self.device = torch.device(config.device)

        # Move model to device
        self.sae.to(self.device)

        # Optimizer
        self.optimizer = Adam(
            self.sae.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Metrics history
        self.metrics_history: List[EnhancedTrainingMetrics] = []
        self.global_step = 0

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"EnhancedSAETrainer initialized")
        logger.info(f"  Model: {config.input_dim} → {sae.dict_size} (k={config.k_sparse})")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Learning rate: {config.learning_rate}")
        logger.info(f"  Sparsity lambda: {config.sparsity_lambda}")
        logger.info(f"  Ghost grad weight: {config.ghost_grad_weight}")

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler with warmup + cosine decay."""
        if not self.config.use_cosine_schedule:
            return None

        # We'll set total_steps after we know the dataset size
        # For now, return None and create in train()
        return None

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> List[EnhancedTrainingMetrics]:
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader returning batches [N, input_dim]
            epoch: Current epoch number (1-indexed)

        Returns:
            List of metrics for logged batches
        """
        self.sae.train()
        epoch_metrics = []
        
        # Get total batches - must work correctly for proper training
        try:
            total_batches = len(dataloader)
            print(f"  → Dataloader has {total_batches} batches")
            logger.info(f"Dataloader reports {total_batches} batches")
        except (TypeError, AttributeError) as e:
            print(f"  ✗ Failed to get dataloader length: {e}")
            logger.error(f"Failed to get dataloader length: {e}")
            raise RuntimeError("Cannot determine number of batches - dataset must implement __len__")

        batch_count = 0
        for batch_idx, batch in enumerate(dataloader, start=1):
            batch_count += 1
            # Extract activations from batch
            # batch can be (layer_name, activations) or just activations
            if isinstance(batch, (list, tuple)):
                activations = batch[1]  # (layer_name, activations)
            else:
                activations = batch

            # Move to device
            activations = activations.to(self.device)

            # ============================================================
            # Forward pass
            # ============================================================
            output = self.sae(activations)

            # ============================================================
            # Compute loss
            # ============================================================
            total_loss, loss_metrics = self.sae.compute_loss(
                activations,
                output,
                ghost_grad_weight=self.config.ghost_grad_weight,
                l1_weight=self.config.sparsity_lambda,
            )

            # ============================================================
            # Backward pass
            # ============================================================
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.sae.parameters(),
                    self.config.grad_clip
                )

            self.optimizer.step()

            # ============================================================
            # CRITICAL: Normalize decoder weights after each step
            # ============================================================
            self.sae.normalize_decoder_step()

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            self.global_step += 1

            # ============================================================
            # Logging
            # ============================================================
            if batch_idx % self.config.log_every_n_batches == 0:
                current_lr = self.optimizer.param_groups[0]['lr']

                metrics = EnhancedTrainingMetrics(
                    epoch=epoch,
                    batch=batch_idx,
                    total_batches=total_batches,
                    mse_loss=loss_metrics['mse'],
                    l1_loss=loss_metrics['l1'],
                    ghost_loss=loss_metrics['ghost'],
                    total_loss=loss_metrics['total'],
                    l0_sparsity=loss_metrics['l0_sparsity'],
                    dead_fraction=loss_metrics['dead_fraction'],
                    current_lr=current_lr,
                )

                print(f"  {metrics}")  # Print directly to console
                logger.info(str(metrics))
                epoch_metrics.append(metrics)
                self.metrics_history.append(metrics)
        
        print(f"  → Processed {batch_count} batches total")
        return epoch_metrics

    def train(
        self,
        dataloader: DataLoader,
        num_epochs: Optional[int] = None,
    ) -> None:
        """
        Full training loop.

        Args:
            dataloader: DataLoader for training data
            num_epochs: Number of epochs (defaults to config.num_epochs)
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        # Create scheduler now that we know dataset size
        if self.config.use_cosine_schedule and self.scheduler is None:
            total_steps = len(dataloader) * num_epochs

            # Warmup scheduler
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config.warmup_steps
            )

            # Cosine scheduler
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - self.config.warmup_steps,
                eta_min=self.config.learning_rate * 0.1
            )

            # Combine
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.config.warmup_steps]
            )

        logger.info(f"Starting training: {num_epochs} epochs, {len(dataloader)} batches/epoch")
        logger.info("=" * 80)

        for epoch in range(1, num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{num_epochs}")
            logger.info("-" * 80)

            epoch_metrics = self.train_epoch(dataloader, epoch)

            # Epoch summary
            if epoch_metrics:
                avg_loss = sum(m.total_loss for m in epoch_metrics) / len(epoch_metrics)
                avg_l0 = sum(m.l0_sparsity for m in epoch_metrics) / len(epoch_metrics)
                avg_dead = sum(m.dead_fraction for m in epoch_metrics) / len(epoch_metrics)

                logger.info(f"\nEpoch {epoch} Summary:")
                logger.info(f"  Avg Loss: {avg_loss:.4f}")
                logger.info(f"  Avg L0: {avg_l0:.1f}")
                logger.info(f"  Dead features: {avg_dead:.1%}")

            # Checkpoint saving
            if epoch % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(f"epoch_{epoch}")
                logger.info(f"  Checkpoint saved: epoch_{epoch}")

            # Feature statistics
            if epoch % self.config.evaluate_every_n_epochs == 0:
                stats = self.sae.get_feature_statistics()
                logger.info(f"  Feature activation rates: min={stats['activation_rates'].min():.4f}, "
                           f"max={stats['activation_rates'].max():.4f}, "
                           f"median={stats['activation_rates'].median():.4f}")

        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)

        # Final checkpoint
        self.save_checkpoint("final")
        logger.info(f"Final checkpoint saved")

        # Final statistics
        final_stats = self.sae.get_feature_statistics()
        logger.info(f"\nFinal Feature Statistics:")
        logger.info(f"  Total features: {self.sae.dict_size}")
        logger.info(f"  Dead features: {final_stats['num_dead']} ({final_stats['dead_fraction']:.1%})")
        logger.info(f"  Active features: {self.sae.dict_size - final_stats['num_dead']}")

    def save_checkpoint(self, name: str) -> Path:
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"

        # Get feature statistics
        stats = self.sae.get_feature_statistics()

        checkpoint = {
            'model_state_dict': self.sae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': {
                'input_dim': self.config.input_dim,
                'dict_size': self.sae.dict_size,
                'k_sparse': self.config.k_sparse,
                'sparsity_lambda': self.config.sparsity_lambda,
                'use_jumprelu': self.config.use_jumprelu,
            },
            'training_state': {
                'global_step': self.global_step,
                'num_forward_passes': self.sae.num_forward_passes.item(),
            },
            'feature_statistics': {
                'activation_counts': stats['activation_counts'].cpu(),
                'activation_rates': stats['activation_rates'].cpu(),
                'num_dead': stats['num_dead'],
            },
            'metrics_history': self.metrics_history,
        }

        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        config: EnhancedTrainingConfig,
        device: str = "cuda",
    ) -> EnhancedSAETrainer:
        """Load model and trainer from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create SAE
        sae = EnhancedSAE(
            input_dim=checkpoint['config']['input_dim'],
            dict_size=checkpoint['config']['dict_size'],
            k_sparse=checkpoint['config']['k_sparse'],
            sparsity_lambda=checkpoint['config']['sparsity_lambda'],
            use_jumprelu=checkpoint['config'].get('use_jumprelu', False),
        )

        # Load state
        sae.load_state_dict(checkpoint['model_state_dict'])

        # Create trainer
        trainer = cls(sae, config)

        # Load optimizer and scheduler state
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and trainer.scheduler:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restore training state
        trainer.global_step = checkpoint['training_state']['global_step']
        sae.num_forward_passes = torch.tensor(
            checkpoint['training_state']['num_forward_passes'],
            device=device
        )

        # Restore feature statistics
        sae.feature_activation_count = checkpoint['feature_statistics']['activation_counts'].to(device)

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"  Global step: {trainer.global_step}")
        logger.info(f"  Dead features: {checkpoint['feature_statistics']['num_dead']}")

        return trainer

    def get_metrics_summary(self) -> Dict[str, float]:
        """Get summary of training metrics."""
        if not self.metrics_history:
            return {}

        last_metrics = self.metrics_history[-1]

        return {
            'final_total_loss': last_metrics.total_loss,
            'final_mse_loss': last_metrics.mse_loss,
            'final_l1_loss': last_metrics.l1_loss,
            'final_ghost_loss': last_metrics.ghost_loss,
            'final_l0_sparsity': last_metrics.l0_sparsity,
            'final_dead_fraction': last_metrics.dead_fraction,
            'total_steps': self.global_step,
        }
