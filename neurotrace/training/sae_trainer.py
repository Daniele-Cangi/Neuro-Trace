# neurotrace/training/sae_trainer.py

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from neurotrace.state_indexer.sae_feature_extractor import LayerSparseAutoencoder

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Configurazione per training SAE.
    """
    # Model
    input_dim: int
    dict_mult: int = 4  # dict_size = dict_mult * input_dim
    sparsity_lambda: float = 1e-3

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 256
    num_epochs: int = 10
    grad_clip: float = 1.0

    # LR scheduling
    use_cosine_schedule: bool = True
    min_lr_factor: float = 0.1  # min_lr = lr * min_lr_factor

    # Device
    device: str = "cuda"

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_batches: Optional[int] = 1000
    save_every_n_epochs: int = 1

    # Logging
    log_every_n_batches: int = 100


@dataclass
class TrainingMetrics:
    """
    Metriche accumulate durante training.
    """
    epoch: int
    batch: int
    total_batches: int

    # Loss components
    mse_loss: float
    l1_loss: float
    total_loss: float

    # Reconstruction quality
    reconstruction_error: float  # MSE
    sparsity: float  # Mean L0 (num non-zero activations)

    # Learning rate
    current_lr: float

    def __str__(self) -> str:
        return (
            f"[Epoch {self.epoch} | Batch {self.batch}/{self.total_batches}] "
            f"Loss: {self.total_loss:.4f} (MSE: {self.mse_loss:.4f}, L1: {self.l1_loss:.4f}) | "
            f"Recon: {self.reconstruction_error:.4f} | "
            f"Sparsity: {self.sparsity:.1f} | "
            f"LR: {self.current_lr:.2e}"
        )


class SAETrainer:
    """
    Trainer per Sparse Autoencoder.

    Training loop con:
    - MSE reconstruction loss
    - L1 sparsity penalty
    - Gradient clipping
    - Cosine LR scheduling
    - Checkpointing
    """

    def __init__(
        self,
        sae: LayerSparseAutoencoder,
        config: TrainingConfig,
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

        # LR scheduler
        self.scheduler: Optional[CosineAnnealingLR] = None

        # Metrics history
        self.metrics_history: List[TrainingMetrics] = []
        self.global_step = 0

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"SAETrainer initialized: {config.input_dim} → {sae.dict_size}")
        logger.info(f"Device: {self.device}, Optimizer: Adam, LR: {config.learning_rate}")

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> List[TrainingMetrics]:
        """
        Allena per un'epoca.

        Args:
            dataloader: DataLoader che ritorna batch [N, input_dim]
            epoch: Numero epoca corrente (1-indexed)

        Returns:
            Lista di TrainingMetrics per i batch loggati
        """
        self.sae.train()
        epoch_metrics = []
        total_batches = len(dataloader)

        for batch_idx, batch in enumerate(dataloader, start=1):
            # batch può essere (layer_name, activations) o solo activations
            if isinstance(batch, (tuple, list)):
                activations = batch[1] if len(batch) > 1 else batch[0]
            else:
                activations = batch

            # Move to device
            activations = activations.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.sae(activations)

            # Loss computation
            mse_loss = torch.mean((output["reconstruction"] - activations) ** 2)
            l1_loss = torch.mean(torch.abs(output["codes"]))
            total_loss = mse_loss + self.config.sparsity_lambda * l1_loss

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.sae.parameters(), self.config.grad_clip)

            self.optimizer.step()

            # Metrics
            with torch.no_grad():
                reconstruction_error = mse_loss.item()
                sparsity = torch.mean((output["codes"].abs() > 1e-6).float()).item() * output["codes"].shape[1]
                current_lr = self.optimizer.param_groups[0]["lr"]

            metrics = TrainingMetrics(
                epoch=epoch,
                batch=batch_idx,
                total_batches=total_batches,
                mse_loss=mse_loss.item(),
                l1_loss=l1_loss.item(),
                total_loss=total_loss.item(),
                reconstruction_error=reconstruction_error,
                sparsity=sparsity,
                current_lr=current_lr,
            )

            self.metrics_history.append(metrics)
            self.global_step += 1

            # Logging
            if batch_idx % self.config.log_every_n_batches == 0:
                logger.info(str(metrics))
                epoch_metrics.append(metrics)

            # Checkpointing (batch-based)
            if (
                self.config.save_every_n_batches
                and self.global_step % self.config.save_every_n_batches == 0
            ):
                self._save_checkpoint(f"step_{self.global_step}")

        return epoch_metrics

    def train(
        self,
        dataloader: DataLoader,
        num_epochs: Optional[int] = None,
    ) -> None:
        """
        Training loop completo.

        Args:
            dataloader: DataLoader per activation dataset
            num_epochs: Numero epoche (override config)
        """
        num_epochs = num_epochs or self.config.num_epochs

        # Setup LR scheduler (needs total steps)
        if self.config.use_cosine_schedule:
            total_steps = len(dataloader) * num_epochs
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.config.learning_rate * self.config.min_lr_factor,
            )

        logger.info(f"Starting training: {num_epochs} epochs, {len(dataloader)} batches/epoch")

        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n{'=' * 70}")
            logger.info(f"EPOCH {epoch}/{num_epochs}")
            logger.info(f"{'=' * 70}")

            epoch_metrics = self.train_epoch(dataloader, epoch)

            # LR step (if using scheduler)
            if self.scheduler:
                self.scheduler.step()

            # Epoch-based checkpointing
            if epoch % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(f"epoch_{epoch}")

            # Summary
            if epoch_metrics:
                avg_loss = sum(m.total_loss for m in epoch_metrics) / len(epoch_metrics)
                avg_sparsity = sum(m.sparsity for m in epoch_metrics) / len(epoch_metrics)
                logger.info(
                    f"Epoch {epoch} Summary: Avg Loss={avg_loss:.4f}, "
                    f"Avg Sparsity={avg_sparsity:.1f}"
                )

        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)

        # Final checkpoint
        self._save_checkpoint("final")

    def _save_checkpoint(self, name: str) -> None:
        """
        Salva checkpoint del SAE.
        """
        checkpoint_path = self.checkpoint_dir / f"sae_{name}.pt"
        torch.save(
            {
                "state_dict": self.sae.state_dict(),
                "config": {
                    "input_dim": self.sae.input_dim,
                    "dict_size": self.sae.dict_size,
                    "sparsity_lambda": self.sae.sparsity_lambda,
                },
                "training_config": self.config.__dict__,
                "global_step": self.global_step,
                "optimizer_state": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )
        logger.info(f"✓ Checkpoint saved: {checkpoint_path}")

    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        device: str = "cpu",
    ) -> LayerSparseAutoencoder:
        """
        Carica SAE da checkpoint.

        Args:
            checkpoint_path: Path al file .pt
            device: Device su cui caricare

        Returns:
            LayerSparseAutoencoder caricato
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Reconstruct SAE
        config = checkpoint["config"]
        sae = LayerSparseAutoencoder(
            input_dim=config["input_dim"],
            dict_size=config["dict_size"],
            sparsity_lambda=config.get("sparsity_lambda", 1e-3),
        )

        # Load weights
        sae.load_state_dict(checkpoint["state_dict"])
        sae.to(device)
        sae.eval()

        logger.info(f"✓ Loaded SAE from {checkpoint_path}")
        logger.info(f"  Config: {config}")

        return sae

    def get_metrics_summary(self) -> Dict[str, float]:
        """
        Ritorna summary delle metriche finali.
        """
        if not self.metrics_history:
            return {}

        recent_metrics = self.metrics_history[-100:]  # last 100 batches
        return {
            "final_total_loss": sum(m.total_loss for m in recent_metrics) / len(recent_metrics),
            "final_mse_loss": sum(m.mse_loss for m in recent_metrics) / len(recent_metrics),
            "final_l1_loss": sum(m.l1_loss for m in recent_metrics) / len(recent_metrics),
            "final_reconstruction_error": sum(m.reconstruction_error for m in recent_metrics) / len(recent_metrics),
            "final_sparsity": sum(m.sparsity for m in recent_metrics) / len(recent_metrics),
            "total_steps": self.global_step,
        }
