# neurotrace/training/sae_checkpoint.py

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch

from neurotrace.state_indexer.sae_feature_extractor import LayerSparseAutoencoder

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """
    Metadata associati ad un checkpoint SAE.
    """
    layer_name: str
    model_name: str
    input_dim: int
    dict_size: int
    sparsity_lambda: float

    # Training info
    training_steps: int
    training_epochs: int
    final_loss: float
    final_sparsity: float

    # Timestamps
    created_at: str

    # Optional
    notes: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "CheckpointMetadata":
        return cls(**data)


class SAECheckpoint:
    """
    Utility per gestione checkpoint SAE con metadata.

    Estende il formato checkpoint base del trainer con informazioni aggiuntive
    per integrazione con Control Plane.
    """

    def __init__(self, checkpoint_dir: str) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        sae: LayerSparseAutoencoder,
        metadata: CheckpointMetadata,
        name: Optional[str] = None,
        optimizer_state: Optional[Dict] = None,
    ) -> Path:
        """
        Salva SAE + metadata in formato completo.

        Args:
            sae: LayerSparseAutoencoder da salvare
            metadata: CheckpointMetadata con info training
            name: Nome checkpoint (default: layer_name_timestamp)
            optimizer_state: Stato optimizer (opzionale)

        Returns:
            Path al checkpoint salvato
        """
        if name is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            name = f"{metadata.layer_name}_{timestamp}"

        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        metadata_path = self.checkpoint_dir / f"{name}_meta.json"

        # Save model checkpoint
        checkpoint = {
            "state_dict": sae.state_dict(),
            "config": {
                "input_dim": sae.input_dim,
                "dict_size": sae.dict_size,
                "sparsity_lambda": sae.sparsity_lambda,
            },
            "metadata": metadata.to_dict(),
        }

        if optimizer_state:
            checkpoint["optimizer_state"] = optimizer_state

        torch.save(checkpoint, checkpoint_path)

        # Save metadata as JSON (for easy browsing)
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        logger.info(f"✓ SAE checkpoint saved: {checkpoint_path}")
        logger.info(f"  Metadata: {metadata_path}")

        return checkpoint_path

    def load(
        self,
        name: str,
        device: str = "cpu",
    ) -> tuple[LayerSparseAutoencoder, CheckpointMetadata]:
        """
        Carica SAE + metadata da checkpoint.

        Args:
            name: Nome checkpoint (senza .pt)
            device: Device target

        Returns:
            (sae, metadata)
        """
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Reconstruct SAE
        config = checkpoint["config"]
        sae = LayerSparseAutoencoder(
            input_dim=config["input_dim"],
            dict_size=config["dict_size"],
            sparsity_lambda=config.get("sparsity_lambda", 1e-3),
        )

        sae.load_state_dict(checkpoint["state_dict"])
        sae.to(device)
        sae.eval()

        # Load metadata
        metadata = CheckpointMetadata.from_dict(checkpoint["metadata"])

        logger.info(f"✓ Loaded SAE: {name}")
        logger.info(f"  Layer: {metadata.layer_name}, Input dim: {metadata.input_dim}, Dict size: {metadata.dict_size}")
        logger.info(f"  Training: {metadata.training_epochs} epochs, {metadata.training_steps} steps")
        logger.info(f"  Final loss: {metadata.final_loss:.4f}, Final sparsity: {metadata.final_sparsity:.1f}")

        return sae, metadata

    def list_checkpoints(self) -> list[str]:
        """
        Lista tutti i checkpoint disponibili.

        Returns:
            Lista di nomi checkpoint (senza .pt)
        """
        checkpoint_files = sorted(self.checkpoint_dir.glob("*.pt"))
        return [f.stem for f in checkpoint_files if not f.stem.endswith("_meta")]

    def get_metadata(self, name: str) -> Optional[CheckpointMetadata]:
        """
        Carica solo metadata senza SAE weights.

        Args:
            name: Nome checkpoint

        Returns:
            CheckpointMetadata o None se non trovato
        """
        metadata_path = self.checkpoint_dir / f"{name}_meta.json"

        if not metadata_path.exists():
            # Fallback: carica da .pt file
            checkpoint_path = self.checkpoint_dir / f"{name}.pt"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                if "metadata" in checkpoint:
                    return CheckpointMetadata.from_dict(checkpoint["metadata"])
            return None

        with open(metadata_path) as f:
            data = json.load(f)

        return CheckpointMetadata.from_dict(data)

    def delete(self, name: str) -> None:
        """
        Elimina checkpoint + metadata.
        """
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        metadata_path = self.checkpoint_dir / f"{name}_meta.json"

        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"Deleted checkpoint: {checkpoint_path}")

        if metadata_path.exists():
            metadata_path.unlink()
            logger.info(f"Deleted metadata: {metadata_path}")


def load_saes_for_model(
    checkpoint_dir: str,
    model_name: str,
    device: str = "cpu",
) -> Dict[str, LayerSparseAutoencoder]:
    """
    Utility: carica tutti i SAE per un modello specifico.

    Args:
        checkpoint_dir: Directory checkpoint
        model_name: Nome modello (es. "gpt2")
        device: Device target

    Returns:
        Dict {layer_name: SAE}
    """
    checkpoint_manager = SAECheckpoint(checkpoint_dir)
    all_checkpoints = checkpoint_manager.list_checkpoints()

    saes = {}
    for name in all_checkpoints:
        metadata = checkpoint_manager.get_metadata(name)
        if metadata and metadata.model_name == model_name:
            sae, _ = checkpoint_manager.load(name, device=device)
            saes[metadata.layer_name] = sae

    logger.info(f"Loaded {len(saes)} SAE(s) for model '{model_name}'")
    return saes
