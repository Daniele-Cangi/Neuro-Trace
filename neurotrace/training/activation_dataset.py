# neurotrace/training/activation_dataset.py

from __future__ import annotations

import glob
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


@dataclass
class ActivationBatch:
    """
    Batch di attivazioni per un singolo layer.
    """
    layer_name: str
    activations: torch.Tensor  # [batch_size, seq_len, hidden_dim]
    example_ids: List[str]
    step_meta: Dict[str, any]


class ActivationDataset(IterableDataset):
    """
    Dataset PyTorch per caricare attivazioni salvate da Phase 1.

    Carica batch di attivazioni da file .pt e li emette come stream per training SAE.

    Layout atteso:
        activations_dir/
            batch_0001.pt
            batch_0002.pt
            ...

    Ogni file .pt contiene un dict:
        {
            "example_ids": List[str],
            "step_meta": Dict,
            "layer_0.block": Tensor[B, S, D],
            "layer_1.block": Tensor[B, S, D],
            ...
        }
    """

    def __init__(
        self,
        activations_dir: str,
        target_layer: Optional[str] = None,
        flatten_sequences: bool = True,
        device: str = "cpu",
        max_batches: Optional[int] = None,
    ) -> None:
        """
        Args:
            activations_dir: Directory contenente batch_*.pt files
            target_layer: Nome layer specifico (es. "layer_9.block"). Se None, itera tutti.
            flatten_sequences: Se True, appiattisce [B, S, D] → [B*S, D]
            device: Device su cui caricare tensori
            max_batches: Limita numero batch (per debug)
        """
        self.activations_dir = Path(activations_dir)
        self.target_layer = target_layer
        self.flatten_sequences = flatten_sequences
        self.device = torch.device(device)
        self.max_batches = max_batches

        # Trova tutti i batch files
        self.batch_files = sorted(glob.glob(str(self.activations_dir / "batch_*.pt")))

        if not self.batch_files:
            raise ValueError(f"No batch_*.pt files found in {activations_dir}")

        if self.max_batches:
            self.batch_files = self.batch_files[:self.max_batches]

        logger.info(
            f"ActivationDataset initialized: {len(self.batch_files)} batches, "
            f"target_layer={target_layer}, flatten={flatten_sequences}"
        )

    def __iter__(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        Yields: (layer_name, activations_tensor)
            - layer_name: es. "layer_9.block"
            - activations_tensor: [N, D] se flatten=True, altrimenti [B, S, D]
        """
        for batch_file in self.batch_files:
            try:
                batch_data = torch.load(batch_file, map_location=self.device)
            except Exception as e:
                logger.warning(f"Failed to load {batch_file}: {e}")
                continue

            # Extract layer activations
            for key, value in batch_data.items():
                # Skip metadata keys
                if key in ("example_ids", "step_meta"):
                    continue

                # Filter by target layer if specified
                if self.target_layer and key != self.target_layer:
                    continue

                if not isinstance(value, torch.Tensor):
                    continue

                # value can be [B, S, D] or already flattened [N, D]
                if value.dim() == 3:
                    # 3D: [B, S, D] - flatten if requested
                    activations = value.to(self.device)
                    if self.flatten_sequences:
                        B, S, D = activations.shape
                        activations = activations.reshape(B * S, D)  # [B*S, D]
                elif value.dim() == 2:
                    # 2D: already flattened [N, D]
                    activations = value.to(self.device)
                else:
                    logger.debug(f"Skipping {key}: unexpected shape {tuple(value.shape)}")
                    continue

                yield key, activations

    def __len__(self) -> int:
        """
        Approximate length (number of batch files).
        Note: actual number of yielded items depends on layers per batch.
        """
        return len(self.batch_files)

    def get_layer_names(self) -> List[str]:
        """
        Scansiona il primo batch per estrarre i nomi dei layer disponibili.
        """
        if not self.batch_files:
            return []

        try:
            batch_data = torch.load(self.batch_files[0], map_location="cpu")
            layer_names = [
                k for k in batch_data.keys()
                if k not in ("example_ids", "step_meta") and isinstance(batch_data[k], torch.Tensor)
            ]
            return sorted(layer_names)
        except Exception as e:
            logger.error(f"Failed to scan layer names: {e}")
            return []

    @staticmethod
    def estimate_hidden_dim(activations_dir: str) -> int:
        """
        Utility: stima hidden_dim dal primo batch trovato.
        """
        batch_files = glob.glob(str(Path(activations_dir) / "batch_*.pt"))
        if not batch_files:
            raise ValueError(f"No batch files found in {activations_dir}")

        batch_data = torch.load(batch_files[0], map_location="cpu")
        for key, value in batch_data.items():
            if key not in ("example_ids", "step_meta") and isinstance(value, torch.Tensor):
                if value.dim() == 3:
                    return value.shape[-1]  # D from [B, S, D]

        raise ValueError("Could not determine hidden_dim from batch files")


class LayerActivationDataset(ActivationDataset):
    """
    Specializzazione di ActivationDataset per un singolo layer.

    Ritorna direttamente tensori senza layer_name prefix.
    Utile per training SAE per-layer.
    """

    def __init__(
        self,
        activations_dir: str,
        layer_name: str,
        flatten_sequences: bool = True,
        device: str = "cpu",
        max_batches: Optional[int] = None,
    ) -> None:
        super().__init__(
            activations_dir=activations_dir,
            target_layer=layer_name,
            flatten_sequences=flatten_sequences,
            device=device,
            max_batches=max_batches,
        )
        self.layer_name = layer_name

    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Yields: activations_tensor [N, D] (senza layer_name prefix)
        """
        for _layer_name, activations in super().__iter__():
            yield activations
