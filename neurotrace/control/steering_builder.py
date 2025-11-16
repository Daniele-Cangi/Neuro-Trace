# neurotrace/control/steering_builder.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Protocol, Tuple

import torch

from .circuit_registry import CircuitRecord


class FeatureStore(Protocol):
    """
    Minimal interface to retrieve SAE directions or feature vectors
    associated with a circuit's components.

    This adapter ti permette di collegarti alla tua implementazione reale
    (es. sae_feature_extractor + vector_state_db).
    """

    def get_sae_directions(
        self,
        model_name: str,
        layer: int,
        feature_indices: List[int],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Return tensor of shape [num_features, hidden_dim] with basis vectors
        for the given SAE feature indices in that layer.
        """
        ...


@dataclass
class LayerSteeringVector:
    """
    Steering vector for a single layer: direction in residual space
    + default alpha scale.
    """
    layer: int
    direction: torch.Tensor         # [hidden_dim]
    default_alpha: float
    alpha_bounds: Tuple[float, float] = (-1.0, 1.0)


@dataclass
class SteeringSpec:
    """
    Full steering specification derived from a circuit.

    This è l'oggetto che il controller userà per applicare hook
    sul residual stream.
    """
    circuit_id: str
    model_name: str
    layer_vectors: Dict[int, LayerSteeringVector] = field(default_factory=dict)
    description: str = ""
    task_tag: str = ""
    semantics_label: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)

    def active_layers(self) -> List[int]:
        return sorted(self.layer_vectors.keys())


class SteeringBuilder:
    """
    Costruisce steering vectors a partire da CircuitRecord + FeatureStore.

    Strategia base: per ogni layer coinvolto nel circuito:
      - prendi gli SAE directions indicati
      - fai media pesata (se disponibile) o media semplice
      - normalizzi
    """

    def __init__(
        self,
        feature_store: FeatureStore,
        default_alpha: float = 0.7,
        alpha_bounds: Tuple[float, float] = (-2.0, 2.0),
        device: torch.device | None = None,
    ) -> None:
        self.feature_store = feature_store
        self.default_alpha = default_alpha
        self.alpha_bounds = alpha_bounds
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_from_circuit(
        self,
        record: CircuitRecord,
        per_layer_scaling: Mapping[int, float] | None = None,
    ) -> SteeringSpec:
        """
        Costruisce uno SteeringSpec pronto per l'uso.

        Args:
            record: CircuitRecord prodotto da critical_path_extractor.
            per_layer_scaling: opzionale fattore moltiplicativo alpha per layer.

        Returns:
            SteeringSpec
        """
        per_layer_scaling = per_layer_scaling or {}
        layer_vectors: Dict[int, LayerSteeringVector] = {}

        # record.features.sae_indices es: {"layer_12": [15, 42, 103], ...}
        sae_indices = record.features.sae_indices or {}

        for layer_key, indices in sae_indices.items():
            if not indices:
                continue

            # layer_key es: "layer_12"
            if layer_key.startswith("layer_"):
                try:
                    layer_idx = int(layer_key.split("_", 1)[1])
                except ValueError:
                    # fallback: skip se naming non standard
                    continue
            else:
                # fallback: assume it's just an int string
                try:
                    layer_idx = int(layer_key)
                except ValueError:
                    continue

            directions = self.feature_store.get_sae_directions(
                model_name=record.model_name,
                layer=layer_idx,
                feature_indices=list(set(indices)),
                device=self.device,
            )  # [n_features, hidden_dim]

            if directions.ndim != 2:
                raise ValueError(
                    f"Expected directions [N, hidden_dim] for layer {layer_idx},"
                    f" got {tuple(directions.shape)}"
                )

            # media semplice come default (puoi sostituire con pesi VLO/attr)
            vec = directions.mean(dim=0)  # [hidden_dim]
            norm = torch.norm(vec)
            if norm < 1e-6:
                # niente direzione significativa → skip
                continue
            vec = vec / norm

            alpha_base = per_layer_scaling.get(layer_idx, self.default_alpha)
            lv = LayerSteeringVector(
                layer=layer_idx,
                direction=vec.detach().clone().to(self.device),
                default_alpha=alpha_base,
                alpha_bounds=self.alpha_bounds,
            )
            layer_vectors[layer_idx] = lv

        return SteeringSpec(
            circuit_id=record.circuit_id,
            model_name=record.model_name,
            layer_vectors=layer_vectors,
            description=record.semantics.description,
            task_tag=record.semantics.task_tag,
            semantics_label=record.semantics.human_label,
            metadata={
                "created_at": record.created_at,
                "source_model_revision": record.model_revision,
            },
        )
