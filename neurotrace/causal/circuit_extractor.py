# neurotrace/causal/circuit_extractor.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch

from neurotrace.control.circuit_registry import (
    CircuitRecord,
    CircuitComponent,
    CircuitCausalMetrics,
    CircuitSemantics,
    CircuitFeatures,
)
from .vlo_tester import VLOResult

logger = logging.getLogger(__name__)


@dataclass
class ComponentSpec:
    """
    Specification di un componente per circuit extraction.
    """
    layer: int
    component_type: str  # "attention_head", "mlp", "residual"
    index: int  # head index, or 0 for full layer
    vlo: float
    faithfulness: float


class CircuitExtractor:
    """
    Extractor che converte componenti causalmente validati in CircuitRecord.

    Workflow:
    1. VLOTester → List[VLOResult]
    2. Filter per VLO threshold
    3. Extract SAE features (opzionale)
    4. Build CircuitRecord
    5. Save to CircuitRegistry
    """

    def __init__(
        self,
        min_vlo: float = 0.5,
        min_faithfulness: float = 0.3,
    ) -> None:
        """
        Args:
            min_vlo: Minimum VLO threshold per considerare componente importante
            min_faithfulness: Minimum faithfulness per validazione causale
        """
        self.min_vlo = min_vlo
        self.min_faithfulness = min_faithfulness

    def extract_from_vlo_results(
        self,
        vlo_results: List[VLOResult],
        circuit_id: str,
        model_name: str,
        task_tag: str,
        human_label: str = "",
        description: str = "",
        examples: Optional[List[str]] = None,
        sae_features: Optional[Dict[str, List[int]]] = None,
        geometric_features: Optional[Dict[str, any]] = None,
    ) -> CircuitRecord:
        """
        Extract CircuitRecord from VLO test results.

        Args:
            vlo_results: VLO test results per components
            circuit_id: Unique ID per circuito
            model_name: Nome modello
            task_tag: Tag task (es. "ioi", "greater_than")
            human_label: Label human-readable
            description: Descrizione circuito
            examples: Esempi di input per task
            sae_features: SAE feature indices per layer {layer_name: [indices]}
            geometric_features: Geometric properties (LID, spectral, etc.)

        Returns:
            CircuitRecord pronto per registry
        """
        # Filter components by thresholds
        valid_results = [
            r for r in vlo_results
            if r.vlo >= self.min_vlo and r.faithfulness >= self.min_faithfulness
        ]

        if not valid_results:
            logger.warning(
                f"No components passed thresholds (min_vlo={self.min_vlo}, "
                f"min_faithfulness={self.min_faithfulness})"
            )

        logger.info(
            f"Circuit extraction: {len(valid_results)}/{len(vlo_results)} "
            f"components passed thresholds"
        )

        # Build components
        components = []
        for result in valid_results:
            # Parse component_name: "layer_9.attention_head.5"
            parts = result.component_name.split(".")
            try:
                layer = int(parts[0].split("_")[1])
                comp_type = parts[1]
                index = int(parts[2]) if len(parts) > 2 else 0
            except (IndexError, ValueError):
                logger.warning(f"Failed to parse component_name: {result.component_name}")
                continue

            component = CircuitComponent(
                layer=layer,
                component_type=comp_type,
                index=index,
                extra={"vlo": result.vlo, "faithfulness": result.faithfulness},
            )
            components.append(component)

        # Aggregate causal metrics
        vlo_values = [r.vlo for r in valid_results]
        faithfulness_values = [r.faithfulness for r in valid_results]

        causal_metrics = CircuitCausalMetrics(
            vlo_mean=float(torch.tensor(vlo_values).mean()) if vlo_values else 0.0,
            vlo_std=float(torch.tensor(vlo_values).std()) if vlo_values else 0.0,
            faithfulness=float(torch.tensor(faithfulness_values).mean()) if faithfulness_values else 0.0,
            causal_scrubbing=0.0,  # placeholder
            effect_size_vs_random=0.0,  # placeholder
        )

        # Build features
        features = CircuitFeatures(
            sae_indices=sae_features or {},
            geometric=geometric_features or {},
        )

        # Build semantics
        semantics = CircuitSemantics(
            task_tag=task_tag,
            human_label=human_label,
            description=description,
            examples=examples or [],
        )

        # Build record
        record = CircuitRecord(
            circuit_id=circuit_id,
            model_name=model_name,
            model_revision="unknown",
            components=components,
            features=features,
            causal_metrics=causal_metrics,
            semantics=semantics,
            created_at=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        )

        return record

    def filter_by_vlo(
        self,
        vlo_results: List[VLOResult],
        top_k: Optional[int] = None,
    ) -> List[VLOResult]:
        """
        Filter and sort VLO results by importance.

        Args:
            vlo_results: List of VLO results
            top_k: Keep only top-k by VLO (None = all)

        Returns:
            Filtered and sorted results
        """
        # Filter by thresholds
        filtered = [
            r for r in vlo_results
            if r.vlo >= self.min_vlo and r.faithfulness >= self.min_faithfulness
        ]

        # Sort by VLO descending
        sorted_results = sorted(filtered, key=lambda r: r.vlo, reverse=True)

        # Top-k
        if top_k is not None:
            sorted_results = sorted_results[:top_k]

        return sorted_results


def extract_circuit_from_components(
    components: List[Tuple[int, str, int]],
    circuit_id: str,
    model_name: str,
    task_tag: str,
    vlo_mean: float = 1.0,
    faithfulness: float = 0.8,
    **kwargs,
) -> CircuitRecord:
    """
    Utility: crea CircuitRecord da lista di componenti senza VLO testing.

    Utile per circuiti hardcoded (es. IOI noto da letteratura).

    Args:
        components: List of (layer, component_type, index)
        circuit_id: Unique ID
        model_name: Model name
        task_tag: Task tag
        vlo_mean: Placeholder VLO
        faithfulness: Placeholder faithfulness
        **kwargs: Additional metadata (human_label, description, examples)

    Returns:
        CircuitRecord
    """
    circuit_components = [
        CircuitComponent(layer=layer, component_type=comp_type, index=idx)
        for layer, comp_type, idx in components
    ]

    causal_metrics = CircuitCausalMetrics(
        vlo_mean=vlo_mean,
        faithfulness=faithfulness,
    )

    semantics = CircuitSemantics(
        task_tag=task_tag,
        human_label=kwargs.get("human_label", ""),
        description=kwargs.get("description", ""),
        examples=kwargs.get("examples", []),
    )

    features = CircuitFeatures(
        sae_indices=kwargs.get("sae_indices", {}),
        geometric=kwargs.get("geometric", {}),
    )

    record = CircuitRecord(
        circuit_id=circuit_id,
        model_name=model_name,
        components=circuit_components,
        causal_metrics=causal_metrics,
        semantics=semantics,
        features=features,
        created_at=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    )

    return record
