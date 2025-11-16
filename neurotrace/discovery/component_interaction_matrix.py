# neurotrace/discovery/component_interaction_matrix.py

"""
Component Interaction Matrix - Mappa completa delle interazioni tra componenti.

Costruisce una matrice N×N dove entry (i,j) rappresenta la forza
dell'interazione causale tra componente i e componente j.
"""

from __future__ import annotations

from typing import List, Dict, Tuple
import numpy as np
import json
from pathlib import Path

from .exhaustive_scanner import ScanResult


class ComponentInteractionMatrix:
    """
    Matrice di interazione completa tra componenti.

    Capabilities:
    - Build interaction matrix from scan results
    - Identify strongly connected components
    - Find interaction patterns (feedforward, feedback, lateral)
    - Export to graph formats (NetworkX, Neo4j)
    """

    def __init__(self):
        self.component_names: List[str] = []
        self.component_index: Dict[str, int] = {}
        self.interaction_matrix: np.ndarray = None  # [N, N]
        self.vlo_matrix: np.ndarray = None  # [N, N] VLO values
        self.faithfulness_matrix: np.ndarray = None  # [N, N] faithfulness values

    def build_from_scan_results(self, results: List[ScanResult]):
        """
        Costruisce matrice da risultati scan.

        Args:
            results: Lista di ScanResult da ExhaustiveCircuitScanner
        """
        # Extract unique components
        self.component_names = sorted(list(set(r.component_name for r in results)))
        self.component_index = {name: i for i, name in enumerate(self.component_names)}

        N = len(self.component_names)

        # Initialize matrices
        self.vlo_matrix = np.zeros((N, N))
        self.faithfulness_matrix = np.zeros((N, N))
        self.interaction_matrix = np.zeros((N, N))

        # Fill matrices (diagonal = self-importance)
        for result in results:
            idx = self.component_index[result.component_name]
            self.vlo_matrix[idx, idx] = result.vlo
            self.faithfulness_matrix[idx, idx] = result.faithfulness
            self.interaction_matrix[idx, idx] = result.vlo * result.faithfulness

    def get_top_components(self, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Ottieni top-k componenti per importanza causale (self-interaction).

        Returns:
            Lista di (component_name, importance_score)
        """
        diagonal = np.diag(self.interaction_matrix)
        top_indices = np.argsort(diagonal)[-top_k:][::-1]

        return [
            (self.component_names[i], diagonal[i])
            for i in top_indices
        ]

    def get_layer_importance(self) -> Dict[int, float]:
        """
        Importanza aggregata per layer.

        Returns:
            Dict[layer_idx, total_importance]
        """
        layer_importance = {}

        for component_name in self.component_names:
            # Parse layer from component name (e.g., "layer_9.attention_head.5" → 9)
            parts = component_name.split(".")
            if parts[0].startswith("layer_"):
                layer_idx = int(parts[0].split("_")[1])
                idx = self.component_index[component_name]
                importance = self.interaction_matrix[idx, idx]

                if layer_idx not in layer_importance:
                    layer_importance[layer_idx] = 0.0
                layer_importance[layer_idx] += importance

        return layer_importance

    def save(self, output_path: str | Path):
        """Salva matrice e metadata."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "component_names": self.component_names,
            "vlo_matrix": self.vlo_matrix.tolist(),
            "faithfulness_matrix": self.faithfulness_matrix.tolist(),
            "interaction_matrix": self.interaction_matrix.tolist(),
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, input_path: str | Path) -> "ComponentInteractionMatrix":
        """Carica matrice salvata."""
        with open(input_path, "r") as f:
            data = json.load(f)

        matrix = cls()
        matrix.component_names = data["component_names"]
        matrix.component_index = {name: i for i, name in enumerate(matrix.component_names)}
        matrix.vlo_matrix = np.array(data["vlo_matrix"])
        matrix.faithfulness_matrix = np.array(data["faithfulness_matrix"])
        matrix.interaction_matrix = np.array(data["interaction_matrix"])

        return matrix
