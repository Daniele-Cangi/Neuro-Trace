# neurotrace/visualization/circuit_graph.py

"""
Visualizzazione interattiva di circuiti causali come grafi.

Usa Pyvis per creare grafi HTML interattivi con:
- Nodi: componenti del circuito (attention heads, MLPs)
- Archi: flusso causale
- Colori: VLO intensity
- Dimensioni: faithfulness
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any
import networkx as nx

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

from neurotrace.control.circuit_registry import CircuitRecord, CircuitComponent


class CircuitGraphVisualizer:
    """
    Visualizza circuiti causali come grafi interattivi.

    Usa Pyvis per creare grafi HTML navigabili con drag-and-drop,
    zoom, e layout automatico (physics-based o hierarchical).
    """

    def __init__(
        self,
        height: str = "750px",
        width: str = "100%",
        bgcolor: str = "#222222",
        font_color: str = "white",
        directed: bool = True,
    ):
        """
        Args:
            height: Altezza grafo HTML
            width: Larghezza grafo HTML
            bgcolor: Colore background
            font_color: Colore font
            directed: Se True, archi direzionali
        """
        if not PYVIS_AVAILABLE:
            raise ImportError(
                "pyvis not installed. Install with: pip install pyvis"
            )

        self.height = height
        self.width = width
        self.bgcolor = bgcolor
        self.font_color = font_color
        self.directed = directed

    def visualize_circuit(
        self,
        circuit: CircuitRecord,
        output_path: str | Path,
        show_metrics: bool = True,
        layout: str = "hierarchical",  # "hierarchical" or "physics"
        node_color_by: str = "vlo",  # "vlo", "faithfulness", "layer"
    ) -> Path:
        """
        Visualizza un singolo circuito.

        Args:
            circuit: CircuitRecord da visualizzare
            output_path: Path file HTML output
            show_metrics: Mostra metriche nei tooltip
            layout: "hierarchical" (layer-based) o "physics" (force-directed)
            node_color_by: Criterio per colore nodi

        Returns:
            Path al file HTML generato
        """
        output_path = Path(output_path)

        # Crea network Pyvis
        net = Network(
            height=self.height,
            width=self.width,
            bgcolor=self.bgcolor,
            font_color=self.font_color,
            directed=self.directed,
            notebook=False,
        )

        # Configura layout
        if layout == "hierarchical":
            net.toggle_physics(False)
            net.set_options("""
            {
                "layout": {
                    "hierarchical": {
                        "enabled": true,
                        "direction": "UD",
                        "sortMethod": "directed",
                        "levelSeparation": 150
                    }
                }
            }
            """)
        else:
            net.toggle_physics(True)

        # Aggiungi nodi per ogni componente
        for comp in circuit.components:
            node_id = comp.component_name

            # Determina colore basato su criterio
            if node_color_by == "vlo":
                color = self._vlo_to_color(comp.vlo)
            elif node_color_by == "faithfulness":
                color = self._faithfulness_to_color(comp.faithfulness)
            else:  # layer
                color = self._layer_to_color(comp.layer_idx)

            # Dimensione basata su faithfulness
            size = 20 + comp.faithfulness * 30  # Range: 20-50

            # Tooltip con metriche
            if show_metrics:
                title = (
                    f"{comp.component_name}\n"
                    f"VLO: {comp.vlo:.3f}\n"
                    f"Faithfulness: {comp.faithfulness:.3f}\n"
                    f"Layer: {comp.layer_idx}\n"
                    f"Type: {comp.component_type}"
                )
            else:
                title = comp.component_name

            # Forma basata su tipo
            if comp.component_type == "attention_head":
                shape = "dot"
            elif comp.component_type == "mlp":
                shape = "square"
            else:
                shape = "triangle"

            # Livello gerarchico (se hierarchical layout)
            level = comp.layer_idx if layout == "hierarchical" else None

            net.add_node(
                node_id,
                label=f"L{comp.layer_idx}.{comp.component_type[:4]}",
                title=title,
                color=color,
                size=size,
                shape=shape,
                level=level,
            )

        # Aggiungi archi (flusso tra layer consecutivi)
        components_sorted = sorted(circuit.components, key=lambda c: c.layer_idx)
        for i in range(len(components_sorted) - 1):
            src = components_sorted[i]
            dst = components_sorted[i + 1]

            # Edge weight basato su VLO medio
            weight = (src.vlo + dst.vlo) / 2
            edge_width = 1 + weight * 3  # Range: 1-4

            net.add_edge(
                src.component_name,
                dst.component_name,
                width=edge_width,
                title=f"VLO: {weight:.3f}",
            )

        # Aggiungi titolo
        net.heading = f"Circuit: {circuit.semantics.human_label}"

        # Salva HTML
        net.save_graph(str(output_path))
        return output_path

    def visualize_multi_circuits(
        self,
        circuits: List[CircuitRecord],
        output_path: str | Path,
        merge_mode: str = "union",  # "union" o "intersection"
    ) -> Path:
        """
        Visualizza più circuiti contemporaneamente.

        Args:
            circuits: Lista di CircuitRecord
            output_path: Path file HTML
            merge_mode: "union" (tutti i nodi) o "intersection" (solo nodi comuni)

        Returns:
            Path al file HTML
        """
        output_path = Path(output_path)

        net = Network(
            height=self.height,
            width=self.width,
            bgcolor=self.bgcolor,
            font_color=self.font_color,
            directed=self.directed,
            notebook=False,
        )

        # Raccogli tutti i componenti
        all_components: Dict[str, List[CircuitComponent]] = {}
        for circuit in circuits:
            for comp in circuit.components:
                comp_name = comp.component_name
                if comp_name not in all_components:
                    all_components[comp_name] = []
                all_components[comp_name].append(comp)

        # Filtra basato su merge_mode
        if merge_mode == "intersection":
            # Solo componenti presenti in tutti i circuiti
            all_components = {
                name: comps for name, comps in all_components.items()
                if len(comps) == len(circuits)
            }

        # Aggiungi nodi
        for comp_name, comps in all_components.items():
            # Media delle metriche
            vlo_mean = sum(c.vlo for c in comps) / len(comps)
            faithfulness_mean = sum(c.faithfulness for c in comps) / len(comps)

            comp = comps[0]  # Usa primo per metadata

            # Colore basato su numero di circuiti che contengono questo nodo
            num_circuits = len(comps)
            color = self._frequency_to_color(num_circuits, len(circuits))

            size = 20 + faithfulness_mean * 30

            title = (
                f"{comp_name}\n"
                f"VLO (avg): {vlo_mean:.3f}\n"
                f"Faithfulness (avg): {faithfulness_mean:.3f}\n"
                f"In {num_circuits}/{len(circuits)} circuits"
            )

            net.add_node(
                comp_name,
                label=f"L{comp.layer_idx}.{comp.component_type[:4]}",
                title=title,
                color=color,
                size=size,
            )

        # Aggiungi archi (considera tutti i circuiti)
        edges_added = set()
        for circuit in circuits:
            comps_sorted = sorted(circuit.components, key=lambda c: c.layer_idx)
            for i in range(len(comps_sorted) - 1):
                src = comps_sorted[i].component_name
                dst = comps_sorted[i + 1].component_name

                # Skip se nodi non nel grafo
                if src not in all_components or dst not in all_components:
                    continue

                edge_key = (src, dst)
                if edge_key not in edges_added:
                    net.add_edge(src, dst)
                    edges_added.add(edge_key)

        net.heading = f"Multi-Circuit View ({len(circuits)} circuits, {merge_mode} mode)"
        net.save_graph(str(output_path))
        return output_path

    def visualize_from_networkx(
        self,
        graph: nx.DiGraph,
        output_path: str | Path,
        node_attrs: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Visualizza grafo NetworkX custom.

        Args:
            graph: NetworkX DiGraph
            output_path: Path HTML
            node_attrs: Attributi personalizzati per nodi

        Returns:
            Path HTML
        """
        output_path = Path(output_path)

        net = Network(
            height=self.height,
            width=self.width,
            bgcolor=self.bgcolor,
            font_color=self.font_color,
            directed=self.directed,
            notebook=False,
        )

        # Converti NetworkX → Pyvis
        net.from_nx(graph)

        # Applica attributi personalizzati
        if node_attrs:
            for node_id, attrs in node_attrs.items():
                if net.get_node(node_id):
                    net.get_node(node_id).update(attrs)

        net.save_graph(str(output_path))
        return output_path

    # ========================================================================
    # Utility: Color mapping
    # ========================================================================

    @staticmethod
    def _vlo_to_color(vlo: float) -> str:
        """VLO → colore (rosso = basso, verde = alto)."""
        # Normalizza VLO [0, 3] → [0, 1]
        normalized = max(0.0, min(1.0, vlo / 3.0))

        # Interpolazione rosso → giallo → verde
        if normalized < 0.5:
            # Rosso → Giallo
            r = 255
            g = int(255 * (normalized * 2))
            b = 0
        else:
            # Giallo → Verde
            r = int(255 * (1 - (normalized - 0.5) * 2))
            g = 255
            b = 0

        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def _faithfulness_to_color(faithfulness: float) -> str:
        """Faithfulness → colore (blu = basso, rosso = alto)."""
        # Normalizza [0, 1]
        normalized = max(0.0, min(1.0, faithfulness))

        # Blu → Rosso
        r = int(255 * normalized)
        g = 0
        b = int(255 * (1 - normalized))

        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def _layer_to_color(layer_idx: int) -> str:
        """Layer index → colore (spettro arcobaleno)."""
        # Assumiamo layer 0-11 (GPT-2)
        hue = (layer_idx / 12) * 360
        # HSV → RGB (semplificato)
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue / 360, 0.8, 0.9)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    @staticmethod
    def _frequency_to_color(count: int, total: int) -> str:
        """Frequenza nodo → colore (grigio = raro, oro = frequente)."""
        normalized = count / total

        # Grigio → Oro
        r = int(128 + 127 * normalized)
        g = int(128 + 87 * normalized)
        b = int(128 - 128 * normalized)

        return f"#{r:02x}{g:02x}{b:02x}"
