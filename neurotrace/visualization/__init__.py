# neurotrace/visualization/__init__.py

"""
NeuroTrace Visualization - Interactive visualization tools.

Questo modulo fornisce:
- CircuitGraphVisualizer: visualizza circuiti come grafi interattivi (Pyvis)
- MetricsPlotter: plot metriche di training e causal discovery (Plotly)
- ActivationExplorer: esplora attivazioni con PCA/t-SNE (Plotly 3D)
- SAEFeatureAnalyzer: analizza feature SAE (top features, reconstruction)
"""

from .circuit_graph import CircuitGraphVisualizer
from .metrics_plotter import MetricsPlotter, TrainingMetricsPlot, VLOMetricsPlot
from .activation_explorer import ActivationExplorer, DimReductionMethod
from .sae_feature_viz import SAEFeatureAnalyzer

__all__ = [
    # Circuit graphs
    "CircuitGraphVisualizer",
    # Metrics
    "MetricsPlotter",
    "TrainingMetricsPlot",
    "VLOMetricsPlot",
    # Activations
    "ActivationExplorer",
    "DimReductionMethod",
    # SAE features
    "SAEFeatureAnalyzer",
]
