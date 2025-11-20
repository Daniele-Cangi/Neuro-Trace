# neurotrace/discovery/__init__.py

"""
NeuroTrace Discovery - Automated circuit discovery at scale.

Questo modulo fornisce:
- ExhaustiveCircuitScanner: test sistematico di TUTTI i componenti
- LatentCircuitDetector: scoperta unsupervised di circuiti nascosti
- ComponentInteractionMatrix: mappa completa delle interazioni
"""

from .exhaustive_scanner import ExhaustiveCircuitScanner, ScanConfig, ScanResult
from .component_interaction_matrix import ComponentInteractionMatrix
from .feature_circuit_discoverer import FeatureCircuitDiscoverer, FeatureImportance

__all__ = [
    "ExhaustiveCircuitScanner",
    "ScanConfig",
    "ScanResult",
    "ComponentInteractionMatrix",
    "FeatureCircuitDiscoverer",
    "FeatureImportance",
]
