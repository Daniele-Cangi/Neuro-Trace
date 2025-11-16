# neurotrace/causal/__init__.py

"""
NeuroTrace Causal - Causal analysis and circuit discovery.

Questo modulo fornisce:
- VLO (Value of Learned Organization): logit difference under intervention
- Circuit extraction: da components → CircuitRecord
- Intervention utilities: ablation, patching, counterfactuals
"""

from .vlo_tester import (
    VLOTester,
    VLOResult,
    InterventionType,
    compute_vlo,
)
from .circuit_extractor import (
    CircuitExtractor,
    ComponentSpec,
    extract_circuit_from_components,
)

__all__ = [
    # VLO
    "VLOTester",
    "VLOResult",
    "InterventionType",
    "compute_vlo",
    # Circuit extraction
    "CircuitExtractor",
    "ComponentSpec",
    "extract_circuit_from_components",
]
