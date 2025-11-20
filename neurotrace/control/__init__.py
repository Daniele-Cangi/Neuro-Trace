# neurotrace/control/__init__.py

"""
NeuroTrace Control Plane - Layer di controllo attivo sui circuiti neurali.

Questo modulo trasforma circuiti scoperti in "oggetti controllabili" tramite:
- CircuitRegistry: persistenza e query dei circuiti
- SteeringBuilder: conversione circuiti → steering vectors
- CircuitController: API orchestrazione steering runtime
"""

from .circuit_registry import (
    CircuitComponent,
    CircuitCausalMetrics,
    CircuitSemantics,
    CircuitFeatures,
    CircuitRecord,
    CircuitRegistry,
)
from .steering_builder import (
    FeatureStore,
    LayerSteeringVector,
    SteeringSpec,
    SteeringBuilder,
)
from .controller import (
    ResidualHookHandle,
    ModelWrapper,
    ActiveCircuit,
    ControlTrace,
    CircuitController,
)
from .sae_feature_store import SAEFeatureStore
from .enhanced_sae_feature_store import EnhancedSAEFeatureStore
from .hierarchical_steering import HierarchicalSteering, SteeringConfig

__all__ = [
    # Registry
    "CircuitComponent",
    "CircuitCausalMetrics",
    "CircuitSemantics",
    "CircuitFeatures",
    "CircuitRecord",
    "CircuitRegistry",
    # Steering
    "FeatureStore",
    "LayerSteeringVector",
    "SteeringSpec",
    "SteeringBuilder",
    # Controller
    "ResidualHookHandle",
    "ModelWrapper",
    "ActiveCircuit",
    "ControlTrace",
    "CircuitController",
    # Feature stores
    "SAEFeatureStore",
    "EnhancedSAEFeatureStore",
    # Hierarchical steering
    "HierarchicalSteering",
    "SteeringConfig",
]
