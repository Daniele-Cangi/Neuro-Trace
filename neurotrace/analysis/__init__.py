# neurotrace/analysis/__init__.py

"""
NeuroTrace Analysis - Geometric and statistical analysis of neural activations.

Questo modulo fornisce:
- Geometric features: LID, spectral analysis, manifold metrics
- Statistical features: variance, entropy, correlation
"""

from .geometric import (
    compute_lid,
    compute_spectral_features,
    GeometricFeatures,
    ActivationGeometry,
)

__all__ = [
    # LID
    "compute_lid",
    # Spectral
    "compute_spectral_features",
    # Aggregated
    "GeometricFeatures",
    "ActivationGeometry",
]
