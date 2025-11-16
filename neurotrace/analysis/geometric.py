# neurotrace/analysis/geometric.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class GeometricFeatures:
    """
    Geometric features di un set di attivazioni.
    """
    # Local Intrinsic Dimension
    lid: float
    lid_std: float

    # Spectral features
    spectral_entropy: float
    participation_ratio: float
    effective_rank: float

    # Variance-based
    explained_variance_ratio: float  # top-k principal components

    # Extras
    num_samples: int
    ambient_dim: int


def compute_lid(
    activations: torch.Tensor,
    k: int = 20,
    method: str = "mle",
) -> tuple[float, float]:
    """
    Compute Local Intrinsic Dimension (LID) using Maximum Likelihood Estimation.

    LID misura la dimensionalità effettiva locale del manifold su cui giacciono
    le attivazioni. Valori bassi indicano struttura compressa/geometrica.

    Args:
        activations: Tensor [N, D] di attivazioni
        k: Numero nearest neighbors per stima locale
        method: "mle" (default) o "twonn"

    Returns:
        (lid_mean, lid_std): Media e std della LID su tutti i punti

    Reference:
        Levina & Bickel (2004) - Maximum Likelihood Estimation of Intrinsic Dimension
    """
    if activations.dim() != 2:
        raise ValueError(f"Expected [N, D] activations, got {tuple(activations.shape)}")

    N, D = activations.shape

    if N < k + 1:
        logger.warning(f"Too few samples ({N}) for k={k}, using k={max(2, N//2)}")
        k = max(2, N // 2)

    # Move to CPU for numpy operations (more stable)
    X = activations.detach().cpu().float().numpy()

    # Compute pairwise distances
    from scipy.spatial.distance import cdist
    dists = cdist(X, X, metric="euclidean")

    # For each point, find k nearest neighbors (excluding self)
    lids = []
    for i in range(N):
        # Sort distances, skip first (self, distance=0)
        sorted_dists = np.sort(dists[i])[1 : k + 1]

        if method == "mle":
            # MLE estimator: LID = -k / sum(log(r_k / r_i))
            # r_k is k-th nearest neighbor distance
            r_k = sorted_dists[-1]
            if r_k < 1e-10:
                continue  # Skip degenerate cases

            log_ratios = np.log(r_k / (sorted_dists + 1e-10))
            lid_estimate = k / np.sum(log_ratios)

        elif method == "twonn":
            # Two-NN estimator (more robust)
            r1, r2 = sorted_dists[0], sorted_dists[1]
            if r2 < 1e-10:
                continue
            lid_estimate = np.log(2) / np.log(r2 / (r1 + 1e-10))

        else:
            raise ValueError(f"Unknown method: {method}")

        # Clamp to reasonable range
        if 0 < lid_estimate < D * 2:
            lids.append(lid_estimate)

    if not lids:
        logger.warning("LID computation failed for all points, returning ambient dim")
        return float(D), 0.0

    lid_mean = float(np.mean(lids))
    lid_std = float(np.std(lids))

    return lid_mean, lid_std


def compute_spectral_features(
    activations: torch.Tensor,
    top_k: int = 50,
) -> dict[str, float]:
    """
    Compute spectral features via SVD della matrice di attivazioni.

    Features:
    - spectral_entropy: entropia degli autovalori (normalizzata)
    - participation_ratio: (sum σ)² / sum(σ²)
    - effective_rank: exp(entropy) approssimato
    - explained_variance_ratio: varianza spiegata da top-k componenti

    Args:
        activations: Tensor [N, D]
        top_k: Numero componenti principali per explained variance

    Returns:
        Dict con feature spettrali
    """
    if activations.dim() != 2:
        raise ValueError(f"Expected [N, D], got {tuple(activations.shape)}")

    N, D = activations.shape

    # Center data
    X = activations - activations.mean(dim=0, keepdim=True)

    # SVD (top-k for efficiency if D is large)
    if D > 500:
        # Use randomized SVD for large matrices
        try:
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=min(top_k, min(N, D) - 1), random_state=42)
            svd.fit(X.detach().cpu().numpy())
            singular_values = svd.singular_values_
        except ImportError:
            # Fallback to torch SVD
            _, s, _ = torch.svd(X.float())
            singular_values = s.detach().cpu().numpy()[:top_k]
    else:
        # Full SVD
        _, s, _ = torch.svd(X.float())
        singular_values = s.detach().cpu().numpy()

    # Eigenvalues (squared singular values)
    eigenvalues = singular_values ** 2
    total_var = eigenvalues.sum()

    if total_var < 1e-10:
        logger.warning("Near-zero variance in activations, returning default features")
        return {
            "spectral_entropy": 0.0,
            "participation_ratio": 1.0,
            "effective_rank": 1.0,
            "explained_variance_ratio": 1.0,
        }

    # Normalize to probabilities
    probs = eigenvalues / total_var

    # Spectral entropy: -sum(p * log(p))
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(len(probs))
    spectral_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Participation ratio: (sum σ)² / sum(σ²)
    participation_ratio = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()

    # Effective rank: exp(entropy)
    effective_rank = np.exp(entropy)

    # Explained variance ratio (top-k)
    top_k_actual = min(top_k, len(eigenvalues))
    explained_var = eigenvalues[:top_k_actual].sum() / total_var

    return {
        "spectral_entropy": float(spectral_entropy),
        "participation_ratio": float(participation_ratio),
        "effective_rank": float(effective_rank),
        "explained_variance_ratio": float(explained_var),
    }


class ActivationGeometry:
    """
    Analyzer per geometric features di attivazioni neurali.

    Usage:
        analyzer = ActivationGeometry()
        features = analyzer.analyze(activations)
    """

    def __init__(
        self,
        lid_k: int = 20,
        lid_method: str = "mle",
        spectral_top_k: int = 50,
    ) -> None:
        self.lid_k = lid_k
        self.lid_method = lid_method
        self.spectral_top_k = spectral_top_k

    def analyze(
        self,
        activations: torch.Tensor,
        compute_lid_flag: bool = True,
    ) -> GeometricFeatures:
        """
        Compute tutte le geometric features per un set di attivazioni.

        Args:
            activations: Tensor [N, D]
            compute_lid_flag: Se True, compute LID (può essere lento per N grande)

        Returns:
            GeometricFeatures
        """
        N, D = activations.shape

        # LID (opzionale, può essere lento)
        if compute_lid_flag and N >= self.lid_k + 1:
            try:
                lid_mean, lid_std = compute_lid(
                    activations, k=self.lid_k, method=self.lid_method
                )
            except Exception as e:
                logger.warning(f"LID computation failed: {e}, using ambient dim")
                lid_mean, lid_std = float(D), 0.0
        else:
            lid_mean, lid_std = float(D), 0.0

        # Spectral features
        try:
            spectral = compute_spectral_features(activations, top_k=self.spectral_top_k)
        except Exception as e:
            logger.warning(f"Spectral computation failed: {e}, using defaults")
            spectral = {
                "spectral_entropy": 0.0,
                "participation_ratio": 1.0,
                "effective_rank": 1.0,
                "explained_variance_ratio": 0.0,
            }

        return GeometricFeatures(
            lid=lid_mean,
            lid_std=lid_std,
            spectral_entropy=spectral["spectral_entropy"],
            participation_ratio=spectral["participation_ratio"],
            effective_rank=spectral["effective_rank"],
            explained_variance_ratio=spectral["explained_variance_ratio"],
            num_samples=N,
            ambient_dim=D,
        )

    def analyze_per_layer(
        self,
        activations_dict: dict[str, torch.Tensor],
    ) -> dict[str, GeometricFeatures]:
        """
        Analyze geometric features per ogni layer.

        Args:
            activations_dict: {layer_name: activations[N, D]}

        Returns:
            {layer_name: GeometricFeatures}
        """
        results = {}
        for layer_name, acts in activations_dict.items():
            logger.info(f"Analyzing geometry for {layer_name}...")
            results[layer_name] = self.analyze(acts)

        return results
