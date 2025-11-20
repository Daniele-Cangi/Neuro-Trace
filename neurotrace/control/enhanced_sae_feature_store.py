# neurotrace/control/enhanced_sae_feature_store.py

"""
Enhanced SAE Feature Store for Control Plane integration.

Connects trained EnhancedSAE models to the steering/control system.
"""

from __future__ import annotations

from typing import List, Dict
from pathlib import Path

import torch

from neurotrace.training.enhanced_sae import EnhancedSAE


class EnhancedSAEFeatureStore:
    """
    Feature store that loads and provides directions from trained EnhancedSAE models.

    This implements the FeatureStore protocol required by steering_builder.py
    but works with our SOTA EnhancedSAE architecture.

    Usage:
        # Load trained SAE
        store = EnhancedSAEFeatureStore()
        store.load_sae("checkpoints/layer0_sae/final.pt", layer=0)

        # Get feature directions
        directions = store.get_sae_directions(
            model_name="gpt2",
            layer=0,
            feature_indices=[2586, 2081, 1123],  # Top IOI features
            device=torch.device("cuda")
        )
    """

    def __init__(self):
        """Initialize empty feature store."""
        self.saes: Dict[int, EnhancedSAE] = {}  # layer -> SAE
        self.sae_paths: Dict[int, str] = {}

    def load_sae(
        self,
        checkpoint_path: str | Path,
        layer: int,
        device: torch.device | None = None
    ) -> None:
        """
        Load a trained EnhancedSAE checkpoint for a specific layer.

        Args:
            checkpoint_path: Path to the .pt checkpoint file
            layer: Which layer this SAE was trained on
            device: Device to load to (default: CPU)
        """
        device = device or torch.device("cpu")
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SAE checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Extract config and create SAE
        if 'config' not in checkpoint:
            raise ValueError(f"Checkpoint missing 'config' key: {checkpoint_path}")

        config = checkpoint['config']

        # Handle both naming conventions (d_in/d_sae vs input_dim/dict_size)
        input_dim = config.get('input_dim') or config.get('d_in')
        dict_size = config.get('dict_size') or config.get('d_sae')

        if not input_dim or not dict_size:
            raise ValueError(f"Config missing dimensions. Keys: {list(config.keys())}")

        # Create EnhancedSAE with saved config
        # Note: EnhancedSAE uses input_dim, dict_size, k_sparse, normalize_decoder
        sae = EnhancedSAE(
            input_dim=input_dim,
            dict_size=dict_size,
            k_sparse=config.get('k_sparse', 64),
            normalize_decoder=config.get('normalize_decoder', True),
            use_jumprelu=config.get('use_jumprelu', False),
            sparsity_lambda=config.get('sparsity_lambda', 1e-3),
            ghost_threshold=config.get('ghost_threshold', 1e-5),
        ).to(device)

        # Load trained weights
        if 'model_state_dict' in checkpoint:
            sae.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            sae.load_state_dict(checkpoint['state_dict'])
        else:
            raise ValueError(f"Checkpoint missing model weights: {checkpoint_path}")

        sae.eval()

        # Store
        self.saes[layer] = sae
        self.sae_paths[layer] = str(checkpoint_path)

        print(f"[OK] Loaded EnhancedSAE for layer {layer}")
        print(f"  Path: {checkpoint_path}")
        print(f"  Config: {input_dim} -> {dict_size} features")
        print(f"  Device: {device}")

    def get_sae_directions(
        self,
        model_name: str,
        layer: int,
        feature_indices: List[int],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Get SAE decoder directions for specific features.

        Args:
            model_name: Model identifier (for validation, optional)
            layer: Layer index
            feature_indices: List of SAE feature indices
            device: Target device (default: SAE's current device)

        Returns:
            Tensor [len(feature_indices), hidden_dim] with normalized directions.
            Each row is the decoder direction for that feature.
        """
        if layer not in self.saes:
            raise ValueError(
                f"No SAE loaded for layer {layer}. "
                f"Available layers: {list(self.saes.keys())}"
            )

        sae = self.saes[layer]
        device = device or next(sae.parameters()).device

        # Validate feature indices
        dict_size = sae.dict_size
        for idx in feature_indices:
            if idx < 0 or idx >= dict_size:
                raise ValueError(
                    f"Feature index {idx} out of bounds for layer {layer} "
                    f"(dict_size={dict_size})"
                )

        # Extract decoder directions
        # EnhancedSAE decoder: Linear(dict_size, input_dim)
        # Weight shape: [input_dim, dict_size] (transposed in Linear)
        # We want direction for feature i: decoder.weight[:, i]

        with torch.no_grad():
            # Get decoder weight matrix [input_dim, dict_size]
            W_dec = sae.decoder.weight.data  # [input_dim, dict_size]

            # Select features [input_dim, len(feature_indices)]
            feature_indices_tensor = torch.tensor(
                feature_indices,
                dtype=torch.long,
                device=W_dec.device
            )
            directions = W_dec[:, feature_indices_tensor].T  # [len(features), input_dim]

            # Normalize each direction (recommended for interpretable steering)
            norms = torch.norm(directions, dim=1, keepdim=True)
            directions = directions / (norms + 1e-8)

            return directions.to(device)

    def get_layer_info(self, layer: int) -> Dict:
        """Get metadata about a loaded SAE."""
        if layer not in self.saes:
            raise ValueError(f"No SAE loaded for layer {layer}")

        sae = self.saes[layer]
        return {
            "layer": layer,
            "input_dim": sae.input_dim,
            "dict_size": sae.dict_size,
            "k_sparse": sae.k_sparse,
            "normalize_decoder": sae.normalize_decoder_weights,
            "use_jumprelu": sae.use_jumprelu,
            "checkpoint_path": self.sae_paths.get(layer, "unknown"),
            "device": str(next(sae.parameters()).device),
        }

    def list_loaded_layers(self) -> List[int]:
        """Get list of layers with loaded SAEs."""
        return sorted(self.saes.keys())
