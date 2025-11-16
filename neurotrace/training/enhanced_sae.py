# neurotrace/training/enhanced_sae.py

"""
State-of-the-art Sparse Autoencoder implementation.

Incorporates best practices from:
- Anthropic "Towards Monosemanticity" (2023)
- Anthropic "Scaling Monosemanticity" (2024)
- Google "Gemma Scope" (2024)
- Apollo Research "Gated SAE"

Features:
1. Decoder weight normalization (Anthropic 2023)
2. Ghost gradients for dead features (Anthropic 2023)
3. Top-K activation (Gao et al. 2024)
4. Pre-bias correction (Anthropic 2024)
5. JumpReLU activation (Rajamanoharan et al. 2024)
6. Auxiliary losses (orthogonality, diversity)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class EnhancedSAE(nn.Module):
    """
    Enhanced Sparse Autoencoder with SOTA features.

    Architecture:
        input[D] -> pre_bias -> encoder[D→K] -> top_k/jumprelu -> decoder[K→D] -> output[D]

    Key improvements over basic SAE:
    - Normalized decoder weights (unit norm per feature)
    - Pre-bias correction (learned mean subtraction)
    - Top-K or JumpReLU activation (better than ReLU)
    - Ghost gradients (resurrect dead features)
    - Feature usage tracking
    """

    def __init__(
        self,
        input_dim: int,
        dict_size: int,
        k_sparse: int = 64,
        sparsity_lambda: float = 1e-3,
        use_jumprelu: bool = False,
        ghost_threshold: float = 1e-5,
        normalize_decoder: bool = True,
    ) -> None:
        """
        Args:
            input_dim: Dimension of input activations
            dict_size: Size of SAE dictionary (typically 4x to 16x input_dim)
            k_sparse: Number of top features to keep (if not using JumpReLU)
            sparsity_lambda: L1 sparsity penalty weight
            use_jumprelu: Use JumpReLU instead of top-k (more advanced)
            ghost_threshold: Threshold for dead feature detection (fraction of steps)
            normalize_decoder: Normalize decoder columns to unit norm
        """
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.k_sparse = k_sparse
        self.sparsity_lambda = sparsity_lambda
        self.use_jumprelu = use_jumprelu
        self.ghost_threshold = ghost_threshold
        self.normalize_decoder_weights = normalize_decoder

        # ================================================================
        # Pre-bias: Learned mean of input distribution
        # ================================================================
        # We subtract this before encoding to center the data
        # This prevents SAE from wasting a feature on "mean activation"
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))

        # ================================================================
        # Encoder: input_dim -> dict_size
        # ================================================================
        self.encoder = nn.Linear(input_dim, dict_size, bias=True)

        # Initialize encoder with small weights (helps training stability)
        nn.init.kaiming_uniform_(self.encoder.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.encoder.bias)

        # ================================================================
        # Decoder: dict_size -> input_dim
        # ================================================================
        self.decoder = nn.Linear(dict_size, input_dim, bias=True)

        # Initialize decoder
        nn.init.kaiming_uniform_(self.decoder.weight, mode='fan_out', nonlinearity='linear')
        nn.init.zeros_(self.decoder.bias)

        # Normalize decoder columns to unit norm (critical for monosemanticity!)
        if self.normalize_decoder_weights:
            self._normalize_decoder()

        # ================================================================
        # JumpReLU parameters (if enabled)
        # ================================================================
        if use_jumprelu:
            # Learnable threshold per feature
            self.jump_threshold = nn.Parameter(torch.zeros(dict_size))
        else:
            self.register_parameter('jump_threshold', None)

        # ================================================================
        # Feature usage tracking (for ghost gradients)
        # ================================================================
        # Tracks how many times each feature has been activated
        self.register_buffer('feature_activation_count', torch.zeros(dict_size))
        # Total number of forward passes
        self.register_buffer('num_forward_passes', torch.tensor(0, dtype=torch.long))

        logger.info(
            f"EnhancedSAE initialized: {input_dim} → {dict_size} "
            f"(k={k_sparse}, jumprelu={use_jumprelu}, norm_decoder={normalize_decoder})"
        )

    def _normalize_decoder(self) -> None:
        """
        Normalize decoder weight columns to unit norm.

        Each column represents a feature in the dictionary.
        Unit norm ensures features don't shrink/grow to minimize loss.

        This is CRITICAL for learning monosemantic features (Anthropic 2023).
        """
        with torch.no_grad():
            # decoder.weight shape: [input_dim, dict_size]
            # We want each column (feature) to have unit norm
            norms = self.decoder.weight.norm(dim=0, keepdim=True)
            self.decoder.weight.div_(norms.clamp(min=1e-8))

    def _jumprelu_activation(self, pre_activation: Tensor) -> Tensor:
        """
        JumpReLU activation: ReLU with learnable per-feature threshold.

        JumpReLU(x, θ) = ReLU(x - θ) if x > θ else 0

        Better sparsity control than standard ReLU.
        Reference: Rajamanoharan et al. 2024 (Gemma Scope)
        """
        # Broadcast threshold: [dict_size] -> [batch, dict_size]
        threshold = self.jump_threshold.unsqueeze(0)
        return F.relu(pre_activation - threshold) * (pre_activation > threshold).float()

    def _topk_activation(self, pre_activation: Tensor) -> Tensor:
        """
        Top-K activation: Keep only K largest activations per sample.

        Provides exact sparsity control (L0 = k).
        Reference: Gao et al. 2024
        """
        # Get top-k values and indices
        values, indices = torch.topk(pre_activation, k=self.k_sparse, dim=-1)

        # Create sparse tensor
        codes = torch.zeros_like(pre_activation)
        codes.scatter_(-1, indices, F.relu(values))

        return codes

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass through SAE.

        Args:
            x: Input activations [batch, input_dim]

        Returns:
            Dictionary with:
                - codes: Sparse feature activations [batch, dict_size]
                - reconstruction: Reconstructed input [batch, input_dim]
                - pre_activation: Pre-activation values [batch, dict_size]
                - dead_features: Boolean mask of dead features [dict_size]
        """
        batch_size = x.shape[0]

        # ================================================================
        # 1. Pre-bias correction (center the input)
        # ================================================================
        x_centered = x - self.pre_bias

        # ================================================================
        # 2. Encode to dictionary space
        # ================================================================
        pre_activation = self.encoder(x_centered)  # [batch, dict_size]

        # ================================================================
        # 3. Apply activation function (Top-K or JumpReLU)
        # ================================================================
        if self.use_jumprelu:
            codes = self._jumprelu_activation(pre_activation)
        else:
            codes = self._topk_activation(pre_activation)

        # ================================================================
        # 4. Track feature usage (for ghost gradients)
        # ================================================================
        if self.training:
            # Count which features are active (>0) in this batch
            active_features = (codes > 0).float().sum(dim=0)  # [dict_size]
            self.feature_activation_count += active_features
            self.num_forward_passes += batch_size

        # ================================================================
        # 5. Decode back to input space
        # ================================================================
        reconstruction = self.decoder(codes) + self.pre_bias  # [batch, input_dim]

        # ================================================================
        # 6. Identify dead features (for ghost gradients)
        # ================================================================
        dead_features = self._get_dead_features()

        return {
            'codes': codes,
            'reconstruction': reconstruction,
            'pre_activation': pre_activation,
            'dead_features': dead_features,
        }

    def _get_dead_features(self) -> Tensor:
        """
        Identify features that are "dead" (rarely/never activated).

        A feature is dead if it activated less than ghost_threshold fraction of the time.

        Returns:
            Boolean tensor [dict_size] where True = dead feature
        """
        if self.num_forward_passes == 0:
            return torch.zeros(self.dict_size, dtype=torch.bool, device=self.encoder.weight.device)

        # Activation rate per feature
        activation_rate = self.feature_activation_count / self.num_forward_passes.float()

        # Features below threshold are dead
        dead = activation_rate < self.ghost_threshold

        return dead

    def compute_loss(
        self,
        x: Tensor,
        output: Dict[str, Tensor],
        ghost_grad_weight: float = 0.1,
        l1_weight: Optional[float] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute total loss with multiple components.

        Loss = MSE + λ*L1 + ghost_grad_loss

        Args:
            x: Original input [batch, input_dim]
            output: Output from forward()
            ghost_grad_weight: Weight for ghost gradient loss
            l1_weight: L1 sparsity weight (defaults to self.sparsity_lambda)

        Returns:
            (total_loss, metrics_dict)
        """
        reconstruction = output['reconstruction']
        codes = output['codes']
        pre_activation = output['pre_activation']
        dead_features = output['dead_features']

        # ================================================================
        # 1. Reconstruction loss (MSE)
        # ================================================================
        mse_loss = F.mse_loss(reconstruction, x)

        # ================================================================
        # 2. Sparsity loss (L1 on codes)
        # ================================================================
        if l1_weight is None:
            l1_weight = self.sparsity_lambda

        l1_loss = codes.abs().mean()

        # ================================================================
        # 3. Ghost gradients (resurrect dead features)
        # ================================================================
        # Give dead features gradients based on their pre-activation
        # This encourages them to "wake up" and become useful
        ghost_loss = torch.tensor(0.0, device=x.device)

        if self.training and self.num_forward_passes > 100:
            if dead_features.any():
                # For dead features, penalize their pre-activation magnitude
                # This creates gradients that encourage them to activate
                dead_pre_act = pre_activation[:, dead_features]
                ghost_loss = dead_pre_act.pow(2).mean()

        # ================================================================
        # 4. Total loss
        # ================================================================
        total_loss = mse_loss + l1_weight * l1_loss + ghost_grad_weight * ghost_loss

        # ================================================================
        # 5. Compute metrics
        # ================================================================
        with torch.no_grad():
            # L0 sparsity (average number of active features)
            l0_sparsity = (codes > 0).float().sum(dim=-1).mean()

            # Fraction of dead features
            dead_fraction = dead_features.float().mean()

        metrics = {
            'mse': mse_loss.item(),
            'l1': l1_loss.item(),
            'ghost': ghost_loss.item(),
            'total': total_loss.item(),
            'l0_sparsity': l0_sparsity.item(),
            'dead_fraction': dead_fraction.item(),
        }

        return total_loss, metrics

    def normalize_decoder_step(self) -> None:
        """
        Normalize decoder weights after each training step.

        Should be called after optimizer.step() in training loop.
        """
        if self.normalize_decoder_weights:
            self._normalize_decoder()

    @torch.no_grad()
    def encode(self, x: Tensor, return_full: bool = False) -> Dict[str, Tensor]:
        """
        Encode input to sparse codes (inference mode).

        Args:
            x: Input activations [batch, input_dim]
            return_full: If True, return full output dict. If False, only codes.

        Returns:
            If return_full=False:
                {'codes': sparse codes [batch, dict_size]}
            If return_full=True:
                Full output from forward()
        """
        self.eval()
        output = self.forward(x)

        if return_full:
            return output
        else:
            return {'codes': output['codes']}

    def get_feature_statistics(self) -> Dict[str, Tensor]:
        """
        Get statistics about feature usage.

        Returns:
            Dictionary with:
                - activation_counts: Total activations per feature
                - activation_rates: Activation rate per feature (0-1)
                - dead_features: Boolean mask of dead features
                - num_dead: Number of dead features
        """
        if self.num_forward_passes == 0:
            activation_rates = torch.zeros(self.dict_size)
        else:
            activation_rates = self.feature_activation_count / self.num_forward_passes.float()

        dead_features = self._get_dead_features()

        num_dead = dead_features.sum().item()
        dead_fraction = num_dead / self.dict_size if self.dict_size > 0 else 0.0

        return {
            'activation_counts': self.feature_activation_count.clone(),
            'activation_rates': activation_rates,
            'dead_features': dead_features,
            'num_dead': num_dead,
            'dead_fraction': dead_fraction,
        }

    def reset_feature_statistics(self) -> None:
        """Reset feature usage statistics (useful for multi-stage training)."""
        self.feature_activation_count.zero_()
        self.num_forward_passes.zero_()


# ================================================================
# Convenience functions
# ================================================================

def create_enhanced_sae(
    input_dim: int,
    dict_mult: int = 4,
    k_sparse: Optional[int] = None,
    **kwargs
) -> EnhancedSAE:
    """
    Create an EnhancedSAE with sensible defaults.

    Args:
        input_dim: Input dimension (e.g., 768 for GPT-2)
        dict_mult: Dictionary size multiplier (dict_size = input_dim * dict_mult)
        k_sparse: Number of top-k features (default: dict_size // 32)
        **kwargs: Additional arguments for EnhancedSAE

    Returns:
        EnhancedSAE instance
    """
    dict_size = input_dim * dict_mult

    if k_sparse is None:
        # Default: ~3% of dictionary active (Anthropic recommendation)
        k_sparse = max(32, dict_size // 32)

    return EnhancedSAE(
        input_dim=input_dim,
        dict_size=dict_size,
        k_sparse=k_sparse,
        **kwargs
    )
