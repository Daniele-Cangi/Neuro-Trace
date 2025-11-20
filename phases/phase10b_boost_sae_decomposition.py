"""
PHASE 10B – BOOST SAE DECOMPOSITION (Layer 10)

Goal:
Decompose the Constrained Task Boost vector (Phase 7D) into the SAE feature space
to understand its composition and test if sparse subsets can reconstruct the effect.
This "closes the circle" between the Atlas (Phase 2/3) and the Boost (Phase 7).

This script:
1. Loads the EnhancedSAE for layer 10.
2. Loads the Constrained Task Boost vector (R=25).
3. Projects the boost into SAE space (alpha coefficients).
4. Reconstructs the boost using Top-K features.
5. Measures reconstruction error and energy preservation.

Usage:
    python phase10b_boost_sae_decomposition.py
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from neurotrace.control import EnhancedSAEFeatureStore

# ============================================================================
# 1. SETUP
# ============================================================================

def load_sae_and_boost(layer_idx: int, device: str) -> Tuple[object, torch.Tensor]:
    """Load SAE and Boost Vector."""
    
    # 1. Load SAE
    sae_path = Path(f"checkpoints/all_layers_sae/layer_{layer_idx}/final.pt")
    if not sae_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found at {sae_path}")
        
    print(f"Loading SAE for layer {layer_idx} from {sae_path}...")
    feature_store = EnhancedSAEFeatureStore()
    feature_store.load_sae(str(sae_path), layer=layer_idx, device=device)
    sae = feature_store.saes[layer_idx]
    
    # 2. Load Boost Vector
    # From Phase 7D: checkpoints/learned_task_boost_layer10_R25.pt
    # It was saved as a direct tensor: torch.save(v_boost, checkpoint_path)
    boost_path = Path(f"checkpoints/learned_task_boost_layer{layer_idx}_R25.pt")
    if not boost_path.exists():
        raise FileNotFoundError(f"Boost vector not found at {boost_path}")
        
    print(f"Loading Boost Vector from {boost_path}...")
    v_boost = torch.load(boost_path, map_location=device)
    
    # Ensure shape [768]
    if v_boost.dim() > 1:
        v_boost = v_boost.squeeze()
        
    return sae, v_boost

# ============================================================================
# 2. DECOMPOSITION LOGIC
# ============================================================================

def decompose_vector(sae, vector: torch.Tensor) -> Dict:
    """
    Project vector into SAE space and analyze reconstruction.
    
    Args:
        sae: The loaded EnhancedSAE model.
        vector: The vector to decompose [hidden_dim].
        
    Returns:
        Dictionary with metrics.
    """
    # 1. Linear Projection (Correct for Direction Vectors)
    # We project the vector onto the encoder weights to measure alignment.
    # We do NOT use sae.encode() because that subtracts pre_bias (mean activation),
    # which is incorrect for a direction/delta vector.
    
    # Encoder weight: [dict_size, hidden_dim]
    W_enc = sae.encoder.weight.detach()
    
    # Alpha coefficients: [dict_size]
    # alpha = W_enc @ vector
    alpha = torch.matmul(W_enc, vector)
        
    # 2. Metrics
    boost_norm = torch.norm(vector).item()
    alpha_norm = torch.norm(alpha).item()
    
    print(f"\nBoost Norm: {boost_norm:.4f}")
    print(f"Alpha Norm (Linear Projection): {alpha_norm:.4f}")
    
    # 3. Top-K Reconstruction
    k_values = [10, 50, 100, 200, 500, 1000, 6144]
    results = []
    
    # Sort features by magnitude
    alpha_abs = torch.abs(alpha)
    sorted_indices = torch.argsort(alpha_abs, descending=True)
    
    # Decoder weight: [hidden_dim, dict_size]
    W_dec = sae.decoder.weight.detach()
    
    print(f"\n{'K':<6} | {'Energy Ratio':<12} | {'Norm Ratio':<10} | {'Recon Error':<12}")
    print("-" * 50)
    
    for k in k_values:
        # Select Top-K indices
        top_k_indices = sorted_indices[:k]
        
        # Create sparse alpha
        alpha_k = torch.zeros_like(alpha)
        alpha_k[top_k_indices] = alpha[top_k_indices]
        
        # Decode back to residual space
        # v_k = W_dec @ alpha_k
        # We do NOT add pre_bias because we are reconstructing a direction.
        v_k = torch.matmul(W_dec, alpha_k)
            
        # Metrics
        v_k_norm = torch.norm(v_k).item()
        recon_error = torch.norm(vector - v_k).item()
        norm_ratio = v_k_norm / boost_norm if boost_norm > 0 else 0.0
        
        # Energy ratio in feature space
        energy_k = torch.norm(alpha_k).item() ** 2
        energy_total = alpha_norm ** 2
        energy_ratio = energy_k / energy_total if energy_total > 0 else 0.0
        
        print(f"{k:<6} | {energy_ratio:<12.2%} | {norm_ratio:<10.2%} | {recon_error:<12.4f}")
        
        results.append({
            "K": k,
            "energy_ratio": energy_ratio,
            "norm_ratio": norm_ratio,
            "recon_error": recon_error,
            "v_k_norm": v_k_norm
        })
        
    return {
        "layer": 10,
        "boost_norm": boost_norm,
        "alpha_norm": alpha_norm,
        "metrics": results,
        "top_10_features": sorted_indices[:10].tolist(),
        "top_10_values": alpha[sorted_indices[:10]].tolist()
    }

# ============================================================================
# 3. MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("PHASE 10B: BOOST SAE DECOMPOSITION")
    print("="*60 + "\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer_idx = 10
    
    try:
        # Load
        sae, v_boost = load_sae_and_boost(layer_idx, device)
        
        # Decompose
        results = decompose_vector(sae, v_boost)
        
        # Save
        json_file = "boost_sae_decomposition_results.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {json_file}")
        
        # Print Top 10 Features
        print("\nTop 10 Features in Boost Vector:")
        for idx, val in zip(results["top_10_features"], results["top_10_values"]):
            print(f"Feature {idx:<5}: {val:+.4f}")
            
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
