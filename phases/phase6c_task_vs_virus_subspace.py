"""
PHASE 6C – TASK VS VIRUS SUBSPACE COMPARISON (Layer 10)

Goal:
Quantitatively compare the "task subspace" (IOI task direction) with the 
"virus subspace" (adversarial direction) at layer 10.

This script:
1. Captures clean residual stream activations at layer 10.
2. Computes the "task direction" via Ridge Regression on logit diffs.
3. Reconstructs the "virus subspace" from Phase 5B sparse viruses via PCA.
4. Computes alignment metrics (cosine similarity, energy, principal angles).

Usage:
    python phase6c_task_vs_virus_subspace.py
"""

import sys
import json
import torch
import math
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from transformers import AutoModelForCausalLM, AutoTokenizer
from neurotrace.datasets import IOIDatasetGenerator
from neurotrace.control import EnhancedSAEFeatureStore

# ============================================================================
# 1. SETUP & UTILS
# ============================================================================

def load_model_and_data(device: str, num_train: int = 2000, num_test: int = 500):
    """Load GPT-2 model and IOI dataset."""
    print(f"Loading GPT-2 model on {device}...")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("Generating IOI dataset...")
    generator = IOIDatasetGenerator()
    total_examples = num_train + num_test
    examples = generator.generate(num_examples=total_examples, ensure_diversity=True)
    
    train_examples = examples[:num_train]
    test_examples = examples[num_train:]
    
    return model, tokenizer, train_examples, test_examples

def compute_metrics(model, tokenizer, examples, batch_size=16):
    """Compute accuracy and logit diff."""
    correct_counts = 0
    total_counts = 0
    logit_diffs = []
    
    prompts = [ex.text for ex in examples]
    correct_answers = [ex.correct_answer for ex in examples]
    incorrect_answers = [ex.incorrect_answer for ex in examples]
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_correct = correct_answers[i:i+batch_size]
        batch_incorrect = incorrect_answers[i:i+batch_size]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        last_token_indices = inputs.attention_mask.sum(dim=1) - 1
        final_logits = logits[torch.arange(logits.shape[0]), last_token_indices, :]
        
        for j, (corr, incorr) in enumerate(zip(batch_correct, batch_incorrect)):
            corr_id = tokenizer.encode(" " + corr)[0]
            incorr_id = tokenizer.encode(" " + incorr)[0]
            
            corr_logit = final_logits[j, corr_id].item()
            incorr_logit = final_logits[j, incorr_id].item()
            
            logit_diffs.append(corr_logit - incorr_logit)
            
            if corr_logit > incorr_logit:
                correct_counts += 1
            total_counts += 1
            
    accuracy = correct_counts / total_counts if total_counts > 0 else 0.0
    mean_logit_diff = np.mean(logit_diffs) if logit_diffs else 0.0
    
    return {
        "accuracy": accuracy,
        "logit_diff": mean_logit_diff
    }

# ============================================================================
# 2. CAPTURE ACTIVATIONS
# ============================================================================

def capture_activations(model, tokenizer, examples, layer_idx, batch_size=16):
    """
    Capture residual stream activations at the specified layer.
    Returns:
        H: [N, d_model] tensor of activations (at last token position)
        y: [N] tensor of logit diffs
    """
    activations = []
    logit_diffs = []
    
    prompts = [ex.text for ex in examples]
    correct_answers = [ex.correct_answer for ex in examples]
    incorrect_answers = [ex.incorrect_answer for ex in examples]
    
    # Hook to capture activations
    captured_batch = {}
    def hook_fn(module, inputs, outputs):
        # outputs[0] is hidden states [B, T, D]
        captured_batch['hidden'] = outputs[0].detach()
        
    handle = model.transformer.h[layer_idx].register_forward_hook(hook_fn)
    
    print(f"Capturing activations from layer {layer_idx}...")
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_correct = correct_answers[i:i+batch_size]
        batch_incorrect = incorrect_answers[i:i+batch_size]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        # Get activations at last token
        last_token_indices = inputs.attention_mask.sum(dim=1) - 1
        batch_hidden = captured_batch['hidden'] # [B, T, D]
        final_hidden = batch_hidden[torch.arange(batch_hidden.shape[0]), last_token_indices, :] # [B, D]
        activations.append(final_hidden.cpu())
        
        # Calculate logit diffs for regression target
        final_logits = logits[torch.arange(logits.shape[0]), last_token_indices, :]
        
        for j, (corr, incorr) in enumerate(zip(batch_correct, batch_incorrect)):
            corr_id = tokenizer.encode(" " + corr)[0]
            incorr_id = tokenizer.encode(" " + incorr)[0]
            diff = final_logits[j, corr_id].item() - final_logits[j, incorr_id].item()
            logit_diffs.append(diff)
            
    handle.remove()
    
    H = torch.cat(activations, dim=0).to(model.device) # [N, D]
    y = torch.tensor(logit_diffs, device=model.device) # [N]
    
    return H, y

# ============================================================================
# 3. TASK SUBSPACE
# ============================================================================

def compute_task_direction(H: torch.Tensor, y: torch.Tensor, lambda_ridge: float = 1e-3):
    """
    Compute task direction w using Ridge Regression: H @ w ~ y
    """
    N, D = H.shape
    I = torch.eye(D, device=H.device)
    
    # w = (H.T @ H + lambda * I)^-1 @ H.T @ y
    # Using torch.linalg.solve for stability
    lhs = H.T @ H + lambda_ridge * I
    rhs = H.T @ y
    
    w = torch.linalg.solve(lhs, rhs)
    w_unit = w / w.norm()
    
    return w_unit

def compute_task_subspace_pca(H: torch.Tensor, y: torch.Tensor, k: int = 4, top_percent: float = 0.3):
    """
    Compute task subspace using PCA on high-logit-diff examples.
    """
    # Select top examples
    threshold = torch.quantile(y, 1 - top_percent)
    mask = y >= threshold
    H_high = H[mask]
    
    # Center
    H_centered = H_high - H_high.mean(dim=0, keepdim=True)
    
    # PCA
    U, S, Vh = torch.linalg.svd(H_centered, full_matrices=False)
    
    # Top k components
    R_task = Vh[:k] # [k, D]
    
    # Orthogonalize (SVD Vh rows are already orthogonal, but let's be safe/consistent with QR)
    # QR expects [D, k]
    Q, _ = torch.linalg.qr(R_task.T)
    B_task = Q[:, :k] # [D, k]
    
    return B_task

# ============================================================================
# 4. VIRUS SUBSPACE RECONSTRUCTION
# ============================================================================

def load_sae_decoder(layer_idx: int, device: str) -> torch.Tensor:
    """Load SAE decoder matrix."""
    checkpoint_path = Path(f"checkpoints/all_layers_sae/layer_{layer_idx}/final.pt")
    feature_store = EnhancedSAEFeatureStore()
    feature_store.load_sae(str(checkpoint_path), layer=layer_idx, device=device)
    sae = feature_store.saes[layer_idx]
    W_dec = sae.decoder.weight.detach().T  # [features, hidden]
    return W_dec

def load_sparse_viruses(layer_idx: int, device: str) -> torch.Tensor:
    """Load alpha vectors."""
    lambdas = [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    alphas = []
    
    print("Loading sparse virus checkpoints...")
    for l1 in lambdas:
        # Try different filename formats
        path_sci_no_underscore = Path(f"checkpoints/sparse_sae_virus_layer{layer_idx}_l1{l1:.0e}.pt")
        path_sci = Path(f"checkpoints/sparse_sae_virus_layer{layer_idx}_l1_{l1:.0e}.pt")
        path_float = Path(f"checkpoints/sparse_sae_virus_layer{layer_idx}_l1_{l1}.pt")
        
        checkpoint_path = None
        if path_sci_no_underscore.exists():
            checkpoint_path = path_sci_no_underscore
        elif path_sci.exists():
            checkpoint_path = path_sci
        elif path_float.exists():
            checkpoint_path = path_float
            
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=device)
            alphas.append(ckpt['alpha'])
            
    if not alphas:
        raise ValueError("No virus checkpoints found!")
        
    return torch.stack(alphas)

def compute_virus_subspace(alphas: torch.Tensor, W_dec: torch.Tensor, k: int = 4):
    """
    Compute virus subspace in residual space.
    """
    # PCA in feature space
    A_centered = alphas - alphas.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(A_centered, full_matrices=False)
    
    # Map top k to residual space
    # Vh: [num_viruses, n_features]
    # W_dec: [n_features, hidden_dim]
    R_virus = Vh[:k] @ W_dec # [k, hidden_dim]
    
    # Orthogonalize
    Qv, _ = torch.linalg.qr(R_virus.T) # [D, D]
    B_virus = Qv[:, :k] # [D, k]
    
    return B_virus

# ============================================================================
# 5. MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("PHASE 6C - TASK VS VIRUS SUBSPACE")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer_idx = 10
    
    # 1. Load Model & Data
    model, tokenizer, train_examples, test_examples = load_model_and_data(device)
    
    # Baseline
    print("\nComputing baseline metrics...")
    baseline_train = compute_metrics(model, tokenizer, train_examples)
    print(f"Baseline Train: Acc={baseline_train['accuracy']:.4f}, Diff={baseline_train['logit_diff']:.4f}")
    
    # 2. Capture Activations
    H_train, y_train = capture_activations(model, tokenizer, train_examples, layer_idx)
    print(f"Captured activations shape: {H_train.shape}")
    
    # 3. Task Direction & Subspace
    print("\nComputing Task Direction (Ridge Regression)...")
    w_task = compute_task_direction(H_train, y_train)
    
    print("Computing Task Subspace (PCA on top examples)...")
    B_task = compute_task_subspace_pca(H_train, y_train, k=4)
    
    # 4. Virus Subspace
    print("\nReconstructing Virus Subspace...")
    W_dec = load_sae_decoder(layer_idx, device)
    alphas = load_sparse_viruses(layer_idx, device)
    B_virus = compute_virus_subspace(alphas, W_dec, k=4)
    
    # 5. Quantitative Comparison
    print("\n=== COMPARISON RESULTS ===")
    
    # A. Cosine Similarity (1D)
    # u_virus is the first component of B_virus
    u_virus = B_virus[:, 0]
    cos_theta = torch.dot(w_task, u_virus).item()
    theta_deg = math.degrees(math.acos(max(min(cos_theta, 1.0), -1.0)))
    
    print(f"1. Task Direction (Ridge) vs Virus PC1:")
    print(f"   Cosine Similarity: {cos_theta:.4f}")
    print(f"   Angle: {theta_deg:.2f} degrees")
    
    # B. Energy in Virus Subspace
    # Project w_task onto B_virus
    coeffs = B_virus.T @ w_task
    energy = coeffs.norm().item()
    print(f"2. Energy of Task Direction in Virus Subspace (4D):")
    print(f"   Energy: {energy:.4f} (max 1.0)")
    
    # C. Principal Angles
    # SVD of B_virus.T @ B_task
    M = B_virus.T @ B_task
    sv = torch.linalg.svdvals(M)
    angles_deg = [math.degrees(math.acos(max(min(c.item(), 1.0), -1.0))) for c in sv]
    
    print(f"3. Principal Angles between Task Subspace (4D) and Virus Subspace (4D):")
    print(f"   Angles (deg): {[f'{a:.2f}' for a in angles_deg]}")
    
    # D. Mean Activation Overlap
    # Check if the MEAN clean activation lies in the virus subspace
    H_mean = H_train.mean(dim=0)
    H_mean_unit = H_mean / H_mean.norm()
    coeffs_mean = B_virus.T @ H_mean_unit
    energy_mean = coeffs_mean.norm().item()
    print(f"4. Energy of MEAN Clean Activation in Virus Subspace (4D):")
    print(f"   Energy: {energy_mean:.4f}")
    
    # 6. Save Results
    results = {
        "layer": layer_idx,
        "baseline": baseline_train,
        "task_direction": {
            "ridge_lambda": 1e-3
        },
        "virus_subspace": {
            "k": 4
        },
        "metrics": {
            "cos_theta_task_virus_1d": cos_theta,
            "theta_task_virus_1d_deg": theta_deg,
            "energy_task_in_virus_subspace": energy,
            "energy_mean_activation_in_virus_subspace": energy_mean,
            "principal_angles_deg": angles_deg
        }
    }
    
    json_file = "phase6c_task_vs_virus_subspace_results.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_file}")
    
    # Markdown Report
    md_file = "PHASE6C_TASK_VS_VIRUS_SUBSPACE.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Phase 6C: Task vs Virus Subspace Comparison\n\n")
        f.write(f"**Layer:** {layer_idx}\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        f.write("## 1. Alignment Metrics\n\n")
        f.write(f"- **Cosine Similarity (Task vs Virus PC1):** {cos_theta:.4f}\n")
        f.write(f"- **Angle:** {theta_deg:.2f}°\n")
        f.write(f"- **Energy of Task Direction in Virus Subspace (4D):** {energy:.4f}\n")
        f.write(f"- **Energy of MEAN Clean Activation in Virus Subspace:** {energy_mean:.4f}\n\n")
        
        f.write("## 2. Subspace Overlap (Principal Angles)\n\n")
        f.write("| Rank | Angle (deg) | Cosine |\n")
        f.write("|---|---|---|\n")
        for i, angle in enumerate(angles_deg):
            f.write(f"| {i+1} | {angle:.2f}° | {math.cos(math.radians(angle)):.4f} |\n")
            
        f.write("\n## 3. Interpretation\n\n")
        if energy_mean > 0.8:
             f.write("**CRITICAL FINDING**: The Virus Subspace captures the **MEAN** clean activation (Energy > 0.8). ")
             f.write("This explains why projecting it out destroys performance: it removes the 'DC component' or average state of the residual stream.\n")
        elif abs(cos_theta) > 0.8 or energy > 0.8:
            f.write("**HIGH OVERLAP**: The virus subspace is strongly aligned with the task direction. ")
            f.write("This confirms that the virus attacks by directly manipulating the features responsible for the task.\n")
        elif abs(cos_theta) > 0.4 or energy > 0.4:
            f.write("**MODERATE OVERLAP**: The virus subspace partially overlaps with the task direction. ")
            f.write("The attack likely distorts the task features while also introducing orthogonal noise.\n")
        else:
            f.write("**LOW OVERLAP**: The virus subspace is largely orthogonal to the task direction. ")
            f.write("This is puzzling given Phase 6B results. It suggests the 'Task Direction' (Ridge) might not capture the critical features that are being destroyed.\n")
            
    print(f"Report saved to {md_file}")

if __name__ == "__main__":
    main()
