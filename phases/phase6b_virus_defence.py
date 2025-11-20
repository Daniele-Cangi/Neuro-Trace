"""
PHASE 6B – VIRUS DEFENCE VIA SUBSPACE PROJECTION (Layer 10)

Goal:
Build a DEFENCE hook on layer 10 that projects the residual stream
orthogonally to the "virus subspace" and measure:
1) Clean accuracy (no attack) with defence ON vs OFF.
2) Attack strength (accuracy drop, logit diff) with:
   - virus only
   - virus + defence

Usage:
    python phase6b_virus_defence.py
"""

import sys
import json
import torch
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
# 1. LOAD MODEL AND DATA
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
    # Generate enough for train + test
    total_examples = num_train + num_test
    examples = generator.generate(num_examples=total_examples, ensure_diversity=True)
    
    train_examples = examples[:num_train]
    test_examples = examples[num_train:]
    
    return model, tokenizer, train_examples, test_examples

# ============================================================================
# 2. LOAD SAE AND VIRUSES
# ============================================================================

def load_sae_decoder(layer_idx: int, device: str) -> torch.Tensor:
    """Load SAE decoder matrix for the specified layer."""
    checkpoint_path = Path(f"checkpoints/all_layers_sae/layer_{layer_idx}/final.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found at {checkpoint_path}")
        
    print(f"Loading SAE for layer {layer_idx} from {checkpoint_path}...")
    feature_store = EnhancedSAEFeatureStore()
    feature_store.load_sae(str(checkpoint_path), layer=layer_idx, device=device)
    
    # Get decoder weights: [n_features, hidden_dim]
    # In EnhancedSAE, decoder is nn.Linear(dict_size, input_dim, bias=False)
    # So decoder.weight is [input_dim, dict_size]
    # We want W_dec such that delta = alpha @ W_dec
    # So W_dec should be decoder.weight.T which is [dict_size, input_dim]
    
    sae = feature_store.saes[layer_idx]
    W_dec = sae.decoder.weight.detach().T  # [features, hidden]
    
    return W_dec

def load_sparse_viruses(layer_idx: int, device: str) -> Tuple[torch.Tensor, List[float]]:
    """
    Load alpha vectors from Phase 5B checkpoints.
    Returns stacked alphas [num_viruses, n_features] and list of lambdas.
    """
    # Define lambdas used in Phase 5B
    lambdas = [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    
    alphas = []
    loaded_lambdas = []
    
    print("Loading sparse virus checkpoints...")
    for l1 in lambdas:
        # Try different filename formats
        # Actual format seems to be l1{value} without underscore
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
            print(f"  Loading {checkpoint_path}...")
            ckpt = torch.load(checkpoint_path, map_location=device)
            alpha = ckpt['alpha'] # [n_features]
            alphas.append(alpha)
            loaded_lambdas.append(l1)
        else:
            print(f"  Warning: Checkpoint for lambda={l1} not found.")
            
    if not alphas:
        raise ValueError("No virus checkpoints found!")
        
    return torch.stack(alphas), loaded_lambdas

# ============================================================================
# 3. PCA AND SUBSPACE
# ============================================================================

def run_pca_on_alphas(alphas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run PCA on stacked alpha vectors.
    Returns: U, S, Vh
    """
    # Center the data
    mean_alpha = alphas.mean(dim=0, keepdim=True)
    alphas_centered = alphas - mean_alpha
    
    # SVD
    # U: [num_viruses, num_viruses]
    # S: [num_viruses]
    # Vh: [num_viruses, n_features]
    U, S, Vh = torch.linalg.svd(alphas_centered, full_matrices=False)
    
    return U, S, Vh

def build_residual_basis(Vh: torch.Tensor, W_dec: torch.Tensor, k: int) -> torch.Tensor:
    """
    Map top k principal components from SAE space to residual space.
    Vh: [num_pc, n_features] (rows are principal directions)
    W_dec: [n_features, hidden_dim]
    Returns: basis [k, hidden_dim] (normalized)
    """
    # Take top k directions in SAE space
    # Vh rows are the principal components
    v_k = Vh[:k] # [k, n_features]
    
    # Map to residual space
    # r = v @ W_dec
    r_k = v_k @ W_dec # [k, hidden_dim]
    
    # Normalize each vector
    norms = torch.norm(r_k, dim=1, keepdim=True)
    r_k_normalized = r_k / (norms + 1e-8)
    
    return r_k_normalized

def project_off_subspace(h: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """
    Project h orthogonally to the subspace defined by basis.
    h: [batch, seq, d_model]
    basis: [k, d_model] (assumed orthonormal or at least normalized)
    """
    B, T, D = h.shape
    h_flat = h.view(-1, D) # [B*T, D]
    
    for i in range(basis.shape[0]):
        r = basis[i].view(1, -1) # [1, D]
        # coeff = <h, r>
        coeff = (h_flat * r).sum(dim=-1, keepdim=True) # [B*T, 1]
        h_flat = h_flat - coeff * r
        
    return h_flat.view(B, T, D)

# ============================================================================
# 4. HOOKS AND EVALUATION
# ============================================================================

class CombinedHook:
    def __init__(self, 
                 apply_attack: bool, 
                 apply_defence: bool, 
                 delta_attack: Optional[torch.Tensor], 
                 defence_basis: Optional[torch.Tensor]):
        self.apply_attack = apply_attack
        self.apply_defence = apply_defence
        self.delta_attack = delta_attack
        self.defence_basis = defence_basis
        
    def __call__(self, module, inputs, outputs):
        # outputs is usually a tuple (hidden_states, present_key_value_states)
        # or just hidden_states depending on model
        if isinstance(outputs, tuple):
            h = outputs[0]
        else:
            h = outputs
            
        # 1. Apply Attack
        if self.apply_attack and self.delta_attack is not None:
            # delta_attack should be [1, 1, D] or broadcastable
            h = h + self.delta_attack
            
        # 2. Apply Defence
        if self.apply_defence and self.defence_basis is not None:
            h = project_off_subspace(h, self.defence_basis)
            
        if isinstance(outputs, tuple):
            return (h,) + outputs[1:]
        else:
            return h

def compute_metrics(model, tokenizer, examples, batch_size=16):
    """Compute accuracy and logit diff."""
    correct_counts = 0
    total_counts = 0
    logit_diffs = []
    
    # Prepare batches
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
            logits = outputs.logits # [B, T, V]
            
        # Extract logits at the last position (prediction for next token)
        # Handle padding: find the last non-pad token
        # attention_mask is [B, T], 1 for token, 0 for pad
        # We want the position of the last '1'
        last_token_indices = inputs.attention_mask.sum(dim=1) - 1 # [B]
        
        # Gather logits
        # logits[b, last_token_indices[b], :]
        final_logits = logits[torch.arange(logits.shape[0]), last_token_indices, :] # [B, V]
        
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
# 5. MAIN
# ============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer_idx = 10
    
    # 1. Load Model & Data
    model, tokenizer, train_examples, test_examples = load_model_and_data(device)
    
    # 2. Baseline Metrics
    print("\nComputing baseline metrics (No Hooks)...")
    baseline_train = compute_metrics(model, tokenizer, train_examples)
    baseline_test = compute_metrics(model, tokenizer, test_examples)
    print(f"Baseline Train: Acc={baseline_train['accuracy']:.4f}, Diff={baseline_train['logit_diff']:.4f}")
    print(f"Baseline Test:  Acc={baseline_test['accuracy']:.4f}, Diff={baseline_test['logit_diff']:.4f}")
    
    # 3. Load SAE and Viruses
    W_dec = load_sae_decoder(layer_idx, device)
    alphas, lambdas = load_sparse_viruses(layer_idx, device)
    
    # 4. PCA
    print("\nRunning PCA on virus alphas...")
    U, S, Vh = run_pca_on_alphas(alphas)
    print(f"Singular values: {S.detach().cpu().numpy()}")
    
    # 5. Select Reference Virus (Attack)
    # We use lambda=1e-3 (0.001) as reference, or the one with index 3 if available
    # lambdas = [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    # 1e-3 is at index 3
    ref_idx = -1
    target_lambda = 1e-3
    for i, l in enumerate(lambdas):
        if abs(l - target_lambda) < 1e-6:
            ref_idx = i
            break
    
    if ref_idx == -1:
        print(f"Warning: Lambda {target_lambda} not found, using index 0")
        ref_idx = 0
        
    alpha_ref = alphas[ref_idx]
    delta_attack = alpha_ref @ W_dec # [hidden_dim]
    delta_attack = delta_attack.view(1, 1, -1) # [1, 1, D]
    
    print(f"Selected reference virus lambda={lambdas[ref_idx]}")
    
    # 6. Evaluation Loop
    k_values = [1, 2, 4]
    results_data = []
    
    for k in k_values:
        print(f"\n=== Evaluating k={k} ===")
        
        # Build basis
        basis = build_residual_basis(Vh, W_dec, k)
        
        # A) CLEAN DEFENCE COST (Defence ON, Attack OFF)
        print("  Evaluating Clean Defence Cost...")
        hook_clean = CombinedHook(apply_attack=False, apply_defence=True, 
                                  delta_attack=None, defence_basis=basis)
        handle = model.transformer.h[layer_idx].register_forward_hook(hook_clean)
        
        def_clean_train = compute_metrics(model, tokenizer, train_examples)
        def_clean_test = compute_metrics(model, tokenizer, test_examples)
        
        handle.remove()
        
        print(f"    Train Acc: {def_clean_train['accuracy']:.4f} (Delta: {def_clean_train['accuracy'] - baseline_train['accuracy']:.4f})")
        print(f"    Test Acc:  {def_clean_test['accuracy']:.4f} (Delta: {def_clean_test['accuracy'] - baseline_test['accuracy']:.4f})")
        
        # B) ATTACK ONLY (Defence OFF, Attack ON)
        print("  Evaluating Attack Only...")
        hook_attack = CombinedHook(apply_attack=True, apply_defence=False, 
                                   delta_attack=delta_attack, defence_basis=None)
        handle = model.transformer.h[layer_idx].register_forward_hook(hook_attack)
        
        att_train = compute_metrics(model, tokenizer, train_examples)
        att_test = compute_metrics(model, tokenizer, test_examples)
        
        handle.remove()
        
        print(f"    Train Acc: {att_train['accuracy']:.4f}")
        print(f"    Test Acc:  {att_test['accuracy']:.4f}")
        
        # C) ATTACK + DEFENCE (Defence ON, Attack ON)
        print("  Evaluating Attack + Defence...")
        hook_full = CombinedHook(apply_attack=True, apply_defence=True, 
                                 delta_attack=delta_attack, defence_basis=basis)
        handle = model.transformer.h[layer_idx].register_forward_hook(hook_full)
        
        full_train = compute_metrics(model, tokenizer, train_examples)
        full_test = compute_metrics(model, tokenizer, test_examples)
        
        handle.remove()
        
        print(f"    Train Acc: {full_train['accuracy']:.4f}")
        print(f"    Test Acc:  {full_test['accuracy']:.4f}")
        
        # Store results
        results_data.append({
            "k": k,
            "basis_dim": k,
            "defence_only": { "train": def_clean_train, "test": def_clean_test },
            "attack_only":  { "train": att_train, "test": att_test },
            "attack_plus_defence": { "train": full_train, "test": full_test }
        })
        
    # 7. Save Results
    output_file = "phase6b_virus_defence_results.json"
    final_output = {
        "layer": layer_idx,
        "baseline": { "train": baseline_train, "test": baseline_test },
        "k_values": results_data
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    # 8. Generate Markdown Report
    md_file = "PHASE6B_VIRUS_DEFENCE.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Phase 6B: Virus Defence via Subspace Projection\n\n")
        f.write(f"**Layer:** {layer_idx}\n")
        f.write(f"**Reference Virus Lambda:** {lambdas[ref_idx]}\n\n")
        
        f.write("## Baseline Performance\n")
        f.write(f"- Train Acc: {baseline_train['accuracy']:.2%}, Diff: {baseline_train['logit_diff']:.2f}\n")
        f.write(f"- Test Acc:  {baseline_test['accuracy']:.2%}, Diff: {baseline_test['logit_diff']:.2f}\n\n")
        
        f.write("## Defence Performance\n\n")
        f.write("| k | Clean Acc (Def) | Attack Acc (No Def) | Attack Acc (With Def) | Recovery |\n")
        f.write("|---|---|---|---|---|\n")
        
        for res in results_data:
            k = res['k']
            clean_acc = res['defence_only']['test']['accuracy']
            att_acc = res['attack_only']['test']['accuracy']
            def_acc = res['attack_plus_defence']['test']['accuracy']
            
            # Recovery: how much of the lost accuracy is recovered?
            # (def_acc - att_acc) / (baseline - att_acc)
            base_acc = baseline_test['accuracy']
            if base_acc - att_acc > 0:
                recovery = (def_acc - att_acc) / (base_acc - att_acc)
            else:
                recovery = 0.0
                
            f.write(f"| {k} | {clean_acc:.1%} | {att_acc:.1%} | {def_acc:.1%} | {recovery:.1%} |\n")
            
    print(f"Report saved to {md_file}")

if __name__ == "__main__":
    main()
