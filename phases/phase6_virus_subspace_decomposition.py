"""
PHASE 6 – VIRUS SUBSPACE DECOMPOSITION (Layer 10, SAE space)

Goal:
1) Load several SAE viruses (alpha vectors) for layer 10 from Phase 5B.
2) Stack them into a matrix in SAE feature space and run PCA to estimate
   a low-dimensional "virus subspace".
3) Reconstruct an approximate virus using only the first k principal components
   and measure how much attack effect is preserved.

Usage:
    python phase6_virus_subspace_decomposition.py
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


def load_sae_decoder(layer_idx: int, device: str) -> torch.Tensor:
    """Load SAE decoder matrix for the specified layer."""
    checkpoint_path = Path(f"checkpoints/all_layers_sae/layer_{layer_idx}/final.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found at {checkpoint_path}")
        
    print(f"Loading SAE for layer {layer_idx} from {checkpoint_path}...")
    feature_store = EnhancedSAEFeatureStore()
    feature_store.load_sae(str(checkpoint_path), layer=layer_idx, device=device)
    
    # Get decoder weights: [n_features, hidden_dim]
    # Note: EnhancedSAE.decoder is a Linear layer, weight is [hidden, features]
    # We want [features, hidden] to multiply alpha [features] @ W_dec
    # Actually, standard linear layer: y = x @ W.T + b
    # If alpha is [1, features], we want delta [1, hidden]
    # delta = alpha @ W_dec
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
    # Based on phase5b_sparse_sae_virus_results.json
    lambdas = [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    
    alphas = []
    loaded_lambdas = []
    
    print("Loading sparse virus checkpoints...")
    for l1 in lambdas:
        # Format filename based on how it was saved in learn_sparse_sae_virus.py
        # Filename format: sparse_sae_virus_layer10_l{val}.pt
        # where val is formatted with scientific notation e.g. 1e-04 -> 1e-04
        # Let's try to find the file
        
        # Construct potential filename patterns
        if l1 == 0.0:
            fname = f"sparse_sae_virus_layer{layer_idx}_l0e+00.pt"
            # Also try alternative formatting if needed
            if not (Path("checkpoints") / fname).exists():
                fname = f"sparse_sae_virus_layer{layer_idx}_l0.0.pt"
        else:
            # Format like 1e-04
            s_fmt = f"{l1:.0e}"
            # The previous script might have used a specific format
            # Let's look at the file list provided in context:
            # sparse_sae_virus_layer10_l10e+00.pt (likely 0.0?)
            # sparse_sae_virus_layer10_l11e-02.pt (1e-2)
            # sparse_sae_virus_layer10_l11e-03.pt (1e-3)
            # sparse_sae_virus_layer10_l11e-04.pt (1e-4)
            # sparse_sae_virus_layer10_l15e-02.pt (5e-2)
            # sparse_sae_virus_layer10_l15e-03.pt (5e-3)
            # sparse_sae_virus_layer10_l15e-04.pt (5e-4)
            
            # Reconstruct exact filenames based on directory listing
            base = f"{l1:.0e}"
            parts = base.split('e')
            mantissa = parts[0]
            exponent = parts[1]
            # The listing shows 'l1' prefix then '1e-04' etc.
            # Wait, the listing shows: sparse_sae_virus_layer10_l11e-04.pt
            # This looks like l1 + 1e-04.
            
            fname = f"sparse_sae_virus_layer{layer_idx}_l1{mantissa}e{exponent}.pt"

        path = Path("checkpoints") / fname
        
        if path.exists():
            print(f"  Loading {path} (lambda={l1})...")
            ckpt = torch.load(path, map_location=device)
            if 'alpha' in ckpt:
                alphas.append(ckpt['alpha'])
                loaded_lambdas.append(l1)
            else:
                print(f"  Warning: 'alpha' not found in {path}")
        else:
            # Try to find by glob if exact name match fails
            print(f"  Warning: {path} not found, skipping lambda={l1}")

    if not alphas:
        raise ValueError("No virus checkpoints found!")
        
    # Stack alphas: [num_viruses, n_features]
    alpha_matrix = torch.stack(alphas, dim=0)
    return alpha_matrix, loaded_lambdas


def run_pca_on_alphas(alpha_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run PCA on the alpha matrix.
    Returns: mean_alpha, principal_components (V), singular_values (S)
    """
    # Center the data
    mean_alpha = alpha_matrix.mean(dim=0)
    centered_alphas = alpha_matrix - mean_alpha
    
    # Run SVD: A = U S V^T
    # V rows are principal components (eigenvectors of A^T A)
    # torch.linalg.svd returns U, S, Vh
    # Vh shape is [min(N, D), D] (if full_matrices=False)
    # Rows of Vh are the principal directions
    U, S, Vh = torch.linalg.svd(centered_alphas, full_matrices=False)
    
    return mean_alpha, Vh, S


def compute_metrics(model, tokenizer, examples, layer_idx, delta_vector, device, batch_size=16):
    """Compute logit diff and accuracy with and without hook."""
    
    # Define hook function
    def apply_delta_hook(module, input, output):
        # output[0] is [batch, seq, hidden]
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        # Add delta to the last token position (or all? usually last for IOI)
        # In previous phases we added to the last token of the prompt
        # Let's add to all positions for simplicity/robustness as in learn_sparse_sae_virus.py
        # "We add the delta to the residual stream at all positions"
        
        # Ensure delta is [1, 1, hidden] for broadcasting
        d = delta_vector.view(1, 1, -1)
        hidden_states = hidden_states + d
        
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    # Prepare batches
    texts = [ex.text for ex in examples]
    answers = [ex.correct_answer for ex in examples]
    wrong_answers = [ex.incorrect_answer for ex in examples]
    
    logit_diffs = []
    accuracies = []
    
    # Register hook if delta is provided
    hook_handle = None
    if delta_vector is not None:
        layer = model.transformer.h[layer_idx]
        hook_handle = layer.register_forward_hook(apply_delta_hook)
    
    try:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_ans = answers[i:i+batch_size]
            batch_wrong = wrong_answers[i:i+batch_size]
            
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            
            # Calculate target positions (last non-pad token)
            attention_mask = inputs.attention_mask
            target_positions = (attention_mask.sum(dim=1) - 1).tolist()
            
            with torch.no_grad():
                outputs = model(**inputs)
                # logits = outputs.logits[:, -1, :] # This was wrong for padded batches
                
            for j, (ans, wrong) in enumerate(zip(batch_ans, batch_wrong)):
                pos = target_positions[j]
                logits = outputs.logits[j, pos, :] # [vocab]
                
                # Prepend space as in learn_sparse_sae_virus.py
                ans_id = tokenizer.encode(" " + ans, add_special_tokens=False)[0]
                wrong_id = tokenizer.encode(" " + wrong, add_special_tokens=False)[0]
                
                # Logit diff
                diff = logits[ans_id].item() - logits[wrong_id].item()
                logit_diffs.append(diff)
                
                # Accuracy (Top-1)
                pred_id = logits.argmax().item()
                if pred_id == ans_id:
                    accuracies.append(1.0)
                else:
                    accuracies.append(0.0)
                    
    finally:
        if hook_handle:
            hook_handle.remove()
            
    return {
        "logit_diff": float(np.mean(logit_diffs)),
        "accuracy": float(np.mean(accuracies))
    }


def main():
    print("=" * 80)
    print("PHASE 6: VIRUS SUBSPACE DECOMPOSITION")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer_idx = 10
    
    # 1. Load Model & Data
    model, tokenizer, train_ex, test_ex = load_model_and_data(device)
    
    # 2. Load SAE Decoder
    W_dec = load_sae_decoder(layer_idx, device)
    print(f"SAE Decoder shape: {W_dec.shape}")
    
    # 3. Load Sparse Viruses
    alpha_matrix, lambdas = load_sparse_viruses(layer_idx, device)
    print(f"Loaded {len(lambdas)} virus vectors. Shape: {alpha_matrix.shape}")
    
    # 4. Run PCA
    print("Running PCA on virus vectors...")
    mean_alpha, Vh, S = run_pca_on_alphas(alpha_matrix)
    print(f"Principal components shape: {Vh.shape}")
    print(f"Singular values: {S.cpu().numpy()}")
    
    # 5. Select Reference Virus
    # We want a strong virus to reconstruct. 
    # From Phase 5B results, lambda=1e-3 had strong effect (-97.5% acc change)
    # Let's find the index of lambda=1e-3 or closest
    target_lambda = 1e-3
    try:
        ref_idx = lambdas.index(target_lambda)
    except ValueError:
        # Fallback to middle one
        ref_idx = len(lambdas) // 2
        
    print(f"Reference virus: lambda={lambdas[ref_idx]}")
    alpha_ref = alpha_matrix[ref_idx]
    
    # Project reference onto PCA basis
    alpha_ref_centered = alpha_ref - mean_alpha
    # coeffs = alpha_ref_centered @ Vh.T
    coeffs = torch.matmul(alpha_ref_centered, Vh.T)
    
    # 6. Sweep k components
    k_values = [1, 2, 4, 8, 16]
    # Limit k to max available components
    max_k = Vh.shape[0]
    k_values = [k for k in k_values if k <= max_k]
    
    results = []
    
    # Compute Baseline first (no hook)
    print("\nComputing baseline metrics...")
    base_train = compute_metrics(model, tokenizer, train_ex, layer_idx, None, device)
    base_test = compute_metrics(model, tokenizer, test_ex, layer_idx, None, device)
    print(f"Baseline Test Acc: {base_test['accuracy']:.1%}, Logit Diff: {base_test['logit_diff']:.2f}")
    
    print("\nSweeping k components...")
    for k in k_values:
        print(f"\n--- k = {k} ---")
        
        # Reconstruct alpha_k
        # alpha_k = mean + sum(coeff_i * V_i)
        # Vh rows are V_i
        reconstruction = torch.matmul(coeffs[:k], Vh[:k])
        alpha_k = mean_alpha + reconstruction
        
        # Map to delta
        delta_k = torch.matmul(alpha_k, W_dec) # [hidden]
        
        # Metrics
        delta_norm = torch.norm(delta_k).item()
        num_active = (torch.abs(alpha_k) > 1e-3).sum().item()
        
        print(f"  Delta norm: {delta_norm:.2f}")
        print(f"  Active features (>1e-3): {num_active}")
        
        # Evaluate
        train_metrics = compute_metrics(model, tokenizer, train_ex, layer_idx, delta_k, device)
        test_metrics = compute_metrics(model, tokenizer, test_ex, layer_idx, delta_k, device)
        
        print(f"  Test Acc: {test_metrics['accuracy']:.1%} (Delta: {test_metrics['accuracy'] - base_test['accuracy']:.1%})")
        print(f"  Test Diff: {test_metrics['logit_diff']:.2f} (Delta: {test_metrics['logit_diff'] - base_test['logit_diff']:.2f})")
        
        res_entry = {
            "k": k,
            "delta_norm": delta_norm,
            "num_effective_features": num_active,
            "train": {
                "baseline_diff": base_train['logit_diff'],
                "steered_diff": train_metrics['logit_diff'],
                "delta_diff": train_metrics['logit_diff'] - base_train['logit_diff'],
                "baseline_acc": base_train['accuracy'],
                "steered_acc": train_metrics['accuracy'],
                "delta_acc": train_metrics['accuracy'] - base_train['accuracy']
            },
            "test": {
                "baseline_diff": base_test['logit_diff'],
                "steered_diff": test_metrics['logit_diff'],
                "delta_diff": test_metrics['logit_diff'] - base_test['logit_diff'],
                "baseline_acc": base_test['accuracy'],
                "steered_acc": test_metrics['accuracy'],
                "delta_acc": test_metrics['accuracy'] - base_test['accuracy']
            }
        }
        results.append(res_entry)
        
    # 7. Save Results
    out_file = "phase6_virus_subspace_results.json"
    with open(out_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "layer": layer_idx,
            "reference_lambda": lambdas[ref_idx],
            "singular_values": S.cpu().tolist(),
            "results": results
        }, f, indent=2)
        
    print(f"\nSaved results to {out_file}")
    
    # 8. Generate Report
    report_path = "PHASE6_VIRUS_SUBSPACE.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 6: Virus Subspace Decomposition Results\n\n")
        f.write(f"**Layer:** {layer_idx}\n")
        f.write(f"**Reference Virus Lambda:** {lambdas[ref_idx]}\n\n")
        
        f.write("## PCA Spectrum\n")
        f.write(f"Singular values: {S.cpu().numpy()}\n\n")
        
        f.write("## Reconstruction Performance\n\n")
        f.write("| k | Active Feats | ||Delta|| | Test Acc | Test ΔAcc | Test Diff | Test ΔDiff |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        
        for r in results:
            t = r["test"]
            f.write(f"| {r['k']} | {r['num_effective_features']} | {r['delta_norm']:.2f} | "
                    f"{t['steered_acc']:.1%} | {t['delta_acc']:+.1%} | "
                    f"{t['steered_diff']:.2f} | {t['delta_diff']:+.2f} |\n")
                    
    print(f"Saved report to {report_path}")
    print("\nDone!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
