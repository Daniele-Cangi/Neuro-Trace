"""
PHASE 6D – CENTERED DEFENCE (Layer 10)

Goal:
Test a "centered projection" defence on layer 10.
Unlike Phase 6B (which projected the raw residual stream), this phase:
1. Subtracts the MEAN clean activation from the residual stream.
2. Projects out the "virus subspace" from the centered residual.
3. Adds the MEAN clean activation back.

This tests the hypothesis that the virus attacks by shifting the mean activation
along a sensitive direction, and that preserving the mean is crucial for performance.

Usage:
    python phase6d_centered_defence.py
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
# 2. MEAN ACTIVATION & VIRUS SUBSPACE
# ============================================================================

def capture_mean_activation(model, tokenizer, examples, layer_idx, batch_size=16):
    """Capture mean residual stream activation at layer_idx."""
    activations = []
    
    prompts = [ex.text for ex in examples]
    
    captured_batch = {}
    def hook_fn(module, inputs, outputs):
        captured_batch['hidden'] = outputs[0].detach()
        
    handle = model.transformer.h[layer_idx].register_forward_hook(hook_fn)
    
    print(f"Capturing mean activation from layer {layer_idx}...")
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            _ = model(**inputs)
            
        last_token_indices = inputs.attention_mask.sum(dim=1) - 1
        batch_hidden = captured_batch['hidden']
        final_hidden = batch_hidden[torch.arange(batch_hidden.shape[0]), last_token_indices, :]
        activations.append(final_hidden.cpu())
            
    handle.remove()
    
    H = torch.cat(activations, dim=0).to(model.device)
    mean_activation = H.mean(dim=0) # [D]
    
    return mean_activation

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

def compute_virus_basis(alphas: torch.Tensor, W_dec: torch.Tensor, k: int = 4):
    """Compute orthonormal virus basis in residual space."""
    A_centered = alphas - alphas.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(A_centered, full_matrices=False)
    
    R_virus = Vh[:k] @ W_dec # [k, hidden_dim]
    
    Qv, _ = torch.linalg.qr(R_virus.T) # [D, D]
    B_virus = Qv[:, :k] # [D, k]
    
    return B_virus

# ============================================================================
# 3. HOOKS
# ============================================================================

class CenteredDefenceHook:
    def __init__(self, mode: str, mean_activation: torch.Tensor, 
                 virus_basis: torch.Tensor, delta: torch.Tensor):
        self.mode = mode
        self.mean_activation = mean_activation
        self.virus_basis = virus_basis
        self.delta = delta
        
    def project_off(self, h_centered, B):
        # h_centered: [B, T, D] or [B, D]
        # B: [D, k]
        
        # Flatten for matmul
        orig_shape = h_centered.shape
        h_flat = h_centered.view(-1, h_centered.shape[-1]) # [N, D]
        
        # coeffs = h @ B -> [N, k]
        coeffs = h_flat @ B
        
        # recon = coeffs @ B.T -> [N, D]
        recon = coeffs @ B.T
        
        h_proj = h_flat - recon
        return h_proj.view(orig_shape)

    def __call__(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            h = outputs[0]
        else:
            h = outputs
            
        # h shape: [Batch, Seq, Hidden]
            
        if self.mode == "attack_only":
            # Broadcast delta [D] to [Batch, Seq, Hidden]
            h_final = h + self.delta.view(1, 1, -1)
            
        elif self.mode == "defence_only":
            # h_centered = h - mean
            h_centered = h - self.mean_activation.view(1, 1, -1)
            # project off
            h_clean = self.project_off(h_centered, self.virus_basis)
            # h_final = h_clean + mean
            h_final = h_clean + self.mean_activation.view(1, 1, -1)
            
        elif self.mode == "attack_plus_defence":
            # 1. Attack
            h_attacked = h + self.delta.view(1, 1, -1)
            # 2. Defence (Centered)
            h_centered = h_attacked - self.mean_activation.view(1, 1, -1)
            h_clean = self.project_off(h_centered, self.virus_basis)
            h_final = h_clean + self.mean_activation.view(1, 1, -1)
            
        else:
            h_final = h
            
        if isinstance(outputs, tuple):
            return (h_final,) + outputs[1:]
        else:
            return h_final

# ============================================================================
# 4. MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("PHASE 6D - CENTERED DEFENCE")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer_idx = 10
    
    # 1. Load Model & Data
    model, tokenizer, train_examples, test_examples = load_model_and_data(device)
    
    # Baseline
    print("\nComputing baseline metrics...")
    baseline_train = compute_metrics(model, tokenizer, train_examples)
    baseline_test = compute_metrics(model, tokenizer, test_examples)
    print(f"BASELINE Train: Acc={baseline_train['accuracy']:.4f}, Diff={baseline_train['logit_diff']:.4f}")
    print(f"BASELINE Test:  Acc={baseline_test['accuracy']:.4f}, Diff={baseline_test['logit_diff']:.4f}")
    
    # 2. Mean Activation & Virus Basis
    mean_activation = capture_mean_activation(model, tokenizer, train_examples, layer_idx)
    
    print("\nReconstructing Virus Subspace...")
    W_dec = load_sae_decoder(layer_idx, device)
    alphas = load_sparse_viruses(layer_idx, device)
    B_virus = compute_virus_basis(alphas, W_dec, k=4)
    
    # 3. Load Adversarial Delta
    # Try to find the delta checkpoint
    delta_path = Path(f"checkpoints/adversarial_delta_layer{layer_idx}.pt")
    if not delta_path.exists():
        # Fallback to virus delta if available, or raise error
        # For now, let's assume we want the one from Phase 5A/4B
        raise FileNotFoundError(f"Adversarial delta not found at {delta_path}")
        
    print(f"Loading adversarial delta from {delta_path}...")
    delta_ckpt = torch.load(delta_path, map_location=device)
    # Check format: might be raw tensor or dict
    if isinstance(delta_ckpt, dict) and 'delta' in delta_ckpt:
        delta = delta_ckpt['delta']
    elif isinstance(delta_ckpt, torch.Tensor):
        delta = delta_ckpt
    else:
        # Phase 5A saved as raw tensor usually
        delta = delta_ckpt
        
    delta = delta.to(device)
    
    # 4. Evaluation Loop
    modes = ["attack_only", "defence_only", "attack_plus_defence"]
    results = {}
    
    for mode in modes:
        print(f"\nEvaluating mode: {mode}...")
        hook = CenteredDefenceHook(mode, mean_activation, B_virus, delta)
        handle = model.transformer.h[layer_idx].register_forward_hook(hook)
        
        train_res = compute_metrics(model, tokenizer, train_examples)
        test_res = compute_metrics(model, tokenizer, test_examples)
        
        handle.remove()
        
        results[mode] = {"train": train_res, "test": test_res}
        
        print(f"  Train Acc={train_res['accuracy']:.4f}, Diff={train_res['logit_diff']:.4f}")
        print(f"  Test  Acc={test_res['accuracy']:.4f}, Diff={test_res['logit_diff']:.4f}")
        
    # 5. Save Results
    final_output = {
        "layer": layer_idx,
        "baseline": {"train": baseline_train, "test": baseline_test},
        "attack_only": results["attack_only"],
        "defence_only_centered": results["defence_only"],
        "attack_plus_defence_centered": results["attack_plus_defence"]
    }
    
    json_file = "phase6d_centered_defence_results.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nResults saved to {json_file}")
    
    # Markdown Report
    md_file = "PHASE6D_CENTERED_DEFENCE.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Phase 6D: Centered Virus Defence\n\n")
        f.write(f"**Layer:** {layer_idx}\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        f.write("## Methodology\n")
        f.write("Unlike Phase 6B, this defence preserves the mean activation:\n")
        f.write("1. $h_{centered} = h - \\mu$\n")
        f.write("2. $h_{clean} = (I - BB^T) h_{centered}$\n")
        f.write("3. $h_{final} = h_{clean} + \\mu$\n\n")
        
        f.write("## Results\n\n")
        f.write("| Scenario | Train Acc | Test Acc | Train Diff | Test Diff |\n")
        f.write("|---|---|---|---|---|\n")
        
        # Baseline
        f.write(f"| Baseline | {baseline_train['accuracy']:.2%} | {baseline_test['accuracy']:.2%} | {baseline_train['logit_diff']:.2f} | {baseline_test['logit_diff']:.2f} |\n")
        
        # Modes
        for mode in modes:
            res = results[mode]
            name = mode.replace("_", " ").title()
            f.write(f"| {name} | {res['train']['accuracy']:.2%} | {res['test']['accuracy']:.2%} | {res['train']['logit_diff']:.2f} | {res['test']['logit_diff']:.2f} |\n")
            
        f.write("\n## Conclusion\n\n")
        
        def_acc = results["defence_only"]["test"]["accuracy"]
        att_def_acc = results["attack_plus_defence"]["test"]["accuracy"]
        base_acc = baseline_test["accuracy"]
        
        if def_acc > 0.9 * base_acc:
            f.write("**SUCCESS (Clean Performance)**: The centered defence preserves model performance on clean data.\n")
            if att_def_acc > 0.5 * base_acc:
                f.write("**SUCCESS (Robustness)**: The defence significantly mitigates the attack.\n")
            else:
                f.write("**PARTIAL SUCCESS**: Clean performance is preserved, but the attack is still effective.\n")
        else:
            f.write("**FAILURE**: Even with centering, the defence degrades model performance significantly.\n")
            
    print(f"Report saved to {md_file}")

if __name__ == "__main__":
    main()
