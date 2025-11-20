"""
PHASE 9B – GATED DEFENCE V2 (Virus Detector)

Goal:
Implement a smart "Gated Defence" that uses the trained Virus Detector (Phase 9A)
to decide when to apply the Boost vector.

Comparison:
1. NO_DEFENCE (Attack only)
2. STATIC_DEFENCE (Attack + Boost always)
3. GATED_DEFENCE_V2 (Attack + Boost only if Detector says "Infected")

Usage:
    python phase9b_gated_defence_v2.py
"""

import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from transformers import AutoModelForCausalLM, AutoTokenizer
from neurotrace.datasets import IOIDatasetGenerator

# ============================================================================
# 1. CONFIG & SETUP
# ============================================================================

CONFIG = {
    "layer": 10,
    "test_size": 500,
    "alpha_attack": 1.0,
    "beta_boost": 3.0,          # From War Surface results
    "detector_threshold": 0.5,
    "hard_threshold": 1.5,      # For Hard Accuracy metric
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "virus_path": "checkpoints/adversarial_delta_layer10.pt",
    "boost_path": "checkpoints/learned_task_boost_layer10_R25.pt",
    "detector_path": "checkpoints/virus_detector_layer10.pt",
    "detector_config": "virus_detector_config.json",
    "seed": 42
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

# ============================================================================
# 2. MODEL DEFINITIONS
# ============================================================================

class VirusDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ============================================================================
# 3. UTILS
# ============================================================================

def load_resources():
    print(f"Loading GPT-2 on {CONFIG['device']}...")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(CONFIG['device'])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading Vectors...")
    virus = torch.load(CONFIG['virus_path'], map_location=CONFIG['device']).float()
    boost = torch.load(CONFIG['boost_path'], map_location=CONFIG['device']).float()
    
    print(f"Loading Detector...")
    with open(CONFIG['detector_config'], 'r') as f:
        det_config = json.load(f)
    
    detector = VirusDetector(input_dim=det_config['input_dim'], hidden_dim=det_config['hidden_dim'])
    detector.load_state_dict(torch.load(CONFIG['detector_path'], map_location=CONFIG['device']))
    detector.to(CONFIG['device'])
    detector.eval()
    
    return model, tokenizer, virus, boost, detector

def get_test_data(num_examples):
    print(f"Generating {num_examples} IOI test examples...")
    generator = IOIDatasetGenerator()
    # Generate a bit more to ensure diversity then slice, or just generate exact
    examples = generator.generate(num_examples=num_examples, ensure_diversity=True)
    return examples

def run_forward_pass(model, tokenizer, examples, layer_idx, perturbation_vector=None):
    """
    Runs a forward pass and returns:
    - logits (at last token)
    - hidden_states (at last token, at layer_idx)
    """
    prompts = [ex.text for ex in examples]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    last_token_indices = inputs.attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(len(prompts), device=model.device)
    
    captured_h = {}
    
    def hook_fn(module, inp, out):
        h = out[0]
        # Capture 'clean' h before perturbation if we want clean h, 
        # but if perturbation_vector is passed, we want to capture the perturbed h?
        # Actually for features we need the state *after* potential attack.
        
        if perturbation_vector is not None:
            h[batch_indices, last_token_indices, :] += perturbation_vector
            
        captured_h['h'] = h[batch_indices, last_token_indices, :].detach().clone()
        return (h,) + out[1:]
    
    handle = model.transformer.h[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            outputs = model(**inputs)
    finally:
        handle.remove()
        
    logits = outputs.logits[batch_indices, last_token_indices, :]
    return logits, captured_h['h']

def calculate_metrics(logits, examples, tokenizer):
    """
    Calculate accuracy, logit_diffs, and correctness list.
    """
    correct_counts = 0
    logit_diffs = []
    is_correct_list = []
    
    for i, ex in enumerate(examples):
        corr_id = tokenizer.encode(" " + ex.correct_answer)[0]
        incorr_id = tokenizer.encode(" " + ex.incorrect_answer)[0]
        
        corr_logit = logits[i, corr_id].item()
        incorr_logit = logits[i, incorr_id].item()
        
        diff = corr_logit - incorr_logit
        logit_diffs.append(diff)
        
        is_correct = corr_logit > incorr_logit
        is_correct_list.append(is_correct)
        if is_correct:
            correct_counts += 1
            
    accuracy = correct_counts / len(examples)
    return accuracy, logit_diffs, is_correct_list

# ============================================================================
# 4. MAIN LOGIC
# ============================================================================

def main():
    # 1. Setup
    model, tokenizer, virus, boost, detector = load_resources()
    examples = get_test_data(CONFIG["test_size"])
    
    # Pre-calculate vectors
    vec_attack = CONFIG["alpha_attack"] * virus
    vec_static_defence = CONFIG["alpha_attack"] * virus + CONFIG["beta_boost"] * boost
    
    virus_unit = virus / (virus.norm() + 1e-8)
    
    print("\n=== Running Evaluation Passes ===")
    
    # PASS 1: CLEAN (Baseline)
    print("1. Running CLEAN pass...")
    logits_clean, h_clean = run_forward_pass(model, tokenizer, examples, CONFIG["layer"], perturbation_vector=None)
    acc_clean, ld_clean, corr_clean = calculate_metrics(logits_clean, examples, tokenizer)
    
    # Identify Hard Subset (based on Baseline)
    hard_indices = [i for i, ld in enumerate(ld_clean) if ld < CONFIG["hard_threshold"]]
    print(f"   Baseline Acc: {acc_clean:.1%}")
    print(f"   Hard Subset Size: {len(hard_indices)} (Threshold < {CONFIG['hard_threshold']})")
    
    # PASS 2: NO DEFENCE (Attack Only)
    print("2. Running NO_DEFENCE pass (Attack Only)...")
    logits_attack, h_attack = run_forward_pass(model, tokenizer, examples, CONFIG["layer"], perturbation_vector=vec_attack)
    acc_attack, ld_attack, corr_attack = calculate_metrics(logits_attack, examples, tokenizer)
    
    # PASS 3: STATIC DEFENCE (Attack + Boost)
    print("3. Running STATIC_DEFENCE pass (Attack + Boost)...")
    logits_static, h_static = run_forward_pass(model, tokenizer, examples, CONFIG["layer"], perturbation_vector=vec_static_defence)
    acc_static, ld_static, corr_static = calculate_metrics(logits_static, examples, tokenizer)
    
    # ========================================================================
    # GATED DEFENCE V2 LOGIC
    # ========================================================================
    print("\n=== Simulating GATED_DEFENCE_V2 ===")
    
    gated_correct_count = 0
    gated_hard_correct_count = 0
    gate_activations = 0
    false_positives = 0
    
    # We need to construct features for the detector based on the ATTACKED state
    # Features: [ld_current, proj_current, norm_current, ld_base, delta_ld]
    
    # Prepare batch tensors for efficiency
    # Note: In the loop we process example by example to simulate the decision
    
    for i in range(len(examples)):
        # Extract features from the "Current" state (which is Attacked)
        curr_ld = ld_attack[i]
        curr_h = h_attack[i]
        curr_norm = curr_h.norm().item()
        curr_proj = torch.dot(curr_h, virus_unit).item()
        
        base_ld = ld_clean[i]
        delta_ld = curr_ld - base_ld
        
        # Feature vector
        features = torch.tensor([
            curr_ld,
            curr_proj,
            curr_norm,
            base_ld,
            delta_ld
        ], dtype=torch.float32).to(CONFIG["device"])
        
        # Detector prediction
        with torch.no_grad():
            logit = detector(features.unsqueeze(0)) # Add batch dim
            prob = torch.sigmoid(logit).item()
            
        is_infected = prob > CONFIG["detector_threshold"]
        
        # Decision
        if is_infected:
            # Apply Boost -> Use Static Defence Result
            final_is_correct = corr_static[i]
            gate_activations += 1
            
            # Check False Positive: Gate fired, but Attack Only was already correct?
            # Or "False Positive" as defined in prompt: 
            # "detector ha attivato il gate ma il modello ATTACKED era comunque corretto"
            if corr_attack[i]:
                false_positives += 1
        else:
            # No Boost -> Use No Defence Result
            final_is_correct = corr_attack[i]
            
        if final_is_correct:
            gated_correct_count += 1
            if i in hard_indices:
                gated_hard_correct_count += 1
                
    # Calculate Metrics
    # 1. No Defence
    no_def_hard_acc = sum(corr_attack[i] for i in hard_indices) / len(hard_indices) if hard_indices else 0.0
    
    # 2. Static Defence
    static_hard_acc = sum(corr_static[i] for i in hard_indices) / len(hard_indices) if hard_indices else 0.0
    
    # 3. Gated Defence
    gated_acc = gated_correct_count / len(examples)
    gated_hard_acc = gated_hard_correct_count / len(hard_indices) if hard_indices else 0.0
    gate_rate = gate_activations / len(examples)
    fp_rate = false_positives / len(examples)
    
    # ========================================================================
    # OUTPUT
    # ========================================================================
    
    print("\n=== Results Summary ===")
    print(f"{'Mode':<20} | {'TestAcc':<8} | {'HardAcc':<8} | {'GateRate':<8} | {'FP_rate':<8}")
    print("-" * 65)
    print(f"{'no_defence':<20} | {acc_attack:<8.1%} | {no_def_hard_acc:<8.1%} | {'-':<8} | {'-':<8}")
    print(f"{'static_defence':<20} | {acc_static:<8.1%} | {static_hard_acc:<8.1%} | {'100%':<8} | {'-':<8}")
    print(f"{'gated_defence_v2':<20} | {gated_acc:<8.1%} | {gated_hard_acc:<8.1%} | {gate_rate:<8.1%} | {fp_rate:<8.1%}")
    
    # Save JSON
    results = {
        "config": CONFIG,
        "metrics": {
            "no_defence": {
                "accuracy": acc_attack,
                "hard_accuracy": no_def_hard_acc
            },
            "static_defence": {
                "accuracy": acc_static,
                "hard_accuracy": static_hard_acc
            },
            "gated_defence_v2": {
                "accuracy": gated_acc,
                "hard_accuracy": gated_hard_acc,
                "gate_rate": gate_rate,
                "fp_rate": fp_rate
            }
        }
    }
    
    out_file = "phase9b_gated_defence_v2_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()
