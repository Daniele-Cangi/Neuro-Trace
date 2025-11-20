"""
PHASE 8A – IMMUNE GATING (Layer 10)

Goal:
Implement a conditional defence mechanism ("immune gating") that activates the
task boost vector ONLY when the model is uncertain (low logit difference).

Mechanism:
1. Measure baseline logit difference (confidence).
2. If confidence < GATE_THRESHOLD, activate Boost (Attack + Boost).
3. Otherwise, remain passive (Attack only).

This simulates an "immune system" that only intervenes when necessary,
minimizing false positives and potential side effects on clean/easy data.

Usage:
    python phase8a_immune_gating.py
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

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
    "train_size": 2000,
    "test_size": 500,
    "hard_threshold": 1.5,      # Definition of "hard" for evaluation metrics
    "gate_threshold": 3.0,      # Threshold to trigger defence (logit diff)
    "alpha_attack": 1.0,        # Virus scale
    "beta_boost": 2.0,          # Boost scale
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "virus_path": "checkpoints/adversarial_delta_layer10.pt",
    "boost_path": "checkpoints/learned_task_boost_layer10_R25.pt"
}

def load_model_and_data(device: str, num_train: int, num_test: int):
    """Load GPT-2 model and generate IOI dataset."""
    print(f"Loading GPT-2 model on {device}...")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Generating IOI dataset ({num_train} train, {num_test} test)...")
    generator = IOIDatasetGenerator()
    total_examples = num_train + num_test
    examples = generator.generate(num_examples=total_examples, ensure_diversity=True)
    
    # We only need test examples for this phase, but keeping signature consistent
    test_examples = examples[num_train:]
    
    return model, tokenizer, test_examples

def load_vectors(device):
    """Load virus and boost vectors."""
    print(f"Loading vectors to {device}...")
    
    if not Path(CONFIG["virus_path"]).exists():
        raise FileNotFoundError(f"Virus vector not found at {CONFIG['virus_path']}")
    virus = torch.load(CONFIG["virus_path"], map_location=device)
    print(f"  Virus loaded: norm={virus.norm().item():.2f}")
    
    if not Path(CONFIG["boost_path"]).exists():
        raise FileNotFoundError(f"Boost vector not found at {CONFIG['boost_path']}")
    boost = torch.load(CONFIG["boost_path"], map_location=device)
    print(f"  Boost loaded: norm={boost.norm().item():.2f}")
    
    return virus, boost

def evaluate_model(
    model, 
    tokenizer, 
    examples, 
    layer_idx, 
    perturbation_fn: Optional[Callable] = None, 
    batch_size=16
):
    """
    Evaluate model with an optional perturbation function.
    perturbation_fn signature: (hidden_states, last_token_indices) -> perturbed_hidden_states
    """
    correct_counts = 0
    total_counts = 0
    logit_diffs = []
    is_correct_list = []
    
    prompts = [ex.text for ex in examples]
    correct_answers = [ex.correct_answer for ex in examples]
    incorrect_answers = [ex.incorrect_answer for ex in examples]
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_correct = correct_answers[i:i+batch_size]
        batch_incorrect = incorrect_answers[i:i+batch_size]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        last_token_indices = inputs.attention_mask.sum(dim=1) - 1
        
        # Define hook for this batch
        handle = None
        if perturbation_fn:
            def hook_fn(module, inp, out):
                h = out[0]
                h_perturbed = perturbation_fn(h, last_token_indices)
                return (h_perturbed,) + out[1:]
            
            handle = model.transformer.h[layer_idx].register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
        finally:
            if handle:
                handle.remove()
            
        final_logits = logits[torch.arange(logits.shape[0]), last_token_indices, :]
        
        for j, (corr, incorr) in enumerate(zip(batch_correct, batch_incorrect)):
            corr_id = tokenizer.encode(" " + corr)[0]
            incorr_id = tokenizer.encode(" " + incorr)[0]
            
            corr_logit = final_logits[j, corr_id].item()
            incorr_logit = final_logits[j, incorr_id].item()
            
            diff = corr_logit - incorr_logit
            logit_diffs.append(diff)
            
            is_correct = corr_logit > incorr_logit
            is_correct_list.append(is_correct)
            
            if is_correct:
                correct_counts += 1
            total_counts += 1
            
    accuracy = correct_counts / total_counts if total_counts > 0 else 0.0
    mean_logit_diff = np.mean(logit_diffs) if logit_diffs else 0.0
    
    return {
        "accuracy": accuracy,
        "logit_diff": mean_logit_diff,
        "per_example_diffs": logit_diffs,
        "per_example_correct": is_correct_list
    }

def identify_hard_subset(metrics, threshold):
    """Identify indices of hard examples (low logit diff)."""
    hard_indices = []
    for i, diff in enumerate(metrics["per_example_diffs"]):
        if diff < threshold:
            hard_indices.append(i)
    return hard_indices

def calculate_subset_accuracy(results, indices):
    """Calculate accuracy for a subset of indices."""
    if not indices:
        return 0.0
    correct = sum(1 for i in indices if results["per_example_correct"][i])
    return correct / len(indices)

# ============================================================================
# 2. MAIN EXECUTION
# ============================================================================

def main():
    device = CONFIG["device"]
    
    # 1. Load Model & Data
    model, tokenizer, test_examples = load_model_and_data(device, CONFIG["train_size"], CONFIG["test_size"])
    
    # 2. Load Vectors
    virus, boost = load_vectors(device)
    
    # 3. Baseline Evaluation (No Hooks)
    print("\n--- Baseline Evaluation ---")
    baseline_results = evaluate_model(model, tokenizer, test_examples, CONFIG["layer"], perturbation_fn=None)
    
    hard_indices = identify_hard_subset(baseline_results, CONFIG["hard_threshold"])
    baseline_hard_acc = calculate_subset_accuracy(baseline_results, hard_indices)
    
    print(f"Baseline Accuracy: {baseline_results['accuracy']:.1%}")
    print(f"Baseline Hard Acc: {baseline_hard_acc:.1%} (Threshold < {CONFIG['hard_threshold']})")
    print(f"Hard examples count: {len(hard_indices)}")
    
    # 4. Define Perturbation Functions
    
    # Scenario 1: No Defence (Attack Only)
    def attack_only_fn(h, last_token_indices):
        # h' = h + alpha * virus
        perturbation = CONFIG["alpha_attack"] * virus
        h[torch.arange(h.shape[0]), last_token_indices, :] += perturbation
        return h

    # Scenario 2: Static Defence (Attack + Boost)
    def static_defence_fn(h, last_token_indices):
        # h' = h + alpha * virus + beta * boost
        perturbation = CONFIG["alpha_attack"] * virus + CONFIG["beta_boost"] * boost
        h[torch.arange(h.shape[0]), last_token_indices, :] += perturbation
        return h
        
    # 5. Run Evaluations
    
    print("\n--- Scenario 1: No Defence (Under Attack) ---")
    no_defence_results = evaluate_model(model, tokenizer, test_examples, CONFIG["layer"], attack_only_fn)
    no_defence_hard_acc = calculate_subset_accuracy(no_defence_results, hard_indices)
    print(f"Accuracy: {no_defence_results['accuracy']:.1%}")
    print(f"Hard Acc: {no_defence_hard_acc:.1%}")
    
    print("\n--- Scenario 2: Static Defence (Always Boost) ---")
    static_defence_results = evaluate_model(model, tokenizer, test_examples, CONFIG["layer"], static_defence_fn)
    static_defence_hard_acc = calculate_subset_accuracy(static_defence_results, hard_indices)
    print(f"Accuracy: {static_defence_results['accuracy']:.1%}")
    print(f"Hard Acc: {static_defence_hard_acc:.1%}")
    
    # 6. Scenario 3: Gated Defence (Simulated)
    print("\n--- Scenario 3: Gated Defence (Immune Gating) ---")
    print(f"Gate Threshold: Logit Diff < {CONFIG['gate_threshold']}")
    
    gated_correct = 0
    gated_hard_correct = 0
    gate_activations = 0
    false_positives = 0
    
    gated_diffs = []
    
    for i in range(len(test_examples)):
        # Check gate condition on BASELINE performance
        baseline_diff = baseline_results["per_example_diffs"][i]
        baseline_is_correct = baseline_results["per_example_correct"][i]
        
        gate_triggered = baseline_diff < CONFIG["gate_threshold"]
        
        if gate_triggered:
            # Gate fires: Use Static Defence result
            is_correct = static_defence_results["per_example_correct"][i]
            diff = static_defence_results["per_example_diffs"][i]
            gate_activations += 1
            
            # False positive: Gate fired but baseline was already correct
            if baseline_is_correct:
                false_positives += 1
        else:
            # Gate closed: Use No Defence result (Attack only)
            is_correct = no_defence_results["per_example_correct"][i]
            diff = no_defence_results["per_example_diffs"][i]
            
        if is_correct:
            gated_correct += 1
            if i in hard_indices:
                gated_hard_correct += 1
                
        gated_diffs.append(diff)
        
    gated_acc = gated_correct / len(test_examples)
    gated_hard_acc = gated_hard_correct / len(hard_indices) if hard_indices else 0.0
    gate_rate = gate_activations / len(test_examples)
    fp_rate = false_positives / len(test_examples)
    
    print(f"Accuracy: {gated_acc:.1%}")
    print(f"Hard Acc: {gated_hard_acc:.1%}")
    print(f"Gate Rate: {gate_rate:.1%} ({gate_activations}/{len(test_examples)})")
    print(f"False Positive Rate: {fp_rate:.1%} ({false_positives} unnecessary boosts)")
    
    # 7. Summary Table
    print("\n=== Results Summary ===")
    print(f"{'Mode':<15} | {'TestAcc':<8} | {'HardAcc':<8} | {'GateRate':<8} | {'FP_rate':<8}")
    print("-" * 60)
    print(f"{'no_defence':<15} | {no_defence_results['accuracy']:<8.1%} | {no_defence_hard_acc:<8.1%} | {'-':<8} | {'-':<8}")
    print(f"{'static_defence':<15} | {static_defence_results['accuracy']:<8.1%} | {static_defence_hard_acc:<8.1%} | {'100%':<8} | {'-':<8}")
    print(f"{'gated_defence':<15} | {gated_acc:<8.1%} | {gated_hard_acc:<8.1%} | {gate_rate:<8.1%} | {fp_rate:<8.1%}")
    
    # 8. Save Results
    results_data = {
        "config": CONFIG,
        "baseline": {
            "accuracy": baseline_results["accuracy"],
            "hard_accuracy": baseline_hard_acc
        },
        "no_defence": {
            "accuracy": no_defence_results["accuracy"],
            "hard_accuracy": no_defence_hard_acc
        },
        "static_defence": {
            "accuracy": static_defence_results["accuracy"],
            "hard_accuracy": static_defence_hard_acc
        },
        "gated_defence": {
            "accuracy": gated_acc,
            "hard_accuracy": gated_hard_acc,
            "gate_rate": gate_rate,
            "false_positive_rate": fp_rate
        }
    }
    
    with open("phase8a_immune_gating_results.json", "w") as f:
        json.dump(results_data, f, indent=4)
    print("\nResults saved to phase8a_immune_gating_results.json")

if __name__ == "__main__":
    main()
