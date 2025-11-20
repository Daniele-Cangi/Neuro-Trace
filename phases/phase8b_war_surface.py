"""
PHASE 8B – WAR SURFACE (Layer 10)

Goal:
Perform a grid search analysis on the interaction between the adversarial "virus" vector
and the "boost" vector in Layer 10 of GPT-2.
We explore the "War Surface" defined by:
    h' = h + alpha * virus + beta * boost

This script:
1. Loads the model and a test set of IOI examples.
2. Identifies "hard" examples based on baseline performance.
3. Loads the virus (Phase 5B/6) and constrained boost (Phase 7D, R=25) vectors.
4. Evaluates the model on a grid of (alpha, beta) scales.
5. Generates a heatmap of Hard Accuracy and saves results to CSV.

Usage:
    python phase8b_war_surface.py
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

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
    "test_size": 300,
    "hard_threshold": 1.5,  # Logit diff threshold for "hard"
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "virus_path": "checkpoints/adversarial_delta_layer10.pt",
    "boost_path": "checkpoints/learned_task_boost_layer10_R25.pt",
    "alphas": [0.0, 0.5, 1.0, 1.5, 2.0],       # Attack scales
    "betas": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]    # Boost scales
}

def load_model_and_data(device: str, num_examples: int):
    """Load GPT-2 model and generate IOI test examples."""
    print(f"Loading GPT-2 model on {device}...")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Generating {num_examples} IOI examples...")
    generator = IOIDatasetGenerator()
    examples = generator.generate(num_examples=num_examples, ensure_diversity=True)
    
    return model, tokenizer, examples

def compute_metrics(model, tokenizer, examples, batch_size=16):
    """Compute accuracy and logit diff, returning detailed per-example stats."""
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
    diffs = metrics["per_example_diffs"]
    
    for i, diff in enumerate(diffs):
        if diff < threshold:
            hard_indices.append(i)
            
    return hard_indices

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

# ============================================================================
# 2. MAIN EXECUTION
# ============================================================================

def main():
    device = CONFIG["device"]
    
    # 1. Load Model & Data
    model, tokenizer, examples = load_model_and_data(device, CONFIG["test_size"])
    
    # 2. Baseline & Hard Subset Definition
    print("\nComputing baseline metrics...")
    baseline_metrics = compute_metrics(model, tokenizer, examples)
    hard_indices = identify_hard_subset(baseline_metrics, CONFIG["hard_threshold"])
    
    print(f"Baseline Accuracy: {baseline_metrics['accuracy']:.1%}")
    print(f"Baseline Logit Diff: {baseline_metrics['logit_diff']:.3f}")
    print(f"Identified {len(hard_indices)} hard examples (logit_diff < {CONFIG['hard_threshold']})")
    
    if len(hard_indices) == 0:
        print("Warning: No hard examples found. Using all examples as 'hard' subset.")
        hard_indices = list(range(len(examples)))

    hard_examples = [examples[i] for i in hard_indices]

    # 3. Load Vectors
    virus, boost = load_vectors(device)
    
    # 4. Grid Search
    results = []
    print("\nStarting War Surface Grid Search...")
    print(f"Alphas (Attack): {CONFIG['alphas']}")
    print(f"Betas (Boost): {CONFIG['betas']}")
    
    # Pre-compute heatmap data structure
    heatmap_data = {alpha: {} for alpha in CONFIG["alphas"]}
    
    for alpha in CONFIG["alphas"]:
        for beta in CONFIG["betas"]:
            # Define hook
            def hook_fn(module, inp, out):
                h = out[0]
                # h shape: [batch, seq_len, hidden_dim]
                # We need to find the last token index for each sequence in the batch
                # Note: In compute_metrics, we process in batches.
                # The hook needs to be robust to batch processing.
                # However, the hook signature doesn't give us the attention mask directly.
                # But since we are adding a constant vector to the *residual stream*,
                # we can add it to the specific position if we knew it, OR
                # we can add it to the last position.
                #
                # CRITICAL: The previous scripts added to the *last token*.
                # But inside the hook, we don't easily have the 'last_token_indices' 
                # unless we pass it or infer it.
                #
                # In `phase7d`, the hook was defined INSIDE the training loop where `last_token_indices` was available.
                # Here, `compute_metrics` handles batching and tokenization internally.
                # We need to inject the hook logic into `compute_metrics` or make `compute_metrics` accept a hook creator.
                #
                # Strategy: We will modify `compute_metrics` slightly to accept an optional hook function
                # OR we can register a hook that adds to the *entire sequence* (which might be noisy)
                # OR we can register a hook that tries to find the EOS token.
                #
                # BETTER STRATEGY: Re-implement a simple evaluation loop here that registers the hook
                # properly for each batch, just like in phase7d training loop.
                return out # Placeholder, actual logic below
            
            # We'll do the evaluation logic manually here to handle the hook correctly
            # reusing the batching logic from compute_metrics but adding the hook
            
            current_correct = 0
            current_hard_correct = 0
            current_logit_diffs = []
            
            prompts = [ex.text for ex in examples]
            correct_answers = [ex.correct_answer for ex in examples]
            incorrect_answers = [ex.incorrect_answer for ex in examples]
            
            # Indices for hard examples
            hard_indices_set = set(hard_indices)
            
            batch_size = 16
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                batch_correct = correct_answers[i:i+batch_size]
                batch_incorrect = incorrect_answers[i:i+batch_size]
                batch_indices = range(i, min(i+batch_size, len(prompts)))
                
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
                last_token_indices = inputs.attention_mask.sum(dim=1) - 1
                
                # Define hook for this batch
                def batch_hook_fn(module, inp, out):
                    h = out[0]
                    # Apply: h' = h + alpha * virus + beta * boost
                    # We add to the last token position
                    perturbation = alpha * virus + beta * boost
                    h[torch.arange(h.shape[0]), last_token_indices, :] += perturbation
                    return (h,) + out[1:]
                
                # Register hook
                handle = model.transformer.h[CONFIG["layer"]].register_forward_hook(batch_hook_fn)
                
                try:
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                finally:
                    handle.remove()
                
                final_logits = logits[torch.arange(logits.shape[0]), last_token_indices, :]
                
                for j, (corr, incorr) in enumerate(zip(batch_correct, batch_incorrect)):
                    global_idx = batch_indices[j]
                    
                    corr_id = tokenizer.encode(" " + corr)[0]
                    incorr_id = tokenizer.encode(" " + incorr)[0]
                    
                    corr_logit = final_logits[j, corr_id].item()
                    incorr_logit = final_logits[j, incorr_id].item()
                    
                    diff = corr_logit - incorr_logit
                    current_logit_diffs.append(diff)
                    
                    is_correct = corr_logit > incorr_logit
                    
                    if is_correct:
                        current_correct += 1
                        if global_idx in hard_indices_set:
                            current_hard_correct += 1
            
            test_acc = current_correct / len(examples)
            hard_acc = current_hard_correct / len(hard_indices) if hard_indices else 0.0
            mean_logit_diff = np.mean(current_logit_diffs)
            
            results.append({
                "alpha": alpha,
                "beta": beta,
                "test_acc": test_acc,
                "hard_acc": hard_acc,
                "mean_logit_diff": mean_logit_diff
            })
            
            heatmap_data[alpha][beta] = hard_acc
            # print(f"  alpha={alpha}, beta={beta} -> Hard Acc: {hard_acc:.1%}")

    # 5. Save Results
    df = pd.DataFrame(results)
    df.to_csv("war_surface.csv", index=False)
    print("\nResults saved to war_surface.csv")
    
    # 6. Print Heatmap
    print("\n=== Hard Accuracy Heatmap (Rows=Alpha, Cols=Beta) ===")
    header = "       " + "".join([f"B={b:<5}" for b in CONFIG["betas"]])
    print(header)
    print("-" * len(header))
    
    for alpha in CONFIG["alphas"]:
        row_str = f"A={alpha:<3} | "
        for beta in CONFIG["betas"]:
            val = heatmap_data[alpha][beta]
            row_str += f"{val:.2f}  "
        print(row_str)
        
    # 7. Summary
    best_row = df.loc[df['hard_acc'].idxmax()]
    print("\n=== Summary ===")
    print(f"Best Configuration: Alpha={best_row['alpha']}, Beta={best_row['beta']}")
    print(f"  -> Hard Acc: {best_row['hard_acc']:.1%}")
    print(f"  -> Test Acc: {best_row['test_acc']:.1%}")
    
    # Find minimal beta for alpha=1.0 to reach 90% hard acc
    alpha_1_df = df[df['alpha'] == 1.0]
    passing_betas = alpha_1_df[alpha_1_df['hard_acc'] >= 0.90]
    
    if not passing_betas.empty:
        min_beta = passing_betas['beta'].min()
        print(f"Minimal Beta to neutralize Alpha=1.0 (Hard Acc >= 90%): {min_beta}")
    else:
        print("No Beta in range fully neutralized Alpha=1.0 (Hard Acc < 90%)")

if __name__ == "__main__":
    main()
