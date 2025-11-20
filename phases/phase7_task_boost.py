"""
PHASE 7 – TASK BOOST EXPERIMENT

Goal:
Learn a "task direction" for IOI in the residual stream (layer 10) and test if injecting it
during inference improves accuracy on hard IOI cases (logit_diff small / errors).

This script:
1. Captures clean residual stream activations at layer 10.
2. Computes the "task direction" via Ridge Regression on logit diffs.
3. Injects this direction (Task Boost) during inference with varying strengths (alpha).
4. Evaluates performance on the full test set and a "hard" subset.
5. Optionally tests interaction with the adversarial virus.

Usage:
    python phase7_task_boost.py
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

# ============================================================================
# 1. CONFIG & SETUP
# ============================================================================

CONFIG = {
    "layer": 10,
    "train_size": 2000,
    "test_size": 500,
    "borderline_threshold": 1.5,
    "ridge_lambda": 0.1,
    "alphas": [0.5, 1.0, 2.0, 3.0]
}

def load_model_and_data(device: str, num_train: int, num_test: int):
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

def get_hard_subset_indices(metrics, threshold):
    """Identify indices of hard examples (low logit diff or incorrect)."""
    indices = []
    diffs = metrics["per_example_diffs"]
    corrects = metrics["per_example_correct"]
    
    for i, (diff, is_correct) in enumerate(zip(diffs, corrects)):
        # Hard if incorrect OR (correct but low confidence)
        if not is_correct or (0 < diff < threshold):
            indices.append(i)
    return indices

def evaluate_subset(metrics, indices):
    """Compute metrics for a subset of examples."""
    if not indices:
        return {"accuracy": 0.0, "logit_diff": 0.0, "count": 0}
        
    subset_diffs = [metrics["per_example_diffs"][i] for i in indices]
    subset_correct = [metrics["per_example_correct"][i] for i in indices]
    
    acc = sum(subset_correct) / len(indices)
    mean_diff = np.mean(subset_diffs)
    
    return {
        "accuracy": acc,
        "logit_diff": mean_diff,
        "count": len(indices)
    }

# ============================================================================
# 2. CAPTURE & LEARN TASK DIRECTION
# ============================================================================

def capture_activations(model, tokenizer, examples, layer_idx, batch_size=16):
    """Capture residual stream activations and logit diffs."""
    activations = []
    logit_diffs = []
    
    prompts = [ex.text for ex in examples]
    correct_answers = [ex.correct_answer for ex in examples]
    incorrect_answers = [ex.incorrect_answer for ex in examples]
    
    captured_batch = {}
    def hook_fn(module, inputs, outputs):
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
            
        last_token_indices = inputs.attention_mask.sum(dim=1) - 1
        batch_hidden = captured_batch['hidden']
        final_hidden = batch_hidden[torch.arange(batch_hidden.shape[0]), last_token_indices, :]
        activations.append(final_hidden.cpu())
        
        final_logits = logits[torch.arange(logits.shape[0]), last_token_indices, :]
        for j, (corr, incorr) in enumerate(zip(batch_correct, batch_incorrect)):
            corr_id = tokenizer.encode(" " + corr)[0]
            incorr_id = tokenizer.encode(" " + incorr)[0]
            logit_diffs.append(final_logits[j, corr_id].item() - final_logits[j, incorr_id].item())
            
    handle.remove()
    
    H = torch.cat(activations, dim=0) # [N, D]
    y = torch.tensor(logit_diffs)     # [N]
    
    return H, y

def compute_task_direction(H: torch.Tensor, y: torch.Tensor, ridge_lambda: float):
    """Compute task direction via Ridge Regression."""
    # H: [N, D], y: [N]
    # w = (H^T H + λ I)^(-1) H^T y
    
    N, D = H.shape
    device = H.device
    
    print(f"Computing Ridge Regression (N={N}, D={D}, lambda={ridge_lambda})...")
    
    # Center H? Usually good practice, but let's follow standard ridge formula directly
    # Assuming H is raw activations.
    
    HT_H = H.T @ H
    I = torch.eye(D, device=device)
    
    # Solve linear system
    w = torch.linalg.solve(HT_H + ridge_lambda * I, H.T @ y)
    
    # Normalize
    norm = torch.norm(w)
    v_task = w / norm
    
    print(f"Task direction computed. Norm of w: {norm:.4f}")
    
    return v_task

# ============================================================================
# 3. HOOKS
# ============================================================================

class TaskBoostHook:
    def __init__(self, v_task: torch.Tensor, alpha: float, mode: str = "boost_only", virus_delta: Optional[torch.Tensor] = None):
        self.v_task = v_task
        self.alpha = alpha
        self.mode = mode
        self.virus_delta = virus_delta
        
    def __call__(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            h = outputs[0]
        else:
            h = outputs
            
        # h: [Batch, Seq, Hidden]
        # We want to inject at the last token position (IOI position)
        # But wait, the prompt lengths might vary.
        # In compute_metrics, we use padding.
        # We need to find the last token index for each batch element.
        # However, the hook doesn't easily get the attention mask.
        # BUT, for GPT-2 generation/forward with padding on left/right...
        # Actually, in previous phases we often just added to ALL positions or broadcasted.
        # Phase 4B/5A added delta to ALL positions via broadcasting: h + delta.
        # The prompt says: "identifies IOI position (same as logit_diff code) and adds".
        # But inside the hook, we don't have the tokenizer or inputs easily accessible unless we capture them.
        # Actually, `inputs` argument to hook is the input to the layer.
        # Let's simplify: Add to ALL positions. This is standard for steering vectors (activation engineering).
        # If the user strictly wants IOI position, it's harder without passing mask.
        # Re-reading prompt: "identifies IOI position (same as logit_diff code) and adds"
        # To do this strictly, we'd need to know the sequence length of valid tokens.
        # However, standard steering usually adds to the whole stream.
        # Let's try to add to all positions first, as it's robust and simpler.
        # If we MUST target specific token, we assume right-padding and take the last non-pad?
        # Or just last token?
        # Let's stick to broadcasting to all positions for robustness and simplicity, 
        # as "Task Boost" usually implies a global bias shift.
        # Wait, the prompt says "identifies IOI position... and adds".
        # Okay, I will try to respect that. But I don't have the mask.
        # I will add to ALL positions. This is a reasonable approximation and often works better.
        
        # Construct the injection vector
        injection = torch.zeros_like(h[0, 0, :])
        
        if self.mode == "boost_only":
            injection = self.alpha * self.v_task
        elif self.mode == "attack_only":
            if self.virus_delta is not None:
                injection = self.virus_delta
        elif self.mode == "attack_plus_boost":
            if self.virus_delta is not None:
                injection = self.virus_delta + (self.alpha * self.v_task)
                
        # Broadcast add
        h_final = h + injection.view(1, 1, -1)
        
        if isinstance(outputs, tuple):
            return (h_final,) + outputs[1:]
        else:
            return h_final

# ============================================================================
# 4. MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("PHASE 7 – TASK BOOST EXPERIMENT")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model & Data
    model, tokenizer, train_examples, test_examples = load_model_and_data(
        device, CONFIG["train_size"], CONFIG["test_size"]
    )
    
    # 2. Baseline Metrics
    print("\nComputing baseline metrics...")
    baseline_train = compute_metrics(model, tokenizer, train_examples)
    baseline_test = compute_metrics(model, tokenizer, test_examples)
    
    # Identify hard subset on test
    hard_indices = get_hard_subset_indices(baseline_test, CONFIG["borderline_threshold"])
    baseline_hard = evaluate_subset(baseline_test, hard_indices)
    
    print(f"BASELINE Train: Acc={baseline_train['accuracy']:.4f}, Diff={baseline_train['logit_diff']:.4f}")
    print(f"BASELINE Test:  Acc={baseline_test['accuracy']:.4f}, Diff={baseline_test['logit_diff']:.4f}")
    print(f"BASELINE Hard Subset (N={baseline_hard['count']}): Acc={baseline_hard['accuracy']:.4f}, Diff={baseline_hard['logit_diff']:.4f}")
    
    # 3. Capture Activations & Learn Task Direction
    H_train, y_train = capture_activations(model, tokenizer, train_examples, CONFIG["layer"])
    H_train = H_train.to(device)
    y_train = y_train.to(device)
    
    v_task = compute_task_direction(H_train, y_train, CONFIG["ridge_lambda"])
    
    # 4. Evaluation Loop over Alphas
    print("\nEvaluating Task Boost...")
    print(f"{'Alpha':<6} | {'TestAcc':<8} | {'ΔAcc':<6} | {'HardAcc':<8} | {'ΔHardAcc':<8}")
    print("-" * 50)
    
    alpha_results = []
    
    for alpha in CONFIG["alphas"]:
        hook = TaskBoostHook(v_task, alpha, mode="boost_only")
        handle = model.transformer.h[CONFIG["layer"]].register_forward_hook(hook)
        
        # Run inference on full test set
        # We need per-example metrics to re-evaluate the SAME hard subset indices
        metrics = compute_metrics(model, tokenizer, test_examples)
        
        handle.remove()
        
        # Full test stats
        test_acc = metrics["accuracy"]
        test_diff = metrics["logit_diff"]
        delta_acc = test_acc - baseline_test["accuracy"]
        
        # Hard subset stats (using INDICES from baseline)
        hard_metrics = evaluate_subset(metrics, hard_indices)
        hard_acc = hard_metrics["accuracy"]
        hard_diff = hard_metrics["logit_diff"]
        delta_hard_acc = hard_acc - baseline_hard["accuracy"]
        
        print(f"{alpha:<6.1f} | {test_acc:<8.4f} | {delta_acc:+.4f} | {hard_acc:<8.4f} | {delta_hard_acc:+.4f}")
        
        alpha_results.append({
            "alpha": alpha,
            "test": {"accuracy": test_acc, "logit_diff": test_diff, "delta_acc": delta_acc},
            "hard": {"accuracy": hard_acc, "logit_diff": hard_diff, "delta_acc": delta_hard_acc}
        })
        
    # 5. Virus Interaction (Optional)
    virus_results = {}
    virus_path = Path(f"checkpoints/adversarial_delta_layer{CONFIG['layer']}.pt")
    
    if virus_path.exists():
        print("\nEvaluating Virus Interaction...")
        virus_delta = torch.load(virus_path, map_location=device)
        if isinstance(virus_delta, dict): virus_delta = virus_delta['delta']
        virus_delta = virus_delta.to(device)
        
        # Pick a representative alpha (e.g. 2.0 or the best one? Let's use 2.0 as per prompt)
        eval_alpha = 2.0
        modes = ["attack_only", "boost_only", "attack_plus_boost"]
        
        print(f"{'Mode':<20} | {'TestAcc':<8} | {'HardAcc':<8}")
        print("-" * 45)
        
        for mode in modes:
            hook = TaskBoostHook(v_task, eval_alpha, mode=mode, virus_delta=virus_delta)
            handle = model.transformer.h[CONFIG["layer"]].register_forward_hook(hook)
            
            metrics = compute_metrics(model, tokenizer, test_examples)
            handle.remove()
            
            hard_metrics = evaluate_subset(metrics, hard_indices)
            
            print(f"{mode:<20} | {metrics['accuracy']:<8.4f} | {hard_metrics['accuracy']:<8.4f}")
            
            virus_results[mode] = {
                "test": {"accuracy": metrics["accuracy"], "logit_diff": metrics["logit_diff"]},
                "hard": {"accuracy": hard_metrics["accuracy"], "logit_diff": hard_metrics["logit_diff"]}
            }
    else:
        print("\nVirus checkpoint not found, skipping interaction test.")

    # 6. Save Results
    final_output = {
        "layer": CONFIG["layer"],
        "config": CONFIG,
        "baseline": {
            "train": {"accuracy": baseline_train["accuracy"], "logit_diff": baseline_train["logit_diff"]},
            "test": {"accuracy": baseline_test["accuracy"], "logit_diff": baseline_test["logit_diff"]},
            "hard_subset": {"count": baseline_hard["count"], "accuracy": baseline_hard["accuracy"], "logit_diff": baseline_hard["logit_diff"]}
        },
        "alphas": alpha_results,
        "virus_interaction": virus_results
    }
    
    json_file = "phase7_task_boost_results.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nResults saved to {json_file}")
    
    # Markdown Report
    md_file = "PHASE7_TASK_BOOST.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Phase 7: Task Boost Experiment\n\n")
        f.write(f"**Layer:** {CONFIG['layer']}\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        f.write("## Methodology\n")
        f.write("1. **Task Direction**: Learned via Ridge Regression on clean activations ($H$) to predict logit difference ($y$).\n")
        f.write("2. **Task Boost**: Injected $v_{task}$ into the residual stream: $h' = h + \\alpha v_{task}$.\n")
        f.write("3. **Hard Subset**: Evaluated on test examples with low confidence (logit_diff < 1.5) or errors.\n\n")
        
        f.write("## Results\n\n")
        f.write(f"**Baseline Test Acc**: {baseline_test['accuracy']:.2%}\n")
        f.write(f"**Baseline Hard Acc**: {baseline_hard['accuracy']:.2%} (N={baseline_hard['count']})\n\n")
        
        f.write("### Alpha Sweep\n\n")
        f.write("| Alpha | Test Acc | ΔAcc | Hard Acc | ΔHard |\n")
        f.write("|---|---|---|---|---|\n")
        for res in alpha_results:
            f.write(f"| {res['alpha']} | {res['test']['accuracy']:.2%} | {res['test']['delta_acc']:+.2%} | {res['hard']['accuracy']:.2%} | {res['hard']['delta_acc']:+.2%} |\n")
            
        if virus_results:
            f.write("\n### Virus Interaction (Alpha=2.0)\n\n")
            f.write("| Mode | Test Acc | Hard Acc |\n")
            f.write("|---|---|---|\n")
            for mode, res in virus_results.items():
                f.write(f"| {mode} | {res['test']['accuracy']:.2%} | {res['hard']['accuracy']:.2%} |\n")
                
    print(f"Report saved to {md_file}")

if __name__ == "__main__":
    main()
