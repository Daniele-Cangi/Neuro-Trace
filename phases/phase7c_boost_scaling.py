"""
PHASE 7C – BOOST SCALING (Layer 10)

Goal:
Evaluate the "Task Boost" vector at different norms (scales) to determine
if the protective effect survives at lower, more "physiological" magnitudes.

This script:
1. Loads the learned boost vector from Phase 7B.
2. Normalizes it and scales it to target norms [5, 10, 20, 30, 50, 80, 112].
3. Evaluates Test Accuracy and Hard Subset Accuracy for:
   - Boost Only
   - Attack Only (Virus)
   - Attack + Boost

Usage:
    python phase7c_boost_scaling.py
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
    "hard_threshold": 1.0,
    "scales": [5.0, 10.0, 20.0, 30.0, 50.0, 80.0, 112.0],
    "device": "cuda" if torch.cuda.is_available() else "cpu"
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

def identify_hard_subset(metrics, threshold):
    """Identify indices of hard examples."""
    indices = []
    diffs = metrics["per_example_diffs"]
    corrects = metrics["per_example_correct"]
    
    for i, (diff, is_correct) in enumerate(zip(diffs, corrects)):
        if not is_correct or (0 < diff < threshold):
            indices.append(i)
    return indices

def evaluate_subset(metrics, indices):
    """Compute metrics for a subset of examples."""
    if not indices:
        return {"accuracy": 0.0, "logit_diff": 0.0}
        
    subset_correct = [metrics["per_example_correct"][i] for i in indices]
    acc = sum(subset_correct) / len(indices)
    return {"accuracy": acc}

# ============================================================================
# 2. HOOKS
# ============================================================================

class ScalingHook:
    def __init__(self, v_boost, virus_delta=None, mode="boost_only"):
        self.v_boost = v_boost
        self.virus_delta = virus_delta
        self.mode = mode
        
    def __call__(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            h = outputs[0]
        else:
            h = outputs
            
        # Broadcast injection to all tokens (standard steering)
        injection = torch.zeros_like(h[0, 0, :])
        
        boost_vec = self.v_boost if self.v_boost is not None else 0
        virus_vec = self.virus_delta if self.virus_delta is not None else 0
        
        if self.mode == "boost_only":
            injection = boost_vec
        elif self.mode == "attack_only":
            injection = virus_vec
        elif self.mode == "attack_plus_boost":
            injection = virus_vec + boost_vec
            
        if isinstance(injection, torch.Tensor):
            h_final = h + injection.view(1, 1, -1)
        else:
            h_final = h
            
        if isinstance(outputs, tuple):
            return (h_final,) + outputs[1:]
        else:
            return h_final

# ============================================================================
# 3. MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("PHASE 7C – BOOST SCALING")
    print("=" * 80)
    
    device = CONFIG["device"]
    
    # 1. Load Data
    model, tokenizer, train_examples, test_examples = load_model_and_data(
        device, CONFIG["train_size"], CONFIG["test_size"]
    )
    
    # 2. Baseline
    print("\nComputing baseline metrics...")
    baseline_test = compute_metrics(model, tokenizer, test_examples)
    hard_indices = identify_hard_subset(baseline_test, CONFIG["hard_threshold"])
    baseline_hard = evaluate_subset(baseline_test, hard_indices)
    
    print(f"BASELINE Test: Acc={baseline_test['accuracy']:.4f}")
    print(f"BASELINE Hard: Acc={baseline_hard['accuracy']:.4f} (N={len(hard_indices)})")
    
    # 3. Load Vectors
    # Boost Vector
    boost_path = Path(f"checkpoints/learned_task_boost_layer{CONFIG['layer']}.pt")
    if not boost_path.exists():
        raise FileNotFoundError(f"Boost vector not found at {boost_path}")
    print(f"Loading boost vector from {boost_path}...")
    v_boost_raw = torch.load(boost_path, map_location=device)
    v_dir = v_boost_raw / torch.norm(v_boost_raw)
    
    # Virus Vector
    virus_path = Path(f"checkpoints/adversarial_delta_layer{CONFIG['layer']}.pt")
    if not virus_path.exists():
        raise FileNotFoundError(f"Virus vector not found at {virus_path}")
    print(f"Loading virus vector from {virus_path}...")
    virus_delta = torch.load(virus_path, map_location=device)
    if isinstance(virus_delta, dict): virus_delta = virus_delta['delta']
    virus_delta = virus_delta.to(device)
    
    # 4. Scaling Loop
    results = {
        "baseline": {
            "test_acc": baseline_test["accuracy"],
            "hard_acc": baseline_hard["accuracy"]
        },
        "scales": []
    }
    
    print(f"\n{'Scale':<6} | {'Mode':<18} | {'TestAcc':<8} | {'HardAcc':<8}")
    print("-" * 50)
    
    for scale in CONFIG["scales"]:
        v_scaled = v_dir * scale
        
        scale_res = {"norm": scale}
        
        modes = ["boost_only", "attack_only", "attack_plus_boost"]
        
        for mode in modes:
            hook = ScalingHook(v_scaled, virus_delta, mode=mode)
            handle = model.transformer.h[CONFIG["layer"]].register_forward_hook(hook)
            
            metrics = compute_metrics(model, tokenizer, test_examples)
            handle.remove()
            
            hard_metrics = evaluate_subset(metrics, hard_indices)
            
            scale_res[mode] = {
                "test_acc": metrics["accuracy"],
                "hard_acc": hard_metrics["accuracy"]
            }
            
            # Only print relevant lines (attack_only is constant across scales, but good to verify)
            if mode != "attack_only" or scale == CONFIG["scales"][0]:
                 print(f"{scale:<6.1f} | {mode:<18} | {metrics['accuracy']:<8.4f} | {hard_metrics['accuracy']:<8.4f}")
                 
        results["scales"].append(scale_res)
        
    # 5. Save Results
    json_file = "phase7c_boost_scaling_results.json"
    def convert(o):
        if isinstance(o, np.float32): return float(o)
        return o
        
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {json_file}")
    
    # 6. Markdown Report
    md_file = "PHASE7C_BOOST_SCALING.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Phase 7C: Boost Scaling Analysis\n\n")
        f.write(f"**Layer:** {CONFIG['layer']}\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        f.write("## Baseline\n")
        f.write(f"- **Test Acc**: {baseline_test['accuracy']:.2%}\n")
        f.write(f"- **Hard Acc**: {baseline_hard['accuracy']:.2%}\n\n")
        
        f.write("## Scaling Results\n\n")
        f.write("| Norm | Mode | Test Acc | Hard Acc |\n")
        f.write("|---|---|---|---|\n")
        
        for item in results["scales"]:
            norm = item["norm"]
            # Boost Only
            bo = item["boost_only"]
            f.write(f"| {norm:.1f} | Boost Only | {bo['test_acc']:.2%} | {bo['hard_acc']:.2%} |\n")
            # Attack + Boost
            ab = item["attack_plus_boost"]
            f.write(f"| {norm:.1f} | Attack + Boost | {ab['test_acc']:.2%} | {ab['hard_acc']:.2%} |\n")
            
        f.write("\n## Conclusion\n")
        
        # Analyze physiological range (20-30)
        physio_scales = [s for s in results["scales"] if 20 <= s["norm"] <= 30]
        if physio_scales:
            best_physio = max(physio_scales, key=lambda x: x["attack_plus_boost"]["test_acc"])
            norm = best_physio["norm"]
            acc = best_physio["attack_plus_boost"]["test_acc"]
            if acc > 0.9:
                f.write(f"**SUCCESS**: At physiological norms (~{norm}), the boost maintains high robustness ({acc:.2%}).\n")
            elif acc > 0.6:
                f.write(f"**PARTIAL**: At physiological norms (~{norm}), the boost offers partial protection ({acc:.2%}).\n")
            else:
                f.write(f"**FAILURE**: At physiological norms, the boost is insufficient to counter the virus.\n")
        
    print(f"Report saved to {md_file}")

if __name__ == "__main__":
    main()
