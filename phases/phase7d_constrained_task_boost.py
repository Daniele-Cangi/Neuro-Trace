"""
PHASE 7D – CONSTRAINED TASK BOOST (Layer 10)

Goal:
Learn a "task steroid" vector v_boost for layer 10 via gradient descent,
but with a STRICT NORM CONSTRAINT (R_TARGET = 25.0).
This tests if we can achieve robustness without "exploding" the activation norm
to unnatural levels (like the ||v||=112 seen in Phase 7B).

This script:
1. Generates IOI data and identifies "hard" vs "easy" examples.
2. Trains v_boost with projected gradient descent (clipping norm to R_TARGET).
3. Evaluates the constrained boost against the virus.

Usage:
    python phase7d_constrained_task_boost.py
"""

import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
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
    "hard_threshold": 1.0,  # Logit diff threshold for "hard"
    "margin": 0.5,          # Allowed degradation for "easy"
    "lambda_easy": 0.1,
    "R_TARGET": 25.0,       # Strict norm constraint
    "lr": 0.01,
    "epochs": 20,
    "batch_size": 32,
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

def identify_subsets(metrics, threshold):
    """Identify hard and easy indices."""
    hard_indices = []
    easy_indices = []
    
    diffs = metrics["per_example_diffs"]
    corrects = metrics["per_example_correct"]
    
    for i, (diff, is_correct) in enumerate(zip(diffs, corrects)):
        if not is_correct or (0 < diff < threshold):
            hard_indices.append(i)
        else:
            easy_indices.append(i)
            
    return hard_indices, easy_indices

# ============================================================================
# 2. TRAINING (CONSTRAINED)
# ============================================================================

class ConstrainedBoostTrainer:
    def __init__(self, model, tokenizer, layer_idx, device):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.device = device
        self.v_boost = nn.Parameter(torch.zeros(768, device=device))
        
    def train(self, train_examples, hard_indices, easy_indices, baseline_diffs):
        optimizer = optim.Adam([self.v_boost], lr=CONFIG["lr"])
        
        hard_set = set(hard_indices)
        easy_set = set(easy_indices)
        
        print(f"\nStarting constrained training (R_TARGET={CONFIG['R_TARGET']})...")
        print(f"Hard examples: {len(hard_indices)}, Easy examples: {len(easy_indices)}")
        
        prompts = [ex.text for ex in train_examples]
        correct_answers = [ex.correct_answer for ex in train_examples]
        incorrect_answers = [ex.incorrect_answer for ex in train_examples]
        
        correct_ids = [self.tokenizer.encode(" " + ans)[0] for ans in correct_answers]
        incorrect_ids = [self.tokenizer.encode(" " + ans)[0] for ans in incorrect_answers]
        
        batch_size = CONFIG["batch_size"]
        num_batches = (len(train_examples) + batch_size - 1) // batch_size
        
        for epoch in range(CONFIG["epochs"]):
            total_loss = 0
            total_hard_loss = 0
            total_easy_loss = 0
            
            perm = np.random.permutation(len(train_examples))
            
            for b in range(num_batches):
                batch_idx = perm[b*batch_size : (b+1)*batch_size]
                
                batch_prompts = [prompts[i] for i in batch_idx]
                batch_corr_ids = torch.tensor([correct_ids[i] for i in batch_idx], device=self.device)
                batch_incorr_ids = torch.tensor([incorrect_ids[i] for i in batch_idx], device=self.device)
                
                batch_is_hard = [i in hard_set for i in batch_idx]
                batch_is_easy = [i in easy_set for i in batch_idx]
                batch_baseline_diffs = torch.tensor([baseline_diffs[i] for i in batch_idx], device=self.device)
                
                inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                last_token_indices = inputs.attention_mask.sum(dim=1) - 1
                
                def hook_fn(module, inp, out):
                    h = out[0]
                    h[torch.arange(h.shape[0]), last_token_indices, :] += self.v_boost
                    return (h,) + out[1:]
                
                handle = self.model.transformer.h[self.layer_idx].register_forward_hook(hook_fn)
                outputs = self.model(**inputs)
                logits = outputs.logits
                handle.remove()
                
                final_logits = logits[torch.arange(logits.shape[0]), last_token_indices, :]
                corr_logits = final_logits.gather(1, batch_corr_ids.unsqueeze(1)).squeeze(1)
                incorr_logits = final_logits.gather(1, batch_incorr_ids.unsqueeze(1)).squeeze(1)
                current_diffs = corr_logits - incorr_logits
                
                loss_hard = torch.tensor(0.0, device=self.device)
                loss_easy = torch.tensor(0.0, device=self.device)
                
                hard_mask = torch.tensor(batch_is_hard, device=self.device)
                if hard_mask.any():
                    loss_hard = -current_diffs[hard_mask].mean()
                    
                easy_mask = torch.tensor(batch_is_easy, device=self.device)
                if easy_mask.any():
                    targets = batch_baseline_diffs[easy_mask] - CONFIG["margin"]
                    currents = current_diffs[easy_mask]
                    penalty = torch.relu(targets - currents)
                    loss_easy = penalty.mean()
                    
                # No norm loss here, we use projection instead
                loss = loss_hard + CONFIG["lambda_easy"] * loss_easy
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([self.v_boost], 5.0)
                optimizer.step()
                
                # NORM CONSTRAINT PROJECTION
                with torch.no_grad():
                    norm = self.v_boost.norm()
                    if norm > CONFIG["R_TARGET"]:
                        self.v_boost.data = self.v_boost.data * (CONFIG["R_TARGET"] / norm)
                
                total_loss += loss.item()
                total_hard_loss += loss_hard.item()
                total_easy_loss += loss_easy.item()
                
            if (epoch + 1) % 5 == 0:
                norm_v = torch.norm(self.v_boost).item()
                print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {total_loss/num_batches:.4f} (H: {total_hard_loss/num_batches:.4f}, E: {total_easy_loss/num_batches:.4f}) | ||v||: {norm_v:.4f}")
                
        return self.v_boost.detach()

# ============================================================================
# 3. EVALUATION HOOKS
# ============================================================================

class EvaluationHook:
    def __init__(self, v_boost, virus_delta=None, mode="boost_only"):
        self.v_boost = v_boost
        self.virus_delta = virus_delta
        self.mode = mode
        
    def __call__(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            h = outputs[0]
        else:
            h = outputs
            
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
# 4. MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("PHASE 7D – CONSTRAINED TASK BOOST")
    print("=" * 80)
    
    device = CONFIG["device"]
    
    # 1. Load Data
    model, tokenizer, train_examples, test_examples = load_model_and_data(
        device, CONFIG["train_size"], CONFIG["test_size"]
    )
    
    # 2. Baseline
    print("\nComputing baseline metrics...")
    baseline_train = compute_metrics(model, tokenizer, train_examples)
    baseline_test = compute_metrics(model, tokenizer, test_examples)
    
    hard_train_idx, easy_train_idx = identify_subsets(baseline_train, CONFIG["hard_threshold"])
    hard_test_idx, easy_test_idx = identify_subsets(baseline_test, CONFIG["hard_threshold"])
    
    print(f"BASELINE Train: Acc={baseline_train['accuracy']:.4f}, Diff={baseline_train['logit_diff']:.4f}")
    print(f"BASELINE Test:  Acc={baseline_test['accuracy']:.4f}, Diff={baseline_test['logit_diff']:.4f}")
    
    # 3. Train
    trainer = ConstrainedBoostTrainer(model, tokenizer, CONFIG["layer"], device)
    v_boost = trainer.train(train_examples, hard_train_idx, easy_train_idx, baseline_train["per_example_diffs"])
    
    # Save Vector
    checkpoint_path = Path(f"checkpoints/learned_task_boost_layer{CONFIG['layer']}_R{int(CONFIG['R_TARGET'])}.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(v_boost, checkpoint_path)
    print(f"\nLearned vector saved to {checkpoint_path}")
    
    # 4. Evaluation
    print("\nEvaluating Constrained Boost...")
    
    def evaluate_mode(mode_name, hook_obj):
        handle = model.transformer.h[CONFIG["layer"]].register_forward_hook(hook_obj)
        metrics = compute_metrics(model, tokenizer, test_examples)
        handle.remove()
        
        hard_diffs = [metrics["per_example_diffs"][i] for i in hard_test_idx]
        hard_correct = [metrics["per_example_correct"][i] for i in hard_test_idx]
        hard_acc = sum(hard_correct)/len(hard_correct) if hard_correct else 0.0
        
        return {
            "test_acc": metrics["accuracy"],
            "test_diff": metrics["logit_diff"],
            "hard_acc": hard_acc,
            "hard_diff": np.mean(hard_diffs) if hard_diffs else 0.0
        }

    # Baseline stats for hard subset
    base_hard_diffs = [baseline_test["per_example_diffs"][i] for i in hard_test_idx]
    base_hard_correct = [baseline_test["per_example_correct"][i] for i in hard_test_idx]
    base_hard_acc = sum(base_hard_correct)/len(base_hard_correct) if base_hard_correct else 0.0
    
    results = {
        "baseline": {
            "test_acc": baseline_test["accuracy"],
            "test_diff": baseline_test["logit_diff"],
            "hard_acc": base_hard_acc,
            "hard_diff": np.mean(base_hard_diffs) if base_hard_diffs else 0.0
        }
    }
    
    # Boost Only
    boost_hook = EvaluationHook(v_boost, mode="boost_only")
    results["boost_only"] = evaluate_mode("boost_only", boost_hook)
    
    # Virus Interaction
    virus_path = Path(f"checkpoints/adversarial_delta_layer{CONFIG['layer']}.pt")
    if virus_path.exists():
        print("Evaluating Virus Interaction...")
        virus_delta = torch.load(virus_path, map_location=device)
        if isinstance(virus_delta, dict): virus_delta = virus_delta['delta']
        
        interaction_modes = ["attack_only", "attack_plus_boost"]
        
        for m in interaction_modes:
            hook = EvaluationHook(v_boost, virus_delta, mode=m)
            results[m] = evaluate_mode(m, hook)
            
    # Print Summary
    print(f"\n{'Mode':<18} | {'TestAcc':<8} | {'ΔTest':<6} | {'HardAcc':<8} | {'ΔHard':<6}")
    print("-" * 65)
    
    modes_to_print = ["baseline", "boost_only"]
    if virus_path.exists():
        modes_to_print.extend(["attack_only", "attack_plus_boost"])
        
    for m in modes_to_print:
        if m in results:
            res = results[m]
            delta_test = res["test_acc"] - results["baseline"]["test_acc"]
            delta_hard = res["hard_acc"] - results["baseline"]["hard_acc"]
            print(f"{m:<18} | {res['test_acc']:<8.4f} | {delta_test:+.4f} | {res['hard_acc']:<8.4f} | {delta_hard:+.4f}")
            
    # 5. Save Results
    json_file = "phase7d_constrained_task_boost_results.json"
    def convert(o):
        if isinstance(o, np.float32): return float(o)
        return o
        
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {json_file}")
    
    # Markdown
    md_file = "PHASE7D_CONSTRAINED_TASK_BOOST.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Phase 7D: Constrained Task Boost Results\n\n")
        f.write(f"**Layer:** {CONFIG['layer']}\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        f.write("## Training Config\n")
        f.write(f"- **Norm Constraint (R_TARGET)**: {CONFIG['R_TARGET']}\n")
        f.write(f"- **Final Vector Norm**: {torch.norm(v_boost).item():.4f}\n\n")
        
        f.write("## Results\n\n")
        f.write("| Mode | Test Acc | Δ Test | Hard Acc | Δ Hard |\n")
        f.write("|---|---|---|---|---|\n")
        
        for m in modes_to_print:
            if m in results:
                res = results[m]
                dt = res["test_acc"] - results["baseline"]["test_acc"]
                dh = res["hard_acc"] - results["baseline"]["hard_acc"]
                f.write(f"| {m} | {res['test_acc']:.2%} | {dt:+.2%} | {res['hard_acc']:.2%} | {dh:+.2%} |\n")
                
        f.write("\n## Conclusion\n")
        
        boost_acc = results["boost_only"]["test_acc"]
        attack_boost_acc = results.get("attack_plus_boost", {}).get("test_acc", 0.0)
        
        if attack_boost_acc > 0.9:
            f.write(f"**SUCCESS**: Even with constrained norm ({CONFIG['R_TARGET']}), the boost neutralizes the virus ({attack_boost_acc:.2%}).\n")
        elif attack_boost_acc > 0.6:
            f.write(f"**PARTIAL**: Constrained boost offers partial protection ({attack_boost_acc:.2%}).\n")
        else:
            f.write(f"**FAILURE**: Constrained boost is insufficient ({attack_boost_acc:.2%}). Higher norm is required.\n")
            
    print(f"Report saved to {md_file}")

if __name__ == "__main__":
    main()
