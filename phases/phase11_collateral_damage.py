"""
PHASE 11 – COLLATERAL DAMAGE ASSESSMENT

Goal:
Measure the "collateral damage" of the static defence (Task Boost Layer 10)
on generic language modeling performance (WikiText-2).
Does the "steroid" that fixes IOI make the model hallucinate or degrade on normal text?

This script:
1. Loads GPT-2 Small.
2. Loads the Constrained Task Boost vector (R=25).
3. Loads a sample of WikiText-2.
4. Computes Perplexity (PPL) for Baseline vs Static Defence (Beta=3.0).

Usage:
    python phase11_collateral_damage.py
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ============================================================================
# 1. SETUP
# ============================================================================

CONFIG = {
    "layer": 10,
    "beta": 3.0,  # Fixed scale from Phase 8B
    "n_samples": 300,
    "max_length": 64,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def load_model_and_boost(device: str) -> Tuple[object, object, torch.Tensor]:
    """Load Model, Tokenizer, and Boost Vector."""
    print(f"Loading GPT-2 model on {device}...")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load Boost Vector
    boost_path = Path(f"checkpoints/learned_task_boost_layer{CONFIG['layer']}_R25.pt")
    if not boost_path.exists():
        raise FileNotFoundError(f"Boost vector not found at {boost_path}")
        
    print(f"Loading Boost Vector from {boost_path}...")
    v_boost = torch.load(boost_path, map_location=device)
    if v_boost.dim() > 1:
        v_boost = v_boost.squeeze()
        
    return model, tokenizer, v_boost

def load_wikitext_samples(tokenizer, n=300, max_length=64):
    """Load and preprocess WikiText-2 samples."""
    print("Loading WikiText-2 dataset...")
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Attempting to use 'wikitext-2-v1' instead...")
        dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")

    samples = []
    count = 0
    
    print(f"Selecting {n} samples...")
    for item in dataset:
        text = item['text'].strip()
        # Filter out headers and very short lines to get "real" text
        if len(text) > 50 and not text.startswith(" ="):
            # Tokenize
            enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
            if enc.input_ids.shape[1] > 10: # Ensure reasonable length
                samples.append(text)
                count += 1
                if count >= n:
                    break
                    
    print(f"Loaded {len(samples)} valid samples.")
    return samples

# ============================================================================
# 2. EVALUATION LOGIC
# ============================================================================

def compute_perplexity(model, tokenizer, texts, device, hook_fn=None, layer_idx=10):
    """
    Compute average NLL and Perplexity.
    If hook_fn is provided, it is registered at layer_idx.
    """
    handle = None
    if hook_fn is not None:
        handle = model.transformer.h[layer_idx].register_forward_hook(hook_fn)
        
    nlls = []
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=CONFIG["max_length"]).to(device)
            input_ids = inputs.input_ids
            target_ids = input_ids.clone()
            
            # GPT-2 calculates loss on shifted labels internally, 
            # but we can just pass labels=input_ids
            outputs = model(input_ids, labels=target_ids)
            
            # loss is the average NLL per token in the batch
            # We want to accumulate total NLL to average later properly?
            # Or just average the per-sequence NLLs.
            # Standard PPL is exp(sum(NLL) / total_tokens)
            
            # outputs.loss is mean NLL.
            # nll_sum = outputs.loss * num_tokens
            # But wait, GPT-2 loss ignores the first token prediction?
            # Actually, let's trust the model's loss calculation which is standard.
            
            # Number of tokens evaluated is seq_len - 1 (since it predicts next token)
            num_eval_tokens = input_ids.shape[1] - 1
            if num_eval_tokens <= 0: continue
            
            nll_sum = outputs.loss.item() * num_eval_tokens
            nlls.append(nll_sum)
            total_tokens += num_eval_tokens
            
    if handle:
        handle.remove()
        
    if total_tokens == 0:
        return 0.0, 0.0, 0
        
    avg_nll = sum(nlls) / total_tokens
    perplexity = np.exp(avg_nll)
    
    return avg_nll, perplexity, total_tokens

# ============================================================================
# 3. MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("PHASE 11: COLLATERAL DAMAGE ASSESSMENT")
    print("="*60 + "\n")
    
    device = CONFIG["device"]
    
    try:
        # 1. Load Resources
        model, tokenizer, v_boost = load_model_and_boost(device)
        samples = load_wikitext_samples(tokenizer, n=CONFIG["n_samples"], max_length=CONFIG["max_length"])
        
        # 2. Define Hook
        # Static Defence: Add beta * v_boost
        def static_defence_hook(module, input, output):
            # output[0] is hidden states [batch, seq, hidden]
            output[0][:] += CONFIG["beta"] * v_boost
            return output
            
        # 3. Evaluate Baseline
        print("\nEvaluating Baseline...")
        base_nll, base_ppl, n_tok = compute_perplexity(model, tokenizer, samples, device)
        
        # 4. Evaluate Static Defence
        print("Evaluating Static Defence (Beta=3.0)...")
        def_nll, def_ppl, _ = compute_perplexity(model, tokenizer, samples, device, 
                                                 hook_fn=static_defence_hook, 
                                                 layer_idx=CONFIG["layer"])
        
        # 5. Results
        print("\n" + "-"*60)
        print(f"{'Mode':<18} | {'N_tokens':<8} | {'NLL':<8} | {'Perplexity':<10}")
        print("-" * 60)
        print(f"{'baseline':<18} | {n_tok:<8} | {base_nll:<8.4f} | {base_ppl:<10.2f}")
        print(f"{'static_defence':<18} | {n_tok:<8} | {def_nll:<8.4f} | {def_ppl:<10.2f}")
        print("-" * 60)
        
        diff_nll = def_nll - base_nll
        diff_ppl = def_ppl - base_ppl
        print(f"\nImpact: NLL {diff_nll:+.4f}, PPL {diff_ppl:+.2f}")
        
        if diff_ppl > 100:
            print("⚠️  CRITICAL DAMAGE: Model is severely degraded on generic text.")
        elif diff_ppl > 20:
            print("⚠️  HIGH DAMAGE: Significant degradation.")
        elif diff_ppl > 5:
            print("⚠️  MODERATE DAMAGE: Noticeable degradation.")
        else:
            print("✅ LOW DAMAGE: Model preserves general capabilities.")
            
        # 6. Save
        results = {
            "layer": CONFIG["layer"],
            "beta": CONFIG["beta"],
            "n_samples": CONFIG["n_samples"],
            "baseline": {"nll": base_nll, "ppl": base_ppl},
            "static_defence": {"nll": def_nll, "ppl": def_ppl},
            "impact": {"delta_nll": diff_nll, "delta_ppl": diff_ppl}
        }
        
        json_file = "phase11_collateral_damage_results.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {json_file}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
