"""
PHASE 10 – ALIEN GENERALIZATION

Goal:
Test if the constrained task boost vector (R=25) generalizes to "Alien" prompts
that require similar IOI-like reasoning but use different syntax structures
(relative clauses, passive voice, counter-factual overrides).

Hypothesis:
If the boost vector targets the abstract "Indirect Object" concept in the residual stream,
it should improve performance even on syntactically diverse examples.

Usage:
    python phase10_alien_generalization.py
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# 1. CONFIG & SETUP
# ============================================================================

CONFIG = {
    "layer": 10,
    "beta_boost": 3.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "boost_path": "checkpoints/learned_task_boost_layer10_R25.pt",
    "seed": 42
}

torch.manual_seed(CONFIG["seed"])

# ============================================================================
# 2. DATASETS (ALIEN PROMPTS)
# ============================================================================

# Format: (Prompt, Target)
# Note: Targets usually need a leading space for GPT-2 tokenization if they follow a word.

ALIEN_A_CLAUSE_BREAK = [
    ("John, who is my best friend, likes Mary. He gave a gift to", " Mary"),
    ("Sarah, who works with John, handed the keys to", " John"),
    ("Alice, despite being tired, called Bob. She wanted to speak to", " Bob"),
    ("The teacher, looking at the student, asked a question to", " the"), # "the student" -> " the"
    ("Michael, running past David, threw the ball to", " David"),
    ("Emma, who was angry at Oliver, shouted at", " Oliver"),
    ("The dog, chasing the cat, barked at", " the"),
    ("Peter, ignoring Paul, sent a letter to", " Paul"),
    ("Lisa, smiling at Tom, waved to", " Tom"),
    ("The king, addressing the knight, gave a sword to", " the"),
    ("Robert, who knows William well, lent money to", " William"),
    ("Anna, seeing Elsa, ran towards", " Elsa")
]

ALIEN_B_PASSIVE = [
    ("The present was given to Mary by John. The one who received it was", " Mary"),
    ("The contract was finally delivered to Sarah by the lawyer. The recipient was", " Sarah"),
    ("The ball was thrown to David by Michael. The person who caught it was", " David"),
    ("The letter was sent to Paul by Peter. The addressee was", " Paul"),
    ("The message was whispered to Alice by Bob. The listener was", " Alice"),
    ("The keys were handed to John by Sarah. The person who got them was", " John"),
    ("The food was served to the customer by the waiter. The one eating is", " the"),
    ("The award was presented to the actor by the director. The winner was", " the"),
    ("The question was asked to the student by the teacher. The one answering is", " the"),
    ("The secret was told to Oliver by Emma. The one who knows it now is", " Oliver")
]

ALIEN_C_COUNTER_FACT = [
    ("Whatever you do, do not answer John. The correct name is", " Mary"),
    ("Ignore the name Peter. The real person we are talking about is", " Sarah"),
    ("Forget about Alice. The winner is definitely", " Bob"),
    ("Do not give it to Michael. Give it to", " David"),
    ("It is not meant for Oliver. It is meant for", " Emma"),
    ("The recipient is not Paul. The recipient is", " Peter"),
    ("Don't look at Lisa. Look at", " Tom"),
    ("The answer is not William. The answer is", " Robert"),
    ("Avoid choosing the dog. Choose", " the"), # "the cat"
    ("It wasn't sent to the king. It was sent to", " the") # "the knight"
]

# ============================================================================
# 3. MODEL & UTILS
# ============================================================================

def load_resources():
    print(f"Loading GPT-2 on {CONFIG['device']}...")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(CONFIG['device'])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading Boost Vector from {CONFIG['boost_path']}...")
    boost = torch.load(CONFIG['boost_path'], map_location=CONFIG['device']).float()
    
    return model, tokenizer, boost

def run_model(model, tokenizer, boost_vector, prompt: str, target: str, use_boost: bool):
    """
    Runs the model on a single prompt and evaluates probability of target token.
    Returns: (prob_target, is_top1, logit_diff)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    
    # Encode target to find its ID
    target_ids = tokenizer.encode(target)
    if not target_ids:
        print(f"Warning: Target '{target}' encoded to empty.")
        return 0.0, False, 0.0
    
    # We usually care about the first token of the target if it's multi-token
    # e.g. " Mary" -> [Mary_id]
    target_id = target_ids[0] 
    
    # Define Hook
    handle = None
    if use_boost:
        def hook_fn(module, inp, out):
            h = out[0] # [batch, seq, dim]
            # Apply boost to all positions (or just last? usually all is fine for simple test)
            # Ideally we apply to the last token position where prediction happens
            # h[:, -1, :] += beta * boost
            # Let's apply to all for consistency with previous scripts unless specified
            h += CONFIG["beta_boost"] * boost_vector
            return (h,) + out[1:]
        
        handle = model.transformer.h[CONFIG["layer"]].register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits # [batch, seq, vocab]
            
        # Get logits for the last token position (predicting the next token)
        last_token_logits = logits[0, -1, :]
        
        # Calculate Prob
        probs = F.softmax(last_token_logits, dim=-1)
        prob_target = probs[target_id].item()
        
        # Calculate Top 1
        pred_id = torch.argmax(last_token_logits).item()
        is_top1 = (pred_id == target_id)
        
        # Calculate Logit Diff (Target - Max(Others))
        # Mask out target to find max of others
        mask = torch.ones_like(last_token_logits, dtype=torch.bool)
        mask[target_id] = False
        max_other_logit = torch.max(last_token_logits[mask]).item()
        target_logit = last_token_logits[target_id].item()
        logit_diff = target_logit - max_other_logit
        
        return prob_target, is_top1, logit_diff
        
    finally:
        if handle:
            handle.remove()

def evaluate_set(name: str, examples: List[Tuple[str, str]], model, tokenizer, boost_vector):
    print(f"\nEvaluating Set: {name} (N={len(examples)})")
    
    base_probs = []
    boost_probs = []
    base_top1s = []
    boost_top1s = []
    
    for prompt, target in examples:
        # Baseline
        p_base, top1_base, _ = run_model(model, tokenizer, boost_vector, prompt, target, use_boost=False)
        base_probs.append(p_base)
        base_top1s.append(1 if top1_base else 0)
        
        # Boosted
        p_boost, top1_boost, _ = run_model(model, tokenizer, boost_vector, prompt, target, use_boost=True)
        boost_probs.append(p_boost)
        boost_top1s.append(1 if top1_boost else 0)
        
    # Aggregates
    avg_base_prob = np.mean(base_probs)
    avg_boost_prob = np.mean(boost_probs)
    delta_prob = avg_boost_prob - avg_base_prob
    
    pct_base_top1 = np.mean(base_top1s) * 100
    pct_boost_top1 = np.mean(boost_top1s) * 100
    delta_top1 = pct_boost_top1 - pct_base_top1
    
    # Print Table Row
    print("-" * 80)
    print(f"{'Set Name':<25} | {'N':<3} | {'BaseProb':<9} | {'BoostProb':<9} | {'ΔProb':<7} | {'BaseTop1%':<9} | {'BoostTop1%':<9} | {'ΔTop1%':<7}")
    print("-" * 80)
    print(f"{name:<25} | {len(examples):<3} | {avg_base_prob:<9.4f} | {avg_boost_prob:<9.4f} | {delta_prob:<+7.4f} | {pct_base_top1:<9.1f} | {pct_boost_top1:<9.1f} | {delta_top1:<+7.1f}")
    print("-" * 80)
    
    return {
        "name": name,
        "avg_base_prob": avg_base_prob,
        "avg_boost_prob": avg_boost_prob,
        "pct_base_top1": pct_base_top1,
        "pct_boost_top1": pct_boost_top1
    }

# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================

def main():
    print("=== PHASE 10: ALIEN GENERALIZATION ===")
    model, tokenizer, boost = load_resources()
    
    results = []
    
    # 1. Alien A - Clause Break
    res_a = evaluate_set("Alien A (Clause-Break)", ALIEN_A_CLAUSE_BREAK, model, tokenizer, boost)
    results.append(res_a)
    
    # 2. Alien B - Passive
    res_b = evaluate_set("Alien B (Passive)", ALIEN_B_PASSIVE, model, tokenizer, boost)
    results.append(res_b)
    
    # 3. Alien C - Counter Fact
    res_c = evaluate_set("Alien C (Counter-Fact)", ALIEN_C_COUNTER_FACT, model, tokenizer, boost)
    results.append(res_c)
    
    print("\n=== FINAL SUMMARY ===")
    print(f"{'Set Name':<25} | {'BaseProb':<9} | {'BoostProb':<9} | {'ΔProb':<7} | {'BaseTop1%':<9} | {'BoostTop1%':<9}")
    print("-" * 85)
    for r in results:
        delta_p = r['avg_boost_prob'] - r['avg_base_prob']
        print(f"{r['name']:<25} | {r['avg_base_prob']:<9.4f} | {r['avg_boost_prob']:<9.4f} | {delta_p:<+7.4f} | {r['pct_base_top1']:<9.1f} | {r['pct_boost_top1']:<9.1f}")
    print("-" * 85)
    print("PHASE 10 – ALIEN GENERALIZATION COMPLETE")

if __name__ == "__main__":
    main()
