"""
PHASE 14 – INTEGRATED DEFENCE SYSTEM (Layer 10)

Goal:
Implement and evaluate a complete 3-level defense system:
1. Domain Guard (Context Classifier): Is this IOI or General Text?
2. Damage Guard (Needs Boost Detector): Does the attack actually cause failure?
3. Active Defense (Task Boost): Inject the boost vector only when needed.

This script integrates all components (Virus, Boost, Detectors) to measure
the final performance on both IOI (under attack) and WikiText (collateral damage).

Usage:
    python phase14_integrated_defence.py
"""

import sys
import json
import torch
import torch.nn as nn
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
from neurotrace.datasets import IOIDatasetGenerator

# ============================================================================
# 1. CONFIG & MODELS
# ============================================================================

CONFIG = {
    "layer": 10,
    "beta": 3.0,          # Boost scale
    "alpha": 1.0,         # Attack scale
    "ioi_samples": 500,
    "wiki_samples": 300,
    "wiki_max_len": 64,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

class SimpleMLP(nn.Module):
    """Simple MLP for detectors."""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def load_resources(device: str):
    """Load Model, Vectors, and Detectors."""
    print(f"Loading GPT-2 model on {device}...")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 1. Vectors
    print("Loading vectors...")
    virus_path = Path(f"checkpoints/adversarial_delta_layer{CONFIG['layer']}.pt")
    boost_path = Path(f"checkpoints/learned_task_boost_layer{CONFIG['layer']}_R25.pt")
    
    virus_data = torch.load(virus_path, map_location=device)
    v_virus = virus_data['delta'] if isinstance(virus_data, dict) else virus_data
    v_boost = torch.load(boost_path, map_location=device)
    if v_boost.dim() > 1: v_boost = v_boost.squeeze()
    
    # 2. Needs Boost Detector (Phase 9C)
    print("Loading Needs Boost Detector...")
    nb_path = Path(f"checkpoints/needs_boost_detector_layer{CONFIG['layer']}.pt")
    # The config file is likely in the root or checkpoints folder, let's check root first as per Phase 9C script
    nb_config_path = Path("needs_boost_detector_config.json")
    if not nb_config_path.exists():
         nb_config_path = Path("checkpoints/needs_boost_detector_config.json")
    
    with open(nb_config_path, "r") as f:
        nb_config = json.load(f)
        
    nb_detector = SimpleMLP(input_dim=nb_config["input_dim"], hidden_dim=nb_config["hidden_dim"]).to(device)
    nb_detector.load_state_dict(torch.load(nb_path, map_location=device))
    nb_detector.eval()
    
    # 3. Context Classifier (Phase 13)
    print("Loading Context Classifier...")
    ctx_path = Path(f"checkpoints/context_classifier_layer{CONFIG['layer']}.pt")
    # Assuming input dim is hidden_dim (768) for context classifier
    ctx_classifier = SimpleMLP(input_dim=768).to(device)
    ctx_classifier.load_state_dict(torch.load(ctx_path, map_location=device))
    ctx_classifier.eval()
    
    return model, tokenizer, v_virus, v_boost, nb_detector, ctx_classifier

def load_datasets(tokenizer):
    """Load IOI and WikiText samples."""
    print("Generating IOI dataset...")
    generator = IOIDatasetGenerator()
    ioi_examples = generator.generate(num_examples=CONFIG["ioi_samples"], ensure_diversity=True)
    
    print("Loading WikiText-2 samples...")
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    except:
        dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
        
    wiki_samples = []
    count = 0
    for item in dataset:
        text = item['text'].strip()
        if len(text) > 50 and not text.startswith(" ="):
            enc = tokenizer(text, truncation=True, max_length=CONFIG["wiki_max_len"], return_tensors="pt")
            if enc.input_ids.shape[1] > 10:
                wiki_samples.append(text)
                count += 1
                if count >= CONFIG["wiki_samples"]:
                    break
                    
    return ioi_examples, wiki_samples

# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

def get_clean_activation(model, input_ids, layer_idx):
    """Get clean activation at layer_idx for the last token."""
    activations = {}
    def hook(module, input, output):
        activations["act"] = output[0][:, -1, :].detach() # [batch, hidden]
    
    handle = model.transformer.h[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        model(input_ids)
    handle.remove()
    return activations["act"]

def extract_nb_features(model, input_ids, layer_idx, v_virus, alpha):
    """Extract features for Needs Boost Detector (Phase 9C logic)."""
    # Features expected by Phase 9C detector:
    # [logit_diff_clean, logit_diff_attacked, delta_logit_diff, proj_clean, proj_attacked]
    
    # We need correct/incorrect answers to compute logit diffs.
    # But here we are inside a generic evaluation loop where we might not have them easily accessible
    # if we just pass input_ids.
    # However, evaluate_ioi_integrated DOES have access to correct/incorrect IDs.
    # We should refactor this to take logits or compute them inside the loop.
    
    # Let's simplify: The detector was trained on specific features. We MUST provide them exactly.
    # If we can't compute logit diff (because we don't know the target), we can't use this detector.
    # BUT: The detector is used inside evaluate_ioi_integrated where we DO know the targets.
    # So we should pass the necessary info to this function.
    pass 

def compute_nb_features_from_logits(clean_logits, attack_logits, clean_act, attack_act, v_virus, c_id, i_id):
    """Compute features for Needs Boost Detector given pre-computed logits/acts."""
    
    # 1. Logit Diffs
    ld_clean = (clean_logits[0, c_id] - clean_logits[0, i_id]).item()
    ld_attacked = (attack_logits[0, c_id] - attack_logits[0, i_id]).item()
    delta_ld = ld_attacked - ld_clean
    
    # 2. Projections
    # Normalize virus
    virus_norm = torch.norm(v_virus)
    virus_unit = v_virus / (virus_norm + 1e-8)
    
    # Project last token activation
    # clean_act shape: [batch, hidden] -> [hidden] (batch=1)
    proj_clean = torch.dot(clean_act.squeeze(0), virus_unit).item()
    proj_attacked = torch.dot(attack_act.squeeze(0), virus_unit).item()
    
    features = torch.tensor([[
        ld_clean,
        ld_attacked,
        delta_ld,
        proj_clean,
        proj_attacked
    ]], device=clean_logits.device)
    
    return features

# ============================================================================
# 3. EVALUATION LOGIC
# ============================================================================

def evaluate_ioi_integrated(model, tokenizer, examples, v_virus, v_boost, nb_detector, ctx_classifier):
    """Evaluate IOI performance under attack with integrated defense."""
    
    modes = ["baseline", "no_defence", "static_defence", "gated_defence_v3"]
    results = {m: {"correct": 0, "hard_correct": 0, "total": 0, "hard_total": 0, "gate_activations": 0} for m in modes}
    
    prompts = [ex.text for ex in examples]
    correct_answers = [ex.correct_answer for ex in examples]
    incorrect_answers = [ex.incorrect_answer for ex in examples]
    
    # Pre-calculate hard examples based on baseline
    # For simplicity in this integrated script, we'll define "hard" dynamically per batch
    # or just track it. Let's track it.
    
    batch_size = 1
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_corr = correct_answers[i:i+batch_size]
        batch_incorr = incorrect_answers[i:i+batch_size]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        input_ids = inputs.input_ids
        
        # 1. Baseline (Clean)
        with torch.no_grad():
            base_out = model(**inputs)
        base_logits = base_out.logits[:, -1, :]
        
        c_id = tokenizer.encode(" " + batch_corr[0])[0]
        i_id = tokenizer.encode(" " + batch_incorr[0])[0]
        
        base_diff = (base_logits[0, c_id] - base_logits[0, i_id]).item()
        is_hard = base_diff < 1.5
        
        # Update Baseline Stats
        if base_diff > 0:
            results["baseline"]["correct"] += 1
            if is_hard: results["baseline"]["hard_correct"] += 1
            
        results["baseline"]["total"] += 1
        if is_hard: results["baseline"]["hard_total"] += 1
        
        # 2. No Defence (Attack Only)
        def attack_hook(module, input, output):
            output[0][:, -1, :] += CONFIG["alpha"] * v_virus
            return output
            
        h = model.transformer.h[CONFIG["layer"]].register_forward_hook(attack_hook)
        with torch.no_grad():
            att_out = model(**inputs)
        h.remove()
        
        att_logits = att_out.logits[:, -1, :]
        att_diff = (att_logits[0, c_id] - att_logits[0, i_id]).item()
        
        if att_diff > 0:
            results["no_defence"]["correct"] += 1
            if is_hard: results["no_defence"]["hard_correct"] += 1
        results["no_defence"]["total"] += 1
        if is_hard: results["no_defence"]["hard_total"] += 1
        
        # 3. Static Defence (Attack + Boost)
        def static_hook(module, input, output):
            output[0][:, -1, :] += CONFIG["alpha"] * v_virus + CONFIG["beta"] * v_boost
            return output
            
        h = model.transformer.h[CONFIG["layer"]].register_forward_hook(static_hook)
        with torch.no_grad():
            stat_out = model(**inputs)
        h.remove()
        
        stat_logits = stat_out.logits[:, -1, :]
        stat_diff = (stat_logits[0, c_id] - stat_logits[0, i_id]).item()
        
        if stat_diff > 0:
            results["static_defence"]["correct"] += 1
            if is_hard: results["static_defence"]["hard_correct"] += 1
        results["static_defence"]["total"] += 1
        if is_hard: results["static_defence"]["hard_total"] += 1
        results["static_defence"]["gate_activations"] += 1 # Always active
        
        # 4. Gated Defence V3 (Integrated)
        # Step A: Domain Guard
        clean_act = get_clean_activation(model, input_ids, CONFIG["layer"])
        is_ioi = ctx_classifier(clean_act).item() > 0.5
        
        final_logits = None
        gate_active = False
        
        if not is_ioi:
            # Not IOI -> No Attack, No Boost (Ideal world assumption: attack only targets IOI)
            # Or if we assume attack is always present but we only defend if IOI?
            # Let's assume the attack is present but we only care to defend if it's IOI context.
            # If it's not IOI, we shouldn't boost.
            # But the attack might still be there.
            # Let's simulate: Attack is present.
            # If not IOI, we do nothing (so attack applies).
            # Wait, if it's not IOI, the "correct answer" is undefined.
            # But here we are evaluating ON IOI DATASET.
            # So is_ioi SHOULD be True. If False, it's a False Negative of Domain Guard.
            # If FN, we don't boost -> we fall back to No Defence (Attack Only).
            
            # Fallback to No Defence logits
            final_logits = att_logits
        else:
            # Is IOI -> Check Damage Guard
            # Extract features under attack
            # We need attacked activation for projection
            # We already ran attack pass (att_out), but didn't save activation.
            # Let's re-run or capture it earlier.
            # Optimization: Capture it during "No Defence" pass.
            
            # Let's just re-run quickly for clarity or assume we captured it.
            # To be safe and clean, let's re-run the attack hook to get activation.
            attack_acts = {}
            def get_att_act(module, input, output):
                output[0][:, -1, :] += CONFIG["alpha"] * v_virus
                attack_acts["act"] = output[0][:, -1, :].detach()
                return output
            h = model.transformer.h[CONFIG["layer"]].register_forward_hook(get_att_act)
            with torch.no_grad():
                model(**inputs)
            h.remove()
            att_act = attack_acts["act"]
            
            nb_feats = compute_nb_features_from_logits(base_logits, att_logits, clean_act, att_act, v_virus, c_id, i_id)
            needs_boost = nb_detector(nb_feats).item() > 0.5
            
            if needs_boost:
                # Apply Boost + Attack
                final_logits = stat_logits
                gate_active = True
            else:
                # Don't Boost -> Attack Only
                final_logits = att_logits
                
        gated_diff = (final_logits[0, c_id] - final_logits[0, i_id]).item()
        
        if gated_diff > 0:
            results["gated_defence_v3"]["correct"] += 1
            if is_hard: results["gated_defence_v3"]["hard_correct"] += 1
        results["gated_defence_v3"]["total"] += 1
        if is_hard: results["gated_defence_v3"]["hard_total"] += 1
        if gate_active: results["gated_defence_v3"]["gate_activations"] += 1
        
    return results

def evaluate_wiki_integrated(model, tokenizer, samples, v_boost, ctx_classifier):
    """Evaluate WikiText collateral damage with integrated defense."""
    
    modes = ["baseline", "static_defence", "domain_guarded"]
    results = {m: {"nll_sum": 0.0, "tokens": 0, "gate_activations": 0} for m in modes}
    
    for text in samples:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=CONFIG["wiki_max_len"]).to(model.device)
        input_ids = inputs.input_ids
        num_tokens = input_ids.shape[1] - 1
        if num_tokens <= 0: continue
        
        # 1. Baseline
        with torch.no_grad():
            base_out = model(input_ids, labels=input_ids)
        results["baseline"]["nll_sum"] += base_out.loss.item() * num_tokens
        results["baseline"]["tokens"] += num_tokens
        
        # 2. Static Defence (Boost Only)
        def boost_hook(module, input, output):
            output[0][:, :, :] += CONFIG["beta"] * v_boost # Apply to all tokens
            return output
            
        h = model.transformer.h[CONFIG["layer"]].register_forward_hook(boost_hook)
        with torch.no_grad():
            stat_out = model(input_ids, labels=input_ids)
        h.remove()
        
        results["static_defence"]["nll_sum"] += stat_out.loss.item() * num_tokens
        results["static_defence"]["tokens"] += num_tokens
        results["static_defence"]["gate_activations"] += 1
        
        # 3. Domain Guarded
        # Check context on the LAST token (approximation for sequence)
        # Ideally we check per token, but for PPL we usually process batch.
        # Let's check the last token's activation to decide for the whole sequence
        # (since we are processing short segments).
        clean_act = get_clean_activation(model, input_ids, CONFIG["layer"])
        is_ioi = ctx_classifier(clean_act).item() > 0.5
        
        if is_ioi:
            # False Positive! Domain Guard thinks this is IOI.
            # We apply the boost (assuming worst case: we try to defend).
            # Note: In full system we would also check Needs Boost, but here we assume
            # if Domain Guard fails, we might trigger defense.
            # Let's be conservative: if Domain Guard says IOI, we apply Static Defence logic
            # (or we could check Needs Boost, but Needs Boost is trained on IOI distribution).
            # Let's assume if Domain Guard says IOI, we enter "Defense Mode".
            # To measure "Collateral Damage of Defense", we apply the boost.
            current_loss = stat_out.loss.item()
            results["domain_guarded"]["gate_activations"] += 1
        else:
            # Correctly identified as General Text -> No Boost
            current_loss = base_out.loss.item()
            
        results["domain_guarded"]["nll_sum"] += current_loss * num_tokens
        results["domain_guarded"]["tokens"] += num_tokens
        
    return results

# ============================================================================
# 4. MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("PHASE 14: INTEGRATED DEFENCE SYSTEM")
    print("="*60 + "\n")
    
    device = CONFIG["device"]
    
    # 1. Load
    model, tokenizer, v_virus, v_boost, nb_detector, ctx_classifier = load_resources(device)
    ioi_ex, wiki_ex = load_datasets(tokenizer)
    
    # 2. Evaluate IOI
    print("\nEvaluating IOI Performance (Under Attack)...")
    ioi_results = evaluate_ioi_integrated(model, tokenizer, ioi_ex, v_virus, v_boost, nb_detector, ctx_classifier)
    
    print(f"{'Mode':<18} | {'TestAcc':<8} | {'HardAcc':<8} | {'GateRate':<8}")
    print("-" * 50)
    for m, res in ioi_results.items():
        acc = res["correct"] / res["total"]
        hard_acc = res["hard_correct"] / res["hard_total"] if res["hard_total"] > 0 else 0.0
        gate_rate = res["gate_activations"] / res["total"]
        print(f"{m:<18} | {acc:<8.2%} | {hard_acc:<8.2%} | {gate_rate:<8.2%}")
        
    # 3. Evaluate WikiText
    print("\nEvaluating WikiText Collateral Damage...")
    wiki_results = evaluate_wiki_integrated(model, tokenizer, wiki_ex, v_boost, ctx_classifier)
    
    print(f"{'Mode':<18} | {'NLL':<8} | {'PPL':<8} | {'FP Rate':<8}")
    print("-" * 50)
    for m, res in wiki_results.items():
        nll = res["nll_sum"] / res["tokens"]
        ppl = np.exp(nll)
        fp_rate = res["gate_activations"] / len(wiki_ex) if m == "domain_guarded" else (1.0 if m == "static_defence" else 0.0)
        print(f"{m:<18} | {nll:<8.4f} | {ppl:<8.2f} | {fp_rate:<8.2%}")
        
    # 4. Save
    final_results = {
        "ioi": ioi_results,
        "wiki": wiki_results
    }
    json_file = "phase14_integrated_defence_results.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to {json_file}")

if __name__ == "__main__":
    main()
