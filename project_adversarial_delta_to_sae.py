"""
Phase 4B-B - Adversarial Delta Decomposition in SAE Feature Space

Project the learned adversarial steering vector (from Phase 4B) onto the SAE
feature basis to identify which features compose the "virus".

Research Questions:
1. How much of the adversarial delta lives in SAE feature space?
2. Which specific features contribute most to the adversarial effect?
3. Can top-K features reconstruct the adversarial effect?
4. Are Phase 3 discovered features related to the adversarial delta?

Method:
1. Load adversarial delta (768-dim residual vector)
2. Load SAE decoder directions (6,144 feature basis)
3. Optimize alpha ∈ R^6144 such that: delta ≈ SAE.decoder(alpha) + pre_bias
4. Measure projection ratio: ||delta_hat|| / ||delta||
5. Identify top-K features (by |alpha|) that compose the virus
6. Test adversarial effect of top-K reconstruction vs full delta
"""

import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from transformers import AutoModelForCausalLM, AutoTokenizer
from neurotrace.datasets import IOIDatasetGenerator
from neurotrace.control import EnhancedSAEFeatureStore

print("=" * 80)
print("PHASE 4B-B - ADVERSARIAL DELTA DECOMPOSITION IN SAE SPACE")
print("=" * 80)
print()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print()

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "target_layer": 10,  # Must match Phase 4B
    "test_size": 500,

    # Decomposition optimization
    "num_epochs": 100,
    "learning_rate": 1e-2,
    "lambda_reg": 1e-3,
    "max_grad_norm": 5.0,

    # Top-K reconstruction
    "top_k_values": [10, 20, 50, 100, 200],
}

print("Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")
print()

target_layer = CONFIG['target_layer']

# ============================================================================
# LOAD ADVERSARIAL DELTA
# ============================================================================
print("=" * 80)
print("LOAD ADVERSARIAL DELTA")
print("=" * 80)
print()

delta_path = Path(f"checkpoints/adversarial_delta_layer{target_layer}.pt")
if not delta_path.exists():
    print(f"ERROR: Delta not found at {delta_path}")
    print(f"Run learn_adversarial_steering_vector.py first!")
    sys.exit(1)

delta = torch.load(delta_path).to(device)
delta_norm = delta.norm().item()

print(f"Loaded delta from: {delta_path}")
print(f"Delta shape: {delta.shape}")
print(f"Delta norm: {delta_norm:.3f}")
print()

# ============================================================================
# LOAD MODEL AND SAE
# ============================================================================
print("=" * 80)
print("LOAD MODEL AND SAE")
print("=" * 80)
print()

print("Loading GPT-2...")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model.eval()
print("Model loaded")
print()

print(f"Loading SAE for layer {target_layer}...")
feature_store = EnhancedSAEFeatureStore()

sae_path = Path(f"checkpoints/all_layers_sae/layer_{target_layer}/final.pt")
if not sae_path.exists():
    print(f"ERROR: SAE not found at {sae_path}")
    sys.exit(1)

feature_store.load_sae(str(sae_path), layer=target_layer, device=device)
sae = feature_store.saes[target_layer]

print(f"SAE loaded: {sae_path}")
print()

# ============================================================================
# EXTRACT SAE FEATURE DIRECTIONS
# ============================================================================
print("=" * 80)
print("EXTRACT SAE FEATURE DIRECTIONS")
print("=" * 80)
print()

# SAE decoder: maps feature activations → residual stream
# decoder: Linear(dict_size → hidden_dim) or has .weight attribute
decoder_weight = sae.decoder.weight  # Shape: [hidden_dim, dict_size] or [dict_size, hidden_dim]

print(f"Decoder weight shape: {decoder_weight.shape}")

# Determine correct orientation
hidden_dim = delta.shape[0]  # 768
dict_size = sae.dict_size  # 6144

if decoder_weight.shape[0] == hidden_dim:
    # Weight is [hidden_dim, dict_size] → transpose to [dict_size, hidden_dim]
    directions = decoder_weight.t()
else:
    # Weight is already [dict_size, hidden_dim]
    directions = decoder_weight

print(f"Directions shape: {directions.shape}")  # Should be [6144, 768]
print(f"Dictionary size: {dict_size}")
print(f"Hidden dimension: {hidden_dim}")
print()

# Check if directions are already normalized (decoder normalization during SAE training)
direction_norms = directions.norm(dim=1)
mean_norm = direction_norms.mean().item()
std_norm = direction_norms.std().item()

print(f"Direction norms: mean={mean_norm:.6f}, std={std_norm:.6f}")

if abs(mean_norm - 1.0) < 0.01 and std_norm < 0.01:
    print("✓ Directions already normalized (decoder was trained with normalization)")
    print("  → Alpha values directly comparable as feature importance")
else:
    print("⚠ Directions not normalized, normalizing now...")
    directions = torch.nn.functional.normalize(directions, dim=1)
    print("  Directions normalized to unit vectors")

print()

# ============================================================================
# OPTIMIZE ALPHA (FEATURE WEIGHTS)
# ============================================================================
print("=" * 80)
print("OPTIMIZE ALPHA (DECOMPOSITION)")
print("=" * 80)
print()

print("Finding optimal alpha such that: delta ≈ sum_i alpha_i * directions_i")
print()

# Initialize alpha
alpha = torch.zeros(dict_size, device=device, requires_grad=True)
optimizer = torch.optim.Adam([alpha], lr=CONFIG['learning_rate'])
lambda_reg = CONFIG['lambda_reg']

print(f"Optimizer: Adam(lr={CONFIG['learning_rate']})")
print(f"L2 regularization: λ={lambda_reg}")
print(f"Epochs: {CONFIG['num_epochs']}")
print()

# Track progress
history = {
    "epoch": [],
    "recon_loss": [],
    "total_loss": [],
    "alpha_norm": [],
    "delta_hat_norm": [],
    "projection_ratio": [],
}

import time
start_time = time.time()

for epoch in range(CONFIG['num_epochs']):
    optimizer.zero_grad()

    # Reconstruct delta from alpha
    delta_hat = torch.matmul(alpha, directions)  # [hidden_dim]

    # Loss: MSE + L2 regularization
    recon_loss = torch.nn.functional.mse_loss(delta_hat, delta)
    reg_loss = lambda_reg * (alpha.norm(p=2) ** 2)
    total_loss = recon_loss + reg_loss

    # Backward
    total_loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_([alpha], max_norm=CONFIG['max_grad_norm'])

    # Update
    optimizer.step()

    # Metrics
    alpha_norm = alpha.norm().item()
    delta_hat_norm = delta_hat.norm().item()
    projection_ratio = delta_hat_norm / delta_norm

    history["epoch"].append(epoch + 1)
    history["recon_loss"].append(recon_loss.item())
    history["total_loss"].append(total_loss.item())
    history["alpha_norm"].append(alpha_norm)
    history["delta_hat_norm"].append(delta_hat_norm)
    history["projection_ratio"].append(projection_ratio)

    # Log every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{CONFIG['num_epochs']}:  "
              f"Loss={total_loss.item():.6f}  "
              f"ReconLoss={recon_loss.item():.6f}  "
              f"||α||={alpha_norm:.3f}  "
              f"||δ_hat||={delta_hat_norm:.3f}  "
              f"Ratio={projection_ratio:.3%}")

elapsed_time = time.time() - start_time
print()
print(f"Optimization complete in {elapsed_time:.1f}s")
print()

# Final reconstruction
with torch.no_grad():
    delta_hat_final = torch.matmul(alpha, directions)
    final_projection_ratio = delta_hat_final.norm().item() / delta_norm
    final_mse = torch.nn.functional.mse_loss(delta_hat_final, delta).item()

print("Final Decomposition:")
print(f"  ||delta||:     {delta_norm:.3f}")
print(f"  ||delta_hat||: {delta_hat_final.norm().item():.3f}")
print(f"  Projection ratio: {final_projection_ratio:.3%}")
print(f"  MSE: {final_mse:.6f}")
print()

# ============================================================================
# IDENTIFY TOP-K FEATURES
# ============================================================================
print("=" * 80)
print("IDENTIFY TOP-K FEATURES (VIRUS GENES)")
print("=" * 80)
print()

# Sort features by |alpha| descending
alpha_np = alpha.detach().cpu().numpy()
feature_importance = [(i, abs(alpha_np[i])) for i in range(len(alpha_np))]
feature_importance.sort(key=lambda x: x[1], reverse=True)

print("Top 50 features by |alpha|:")
print()
for rank, (feat_idx, importance) in enumerate(feature_importance[:50], 1):
    alpha_val = alpha_np[feat_idx]
    print(f"  {rank:2d}. Feature {feat_idx:4d}:  alpha={alpha_val:+.4f}  |alpha|={importance:.4f}")
print()

# Save top features
top_features = [
    {
        "rank": rank,
        "feature_idx": int(feat_idx),
        "alpha": float(alpha_np[feat_idx]),
        "abs_alpha": float(importance),
        "layer": target_layer,
    }
    for rank, (feat_idx, importance) in enumerate(feature_importance, 1)
]

# ============================================================================
# TOP-K RECONSTRUCTION TEST
# ============================================================================
print("=" * 80)
print("TOP-K RECONSTRUCTION TEST")
print("=" * 80)
print()

print("Testing how many features needed to reconstruct adversarial effect...")
print()

# Load test dataset
print("Generating test dataset...")
test_generator = IOIDatasetGenerator(seed=99)
test_examples = test_generator.generate(num_examples=CONFIG['test_size'])

texts = [ex.text for ex in test_examples]
encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

test_input_ids = encoding["input_ids"].to(device)
test_attention_mask = encoding["attention_mask"].to(device)
test_target_positions = (test_attention_mask.sum(dim=1) - 1).tolist()

test_correct_ids = torch.tensor([
    tokenizer.encode(" " + ex.correct_answer, add_special_tokens=False)[0]
    for ex in test_examples
], device=device)

test_incorrect_ids = torch.tensor([
    tokenizer.encode(" " + ex.incorrect_answer, add_special_tokens=False)[0]
    for ex in test_examples
], device=device)

print(f"Test dataset: {len(test_examples)} examples")
print()

# Baseline (no steering)
print("Computing baseline...")

def compute_metrics(model, input_ids, attention_mask, target_positions, correct_ids, incorrect_ids, batch_size=50):
    """Compute logit diffs and accuracy."""
    all_logit_diffs = []

    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            batch_input = input_ids[i:i+batch_size]
            batch_mask = attention_mask[i:i+batch_size]

            outputs = model(input_ids=batch_input, attention_mask=batch_mask)
            batch_logits = outputs.logits

            for j in range(len(batch_logits)):
                idx = i + j
                pos = target_positions[idx]
                logit_correct = batch_logits[j, pos, correct_ids[idx]].item()
                logit_incorrect = batch_logits[j, pos, incorrect_ids[idx]].item()
                all_logit_diffs.append(logit_correct - logit_incorrect)

    mean_diff = np.mean(all_logit_diffs)
    accuracy = sum(1 for d in all_logit_diffs if d > 0) / len(all_logit_diffs)
    return mean_diff, accuracy

baseline_mean, baseline_acc = compute_metrics(
    model, test_input_ids, test_attention_mask, test_target_positions,
    test_correct_ids, test_incorrect_ids
)

print(f"Baseline: {baseline_mean:.3f} ({baseline_acc*100:.1f}%)")
print()

# Test full delta (from Phase 4B)
print("Testing full adversarial delta...")

def make_hook(delta_vec):
    """Create hook to inject delta."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            modified = hidden_states + delta_vec
            return (modified,) + output[1:]
        else:
            return output + delta_vec
    return hook

hook_fn = make_hook(delta)
handle = model.transformer.h[target_layer].register_forward_hook(hook_fn)

full_delta_mean, full_delta_acc = compute_metrics(
    model, test_input_ids, test_attention_mask, test_target_positions,
    test_correct_ids, test_incorrect_ids
)

handle.remove()

full_delta_effect = full_delta_mean - baseline_mean
full_delta_acc_change = (full_delta_acc - baseline_acc) * 100

print(f"Full delta: {full_delta_mean:.3f} ({full_delta_acc*100:.1f}%)")
print(f"  Effect: {full_delta_effect:+.3f}  Accuracy Δ: {full_delta_acc_change:+.1f}%")
print()

# Test top-K reconstructions
print("Testing top-K feature reconstructions...")
print()

topk_results = []

for k in CONFIG['top_k_values']:
    # Reconstruct delta from top-K features
    top_k_indices = [feat_idx for feat_idx, _ in feature_importance[:k]]
    top_k_alpha = torch.zeros_like(alpha)
    top_k_alpha[top_k_indices] = alpha[top_k_indices]

    with torch.no_grad():
        delta_topk = torch.matmul(top_k_alpha, directions)
        topk_norm = delta_topk.norm().item()
        topk_ratio = topk_norm / delta_norm

    # Test adversarial effect
    hook_fn = make_hook(delta_topk)
    handle = model.transformer.h[target_layer].register_forward_hook(hook_fn)

    topk_mean, topk_acc = compute_metrics(
        model, test_input_ids, test_attention_mask, test_target_positions,
        test_correct_ids, test_incorrect_ids
    )

    handle.remove()

    topk_effect = topk_mean - baseline_mean
    topk_acc_change = (topk_acc - baseline_acc) * 100
    effect_preservation = abs(topk_effect) / abs(full_delta_effect) if full_delta_effect != 0 else 0

    print(f"Top-{k:3d}:  ||δ||={topk_norm:.3f} ({topk_ratio:.1%})  "
          f"Effect={topk_effect:+.3f} ({effect_preservation:.1%} of full)  "
          f"Acc={topk_acc*100:.1f}% (Δ{topk_acc_change:+.1f}%)")

    topk_results.append({
        "k": k,
        "delta_norm": topk_norm,
        "norm_ratio": topk_ratio,
        "logit_diff_mean": topk_mean,
        "effect": topk_effect,
        "effect_preservation": effect_preservation,
        "accuracy": topk_acc,
        "accuracy_change": topk_acc_change,
    })

print()

# ============================================================================
# CROSS-REFERENCE WITH PHASE 3 FEATURES
# ============================================================================
print("=" * 80)
print("CROSS-REFERENCE WITH PHASE 3 DISCOVERED FEATURES")
print("=" * 80)
print()

# Load Phase 3 results
discovery_path = Path("feature_circuit_discovery.json")
if discovery_path.exists():
    with open(discovery_path) as f:
        discovery_results = json.load(f)

    discovered_features = discovery_results["discovered_features"]

    # Filter to this layer
    layer_features = [f for f in discovered_features if f["layer"] == target_layer]

    if layer_features:
        print(f"Phase 3 discovered {len(layer_features)} features in layer {target_layer}")
        print()

        # Check overlap with top virus features
        top_virus_indices = set(feat_idx for feat_idx, _ in feature_importance[:100])
        phase3_indices = set(f["feature_idx"] for f in layer_features)

        overlap = top_virus_indices & phase3_indices
        print(f"Overlap with top-100 virus features: {len(overlap)}/100 ({len(overlap)/100*100:.1f}%)")

        if overlap:
            print()
            print("Overlapping features (in both Phase 3 and virus top-100):")
            for feat_idx in sorted(overlap):
                # Find in Phase 3
                phase3_feat = next(f for f in layer_features if f["feature_idx"] == feat_idx)
                # Find in virus
                virus_rank = next(r for r, (fi, _) in enumerate(feature_importance, 1) if fi == feat_idx)
                virus_alpha = alpha_np[feat_idx]

                print(f"  Feature {feat_idx:4d}:  "
                      f"Phase3 corr={phase3_feat['correlation_with_success']:+.3f}  "
                      f"Virus rank={virus_rank:3d}  alpha={virus_alpha:+.4f}")
        else:
            print("  No overlap found (virus uses different features than Phase 3)")
    else:
        print(f"Phase 3 found no features in layer {target_layer}")
else:
    print("Phase 3 results not found (feature_circuit_discovery.json)")

print()

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("=" * 80)
print("SAVE RESULTS")
print("=" * 80)
print()

results = {
    "timestamp": datetime.now().isoformat(),
    "config": CONFIG,
    "decomposition": {
        "target_layer": target_layer,
        "delta_norm": delta_norm,
        "delta_hat_norm": delta_hat_final.norm().item(),
        "projection_ratio": final_projection_ratio,
        "mse": final_mse,
        "alpha_norm": alpha.norm().item(),
        "num_nonzero_features": int((alpha.abs() > 1e-6).sum().item()),
        "optimization_time_seconds": elapsed_time,
        "history": history,
    },
    "top_features": top_features[:200],  # Save top 200
    "reconstruction_tests": {
        "baseline": {
            "logit_diff_mean": baseline_mean,
            "accuracy": baseline_acc,
        },
        "full_delta": {
            "logit_diff_mean": full_delta_mean,
            "accuracy": full_delta_acc,
            "effect": full_delta_effect,
            "accuracy_change": full_delta_acc_change,
        },
        "topk_results": topk_results,
    },
}

output_path = Path(f"adversarial_delta_feature_decomposition_layer{target_layer}.json")
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved: {output_path}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ADVERSARIAL DELTA DECOMPOSITION COMPLETE")
print("=" * 80)
print()

print(f"Layer {target_layer} Analysis:")
print()

print("Decomposition Quality:")
print(f"  ||delta||:           {delta_norm:.3f}")
print(f"  ||delta_hat (all)||: {delta_hat_final.norm().item():.3f}")
print(f"  Projection ratio:    {final_projection_ratio:.3%}")
print(f"  MSE:                 {final_mse:.6f}")
print()

print("Feature Sparsity:")
print(f"  Total features:      {dict_size}")
print(f"  Non-zero alpha:      {(alpha.abs() > 1e-6).sum().item()}")
print(f"  Top-50 explain:      {topk_results[2]['norm_ratio']:.1%} of norm" if len(topk_results) > 2 else "N/A")
print()

print("Adversarial Effect Reconstruction:")
print(f"  Baseline:            {baseline_mean:.3f} ({baseline_acc*100:.1f}%)")
print(f"  Full delta:          {full_delta_mean:.3f} ({full_delta_acc*100:.1f}%)  Effect: {full_delta_effect:+.3f}")
print()

for result in topk_results:
    print(f"  Top-{result['k']:3d}:             "
          f"{result['logit_diff_mean']:.3f} ({result['accuracy']*100:.1f}%)  "
          f"Effect: {result['effect']:+.3f} ({result['effect_preservation']:.1%} of full)")
print()

if final_projection_ratio > 0.8:
    print("✅ HIGH projection ratio: Virus lives primarily in SAE feature space!")
elif final_projection_ratio > 0.5:
    print("⚠️  MODERATE projection ratio: Virus partially in SAE space")
else:
    print("❌ LOW projection ratio: Virus orthogonal to SAE features")

print()

# Find best top-K
if topk_results:
    best_topk = max(topk_results, key=lambda x: x['effect_preservation'])
    print(f"Best reconstruction: Top-{best_topk['k']} preserves {best_topk['effect_preservation']:.1%} of effect")
    print(f"  Using only {best_topk['k']}/6144 features ({best_topk['k']/6144*100:.1f}%)")

print()
print("Next steps:")
print("  - Analyze which Phase 3 features overlap with virus top-K")
print("  - Visualize feature importance distribution")
print("  - Test defensive steering (invert top-K features)")
print()
