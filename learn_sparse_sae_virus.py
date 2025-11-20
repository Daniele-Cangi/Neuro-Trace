"""
Phase 5B - Sparse SAE Virus

Learn adversarial steering vectors DIRECTLY in SAE feature space (α coefficients)
with L1 regularization to enforce sparsity. Trace the sparsity-performance curve:
#features active → accuracy drop.

Research Question: Can we achieve -60% drop with <100 interpretable features?

Method:
1. Optimize α ∈ ℝⁿ (SAE coefficients) instead of δ ∈ ℝ⁷⁶⁸
2. Construct delta = α @ W_dec (decoder weight matrix)
3. Add L1 regularization to enforce sparsity: loss = logit_diff + λ₁||α||₁ + λ₂||α||₂²
4. Sweep λ₁ values to explore sparsity-performance tradeoff
5. Answer: What's the minimal feature set for strong adversarial control?

This tests if sparse, interpretable feature combinations can achieve
the same adversarial power as dense residual vectors (Phase 4B).
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

print("=" * 80)
print("PHASE 5B - SPARSE SAE VIRUS")
print("=" * 80)
print("Learning adversarial vectors in SAE feature space with L1 sparsity")
print()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print()

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "target_layer": 10,  # Same as Phase 4B
    "dataset_size": 2000,
    "borderline_threshold": 1.5,
    "test_size": 500,

    # Optimization
    "learning_rate": 1e-2,
    "lambda_l2": 1e-4,  # Small L2 for stability
    "num_epochs": 20,
    "batch_size": 32,
    "max_grad_norm": 5.0,

    # L1 sweep (sparsity control)
    "lambda_l1_values": [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],

    # Sparsity threshold
    "active_threshold": 1e-3,
}

print("Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")
print()

TARGET_LAYER = CONFIG['target_layer']

# ============================================================================
# LOAD MODEL
# ============================================================================
print("=" * 80)
print("LOAD MODEL")
print("=" * 80)
print()

print("Loading GPT-2...")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model.eval()
print("Model loaded")
print()

hidden_dim = model.config.n_embd  # 768
print(f"Hidden dimension: {hidden_dim}")
print()

# ============================================================================
# LOAD SAE FOR TARGET LAYER
# ============================================================================
print("=" * 80)
print(f"LOAD SAE (Layer {TARGET_LAYER})")
print("=" * 80)
print()

sae_path = Path(f"checkpoints/all_layers_sae/layer_{TARGET_LAYER}/final.pt")
if not sae_path.exists():
    print(f"ERROR: SAE not found at {sae_path}")
    print("Run train_atlas_simple.py first to train SAEs for all layers")
    exit(1)

print(f"Loading SAE from {sae_path}...")
sae_checkpoint = torch.load(sae_path, map_location=device, weights_only=False)

# Extract decoder weight matrix
if 'decoder.weight' in sae_checkpoint:
    decoder_weight = sae_checkpoint['decoder.weight']  # [n_features, hidden_dim]
elif 'model_state_dict' in sae_checkpoint and 'decoder.weight' in sae_checkpoint['model_state_dict']:
    decoder_weight = sae_checkpoint['model_state_dict']['decoder.weight']
else:
    print("ERROR: Could not find decoder.weight in checkpoint")
    print(f"Available keys: {sae_checkpoint.keys()}")
    exit(1)

decoder_weight = decoder_weight.to(device)

# Ensure correct shape: [n_features, hidden_dim]
if decoder_weight.shape[1] != hidden_dim:
    # Weight is [hidden_dim, n_features], need to transpose
    decoder_weight = decoder_weight.T

n_features = decoder_weight.shape[0]

print(f"SAE loaded successfully")
print(f"  Decoder weight shape: {decoder_weight.shape}")
print(f"  Number of features: {n_features}")
print(f"  Hidden dimension: {decoder_weight.shape[1]}")
print()

# Check decoder normalization
direction_norms = decoder_weight.norm(dim=1)
mean_norm = direction_norms.mean().item()
std_norm = direction_norms.std().item()
print(f"Decoder normalization check:")
print(f"  Mean norm: {mean_norm:.6f}")
print(f"  Std norm: {std_norm:.6f}")
if abs(mean_norm - 1.0) < 0.01 and std_norm < 0.01:
    print("  ✓ Decoder normalized (α values directly interpretable)")
else:
    print("  ⚠ Decoder not normalized")
print()

# ============================================================================
# GENERATE DATASETS (Same as Phase 4B)
# ============================================================================
print("=" * 80)
print("GENERATE DATASETS")
print("=" * 80)
print()

print(f"Generating {CONFIG['dataset_size']} training IOI examples...")
generator = IOIDatasetGenerator(seed=42)
train_examples = generator.generate(num_examples=CONFIG['dataset_size'])

print(f"Generating {CONFIG['test_size']} test IOI examples...")
test_generator = IOIDatasetGenerator(seed=99)
test_examples = test_generator.generate(num_examples=CONFIG['test_size'])

print(f"Generated {len(train_examples)} train + {len(test_examples)} test examples")
print()

# ============================================================================
# TOKENIZE DATASETS
# ============================================================================

def tokenize_dataset(examples, tokenizer, device):
    """Tokenize IOI examples and extract metadata."""
    texts = [ex.text for ex in examples]
    encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    target_positions = (attention_mask.sum(dim=1) - 1).tolist()

    correct_ids = torch.tensor([
        tokenizer.encode(" " + ex.correct_answer, add_special_tokens=False)[0]
        for ex in examples
    ], device=device)

    incorrect_ids = torch.tensor([
        tokenizer.encode(" " + ex.incorrect_answer, add_special_tokens=False)[0]
        for ex in examples
    ], device=device)

    return input_ids, attention_mask, target_positions, correct_ids, incorrect_ids

print("Tokenizing datasets...")
train_input_ids, train_attention_mask, train_target_positions, train_correct_ids, train_incorrect_ids = \
    tokenize_dataset(train_examples, tokenizer, device)

test_input_ids, test_attention_mask, test_target_positions, test_correct_ids, test_incorrect_ids = \
    tokenize_dataset(test_examples, tokenizer, device)

print(f"Train shape: {train_input_ids.shape}")
print(f"Test shape: {test_input_ids.shape}")
print()

# ============================================================================
# BASELINE EVALUATION
# ============================================================================
print("=" * 80)
print("BASELINE EVALUATION")
print("=" * 80)
print()

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
    return mean_diff, accuracy, all_logit_diffs

print("Computing baseline on train set...")
train_baseline_mean, train_baseline_acc, train_baseline_diffs = compute_metrics(
    model, train_input_ids, train_attention_mask, train_target_positions,
    train_correct_ids, train_incorrect_ids
)

print(f"Train Baseline: {train_baseline_mean:.3f} ({train_baseline_acc*100:.1f}%)")

print("Computing baseline on test set...")
test_baseline_mean, test_baseline_acc, test_baseline_diffs = compute_metrics(
    model, test_input_ids, test_attention_mask, test_target_positions,
    test_correct_ids, test_incorrect_ids
)

print(f"Test Baseline: {test_baseline_mean:.3f} ({test_baseline_acc*100:.1f}%)")
print()

# Identify borderline cases
borderline_threshold = CONFIG['borderline_threshold']
borderline_indices = [
    i for i, diff in enumerate(train_baseline_diffs)
    if 0.0 < diff < borderline_threshold
]

print(f"Borderline cases: {len(borderline_indices)} ({len(borderline_indices)/len(train_examples)*100:.1f}%)")
print()

# ============================================================================
# L1 SWEEP: TRAIN SPARSE SAE VIRUSES
# ============================================================================
print("=" * 80)
print("L1 SWEEP - SPARSE SAE VIRUS TRAINING")
print("=" * 80)
print()

def make_hook(delta_vec):
    """Create hook to inject delta into residual stream (same as Phase 4B)."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            modified = hidden_states + delta_vec
            return (modified,) + output[1:]
        else:
            return output + delta_vec
    return hook

# Results storage
all_results = []

import time
total_start_time = time.time()

lambda_l1_values = CONFIG['lambda_l1_values']

for lambda_l1 in lambda_l1_values:
    print("=" * 80)
    print(f"LAMBDA_L1 = {lambda_l1}")
    print("=" * 80)
    print()

    # Initialize alpha (SAE coefficients)
    alpha = torch.zeros(n_features, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([alpha], lr=CONFIG['learning_rate'])
    lambda_l2 = CONFIG['lambda_l2']

    print(f"Training sparse SAE virus with λ₁={lambda_l1}, λ₂={lambda_l2}...")

    training_start_time = time.time()

    # Training loop
    for epoch in range(CONFIG['num_epochs']):
        epoch_losses = []
        epoch_logit_diffs = []
        epoch_l1s = []
        epoch_l2s = []

        # Shuffle borderline indices
        indices = np.random.permutation(borderline_indices)

        for batch_start in range(0, len(indices), CONFIG['batch_size']):
            batch_idx = indices[batch_start:batch_start + CONFIG['batch_size']]

            # Get batch data
            batch_input = train_input_ids[batch_idx]
            batch_mask = train_attention_mask[batch_idx]
            batch_positions = [train_target_positions[i] for i in batch_idx]
            batch_correct = train_correct_ids[batch_idx]
            batch_incorrect = train_incorrect_ids[batch_idx]

            # Construct delta from alpha
            delta = torch.matmul(alpha, decoder_weight)  # [hidden_dim]

            # Register hook on target layer
            hook_fn = make_hook(delta)
            handle = model.transformer.h[TARGET_LAYER].register_forward_hook(hook_fn)

            # Forward pass
            outputs = model(input_ids=batch_input, attention_mask=batch_mask)
            logits = outputs.logits

            # Remove hook
            handle.remove()

            # Compute logit differences
            batch_logit_diffs = []
            for i in range(len(batch_input)):
                pos = batch_positions[i]
                logit_correct = logits[i, pos, batch_correct[i]]
                logit_incorrect = logits[i, pos, batch_incorrect[i]]
                batch_logit_diffs.append(logit_correct - logit_incorrect)

            batch_logit_diffs = torch.stack(batch_logit_diffs)
            mean_logit_diff = batch_logit_diffs.mean()

            # Loss with L1 and L2 regularization
            l1_reg = alpha.abs().sum()
            l2_reg = alpha.norm(p=2) ** 2
            loss = mean_logit_diff + lambda_l1 * l1_reg + lambda_l2 * l2_reg

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([alpha], max_norm=CONFIG['max_grad_norm'])
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_logit_diffs.append(mean_logit_diff.item())
            epoch_l1s.append(l1_reg.item())
            epoch_l2s.append(l2_reg.item())

        # Log every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            epoch_loss = np.mean(epoch_losses)
            epoch_mean_logit_diff = np.mean(epoch_logit_diffs)
            epoch_mean_l1 = np.mean(epoch_l1s)
            epoch_mean_l2 = np.mean(epoch_l2s)

            with torch.no_grad():
                delta = torch.matmul(alpha, decoder_weight)
                delta_norm = delta.norm().item()
                num_active = (alpha.abs() > CONFIG['active_threshold']).sum().item()

            print(f"  Epoch {epoch+1:2d}: Loss={epoch_loss:.4f}  LogitDiff={epoch_mean_logit_diff:+.3f}  "
                  f"||δ||={delta_norm:.3f}  ||α||₁={epoch_mean_l1:.1f}  #active={num_active}")

    training_time = time.time() - training_start_time
    print(f"  Training complete in {training_time:.1f}s")
    print()

    # Final metrics
    with torch.no_grad():
        delta = torch.matmul(alpha, decoder_weight)
        delta_norm = delta.norm().item()
        alpha_l1 = alpha.abs().sum().item()
        num_active = (alpha.abs() > CONFIG['active_threshold']).sum().item()

    # Save checkpoint
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoints_dir / f"sparse_sae_virus_layer{TARGET_LAYER}_l1{lambda_l1:.0e}.pt"
    torch.save({
        'alpha': alpha.detach().cpu(),
        'delta': delta.detach().cpu(),
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'decoder_weight': decoder_weight.cpu(),
        'config': CONFIG,
    }, checkpoint_path)
    print(f"  Saved: {checkpoint_path}")
    print()

    # Evaluate on train set
    hook_fn = make_hook(delta)
    handle = model.transformer.h[TARGET_LAYER].register_forward_hook(hook_fn)

    train_steered_mean, train_steered_acc, _ = compute_metrics(
        model, train_input_ids, train_attention_mask, train_target_positions,
        train_correct_ids, train_incorrect_ids
    )

    handle.remove()

    train_effect = train_steered_mean - train_baseline_mean
    train_acc_change = (train_steered_acc - train_baseline_acc) * 100

    # Evaluate on test set
    hook_fn = make_hook(delta)
    handle = model.transformer.h[TARGET_LAYER].register_forward_hook(hook_fn)

    test_steered_mean, test_steered_acc, _ = compute_metrics(
        model, test_input_ids, test_attention_mask, test_target_positions,
        test_correct_ids, test_incorrect_ids
    )

    handle.remove()

    test_effect = test_steered_mean - test_baseline_mean
    test_acc_change = (test_steered_acc - test_baseline_acc) * 100

    # Store results
    result = {
        "lambda_l1": lambda_l1,
        "lambda_l2": lambda_l2,
        "delta_norm": delta_norm,
        "alpha_l1": alpha_l1,
        "num_active": num_active,
        "training_time_seconds": training_time,
        "train": {
            "baseline_logit_diff": train_baseline_mean,
            "baseline_accuracy": train_baseline_acc,
            "steered_logit_diff": train_steered_mean,
            "steered_accuracy": train_steered_acc,
            "effect": train_effect,
            "accuracy_change": train_acc_change,
        },
        "test": {
            "baseline_logit_diff": test_baseline_mean,
            "baseline_accuracy": test_baseline_acc,
            "steered_logit_diff": test_steered_mean,
            "steered_accuracy": test_steered_acc,
            "effect": test_effect,
            "accuracy_change": test_acc_change,
        },
    }

    all_results.append(result)

    print(f"  Results:")
    print(f"    Delta norm: ||δ|| = {delta_norm:.3f}")
    print(f"    Alpha L1: ||α||₁ = {alpha_l1:.1f}")
    print(f"    Active features: {num_active} ({num_active/n_features*100:.1f}%)")
    print(f"    Train: {train_baseline_mean:.3f} → {train_steered_mean:.3f} (Effect: {train_effect:+.3f}, Acc Δ: {train_acc_change:+.1f}%)")
    print(f"    Test:  {test_baseline_mean:.3f} → {test_steered_mean:.3f} (Effect: {test_effect:+.3f}, Acc Δ: {test_acc_change:+.1f}%)")
    print()

total_elapsed_time = time.time() - total_start_time
print(f"Total L1 sweep time: {total_elapsed_time:.1f}s ({total_elapsed_time/60:.1f} min)")
print()

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("=" * 80)
print("SAVE RESULTS")
print("=" * 80)
print()

output = {
    "timestamp": datetime.now().isoformat(),
    "config": CONFIG,
    "baseline": {
        "train": {
            "logit_diff": train_baseline_mean,
            "accuracy": train_baseline_acc,
        },
        "test": {
            "logit_diff": test_baseline_mean,
            "accuracy": test_baseline_acc,
        },
    },
    "results": all_results,
    "total_time_seconds": total_elapsed_time,
}

output_path = Path("phase5b_sparse_sae_virus_results.json")
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Results saved: {output_path}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("SPARSE SAE VIRUS SWEEP COMPLETE")
print("=" * 80)
print()

print("Sparsity-Performance Tradeoff:")
print()
print(f"{'λ₁':>8s}  {'#Active':>8s}  {'||δ||':>8s}  {'Test Δacc':>10s}  {'Test Δdiff':>11s}")
print("-" * 60)
for result in all_results:
    lambda_l1 = result['lambda_l1']
    num_active = result['num_active']
    delta_norm = result['delta_norm']
    test_acc_change = result['test']['accuracy_change']
    test_effect = result['test']['effect']

    print(f"{lambda_l1:>8.0e}  {num_active:>8d}  {delta_norm:>8.3f}  {test_acc_change:>+10.1f}%  {test_effect:>+11.3f}")

print()

# Find best sparse result (most negative accuracy change with fewest features)
sparse_results = [r for r in all_results if r['num_active'] > 0]
if sparse_results:
    best_sparse = min(sparse_results, key=lambda x: x['test']['accuracy_change'])
    print(f"Best sparse result:")
    print(f"  λ₁ = {best_sparse['lambda_l1']:.0e}")
    print(f"  Active features: {best_sparse['num_active']} ({best_sparse['num_active']/n_features*100:.1f}%)")
    print(f"  Test accuracy drop: {best_sparse['test']['accuracy_change']:+.1f}%")
    print(f"  Test effect: {best_sparse['test']['effect']:+.3f}")
    print()

# Check if any setting achieves strong effect with <100 features
strong_sparse = [r for r in all_results if r['num_active'] < 100 and r['test']['accuracy_change'] <= -40]
if strong_sparse:
    print(f"✓ Found {len(strong_sparse)} setting(s) with <100 features and ≤-40% accuracy drop:")
    for r in strong_sparse:
        print(f"  λ₁={r['lambda_l1']:.0e}: {r['num_active']} features → {r['test']['accuracy_change']:+.1f}%")
else:
    print("✗ No setting achieved ≤-40% drop with <100 features")

print()

print("Next steps:")
print("  - Generate PHASE5B_SPARSE_SAE_VIRUS.md report")
print("  - Analyze top features in best sparse model")
print("  - Compare with Phase 4B-B decomposition")
print()
