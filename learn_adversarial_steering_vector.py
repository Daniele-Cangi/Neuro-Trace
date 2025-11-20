"""
Phase 4B - Adversarial Steering Vector Learning

Learn a residual stream delta vector that adversarially reduces IOI logit difference.
This is a gradient-based approach to find universal steering directions, independent
of SAE feature space.

Method:
1. Select borderline IOI examples (0 < logit_diff < 1.5)
2. Optimize delta ∈ R^768 to minimize: loss = mean_logit_diff + λ||delta||²
3. Inject delta into residual stream at target layer via hook
4. Validate on full dataset and separate test set

This tests whether IOI can be adversarially attacked via learned residual perturbations,
revealing causal directions in the residual stream (not feature space).
"""

import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from transformers import AutoModelForCausalLM, AutoTokenizer
from neurotrace.datasets import IOIDatasetGenerator

print("=" * 80)
print("PHASE 4B - ADVERSARIAL STEERING VECTOR LEARNING")
print("=" * 80)
print()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print()

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "dataset_size": 2000,
    "borderline_threshold": 1.5,
    "test_size": 500,

    # Optimization
    "target_layer": 10,  # Layer to inject delta (0-11)
    "learning_rate": 1e-2,
    "lambda_reg": 1e-3,  # L2 regularization on delta
    "num_epochs": 20,
    "batch_size": 32,

    # Gradient clipping
    "max_grad_norm": 5.0,
}

print("Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")
print()

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
model.eval()  # Keep in eval mode (no dropout)
print("Model loaded")
print()

hidden_dim = model.config.n_embd  # 768 for GPT-2
print(f"Hidden dimension: {hidden_dim}")
print()

# ============================================================================
# GENERATE DATASETS
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

print("Tokenizing train dataset...")
train_input_ids, train_attention_mask, train_target_positions, train_correct_ids, train_incorrect_ids = \
    tokenize_dataset(train_examples, tokenizer, device)

print("Tokenizing test dataset...")
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

def compute_logit_diffs(model, input_ids, attention_mask, target_positions,
                        correct_ids, incorrect_ids, batch_size=50):
    """Compute logit differences for a dataset."""
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

    return all_logit_diffs

print("Computing baseline on train set...")
train_baseline_diffs = compute_logit_diffs(
    model, train_input_ids, train_attention_mask, train_target_positions,
    train_correct_ids, train_incorrect_ids
)

train_baseline_mean = np.mean(train_baseline_diffs)
train_baseline_accuracy = sum(1 for d in train_baseline_diffs if d > 0) / len(train_baseline_diffs)

print(f"Train Baseline:")
print(f"  Mean logit diff: {train_baseline_mean:.3f}")
print(f"  Accuracy: {train_baseline_accuracy*100:.1f}%")
print()

print("Computing baseline on test set...")
test_baseline_diffs = compute_logit_diffs(
    model, test_input_ids, test_attention_mask, test_target_positions,
    test_correct_ids, test_incorrect_ids
)

test_baseline_mean = np.mean(test_baseline_diffs)
test_baseline_accuracy = sum(1 for d in test_baseline_diffs if d > 0) / len(test_baseline_diffs)

print(f"Test Baseline:")
print(f"  Mean logit diff: {test_baseline_mean:.3f}")
print(f"  Accuracy: {test_baseline_accuracy*100:.1f}%")
print()

# ============================================================================
# IDENTIFY BORDERLINE CASES
# ============================================================================
print("=" * 80)
print("IDENTIFY BORDERLINE CASES")
print("=" * 80)
print()

borderline_threshold = CONFIG['borderline_threshold']
borderline_indices = [
    i for i, diff in enumerate(train_baseline_diffs)
    if 0.0 < diff < borderline_threshold
]

print(f"Borderline cases (0 < logit_diff < {borderline_threshold}):")
print(f"  Count: {len(borderline_indices)} ({len(borderline_indices)/len(train_baseline_diffs)*100:.1f}%)")

if len(borderline_indices) > 0:
    borderline_mean = np.mean([train_baseline_diffs[i] for i in borderline_indices])
    print(f"  Mean logit diff: {borderline_mean:.3f}")
else:
    print("  ⚠️  WARNING: No borderline cases found! Using full dataset.")
    borderline_indices = list(range(len(train_examples)))

print()

# ============================================================================
# INITIALIZE ADVERSARIAL DELTA
# ============================================================================
print("=" * 80)
print("INITIALIZE ADVERSARIAL DELTA")
print("=" * 80)
print()

target_layer = CONFIG['target_layer']
print(f"Target layer for injection: {target_layer}")
print()

# Initialize delta as zero with requires_grad=True
delta = torch.zeros(hidden_dim, device=device, requires_grad=True)
print(f"Delta shape: {delta.shape}")
print(f"Initial norm: {delta.norm().item():.6f}")
print()

# Optimizer
optimizer = torch.optim.Adam([delta], lr=CONFIG['learning_rate'])
lambda_reg = CONFIG['lambda_reg']

print(f"Optimizer: Adam(lr={CONFIG['learning_rate']})")
print(f"L2 regularization: λ={lambda_reg}")
print()

# ============================================================================
# TRAINING LOOP
# ============================================================================
print("=" * 80)
print("TRAINING ADVERSARIAL DELTA")
print("=" * 80)
print()

def make_hook(delta_vec):
    """Create hook to inject delta into residual stream."""
    def hook(module, input, output):
        # GPT-2 blocks return tuple: (hidden_states, *extras)
        if isinstance(output, tuple):
            hidden_states = output[0]
            modified = hidden_states + delta_vec
            return (modified,) + output[1:]
        else:
            return output + delta_vec
    return hook

batch_size = CONFIG['batch_size']
num_epochs = CONFIG['num_epochs']
max_grad_norm = CONFIG['max_grad_norm']

# Training history
history = {
    "epoch": [],
    "loss": [],
    "mean_logit_diff": [],
    "delta_norm": [],
    "grad_norm": [],
}

print(f"Training on {len(borderline_indices)} borderline examples")
print(f"Batch size: {batch_size}, Epochs: {num_epochs}")
print()

import time
start_time = time.time()

for epoch in range(num_epochs):
    epoch_losses = []
    epoch_logit_diffs = []

    # Shuffle borderline indices
    indices = np.random.permutation(borderline_indices)

    for batch_start in range(0, len(indices), batch_size):
        batch_idx = indices[batch_start:batch_start + batch_size]

        # Get batch data
        batch_input = train_input_ids[batch_idx]
        batch_mask = train_attention_mask[batch_idx]
        batch_positions = [train_target_positions[i] for i in batch_idx]
        batch_correct = train_correct_ids[batch_idx]
        batch_incorrect = train_incorrect_ids[batch_idx]

        # Register hook
        hook_fn = make_hook(delta)
        handle = model.transformer.h[target_layer].register_forward_hook(hook_fn)

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

        # Loss: minimize logit_diff + L2 regularization
        loss = mean_logit_diff + lambda_reg * (delta.norm(p=2) ** 2)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_([delta], max_norm=max_grad_norm)

        # Update
        optimizer.step()

        # Record
        epoch_losses.append(loss.item())
        epoch_logit_diffs.append(mean_logit_diff.item())

    # Epoch statistics
    epoch_loss = np.mean(epoch_losses)
    epoch_mean_logit_diff = np.mean(epoch_logit_diffs)
    delta_norm = delta.norm().item()

    history["epoch"].append(epoch + 1)
    history["loss"].append(epoch_loss)
    history["mean_logit_diff"].append(epoch_mean_logit_diff)
    history["delta_norm"].append(delta_norm)
    history["grad_norm"].append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)

    print(f"Epoch {epoch+1:2d}/{num_epochs}:  "
          f"Loss={epoch_loss:.4f}  "
          f"LogitDiff={epoch_mean_logit_diff:+.3f}  "
          f"||δ||={delta_norm:.3f}  "
          f"||grad||={grad_norm:.2f}")

elapsed_time = time.time() - start_time
print()
print(f"Training complete in {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
print()

# ============================================================================
# SAVE DELTA
# ============================================================================
print("=" * 80)
print("SAVE ADVERSARIAL DELTA")
print("=" * 80)
print()

checkpoints_dir = Path("checkpoints")
checkpoints_dir.mkdir(exist_ok=True)

delta_path = checkpoints_dir / f"adversarial_delta_layer{target_layer}.pt"
torch.save(delta.detach().cpu(), delta_path)

print(f"Delta saved: {delta_path}")
print(f"Delta norm: {delta.norm().item():.3f}")
print()

# ============================================================================
# EVALUATION
# ============================================================================
print("=" * 80)
print("EVALUATION")
print("=" * 80)
print()

def evaluate_with_delta(model, delta_vec, target_layer, input_ids, attention_mask,
                        target_positions, correct_ids, incorrect_ids, batch_size=50):
    """Evaluate model with delta injection."""
    all_logit_diffs = []

    hook_fn = make_hook(delta_vec)
    handle = model.transformer.h[target_layer].register_forward_hook(hook_fn)

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

    handle.remove()
    return all_logit_diffs

print("Evaluating on TRAIN set (borderline)...")
borderline_input_ids = train_input_ids[borderline_indices]
borderline_attention_mask = train_attention_mask[borderline_indices]
borderline_positions = [train_target_positions[i] for i in borderline_indices]
borderline_correct = train_correct_ids[borderline_indices]
borderline_incorrect = train_incorrect_ids[borderline_indices]

borderline_steered_diffs = evaluate_with_delta(
    model, delta, target_layer,
    borderline_input_ids, borderline_attention_mask, borderline_positions,
    borderline_correct, borderline_incorrect
)

borderline_baseline = [train_baseline_diffs[i] for i in borderline_indices]
borderline_baseline_mean = np.mean(borderline_baseline)
borderline_steered_mean = np.mean(borderline_steered_diffs)
borderline_effect = borderline_steered_mean - borderline_baseline_mean

borderline_baseline_acc = sum(1 for d in borderline_baseline if d > 0) / len(borderline_baseline)
borderline_steered_acc = sum(1 for d in borderline_steered_diffs if d > 0) / len(borderline_steered_diffs)
borderline_acc_change = (borderline_steered_acc - borderline_baseline_acc) * 100

print(f"Borderline (n={len(borderline_indices)}):")
print(f"  Baseline:  {borderline_baseline_mean:.3f} ({borderline_baseline_acc*100:.1f}%)")
print(f"  Steered:   {borderline_steered_mean:.3f} ({borderline_steered_acc*100:.1f}%)")
print(f"  Effect:    {borderline_effect:+.3f}  Accuracy Δ: {borderline_acc_change:+.1f}%")
print()

print("Evaluating on FULL TRAIN set...")
train_steered_diffs = evaluate_with_delta(
    model, delta, target_layer,
    train_input_ids, train_attention_mask, train_target_positions,
    train_correct_ids, train_incorrect_ids
)

train_steered_mean = np.mean(train_steered_diffs)
train_effect = train_steered_mean - train_baseline_mean

train_steered_acc = sum(1 for d in train_steered_diffs if d > 0) / len(train_steered_diffs)
train_acc_change = (train_steered_acc - train_baseline_accuracy) * 100

print(f"Full Train (n={len(train_examples)}):")
print(f"  Baseline:  {train_baseline_mean:.3f} ({train_baseline_accuracy*100:.1f}%)")
print(f"  Steered:   {train_steered_mean:.3f} ({train_steered_acc*100:.1f}%)")
print(f"  Effect:    {train_effect:+.3f}  Accuracy Δ: {train_acc_change:+.1f}%")
print()

print("Evaluating on TEST set...")
test_steered_diffs = evaluate_with_delta(
    model, delta, target_layer,
    test_input_ids, test_attention_mask, test_target_positions,
    test_correct_ids, test_incorrect_ids
)

test_steered_mean = np.mean(test_steered_diffs)
test_effect = test_steered_mean - test_baseline_mean

test_steered_acc = sum(1 for d in test_steered_diffs if d > 0) / len(test_steered_diffs)
test_acc_change = (test_steered_acc - test_baseline_accuracy) * 100

print(f"Test (n={len(test_examples)}):")
print(f"  Baseline:  {test_baseline_mean:.3f} ({test_baseline_accuracy*100:.1f}%)")
print(f"  Steered:   {test_steered_mean:.3f} ({test_steered_acc*100:.1f}%)")
print(f"  Effect:    {test_effect:+.3f}  Accuracy Δ: {test_acc_change:+.1f}%")
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
    "training": {
        "borderline_count": len(borderline_indices),
        "training_time_seconds": elapsed_time,
        "final_delta_norm": delta.norm().item(),
        "history": history,
    },
    "evaluation": {
        "borderline": {
            "count": len(borderline_indices),
            "baseline_mean": borderline_baseline_mean,
            "steered_mean": borderline_steered_mean,
            "effect": borderline_effect,
            "baseline_accuracy": borderline_baseline_acc,
            "steered_accuracy": borderline_steered_acc,
            "accuracy_change": borderline_acc_change,
        },
        "train": {
            "count": len(train_examples),
            "baseline_mean": train_baseline_mean,
            "steered_mean": train_steered_mean,
            "effect": train_effect,
            "baseline_accuracy": train_baseline_accuracy,
            "steered_accuracy": train_steered_acc,
            "accuracy_change": train_acc_change,
        },
        "test": {
            "count": len(test_examples),
            "baseline_mean": test_baseline_mean,
            "steered_mean": test_steered_mean,
            "effect": test_effect,
            "baseline_accuracy": test_baseline_accuracy,
            "steered_accuracy": test_steered_acc,
            "accuracy_change": test_acc_change,
        },
    },
}

results_path = Path(f"adversarial_steering_layer{target_layer}_results.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved: {results_path}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ADVERSARIAL STEERING VECTOR LEARNING COMPLETE")
print("=" * 80)
print()

print("Configuration:")
print(f"  Target layer: {target_layer}")
print(f"  Delta norm: {delta.norm().item():.3f}")
print(f"  Training time: {elapsed_time:.1f}s")
print()

print("Results:")
print(f"  Borderline effect: {borderline_effect:+.3f}  Accuracy Δ: {borderline_acc_change:+.1f}%")
print(f"  Train effect:      {train_effect:+.3f}  Accuracy Δ: {train_acc_change:+.1f}%")
print(f"  Test effect:       {test_effect:+.3f}  Accuracy Δ: {test_acc_change:+.1f}%")
print()

if abs(test_effect) > 0.5:
    print("✅ SUCCESS: Found adversarial steering vector with strong effect!")
    print(f"   Test accuracy dropped from {test_baseline_accuracy*100:.1f}% to {test_steered_acc*100:.1f}%")
    print()
    print("   This validates that residual stream directions can causally control IOI.")
elif abs(test_effect) > 0.2:
    print("⚠️  MODERATE: Found weak adversarial effect")
    print(f"   Test effect: {test_effect:+.3f}")
    print()
    print("   May need higher learning rate, more epochs, or different layer.")
else:
    print("❌ WEAK: No strong adversarial effect found")
    print(f"   Test effect: {test_effect:+.3f}")
    print()
    print("   Consider:")
    print("   - Try different target layers (0, 5, 9, 11)")
    print("   - Increase learning rate or epochs")
    print("   - Test layer-specific vs universal delta")

print()
print("Next steps:")
print("  - Try multiple layers (sweep 0-11)")
print("  - Visualize delta in SAE feature space (project onto Atlas)")
print("  - Test multi-layer delta combinations")
print()

# ============================================================================
# TODO: Phase 4B-B (SAE Feature Space Delta)
# ============================================================================
print("=" * 80)
print("TODO: Phase 4B-B - SAE Feature Space Delta")
print("=" * 80)
print()
print("Future extension:")
print("  Instead of learning delta ∈ R^768 (free residual),")
print("  learn alpha ∈ R^6144 (SAE feature space):")
print()
print("    delta = SAE.decoder(alpha) + SAE.pre_bias")
print()
print("  This constrains delta to be a linear combination of discovered features,")
print("  making it interpretable via Phase 3 feature analysis.")
print()
print("  Implementation:")
print("    1. Load SAE for target layer")
print("    2. Initialize alpha = torch.zeros(6144, requires_grad=True)")
print("    3. In hook: delta = sae.decoder(alpha) + sae.pre_bias")
print("    4. Optimize alpha with same loss function")
print()
print("  Benefits:")
print("    - Interpretable: can see which features contribute to adversarial effect")
print("    - Constrained: stays in learned feature manifold")
print("    - Connects Phase 3 (features) with Phase 4B (steering)")
print()
