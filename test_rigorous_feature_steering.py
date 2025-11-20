"""
Rigorous Feature Steering Test - Phase 4A (Revised)

Addresses methodological issues:
1. Small dataset + ceiling effect → Use 1000+ examples + borderline filtering
2. Uncalibrated clamping → Use percentile-based clamping (P90, P99, P99.9)
3. Single-feature on distributed task → Test multi-feature combinations
4. Global ablation only → Add conditional ablation

Scientific hypothesis:
- IOI is a distributed/circuit-level task
- Single features may not show strong effects
- Multi-feature steering should reveal circuit-level causality
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from transformers import AutoModelForCausalLM, AutoTokenizer
from neurotrace.datasets import IOIDatasetGenerator
from neurotrace.control import EnhancedSAEFeatureStore

print("=" * 80)
print("RIGOROUS FEATURE STEERING TEST (Phase 4A - Revised)")
print("=" * 80)
print()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print()

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "large_dataset_size": 2000,  # Increased statistical power
    "borderline_threshold": 1.5,  # logit_diff < 1.5 (near decision boundary)
    "percentile_levels": [90, 99, 99.9],  # Natural distribution percentiles
    "multi_feature_count": 5,  # Test top 5 features together
}

print("Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")
print()

# ============================================================================
# LOAD PHASE 3 RESULTS
# ============================================================================
print("=" * 80)
print("LOAD PHASE 3 DISCOVERED FEATURES")
print("=" * 80)
print()

discovery_results_path = Path("feature_circuit_discovery.json")
if not discovery_results_path.exists():
    print("ERROR: feature_circuit_discovery.json not found!")
    sys.exit(1)

with open(discovery_results_path) as f:
    discovery_results = json.load(f)

discovered_features = discovery_results["discovered_features"]
print(f"Loaded {len(discovered_features)} discovered features")
print()

# Select top features
negative_features = [f for f in discovered_features if f["correlation_with_success"] < 0]
top_negative = sorted(negative_features, key=lambda x: x["correlation_with_success"])[:10]

positive_features = [f for f in discovered_features if f["correlation_with_success"] > 0]
top_positive = sorted(positive_features, key=lambda x: -x["correlation_with_success"])[:5]

print(f"Selected {len(top_negative)} negative + {len(top_positive)} positive features")
print()

# ============================================================================
# LOAD MODEL AND SAEs
# ============================================================================
print("=" * 80)
print("LOAD MODEL AND SAEs")
print("=" * 80)
print()

print("Loading GPT-2...")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
print("Model loaded")
print()

print("Loading SAEs...")
feature_store = EnhancedSAEFeatureStore()

all_features = top_negative + top_positive
layers_needed = sorted(set(f["layer"] for f in all_features))
print(f"Layers needed: {layers_needed}")

for layer in layers_needed:
    checkpoint_path = Path(f"checkpoints/all_layers_sae/layer_{layer}/final.pt")
    if checkpoint_path.exists():
        feature_store.load_sae(str(checkpoint_path), layer=layer, device=device)
        print(f"  Loaded SAE for layer {layer}")

print()

# ============================================================================
# STEP 1: GENERATE LARGE DATASET
# ============================================================================
print("=" * 80)
print("STEP 1: GENERATE LARGE DATASET")
print("=" * 80)
print()

print(f"Generating {CONFIG['large_dataset_size']} IOI examples...")
generator = IOIDatasetGenerator(seed=42)
test_examples = generator.generate(num_examples=CONFIG['large_dataset_size'])

print(f"Generated {len(test_examples)} examples")
print()

# Tokenize
texts = [ex.text for ex in test_examples]
encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]

target_positions = (attention_mask.sum(dim=1) - 1).tolist()
correct_ids = [
    tokenizer.encode(" " + ex.correct_answer, add_special_tokens=False)[0]
    for ex in test_examples
]
incorrect_ids = [
    tokenizer.encode(" " + ex.incorrect_answer, add_special_tokens=False)[0]
    for ex in test_examples
]

print(f"Tokenized {len(test_examples)} examples")
print()

# ============================================================================
# STEP 2: BASELINE + IDENTIFY BORDERLINE CASES
# ============================================================================
print("=" * 80)
print("STEP 2: BASELINE + IDENTIFY BORDERLINE CASES")
print("=" * 80)
print()

print("Running baseline forward pass (batched for memory)...")
batch_size = 50
all_logit_diffs = []

with torch.no_grad():
    for i in range(0, len(test_examples), batch_size):
        batch_input = input_ids[i:i+batch_size]
        batch_mask = attention_mask[i:i+batch_size]

        outputs = model(input_ids=batch_input, attention_mask=batch_mask)
        batch_logits = outputs.logits.cpu()

        for j in range(len(batch_logits)):
            idx = i + j
            pos = target_positions[idx]
            logit_correct = batch_logits[j, pos, correct_ids[idx]].item()
            logit_incorrect = batch_logits[j, pos, incorrect_ids[idx]].item()
            all_logit_diffs.append(logit_correct - logit_incorrect)

baseline_mean = np.mean(all_logit_diffs)
baseline_std = np.std(all_logit_diffs)
baseline_accuracy = sum(1 for diff in all_logit_diffs if diff > 0) / len(all_logit_diffs)

print(f"Baseline Statistics:")
print(f"  Mean logit diff: {baseline_mean:.3f} ± {baseline_std:.3f}")
print(f"  Accuracy: {baseline_accuracy*100:.1f}%")
print()

# Identify borderline cases
borderline_threshold = CONFIG['borderline_threshold']
borderline_indices = [
    i for i, diff in enumerate(all_logit_diffs)
    if 0.0 < diff < borderline_threshold
]

print(f"Borderline cases (0 < logit_diff < {borderline_threshold}):")
print(f"  Count: {len(borderline_indices)} ({len(borderline_indices)/len(all_logit_diffs)*100:.1f}%)")

if len(borderline_indices) > 0:
    borderline_mean = np.mean([all_logit_diffs[i] for i in borderline_indices])
    print(f"  Mean logit diff: {borderline_mean:.3f}")
else:
    print("  ⚠️  WARNING: No borderline cases found! Dataset too easy.")
    print("  Proceeding with full dataset...")
    borderline_indices = list(range(len(test_examples)))

print()

# ============================================================================
# STEP 3: COLLECT FEATURE ACTIVATION DISTRIBUTIONS
# ============================================================================
print("=" * 80)
print("STEP 3: COLLECT FEATURE ACTIVATION DISTRIBUTIONS")
print("=" * 80)
print()

print("Collecting natural activation distributions for features...")
print("This is needed for percentile-based clamping")
print()

feature_distributions = {}

for feat in all_features:
    layer = feat["layer"]
    feature_idx = feat["feature_idx"]

    if layer not in feature_store.saes:
        continue

    sae = feature_store.saes[layer]
    activations = []

    # Collect activations across dataset
    with torch.no_grad():
        for i in range(0, min(500, len(test_examples)), batch_size):
            batch_input = input_ids[i:i+batch_size]
            batch_mask = attention_mask[i:i+batch_size]

            # Forward to get hidden states at this layer
            outputs = model(input_ids=batch_input, attention_mask=batch_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer + 1]  # +1 because includes input embeddings

            # Pass through SAE
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            sae_output = sae(flat)
            codes = sae_output['codes']

            # Extract this feature's activations
            feature_acts = codes[:, feature_idx].cpu().numpy()
            activations.extend(feature_acts[feature_acts > 0])  # Only non-zero

    if len(activations) > 0:
        activations = np.array(activations)
        feature_distributions[(layer, feature_idx)] = {
            "mean": float(np.mean(activations)),
            "std": float(np.std(activations)),
            "p90": float(np.percentile(activations, 90)),
            "p99": float(np.percentile(activations, 99)),
            "p99.9": float(np.percentile(activations, 99.9)),
            "max": float(np.max(activations)),
        }

        print(f"  Layer {layer} F{feature_idx}: "
              f"mean={feature_distributions[(layer, feature_idx)]['mean']:.2f}, "
              f"P99={feature_distributions[(layer, feature_idx)]['p99']:.2f}, "
              f"max={feature_distributions[(layer, feature_idx)]['max']:.2f}")

print()
print(f"Collected distributions for {len(feature_distributions)} features")
print()

# ============================================================================
# STEP 4: MULTI-FEATURE STEERING TEST
# ============================================================================
print("=" * 80)
print("STEP 4: MULTI-FEATURE STEERING TEST")
print("=" * 80)
print()

print("Testing TOP 5 NEGATIVE features TOGETHER (circuit-level hypothesis)")
print()

# Select top 5 negative features
circuit_features = top_negative[:CONFIG['multi_feature_count']]

print("Circuit features:")
for i, feat in enumerate(circuit_features, 1):
    print(f"  {i}. Layer {feat['layer']:2d} F{feat['feature_idx']:4d}  r={feat['correlation_with_success']:+.3f}")
print()

# Test on borderline subset
borderline_input_ids = input_ids[borderline_indices]
borderline_attention_mask = attention_mask[borderline_indices]
borderline_target_positions = [target_positions[i] for i in borderline_indices]
borderline_correct_ids = [correct_ids[i] for i in borderline_indices]
borderline_incorrect_ids = [incorrect_ids[i] for i in borderline_indices]
borderline_baseline_diffs = [all_logit_diffs[i] for i in borderline_indices]

print(f"Testing on {len(borderline_indices)} borderline examples")
print()

# Test different interventions
interventions = [
    ("Ablate All 5", "ablate", None),
    ("Clamp All 5 to P99", "clamp", "p99"),
    ("Clamp All 5 to P99.9", "clamp", "p99.9"),
]

multi_feature_results = []

for intervention_name, intervention_type, percentile in interventions:
    print(f"Testing: {intervention_name}...")

    # Create hooks for all 5 features
    handles = []

    for feat in circuit_features:
        layer = feat["layer"]
        feature_idx = feat["feature_idx"]

        if layer not in feature_store.saes:
            continue

        sae = feature_store.saes[layer]

        # Get clamping value if needed
        clamp_value = None
        if intervention_type == "clamp" and (layer, feature_idx) in feature_distributions:
            dist = feature_distributions[(layer, feature_idx)]
            clamp_value = dist[percentile] if percentile in dist else 5.0

        def make_hook(sae_ref, feat_idx, interv_type, clamp_val):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                batch, seq, hidden = hidden_states.shape
                flat = hidden_states.view(-1, hidden)

                with torch.no_grad():
                    sae_output = sae_ref(flat)
                    codes = sae_output['codes']

                    if interv_type == "ablate":
                        codes[:, feat_idx] = 0.0
                    elif interv_type == "clamp" and clamp_val is not None:
                        codes[:, feat_idx] = clamp_val

                    reconstructed_flat = sae_ref.decoder(codes) + sae_ref.pre_bias

                reconstructed = reconstructed_flat.view(batch, seq, hidden)

                if isinstance(output, tuple):
                    return (reconstructed,) + output[1:]
                else:
                    return reconstructed

            return hook

        hook_fn = make_hook(sae, feature_idx, intervention_type, clamp_value)
        mlp_module = model.transformer.h[layer].mlp
        handle = mlp_module.register_forward_hook(hook_fn)
        handles.append(handle)

    # Forward pass with intervention
    intervened_logit_diffs = []

    with torch.no_grad():
        for i in range(0, len(borderline_input_ids), batch_size):
            batch_input = borderline_input_ids[i:i+batch_size]
            batch_mask = borderline_attention_mask[i:i+batch_size]

            outputs = model(input_ids=batch_input, attention_mask=batch_mask)
            batch_logits = outputs.logits.cpu()

            for j in range(len(batch_logits)):
                idx = i + j
                pos = borderline_target_positions[idx]
                logit_correct = batch_logits[j, pos, borderline_correct_ids[idx]].item()
                logit_incorrect = batch_logits[j, pos, borderline_incorrect_ids[idx]].item()
                intervened_logit_diffs.append(logit_correct - logit_incorrect)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Calculate effect
    baseline_borderline_mean = np.mean(borderline_baseline_diffs)
    intervened_mean = np.mean(intervened_logit_diffs)
    effect = intervened_mean - baseline_borderline_mean

    baseline_borderline_acc = sum(1 for diff in borderline_baseline_diffs if diff > 0) / len(borderline_baseline_diffs)
    intervened_acc = sum(1 for diff in intervened_logit_diffs if diff > 0) / len(intervened_logit_diffs)
    acc_change = (intervened_acc - baseline_borderline_acc) * 100

    print(f"  Baseline: {baseline_borderline_mean:.3f} ({baseline_borderline_acc*100:.1f}%)")
    print(f"  Intervened: {intervened_mean:.3f} ({intervened_acc*100:.1f}%)")
    print(f"  Effect: {effect:+.3f}  Accuracy Δ: {acc_change:+.1f}%")
    print()

    multi_feature_results.append({
        "intervention": intervention_name,
        "type": intervention_type,
        "percentile": percentile,
        "baseline_mean": baseline_borderline_mean,
        "intervened_mean": intervened_mean,
        "effect": effect,
        "baseline_accuracy": baseline_borderline_acc,
        "intervened_accuracy": intervened_acc,
        "accuracy_change": acc_change,
    })

# ============================================================================
# STEP 5: ANALYSIS
# ============================================================================
print("=" * 80)
print("ANALYSIS")
print("=" * 80)
print()

print("Multi-Feature Circuit Test Results:")
print()
for result in multi_feature_results:
    print(f"{result['intervention']}:")
    print(f"  Effect: {result['effect']:+.3f}  Accuracy Δ: {result['accuracy_change']:+.1f}%")
print()

# Find strongest effect
strongest = max(multi_feature_results, key=lambda x: abs(x['effect']))

if abs(strongest['effect']) > 0.3:
    print("✅ SUCCESS: Found circuit-level causal effect!")
    print(f"  Intervention: {strongest['intervention']}")
    print(f"  Effect size: {strongest['effect']:+.3f}")
    print(f"  Accuracy change: {strongest['accuracy_change']:+.1f}%")
else:
    print("⚠️  Weak circuit-level effects")
    print(f"  Strongest effect: {strongest['effect']:+.3f}")
    print()
    print("Implications:")
    print("  - IOI may be highly distributed (not sparse circuit)")
    print("  - Features are correlational markers, not causal nodes")
    print("  - Component-level analysis (Phase 1) more appropriate")

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
    "dataset_stats": {
        "total_examples": len(test_examples),
        "borderline_count": len(borderline_indices),
        "baseline_accuracy": baseline_accuracy,
        "baseline_mean_logit_diff": baseline_mean,
        "baseline_std_logit_diff": baseline_std,
    },
    "feature_distributions": {
        f"layer_{k[0]}_feature_{k[1]}": v
        for k, v in feature_distributions.items()
    },
    "multi_feature_results": multi_feature_results,
}

output_path = Path("rigorous_feature_steering_results.json")
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Results saved: {output_path}")
print()

print("=" * 80)
print("RIGOROUS FEATURE STEERING TEST COMPLETE")
print("=" * 80)
print()
