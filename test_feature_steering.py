"""
Feature Hyper-Activation Steering Test

INVERSE VLO TEST: Instead of ablating features, we FORCE them active.

For negative features ("IOI Killer"), ablation on high-performing model gives
false negatives. The correct test is CLAMPING: force activation high and
measure if accuracy collapses.

Target: Layer 9, Feature 3428 ("IOI Killer", r=-0.798)
Hypothesis: Forcing this feature active should crash accuracy from 97% → 0%
"""

import sys
import json
import torch
from pathlib import Path
from datetime import datetime

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from transformers import AutoModelForCausalLM, AutoTokenizer
from neurotrace.datasets import IOIDatasetGenerator
from neurotrace.control import EnhancedSAEFeatureStore

print("=" * 80)
print("FEATURE HYPER-ACTIVATION STEERING TEST")
print("=" * 80)
print("Testing causal importance via FORCED ACTIVATION (not ablation)")
print()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
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

# Select top negative features for clamping test
negative_features = [f for f in discovered_features if f["correlation_with_success"] < 0]
top_negative = sorted(negative_features, key=lambda x: x["correlation_with_success"])[:5]

print("Top 5 Negative Features (forcing active should CRASH accuracy):")
for i, feat in enumerate(top_negative, 1):
    print(f"  {i}. Layer {feat['layer']:2d} Feature {feat['feature_idx']:4d}  "
          f"Corr={feat['correlation_with_success']:+.3f}  "
          f"Freq={feat['activation_frequency']*100:.1f}%")
print()

# Also test positive features (forcing active should IMPROVE accuracy)
positive_features = [f for f in discovered_features if f["correlation_with_success"] > 0]
top_positive = sorted(positive_features, key=lambda x: -x["correlation_with_success"])[:3]

print("Top 3 Positive Features (forcing active should IMPROVE accuracy):")
for i, feat in enumerate(top_positive, 1):
    print(f"  {i}. Layer {feat['layer']:2d} Feature {feat['feature_idx']:4d}  "
          f"Corr={feat['correlation_with_success']:+.3f}  "
          f"Freq={feat['activation_frequency']*100:.1f}%")
print()

features_to_test = top_negative + top_positive
print(f"Total features to test: {len(features_to_test)}")
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

layers_needed = sorted(set(f["layer"] for f in features_to_test))
print(f"Layers needed: {layers_needed}")

for layer in layers_needed:
    checkpoint_path = Path(f"checkpoints/all_layers_sae/layer_{layer}/final.pt")
    if checkpoint_path.exists():
        feature_store.load_sae(str(checkpoint_path), layer=layer, device=device)
        print(f"  Loaded SAE for layer {layer}")
    else:
        print(f"  WARNING: Layer {layer} SAE not found!")

print()

# ============================================================================
# GENERATE TEST DATASET
# ============================================================================
print("=" * 80)
print("GENERATE TEST DATASET")
print("=" * 80)
print()

print("Generating IOI examples...")
generator = IOIDatasetGenerator(seed=42)

# Use 100 examples for faster testing
num_test = 100
test_examples = generator.generate(num_examples=num_test)

print(f"Generated {len(test_examples)} IOI examples")
print(f"Example: '{test_examples[0].text}'")
print(f"  Correct: {test_examples[0].correct_answer}")
print(f"  Incorrect: {test_examples[0].incorrect_answer}")
print()

# Tokenize examples
texts = [ex.text for ex in test_examples]
encoding = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
).to(device)

input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]

# Get target positions (last token)
target_positions = (attention_mask.sum(dim=1) - 1).tolist()

# Get correct/incorrect token IDs
correct_ids = [
    tokenizer.encode(" " + ex.correct_answer, add_special_tokens=False)[0]
    for ex in test_examples
]
incorrect_ids = [
    tokenizer.encode(" " + ex.incorrect_answer, add_special_tokens=False)[0]
    for ex in test_examples
]

print(f"Tokenized {len(test_examples)} examples")
print(f"Input shape: {input_ids.shape}")
print()

# ============================================================================
# BASELINE (NO INTERVENTION)
# ============================================================================
print("=" * 80)
print("MEASURE BASELINE")
print("=" * 80)
print()

print("Running baseline forward pass...")
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    baseline_logits = outputs.logits

# Calculate baseline logit difference
baseline_logit_diffs = []
for i in range(len(test_examples)):
    pos = target_positions[i]
    logit_correct = baseline_logits[i, pos, correct_ids[i]].item()
    logit_incorrect = baseline_logits[i, pos, incorrect_ids[i]].item()
    baseline_logit_diffs.append(logit_correct - logit_incorrect)

baseline_mean = sum(baseline_logit_diffs) / len(baseline_logit_diffs)
baseline_accuracy = sum(1 for diff in baseline_logit_diffs if diff > 0) / len(baseline_logit_diffs)

print(f"Baseline Mean Logit Diff: {baseline_mean:.3f}")
print(f"Baseline Accuracy: {baseline_accuracy*100:.1f}%")
print()

# ============================================================================
# HYPER-ACTIVATION STEERING TEST
# ============================================================================
print("=" * 80)
print("HYPER-ACTIVATION STEERING TEST")
print("=" * 80)
print()

print(f"Testing {len(features_to_test)} features...")
print(f"Method: FORCE feature activation → Measure impact")
print()

import time
start_time = time.time()

# Test different clamping strengths
clamping_strengths = [2.0, 5.0, 10.0]

steering_results = []

for idx, feat in enumerate(features_to_test, 1):
    layer = feat["layer"]
    feature_idx = feat["feature_idx"]
    correlation = feat["correlation_with_success"]

    print(f"[{idx}/{len(features_to_test)}] Testing Layer {layer}, Feature {feature_idx} (r={correlation:+.3f})...")

    # Get SAE for this layer
    if layer not in feature_store.saes:
        print(f"  ERROR: SAE for layer {layer} not loaded!")
        continue

    sae = feature_store.saes[layer]

    feature_results = {
        "layer": layer,
        "feature_idx": feature_idx,
        "correlation_with_success": correlation,
        "activation_frequency": feat["activation_frequency"],
        "baseline_logit_diff": baseline_mean,
        "baseline_accuracy": baseline_accuracy,
        "clamping_tests": []
    }

    # Test different clamping strengths
    for clamp_value in clamping_strengths:
        # Register hook to FORCE this specific feature active
        def steering_hook(module, input, output):
            # Extract hidden states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            batch, seq, hidden = hidden_states.shape

            # Flatten for SAE
            flat = hidden_states.view(-1, hidden)

            with torch.no_grad():
                # Forward through SAE
                sae_output = sae(flat)
                codes = sae_output['codes']  # [batch*seq, dict_size]

                # CLAMP: Force feature to high activation
                codes[:, feature_idx] = codes[:, feature_idx] + clamp_value

                # Reconstruct with clamped codes
                reconstructed_flat = sae.decoder(codes) + sae.pre_bias

            # Reshape back
            reconstructed = reconstructed_flat.view(batch, seq, hidden)

            if isinstance(output, tuple):
                return (reconstructed,) + output[1:]
            else:
                return reconstructed

        # Register hook on MLP output
        mlp_module = model.transformer.h[layer].mlp
        handle = mlp_module.register_forward_hook(steering_hook)

        # Forward pass with steering
        with torch.no_grad():
            outputs_steered = model(input_ids, attention_mask=attention_mask)
            steered_logits = outputs_steered.logits

        # Remove hook
        handle.remove()

        # Calculate steered logit difference
        steered_logit_diffs = []
        for i in range(len(test_examples)):
            pos = target_positions[i]
            logit_correct = steered_logits[i, pos, correct_ids[i]].item()
            logit_incorrect = steered_logits[i, pos, incorrect_ids[i]].item()
            steered_logit_diffs.append(logit_correct - logit_incorrect)

        steered_mean = sum(steered_logit_diffs) / len(steered_logit_diffs)
        steered_accuracy = sum(1 for diff in steered_logit_diffs if diff > 0) / len(steered_logit_diffs)

        # Calculate steering effect
        steering_effect = steered_mean - baseline_mean
        accuracy_change = (steered_accuracy - baseline_accuracy) * 100

        feature_results["clamping_tests"].append({
            "clamp_value": clamp_value,
            "steered_logit_diff": steered_mean,
            "steered_accuracy": steered_accuracy,
            "steering_effect": steering_effect,
            "accuracy_change": accuracy_change,
        })

    # Print results for this feature
    print(f"  Baseline: {baseline_mean:.3f} ({baseline_accuracy*100:.1f}%)")
    for test in feature_results["clamping_tests"]:
        print(f"  Clamp +{test['clamp_value']:.1f}: {test['steered_logit_diff']:.3f} ({test['steered_accuracy']*100:.1f}%)  "
              f"Effect: {test['steering_effect']:+.3f}  AccΔ: {test['accuracy_change']:+.1f}%")
    print()

    steering_results.append(feature_results)

elapsed_time = time.time() - start_time

print(f"Steering test complete in {elapsed_time:.1f} seconds")
print()

# ============================================================================
# ANALYSIS
# ============================================================================
print("=" * 80)
print("ANALYSIS: CAUSAL FEATURES")
print("=" * 80)
print()

# Find features with strongest effect at max clamping
print("Features Ranked by Steering Effect (at clamp=10.0):")
print()

results_with_max_effect = []
for result in steering_results:
    max_clamp_test = result["clamping_tests"][-1]  # Highest clamping value
    results_with_max_effect.append({
        **result,
        "max_steering_effect": max_clamp_test["steering_effect"],
        "max_accuracy_change": max_clamp_test["accuracy_change"],
    })

# Sort by absolute steering effect
sorted_results = sorted(results_with_max_effect, key=lambda x: abs(x["max_steering_effect"]), reverse=True)

for i, result in enumerate(sorted_results, 1):
    print(f"{i:2d}. Layer {result['layer']:2d} Feature {result['feature_idx']:4d}")
    print(f"    Correlation: {result['correlation_with_success']:+.3f}")
    print(f"    Max Steering Effect: {result['max_steering_effect']:+.3f}  AccΔ: {result['max_accuracy_change']:+.1f}%")
    print()

# ============================================================================
# VALIDATION
# ============================================================================
print("=" * 80)
print("VALIDATION: Did We Find Causal Features?")
print("=" * 80)
print()

# Check if any feature has strong effect
strong_effects = [r for r in sorted_results if abs(r["max_steering_effect"]) > 0.5]

if strong_effects:
    print(f"✅ SUCCESS: Found {len(strong_effects)} features with |effect| > 0.5")
    print()
    print("Top Causal Features:")
    for result in strong_effects[:3]:
        print(f"  - Layer {result['layer']:2d} F{result['feature_idx']:4d}  "
              f"Effect={result['max_steering_effect']:+.3f}  AccΔ={result['max_accuracy_change']:+.1f}%")
    print()
else:
    print(f"⚠️  NO strong effects found (all |effect| < 0.5)")
    print()
    print("Possible explanations:")
    print("  1. Features are correlated but not causal")
    print("  2. Clamping strength too low (try higher values)")
    print("  3. IOI is truly distributed/dense (not sparse features)")
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
    "config": {
        "num_examples": num_test,
        "features_tested": len(features_to_test),
        "clamping_strengths": clamping_strengths,
        "baseline_accuracy": baseline_accuracy,
        "baseline_logit_diff": baseline_mean,
    },
    "steering_results": sorted_results,
    "statistics": {
        "strong_effects_count": len(strong_effects),
        "elapsed_time_seconds": elapsed_time,
    }
}

output_path = Path("feature_steering_results.json")
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Results saved: {output_path}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("HYPER-ACTIVATION STEERING TEST COMPLETE")
print("=" * 80)
print()
print("Results:")
print(f"  - Features tested: {len(steering_results)}")
print(f"  - Strong causal effects: {len(strong_effects)}")
print(f"  - Baseline accuracy: {baseline_accuracy*100:.1f}%")
print(f"  - Time elapsed: {elapsed_time:.1f}s")
print()

if strong_effects:
    print("🎯 CONCLUSION: Feature-level causality VALIDATED")
    print()
    print("Next steps:")
    print("  - Use validated features for steering (Phase 4B)")
    print("  - Build multi-layer circuits from causal features")
else:
    print("⚠️  CONCLUSION: No strong feature-level causality found")
    print()
    print("Next steps:")
    print("  - Try higher clamping values (20.0, 50.0)")
    print("  - Test feature combinations (multi-feature steering)")
    print("  - Consider that IOI may be dense/distributed")
print()
