"""
Feature-Level VLO Testing

Validates that features discovered in Phase 3 are CAUSALLY important,
not just correlated with task success.

Method:
1. Load top features from feature_circuit_discovery.json
2. For each feature:
   - Ablate the feature (set activation to 0)
   - Measure VLO (change in logit difference)
3. Compare correlation vs causation

Expected Results:
- Layer 9 F3428 ("IOI Killer", r=-0.798): VLO > 2.0 (ablating improves)
- Layer 11 F1724 ("Success Marker", r=+0.361): VLO < -1.0 (ablating hurts)
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
print("FEATURE-LEVEL VLO TESTING")
print("=" * 80)
print("Validating causal importance of discovered features")
print()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print()

# ============================================================================
# STEP 1: LOAD DISCOVERED FEATURES
# ============================================================================
print("=" * 80)
print("STEP 1: LOAD PHASE 3 RESULTS")
print("=" * 80)
print()

discovery_results_path = Path("feature_circuit_discovery.json")
if not discovery_results_path.exists():
    print("ERROR: feature_circuit_discovery.json not found!")
    print("Run discover_feature_circuits.py first")
    sys.exit(1)

with open(discovery_results_path) as f:
    discovery_results = json.load(f)

discovered_features = discovery_results["discovered_features"]
print(f"Loaded {len(discovered_features)} discovered features")
print()

# Select top features for VLO testing
print("Selecting features for VLO testing:")
print()

# Top 5 negative features (expect positive VLO when ablated)
negative_features = [f for f in discovered_features if f["correlation_with_success"] < 0]
top_negative = sorted(negative_features, key=lambda x: x["correlation_with_success"])[:5]

print("Top 5 Negative Features (ablating should IMPROVE accuracy):")
for i, feat in enumerate(top_negative, 1):
    print(f"  {i}. Layer {feat['layer']:2d} Feature {feat['feature_idx']:4d}  "
          f"Corr={feat['correlation_with_success']:+.3f}  "
          f"Freq={feat['activation_frequency']*100:.1f}%")
print()

# Top 3 positive features (expect negative VLO when ablated)
positive_features = [f for f in discovered_features if f["correlation_with_success"] > 0]
top_positive = sorted(positive_features, key=lambda x: -x["correlation_with_success"])[:3]

print("Top 3 Positive Features (ablating should HURT accuracy):")
for i, feat in enumerate(top_positive, 1):
    print(f"  {i}. Layer {feat['layer']:2d} Feature {feat['feature_idx']:4d}  "
          f"Corr={feat['correlation_with_success']:+.3f}  "
          f"Freq={feat['activation_frequency']*100:.1f}%")
print()

# Combine for testing
features_to_test = top_negative + top_positive
print(f"Total features to test: {len(features_to_test)}")
print()

# ============================================================================
# STEP 2: LOAD MODEL AND SAEs
# ============================================================================
print("=" * 80)
print("STEP 2: LOAD MODEL AND SAEs")
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

# Get unique layers needed
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
# STEP 3: GENERATE TEST DATASET
# ============================================================================
print("=" * 80)
print("STEP 3: GENERATE TEST DATASET")
print("=" * 80)
print()

print("Generating IOI examples...")
generator = IOIDatasetGenerator(seed=42)

# Use 200 examples for robust VLO measurement
num_test = 200
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
# STEP 4: BASELINE (NO INTERVENTION)
# ============================================================================
print("=" * 80)
print("STEP 4: MEASURE BASELINE")
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
# STEP 5: FEATURE ABLATION VLO TESTING
# ============================================================================
print("=" * 80)
print("STEP 5: FEATURE ABLATION VLO TESTING")
print("=" * 80)
print()

print(f"Testing {len(features_to_test)} features...")
print(f"Method: Ablate feature activation → Measure VLO")
print()

import time
start_time = time.time()

vlo_results = []

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

    # Register hook to ablate this specific feature
    def ablation_hook(module, input, output):
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

            # ABLATE: Zero out the target feature
            codes[:, feature_idx] = 0.0

            # Reconstruct with ablated codes
            reconstructed_flat = sae.decoder(codes) + sae.pre_bias

        # Reshape back
        reconstructed = reconstructed_flat.view(batch, seq, hidden)

        if isinstance(output, tuple):
            return (reconstructed,) + output[1:]
        else:
            return reconstructed

    # Register hook on MLP output
    mlp_module = model.transformer.h[layer].mlp
    handle = mlp_module.register_forward_hook(ablation_hook)

    # Forward pass with ablation
    with torch.no_grad():
        outputs_ablated = model(input_ids, attention_mask=attention_mask)
        ablated_logits = outputs_ablated.logits

    # Remove hook
    handle.remove()

    # Calculate ablated logit difference
    ablated_logit_diffs = []
    for i in range(len(test_examples)):
        pos = target_positions[i]
        logit_correct = ablated_logits[i, pos, correct_ids[i]].item()
        logit_incorrect = ablated_logits[i, pos, incorrect_ids[i]].item()
        ablated_logit_diffs.append(logit_correct - logit_incorrect)

    ablated_mean = sum(ablated_logit_diffs) / len(ablated_logit_diffs)
    ablated_accuracy = sum(1 for diff in ablated_logit_diffs if diff > 0) / len(ablated_logit_diffs)

    # Calculate VLO
    vlo = ablated_mean - baseline_mean

    # Calculate effect size (accuracy change)
    accuracy_change = (ablated_accuracy - baseline_accuracy) * 100

    print(f"  Baseline: {baseline_mean:.3f} ({baseline_accuracy*100:.1f}%)")
    print(f"  Ablated:  {ablated_mean:.3f} ({ablated_accuracy*100:.1f}%)")
    print(f"  VLO: {vlo:+.3f}  Accuracy Change: {accuracy_change:+.1f}%")
    print()

    vlo_results.append({
        "layer": layer,
        "feature_idx": feature_idx,
        "correlation_with_success": correlation,
        "activation_frequency": feat["activation_frequency"],
        "baseline_logit_diff": baseline_mean,
        "ablated_logit_diff": ablated_mean,
        "vlo": vlo,
        "baseline_accuracy": baseline_accuracy,
        "ablated_accuracy": ablated_accuracy,
        "accuracy_change": accuracy_change,
    })

elapsed_time = time.time() - start_time

print(f"VLO testing complete in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
print()

# ============================================================================
# STEP 6: ANALYZE RESULTS
# ============================================================================
print("=" * 80)
print("STEP 6: ANALYSIS")
print("=" * 80)
print()

# Sort by VLO magnitude
vlo_results_sorted = sorted(vlo_results, key=lambda x: abs(x["vlo"]), reverse=True)

print("Features Ranked by VLO (Causal Importance):")
print()
for i, result in enumerate(vlo_results_sorted, 1):
    print(f"{i:2d}. Layer {result['layer']:2d} Feature {result['feature_idx']:4d}")
    print(f"    Correlation: {result['correlation_with_success']:+.3f}")
    print(f"    VLO: {result['vlo']:+.3f}  Accuracy: {result['accuracy_change']:+.1f}%")
    print()

# Compare correlation vs causation
print("=" * 80)
print("CORRELATION vs CAUSATION")
print("=" * 80)
print()

import numpy as np

correlations = [r["correlation_with_success"] for r in vlo_results]
vlos = [r["vlo"] for r in vlo_results]

# Calculate correlation between correlation and VLO
corr_array = np.array(correlations)
vlo_array = np.array(vlos)
pearson_corr = np.corrcoef(corr_array, vlo_array)[0, 1]

print(f"Correlation (Phase 3) vs VLO (Phase 4A): r={pearson_corr:.3f}")
print()

if pearson_corr > 0.7:
    print("✅ STRONG agreement: Correlation predicts causation!")
elif pearson_corr > 0.4:
    print("⚠️  MODERATE agreement: Correlation somewhat predicts causation")
else:
    print("❌ WEAK agreement: Correlation does NOT predict causation well")
print()

# ============================================================================
# STEP 7: SAVE RESULTS
# ============================================================================
print("=" * 80)
print("STEP 7: SAVE RESULTS")
print("=" * 80)
print()

output = {
    "timestamp": datetime.now().isoformat(),
    "config": {
        "num_examples": num_test,
        "features_tested": len(features_to_test),
        "baseline_accuracy": baseline_accuracy,
        "baseline_logit_diff": baseline_mean,
    },
    "vlo_results": vlo_results_sorted,
    "statistics": {
        "correlation_vs_vlo_pearson": float(pearson_corr),
        "elapsed_time_seconds": elapsed_time,
    }
}

output_path = Path("feature_vlo_results.json")
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Results saved: {output_path}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("FEATURE VLO TESTING COMPLETE")
print("=" * 80)
print()
print("Results:")
print(f"  - Features tested: {len(vlo_results)}")
print(f"  - Correlation vs VLO: r={pearson_corr:.3f}")
print(f"  - Baseline accuracy: {baseline_accuracy*100:.1f}%")
print(f"  - Time elapsed: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
print()

# Highlight top causal features
print("Top 3 Causal Features (by |VLO|):")
for i, result in enumerate(vlo_results_sorted[:3], 1):
    print(f"  {i}. Layer {result['layer']:2d} F{result['feature_idx']:4d}  "
          f"VLO={result['vlo']:+.3f}  AccΔ={result['accuracy_change']:+.1f}%")
print()

print("Next steps:")
print("  - Use VLO-validated features for steering (Phase 4B)")
print("  - Build circuits from causally important features")
print()
