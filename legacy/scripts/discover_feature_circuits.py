"""
Feature-Based Circuit Discovery from Neural Atlas

Discovers WHICH SPECIFIC FEATURES (out of 36,864) drive IOI task behavior.

Unlike component-level discovery (layer_0.mlp as a whole), this identifies
individual features within each layer that are causally important.

Uses correlation analysis + activation patterns across all 12 layers.
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
from neurotrace.discovery import FeatureCircuitDiscoverer
from neurotrace.control import EnhancedSAEFeatureStore

print("=" * 80)
print("FEATURE-BASED CIRCUIT DISCOVERY")
print("=" * 80)
print("Analyzing 73,728 Atlas features to find IOI circuit")
print()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print()

# ============================================================================
# STEP 1: LOAD MODEL
# ============================================================================
print("=" * 80)
print("STEP 1: LOAD GPT-2 MODEL")
print("=" * 80)
print()

print("Loading GPT-2...")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded: GPT-2 (124M parameters)")
print()

# ============================================================================
# STEP 2: LOAD NEURAL ATLAS (12 SAEs)
# ============================================================================
print("=" * 80)
print("STEP 2: LOAD NEURAL ATLAS")
print("=" * 80)
print()

feature_store = EnhancedSAEFeatureStore()
atlas_dir = Path("checkpoints/all_layers_sae")

if not atlas_dir.exists():
    print("ERROR: Atlas not found!")
    print(f"Expected: {atlas_dir}")
    print("Run: python train_atlas_simple.py")
    sys.exit(1)

# Load all 12 SAEs
loaded_layers = []
for layer_idx in range(12):
    checkpoint_path = atlas_dir / f"layer_{layer_idx}" / "final.pt"
    if checkpoint_path.exists():
        feature_store.load_sae(str(checkpoint_path), layer=layer_idx, device=device)
        loaded_layers.append(layer_idx)
    else:
        print(f"WARNING: Layer {layer_idx} SAE not found")

print()
print(f"Loaded SAEs: {len(loaded_layers)}/12 layers")
print(f"Total features: {len(loaded_layers) * 6144}")
print()

if len(loaded_layers) < 12:
    print(f"WARNING: Missing layers {[i for i in range(12) if i not in loaded_layers]}")
    print()

# ============================================================================
# STEP 3: GENERATE IOI DATASET
# ============================================================================
print("=" * 80)
print("STEP 3: GENERATE IOI DATASET")
print("=" * 80)
print()

print("Generating IOI test examples...")
generator = IOIDatasetGenerator(seed=42)

# Use 100 examples (faster for feature analysis)
num_test = 100
test_examples = generator.generate(num_examples=num_test)

print(f"Generated {len(test_examples)} IOI examples")
print(f"Example: '{test_examples[0].text}'")
print(f"  Correct: {test_examples[0].correct_answer}")
print(f"  Incorrect: {test_examples[0].incorrect_answer}")
print()

# ============================================================================
# STEP 4: DISCOVER IMPORTANT FEATURES
# ============================================================================
print("=" * 80)
print("STEP 4: FEATURE DISCOVERY")
print("=" * 80)
print()

print("Initializing FeatureCircuitDiscoverer...")
discoverer = FeatureCircuitDiscoverer(
    feature_store=feature_store,
    model=model,
    tokenizer=tokenizer,
    device=device
)

print()
print("Running feature discovery...")
print(f"  Analyzing: {len(loaded_layers)} layers × 6,144 features = {len(loaded_layers) * 6144} total")
print(f"  Dataset: {num_test} IOI examples")
print(f"  Method: Correlation analysis + activation patterns")
print()

import time
start_time = time.time()

important_features = discoverer.discover_from_examples(
    examples=test_examples,
    top_k_per_layer=20,  # Top 20 features per layer
    min_correlation=0.2,  # Minimum correlation with success
    verbose=True
)

elapsed_time = time.time() - start_time

print(f"Discovery complete in {elapsed_time:.1f} seconds")
print()

# ============================================================================
# STEP 5: ANALYZE RESULTS
# ============================================================================
print("=" * 80)
print("STEP 5: ANALYSIS")
print("=" * 80)
print()

if important_features:
    print(f"Found {len(important_features)} important features")
    print()

    # Group by layer
    layer_counts = {}
    for feat in important_features:
        layer_counts[feat.layer] = layer_counts.get(feat.layer, 0) + 1

    print("Features per layer:")
    for layer in sorted(layer_counts.keys()):
        count = layer_counts[layer]
        bar = "█" * (count // 2)
        print(f"  Layer {layer:2d}: {count:3d} features  {bar}")
    print()

    # Top 30 features overall
    print("Top 30 features by correlation:")
    for i, feat in enumerate(important_features[:30], 1):
        sign = "+" if feat.correlation_with_success >= 0 else "-"
        print(f"  {i:2d}. Layer {feat.layer:2d} Feature {feat.feature_idx:4d}  "
              f"Corr={sign}{abs(feat.correlation_with_success):.3f}  "
              f"MeanAct={feat.mean_activation:.3f}  "
              f"Freq={feat.activation_frequency*100:5.1f}%")
    print()

    # Check Layer 0 features (since layer_0.mlp was dominant in component discovery)
    layer_0_features = [f for f in important_features if f.layer == 0]
    if layer_0_features:
        print(f"Layer 0 features (dominant layer): {len(layer_0_features)}")
        print("Top 10 Layer 0 features:")
        for i, feat in enumerate(layer_0_features[:10], 1):
            print(f"  {i:2d}. Feature {feat.feature_idx:4d}  "
                  f"Corr={feat.correlation_with_success:+.3f}  "
                  f"MeanAct={feat.mean_activation:.3f}")
        print()

else:
    print("WARNING: No important features found!")
    print("This might mean:")
    print("  - Correlation threshold too high")
    print("  - Not enough examples")
    print()

# ============================================================================
# STEP 6: SAVE RESULTS
# ============================================================================
print("=" * 80)
print("STEP 6: SAVE RESULTS")
print("=" * 80)
print()

# Convert to JSON-serializable format
from dataclasses import asdict

results = {
    "timestamp": datetime.now().isoformat(),
    "config": {
        "num_examples": num_test,
        "layers_analyzed": loaded_layers,
        "total_features": len(loaded_layers) * 6144,
        "top_k_per_layer": 20,
        "min_correlation": 0.2,
    },
    "discovered_features": [
        {
            "layer": f.layer,
            "feature_idx": f.feature_idx,
            "mean_activation": float(f.mean_activation),
            "activation_frequency": float(f.activation_frequency),
            "correlation_with_success": float(f.correlation_with_success),
        }
        for f in important_features
    ],
    "summary": {
        "total_important_features": len(important_features),
        "features_per_layer": {
            str(layer): sum(1 for f in important_features if f.layer == layer)
            for layer in loaded_layers
        },
        "elapsed_time_seconds": elapsed_time,
    }
}

output_path = Path("feature_circuit_discovery.json")
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved: {output_path}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("FEATURE DISCOVERY COMPLETE")
print("=" * 80)
print()
print("Results:")
print(f"  - Features analyzed: {len(loaded_layers) * 6144}")
print(f"  - Important features found: {len(important_features)}")
print(f"  - Layers with features: {len(layer_counts) if important_features else 0}")
print(f"  - Time elapsed: {elapsed_time:.1f}s")
print()
print("Next steps:")
print("  - Test causal importance via feature ablation")
print("  - Build multi-layer feature circuits")
print("  - Use for hierarchical steering")
print()
