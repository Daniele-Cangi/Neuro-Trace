# Available Tools - Feature Discovery & Circuit Analysis

**Last Updated**: 2025-11-19
**Status**: Post-Atlas (12/12 SAEs trained and validated)

This document catalogs existing scripts and infrastructure for Phase 3 (Feature Discovery).

---

## Ready-to-Use Scripts

### 1. `discover_feature_circuits.py` ✅

**Purpose**: Discover which specific features (out of 73,728) drive IOI task

**What it does**:
- Loads all 12 SAE layers
- Runs IOI examples through model + SAEs
- Tracks feature activation patterns
- Identifies task-critical features via correlation

**Status**: EXISTS - Ready to run with updated SAE paths

**Usage**:
```bash
python discover_feature_circuits.py
```

**Expected Output**:
- Feature activation database
- IOI-critical features list
- Feature importance rankings

---

### 2. `discover_real_circuits.py` ✅

**Purpose**: VLO-validated circuit discovery (component-level)

**What it does**:
- Exhaustive component scanning
- VLO (Value of Learned Organization) testing
- Circuit extraction and validation

**Status**: EXISTS - Uses ExhaustiveCircuitScanner

**Note**: This is component-level (entire MLPs), not feature-level

---

## Core Infrastructure Classes

### 3. `FeatureCircuitDiscoverer` ✅

**Location**: `neurotrace/discovery/feature_circuit_discoverer.py`

**Purpose**: Feature-level circuit discovery engine

**Key Methods**:
- `discover_from_examples()` - Analyze which features activate
- `compute_feature_importance()` - Correlation with task success
- `test_feature_ablation()` - Causal validation

**Status**: IMPLEMENTED - Class exists and is ready

**Integration**: Used by `discover_feature_circuits.py`

---

### 4. `EnhancedSAEFeatureStore` ✅

**Location**: `neurotrace/control/enhanced_sae_feature_store.py`

**Purpose**: Load and manage all 12 SAE layers

**Key Methods**:
```python
feature_store = EnhancedSAEFeatureStore()

# Load all layers
for layer in range(12):
    feature_store.load_sae(
        checkpoint_path=f"checkpoints/all_layers_sae/layer_{layer}/final.pt",
        layer=layer,
        device="cuda"
    )

# Extract features from activations
features = feature_store.extract_features(
    layer=0,
    activations=hidden_states
)
```

**Status**: PRODUCTION-READY

---

### 5. `SAEFeatureExtractor` ✅

**Location**: `neurotrace/state_indexer/sae_feature_extractor.py`

**Purpose**: Basic SAE operations (encode/decode)

**Note**: Uses simpler SAE architecture (ReLU-based)
- Our Atlas uses **EnhancedSAE** (Top-K based)
- Use `EnhancedSAEFeatureStore` instead for Atlas features

---

## Supporting Infrastructure

### 6. `CircuitRegistry` ✅

**Location**: `neurotrace/control/circuit_registry.py`

**Purpose**: Store and retrieve discovered circuits

**Methods**:
- `register_circuit()` - Save discovered circuit
- `get_circuit()` - Load circuit by name
- `list_circuits()` - Enumerate all circuits

**Storage Format**: JSON files in `circuits/` directory

---

### 7. `ExhaustiveCircuitScanner` ✅

**Location**: `neurotrace/discovery/exhaustive_scanner.py`

**Purpose**: Component-level circuit scanning

**Status**: Production-ready, used in Phase 1

**Note**: Works on components (MLP layers), not individual features

---

### 8. `VLOTester` ✅

**Location**: `neurotrace/causal/vlo_tester.py`

**Purpose**: Compute Value of Learned Organization metric

**What it does**: Measures causal importance via ablation

**Status**: Validated in Phase 1 (found Layer 0 MLP VLO=5.276)

---

## What We Need to Update

### Path Updates Required

All scripts reference old SAE paths. Need to update:

**Old paths** (deprecated):
```python
"checkpoints/layer0_sae/final.pt"
"runs/deep_ioi_capture/20251116_171258/"
```

**New paths** (current):
```python
"checkpoints/all_layers_sae/layer_{i}/final.pt"
"D:/NeuroTrace/20251118_123433/"
```

**Files to update**:
1. `discover_feature_circuits.py` - SAE checkpoint paths
2. `discover_real_circuits.py` - SAE checkpoint paths
3. Any hardcoded paths in discovery classes

---

## Quick Start Guide

### Phase 3: Feature Discovery (Use Existing Tools)

#### Step 1: Update Paths in `discover_feature_circuits.py`

```python
# Line ~60-70: Update SAE loading
SAE_CHECKPOINT_DIR = Path("checkpoints/all_layers_sae")
IOI_DATASET = Path("D:/NeuroTrace/20251118_123433/ioi_dataset.json")
```

#### Step 2: Run Feature Discovery

```bash
python discover_feature_circuits.py
```

**Expected Runtime**: 2-3 hours (analyzing 73,728 features on 1000+ examples)

**Output Files**:
- `feature_activations.json` - Raw activation data
- `feature_importance.json` - Ranked feature list
- `ioi_critical_features.json` - Top features for IOI task

#### Step 3: Analyze Results

The script will identify:
- Which features activate most frequently on correct IOI predictions
- Feature correlation with task success
- Cross-layer feature patterns

---

## Feature Analysis Pipeline (Existing Code)

```python
# 1. Load feature store
feature_store = EnhancedSAEFeatureStore()
for layer in range(12):
    feature_store.load_sae(
        checkpoint_path=f"checkpoints/all_layers_sae/layer_{layer}/final.pt",
        layer=layer,
        device="cuda"
    )

# 2. Initialize discoverer
discoverer = FeatureCircuitDiscoverer(
    feature_store=feature_store,
    model=model,
    tokenizer=tokenizer,
    device="cuda"
)

# 3. Discover features
feature_importances = discoverer.discover_from_examples(
    examples=ioi_examples,
    top_k_per_layer=50,
    min_correlation=0.3
)

# 4. Extract circuits
circuits = discoverer.extract_circuits(
    feature_importances,
    min_path_length=2
)
```

This code **already exists** in the infrastructure!

---

## What's Missing vs What Exists

### ✅ Already Implemented:
1. Feature extraction from SAEs
2. Activation pattern analysis
3. Correlation computation
4. Feature ablation testing
5. Circuit extraction logic
6. Circuit storage (CircuitRegistry)

### ❌ Not Yet Implemented:
1. **Feature visualization** (top-activating examples per feature)
2. **Cross-layer feature graph** (which features feed into which)
3. **Feature clustering** (semantic grouping)
4. **Minimal circuit search** (smallest set of features for task)

### 🔧 Needs Update:
1. Paths to Atlas SAEs (old → new)
2. Feature count (3,072 → 6,144 per layer)
3. Sparsity parameter (k=64 → k=128)

---

## Recommended Next Action

**DO NOT write new scripts!** We have everything we need.

### Immediate Steps:

1. **Update paths** in `discover_feature_circuits.py`:
   - SAE checkpoints → `checkpoints/all_layers_sae/`
   - IOI dataset → `D:/NeuroTrace/20251118_123433/ioi_dataset.json`

2. **Run existing script**:
   ```bash
   python discover_feature_circuits.py
   ```

3. **Analyze output**:
   - Review `feature_importance.json`
   - Identify top IOI-critical features
   - Compare with Phase 1 component-level results

### Follow-up (After Discovery):

4. **Validate circuits** using existing `VLOTester`
5. **Store circuits** using existing `CircuitRegistry`
6. **Visualize** (only missing piece - can add to existing infrastructure)

---

## Summary

**We have 90% of the infrastructure ready!**

- Feature discovery: ✅ `FeatureCircuitDiscoverer`
- SAE management: ✅ `EnhancedSAEFeatureStore`
- Circuit validation: ✅ `VLOTester`
- Circuit storage: ✅ `CircuitRegistry`

**What we need to do**:
1. Update 2-3 file paths in existing scripts
2. Run `discover_feature_circuits.py`
3. Analyze results

**Estimated time**: 1-2 hours to update paths + 2-3 hours runtime = **Done in one day**.

No need to write new code from scratch!
