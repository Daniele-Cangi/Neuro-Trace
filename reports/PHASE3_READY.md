# Phase 3: Ready to Launch

**Status**: All paths updated, infrastructure validated
**Date**: 2025-11-19
**Atlas**: 12/12 layers trained and validated (100% success)

---

## Updates Completed ✓

### 1. Script Path Updates

All existing discovery scripts have been updated with correct feature counts (3,072 → 6,144 per layer):

#### `discover_feature_circuits.py` ✓
- Line 32: Updated to "73,728 Atlas features" (was 36,864)
- Line 84: Updated to `len(loaded_layers) * 6144` (was 3072)
- Line 130: Updated to `6,144 features` in output
- Line 218: Updated config total_features to `6144`
- Line 257: Updated summary to `6144`

**Path verification**:
- ✓ SAE checkpoints: `checkpoints/all_layers_sae/layer_{i}/final.pt` (CORRECT)
- ✓ IOI dataset: Generated on-the-fly (no path needed)

#### `discover_real_circuits.py` ✓
- Line 4: Updated docstring to "73,728 Atlas features" (was 36,864)

**Path verification**:
- ✓ No hardcoded paths (uses component-level scanning, not feature-level)
- ✓ IOI dataset: Generated on-the-fly

#### `neurotrace/discovery/feature_circuit_discoverer.py` ✓
- Line 6: Updated docstring to "73,728 Atlas features" (was 36,864)
- Line 97: Updated init message to `6144` per layer
- Line 334: Updated discovery message to `6144` per layer

---

## Infrastructure Status

### Core Classes - All Ready ✓

1. **EnhancedSAEFeatureStore** (`neurotrace/control/enhanced_sae_feature_store.py`)
   - Purpose: Load and manage all 12 SAE layers
   - Status: Production-ready
   - Usage:
   ```python
   feature_store = EnhancedSAEFeatureStore()
   for layer in range(12):
       feature_store.load_sae(
           checkpoint_path=f"checkpoints/all_layers_sae/layer_{layer}/final.pt",
           layer=layer,
           device="cuda"
       )
   ```

2. **FeatureCircuitDiscoverer** (`neurotrace/discovery/feature_circuit_discoverer.py`)
   - Purpose: Discover which features drive IOI task
   - Status: Production-ready, paths updated
   - Usage:
   ```python
   discoverer = FeatureCircuitDiscoverer(feature_store, model, tokenizer, device="cuda")
   important_features = discoverer.discover_from_examples(
       examples=ioi_examples,
       top_k_per_layer=20,
       min_correlation=0.2
   )
   ```

3. **ExhaustiveCircuitScanner** (`neurotrace/discovery/exhaustive_scanner.py`)
   - Purpose: Component-level VLO testing
   - Status: Validated in Phase 1
   - Note: Component-level (156 components), not feature-level (73,728 features)

4. **VLOTester** (`neurotrace/causal/vlo_tester.py`)
   - Purpose: Compute causal importance via ablation
   - Status: Validated (found Layer 0 MLP VLO=5.276)

5. **CircuitRegistry** (`neurotrace/control/circuit_registry.py`)
   - Purpose: Store and retrieve discovered circuits
   - Status: Production-ready

---

## Ready to Run

### Option 1: Feature-Level Discovery (RECOMMENDED)

**Script**: `discover_feature_circuits.py`

**What it does**:
- Analyzes all 73,728 features across 12 layers
- Identifies which specific features activate on IOI examples
- Ranks features by correlation with task success
- Faster than component scanning (100 examples in ~2-3 minutes)

**Command**:
```bash
python discover_feature_circuits.py
```

**Expected Output**:
- `feature_circuit_discovery.json` with:
  - Feature activations per layer
  - Correlation scores
  - Activation frequencies
  - Top features per layer

**Estimated Runtime**: 2-3 minutes (100 IOI examples)

**Configuration** (edit lines 102-142 in script if needed):
```python
num_test = 100  # Number of IOI examples
top_k_per_layer=20  # Top features per layer
min_correlation=0.2  # Minimum correlation threshold
```

---

### Option 2: Component-Level VLO Testing

**Script**: `discover_real_circuits.py`

**What it does**:
- Exhaustive scanning of 156 components (144 attention heads + 12 MLPs)
- VLO testing via ablation
- Extracts validated circuits
- Slower but more comprehensive

**Command**:
```bash
python discover_real_circuits.py
```

**Expected Output**:
- `circuit_discovery_results.json` with VLO scores
- Circuit saved to `circuits/atlas_circuits.db`

**Estimated Runtime**: 15-30 minutes (200 IOI examples)

---

## Atlas Configuration (Locked)

All 12 SAE layers trained with:
- **dict_mult**: 8 (6,144 features per layer)
- **k_sparse**: 128 (2.1% sparsity)
- **Architecture**: EnhancedSAE with Top-K activation
- **Training data**: 100k examples from D:/NeuroTrace/20251118_123433/
- **Validation**: 12/12 layers passed (<10% accuracy loss)

**Checkpoint paths**:
```
checkpoints/all_layers_sae/layer_0/final.pt
checkpoints/all_layers_sae/layer_1/final.pt
...
checkpoints/all_layers_sae/layer_11/final.pt
```

---

## Next Steps After Discovery

### Immediate Analysis (Same Day)
1. Review `feature_circuit_discovery.json` output
2. Identify top IOI-critical features
3. Compare with Phase 1 findings (Layer 0 MLP dominance)

### Follow-up Work (Week 1-2)
1. **Feature Ablation**: Test causal importance of discovered features
2. **Cross-Layer Analysis**: Build feature dependency graph
3. **Circuit Validation**: Verify feature circuits with VLO testing
4. **Visualization**: Create interactive feature activation plots

### Future Phases
- **Phase 4**: Active steering via feature intervention
- **Phase 5**: Publication preparation

---

## Questions Before Running?

### Q: Should we increase num_test examples?
**A**: 100 examples is good for initial discovery. Can increase to 500-1000 for statistical robustness later.

### Q: Should we run both scripts?
**A**: Start with `discover_feature_circuits.py` (faster, feature-level). Run `discover_real_circuits.py` later for component-level comparison.

### Q: What if no features are found?
**A**: Lower `min_correlation` threshold from 0.2 to 0.1, or increase `num_test` examples.

### Q: Should we test on GPU?
**A**: Yes, script auto-detects CUDA. Will run on GPU if available (~2-3 min), CPU if not (~10-15 min).

---

## Summary

**Infrastructure**: ✓ All ready
**Paths**: ✓ All updated
**Atlas**: ✓ 12/12 validated
**Scripts**: ✓ 2 options ready to run
**Documentation**: ✓ Complete

**Recommendation**: Run `discover_feature_circuits.py` first.

**Command**:
```bash
python discover_feature_circuits.py
```

**Expected result**: JSON file with IOI-critical features ranked by correlation, ~2-3 minutes runtime.

---

## Changelog

- **2025-11-19**: All paths updated from 3,072 to 6,144 features per layer
- **2025-11-19**: Verified SAE checkpoint paths (checkpoints/all_layers_sae/)
- **2025-11-19**: Confirmed infrastructure classes ready
