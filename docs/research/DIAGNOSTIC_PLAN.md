# NeuroTrace Diagnostic Plan & Key Questions

**Date**: 2025-11-16
**Status**: INVESTIGATION IN PROGRESS
**Goal**: Understand what actually works vs what we think works

---

## Critical Questions to Answer

### 1. Are we intercepting the network 1:1?
**Status**: UNKNOWN - Never formally tested

**What we need to verify**:
- [ ] Activation capture is deterministic (same input → same activations)
- [ ] No numerical corruption during hook operations
- [ ] Gradient flow is preserved (if needed for future work)
- [ ] Activations match across multiple runs (variance < 1e-6)

**Test Method**:
- Run same prompt twice
- Capture activations via hooks both times
- Compare with torch.allclose()
- Measure max absolute difference

**Expected Results**:
- PASS: max_diff < 1e-6 (numerical precision only)
- WARNING: max_diff < 1e-4 (acceptable but investigate)
- FAIL: max_diff > 1e-4 (serious problem)

---

### 2. Do our SAEs actually reconstruct well?
**Status**: PARTIALLY KNOWN - Only tested during training, not in deployment

**What we know**:
- Layer 0 SAE training MSE: ~0.0124 (EXCELLENT)
- But: Never tested reconstruction on NEW data
- But: Never compared reconstruction vs original model outputs

**What we need to verify**:
- [ ] Reconstruction MSE on held-out test set
- [ ] Reconstruction quality per layer (all 12 layers)
- [ ] Does reconstruction preserve semantic meaning?
- [ ] L0 sparsity matches expected (k=64 for top-k)

**Test Method**:
1. Load real activations from validation set
2. Encode → Decode through SAE
3. Measure MSE: `mean((original - reconstructed)^2)`
4. Compare to publication standards (target: <0.05)

**Expected Results**:
- PASS: MSE < 0.05 (publication quality)
- WARNING: 0.05 < MSE < 0.12 (acceptable)
- FAIL: MSE > 0.12 (insufficient quality)

---

### 3. Does steering actually change outputs?
**Status**: DEMONSTRATED BUT NOT QUANTIFIED

**What we know**:
- Demo shows text changes with steering enabled
- But: No quantitative measurement of effect size
- But: No statistical significance testing
- But: Only tested on 3 prompts

**What we need to verify**:
- [ ] Steering effect is consistent across many prompts
- [ ] Effect size is measurable and significant
- [ ] Different alpha values produce different effects
- [ ] Steering is reversible (disable → returns to baseline)

**Test Method**:
1. Generate 50+ prompts (baseline)
2. Generate same prompts with steering (alpha=1.0)
3. Measure differences:
   - Text similarity (edit distance, BLEU, etc.)
   - Logit differences
   - Activation differences at target layer
4. Statistical test: paired t-test for significance

**Expected Results**:
- PASS: >80% of prompts show measurable change, p < 0.01
- WARNING: 50-80% show change
- FAIL: <50% show change (steering not working)

---

### 4. Is the Neural Atlas functional?
**Status**: EXISTS BUT NEVER USED

**What we have**:
- Layer 0 SAE trained (3,072 features)
- Layers 1-11: NO SAEs trained
- No feature labeling system
- No cross-layer comparison tools
- No search/query interface

**What we need to verify**:
- [ ] Can we load and inspect Layer 0 features?
- [ ] Can we identify top features by activation frequency?
- [ ] Can we find similar features (cosine similarity)?
- [ ] Does the atlas provide interpretable insights?

**Test Method**:
1. Load Layer 0 SAE
2. For each feature:
   - Get decoder direction
   - Find top activating examples
   - Label feature (manual or automated)
3. Build feature similarity matrix
4. Test search: "find features related to [concept]"

**Expected Results**:
- PASS: Features are interpretable, search works
- WARNING: Some features interpretable, search is noisy
- FAIL: Features are incomprehensible, search fails

---

### 5. Can we systematically analyze our data?
**Status**: DATA EXISTS BUT NO ANALYSIS PIPELINE

**Current Problem**:
- JSON files scattered everywhere
- No aggregation or comparison
- No trend analysis
- No visualization

**What we need**:
- [ ] JSON aggregator (collect all results)
- [ ] Systematic comparison tool
- [ ] Trend analysis (how do metrics evolve?)
- [ ] Automated reporting

**Required Tools**:
1. `neurotrace/analysis/json_aggregator.py` - collect all JSONs
2. `neurotrace/analysis/comparative_analysis.py` - compare results
3. `neurotrace/analysis/report_generator.py` - generate summaries

---

## Diagnostic Test Suite

**Location**: `tests/validation/test_system_diagnostic.py`

**Tests**:
1. `test_activation_capture_fidelity()` - Q1: 1:1 interception?
2. `test_sae_reconstruction_quality()` - Q2: SAE quality?
3. `test_steering_causality()` - Q3: Steering works?
4. `test_atlas_navigation()` - Q4: Atlas functional?
5. `test_systematic_data_analysis()` - Q5: Can we analyze data?

**Outputs**:
- `diagnostic_report.json` - machine-readable results
- Console output - human-readable summary
- Pass/Warning/Fail status for each test
- Score (0.0-1.0) for each component
- Critical issues and recommendations

---

## Expected Findings

### Optimistic Scenario:
- Activation capture: PASS (1:1 fidelity)
- SAE reconstruction: PASS (MSE < 0.05)
- Steering causality: PASS (80%+ effect rate)
- Neural Atlas: WARNING (only Layer 0 exists)
- Data analysis: WARNING (no tools yet)

**Overall**: NEEDS_ATTENTION (Atlas incomplete, tools missing)

### Realistic Scenario:
- Activation capture: PASS
- SAE reconstruction: WARNING (MSE 0.05-0.12, acceptable but not SOTA)
- Steering causality: WARNING (50-80% effect rate)
- Neural Atlas: FAIL (missing 11 layers)
- Data analysis: FAIL (no systematic tools)

**Overall**: CRITICAL (major gaps in testing and infrastructure)

### Pessimistic Scenario:
- Activation capture: WARNING (small numerical issues)
- SAE reconstruction: FAIL (MSE > 0.12, poor quality)
- Steering causality: FAIL (<50% effect, not working reliably)
- Neural Atlas: FAIL
- Data analysis: FAIL

**Overall**: CRITICAL (fundamental issues, needs rework)

---

## Action Items Based on Results

### If Optimistic:
1. Complete Neural Atlas (train layers 1-11)
2. Build data analysis tools
3. Add web interface
4. Write paper

### If Realistic:
1. **PRIORITY**: Fix SAE quality (re-train with better hyperparams)
2. **PRIORITY**: Improve steering reliability (better feature selection)
3. Complete Neural Atlas
4. Build systematic testing
5. Only then: web interface

### If Pessimistic:
1. **CRITICAL**: Debug activation capture (investigate numerical issues)
2. **CRITICAL**: Complete SAE rework (architecture, training, data)
3. **CRITICAL**: Validate steering mechanism (are hooks correct?)
4. Put Atlas and interface on hold
5. Focus on fundamental correctness first

---

## Next Steps After Diagnostic

### Immediate (based on results):
1. Review diagnostic_report.json
2. Identify CRITICAL issues (score < 0.5)
3. Fix critical issues before proceeding
4. Re-run diagnostic to verify fixes

### Short-term:
1. Implement missing tools (JSON aggregator, etc.)
2. Complete Neural Atlas (train all layers)
3. Build feature labeling system
4. Create Atlas query interface

### Medium-term:
1. Web UI for exploration
2. 3D visualization
3. Advanced steering (multi-circuit, adaptive)
4. Publication preparation

---

## Key Metrics to Track

### System Health:
- Activation fidelity: max_diff < 1e-6
- SAE reconstruction: MSE < 0.05 per layer
- Steering effect rate: >80%
- Atlas coverage: 12/12 layers
- Test coverage: >80% unit tests

### Research Quality:
- Feature interpretability: >90% features labeled
- Circuit validity: VLO correlation with ground truth
- Steering precision: controllable effect size
- Cross-layer consistency: feature evolution tracked

---

## Open Research Questions (Post-Diagnostic)

1. **Layer-wise feature evolution**: How do features transform from structural (Layer 0) to semantic (Layer 9)?
2. **Circuit composability**: Can we combine multiple circuits without interference?
3. **Generalization**: Do circuits transfer to other models (GPT-2 Medium, Large)?
4. **Scaling**: Does our method work on larger models (GPT-J, Llama)?
5. **Real-world tasks**: Beyond IOI, what can we control?

---

## Document Updates After Diagnostic

**Files to update based on results**:
- `README.md` - update status, be honest about limitations
- `PRODUCTION_ROADMAP.md` - prioritize based on findings
- `FINAL_RESULTS.md` - add diagnostic results, caveats
- `docs/research/LIMITATIONS.md` (new) - document what doesn't work

**Principle**: Scientific honesty over hype. We report what we found, not what we hoped to find.

---

**Status**: Diagnostic running... awaiting results.
