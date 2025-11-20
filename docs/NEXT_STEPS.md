# Next Steps - Post-Atlas Completion

**Status**: Atlas Complete (12/12 layers validated)
**Date**: November 19, 2025
**Ready for**: Phase 3 - Feature Analysis & Circuit Discovery

---

## Phase 3: Feature-Level Analysis (READY TO START)

Now that we have 12 validated SAEs (73,728 total features), we can proceed with feature-level interpretability.

### 3.1 Feature Discovery & Characterization

**Goal**: Understand what each of the 73,728 features represents

#### A. Automated Feature Analysis
```python
# For each layer's 6,144 features:
# 1. Top-activating examples
# 2. Semantic clustering
# 3. Task-specific importance
```

**Priority Layers**:
- Layer 0: Structural patterns (established from Phase 1)
- Layers 7-8: Highest performance improvement (+1.2%)
- Layer 11: Highest MSE (0.0837) - understand why

**Implementation**: `neurotrace/discovery/feature_analysis.py`

**Estimated Time**: 2-3 hours per layer (automated)

---

#### B. IOI-Specific Feature Identification

**Goal**: Find which features are crucial for the IOI task

**Method**:
1. Run IOI examples through each SAE
2. Track which features activate for correct predictions
3. Identify "IOI-critical features" (features that when ablated, drop accuracy)

**Expected Findings**:
- Name-position features (Layer 0-2)
- Syntactic structure features (Layer 3-5)
- Semantic binding features (Layer 6-8)
- Output mapping features (Layer 9-11)

**Implementation**: Extend `validate_reconstruction()` to track feature activations

---

### 3.2 Circuit Discovery with SAE Features

**Goal**: Build circuits using sparse features (not just components)

#### Previous Approach (Phase 1)
- Component-level circuits (entire MLP layers)
- Result: Layer 0 dominates (70% importance)

#### New Approach (Phase 3)
- **Feature-level circuits**: Specific features from different layers
- **Multi-layer paths**: Track information flow through features
- **Causal validation**: Ablate individual features

**Example Circuit**:
```
Layer 0, Feature 127 (name position detector)
  ↓
Layer 3, Feature 892 (syntactic role binder)
  ↓
Layer 7, Feature 2341 (semantic resolution)
  ↓
Layer 11, Feature 5876 (output selector)
```

**Implementation**: `neurotrace/discovery/feature_circuits.py`

---

### 3.3 Cross-Layer Feature Analysis

**Goal**: Understand how features compose across layers

**Questions to Answer**:
1. Which Layer 0 features feed into which Layer 3 features?
2. Are there "feature hierarchies" (low-level → high-level)?
3. Do some features act as "hubs" (many connections)?

**Method**: Activation correlation analysis
- Run 10K examples
- Track feature co-activation patterns
- Build feature dependency graph

**Visualization**: Interactive graph showing feature connections

---

## Phase 4: Active Steering & Control (AFTER PHASE 3)

Once we understand feature semantics, we can steer model behavior.

### 4.1 Feature Intervention

**Capability**: Modify specific features to change model output

**Use Cases**:
1. **Force correct IOI answer**: Amplify "correct indirect object" features
2. **Bias correction**: Suppress biased name-selection features
3. **Behavior editing**: Change model's decision-making path

**Implementation**: Already exists in `neurotrace/control/hierarchical_steering.py`

---

### 4.2 Circuit-Level Steering

**Goal**: Intervene on entire circuits (not just single features)

**Example**:
```python
# Disable "syntactic shortcut" circuit (Layer 0 → Layer 11 direct path)
# Force model to use "semantic understanding" circuit (Layer 0 → ... → Layer 7 → Layer 11)
```

**Expected Result**: Model performance drops but reasoning becomes more interpretable

---

## Phase 5: Publication & Generalization

### 5.1 Paper Preparation

**Findings to Report**:
1. **Phase 1**: Layer 0 structural dominance in small LMs
2. **Phase 2**: 100% SAE validation success with dict_mult=8
3. **Phase 3**: Complete feature-level IOI circuit map
4. **Phase 4**: Successful circuit-level steering

**Target Venues**:
- ICML 2026 (Interpretability Workshop)
- NeurIPS 2026 (Mechanistic Interpretability)
- ICLR 2027 (Main Conference)

---

### 5.2 Framework Generalization

**Current**: GPT-2 + IOI task
**Future**:
1. **Other models**: GPT-2 Medium, LLaMA-7B, Mistral-7B
2. **Other tasks**: Translation, Q&A, Reasoning
3. **Other architectures**: Vision Transformers, Diffusion Models

**Code Refactoring Needed**:
- Abstract model-specific code
- Generalize task evaluation
- Create model registry

---

## Immediate Action Items (Next Session)

### Priority 1: Feature Discovery (START HERE)

**Script**: `discover_ioi_features.py`

```python
# 1. Load all 12 SAEs
# 2. Run 1,000 IOI examples
# 3. For each feature:
#    - Track activation frequency
#    - Find top-activating examples
#    - Compute task correlation
# 4. Save feature database
```

**Output**: `feature_database.json` with 73,728 feature descriptions

**Estimated Time**: 3-4 hours (automated overnight)

---

### Priority 2: Build Feature-Level Circuits

**Script**: `build_feature_circuits.py`

```python
# 1. Identify "critical features" (high IOI correlation)
# 2. Run causal ablation on individual features
# 3. Build feature→feature dependency graph
# 4. Extract minimal circuits
```

**Output**: `feature_circuits.json` with validated feature paths

**Estimated Time**: 4-6 hours (includes validation)

---

### Priority 3: Validate Against Phase 1 Results

**Goal**: Reconcile component-level findings (Phase 1) with feature-level findings (Phase 3)

**Questions**:
- Does Layer 0's 70% importance come from specific features or distributed?
- Can we identify the exact "structural shortcut" features in Layer 0?
- Do later layers genuinely contribute less, or do they use fewer but critical features?

---

## Technical Debt & Cleanup

### Code Organization
- [ ] Delete deprecated scripts (old training attempts)
- [ ] Move old capture to archive: `runs/deep_ioi_capture/20251116_171258/`
- [ ] Update all hardcoded paths to use config file
- [ ] Add type hints to core functions

### Documentation
- [x] Update README with Phase 2 results
- [ ] Create `ATLAS_TRAINING.md` with detailed training guide
- [ ] Document SAE architecture decisions (why dict_mult=8, k=128)
- [ ] Write troubleshooting guide for common issues

### Testing
- [ ] Unit tests for SAE forward/backward pass
- [ ] Integration tests for feature discovery pipeline
- [ ] Regression tests for circuit validation
- [ ] Performance benchmarks

---

## Resources Needed

### Compute
- **Current**: Single GPU (sufficient for inference)
- **Phase 3**: Multi-GPU helpful for parallel feature analysis
- **Phase 4**: Same as current

### Storage
- **Current**: ~100GB (activation cache)
- **Phase 3**: +50GB (feature database, activation correlations)
- **Phase 4**: +10GB (steering experiments)

### Time Estimates
- **Feature Discovery**: 1 week (mostly automated)
- **Circuit Building**: 2 weeks (validation intensive)
- **Steering**: 1 week (experimental)
- **Paper Writing**: 4 weeks (iterative)

**Total to Publication**: ~8-10 weeks

---

## Success Criteria

### Phase 3 Complete When:
- [x] 73,728 features characterized (automated analysis)
- [ ] >100 IOI-critical features identified
- [ ] Feature circuits discovered and validated (<10% accuracy drop)
- [ ] Cross-layer feature graph visualized

### Phase 4 Complete When:
- [ ] Feature intervention working (can change IOI prediction)
- [ ] Circuit-level steering demonstrated
- [ ] Comparative analysis: component vs feature steering

### Publication Ready When:
- [ ] Novel findings clearly documented
- [ ] All experiments reproducible
- [ ] Code open-sourced and documented
- [ ] Figures publication-quality

---

## Questions for Discussion

1. **Feature Analysis Scope**: Analyze all 73K features or focus on high-variance ones?
2. **Circuit Complexity**: Target minimal circuits or comprehensive mappings?
3. **Generalization Priority**: GPT-2 Medium next, or different task on GPT-2 Small?
4. **Publication Strategy**: Workshop paper (faster) or main conference (higher impact)?

---

## Changelog

- **2025-11-19**: Document created after Atlas Phase 2 completion
- **Next Update**: After Phase 3 feature discovery begins
