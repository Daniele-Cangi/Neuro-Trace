# NeuroTrace - Discovery Validation Plan

**Date**: 2025-11-16
**Purpose**: Validate Layer 0 MLP dominance finding with larger dataset
**Status**: 🔄 **IN PROGRESS**

---

## Background

The initial automated discovery run (100 IOI examples) revealed a surprising finding:

**Layer 0 MLP is the dominant component** with:
- VLO = 1.874 (massively higher than any other component)
- Faithfulness = 4.433 (exceptionally high)
- 92% of total layer importance comes from Layer 0

This finding **contradicts published literature** on IOI circuits (Elhage et al.), which found:
- Name Mover Heads in layers 9-10 (not Layer 0)
- S-Inhibition Heads in layers 7-8
- Duplicate Token Heads in layer 0 (attention, not MLP)

---

## Hypothesis

**H0 (Null)**: Layer 0 MLP dominance is an artifact of small sample size (100 examples)

**H1 (Alternative)**: Layer 0 MLP dominance is a real, robust mechanism in GPT-2's IOI processing

---

## Validation Methodology

### Validation Run Specifications

| Parameter | Initial Run | Validation Run | Ratio |
|-----------|-------------|----------------|-------|
| **Dataset Size** | 100 examples | 1000 examples | 10x |
| **Bootstrap Samples** | 0 | 100 | Statistical validation |
| **Components Scanned** | 156 | 156 | Same |
| **VLO Threshold** | 0.3 | 0.3 | Same |
| **Faithfulness Threshold** | 0.2 | 0.2 | Same |
| **Model** | GPT-2 (124M) | GPT-2 (124M) | Same |
| **Device** | CUDA | CUDA | Same |
| **Template Diversity** | High (10+ templates) | High (10+ templates) | Same |

### Evaluation Criteria

**STRONG VALIDATION** (H1 confirmed):
- Layer 0 MLP VLO > 1.0 in validation run
- VLO change < ±30% from initial run
- Remains #1 ranked component
- Bootstrap confidence interval excludes 0
- Rank correlation (Spearman r) > 0.7

**MODERATE VALIDATION** (H1 likely):
- Layer 0 MLP VLO > 0.5
- VLO change < ±50%
- Remains in top 3 components
- Significant positive VLO (p < 0.05)

**WEAK/NO VALIDATION** (H0 confirmed):
- Layer 0 MLP VLO < 0.3
- VLO change > ±70%
- Ranking drops below top 10
- Confidence interval includes 0

---

## Possible Outcomes & Interpretations

### Outcome 1: Strong Validation (Most Likely)

**If Layer 0 MLP VLO ≈ 1.5-2.0 in validation run:**

**Interpretation**:
- This is a **real, robust mechanism** in GPT-2 IOI processing
- Not an artifact of sample size
- Potentially a **novel scientific discovery**

**Hypotheses for mechanism**:
1. **Early Name Detection**: Layer 0 MLP identifies proper nouns in context
2. **Model Size Effect**: Smaller GPT-2 (124M) uses different strategy than GPT-2-medium/large
3. **Template Generalization**: High diversity (10+ templates) reveals different circuit than single-template studies
4. **Positional Encoding**: Early layer processes positional information critical for name resolution

**Next steps**:
- ✅ Train SAE on Layer 0 MLP activations to identify specific features
- ✅ Visualize attention patterns in Layer 0
- ✅ Test on GPT-2-medium/large to check scaling behavior
- ✅ Ablation studies: Remove Layer 0 MLP → measure degradation
- ✅ Comparative analysis with single-template IOI dataset

---

### Outcome 2: Moderate Validation

**If Layer 0 MLP VLO ≈ 0.5-1.0:**

**Interpretation**:
- Layer 0 MLP is important but **effect size diminishes** with larger dataset
- May be partially real, partially artifact

**Next steps**:
- Investigate variance: Does VLO stabilize at certain dataset size?
- Component interaction analysis: Is Layer 0 synergistic with other components?
- Cross-validation: Run with 500, 2000, 5000 examples to find asymptote

---

### Outcome 3: Weak/No Validation

**If Layer 0 MLP VLO < 0.3:**

**Interpretation**:
- Finding was an **artifact of small sample size**
- GPT-2 IOI circuit likely matches published literature (layers 7-10)

**Next steps**:
- Re-analyze initial run: Was there selection bias in 100 examples?
- Identify true dominant components from validation run
- Focus on Layer 7-10 attention heads (as per literature)

---

## Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| **Dataset Generation** | ~30 seconds | ✅ **COMPLETE** |
| **Model Loading** | ~10 seconds | ✅ **COMPLETE** |
| **Tokenization** | ~5 seconds | ✅ **COMPLETE** |
| **Exhaustive Scan** | ~15-20 minutes | 🔄 **IN PROGRESS** |
| **Matrix Building** | ~5 seconds | ⏳ Pending |
| **Circuit Extraction** | ~5 seconds | ⏳ Pending |
| **Visualization** | ~10 seconds | ⏳ Pending |
| **Comparison Analysis** | ~5 seconds | ⏳ Pending |
| **Documentation Update** | ~5 minutes | ⏳ Pending |

**Estimated Total**: ~20-25 minutes

---

## Scan Progress

**Current Status**: Scanning 156 components (144 attention heads + 12 MLPs)

**Components Scanned**:
- Attention Heads (Layers 0-11, Heads 0-11): 0/144
- MLPs (Layers 0-11): 0/12

**Progress**: 0% (just started)

**Checkpoints**: Saved every 50 components to:
- `runs/discovery_validation/20251116_014719/checkpoints/scan_checkpoint_50.json`
- `runs/discovery_validation/20251116_014719/checkpoints/scan_checkpoint_100.json`
- `runs/discovery_validation/20251116_014719/checkpoints/scan_checkpoint_150.json`

---

## Comparative Analysis Plan

Once validation completes, we will run `compare_discovery_runs.py` which performs:

### 1. Layer 0 MLP Direct Comparison
- Initial VLO vs Validation VLO
- Absolute and percentage change
- Statistical significance (if bootstrap data available)

### 2. Component Ranking Analysis
- Top 10 components from each run
- Rank correlation (Spearman r)
- Stability metrics

### 3. Layer Importance Distribution
- Layer-wise aggregation of VLO
- Distribution shifts between runs
- Dominant layer identification

### 4. Statistical Validation
- Bootstrap confidence intervals (if available)
- P-values for top components
- Effect size stability

---

## Expected Files Generated

```
runs/discovery_validation/20251116_014719/
├── ioi_dataset.json                        ✅ Generated (1000 examples)
├── scan_results.json                       ⏳ Pending
├── interaction_matrix.json                 ⏳ Pending
├── circuits.db                             ⏳ Pending
├── checkpoints/
│   ├── scan_checkpoint_50.json            ⏳ Pending
│   ├── scan_checkpoint_100.json           ⏳ Pending
│   └── scan_checkpoint_150.json           ⏳ Pending
└── visualizations/
    ├── vlo_results.html                    ⏳ Pending
    ├── vlo_distribution.html               ⏳ Pending
    └── circuit_graph.html                  ⏳ Pending
```

---

## Scientific Implications

### If Validation Confirms Layer 0 MLP Dominance:

**Potential Contributions**:

1. **Novel Mechanism Discovery**
   - First documentation of early-layer MLP dominance in IOI task
   - Challenges existing understanding of IOI circuits
   - May apply to other name resolution tasks

2. **Model Size Effects**
   - Demonstrates circuit emergence varies with model scale
   - GPT-2 (124M) may use fundamentally different strategy than larger models
   - Important for mechanistic interpretability at different scales

3. **Template Generalization**
   - Shows importance of dataset diversity in circuit discovery
   - Single-template studies may miss critical mechanisms
   - Robust circuits emerge when tested on diverse prompts

4. **Methodological Advancement**
   - Validates exhaustive automated scanning approach
   - Demonstrates value of bootstrap confidence intervals
   - Establishes best practices for circuit discovery validation

**Publication Potential**:
- Conference paper: NeurIPS, ICML, ICLR (Mechanistic Interpretability track)
- Workshop: WANT (Workshop on Alignment of Neural Networks and Transformers)
- Preprint: arXiv with code/data release

---

## Risk Mitigation

**Risk 1**: Validation run fails due to memory/compute issues
- **Mitigation**: Checkpointing every 50 components allows resume
- **Fallback**: Run with 500 examples instead of 1000

**Risk 2**: Results are inconclusive (moderate validation)
- **Mitigation**: Run additional validation at 2000, 5000 examples
- **Analysis**: Plot VLO vs dataset size to find asymptotic behavior

**Risk 3**: Different dominant component emerges
- **Mitigation**: This is valid scientific outcome (H0 confirmed)
- **Action**: Pivot to analyzing actual dominant component

---

## Next Phase After Validation

Regardless of validation outcome, the next phase will be:

**Phase 2: Deep SAE Mapping of Dominant Component(s)**

If Layer 0 MLP validated:
1. Train SAE (16k dictionary) on Layer 0 MLP activations
2. Identify monosemantic features activated by IOI examples
3. Visualize top activating examples for each feature
4. Build feature→behavior mapping

If different component emerges:
1. SAE training on validated dominant components
2. Comparative analysis: Why did initial run mislead?
3. Dataset sensitivity analysis

---

## Validation Completion Checklist

- [ ] Scan completes successfully (156/156 components)
- [ ] No errors or crashes during execution
- [ ] All checkpoints saved correctly
- [ ] Scan results JSON valid and complete
- [ ] Circuit extracted and saved to registry
- [ ] Visualizations generated (3 HTML files)
- [ ] Comparison script run successfully
- [ ] Results documented in DISCOVERY_RESULTS.md
- [ ] Validation verdict determined (Strong/Moderate/Weak)
- [ ] Next steps planned based on outcome

---

**Generated**: 2025-11-16 01:47 UTC
**Runtime**: In progress (~0-5 minutes elapsed)
**Expected Completion**: 2025-11-16 02:05 UTC (~20 minutes total)
**Status**: 🔄 **SCANNING IN PROGRESS**
