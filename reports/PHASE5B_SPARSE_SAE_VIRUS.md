# Phase 5B: Sparse SAE Virus

**Date**: 2025-11-20
**Total Time**: 559.0s (9.3 min)

## Overview

Learned adversarial steering vectors **directly in SAE feature space** (α coefficients)
with L1 regularization to enforce sparsity. This tests whether sparse, interpretable
feature combinations can achieve the same adversarial power as dense residual vectors.

## Configuration

- **Target layer**: 10
- **Training examples**: 2000
- **Test examples**: 500
- **Borderline threshold**: 1.5
- **Epochs**: 20
- **Learning rate**: 0.01
- **L2 regularization**: λ₂=0.0001
- **L1 values tested**: [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]

## Baseline Performance

- **Train**: 3.492 (97.9% accuracy)
- **Test**: 3.474 (97.2% accuracy)

## Sparsity-Performance Tradeoff

### Summary Table

| λ₁ | #Active | % Active | ‖δ‖ | ‖α‖₁ | Test Δacc | Test Δdiff |
|---------|---------|----------|--------|--------|-----------|------------|
| 0e+00 |    6137 |    99.9% | 204.857 | 2926.8 |     -96.6% |    -31.486 |
| 1e-04 |    6132 |    99.8% | 203.827 | 2886.8 |     -96.6% |    -31.511 |
| 5e-04 |    6134 |    99.8% | 202.295 | 2807.6 |     -96.6% |    -31.522 |
| 1e-03 |    6129 |    99.8% | 200.515 | 2699.2 |     -96.8% |    -31.630 |
| 5e-03 |    5961 |    97.0% | 188.026 | 1929.4 |     -96.2% |    -29.812 |
| 1e-02 |    3718 |    60.5% |  1.467 |   13.3 |      +0.0% |     -0.057 |
| 5e-02 |    3281 |    53.4% |  0.145 |    8.0 |      +0.0% |     -0.004 |

### Detailed Results

#### λ₁ = 0e+00

- **Active features**: 6137 (99.9%)
- **Delta norm**: 204.857
- **Alpha L1**: 2926.8
- **Training time**: 73.8s

**Train Set**:
- Baseline: 3.492 (97.9%)
- Steered: -28.150 (0.4%)
- Effect: -31.642
- Accuracy change: -97.5%

**Test Set**:
- Baseline: 3.474 (97.2%)
- Steered: -28.012 (0.6%)
- Effect: -31.486
- Accuracy change: -96.6%

#### λ₁ = 1e-04

- **Active features**: 6132 (99.8%)
- **Delta norm**: 203.827
- **Alpha L1**: 2886.8
- **Training time**: 74.1s

**Train Set**:
- Baseline: 3.492 (97.9%)
- Steered: -28.183 (0.4%)
- Effect: -31.675
- Accuracy change: -97.4%

**Test Set**:
- Baseline: 3.474 (97.2%)
- Steered: -28.037 (0.6%)
- Effect: -31.511
- Accuracy change: -96.6%

#### λ₁ = 5e-04

- **Active features**: 6134 (99.8%)
- **Delta norm**: 202.295
- **Alpha L1**: 2807.6
- **Training time**: 70.3s

**Train Set**:
- Baseline: 3.492 (97.9%)
- Steered: -28.226 (0.4%)
- Effect: -31.719
- Accuracy change: -97.5%

**Test Set**:
- Baseline: 3.474 (97.2%)
- Steered: -28.048 (0.6%)
- Effect: -31.522
- Accuracy change: -96.6%

#### λ₁ = 1e-03

- **Active features**: 6129 (99.8%)
- **Delta norm**: 200.515
- **Alpha L1**: 2699.2
- **Training time**: 71.5s

**Train Set**:
- Baseline: 3.492 (97.9%)
- Steered: -28.319 (0.4%)
- Effect: -31.811
- Accuracy change: -97.5%

**Test Set**:
- Baseline: 3.474 (97.2%)
- Steered: -28.156 (0.4%)
- Effect: -31.630
- Accuracy change: -96.8%

#### λ₁ = 5e-03

- **Active features**: 5961 (97.0%)
- **Delta norm**: 188.026
- **Alpha L1**: 1929.4
- **Training time**: 72.7s

**Train Set**:
- Baseline: 3.492 (97.9%)
- Steered: -26.539 (0.9%)
- Effect: -30.031
- Accuracy change: -97.0%

**Test Set**:
- Baseline: 3.474 (97.2%)
- Steered: -26.338 (1.0%)
- Effect: -29.812
- Accuracy change: -96.2%

#### λ₁ = 1e-02

- **Active features**: 3718 (60.5%)
- **Delta norm**: 1.467
- **Alpha L1**: 13.3
- **Training time**: 72.6s

**Train Set**:
- Baseline: 3.492 (97.9%)
- Steered: 3.434 (97.8%)
- Effect: -0.058
- Accuracy change: -0.1%

**Test Set**:
- Baseline: 3.474 (97.2%)
- Steered: 3.417 (97.2%)
- Effect: -0.057
- Accuracy change: +0.0%

#### λ₁ = 5e-02

- **Active features**: 3281 (53.4%)
- **Delta norm**: 0.145
- **Alpha L1**: 8.0
- **Training time**: 71.8s

**Train Set**:
- Baseline: 3.492 (97.9%)
- Steered: 3.489 (97.9%)
- Effect: -0.004
- Accuracy change: +0.0%

**Test Set**:
- Baseline: 3.474 (97.2%)
- Steered: 3.470 (97.2%)
- Effect: -0.004
- Accuracy change: +0.0%

## Analysis

### Best Overall Performance: λ₁=1e-03

- **Test accuracy drop**: -96.8%
- **Test effect**: -31.630
- **Active features**: 6129 (99.8%)
- **Delta norm**: 200.515

### Sparsest Strong Attack: λ₁=5e-03

- **Test accuracy drop**: -96.2%
- **Active features**: 5961 (97.0%)
- **Delta norm**: 188.026

### ❌ Ultra-Sparse Attack Not Achieved

**No setting achieved ≤-40% drop with <100 features**


### Sparsity Statistics

- **Accuracy drop range**: [-96.8%, 0.0%]
- **Active features range**: [3281, 6137]

- **Correlation** (# features vs accuracy drop): r=-0.994 (p=0.000)
  - Significant negative correlation: More features → More attack power

## Key Findings

1. **⚠️ Strong attacks require moderate sparsity**: Best result uses 5961 features (97.0%)
2. **Gap between sparse and dense**: Ultra-sparse (<100 features) insufficient for strong control

## Comparison with Phase 4B

| Method | Space | Features/Dims | Effect | Accuracy Δ |
|--------|-------|---------------|--------|------------|
| Phase 4B | Residual δ | 768 dims | -4.180 | -60.4% |
| Phase 4B-B | Top-200 SAE | 200 features | -0.389 | -1.4% |
| **Phase 5B (best)** | **SAE α (L1)** | **6129 features** | **-31.630** | **-96.8%** |

**Phase 5B achieves comparable performance to Phase 4B** by directly optimizing in SAE space with sparsity constraints.

## Implications

1. **Sparsity-performance tradeoff exists**: Stronger attacks require more features
2. **Interpretability limit**: Cannot achieve Phase 4B power with ultra-sparse features
3. **Dense mechanisms dominate**: IOI control fundamentally requires distributed features

## Checkpoints

Sparse SAE virus checkpoints saved:

- `checkpoints/sparse_sae_virus_layer10_l10e+00.pt`
- `checkpoints/sparse_sae_virus_layer10_l11e-04.pt`
- `checkpoints/sparse_sae_virus_layer10_l15e-04.pt`
- `checkpoints/sparse_sae_virus_layer10_l11e-03.pt`
- `checkpoints/sparse_sae_virus_layer10_l15e-03.pt`
- `checkpoints/sparse_sae_virus_layer10_l11e-02.pt`
- `checkpoints/sparse_sae_virus_layer10_l15e-02.pt`

---

**Next Steps**: Analyze top features in best sparse model, multi-layer sparse attacks, defensive steering