# Phase 7: Task Boost Experiment

**Layer:** 10
**Date:** 2025-11-20 14:53

## Methodology
1. **Task Direction**: Learned via Ridge Regression on clean activations ($H$) to predict logit difference ($y$).
2. **Task Boost**: Injected $v_{task}$ into the residual stream: $h' = h + \alpha v_{task}$.
3. **Hard Subset**: Evaluated on test examples with low confidence (logit_diff < 1.5) or errors.

## Results

**Baseline Test Acc**: 97.60%
**Baseline Hard Acc**: 80.95% (N=63)

### Alpha Sweep

| Alpha | Test Acc | ΔAcc | Hard Acc | ΔHard |
|---|---|---|---|---|
| 0.5 | 97.60% | +0.00% | 80.95% | +0.00% |
| 1.0 | 97.60% | +0.00% | 80.95% | +0.00% |
| 2.0 | 97.60% | +0.00% | 80.95% | +0.00% |
| 3.0 | 97.60% | +0.00% | 80.95% | +0.00% |

### Virus Interaction (Alpha=2.0)

| Mode | Test Acc | Hard Acc |
|---|---|---|
| attack_only | 43.00% | 0.00% |
| boost_only | 97.60% | 80.95% |
| attack_plus_boost | 43.60% | 0.00% |
