# Phase 7B: Learned Task Boost Results

**Layer:** 10
**Date:** 2025-11-20 16:39

## Training Config
- **Hard Threshold**: Logit Diff < 1.0
- **Easy Margin**: 0.5
- **Learned Vector Norm**: 112.2552

## Results

| Mode | Test Acc | Δ Test | Hard Acc | Δ Hard |
|---|---|---|---|---|
| baseline | 97.60% | +0.00% | 70.00% | +0.00% |
| boost_only | 99.80% | +2.20% | 100.00% | +30.00% |
| attack_only | 43.00% | -54.60% | 0.00% | -70.00% |
| attack_plus_boost | 98.40% | +0.80% | 92.50% | +22.50% |
