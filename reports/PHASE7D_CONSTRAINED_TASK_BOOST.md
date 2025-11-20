# Phase 7D: Constrained Task Boost Results

**Layer:** 10
**Date:** 2025-11-20 17:09

## Training Config
- **Norm Constraint (R_TARGET)**: 25.0
- **Final Vector Norm**: 25.0000

## Results

| Mode | Test Acc | Δ Test | Hard Acc | Δ Hard |
|---|---|---|---|---|
| baseline | 97.60% | +0.00% | 70.00% | +0.00% |
| boost_only | 99.40% | +1.80% | 92.50% | +22.50% |
| attack_only | 43.00% | -54.60% | 0.00% | -70.00% |
| attack_plus_boost | 84.40% | -13.20% | 52.50% | -17.50% |

## Conclusion
**PARTIAL**: Constrained boost offers partial protection (84.40%).
