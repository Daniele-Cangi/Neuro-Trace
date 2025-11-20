# Phase 6D: Centered Virus Defence

**Layer:** 10
**Date:** 2025-11-20 14:22

## Methodology
Unlike Phase 6B, this defence preserves the mean activation:
1. $h_{centered} = h - \mu$
2. $h_{clean} = (I - BB^T) h_{centered}$
3. $h_{final} = h_{clean} + \mu$

## Results

| Scenario | Train Acc | Test Acc | Train Diff | Test Diff |
|---|---|---|---|---|
| Baseline | 97.85% | 97.60% | 3.49 | 3.61 |
| Attack Only | 37.35% | 43.00% | -0.65 | -0.47 |
| Defence Only | 96.30% | 97.00% | 3.43 | 3.53 |
| Attack Plus Defence | 60.40% | 61.40% | 0.74 | 0.91 |

## Conclusion

**SUCCESS (Clean Performance)**: The centered defence preserves model performance on clean data.
**SUCCESS (Robustness)**: The defence significantly mitigates the attack.
