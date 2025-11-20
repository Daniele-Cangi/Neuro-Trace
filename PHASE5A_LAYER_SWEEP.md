# Phase 5A: Layer Vulnerability Sweep

**Date**: 2025-11-20
**Total Time**: 952.9s (15.9 min)

## Overview

Trained adversarial steering vectors for all 12 layers using Phase 4B method.
Each layer was optimized independently to minimize IOI logit difference on borderline examples.

## Configuration

- **Training examples**: 2000
- **Test examples**: 500
- **Borderline threshold**: 1.5
- **Epochs per layer**: 20
- **Learning rate**: 0.01
- **L2 regularization**: λ=0.001

## Baseline Performance

- **Train**: 3.492 (97.9% accuracy)
- **Test**: 3.474 (97.2% accuracy)

## Layer Vulnerability Results

### Summary Table

| Rank | Layer | ||δ|| | Train Δacc | Test Δacc | Train Δdiff | Test Δdiff |
|------|-------|--------|------------|-----------|-------------|------------|
|  1 |  0 | 23.94 | -97.8% | -97.2% | -28.758 | -28.543 |
|  2 |  1 | 24.36 | -97.8% | -97.2% | -26.543 | -26.337 |
|  3 |  2 | 25.76 | -97.8% | -97.2% | -31.007 | -30.491 |
|  4 |  3 | 26.12 | -97.9% | -97.2% | -25.824 | -25.522 |
|  5 |  4 | 27.40 | -97.9% | -97.2% | -21.818 | -21.613 |
|  6 |  5 | 28.68 | -97.9% | -97.2% | -33.346 | -32.669 |
|  7 |  6 | 31.16 | -97.9% | -97.2% | -35.213 | -34.560 |
|  8 |  7 | 34.52 | -97.9% | -97.2% | -21.555 | -21.332 |
|  9 |  8 | 35.71 | -95.3% | -95.8% | -13.798 | -13.837 |
| 10 |  9 | 34.44 | -91.5% | -92.2% | -9.200 | -9.272 |
| 11 | 10 | 29.65 | -60.5% | -60.2% | -4.139 | -4.171 |
| 12 | 11 | 10.15 | -0.8% | -0.4% | -0.024 | -0.023 |

### Detailed Results by Layer

#### Layer 0

- **Delta norm**: 23.940
- **Training time**: 67.5s

**Train Set**:
- Baseline: 3.492 (97.9%)
- Steered: -25.266 (0.1%)
- Effect: -28.758
- Accuracy change: -97.8%

**Test Set**:
- Baseline: 3.474 (97.2%)
- Steered: -25.069 (0.0%)
- Effect: -28.543
- Accuracy change: -97.2%

#### Layer 1

- **Delta norm**: 24.362
- **Training time**: 76.0s

**Train Set**:
- Baseline: 3.492 (97.9%)
- Steered: -23.051 (0.1%)
- Effect: -26.543
- Accuracy change: -97.8%

**Test Set**:
- Baseline: 3.474 (97.2%)
- Steered: -22.863 (0.0%)
- Effect: -26.337
- Accuracy change: -97.2%

#### Layer 2

- **Delta norm**: 25.760
- **Training time**: 69.9s

**Train Set**:
- Baseline: 3.492 (97.9%)
- Steered: -27.515 (0.1%)
- Effect: -31.007
- Accuracy change: -97.8%

**Test Set**:
- Baseline: 3.474 (97.2%)
- Steered: -27.017 (0.0%)
- Effect: -30.491
- Accuracy change: -97.2%

#### Layer 3

- **Delta norm**: 26.118
- **Training time**: 70.5s

**Train Set**:
- Baseline: 3.492 (97.9%)
- Steered: -22.332 (0.0%)
- Effect: -25.824
- Accuracy change: -97.9%

**Test Set**:
- Baseline: 3.474 (97.2%)
- Steered: -22.048 (0.0%)
- Effect: -25.522
- Accuracy change: -97.2%

#### Layer 4

- **Delta norm**: 27.397
- **Training time**: 71.6s

**Train Set**:
- Baseline: 3.492 (97.9%)
- Steered: -18.326 (0.0%)
- Effect: -21.818
- Accuracy change: -97.9%

**Test Set**:
- Baseline: 3.474 (97.2%)
- Steered: -18.139 (0.0%)
- Effect: -21.613
- Accuracy change: -97.2%

#### Layer 5

- **Delta norm**: 28.676
- **Training time**: 71.1s

**Train Set**:
- Baseline: 3.492 (97.9%)
- Steered: -29.854 (0.0%)
- Effect: -33.346
- Accuracy change: -97.9%

**Test Set**:
- Baseline: 3.474 (97.2%)
- Steered: -29.195 (0.0%)
- Effect: -32.669
- Accuracy change: -97.2%

#### Layer 6

- **Delta norm**: 31.163
- **Training time**: 71.1s

**Train Set**:
- Baseline: 3.492 (97.9%)
- Steered: -31.720 (0.0%)
- Effect: -35.213
- Accuracy change: -97.9%

**Test Set**:
- Baseline: 3.474 (97.2%)
- Steered: -31.086 (0.0%)
- Effect: -34.560
- Accuracy change: -97.2%

#### Layer 7

- **Delta norm**: 34.516
- **Training time**: 77.5s

**Train Set**:
- Baseline: 3.492 (97.9%)
- Steered: -18.063 (0.0%)
- Effect: -21.555
- Accuracy change: -97.9%

**Test Set**:
- Baseline: 3.474 (97.2%)
- Steered: -17.858 (0.0%)
- Effect: -21.332
- Accuracy change: -97.2%

#### Layer 8

- **Delta norm**: 35.713
- **Training time**: 72.4s

**Train Set**:
- Baseline: 3.492 (97.9%)
- Steered: -10.306 (2.5%)
- Effect: -13.798
- Accuracy change: -95.3%

**Test Set**:
- Baseline: 3.474 (97.2%)
- Steered: -10.363 (1.4%)
- Effect: -13.837
- Accuracy change: -95.8%

#### Layer 9

- **Delta norm**: 34.438
- **Training time**: 71.0s

**Train Set**:
- Baseline: 3.492 (97.9%)
- Steered: -5.707 (6.3%)
- Effect: -9.200
- Accuracy change: -91.5%

**Test Set**:
- Baseline: 3.474 (97.2%)
- Steered: -5.798 (5.0%)
- Effect: -9.272
- Accuracy change: -92.2%

#### Layer 10

- **Delta norm**: 29.654
- **Training time**: 71.2s

**Train Set**:
- Baseline: 3.492 (97.9%)
- Steered: -0.647 (37.4%)
- Effect: -4.139
- Accuracy change: -60.5%

**Test Set**:
- Baseline: 3.474 (97.2%)
- Steered: -0.697 (37.0%)
- Effect: -4.171
- Accuracy change: -60.2%

#### Layer 11

- **Delta norm**: 10.154
- **Training time**: 71.6s

**Train Set**:
- Baseline: 3.492 (97.9%)
- Steered: 3.468 (97.0%)
- Effect: -0.024
- Accuracy change: -0.8%

**Test Set**:
- Baseline: 3.474 (97.2%)
- Steered: 3.451 (96.8%)
- Effect: -0.023
- Accuracy change: -0.4%

## Analysis

### Most Vulnerable Layer: **Layer 0**

- Test accuracy drop: **-97.2%**
- Test effect: **-28.543**
- Delta norm: 23.940

### Least Vulnerable Layer: **Layer 11**

- Test accuracy drop: **-0.4%**
- Test effect: **-0.023**
- Delta norm: 10.154

### Statistics

- Mean test accuracy change: -85.5% ± 27.6%
- Range: [-97.2%, -0.4%]
- Vulnerability spread: 96.8%

### Layer Groups

- **Early layers (0-3)**: -97.2% average accuracy drop
- **Mid layers (4-7)**: -97.2% average accuracy drop
- **Late layers (8-11)**: -62.1% average accuracy drop

## Key Findings

- **Mid layers (4-7) are MOST vulnerable** to adversarial steering

- Correlation between layer depth and vulnerability: r=0.630 (p=0.028)
  - **Significant positive correlation**: Shallower layers are MORE vulnerable

## Implications

1. **Layer-specific vulnerability** varies significantly across the network
2. **Vulnerability range**: 96.8% spread between most/least vulnerable layers
3. **Strategic targeting**: Adversarial attacks should focus on most vulnerable layers
4. **Defensive priorities**: Protect most vulnerable layers with countermeasures

## Checkpoints

Adversarial deltas saved for all layers:

- `checkpoints/adversarial_delta_layer0.pt`
- `checkpoints/adversarial_delta_layer1.pt`
- `checkpoints/adversarial_delta_layer2.pt`
- `checkpoints/adversarial_delta_layer3.pt`
- `checkpoints/adversarial_delta_layer4.pt`
- `checkpoints/adversarial_delta_layer5.pt`
- `checkpoints/adversarial_delta_layer6.pt`
- `checkpoints/adversarial_delta_layer7.pt`
- `checkpoints/adversarial_delta_layer8.pt`
- `checkpoints/adversarial_delta_layer9.pt`
- `checkpoints/adversarial_delta_layer10.pt`
- `checkpoints/adversarial_delta_layer11.pt`

---

**Next Steps**: Multi-layer delta combinations, defensive steering, layer-wise analysis