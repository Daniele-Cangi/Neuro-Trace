# Phase 6C: Task vs Virus Subspace Comparison

**Layer:** 10
**Date:** 2025-11-20 14:11

## 1. Alignment Metrics

- **Cosine Similarity (Task vs Virus PC1):** 0.0060
- **Angle:** 89.66°
- **Energy of Task Direction in Virus Subspace (4D):** 0.0087
- **Energy of MEAN Clean Activation in Virus Subspace:** 0.8847

## 2. Subspace Overlap (Principal Angles)

| Rank | Angle (deg) | Cosine |
|---|---|---|
| 1 | 47.74° | 0.6725 |
| 2 | 84.58° | 0.0944 |
| 3 | 85.80° | 0.0732 |
| 4 | 89.30° | 0.0123 |

## 3. Interpretation

**CRITICAL FINDING**: The Virus Subspace captures the **MEAN** clean activation (Energy > 0.8). This explains why projecting it out destroys performance: it removes the 'DC component' or average state of the residual stream.
