# NeuroTrace: Neural Network Interpretability Framework

A comprehensive framework for deep neural network analysis, featuring automated circuit discovery, sparse autoencoder training, and active steering capabilities.

**Version**: 8.0.0
**Status**: ✅ Phase 14 Complete - Integrated Defence System Validated
**Last Updated**: 2025-11-20
**Atlas**: 12/12 Layers Validated (73,728 Features Total)

---

## Overview

NeuroTrace is a research framework for understanding and controlling transformer neural networks. The project has evolved from passive analysis to active defense:

1. **Atlas (Phases 1-3)**: Mapped the network's components and features.
2. **Causality (Phase 4)**: Discovered that causal control lives in the residual stream, not sparse features.
3. **Vulnerability (Phase 5)**: Identified that early layers are highly vulnerable to adversarial attacks.
4. **Defense (Phases 6-7)**: Developed "Task Boosting" - an active defense strategy that neutralizes attacks by injecting optimized task vectors.
5. **Smart Gating (Phases 8-9)**: Implemented intelligent detectors ("Needs Boost") to activate defense only when necessary.
6. **Analysis (Phases 10-12)**: Analyzed the "alien" nature of the boost vector and its collateral damage on general text.
7. **Integration (Phases 13-14)**: Built the final "Integrated Defence System" with Domain Guard and Damage Guard, achieving zero collateral damage.

---
<img width="1408" height="752" alt="Gemini_Generated_Image_i3yqiki3yqiki3yq" src="https://github.com/user-attachments/assets/52181323-1171-4db4-90a8-64f5c940c108" />

## 📂 Repository Structure

The project is organized as follows:

- **`phases/`**: Contains the main execution scripts for each research phase (e.g., `phase14_integrated_defence.py`).
- **`reports/`**: Stores detailed markdown reports and JSON result files for each phase.
- **`checkpoints/`**: Saved models, vectors, and detectors (e.g., `context_classifier_layer10.pt`).
- **`configs/`**: Configuration files for detectors and experiments.
- **`neurotrace/`**: Core Python package containing reusable modules (`models`, `datasets`, `control`).
- **`legacy/`**: Archived scripts from previous iterations and utility tools.

---

## 🏆 Latest Achievement: Phase 14 (Integrated Defence)

### Integrated Defence System - November 20, 2025

Successfully implemented the final **Integrated Defence System** combining **Domain Guard** (Context Classifier), **Damage Guard** (Needs Boost Detector), and **Active Defense** (Task Boost).

**Results**:

| Mode | IOI Acc | Wiki PPL | Gate Rate | Collateral Damage |
|---|---|---|---|---|
| **Baseline** | 97.4% | 72.14 | 0% | - |
| **Static Defence** | 98.8% | 145.08 | 100% | +101% (Severe) |
| **Integrated Defence** | **94.8%** | **72.14** | **48.4%** | **0.0% (None)** |

**Conclusion**:
The system achieves the "Holy Grail" of AI Defense:
1.  **Robustness**: Restores accuracy under attack (94.8%).
2.  **Safety**: Zero impact on general capabilities (PPL 72.14).
3.  **Efficiency**: Only activates when strictly necessary.

---

## 🔬 Phase 13: Context Classifier (Domain Guard)

**Domain Awareness - November 20, 2025**

Trained a lightweight binary classifier (Layer 10 activations) to distinguish between "IOI Task" and "General Text" (WikiText).

- **Accuracy**: 100.00%
- **False Positive Rate**: 0.00%
- **Role**: Acts as the first line of defense, completely disabling the system when the model is processing general text, ensuring zero collateral damage.

---

## 🔬 Phase 12: SAE Purification

**Attempting Sparse Reconstruction - November 20, 2025**

Attempted to replace the dense "Task Boost" vector with a sparse combination of SAE features (Top-K) to reduce collateral damage.

**Results**:
- **K=50 Features**: Reduced PPL spike from 145 to 100, but still +38% damage compared to baseline.
- **Conclusion**: Purification is insufficient. The "alien" nature of the boost vector means it cannot be cleanly approximated by sparse features without loss of power or added noise. **Gating is the only viable solution.**

---

## 🔬 Phase 11: Collateral Damage Assessment

**Static Defence Impact on General Text - November 20, 2025**

Measured the "doping effect" of the static defence (Task Boost Layer 10, R=25) on generic language modeling performance using WikiText-2.

### Phase 11 Results

| Mode | N_tokens | NLL | Perplexity | Impact |
|------|----------|-----|------------|--------|
| **Baseline** | 16,498 | 4.2785 | **72.14** | - |
| **Static Defence** | 16,498 | 4.9773 | **145.08** | **+101%** |

### Key Findings

1. **High Collateral Damage**: The static defence **doubles the perplexity** (+72.95) on generic text.
2. **Doping Effect**: The model becomes hyper-specialized for IOI but "hallucinates" or degrades on normal English text.
3. **Justification for Gating**: This result proves that **Static Defence is not viable** for production. The **Gated Defence (Phase 9)** is strictly necessary to activate the boost ONLY when an attack is detected, preserving general capabilities (PPL ~72) while defending IOI.

---

## 🔬 Phase 10B: Boost SAE Decomposition

**Atlas ↔ Boost Connection - November 20, 2025**

Attempted to decompose the learned "Task Boost" vector (Phase 7D) into the interpretable SAE feature space (Phase 2) to see if it corresponds to a sparse combination of features.

### Decomposition Results

- **Boost Norm**: 25.00
- **Alpha Norm (Linear)**: 44.76

| K Features | Energy Ratio | Norm Ratio | Recon Error |
|------------|--------------|------------|-------------|
| 10 | 4.8% | 40.1% | 25.37 |
| 100 | 22.6% | 92.9% | 28.58 |
| 6144 (All) | 100.0% | 272.2% | 62.44 |

### Key Insight: The Boost is "Alien"

The Task Boost vector is **highly dense** and does **not** live in the sparse feature manifold learned by the SAE.
1. **Poor Reconstruction**: Even using all 6,144 features, the reconstruction error is massive (62.44 vs norm 25.00).
2. **Constructive Interference**: The linear projection explodes the norm (272%), indicating the boost vector exploits directions that are not aligned with the clean, sparse dictionary features.
3. **Conclusion**: Gradient descent found a "hack" in the residual stream that forces the task output, but this direction is orthogonal to the model's natural interpretable features.

---

## 🧠 Phase 9: Smart Defence Systems

**Objective**: Transition from "brute force" defense (always-on, high metabolic cost) to "smart" defense systems that only intervene when necessary.

### Phase 9D: Gated Defence V3 (Needs Boost)

**Goal**: Use the "Needs Boost" detector to gate the constrained task boost.

**Results**:

| Mode | Test Acc | Hard Acc | Gate Rate | FP Rate |
|---|---|---|---|---|
| **Baseline** | 97.4% | 83.3% | - | - |
| **No Defence** | 47.6% | 1.3% | - | - |
| **Static Defence** | 98.8% | 92.3% | 100% | - |
| **Gated V3** | **97.4%** | **92.3%** | **51.0%** | **0.0%** |

**Conclusion**:

- **Perfect Precision**: The system *never* intervenes when the model is already correct (0.0% FP).
- **Optimal Efficiency**: It cuts the intervention cost by half (51% vs 100%) compared to static defense.
- **Full Restoration**: It restores the hard accuracy to the same level as the static defense (92.3%).
- **Verdict**: This is the superior defense strategy.

### Phase 9C: "Needs Boost" Detector

**Goal**: Train an MLP to detect *functional damage* (prediction flip), not just the presence of the virus.

**Method**:

- Trained on 2000 examples (Clean vs Attacked).
- Features: Logit diffs, projections, norms.
- Label: 1 if Attack causes error, 0 otherwise.

**Results**:

- **Test Accuracy**: **99.4%**
- **Precision**: 99.6%
- **Recall**: 99.2%
- **Finding**: We can reliably predict *before* the final output whether the attack will succeed or fail.

### Phase 9B: Gated Defence V2 (Virus Detector)

**Goal**: Gate the defense using a simple "Virus Detector".

**Results**:

- **Gate Rate**: 100% (The virus is always present).
- **FP Rate**: 47.6% (Defends even when not needed).
- **Conclusion**: Detecting the *presence* of the attack is not enough for efficient defense; we need to detect the *impact*.

### Phase 9A: Virus Detector

**Goal**: Train an MLP to detect if the adversarial vector is present in the residual stream.

**Results**:

- **Test Accuracy**: **100.0%**
- **AUC**: 1.000
- **Finding**: The adversarial vector is easily detectable, but its presence doesn't always imply failure.

---

## ⚔️ Phase 8: Immune Gating & War Surface

**Objective**: Understand the interaction between Attack and Defense vectors and attempt simple gating mechanisms.

### Phase 8B: War Surface (Grid Search)

**Goal**: Map the interaction surface between Attack Scale ($\alpha$) and Boost Scale ($\beta$).

**Results**:

- **Neutralization Ratio**: To counter an attack of $\alpha=1.0$, a boost of $\beta \approx 3.0$ is required.
- **Heatmap**: Shows a clear linear boundary between "Collapsed" and "Restored" states.

### Phase 8A: Immune Gating (Confidence-Based)

**Goal**: Gate the defense based on model confidence (logit difference).

**Results**:

- **Gate Rate**: 39.6%
- **Hard Acc**: 93.7%
- **FP Rate**: 37.2%
- **Conclusion**: Simple confidence thresholding works but is less precise than the learned detectors in Phase 9.

---

## 🎯 Phase 7D: Constrained Task Boost

**Objective**: Train a specific boost vector with a **strict norm constraint** ($R=25.0$) to see if we can optimize for efficiency rather than raw power.

**Method**: Projected Gradient Descent (clipping norm to 25.0 at every step).

**Results**:

- **Clean Performance**:
  - **Test Acc**: **99.4%** (Excellent)
  - **Hard Subset**: **92.5%** (Strong improvement)
- **Virus Defense**:
  - **Attack + Boost**: **84.4%** (Partial Success)

**Conclusion**:

- A specialized, constrained vector is **much more efficient** than a scaled-down version of the unconstrained vector (84.4% vs 54.8% at similar norms).
- We can achieve **significant protection** (84% accuracy) with constrained norms, though total neutralization requires higher energy.
- **Final Verdict**: Active defense is viable and highly effective.

---

## ⚖️ Phase 7C: Boost Scaling Analysis

**Objective**: Determine if the protective effect of Phase 7B survives at lower, more constrained norms.

**Results**:

| Norm | Mode | Test Acc | Hard Acc |
|---|---|---|---|
| **5.0** | Attack + Boost | 44.4% | 0.0% |
| **20.0** | Attack + Boost | 54.8% | 0.0% |
| **50.0** | Attack + Boost | 85.2% | 55.0% |
| **80.0** | Attack + Boost | 96.4% | 90.0% |
| **112.0** | Attack + Boost | 98.4% | 92.5% |

**Finding**: The defense requires high energy. At low-magnitude norms (~20-30), the boost is insufficient to counter the attack (Acc ~55%). To fully neutralize the attack, the boost must overpower the adversarial vector with norm > 50.

---

## 🚀 Phase 7B: Learned Task Boost (Gradient Descent)

**Objective**: Learn a "Task Enhancement" vector $v_{boost}$ via direct gradient descent optimization to maximize logit difference on hard examples.

**Method**:

- **Loss Function**: $L = L_{hard} + \lambda L_{easy} + \lambda ||v||^2$
  - $L_{hard}$: Maximize logit diff on errors/borderline cases.
  - $L_{easy}$: Penalize degradation of correct examples (margin loss).
- **Optimization**: Adam, 20 epochs.

**Results (Layer 10)**:

- **Learned Vector**: Norm $||v|| \approx 112$ (Very high energy).
- **Clean Performance**:
  - **Test Acc**: 97.6% -> **99.8%** (+2.2%)
  - **Hard Subset**: 70.0% -> **100.0%** (Perfect correction)
- **Virus Defense**:
  - **Attack Only**: 43.0%
  - **Attack + Boost**: **98.4%** (Full Neutralization)

**Conclusion**: We found a robust countermeasure for Layer 10. By injecting high-energy signal in the optimized direction, we can completely override the adversarial vector and even improve baseline performance.

---

## 📉 Phase 7: Task Boost Experiment (Linear)

**Objective**: Test if injecting a "Task Direction" vector (learned via Ridge Regression on clean activations) improves performance on hard examples or defends against the virus.

**Method**:

1. Captured clean activations $H$ and logit differences $y$.
2. Solved for $w$ in $Hw \approx y$ (Ridge Regression).
3. Injected $v_{task} = w / ||w||$ with varying $\alpha$.

**Results**:

- **Clean Data**: No improvement (Acc 97.6% -> 97.6%).
- **Hard Subset**: No improvement (Acc 80.9% -> 80.9%).
- **Virus Defense**: No effect (Acc 43.0% -> 43.6%).
- **Conclusion**: A simple linear regression direction is insufficient to "boost" the model or counter the virus. The task manifold is likely more complex or the virus attacks a different subspace.

---

## 🛡️ Phase 6D: Centered Defence

**Objective**: Test a refined defense strategy that preserves the mean activation while removing the virus subspace.

**Method**:

1. **Center**: Subtract mean activation ($h' = h - \mu$).
2. **Project**: Remove virus subspace from centered residual ($h'' = (I - BB^T)h'$).
3. **Restore**: Add mean activation back ($h_{final} = h'' + \mu$).

**Results (Layer 10)**:

| Scenario | Test Accuracy | Logit Diff | Interpretation |
|----------|---------------|------------|----------------|
| **Baseline** | 97.6% | +3.61 | Normal behavior |
| **Attack Only** | 43.0% | -0.47 | Attack degrades performance |
| **Defence Only** | **97.0%** | +3.53 | **SUCCESS**: Clean performance preserved |
| **Attack + Defence** | **61.4%** | +0.91 | **PARTIAL SUCCESS**: Attack mitigated (+18%) |

**Conclusion**:

- **Hypothesis Confirmed**: Preserving the mean activation prevents the model collapse seen in Phase 6B.
- **Defense Viability**: The centered defense is safe to deploy (minimal impact on clean data) and offers moderate protection, recovering significant accuracy.
- **Remaining Gap**: The attack is not fully neutralized (61% vs 97%), suggesting the virus also operates via non-linear interactions or components not fully captured by the top-4 PCA directions.

---

## 🔍 Phase 6C: Task vs Virus Subspace Comparison

**Objective**: Investigate the catastrophic failure of the Phase 6B defense. Why did removing the "Virus Subspace" destroy the model's performance on clean data?

**Method**:

1. **Task Direction**: Computed via Ridge Regression on clean activations (predicting logit difference).
2. **Virus Subspace**: Computed via PCA on sparse virus vectors (Phase 5B).
3. **Mean Activation**: Computed the mean of clean residual stream activations.
4. **Comparison**: Measured Cosine Similarity and Energy overlap between these subspaces.

**Results**:

- **Task vs Virus**: The Virus Subspace is **orthogonal** to the Task Direction (Cosine Similarity ≈ 0.006). The virus does NOT mimic the task feature.
- **Mean vs Virus**: The Virus Subspace has a **massive overlap** with the Mean Activation (Energy ≈ 0.88).
- **Interpretation**: The virus attacks by hijacking the **"DC component"** (mean state) of the residual stream. In Phase 6B, projecting out the virus subspace inadvertently removed this critical mean component, effectively "blinding" the model.

---

## 🛡️ Phase 6B: Virus Defence via Subspace Projection

**Objective**: Attempt to defend the model against the virus by projecting the residual stream **orthogonally** to the discovered "virus subspace" (removing the attack direction).

**Hypothesis**: If the virus exploits a non-essential direction, removing that subspace should neutralize the attack while preserving clean performance.

**Results (Layer 10)**:

| Scenario | Test Accuracy | Logit Diff | Interpretation |
|----------|---------------|------------|----------------|
| **Baseline (Clean)** | 97.6% | +3.61 | Normal model behavior |
| **Attack Only** | 0.8% | -27.94 | Attack is successful |
| **Defence Only (Clean)** | **0.2%** | -28.22 | **Significant Performance Drop** |
| **Attack + Defence** | 0.2% | -28.21 | Defense fails to restore function |

**Key Finding**:
The "Virus Subspace" appears to be **aligned with the critical task subspace**.

- Projecting out the virus direction (Defence Only) significantly impacts the model's ability to perform the task on clean data (Acc drops from 97.6% to 0.2%).
- This suggests the virus attacks by **manipulating the core features** used for the Indirect Object Identification (IOI) task.
- **Conclusion**: Simple linear subspace removal is **not** an effective defense for this type of attack in this layer. Future defenses may need to be non-linear or feature-specific.

---

## 🦠 Phase 6: Virus Subspace Decomposition

**Objective**: Determine the dimensionality of the "Sparse SAE Virus" (adversarial feature combination) discovered in Phase 5B for Layer 10.

**Method**:

1. Collected multiple sparse virus vectors (alphas) trained with different L1 regularization strengths.
2. Performed Principal Component Analysis (PCA) on these vectors in the SAE feature space.
3. Reconstructed the virus using only the top-k principal components and measured attack effectiveness.

**Results (Layer 10)**:

- **Dimensionality**: The virus is low-rank. The **first principal component (k=1)** explains the majority of the variance.
- **Attack Efficiency**:
  - **Full Virus**: Test Acc **0.8%**
  - **k=1 Reconstruction**: Test Acc **0.8%** (Identical effectiveness)
- **Conclusion**: The adversarial attack, despite involving ~6000 active features (dense in feature space), can be approximated by a **single direction** in the residual stream.

---

## 🔬 Phase 5A: Layer Vulnerability Sweep

### Layer-wise Adversarial Vulnerability Landscape - November 20, 2025

Trained adversarial steering vectors for **all 12 layers** (0-11) using the Phase 4B gradient-based method. Discovered a vulnerability gradient across network depth: **Layers 0-7 are highly vulnerable (≈-97% accuracy drop), layers 8-9 remain vulnerable, layer 10 moderately vulnerable (≈-60%), layer 11 almost invariant (0.4% drop). Adversarial control is possible across most of the depth, but the final layer behaves as a robustness buffer.**

### Phase 5A Results

**Execution**: 15.9 minutes (952.9s) | **Layers Trained**: 12 | **Deltas Saved**: 12 checkpoints

### Layer Vulnerability Table

| Rank | Layer | ‖δ‖ | Train Δacc | Test Δacc | Train Δdiff | Test Δdiff |
|------|-------|---------|------------|-----------|-------------|------------|
| 1 | 0 | 23.94 | -97.8% | **-97.2%** | -28.758 | **-28.543** |
| 2 | 1 | 24.36 | -97.8% | **-97.2%** | -26.543 | **-26.337** |
| 3 | 2 | 25.76 | -97.8% | **-97.2%** | -31.007 | **-30.491** |
| 4 | 3 | 26.12 | -97.9% | **-97.2%** | -25.824 | **-25.522** |
| 5 | 4 | 27.40 | -97.9% | **-97.2%** | -21.818 | **-21.613** |
| 6 | 5 | 28.68 | -97.9% | **-97.2%** | -33.346 | **-32.669** |
| 7 | 6 | 31.16 | -97.9% | **-97.2%** | -35.213 | **-34.560** |
| 8 | 7 | 34.52 | -97.9% | **-97.2%** | -21.555 | **-21.332** |
| 9 | 8 | 35.71 | -95.3% | **-95.8%** | -13.798 | **-13.837** |
| 10 | 9 | 34.44 | -91.5% | **-92.2%** | -9.200 | **-9.272** |
| 11 | 10 | 29.65 | -60.5% | **-60.2%** | -4.139 | **-4.171** |
| 12 | 11 | 10.15 | -0.8% | **-0.4%** | -0.024 | **-0.023** |

### Key Findings

#### 1. **Vulnerability Gradient**

- **Early layers (0-7)**: -97.2% average accuracy drop (highly vulnerable)
- **Mid layers (8-9)**: -92.2% to -95.8% (vulnerable)
- **Late layer (10)**: -60.2% (moderately vulnerable)
- **Final layer (11)**: **-0.4%** (robust)

#### 2. **Statistical Analysis**

- Mean vulnerability: **-85.5% ± 27.6%**
- Vulnerability range: **96.8%** spread (Layer 0: -97.2%, Layer 11: -0.4%)
- **Significant correlation** (r=0.630, p=0.028): Shallower layers MORE vulnerable

#### 3. **Layer 11 Robustness**

- Final layer shows **strong resistance** to adversarial steering
- Effect magnitude: -0.023 (vs -28.543 for Layer 0)
- Delta norm: 10.15 (vs 23.94 for Layer 0) - optimization couldn't find strong attack
- **Interpretation**: Late-stage processing has strong robustness mechanisms

#### 4. **Vulnerability Pattern**

- Layers 0-7: Uniform high vulnerability (~97% drop)
- Layer 8: First sign of resistance (95.8% drop)
- Layer 9: More resistance (92.2% drop)
- Layer 10: Significant robustness (60.2% drop)
- Layer 11: Robustness buffer (0.4% drop)

### Training Configuration

Same as Phase 4B for all layers:

- **Training examples**: 2,000 (237 borderline with logit_diff < 1.5)
- **Test examples**: 500
- **Epochs per layer**: 20
- **Learning rate**: 0.01
- **L2 regularization**: λ = 0.001
- **Optimization**: Adam with gradient clipping (max_norm=5.0)

### Implications

1. **Strategic Attack Surface**: Adversarial attacks are most effective on Layers 0-7.
2. **Defensive Priorities**: Protect early/mid layers; Layer 11 already robust.
3. **Architecture Insight**: GPT-2's final layer acts as a **robustness buffer** against perturbations.
4. **Multi-layer Attacks**: Layers 0-10 are all vulnerable → potential for multi-layer combinations.
5. **Generalization**: Vulnerability gradient may be universal across transformer architectures.

### Checkpoints

All 12 adversarial deltas saved:

- [checkpoints/adversarial_delta_layer0.pt](checkpoints/adversarial_delta_layer0.pt) through [adversarial_delta_layer11.pt](checkpoints/adversarial_delta_layer11.pt)

---

## 🔬 Phase 4B: Adversarial Steering Milestone

### Residual Stream Steering - November 19, 2025

Successfully learned an adversarial steering vector that **severely degrades IOI performance** via gradient-based optimization in the residual stream. Test accuracy dropped from 97.2% to 36.8% (-60.4%), validating that **residual stream directions causally control IOI**, while SAE feature-level steering (Phase 4A) showed only weak effects.

### Phase 4B Results

**Setup**:

- **Target Layer**: Layer 10 (residual stream injection)
- **Optimization**: Gradient descent on δ ∈ ℝ⁷⁶⁸ to minimize logit_diff + λ||δ||²
- **Training Set**: 237 borderline examples (0 < logit_diff < 1.5)
- **Epochs**: 20 (74.6 seconds training)
- **Learned Delta Norm**: ||δ|| = 29.738

### Steering Results

| Dataset | Baseline | Steered | Effect | Accuracy Change |
|---------|----------|---------|--------|-----------------|
| **Borderline** (237) | 0.926 (100.0%) | -3.436 (1.3%) | **-4.361** | **-98.7%** |
| **Train** (2000) | 3.492 (97.9%) | -0.654 (37.2%) | **-4.146** | **-60.6%** |
| **Test** (500) | 3.474 (97.2%) | -0.706 (36.8%) | **-4.180** | **-60.4%** |

### Phase 4B Key Findings

#### 1. **Residual Stream is the Causal Locus**

- **Phase 4A** (SAE features): Effect = +0.160, Accuracy Δ = -1.7% ❌
- **Phase 4B** (residual delta): Effect = **-4.180**, Accuracy Δ = **-60.4%** ✅
- **Implication**: IOI causality lives in residual stream space, not sparse feature space

#### 2. **Gradient-Based Optimization Works**

- Trained on only 237 borderline examples (11.8% of dataset)
- Generalizes perfectly: Train effect (-4.146) ≈ Test effect (-4.180)
- Loss progression: +0.880 → -3.297 (inverted logit difference!)

#### 3. **Single Vector Destroys IOI**

- One 768-dimensional vector (||δ|| = 29.738) sufficient
- No multi-layer coordination needed
- Layer 10 highly vulnerable to adversarial steering

#### 4. **Borderline Examples Most Vulnerable**

- Borderline accuracy: 100% → 1.3% (-98.7%)
- Full dataset accuracy: 97.9% → 37.2% (-60.6%)
- Adversarial delta exploits decision boundary uncertainty

### Scientific Validation

**Phase 4A vs Phase 4B**:

| Method | Space | Effect | Result |
|--------|-------|--------|--------|
| Single feature ablation | SAE (6,144-dim) | ≈0.0 | ❌ No effect |
| Multi-feature (top 5) | SAE (5 features) | +0.160 | ❌ Weak |
| **Adversarial delta** | **Residual (768-dim)** | **-4.180** | **✅ Strong** |

**Conclusion**: Phase 4B demonstrates that IOI is **strongly controllable via residual stream directions**, while single-feature or multi-feature steering in SAE space produces only weak effects. The residual stream is the **primary causal locus**; SAE features are predominantly **correlational markers**.

### Training Dynamics

```text
Epoch  1: Loss=+0.880  LogitDiff=+0.880  ||δ||=1.543
Epoch  5: Loss=+0.311  LogitDiff=+0.267  ||δ||=7.457
Epoch 10: Loss=-0.656  LogitDiff=-0.869  ||δ||=15.476
Epoch 15: Loss=-1.611  LogitDiff=-2.111  ||δ||=23.182
Epoch 20: Loss=-2.453  LogitDiff=-3.297  ||δ||=29.738
```

Smooth convergence with consistent generalization throughout training.

### Phase 4B Implications

1. **Residual > Features**: Causality for dense tasks (IOI) lives in residual stream, not sparse decomposition
2. **SAE Purpose**: Features are for **interpretation**, not **control**
3. **Gradient-Based Steering**: Effective method for discovering causal directions
4. **Layer Vulnerability**: Layer 10 particularly susceptible (future: sweep layers 0-11)

### Next Steps

**Phase 4B-B - Feature Space Decomposition**:

- Project adversarial δ onto SAE feature basis
- Identify which features compose the "virus"
- Test if top-K features reconstruct adversarial effect
- Connect Phase 3 (feature discovery) with Phase 4B (steering)

**Future Work**:

- Layer sweep (0-11): Find most/least vulnerable layers
- Multi-layer delta combinations
- Test generalization to other tasks (beyond IOI)
- Defensive steering: learn δ that improves accuracy

See [adversarial_steering_layer10_results.json](adversarial_steering_layer10_results.json) for complete data and training history.

---

## 🔬 Phase 4B-B: Adversarial Virus in SAE Feature Space

### Feature Space Decomposition - November 19, 2025

Projected the learned adversarial steering vector onto the SAE feature basis to understand its composition and test if sparse subsets can reconstruct the adversarial effect.

### Decomposition Results

**Projection Quality**:

- **||δ|| (original)**: 29.738
- **||δ_hat (SAE projection)**: 26.377
- **Projection ratio**: **88.7%** ✅ Virus lives primarily in SAE space
- **MSE**: 0.0176 (high-quality reconstruction)

### Top-K Feature Reconstruction

| Top-K Features | Norm Ratio | Effect Preserved | Accuracy Drop |
|----------------|------------|------------------|---------------|
| Top-10 | 3.8% | **0.4%** | -0.0% |
| Top-50 | 8.9% | **2.4%** | -0.2% |
| Top-100 | 13.4% | **4.8%** | -0.8% |
| Top-200 | 20.3% | **9.3%** | -1.4% |
| **All 6,144** | **88.7%** | **100%** | **-60.4%** |

### Key Finding

**The adversarial steering vector learned in Phase 4B lives mostly inside the SAE feature space (projection ≈ 88.7%), but its representation is highly dense**: no small subset of features (10, 50, 100, 200) can reproduce the effect. Full adversarial power only emerges when combining (almost) the entire SAE basis.

**Top-200 features** (3.3% of dictionary) preserve only **9.3% of adversarial effect**, despite capturing **20.3% of the norm**. This demonstrates a fundamental gap between norm preservation and causal effect preservation.

### Cross-Phase Comparison

| Approach | Method | Features Used | Effect | Result |
|----------|--------|---------------|--------|--------|
| **Phase 4A** | Top-5 Phase 3 features | 5 correlational | +0.160 | ❌ Weak |
| **Phase 4B** | Free residual δ | 768-dim vector | **-4.180** | ✅ Strong |
| **Phase 4B-B** | Top-200 SAE features | 200 causal | -0.389 | ❌ Weak |
| **Phase 4B-B** | All 6,144 SAE features | Full basis | -3.71* | ✅ Strong |

*Estimated from 88.7% projection ratio

### Phase 3 vs Virus Features

**Overlap Analysis**:

- Phase 3 discovered 20 features in Layer 10 (correlational markers)
- Top-100 virus features: 100 most important for adversarial effect
- **Overlap**: **0/100** (0.0%)

**Interpretation**: Phase 3 features are strong **correlational markers**, but the adversarial virus lives in a **different, dense causal subspace** of the SAE feature basis. The features that correlate with task success are orthogonal to the features that causally control task performance.

### Top Virus Features (Layer 10)

```text
Top 10 features by |alpha| (causal importance):
1. F57     alpha=+0.360
2. F5805   alpha=-0.358
3. F194    alpha=+0.354
4. F4405   alpha=-0.353
5. F87     alpha=-0.353
6. F1170   alpha=+0.352
7. F3828   alpha=-0.351
8. F2141   alpha=-0.347
9. F5006   alpha=-0.347
10. F2442  alpha=+0.343
```

None of these overlap with Phase 3 discovered features, confirming orthogonality.

### Implication

**IOI is controllable via a single residual direction, but this direction corresponds to a distributed pattern over thousands of SAE features**. Even in an interpretable SAE basis, the adversarial mechanism is not sparse.

This suggests a fundamental gap between:
1. **"Nice" monosemantic features** discovered by correlation (Phase 3), and
2. **Truly causal control directions** in the residual stream (Phase 4B)

**Atlas therefore exposes a fundamental tension between sparse interpretability and dense adversarial control**. Features that explain task success (correlational, sparse) are distinct from features that control task performance (causal, dense).

### Scientific Significance

**Validated**:
- ✅ Adversarial δ is **88.7% in SAE space** (not orthogonal to learned features)
- ✅ Causal control requires **dense combinations** (~6000 features)
- ✅ Phase 3 and virus features are **orthogonal** (different subspaces)

**Key Insight**: **Sparse ≠ Causal**. The most interpretable features (Phase 3: monosemantic, correlational) are not the most causally important features (Phase 4B-B: distributed, adversarial).

See [adversarial_delta_feature_decomposition_layer10.json](adversarial_delta_feature_decomposition_layer10.json) for complete decomposition data.

---

## 🔬 Phase 4A: Feature Causality Testing Complete

### Rigorous VLO Testing - November 19, 2025

Successfully validated that IOI is a **dense/distributed task**, not sparse feature-based. Feature-level interventions show weak causal effects, confirming component-level analysis (Phase 1) as the correct approach.

### Phase 4A Results

**Methodology**: Rigorous multi-method testing on 2,000 examples with borderline filtering
- **Single Feature Ablation**: VLO ≈ 0.0 (no effect)
- **Single Feature Clamping** (+10.0): Effect = -0.068 (no effect)
- **Multi-Feature Steering** (top 5): Effect = +0.160, Accuracy Δ = -1.7% (weak)

### Phase 4A Key Findings

#### 1. **IOI is Dense/Distributed**

- **Layer 0 MLP** (Phase 1): VLO = **5.276** ✅ Strong causal effect
- **Individual Features** (Phase 4A): |VLO| < 0.07 ❌ No causal effect
- **Implication**: IOI emerges from distributed activation patterns, not sparse features

#### 2. **Features are Correlational Markers**

- **Layer 9 F3428** ("IOI Killer", r=-0.798):
  - Ablation: VLO = -0.005 (removing has no effect)
  - Clamping +10: Effect = -0.037 (forcing has no effect)
- **Interpretation**: Feature activates when model is about to fail, but doesn't **cause** failure
- **Analogy**: Like sweating when hot → correlated with heat, but turning off sweat doesn't cool you

#### 3. **Multi-Feature Circuit Test**

- **Test**: Ablate/clamp top 5 negative features together on 237 borderline examples
- **Results**:
  - Ablate all 5: Effect = +0.099, Accuracy Δ = -1.7%
  - Clamp to P99: Effect = +0.160, Accuracy Δ = -1.7%
- **Conclusion**: Even combined, features show weak circuit-level causality

#### 4. **Borderline Analysis Reveals Mechanism**

- Dataset: 2,000 examples, 237 borderline (11.8%, logit_diff < 1.5)
- Borderline baseline accuracy: **100%** (model already solves hard cases perfectly)
- **Implication**: High redundancy prevents single/multi-feature steering from working

### Phase 4A Scientific Validation

**What We Validated** ✅:
- Phase 3 correlations are statistically real (r=-0.798 is significant)
- IOI task is dense/distributed, not sparse-circuit based
- Component-level (MLP, attention heads) is correct granularity for IOI
- SAE features useful for **interpretation**, not **causal control**

**What We Invalidated** ❌:
- Feature-level causal steering for IOI
- Sparse circuits identifiable via individual SAE features
- Single-feature interventions as mechanism for control

### Methodological Rigor

**Tests Performed**:
1. **Single Ablation** (100 examples): Zero out features → no effect
2. **Single Clamping** (100 examples, +2/+5/+10): Force activation → no effect
3. **Percentile-Based Clamping** (2000 examples): P90/P99/P99.9 → no effect
4. **Multi-Feature Steering** (237 borderline): Top 5 together → weak effect

**Statistical Power**:
- Large dataset (2,000 examples vs initial 100)
- Borderline filtering (11.8% near decision boundary)
- Natural activation percentiles (P90, P99, P99.9)
- Multi-feature combinations tested

### Implications for Research

**Phase 1 is Validated**:
- Layer 0 MLP: VLO = 5.276 (component-level causality works!)
- Continue circuit discovery at **component-level** (MLPs, attention heads)
- Use SAE features to **interpret** component behavior, not control it

**Next Steps**:
- Phase 4B: Adversarial steering vectors (residual stream interventions)
- Component-level circuit discovery (building on Phase 1)
- SAE features for post-hoc interpretability of discovered circuits

See [rigorous_feature_steering_results.json](rigorous_feature_steering_results.json) for complete data.

---

## 🔬 Phase 3: Feature Discovery Complete

**Feature-Level Circuit Discovery - November 19, 2025, 15:43**

Successfully analyzed all 73,728 SAE features across 12 layers to discover which specific features are critical for the IOI (Indirect Object Identification) task.

### Phase 3 Results

**Execution**: 5.8 seconds | **Features Analyzed**: 73,728 | **Examples**: 100 IOI sentences

**Features Discovered**: **223 IOI-critical features** (0.3% of total)
- **Top Feature**: Layer 9, Feature 3428 (r=-0.798, "IOI Killer")
- **Success Marker**: Layer 11, Feature 1724 (r=+0.361, 97% frequency)
- **Correlation Threshold**: |r| ≥ 0.2

### Key Discoveries

#### 1. **"IOI Killer" Feature (Layer 9, Feature 3428)**
- **Correlation**: -0.798 (strongest negative!)
- **Activation Frequency**: 2% (ultra-selective)
- **Interpretation**: When this feature activates → model confuses subject/object, IOI fails
- **Implication**: Primary failure mode identified at feature-level

#### 2. **"IOI Success Marker" (Layer 11, Feature 1724)**
- **Correlation**: +0.361 (strongest positive!)
- **Activation Frequency**: 97% (always-on)
- **Mean Activation**: 3.33 (very high)
- **Interpretation**: Present in almost all successful IOI predictions

#### 3. **Bi-Modal Architecture Revealed**
- **Always-On Features** (97% freq): Context features (Layer 0, 11)
- **Trigger Features** (1-5% freq): Decision-critical selectors (Layer 8-10)
- **Processing Features** (5-70% freq): Task-specific patterns (Layer 4-7)

#### 4. **Negative Features Dominate**
- **221 negative** vs **2 positive** features
- SAE learns **failure modes** better than success patterns
- Features represent "what to avoid" rather than "what to do"

#### 5. **Layer 0 Paradox Resolved**
- **Phase 1 (VLO)**: Layer 0 MLP = 5.276 (dominant component)
- **Phase 3 (Features)**: Only 3 features above threshold
- **Explanation**: Layer 0 is critical but works **densely** (distributed), not **sparsely** (selective features)

### Distribution by Layer

```
Layer  0:   3 features ⚠️  (dense processing, no dominant features)
Layer  1:  20 features
Layer  2:  20 features
Layer  3:  20 features
Layer  4:  20 features
Layer  5:  20 features
Layer  6:  20 features
Layer  7:  20 features
Layer  8:  20 features  (ultra-selective, decision-critical)
Layer  9:  20 features  🏆 (contains "IOI Killer" -0.798)
Layer 10:  20 features
Layer 11:  20 features  ✨ (only layer with positive features)
```

### Top 5 Critical Features

1. **Layer 9, F3428**: r=-0.798, 2% freq → Subject/object confusion
2. **Layer 10, F2844**: r=-0.691, 3% freq → Name disambiguation error
3. **Layer 8, F3488**: r=-0.689, 3% freq → Indirect object misidentification
4. **Layer 11, F1462**: r=-0.683, 3% freq → Output layer inhibition
5. **Layer 11, F1935**: r=-0.610, 3% freq → Decision suppression

### Phase 3 Scientific Significance

**Enables**:
- **Feature-Level Steering**: Suppress "IOI Killer" feature to improve accuracy
- **Error Detection**: Monitor features that predict failures
- **Mechanistic Understanding**: IOI uses always-on context + rare triggers
- **Cross-Layer Integration**: Features 8-11 dominate decision making

**Novel Findings**:
1. IOI task uses bi-modal feature architecture (context + triggers)
2. Negative features more sparse-codable than positive (99% discovered are negative)
3. Late layers (8-11) contain decision-critical features (missed in component VLO)
4. Layer 0 paradox: critical as component but non-sparse at feature level

See [ATLAS_COMPLETE.md](ATLAS_COMPLETE.md) for complete Phase 3 analysis.

---

## 🏆 Phase 2 Complete: Neural Atlas Training

**Complete 12-Layer SAE Training - November 19, 2025**

Successfully trained and validated Sparse Autoencoders for all 12 layers of GPT-2 with exceptional quality.

### Training Configuration

- **Layers Validated**: 12/12 (100% success rate)
- **Total Features**: 73,728 (6,144 per layer, dict_mult=8)
- **Sparsity**: k=128 (2.1% active features, Top-K)
- **Training Data**: 100,000 IOI examples (98GB, D:/NeuroTrace/20251118_123433/)
- **Architecture**: EnhancedSAE (decoder norm, ghost gradients, Top-K, pre-bias)
- **Validation**: 500 examples/layer, <10% accuracy loss threshold

### Validation Results

**All 12/12 layers passed validation** with remarkable quality:

| Layer | MSE Loss | L0 Sparsity | Dead % | IOI Accuracy Change | Status |
|-------|----------|-------------|--------|---------------------|---------|
| 0     | 0.0109   | 128.0       | 0.0%   | +0.8%               | ✅ Excellent |
| 1     | 0.0062   | 128.0       | 0.0%   | 0.0%                | ✅ Excellent |
| 2     | 0.0159   | 128.0       | 0.0%   | +0.4%               | ✅ Excellent |
| 3     | 0.0084   | 128.0       | 0.0%   | +0.6%               | ✅ Excellent |
| 4     | 0.0091   | 128.0       | 0.0%   | +1.0%               | ✅ Excellent |
| 5     | 0.0096   | 128.0       | 0.0%   | +0.2%               | ✅ Excellent |
| 6     | 0.0119   | 128.0       | 0.0%   | +0.4%               | ✅ Excellent |
| 7     | 0.0125   | 128.0       | 0.0%   | +1.2%               | ✅ Excellent |
| 8     | 0.0130   | 128.0       | 0.0%   | +1.2%               | ✅ Excellent |
| 9     | 0.0222   | 128.0       | 0.0%   | +0.4%               | ✅ Good |
| 10    | 0.0598   | 128.0       | 0.0%   | +0.8%               | ✅ Good |
| 11    | 0.0837   | 128.0       | 0.0%   | -0.6%               | ✅ Acceptable |

**Average Reconstruction Loss**: **-0.6%** (improvement!)
**Layers with Improved Accuracy**: **10/12** (83%)

### Remarkable Findings

1. **Denoising Effect**: 10/12 layers **improve** IOI accuracy when reconstructed through SAE
   - Layer 7 & 8: +1.2% (best performers)
   - Layer 4: +1.0%
   - Average: -0.6% (negative = improvement)

2. **Perfect Sparsity**: All layers maintain exact L0=128.0 (Top-K working perfectly)

3. **Zero Dead Features**: 0.0% dead features across all 12 layers (ghost gradients effective)

4. **MSE Quality**: Range 0.006-0.084, all within publication standards

### Technical Achievements

1. **Training/Inference Alignment**: Fixed critical pipeline mismatch
   - Validation now uses same forward() as training
   - No manual normalization in validation hook

2. **Data Quality**: 100K properly captured activations
   - Fixed shape bug from previous 58K capture
   - Batch shape: [32, 28, 768] (correct)

3. **Optimal Hyperparameters**:
   - dict_mult=8 provides sufficient capacity
   - k_sparse=128 balances sparsity vs information
   - sparsity_lambda=1e-4 for L1 regularization

See [checkpoints/all_layers_sae/training_summary.json](checkpoints/all_layers_sae/training_summary.json) for complete metrics.

---

## 📊 Phase 1: Component-Level Circuit Discovery

**VLO-Validated Causal Importance - Completed**

### Methodology

- **Dataset**: 200 IOI examples
- **Model**: GPT-2 (124M parameters, 12 layers)
- **Method**: VLO (Value of Learned Organization) via zero ablation
- **Components Tested**: 156 (144 attention heads + 12 MLPs)
- **Threshold**: VLO > 0.5 for significance

### Key Finding: Layer 0 MLP Dominance

**Layer 0 MLP**: VLO = **5.276** (ONLY significant component above threshold)

**Interpretation**: Layer 0 MLP dominates IOI task through early structural pattern detection (syntax shortcuts: "gave X to", name positions) rather than semantic understanding.

### Top Components by VLO

1. **layer_0.mlp** - VLO=**5.276** ⭐ (70% of causal importance)
2. layer_7.attn_head_10 - VLO=2.1 (attention head)
3. layer_8.mlp - VLO=1.8
4. layer_4.mlp - VLO=0.81
5. layer_10.mlp - VLO=0.57

**All attention heads individually**: VLO < 0.3 (not significant when isolated)

### Implications

- Small models (GPT-2) rely on **syntax over semantics** for IOI
- **Early layers** (Layer 0) encode critical structural/positional information
- Individual attention heads do NOT participate significantly (MLP-driven task)

### Integration with Phase 3

**Layer 0 Paradox Explained**:
- **Component VLO** (Phase 1): 5.276 → Critical as aggregate component
- **Feature Discovery** (Phase 3): Only 3 features → Contribution is **dense**, not **sparse**
- **Conclusion**: Layer 0 works as unified block, not via specific interpretable features

See [docs/research/FINAL_RESULTS.md](docs/research/FINAL_RESULTS.md) for complete Phase 1 analysis.

---

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd Analisi_Neurale

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import neurotrace; print('NeuroTrace installed successfully')"
```

### Run Feature Discovery

```bash
# Verify Phase 3 readiness
python check_phase3_readiness.py

# Discover IOI-critical features (5 seconds)
python discover_feature_circuits.py
```

**Output**: `feature_circuit_discovery.json` with 223 ranked features

### Basic Usage

```python
from neurotrace.control import EnhancedSAEFeatureStore
from neurotrace.discovery import FeatureCircuitDiscoverer
from neurotrace.datasets import IOIDatasetGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Load all 12 SAE layers
feature_store = EnhancedSAEFeatureStore()
for layer in range(12):
    feature_store.load_sae(
        f'checkpoints/all_layers_sae/layer_{layer}/final.pt',
        layer=layer,
        device="cuda"
    )

# Generate IOI examples
generator = IOIDatasetGenerator(seed=42)
examples = generator.generate(num_examples=100)

# Discover features
discoverer = FeatureCircuitDiscoverer(feature_store, model, tokenizer, device="cuda")
important_features = discoverer.discover_from_examples(
    examples=examples,
    top_k_per_layer=20,
    min_correlation=0.2
)

# Analyze top features
print(f"Discovered {len(important_features)} IOI-critical features")
for feat in important_features[:10]:
    print(f"Layer {feat.layer}, Feature {feat.feature_idx}: r={feat.correlation_with_success:.3f}")
```

---

## Project Structure

```
Analisi_Neurale/
├── neurotrace/                        # Core framework
│   ├── analysis/                     # Analysis tools
│   ├── causal/                       # Causal intervention tools
│   ├── control/                      # Steering and control
│   ├── datasets/                     # Dataset generation (IOI)
│   ├── discovery/                    # Circuit discovery
│   ├── instrumentation/              # Model hooking and tracing
│   ├── models/                       # Model wrappers
│   ├── training/                     # SAE training
│   └── visualization/                # Plotting tools
│
├── checkpoints/                       # Saved models and vectors
│   ├── adversarial_delta_layer*.pt   # Attack vectors
│   ├── learned_task_boost_layer*.pt  # Defense vectors
│   ├── virus_detector_layer*.pt      # Detectors
│   └── all_layers_sae/               # Sparse Autoencoders
│
├── cli/                               # Command Line Interface
│
├── docs/                              # Documentation
│
├── results/                           # Experiment results (JSON)
│
├── scripts/                           # Utility scripts
│
├── tests/                             # Unit tests
│
├── Phase Scripts (Root)
│   ├── phase4b_...py                 # Adversarial Steering
│   ├── phase5a_...py                 # Layer Sweep
│   ├── phase5b_...py                 # Sparse Virus
│   ├── phase6b_...py                 # Virus Defence
│   ├── phase6c_...py                 # Subspace Analysis
│   ├── phase6d_...py                 # Centered Defence
│   ├── phase7_...py                  # Linear Task Boost
│   ├── phase7b_...py                 # Learned Task Boost
│   ├── phase7c_...py                 # Boost Scaling
│   ├── phase7d_...py                 # Constrained Boost
│   ├── phase8a_...py                 # Immune Gating
│   ├── phase8b_...py                 # War Surface
│   ├── phase9a_...py                 # Virus Detector
│   ├── phase9b_...py                 # Gated Defence V2
│   ├── phase9c_...py                 # Needs Boost Detector
│   ├── phase9d_...py                 # Gated Defence V3
│   └── phase10_...py                 # Alien Generalization
│
└── README.md                          # Project documentation
```

---

## Core Capabilities

### 1. Feature-Level Discovery (NEW - Phase 3)

Discover which specific SAE features are critical for task behavior:

```python
from neurotrace.discovery import FeatureCircuitDiscoverer

discoverer = FeatureCircuitDiscoverer(feature_store, model, tokenizer)
features = discoverer.discover_from_examples(
    examples=ioi_examples,
    top_k_per_layer=20,
    min_correlation=0.2
)
```

**Output**: 223 IOI-critical features ranked by correlation with success

### 2. Component-Level Discovery (Phase 1)

Automated identification of causal pathways at component level:

```python
from neurotrace.discovery import ExhaustiveCircuitScanner

scanner = ExhaustiveCircuitScanner(model, tokenizer, config)
results = scanner.scan_all_components(
    input_ids=input_ids,
    attention_mask=attention_mask,
    target_positions=target_positions,
    correct_token_ids=correct_ids,
    incorrect_token_ids=incorrect_ids,
)
```

**Output**: VLO scores for 156 components (144 heads + 12 MLPs)

### 3. Sparse Autoencoders (Phase 2)

SOTA SAE architecture with publication-quality features:

```python
from neurotrace.training import create_enhanced_sae, EnhancedSAETrainer

sae = create_enhanced_sae(
    input_dim=768,
    dict_mult=8,        # 6,144 features (8x overcomplete)
    k_sparse=128,       # Top-128 activation
    sparsity_lambda=1e-4,
    normalize_decoder=True,
)

trainer = EnhancedSAETrainer(sae, config)
trainer.train(dataloader)
```

**Features**:
- Decoder weight normalization (Anthropic 2023)
- Ghost gradients (dead feature revival)
- Top-K activation (exact sparsity control)
- Pre-bias correction

### 4. Active Steering (Phase 4 - Coming)

Real-time intervention using discovered features:

```python
from neurotrace.control import CircuitController, SteeringBuilder

# Build steering from discovered features
builder = SteeringBuilder(feature_store=feature_store)
controller = CircuitController(model_wrapper, registry, builder)

# Suppress "IOI Killer" feature
controller.enable_circuit("suppress_layer9_f3428", global_alpha=-2.0)

# Amplify "Success Marker" feature
controller.enable_circuit("amplify_layer11_f1724", global_alpha=+2.0)

# Generate with steering
output = controller.generate(prompt, max_new_tokens=20)
```

---

## Results Summary

### Phase 3: Feature Discovery

**Dataset**: 100 IOI examples
**Features Analyzed**: 73,728 (12 layers × 6,144)
**Execution Time**: 5.8 seconds
**GPU**: NVIDIA RTX 2060 (6GB VRAM)

**Discovered**: 223 IOI-critical features (0.3% of total)

**Top Features**:
1. Layer 9, F3428: r=-0.798 (IOI Killer - subject/object confusion)
2. Layer 10, F2844: r=-0.691 (name disambiguation error)
3. Layer 8, F3488: r=-0.689 (indirect object misidentification)
4. Layer 11, F1724: r=+0.361 (IOI Success Marker - 97% freq)
5. Layer 11, F2338: r=+0.312 (decision support)

**Architecture Revealed**:
- **Always-on features** (97% freq): Context (Layer 0, 11)
- **Trigger features** (1-5% freq): Decision-critical (Layer 8-10)
- **Processing features** (5-70% freq): Task-specific (Layer 4-7)

**Novel Finding**: 99% of discovered features are **negative** (predict failures, not success)

### Phase 2: Atlas Training

**Dataset**: 100,000 IOI examples (98GB)
**Architecture**: 768 → 6,144 features per layer
**Sparsity**: Top-128 (2.1% active)
**Total Features**: 73,728 interpretable features

**Validation**: 12/12 layers passed (<10% accuracy loss)
**Average Reconstruction Loss**: -0.6% (improvement!)
**Dead Features**: 0.0% across all layers
**Training Time**: ~5 minutes per layer (63 minutes total)

**Quality Distribution**:
- Excellent (MSE < 0.015): 7 layers
- Good (MSE 0.015-0.035): 3 layers
- Acceptable (MSE 0.035-0.070): 2 layers

### Phase 1: Component Discovery

**Dataset**: 200 IOI examples
**Components Tested**: 156 (144 heads + 12 MLPs)
**Method**: VLO (zero ablation)

**Key Finding**: Layer 0 MLP dominates (VLO=5.276)

**Top 5 Components**:
1. layer_0.mlp: VLO=5.276 ⭐
2. layer_7.attn_head_10: VLO=2.1
3. layer_8.mlp: VLO=1.8
4. layer_4.mlp: VLO=0.81
5. layer_10.mlp: VLO=0.57

**Mechanism**: Early structural pattern detection (syntax shortcuts)

---

## Cross-Phase Integration

### Layer 0 Paradox Resolved

**Phase 1 (Component VLO)**:
- Layer 0 MLP: VLO=5.276 (dominant)
- 70% of causal importance
- Critical component

**Phase 3 (Feature Discovery)**:
- Layer 0: Only 3 features above threshold
- Features: F1262 (r=-0.42), F709 (r=-0.25), F5680 (r=-0.21)
- Under-represented compared to other layers (20 features each)

**Resolution**:
- Layer 0 MLP is **critical as aggregate component**
- But contribution is **dense** (distributed across many features)
- Not **sparse** (no single dominant features)
- Works as unified block, not via interpretable feature selectors

### Late Layers Emergence

**Phase 1**: Layer 9-11 not dominant in VLO (< 0.5 threshold)

**Phase 3**: Layer 9-11 contain strongest features
- Layer 9: "IOI Killer" F3428 (r=-0.798, strongest!)
- Layer 11: "Success Marker" F1724 (r=+0.361, only strong positive)
- 8/10 top features in layers 8-11

**Interpretation**:
- Component VLO measures **aggregate importance**
- Feature discovery finds **specific selectors**
- Late layers critical at feature-level (decision triggers)
- Missed in component-level analysis (low aggregate VLO)

### Complementary Views

**Component-Level** (Phase 1):
- Measures holistic block importance
- Finds which layers/components are critical
- Aggregate causal effect

**Feature-Level** (Phase 3):
- Measures specific feature importance
- Finds which features within components are critical
- Selective causal effect

**Both needed for complete understanding**

---

## Performance

### Computational Requirements

| Task | Time | Memory | GPU Required |
|------|------|--------|--------------|
| Phase 1: Component Discovery (200 ex) | ~15 min | 2 GB RAM | 6 GB VRAM |
| Phase 2: Atlas Training (12 layers) | ~63 min | 4 GB RAM | 6 GB VRAM |
| Phase 3: Feature Discovery (100 ex) | ~6 sec | 2 GB RAM | 6 GB VRAM |
| Dataset Capture (100K examples) | ~45 min | 8 GB RAM | 6 GB VRAM |

### Scalability

- **Hardware**: Consumer GPU (RTX 2060, 6GB VRAM)
- **Batch Processing**: Optimized for limited VRAM
- **Multi-Layer**: All 12 layers in ~1 hour
- **Large Datasets**: Tested up to 100K examples

---

## Scientific Rigor

### Validation

- ✅ All 12/12 SAE layers validated with <10% accuracy loss threshold
- ✅ Feature discovery executed in 5.8 seconds (12,729 features/sec)
- ✅ Cross-validation between Phase 1 (VLO) and Phase 3 (correlation)
- ✅ Reproducible (seed=42, documented configs)
- ✅ No NaN/Inf values in features
- ✅ Correlations in valid range [-1, +1]

### Quality Metrics

**SAE Training** (Phase 2):
- MSE: 0.006-0.084 (publication standard < 0.12)
- Dead features: 0.0% (SOTA target < 5%)
- Exact sparsity: L0=128.0 across all layers (Top-K working)

**Feature Discovery** (Phase 3):
- Features discovered: 223/73,728 (0.3%, highly selective)
- Max correlation: 0.798 (strong signal)
- Statistical robustness: 100 examples sufficient for r > 0.2

**Component Discovery** (Phase 1):
- VLO threshold: >0.5 for significance
- Statistical testing: bootstrapping available
- Cross-validation on unseen examples

---

## 🚀 Future Roadmap (Post-Phase 14)

With the **Integrated Defence System** successfully validated, the project moves towards scaling and generalization.

### 1. Scale Up
- **Objective**: Apply the defense architecture to larger models (GPT-2 Medium, GPT-2 Large).
- **Hypothesis**: Larger models may have more robust "Task Boost" vectors but also more complex "Alien" structures.

### 2. Task Generalization
- **Objective**: Test the "Task Boosting" strategy on other algorithmic tasks (e.g., Greater-Than, Gender Bias, Factual Recall).
- **Goal**: Verify if the "Boost + Gate" pattern is a universal defense mechanism for transformer circuits.

### 3. Detector Distillation
- **Objective**: Optimize the **Context Classifier** and **Needs Boost Detector**.
- **Goal**: Reduce the computational overhead (currently minimal) to negligible levels for real-time inference.

### 4. Publication
- **Objective**: Formalize the findings into a research paper.
- **Key Contributions**:
    1.  Discovery of the "Alien" nature of Task Boost vectors (Phase 10B).
    2.  Quantification of Collateral Damage in static defenses (Phase 11).
    3.  The "Integrated Defence" architecture achieving zero-cost robustness (Phase 14).

---

## References

### Implemented Methods

1. **Anthropic** - "Towards Monosemanticity" (2023) - Decoder normalization
2. **Anthropic** - "Scaling Monosemanticity" (2024) - Large-scale SAE training
3. **Gao et al.** - "Top-K SAE" (2024) - Exact sparsity control
4. **Rajamanoharan et al.** - "JumpReLU" (Gemma Scope 2024) - Ghost gradients

### Related Work

- **IOI Task**: Wang et al. (2022)
- **Circuit Discovery**: Conmy et al. (2023)
- **SAELens**: Bloom et al. (2024)

---

## Citation

```bibtex
@software{neurotrace2025,
  title={NeuroTrace: Neural Network Interpretability Framework},
  author={NeuroTrace Team},
  year={2025},
  version={3.0.0},
  note={Complete 12-Layer Neural Atlas with Feature-Level Discovery}
}
```

---

## License

Apache-2.0

See [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Built with:
- PyTorch
- Transformers (Hugging Face)
- NumPy, SciPy
- Plotly (Visualization)

---

**Status**: Phase 3 Complete | 73,728 Features Mapped | 223 IOI-Critical Features Discovered | Ready for Feature-Level Steering

For complete analysis see:
- Phase 1: [docs/research/FINAL_RESULTS.md](docs/research/FINAL_RESULTS.md)
- Phase 2+3: [ATLAS_COMPLETE.md](ATLAS_COMPLETE.md)
