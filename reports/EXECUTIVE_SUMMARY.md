# Executive Summary: The "Virus & Defense" Journey on GPT-2 Small

This section summarizes the entire "virus & defense" path on GPT-2 Small, from SAE study to the integrated immune system. It is designed as a project overview, highlighting not just positive results but also the failures that guided the redesign.

## 🌍 Panorama in 10 Lines

*   **Task**: Indirect Object Identification (IOI) ("Mary/John") on GPT-2 Small, with baseline accuracy ≈ 97–98%.
*   **Phase 1–3**: Layer-wise SAE training and discovery of IOI-correlated features → excellent for diagnosis, weak for control.
*   **Phase 4–6**: Construction of an attack vector (virus) in the residual stream → controlled accuracy collapse to ≈ 36–40%, and then ≈ 2–8% via layer-specific attacks.
*   **Key Result**: IOI is controllable via the residual stream, but the representation is dense and distributed; SAE features are largely correlational, not causal.
*   **Phase 7–8**: Construction of boost vectors (task vectors) that repair damage and, if applied statically, exceed the baseline (cognitive doping).
*   **Problem**: Global Boost = severe collateral damage (perplexity doubled on WikiText).
*   **Phase 8–9**: Introduction of detectors (virus / needs-boost) and gating → targeted defense only where the model is about to fail.
*   **Phase 10**: "Alien" generalization test → the boost improves IOI even outside standard templates, but is not yet a pure "universal".
*   **Phase 10B–11**: SAE analysis of the boost and collateral damage → boost is off-manifold, composed of canceling features, confirming global toxicity.
*   **Phase 13–14**: Context classifier (IOI vs WikiText) + integrated system → active defense only in the IOI domain, no impact on generic text.

---

## 🔬 Phase 0–3: Control, SAE, and Feature Discovery

### Phase 0/1 – Setup IOI Control Plane
*   Definition of standard IOI task (synthetically generated dataset).
*   **GPT-2 Small Baseline**:
    *   Accuracy ≈ 97–98% on IOI test set.
    *   Logit diff stable (~3.5) between correct token and foil.

### Phase 2 – SAE Training Pipeline
*   Training of SAE (EnhancedSAE) on all 12 layers: `d_model = 768` → `6144` features per layer, with MSE + L1 sparsity.
*   **Quality Verification**:
    *   Low reconstruction error.
    *   Controlled sparsity.
    *   **IOI Test**: Baseline vs reconstructed accuracy practically identical (e.g., layer 2: 68.8% → 69.6%), proving SAEs preserve information.

### Phase 3 – Feature Discovery
*   Automatic search for features with high correlation to IOI success/failure.
*   Production of a feature list:
    *   **Positive set**: Correlated with correct behavior.
    *   **Negative set**: Correlated with errors.
*   *Result*: This gives a diagnostic map of the model, but not yet a causal control mechanism.

---

## ⚔️ Phase 4: From Sparse Steering to Dense Virus

### Phase 4A – Feature Steering ("Clean" Failure)
*   **Test**: Causality testing via forced activation of single SAE features (positive/negative) and multi-feature steering (top 5 negative) on borderline examples.
*   **Result**: No strong effect (`|Δ logit_diff| ≪ 0.5`), minor accuracy variations (1–2% max).
*   **Conclusion**: SAE features found in Phase 3 are correlational markers, not causal levers. IOI is not explainable by a few "strong" sparse features.

### Phase 4B – Adversarial Residual Delta (Virus $v_{att}$)
*   **Strategy Shift**: Instead of acting in SAE space, directly optimize a vector $\delta$ in the residual stream of a layer.
*   **Results**: Training on borderline examples → adversarial vector inverting logit diff.
    *   **IOI Test**: Accuracy from ≈ 97% → ≈ 36–40%.
*   **Key Discovery**: The residual stream contains densely causal directions for the IOI task. Direct steering on the residual is much more powerful than control via single SAE features.

### Phase 4B-B – Virus in SAE Space
*   Projection of virus $v_{att}$ into layer SAE space:
    *   ≈ 89% of virus energy lives in SAE space.
    *   But it is distributed over all 6144 features: **no useful sparsity**.
    *   Zero overlap with top Phase 3 features.
*   **Conclusion**: SAE space is a good atlas of internal state, but truly causal directions appear dense there too.

---

## 🛡️ Phase 5: Layer Vulnerability

### Phase 5A – Layer Vulnerability Sweep
*   For each layer 0–11, train a dedicated adversarial vector $v_{att}^l$.
*   **Results (under attack)**:
    *   **Layer 0–7**: Accuracy drop ≈ −97% (task practically destroyed).
    *   **Layer 8–9**: Drop 92–96%.
    *   **Layer 10**: Drop ~60%.
    *   **Layer 11**: Almost invulnerable (≈ −0.4%).
*   **Interpretation**: Almost all layers are attackable with strong residual deltas. The last layer acts as a robustness buffer: resists perturbations, likely because it aligns output with vocabulary.

### Phase 5B – Sparse SAE Virus (Attempt)
*   Attempt to build a virus in sparse SAE space (few coefficients).
*   **Result**: Much weaker effects compared to dense virus.
*   **Conclusion**: Even the virus, to be effective, requires dense directions.

---

## 🧬 Phase 6: Virus Subspace, Defence, and Centering

### Phase 6A–B – Virus Subspace Decomposition
*   PCA on virus in residual/SAE space: Few principal components explain most of the effect, but representation remains dense, not sparse-friendly.
*   **Idea**: Work with a low-dimensional basis (PC1–PC4) as "sick" subspace.

### Phase 6C – Task vs Virus Subspace
*   Comparison between: "Good" task direction (via ridge regression) and Virus subspace.
*   **Results**: Cosine sim ≈ 0 between task direction and virus PC1 → almost orthogonal. But mean activation has very high energy in virus subspace.
*   **Interpretation**: Task direction and virus direction do not coincide, but the virus lives in an activation region where the model often passes → structural vulnerability.

### Phase 6D – Centered Defence
*   **Method**: Remove layer mean, project out of virus subspace, re-add mean.
*   **Result**:
    *   *Defence only*: Keeps accuracy almost intact.
    *   *Attack + Centred Defence*: Attenuates effect but is not a perfect cure.
*   **Conclusion**: Conceptually useful (separates structure / virus), but not yet the definitive solution.

---

## 🚀 Phase 7: Task Boost – From Pill to Controlled Doping

### Phase 7A – Task Boost (Ridge Direction)
*   Initial attempt: Use linear task direction (ridge) as boost.
*   **Result**: Almost no improvement on IOI, nor true defense against virus.
*   **Conclusion**: Linear direction is too weak / poorly aligned with true internal dynamics.

### Phase 7B – Learned Task Boost ($v_{boost}$)
*   Direct optimization of a vector $v_{boost}$ at layer 10 to: maximize logit diff on hard examples, and resist the virus.
*   **Results**:
    *   **HardAcc**: 70% → 100% (without virus) with `boost_only`.
    *   **With Virus**: `attack_only` collapses to 0%, `attack+boost` returns > 98%.
*   **Norms**: $||v_{boost}|| \approx 112$ → very "energetic" vector.

### Phase 7C – Boost Scaling
*   Scaling test $v_{boost}$ with factors 5, 10, 20, 30, 50, 80, 112.
*   **Observations**: Already at scale 20–30, boost strongly improves IOI. `attack + boost` starts recovering up to ~0.96–0.98.
*   **Conclusion**: There exists a family of scales for which $v_{boost}$ is a powerful nootropic for IOI.

### Phase 7D – Constrained Task Boost (Fixed Norm R=25)
*   Re-training boost with fixed norm (R=25): Avoids norm explosion.
*   **Results**:
    *   `boost_only`: TestAcc ≈ 99.4%, HardAcc ≈ 92.5%.
    *   `attack + boost` ($\alpha=1$): TestAcc ≈ 84.4%, HardAcc ≈ 52.5%.
*   **Verdict**: $v_{boost}^{R25}$ becomes the compromise to use as defense "drug".

---

## 🛡️ Phase 8: War Surface & Immune Gating

### Phase 8B – War Surface ($\alpha, \beta$)
*   Grid search on: attack strength $\alpha \in \{0.0, 0.5, 1.0, 1.5, 2.0\}$, boost strength $\beta \in \{0.0, 1.0, 2.0, 3.0, 4.0, 5.0\}$.
*   **Heatmap (HardAcc)**:
    *   Without attack ($\alpha=0$): $\beta=3$ brings hard acc to 98%.
    *   With standard attack ($\alpha=1$): $\beta \ge 3$ brings hard acc $\ge 0.9$.
*   **Conclusion**: There is a true phase line $\beta^*(\alpha)$ beyond which the virus is neutralized. $v_{boost}$ is both an enhancer ($\alpha=0$) and a cure ($\alpha>0$).

### Phase 8A – Immune Gating v1 (Logit Threshold)
*   Gating conditioned on `logit_diff`: If diff < threshold → apply boost, else → no.
*   **Result**: Good HardAcc, but TestAcc drops compared to static defence, high FP (boost activated unnecessarily on healthy cases).
*   **Conclusion**: Gating idea is right, but signal (only logit_diff) is too crude.

---

## 🧠 Phase 9: Detector & Third Generation Gated Defence

### Phase 9A – Virus Detector (Clean vs Attacked)
*   MLP distinguishing layer10 activations clean vs attacked.
*   **Result**: Test accuracy 100%, AUC 1.0.
*   **But**: Detects presence of specific attack, not "functional damage".

### Phase 9B – Gated Defence v2 (with Virus Detector)
*   Gating based on "is virus present?".
*   **Result**: Virus always present in test scenario → gateRate ≈ 100%. Effectively reverts to masked static defence.

### Phase 9C – Needs-Boost Detector (Damage, not Virus)
*   **New Detector**: Label = "does this example fail under attack without boost?"
*   **Features**: Combination of logit diff, virus component, etc.
*   **Result**: TestAcc ≈ 99.4%, precision/recall class=1 ~99%. Identifies which examples fail on their own under attack.

### Phase 9D – Gated Defence v3 (Needs Boost)
*   **Scheme**: Under attack, use detector to decide: if "needs boost" → apply $v_{boost}$, else → leave model intact.
*   **Results**:
    *   **Baseline**: 97.4% / 83.3% (Hard).
    *   **No Defence**: 47.6% / 1.3%.
    *   **Static Defence**: 98.8% / 92.3%.
    *   **Gated Defence v3**: TestAcc: 97.4% (same as baseline), HardAcc: 92.3% (same as static), GateRate: 51%, FP_rate: 0.0%.
*   **Conclusion**: Surgery: intervene only where task collapses, no damage on healthy cases.

---

## 👽 Phase 10: Alien Generalization & Boost Decomposition

### Phase 10 – Alien Generalization (Set A/B/C)
*   Three types of "alien" IOI-like prompts: A: clause-break, B: passives, C: counter-factuals.
*   **Results**:
    *   **Alien A**: prob target 0.115 → 0.659, Top1 25% → 100%.
    *   **Alien B**: prob 0.048 → 0.127, Top1 10% → 20%.
    *   **Alien C**: almost unchanged.
*   **Interpretation**: $v_{boost}$ increases "IOI competence" even outside standard template, but is not yet a "universal reasoner".

### Phase 10B – Boost SAE Decomposition (Corrected)
*   Decomposition of $v_{boost}$ (R=25) in layer 10 SAE space.
*   **Key Data**:
    *   Norm ratio up to 272% → vector in SAE is made of large coefficients that cancel out.
    *   Reconstruction error grows with K → adding more SAE features worsens reconstruction.
    *   Top feature with large mixed coefficients (+/−).
*   **Conclusion**: $v_{boost}$ is **off-manifold** regarding natural SAE feature space. It is an SGD-optimized shortcut, not a clean combination of interpretable concepts.

---

## 💥 Phase 11: Collateral Damage

*   **Test on WikiText-2 (300 samples)**
    *   **Baseline**: NLL ≈ 4.28, PPL ≈ 72.
    *   **Static Defence** (boost always on): NLL ≈ 4.98, PPL ≈ 145.
*   **Interpretation**: $v_{boost}$, if applied everywhere, destroys global linguistic quality. Confirms vector cannot be used as "permanent modification" of the model.

---

## 🏆 Phase 13–14: Context Guard & Integrated Defence

### Phase 13 – Context Classifier (IOI vs WikiText)
*   **Context Classifier**: Input: layer10 activations (clean), Output: "is this prompt IOI yes/no?".
*   **Results**: TestAcc = 100%, False Positives on Wiki ≈ 0%.
*   **Role**: Frontier block: decides whether to activate IOI defense system.

### Phase 14 – Integrated Defence System
*   **Combination of Three Components**:
    1.  **Context Classifier (Phase 13)**: Activates system only in IOI domain.
    2.  **Needs-Boost Detector (Phase 9C)**: In IOI domain, decides if example is vulnerable.
    3.  **Constrained Boost $v_{boost}^{R25}$ (Phase 7D)**: Injected only if `context=IOI` and `needs_boost=true`.
*   **Final Results**:
    *   **On IOI** (under layer10 attack with virus):
        *   *Baseline*: 97.4% / 83.3% (Hard).
        *   *No Defence*: 47.6% / 1.3%.
        *   *Static Defence*: 98.8% / 92.3%.
        *   *Integrated Gated Defence v3*: TestAcc: 94.8%, HardAcc: 84.6%, GateRate ≈ 48.4%.
    *   **On WikiText**:
        *   *Baseline*: NLL 4.2785, PPL 72.14, FP=0%.
        *   *Static Defence*: PPL 145.08, FP=100%.
        *   *Domain Guarded*: NLL 4.2785, PPL 72.14, FP=0%.
*   **Conclusion**: The integrated system is an **immune shell** around GPT-2:
    *   **In IOI Domain**: Powerful attack, adaptive defense, performance recovery.
    *   **Outside Domain**: Original model intact, no PPL cost.
