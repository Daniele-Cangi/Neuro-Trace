# NeuroTrace: Neural Network Interpretability Framework

NeuroTrace is an experimental research framework for studying transformer internals through component ablation, sparse autoencoders, residual-stream interventions, adversarial steering, and gated control.

The repository documents a sequence of experiments on **GPT-2 (124M)** and the **Indirect Object Identification (IOI)** task. Results are reported within those experimental conditions and should not be interpreted as universal claims about transformer architectures, production robustness, or general AI safety.

**Version:** 8.0.0  
**Status:** research prototype; Phase 14 benchmark sequence completed  
**Primary model/task studied:** GPT-2 / IOI

<img width="1408" height="752" alt="NeuroTrace conceptual overview" src="https://github.com/user-attachments/assets/52181323-1171-4db4-90a8-64f5c940c108" />

---

## Research question

NeuroTrace asks a narrower question than "can we interpret or control a transformer?":

> Within a declared model, task, dataset, and intervention protocol, which internal representations correlate with behavior, which interventions causally change that behavior, and what trade-offs appear when those interventions are used defensively?

The project evolved through four connected lines of work:

1. **Component-level analysis** — measure the effect of ablating attention heads and MLP blocks.
2. **Sparse representation analysis** — train SAEs across all GPT-2 layers and identify features correlated with IOI outcomes.
3. **Causal intervention** — compare sparse-feature steering with direct residual-stream steering.
4. **Gated intervention** — test whether learned detectors can limit when a task-specific intervention is applied.

---

## Current evidence summary

### 1. Component-level interventions

Phase 1 tested 156 GPT-2 components on IOI examples using zero-ablation / VLO-style measurements. In that experiment, the Layer 0 MLP produced the strongest measured effect.

This is evidence about the tested IOI setup, not a claim that early MLPs generally dominate transformer reasoning.

See [`docs/research/FINAL_RESULTS.md`](docs/research/FINAL_RESULTS.md).

### 2. Sparse autoencoder atlas

Phase 2 trained sparse autoencoders for all 12 GPT-2 layers, with 6,144 features per layer and Top-K sparsity (`k=128`). The repository records 73,728 learned SAE features in total.

All 12 layer-specific SAEs passed the repository's declared reconstruction/IOI validation criterion. These checks establish internal experimental quality for this setup; they are not an external certification or a general benchmark of SAE quality.

See [`ATLAS_COMPLETE.md`](ATLAS_COMPLETE.md) and the checkpoint training summaries.

### 3. Correlation is not causality

Phase 3 identified features correlated with IOI success or failure. Later intervention experiments showed that several apparently strong feature-level markers had weak causal effect when directly ablated or clamped.

That negative result became an important part of the project:

- feature correlations can be useful diagnostics;
- sparse features that predict failure are not automatically causal control points;
- the tested IOI behavior appears more distributed than the original feature-level interpretation suggested.

See [`rigorous_feature_steering_results.json`](rigorous_feature_steering_results.json).

### 4. Residual-stream steering

Phase 4B optimized a 768-dimensional residual-stream intervention at Layer 10. Under the declared IOI evaluation, the learned direction produced a much larger behavioral effect than the tested SAE feature interventions.

For the reported 500-example test set:

| Condition | IOI accuracy |
|---|---:|
| Baseline | 97.2% |
| Residual steering | 36.8% |

This demonstrates strong controllability **for that model/task/layer/protocol**. It does not establish that the residual stream is the universal causal locus of transformer behavior.

See [`adversarial_steering_layer10_results.json`](adversarial_steering_layer10_results.json).

### 5. Layer vulnerability sweep

Phase 5 repeated the adversarial-steering procedure across all 12 GPT-2 layers. The measured effect was strongest in early/middle layers and much smaller at Layer 11 under the tested optimization setup.

The observed depth pattern is a result of this experiment. It remains a hypothesis to test on other tasks, seeds, model families, optimization budgets, and intervention constraints.

### 6. Sparse decomposition of the intervention

The learned residual intervention was projected into the SAE basis. The repository reports that a large fraction of its norm can be represented in that basis, while small Top-K feature subsets preserve little of the measured causal effect.

The useful interpretation is therefore narrower than the earlier "alien vector" language:

> the optimized intervention is poorly approximated by a small sparse subset of the learned SAE dictionary, even when the broader SAE basis captures much of the direction.

See [`adversarial_delta_feature_decomposition_layer10.json`](adversarial_delta_feature_decomposition_layer10.json).

### 7. Defensive intervention and gating

Phases 6-9 explored several defensive strategies:

- direct subspace projection;
- mean-preserving / centered projection;
- learned task-boost vectors;
- constrained task boosts;
- confidence-based gating;
- learned attack/damage detectors.

Several early defensive approaches failed or damaged clean-task performance. Those failures are retained because they motivated the gated design.

In the Phase 9D evaluation, the learned `Needs Boost` gate produced the following benchmark-specific result:

| Mode | Test Acc | Hard Acc | Gate Rate | FP Rate |
|---|---:|---:|---:|---:|
| Baseline | 97.4% | 83.3% | — | — |
| No defence | 47.6% | 1.3% | — | — |
| Static defence | 98.8% | 92.3% | 100% | — |
| Gated V3 | 97.4% | 92.3% | 51.0% | 0.0% |

`0.0% FP` means no false positives were observed on that evaluated split. It is not a claim of zero false positives in general deployment.

### 8. Integrated defence experiment

Phase 14 combines:

- **Domain Guard** — a context classifier;
- **Damage Guard** — the `Needs Boost` detector;
- **Task Boost** — the learned intervention.

The recorded Phase 14 evaluation reports:

| Mode | IOI Acc | WikiText-2 PPL | Gate Rate |
|---|---:|---:|---:|
| Baseline | 97.4% | 72.14 | 0% |
| Static defence | 98.8% | 145.08 | 100% |
| Integrated defence | 94.8% | 72.14 | 48.4% |

Within this evaluation, the integrated configuration avoided the perplexity increase observed with the always-on static intervention while applying the boost to fewer examples.

This should be read as a **benchmark result**, not as "zero collateral damage" or a generally validated AI-defense system. Broader claims require independent tasks, models, datasets, attacks, seeds, and external evaluation.

---

## What changed during the research

NeuroTrace intentionally preserves failed hypotheses and revisions.

Some examples:

- Strong feature correlations initially looked like candidate causal mechanisms; direct feature interventions later showed weak effects.
- Simple subspace removal damaged clean behavior, motivating mean-preserving and later gated approaches.
- A large learned task-boost vector performed strongly on the target task but increased WikiText-2 perplexity when applied statically.
- Gating reduced that measured side effect in the evaluated configuration, but its generality remains unproven.

The project therefore treats later phases as corrections and refinements of earlier interpretations, not as a sequence of ever-stronger universal claims.

---

## Scope and non-claims

NeuroTrace currently does **not** claim:

- a universal transformer defense mechanism;
- production-ready adversarial robustness;
- generalization beyond the evaluated models/tasks without additional experiments;
- that SAE correlations are causal by default;
- that a 0% observed error or side-effect rate implies a true population rate of zero;
- that the tested residual-stream interventions are safe for arbitrary prompts or domains;
- external validation, certification, or state-of-the-art performance.

The strongest claims supported by this repository are local to the declared experimental protocols.

---

## Repository structure

```text
neurotrace/        reusable analysis, discovery, control and training modules
phases/            phase-oriented experiment scripts
reports/           experiment reports and result summaries
checkpoints/       SAEs, intervention vectors and detector checkpoints
configs/           experiment and detector configuration
results/           machine-readable experiment outputs
legacy/            preserved earlier scripts and utilities
docs/              research documentation
```

Several historical phase scripts also remain at repository root for reproducibility and provenance.

---

## Quick start

```bash
git clone https://github.com/Daniele-Cangi/Neuro-Trace.git
cd Neuro-Trace
pip install -r requirements.txt
```

Verify the package import:

```bash
python -c "import neurotrace; print('NeuroTrace installed successfully')"
```

Run the feature-discovery readiness check:

```bash
python check_phase3_readiness.py
```

Run the feature-discovery experiment:

```bash
python discover_feature_circuits.py
```

The repository contains later phase scripts for residual steering, vulnerability sweeps, defensive projections, task boosts, gating, and integrated evaluation. Review the corresponding script and configuration before reproducing a phase; the phases do not all share identical datasets or intervention assumptions.

---

## Experimental methodology

Across the repository, experiments use combinations of:

- deterministic dataset generation and declared seeds;
- component ablation;
- SAE training and reconstruction checks;
- feature correlation analysis;
- direct intervention / steering;
- held-out test splits;
- layer sweeps;
- task-specific and general-text metrics;
- saved JSON results and checkpoints.

The project distinguishes three kinds of evidence:

1. **Correlation evidence** — a representation predicts an outcome.
2. **Intervention evidence** — changing a representation changes the measured outcome.
3. **Generalization evidence** — a finding survives new tasks, models, datasets or domains.

Most of the repository is strong on the first two within GPT-2/IOI. The third remains the main open research requirement.

---

## Selected measured results

| Experiment | Reported result | Appropriate interpretation |
|---|---|---|
| SAE atlas | 12/12 layers passed repository validation criterion | The trained SAEs met the project's declared checks |
| Phase 4B residual steering | 97.2% → 36.8% IOI test accuracy | Strong intervention effect in the tested Layer-10 setup |
| Phase 5 layer sweep | large depth-dependent variation | Candidate vulnerability pattern requiring cross-model replication |
| Phase 9D gated defence | 0.0% observed FP on evaluated split | No false positives observed in that split |
| Phase 11 static boost | WikiText-2 PPL 72.14 → 145.08 | Large measured general-text degradation from always-on intervention |
| Phase 14 integrated setup | WikiText-2 PPL 72.14 with 48.4% gate rate | No measured PPL increase in that evaluation, with selective intervention |

---

## Research directions

The next useful tests are not stronger wording; they are broader evidence:

- reproduce the intervention/gating results across multiple seeds;
- evaluate larger and different transformer families;
- move beyond IOI to unrelated tasks;
- test detector calibration under distribution shift;
- use larger and independent general-text evaluations;
- compare against simpler baselines and alternative intervention methods;
- quantify confidence intervals and failure rates rather than relying on single-point metrics.

---

## References

The repository builds on ideas from mechanistic interpretability, sparse autoencoders, circuit discovery and activation steering, including work associated with Anthropic's monosemanticity research, Top-K SAEs, IOI circuit analysis, and related interpretability tooling.

See the detailed project documentation and experiment files for phase-specific references.

---

## License

Apache-2.0. See [`LICENSE`](LICENSE).

---

NeuroTrace is best read as an evolving experimental record: hypotheses are allowed to fail, later phases may overturn earlier interpretations, and reported numbers remain tied to the conditions under which they were measured.
