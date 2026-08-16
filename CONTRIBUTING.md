# Contributing to NeuroTrace

NeuroTrace welcomes contributions that make its interpretability and intervention results easier to **reproduce, falsify, compare, or generalize**.

The repository already contains a substantial GPT-2 / IOI experimental record. The most valuable next contributions are therefore not broader claims or more phases for their own sake, but experiments that test whether the observed effects survive changes in seeds, tasks, model families, datasets, intervention budgets, and distribution.

Before starting substantial work, read:

1. [`README.md`](README.md)
2. the report or result artifact for the phase you want to extend
3. the corresponding script/configuration under `phases/`, `configs/`, `reports/`, or `docs/`

## Evidence model

NeuroTrace distinguishes three kinds of evidence:

1. **Correlation evidence** — an internal representation predicts or tracks an outcome.
2. **Intervention evidence** — changing an internal representation changes measured behavior.
3. **Generalization evidence** — a finding survives meaningful changes in model, task, dataset, seed, domain, or protocol.

These are not interchangeable.

A correlated SAE feature is not automatically a causal control point. A strong intervention on GPT-2/IOI is not automatically a general transformer mechanism. A detector that shows no false positives on one evaluated split is not a zero-false-positive system.

## Current contribution frontier

### 1. Reproducibility across seeds and reruns

Useful work includes:

- rerunning key intervention and gating experiments across multiple declared seeds;
- quantifying run-to-run variation;
- confidence intervals or bootstrap uncertainty for reported metrics;
- deterministic configuration capture;
- model/checkpoint/data manifests;
- scripts that replay saved result artifacts without retraining;
- identifying phase results that are sensitive to initialization or optimization budget.

A negative replication is useful evidence.

### 2. Cross-task and cross-model replication

The strongest open question is whether the patterns found on GPT-2 / IOI survive outside that setting.

Useful contributions include:

- reproducing one bounded experiment on a different transformer family;
- testing an unrelated task rather than another IOI variant;
- comparing layer-depth patterns after normalizing for architecture depth;
- testing whether sparse-feature correlation and causal intervention separate in the same way elsewhere;
- testing whether residual-stream steering remains stronger than the compared sparse interventions under a matched protocol;
- documenting where the original result fails to transfer.

Do not describe a failed transfer as an implementation bug unless the evidence supports that conclusion.

### 3. Gating and detector behavior under distribution shift

The Phase 9/14 gating results are promising within the evaluated setup, but their robustness outside that distribution is unknown.

Useful contributions include:

- held-out distribution-shift suites;
- detector calibration and threshold stability;
- false-positive / false-negative analysis with uncertainty;
- adversarial or naturally shifted inputs that do not reuse training/tuning examples;
- independent general-text evaluation;
- gate-rate / task-quality / collateral-damage trade-offs;
- explicit `abstain` or `unknown` behavior when the detector is outside its supported regime.

Do not tune on the final evaluation set and then present it as independent evidence.

### 4. Stronger simple baselines and alternative interventions

Useful work includes:

- matched random-direction or random-subspace controls;
- norm-matched steering baselines;
- simpler classifiers for gate decisions;
- alternative sparse representations;
- activation patching or other intervention methods under comparable budgets;
- ablations that isolate whether a reported effect comes from direction, magnitude, layer choice, optimization procedure, or dataset structure.

A simpler method matching a NeuroTrace result is a valuable contribution.

## Preserve the experimental record

Historical phase outputs, reports, checkpoints, and machine-readable results are part of the research provenance.

Do not:

- overwrite an existing result artifact with a rerun;
- silently change the meaning of an old metric;
- replace a failed result with a successful one;
- retune a completed evaluation and present it as the same experiment;
- remove earlier failed hypotheses merely because later phases revised them.

New experiments should receive a distinct result identity, configuration, and output path.

## Proposing a new experiment

Before a large experimental PR, open an issue describing:

- the research question;
- the existing NeuroTrace result being tested or challenged;
- model and task;
- dataset and split roles;
- intervention or detector under test;
- comparison baseline(s);
- seeds / replicas;
- primary metrics;
- uncertainty treatment;
- development vs held-out evidence;
- what outcome would count against the proposed hypothesis.

Prefer a small experiment that distinguishes explanations over a broad new phase with many coupled changes.

## Data and evaluation discipline

When practical, distinguish:

- **training data** — may update learned parameters;
- **development/calibration data** — may influence thresholds, hyperparameters, layer choice, or stopping criteria;
- **held-out evaluation data** — should not influence the method being evaluated.

For cross-task or cross-model claims, document any changes needed to adapt the protocol. If adaptation changes the scientific question, say so explicitly.

## Intervention comparisons

When comparing two interventions, match relevant budgets where possible:

- intervention norm;
- number of tuned parameters;
- optimization steps;
- access to labels/data;
- layer-selection procedure;
- evaluation examples;
- compute or search budget.

If budgets cannot be matched, report the difference rather than hiding it behind one score.

## Reproducibility expectations

A research PR should state:

- exact model/revision;
- dataset identity and split construction;
- seed(s);
- configuration file or full parameter set;
- command(s) executed;
- hardware/runtime when materially relevant;
- raw result location;
- derived metrics;
- known limitations;
- whether the result was visible during method development.

Machine-readable outputs are preferred alongside narrative summaries.

## Code contributions

Reusable logic should generally live under `neurotrace/` rather than being duplicated across phase scripts.

Historical scripts may remain frozen when they are needed for provenance. If you need to change a historical path for reproducibility, preserve the old behavior or version the new path rather than silently changing old semantics.

Add deterministic tests for pure logic where practical.

## Claims and terminology

Keep claims tied to the tested conditions.

Please avoid language such as:

- universal transformer vulnerability;
- proven causal mechanism across models;
- general AI defense;
- zero collateral damage;
- zero false positives;
- production robustness;

unless the submitted evidence actually supports that scope.

Prefer wording such as:

- "observed in the evaluated split";
- "under this intervention protocol";
- "replicated across N seeds";
- "did not transfer to model/task X";
- "no false positives were observed in this sample".

## Negative results

Negative and inconclusive results are welcome when the experiment is well specified.

Examples:

- a layer-depth pattern disappears on another model;
- a gate fails under distribution shift;
- an SAE feature remains predictive but non-causal;
- a simpler baseline matches the learned intervention;
- an intervention works but causes unacceptable general-text degradation.

Those results narrow the claim and improve the project.

## Pull request expectations

Keep one conceptual contribution per PR where practical.

A PR should explain:

- what question it addresses;
- what changed;
- which existing result it relates to;
- how it was validated;
- what evidence was added;
- what failed or remained uncertain;
- whether any old results remain directly comparable.

## Good first contributions

Small useful contributions may include:

- deterministic replay of an existing JSON result;
- seed/config capture for one phase;
- confidence intervals over an existing machine-readable result;
- a matched random-direction baseline;
- integrity checks for dataset split overlap;
- documentation corrections that narrow a claim to what the evidence supports.

## License

NeuroTrace is licensed under Apache License 2.0. Unless explicitly stated otherwise, contributions intentionally submitted for inclusion in the project are provided under the same license terms.
