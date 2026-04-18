# Phase 1-3 Implementation Review

This document reviews the current implementation state of Phases 1 through 3 against the official assignment requirements in:

- `documents/canvas/assignment/md/RL_Report_Spring_2026_v1-2.md`
- `documents/canvas/assignment/md/RL_Report_Spring_2026_FAQ_v2.md`

This version reflects the **updated rerun artifacts** currently present under `artifacts/`. Relative to the earlier review, the implementation has clearly improved: CartPole rollout budgets are larger, DP hyperparameter validation is now persisted as CSV artifacts, and PI stop reasons are now made explicit in logs and checkpoint JSONs. The main remaining issues are **CartPole model sparsity**, **incomplete PI policy stabilization on Phase 3 default/fine grids**, and **a new consistency gap between the main reported results and the saved hyperparameter-sweep rows**.

## What is already working well

### Phase 1

- `artifacts/metadata/phase1.json` records both environment setups.
- Blackjack model construction still looks correct:
  - `n_states = 290`
  - `n_actions = 2`
  - row-stochasticity max deviation `4.44e-16`
  - reward range about `[-1, 1]`
- CartPole model construction is more ambitious than before:
  - default-grid rollout budget increased to `2000000`
  - persisted coverage improved to `25.17%`
  - smoothed fraction decreased to `57.12%`
- Phase 1 still generates the expected persisted artifacts:
  - `artifacts/metadata/cartpole_model.npz`
  - `artifacts/figures/phase1/bin_edges.png`
  - `artifacts/figures/phase1/coverage_heatmap.png`
  - `artifacts/figures/phase1/visit_histogram.png`

### Phase 2

- The required Blackjack DP outputs are present:
  - `vi_convergence.csv`
  - `pi_convergence.csv`
  - `policy_eval_aggregate.csv`
  - `summary.csv`
  - `hp_validation.csv`
  - the expected convergence and heatmap figures
- Main reported results are still strong:
  - VI converges in `9` iterations
  - PI converges in `3` iterations
  - both main runs evaluate to `-0.0288`
  - main saved policy agreement is `1.0`
- The DP hyperparameter-validation requirement is now being addressed explicitly:
  - `phase2.json` records validated hyperparameters `["gamma", "delta"]`
  - `artifacts/metrics/phase2_vi_pi_blackjack/hp_validation.csv` persists the sweep rows

### Phase 3

- The expected CartPole grid-ablation outputs are present:
  - `vi_convergence.csv`
  - `pi_convergence.csv`
  - `policy_eval_aggregate.csv`
  - `policy_agreement.csv`
  - `discretization_study.csv`
  - `hp_validation.csv`
  - the expected convergence and discretization-study figures
- The rerun materially improved the CartPole DP results:
  - default-grid performance increased to about `456.8` / `466.2`
  - fine-grid performance increased to about `438.1`
  - fine-grid coverage increased from the earlier run to `16.3%`
- The artifacts now make PI stopping behavior explicit:
  - coarse: `stop_reason = "policy_stable"`
  - default: `stop_reason = "value_threshold"`
  - fine: `stop_reason = "value_threshold"`

That means the pipeline is not just functioning mechanically; it is iterating in the right direction. The review points below focus on what still limits the report-quality interpretation of the saved results.

## Review Point 1 — CartPole model coverage improved, but it is still too sparse for a clean discretization conclusion

### Assignment basis

From `RL_Report_Spring_2026_v1-2.md`:

> **Value Iteration (VI) and Policy Iteration (PI):** Compare convergence rates and assess how discretization influences results in the CartPole environment.

> Discuss the effect of discretization on CartPole's solution:
> - Did different levels of discretization impact performance?
> - Were there trade-offs between computational efficiency and accuracy?

These requirements are strongest when the CartPole transition model is good enough that the observed differences are mostly due to discretization, not due to sparse empirical coverage.

### Artifact evidence

From `artifacts/metadata/phase1.json`, the default-grid CartPole model now uses:

- `model_rollout_steps = 2000000`
- `coverage_pct = 0.2517`
- `smoothed_pct = 0.5712`

From `artifacts/logs/phase1.log`:

> `Coverage: 25.2% covered (>=5 visits), 57.1% Laplace-only`

and:

> `CartPole coverage 25.2% is below the 80% target`

Phase 3 now reports:

| Grid | States | Rollout steps | Coverage | Smoothed | Mean episode length |
| --- | ---: | ---: | ---: | ---: | ---: |
| coarse | 72 | 500000 | 89.58% | 6.25% | 500.00 / 500.00 |
| default | 864 | 2000000 | 25.17% | 57.12% | 456.79 / 466.23 |
| fine | 4000 | 5000000 | 16.30% | 75.32% | 438.10 / 437.93 |

from `artifacts/metadata/phase3.json` and `discretization_study.csv`.

### Review

This point is **better than before**, but it is still the main limitation of the CartPole DP story.

The rerun clearly helped: default-grid coverage increased and both default/fine policies improved substantially. That is strong evidence that the earlier result really was being held back by model quality. But the absolute coverage levels are still low enough that the default and fine grids remain heavily confounded by sparsity:

- default still relies on smoothing for more than half the state-action space
- fine still relies on smoothing for roughly three quarters of the state-action space

So the current writeup can make a **qualified** discretization claim, but not yet the cleanest possible one. The safe interpretation is:

1. coarse discretization is well supported by the rollout model and performs excellently
2. increasing grid resolution improves representational capacity in principle
3. in practice, with the current rollout budget, finer grids still suffer from under-covered empirical models

### Recommendation

The report should now explicitly frame the CartPole result as a **joint discretization-and-model-coverage trade-off**, not only a discretization trade-off. If there is time for another iteration, the highest-value improvement remains better CartPole model quality.

## Review Point 2 — Phase 3 PI reporting is more honest now, but default/fine still do not fully stabilize

### Assignment basis

From `RL_Report_Spring_2026_v1-2.md`:

> Compare the performance of VI vs. PI:
> - How many iterations were needed for convergence?
> - Which algorithm converged faster? Why?
> - Did they produce the same optimal policy?

To answer those questions rigorously, it helps if the saved PI runs end with policy stabilization rather than only a very small value-function delta.

### Artifact evidence

The final rows of `artifacts/metrics/phase3_vi_pi_cartpole/pi_convergence.csv` now show:

- coarse: `policy_changes = 0`, `stop_reason = policy_stable`
- default: `policy_changes = 5`, `stop_reason = value_threshold`
- fine: `policy_changes = 19`, `stop_reason = value_threshold`

The Phase 3 log now explicitly warns on the non-coarse grids:

> `stopping on value threshold ... but policy_changes=5 — full policy convergence not reached`

and:

> `stopping on value threshold ... but policy_changes=19 — full policy convergence not reached`

### Review

This is a meaningful improvement over the earlier run because the artifact layer is now honest about what happened. The earlier issue was partly that the output looked like PI had simply converged. The new run fixes that by persisting `stop_reason` and warning in the logs.

But the underlying behavior is still unresolved for the default and fine grids. Those PI runs remain usable as approximate comparison points, yet they should not be described as fully policy-stable convergence results.

That matters for two reasons:

1. it weakens the strongest version of any claim that VI and PI reached the same optimal policy on those grids
2. it makes the PI wall-clock and iteration counts slightly ambiguous, because they are counts to a value threshold rather than to stable policy convergence

### Recommendation

If another code iteration is possible, PI should either:

- continue until policy stabilization, or
- report two stopping notions separately: value-threshold reached vs. policy-stable reached

If no further implementation change is made, the report should explicitly state that default/fine PI stopped on a value threshold before full policy stabilization.

## Review Point 3 — DP hyperparameter validation is now present, but the saved sweep outputs do not fully agree with the main reported results

### Assignment basis

From `RL_Report_Spring_2026_FAQ_v2.md`:

> **Hyperparameters:** For each model you submit (VI, PI, SARSA, Q-Learning, and any DQN/Rainbow EC), you must validate at least **two (2) hyperparameters**.

The FAQ also says:

> Explicitly list the **at least 2 validated hyperparameters** per model and summarize sensitivity (what mattered vs. noise).

and:

> Report ranges, sampling distributions, and the final selection with justification.

From `RL_Report_Spring_2026_v1-2.md`:

> Whatever hypothesis you choose, you will need to back it up with experimentation and thorough discussion. It is not enough to just show results.

### Artifact evidence

This requirement is now substantially addressed:

- `phase2.json` and `phase3.json` both record validated hyperparameters `["gamma", "delta"]`
- `artifacts/metrics/phase2_vi_pi_blackjack/hp_validation.csv` exists
- `artifacts/metrics/phase3_vi_pi_cartpole/hp_validation.csv` exists

However, the selected sweep rows do not fully match the main reported results.

For Phase 2:

- main summary reports `mean_eval_return = -0.0288`
- `hp_validation.csv` at `gamma=0.99, delta=1e-6` reports `mean_eval_return = -0.0408`

For Phase 3 default grid:

- main aggregate reports `456.794` / `466.230`
- `hp_validation.csv` at `gamma=0.99, delta=1e-6` reports `454.79` / `465.98`

The Phase 3 gap is small; the Phase 2 gap is material.

### Review

This means the repository is now in a better place with respect to the FAQ requirement, but the validation story is not fully self-consistent yet.

The problem is no longer "there is no DP hyperparameter validation." The problem is now "the saved validation artifacts do not appear to be exactly the same experiment path as the main reported configuration, or the difference is not documented." That weakens the justification of the final selected settings.

Possible explanations include:

- different evaluation episode counts
- different seed handling
- slightly different execution paths between the main run and the sweep helper
- one artifact being stale relative to another

### Recommendation

Before treating the DP phases as fully report-ready, reconcile the main selected configuration with the sweep row at that same configuration. The cleanest end state is:

- the selected row in `hp_validation.csv` exactly matches the corresponding main reported result, or
- the report and metadata explicitly explain why the sweep numbers are intentionally different

## Overall conclusion

The updated artifacts show **clear progress**:

- CartPole model quality is better than in the earlier run
- Phase 3 performance improved meaningfully
- DP hyperparameter validation is now persisted
- PI stopping semantics are now transparent

So the rerun successfully addressed two major weaknesses from the earlier review.

### Final assessment

- **Phase 1:** improved, but CartPole model quality is still below the intended target
- **Phase 2:** stronger than before, but the main-result vs validation-sweep mismatch should be reconciled
- **Phase 3:** much better and now honestly instrumented, but still not fully clean as a discretization study because of low coverage and non-policy-stable PI on default/fine grids

### Recommended next actions

1. Reconcile the main reported DP results with the selected hyperparameter-sweep rows.
2. Decide whether to invest in one more CartPole model-quality improvement pass.
3. If no more reruns are planned, write the report using the current caveats explicitly: sparse model coverage and PI value-threshold stopping on default/fine grids.
