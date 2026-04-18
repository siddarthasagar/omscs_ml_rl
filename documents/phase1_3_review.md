# Phase 1-3 Implementation Review

This document reviews the current implementation state of Phases 1 through 3 against the official assignment requirements in:

- `documents/canvas/assignment/md/RL_Report_Spring_2026_v1-2.md`
- `documents/canvas/assignment/md/RL_Report_Spring_2026_FAQ_v2.md`

This version reflects the current artifacts under `artifacts/` and corrects a few stale claims from the previous review. Relative to the earlier state, the implementation has clearly improved: CartPole rollout budgets are larger, DP hyperparameter validation is now persisted as CSV artifacts, and PI stop reasons are now made explicit in logs and checkpoint JSONs.

The main remaining issues are:

- **CartPole model sparsity**, especially on the default and fine grids
- **VI/PI divergence on the Phase 3 default and fine grids**, even after PI now reaches full policy stabilization
- **Phase 3 hyperparameter-validation narrative is stale relative to the latest rerun**, especially for VI sensitivity to `delta`
- **under-documented differences between main evaluation outputs and hyperparameter-sweep outputs**, which are explainable from the code but not yet clearly stated in the artifact narrative

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
- Phase 1 generates the expected persisted artifacts:
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
- Main reported results are strong and internally coherent:
  - VI converges in `9` iterations
  - PI converges in `3` iterations
  - both main runs evaluate to `-0.0288`
  - saved policy agreement is `1.0`
  - PI stops with `policy_changes_at_convergence = 0` and `stop_reason = "policy_stable"`
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
- The latest rerun materially changes the CartPole DP comparison:
  - coarse grid remains `500.00 / 500.00` mean episode length for VI / PI
  - default grid is now `456.79 / 465.47`
  - fine grid is now `434.88 / 436.71`
  - fine-grid coverage remains `16.30%`
- The PI stabilization issue is now resolved in the saved artifacts:
  - coarse: `stop_reason = "policy_stable"`, `policy_changes_at_convergence = 0`
  - default: `stop_reason = "policy_stable"`, `policy_changes_at_convergence = 0`
  - fine: `stop_reason = "policy_stable"`, `policy_changes_at_convergence = 0`
- VI and PI still agree strongly on the non-coarse grids, but they no longer match exactly:
  - default agreement: `98.38%`
  - fine agreement: `98.95%`
  - `policy_agreement.csv` records `exact_match = False` for both default and fine

That means the pipeline is not just functioning mechanically; it is producing sensible, reviewable artifacts. The remaining issues are about how strong the report claims can be, not whether Phases 1 through 3 are fundamentally broken.

## Review Point 1 - CartPole model coverage is still too sparse for a clean discretization conclusion

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

Phase 3 currently reports:

| Grid | States | Rollout steps | Coverage | Smoothed | Mean episode length (VI / PI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| coarse | 72 | 500000 | 89.58% | 6.25% | 500.00 / 500.00 |
| default | 864 | 2000000 | 25.17% | 57.12% | 456.79 / 465.47 |
| fine | 4000 | 5000000 | 16.30% | 75.32% | 434.88 / 436.71 |

from `artifacts/metadata/phase3.json` and `discretization_study.csv`.

### Review

This point is still one of the main limitations of the CartPole DP story.

The rerun clearly helped at the policy-performance level: all saved PI runs are now policy-stable, and the current VI results no longer show the earlier extraction-driven instability. But the absolute coverage levels are still low enough that the default and fine grids remain heavily confounded by sparsity:

- default still relies on smoothing for more than half the state-action space
- fine still relies on smoothing for roughly three quarters of the state-action space

So the current writeup can make a **qualified** discretization claim, but not yet the cleanest possible one. The safe interpretation is:

1. coarse discretization is well supported by the rollout model and performs excellently
2. increasing grid resolution improves representational capacity in principle
3. in practice, with the current rollout budget, finer grids still suffer from under-covered empirical models

### Recommendation

The report should explicitly frame the CartPole result as a **joint discretization-and-model-coverage trade-off**, not only a discretization trade-off.

If there is time for another iteration, the highest-value improvement remains better CartPole model quality. If there is not time for another rerun, this issue becomes a reporting caveat rather than a blocker.

## Review Point 2 - Phase 3 PI stabilization is fixed, but VI and PI now separate materially on both non-coarse grids

### Assignment basis

From `RL_Report_Spring_2026_v1-2.md`:

> Compare the performance of VI vs. PI:
> - How many iterations were needed for convergence?
> - Which algorithm converged faster? Why?
> - Did they produce the same optimal policy?

To answer those questions rigorously, it helps if both methods are genuinely comparable at convergence and if any remaining disagreement is interpreted correctly.

### Artifact evidence

The final rows of `artifacts/metrics/phase3_vi_pi_cartpole/pi_convergence.csv` now show:

- coarse: `policy_changes = 0`, `stop_reason = policy_stable`
- default: `policy_changes = 0`, `stop_reason = policy_stable`
- fine: `policy_changes = 0`, `stop_reason = policy_stable`

That resolves the earlier PI-stopping caveat.

However, `artifacts/metrics/phase3_vi_pi_cartpole/policy_agreement.csv` now reports:

- coarse: `100.0%`, `exact_match = True`
- default: `98.38%`, `exact_match = False`
- fine: `98.95%`, `exact_match = False`

And `artifacts/metrics/phase3_vi_pi_cartpole/policy_eval_aggregate.csv` reports:

- default: VI `456.794`, PI `465.468`
- fine: VI `434.880`, PI `436.712`

At the per-seed level:

- PI outperforms VI on all five default-grid seeds in `policy_eval_per_seed.csv`
- PI slightly outperforms VI on three of five fine-grid seeds, and the aggregate mean also favors PI

### Review

This is a real improvement over the earlier run because the PI non-convergence issue is gone. The saved PI results are now easier to defend.

But fixing that issue exposed a different interpretive problem: on the non-coarse grids, VI and PI are no longer effectively interchangeable under the saved evaluation.

That matters for two reasons:

1. the report should no longer imply that VI and PI reached the same policy on the non-coarse grids
2. under the current saved evaluation, PI is better on the default grid and slightly better on the fine grid

This does not necessarily mean the implementation is broken. Possible explanations include:

- non-unique near-optimal policies under the discretized empirical model
- a VI stopping threshold or policy-extraction detail that is slightly less favorable than PI on this model
- evaluation sensitivity under sparse model regions

But whatever the explanation, the saved results should now be written as substantive differences, not as near-identity.

### Recommendation

If another code iteration is possible, the next high-value investigation is no longer PI stopping. It is understanding why VI and stable PI disagree on the default and fine grids.

Specifically:

- verify whether VI policy extraction and stopping criteria are as intended
- confirm whether the differences persist under a larger evaluation budget
- if no implementation change is made, report the current result honestly: `PI > VI` on the default grid and a smaller `PI > VI` gap on the fine grid under the saved setup, rather than claiming equivalence

## Review Point 3 - DP hyperparameter validation is present, but the saved Phase 3 narrative is stale relative to the latest rerun

### Assignment basis

From `RL_Report_Spring_2026_FAQ_v2.md`:

> **Hyperparameters:** For each model you submit (VI, PI, SARSA, Q-Learning, and any DQN/Rainbow EC), you must validate at least **two (2) hyperparameters**.

The FAQ also says:

> Explicitly list the **at least 2 validated hyperparameters** per model and summarize sensitivity (what mattered vs noise).

and:

> Report ranges, sampling distributions, and the final selection with justification.

From `RL_Report_Spring_2026_v1-2.md`:

> Whatever hypothesis you choose, you will need to back it up with experimentation and thorough discussion. It is not enough to just show results.

### Artifact evidence

This requirement is now substantially addressed:

- `phase2.json` and `phase3.json` both record validated hyperparameters `["gamma", "delta"]`
- `artifacts/metrics/phase2_vi_pi_blackjack/hp_validation.csv` exists
- `artifacts/metrics/phase3_vi_pi_cartpole/hp_validation.csv` exists

The selected sweep rows do not numerically match the main reported results.

For Phase 2:

- main summary reports `mean_eval_return = -0.0288`
- `hp_validation.csv` at `gamma=0.99, delta=1e-6` reports `mean_eval_return = -0.0408`

For Phase 3 default grid:

- main aggregate reports `456.794` / `465.468`
- `hp_validation.csv` at `gamma=0.99, delta=1e-6` reports `454.79` / `464.64`

But the more important new detail is inside `artifacts/metrics/phase3_vi_pi_cartpole/hp_validation.csv` itself.

For VI on the default grid:

- gamma sweep: `454.83`, `454.79`, `454.69`, `454.87`
- delta sweep: `457.91`, `457.90`, `457.90`, `454.87`

For PI on the default grid:

- gamma sweep: `463.68`, `463.68`, `463.68`, `464.64`
- delta sweep: `464.64` across all tested deltas

However, the code explains most of this gap:

- Phase 2 main evaluation uses `1000` episodes per seed, while the HP sweep uses `500`
- Phase 3 main evaluation uses `100` episodes per seed, while the HP sweep uses `50`

So these outputs are not necessarily contradictory. They are generated under different evaluation budgets.

### Review

This means the repository is in a better place with respect to the FAQ requirement than the previous review implied.

The problem is no longer `there is no DP hyperparameter validation`, and it is not automatically `the artifacts are inconsistent`. The more precise issue is:

- the repository now validates DP hyperparameters
- the sweep outputs are directionally useful
- but the artifact and report narrative do not yet explain that the sweep helper intentionally uses lighter evaluation budgets than the final main-report runs
- and, on Phase 3, the saved metadata note still does not match the current sweep rows

That is chiefly a documentation and interpretation issue, not clear evidence of a broken experiment path. On Phase 3, the selected-row gap is still small enough to look consistent with the lighter sweep budget. The current VI sweep no longer shows catastrophic collapse, but it does still show that `delta` changes final performance modestly, so the saved note claiming that `delta` only changes convergence speed for values `<= 1e-3` remains inaccurate.

### Recommendation

Before sign-off, document this explicitly in the report and, ideally, in the metadata narrative:

- the sweep outputs were generated with lighter evaluation budgets to keep validation cheap
- the selected sweep rows should be compared directionally, not expected to match the final main-report values exactly
- for CartPole VI on the default grid, `gamma` is now fairly flat in the saved sweep, while `delta` still changes final policy quality modestly

If you want the cleanest possible story, rerun the selected final HP row under the full main evaluation budget and note that it agrees with the main result. But that is a strengthening step, not the only acceptable resolution.

## Overall conclusion

The updated artifacts show clear progress:

- CartPole model quality is better than in the earlier run
- Phase 3 performance improved materially
- DP hyperparameter validation is now persisted
- PI now reaches full policy stabilization on all three saved grids

So the rerun successfully removed one major weakness from the earlier review. The remaining gaps are narrower and better understood, but the non-coarse-grid VI/PI discrepancies and the stale Phase 3 HP-validation note are now the main comparison issues.

## Visualization recommendations for the report

These are not experimental blockers, but they matter for how convincingly the report communicates the results under a tight page budget.

### Blackjack

- A research check changes the earlier recommendation here: **3D Blackjack value surfaces are legitimate and canonical in Sutton-and-Barto-style presentations**, so they are not a gimmick.
- However, that does **not** automatically make them the best figure for this report. In a short paper, 3D plots cost space and are weaker for exact comparison than a clean 2D view.
- The current value heatmap is therefore acceptable to keep if report space is tight.
- If you want one explicitly textbook-style Blackjack figure, the best use of 3D is still:
  - X-axis = dealer card
  - Y-axis = player sum
  - Z-axis = value `V(s)`
  - one panel for no usable ace
  - one panel for usable ace
- The **highest-value Blackjack visual** is still a crisp 2D policy decision-region plot:
  - solid discrete colors for hit vs stick
  - no smoothing or interpolation
  - sharp separation between action regions
  - one panel for usable ace and one for no usable ace

### CartPole

- The current CartPole convergence and discretization figures should remain the **core figures** because they directly support the assignment questions.
- For **VI vs PI comparison**, same-figure plots are usually better than fully independent figures **when the metric is genuinely comparable on the same axis**.
- The current discretization-study figure already follows the right pattern: VI and PI are plotted together for policy quality and planning time, so the reader can compare them directly without mentally aligning separate plots.
- Raw VI and PI convergence should **not** be forced onto one overlaid line chart just because both use iteration on the x-axis.
- In this project, a VI iteration and a PI iteration are not the same unit of work, and the PI trace also carries policy-change events. A single overlay would therefore make the comparison look cleaner than it really is.
- If the convergence presentation is revised, the better format is a **shared comparison figure with aligned subplots** or side-by-side panels using matched scales, not one merged overlay.
- If one additional policy-visualization figure is added, the highest-value option is a **sliced decision-boundary plot** in `(theta, thetadot)` space while fixing `x = 0` and `xdot = 0`.
- That kind of plot is more useful for this report than a raw rollout trajectory because it shows what the learned controller chooses across a slice of state space, not just one realized episode.
- Phase-space trajectory plots can still be useful as supplementary figures, but they should not replace the discretization-study figure or the convergence figures.

### Recommendation

If report space is tight, the highest-value visualization upgrades are:

1. Keep the existing CartPole discretization figure as the main VI-vs-PI comparison figure.
2. Make the Blackjack policy plots look like crisp decision-region maps.
3. If you rework CartPole convergence presentation, prefer a shared figure with aligned VI and PI subplots rather than two totally separate figures or one overlaid graph.
4. Add Blackjack 3D value-function surfaces only if you want an explicit Sutton-and-Barto-style figure and still have room.
5. Add optional CartPole `(theta, thetadot)` sliced policy maps if room remains.

I would not prioritize CartPole rollout trajectories over the current convergence and discretization figures.

## Final assessment

- **Phase 1:** functionally done, with the standing CartPole coverage caveat
- **Phase 2:** functionally done; the main remaining task is documenting why HP-sweep numbers and main-report numbers differ
- **Phase 3:** materially stronger than before because PI now fully stabilizes and VI extraction is cleaner, but still not ready for an unqualified claim because of low model coverage, remaining VI/PI disagreement, and a stale HP-validation narrative

## Sign-off status

**Recommended status: conditional sign-off for Phases 1 through 3, still not unconditional sign-off.**

I would treat these phases as done once the reporting caveats below are carried into the report or final review notes. I would only hold back sign-off if you want to claim a stronger CartPole discretization result than the current artifacts support, if you want to claim that VI and PI produced the same policy on the non-coarse grids, or if you want to leave the current Phase 3 HP-validation note uncorrected.

## Next steps required before sign-off

1. Update the report text to describe CartPole as a **discretization plus model-coverage trade-off**, not as a pure discretization study.
2. State explicitly that Phase 3 PI now reaches policy-stable convergence on all saved grids, but VI and PI do **not** produce identical policies on the default and fine grids.
3. Report the current non-coarse-grid comparison honestly: `PI > VI` on the default grid, and a smaller `PI > VI` gap on the fine grid, under the saved evaluation.
4. Update the Phase 3 hyperparameter-validation narrative to reflect the current CSV rows: the earlier severe VI brittleness is no longer present, but `delta` still changes final VI performance modestly and the current metadata note should be corrected.
5. Document that the DP hyperparameter sweeps used smaller evaluation budgets than the final main runs, so `hp_validation.csv` should be interpreted directionally rather than as an exact numeric duplicate of the final report row.

## Optional strengthening steps

1. Increase CartPole rollout budgets again and rerun Phase 1 plus Phase 3 if you want a cleaner discretization conclusion.
2. Investigate why VI and stable PI disagree on the default and fine grids, especially whether the VI stopping criterion or policy extraction should be tightened.
3. Rerun the selected DP hyperparameter row at the full main evaluation budget and persist that comparison if you want a perfectly apples-to-apples validation story.

If those optional strengthening steps are skipped, the phases can still be signed off as complete for the DP portion of the project, provided the report keeps the current caveats explicit.
