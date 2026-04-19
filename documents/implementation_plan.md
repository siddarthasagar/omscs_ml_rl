# Implementation Plan — CS7641 RL Spring 2026

## Overview

Two MDPs × four algorithms = eight experiment cells, plus a CartPole discretization study and optional Rainbow DQN extra credit. Everything runs through the phase-gate pipeline: each phase saves a `phase{N}.json` checkpoint for human review before the next phase begins.

## Go-Forward Phase Design Contract

All phases implemented or refactored from this point forward follow the artifact-first lifecycle defined in the steering docs:

- `run()` writes all required artifacts for the phase
- `visualize(checkpoint_path)` reloads only those artifacts from disk and renders figures
- downstream phases consume only declared upstream artifact inputs
- a phase may have zero upstream phase dependencies if it is logically self-contained

This is the target design for the Phase 1–3 refactor and the expected template for Phases 4+.

## Plot Palette Contract

Plotting should use one centralized semantic palette registry in `src/utils/plotting.py` rather than choosing colors figure by figure.

- `ALGO_COLORS`: VI is blue `#4C72B0`, PI is brown `#8C564B`
- `CP_GRID_COLORS`: coarse is muted gray `#BAB0AC`, default is muted teal `#76B7B2`, fine is muted mauve `#B07AA1`
- action maps use their own action palette and do not reuse algorithm colors

Interpretation rule: use algorithm colors for VI-vs-PI comparisons, use grid colors for coarse/default/fine comparisons, and use action colors only when the plotted categories are actions.

---

## Hypothesis (state before experiments begin)

> DP methods (VI/PI) will converge to near-optimal policies in far fewer planning iterations on Blackjack (small exact model) than model-free methods need in episodes, but require substantial upfront sample collection to estimate the CartPole model. Model-free methods avoid that upfront cost but will need far more environment interaction to approach VI/PI policy quality on CartPole. On Blackjack, Q-Learning will converge faster than SARSA due to off-policy bootstrapping, but will show higher variance under sparse stochastic rewards.

*(Refine before submission — this is the anchor for the report's analysis.)*

---

## Seed Semantics

Seeds mean different things for DP vs model-free methods. These must not be conflated.

| Method | What seeds vary | What is deterministic |
|---|---|---|
| VI / PI on Blackjack | Policy evaluation rollouts only (5 seeds × N episodes vs greedy policy) | Planning itself — one run on the exact analytic T/R model |
| VI / PI on CartPole | Policy evaluation rollouts only (5 seeds × N episodes) | Planning — one run on the shared fitted T/R model |
| SARSA / Q-Learning | Full training loop — exploration, Q-init tie-breaks, episode ordering | Nothing; all runs are stochastic |

Reported as: VI/PI convergence curves show a single planning trace (no seed bands). Policy quality results (mean episode length, mean return) show mean ± IQR across 5 evaluation seeds.

---

## Comparison Axes

DP and model-free methods are **not** comparable on a shared sample-efficiency axis. Five separate comparison outputs — never merged:

| Comparison | X-axis | Regime | Methods |
|---|---|---|---|
| Planning efficiency | VI/PI iterations | n/a (deterministic) | VI vs PI, per MDP |
| Learning efficiency | Episodes | controlled | SARSA vs Q-Learning, per MDP |
| Stability | — (aggregate stat) | controlled | SARSA vs Q-Learning, per MDP |
| Final performance | — (aggregate stat) | tuned | All methods, per MDP |
| Wall-clock cost | Seconds | controlled-run only (model-free) | All methods, per MDP |

Five separate comparison outputs — never merged into one chart. Regime is fixed per chart (see Phase 6). Primary learning-curve axis is **episodes** (per FAQ). Cumulative steps logged as secondary column in all model-free CSVs, not used as a chart axis.

**Wall-clock accounting rule:** model-free bars represent the **controlled-regime final reporting run** (5 seeds × full episode budget under the fixed baseline schedule). HP search cost is reported separately as a one-time overhead note in the chart caption — same pattern as CartPole model-build overhead. Tuned-regime run cost is not included in the bar.

**Model-free convergence rule:** `convergence_episode` = first episode E where `|mean_return(E-W:E) − mean_return(E-2W:E-W)| < RL_CONVERGENCE_DELTA` for `RL_CONVERGENCE_M` consecutive window-pairs. Constants in `config.py`: `W=100`, `RL_CONVERGENCE_DELTA=0.01`, `RL_CONVERGENCE_M=3`. Applied per-seed; `convergence_episode_iqr` is the IQR of per-seed convergence episodes across the 5 seeds.

---

## Phase Breakdown

### Phase 1 — Environment Setup, MDP Characterization & CartPole Model

**Goal:** Verify both environments, implement CartPole discretizer, construct and validate the shared T/R model.

**DP model construction:**
- **Blackjack:** `bettermdptools` analytic enumeration — exact T/R, no rollout.
- **CartPole:** one shared T/R model estimated from rollouts. Built once; 5 seeds used only for downstream policy evaluation, not model construction.

**CartPole model estimation procedure:**
1. Roll out `CartPole-v1` under uniform random policy for `CARTPOLE_MODEL_ROLLOUT_STEPS` steps.
2. Bin each `(obs, a, obs', r, done)` using `CartPoleDiscretizer`.
3. Accumulate counts `N(s, a, s')` and reward sums `R(s, a)`.
4. Laplace smoothing (`+1` pseudocount) for unseen `(s, a)` pairs.
5. Terminal/absorbing state: episode-ending transitions map to dedicated `s_term` with `T(s_term, a, s_term)=1`, `R(s_term, a)=0`.
6. Save matrices to `artifacts/metadata/cartpole_model.npz`.

**Coverage diagnostics (logged in `phase1.json`):**
- `coverage_pct`: fraction of `(s,a)` pairs with ≥ `MIN_VISITS` real transitions (target ≥ 80%)
- `smoothed_pct`: fraction receiving Laplace-only coverage
- `mean_visits_covered`: mean visit count for covered pairs
- These are also reported per-grid in Phase 3.

**CartPole discretizer:**
- Default grid `(3, 3, 8, 12)` for `(x, ẋ, θ, θ̇)`
- Clamps: `[-2.4,2.4]`, `[-3,3]`, `[-0.2,0.2]`, `[-3.5,3.5]`
- Non-uniform bin edges for θ and θ̇ (finer near zero) — edges computed in `cartpole_discretizer.py`, exported to `config.py`, and reported verbatim in the reproducibility sheet

**Tests — `tests/test_envs.py` (Gate 1):**
- Blackjack T/R: shape, row-stochasticity, reward range
- CartPole discretizer: all outputs in `[0, n_states)`, full state space enumerable, clamps respected
- CartPole T/R: non-negativity, rows sum to 1, absorbing state well-formed, coverage ≥ threshold

**Output → `artifacts/metadata/phase1.json`:**
```json
{
  "blackjack": { "n_states": ..., "n_actions": 2, "model_source": "bettermdptools_analytic" },
  "cartpole": {
    "bins": [3, 3, 8, 12],
    "n_states": ...,
    "clamps": { "x": [-2.4,2.4], "xdot": [-3.0,3.0], "theta": [-0.2,0.2], "thetadot": [-3.5,3.5] },
    "bin_edges": { "x": [...], "xdot": [...], "theta": [...], "thetadot": [...] },
    "model_rollout_steps": 500000,
    "model_source": "rollout_estimation_laplace_smoothed",
    "coverage_pct": ...,
    "smoothed_pct": ...,
    "mean_visits_covered": ...,
    "min_visits_threshold": 5,
    "absorbing_state_index": ...
  }
}
```

---

### Phase 2 — Model-Based: VI & PI on Blackjack

**Goal:** Establish convergence behaviour on the exact discrete stochastic model.

**Input contract:** none from prior phases. Phase 2 is self-contained because Blackjack planning uses the exact analytic T/R model rather than a fitted upstream artifact.

**Seed usage:** One deterministic planning run per algorithm. Policy quality evaluated over 5 independent seeds (env rollouts of greedy policy).

**Tasks:**
- Implement `src/algorithms/value_iteration.py`, `src/algorithms/policy_iteration.py`
- Load Blackjack T/R from `bettermdptools`
- Run VI once; run PI once. Validate δ and γ as the two required hyperparameters.
- Track per-iteration: max ΔV, policy-change count (PI), wall-clock
- Convergence: `max_s |V_{k+1}(s) − V_k(s)| < δ` for m consecutive sweeps
- Evaluate greedy policy over 5 seeds × 1000 episodes each; report mean return ± IQR

**Figures → `artifacts/figures/phase2_vi_pi_blackjack/`:**
- `blackjack_vi_convergence.png` — ΔV vs iteration (single planning trace)
- `blackjack_pi_convergence.png` — ΔV + policy-change count vs iteration (single trace)
- `blackjack_vi_policy_heatmap.png` — optimal action over (player sum × dealer card), usable ace / no ace panels
- `blackjack_vi_value_heatmap.png` — V(s) heatmap

**Metrics → `artifacts/metrics/phase2_vi_pi_blackjack/`:**
- `vi_convergence.csv` — iteration, delta_v, wall_clock_s
- `pi_convergence.csv` — iteration, delta_v, policy_changes, wall_clock_s
- `policy_eval_per_seed.csv` — algorithm, seed, mean_return
- `policy_eval_aggregate.csv` — algorithm, mean_return, eval_return_iqr (IQR across 5 evaluation seeds)
- `summary.csv` — algorithm, iterations_to_convergence, wall_clock_s, mean_eval_return, eval_return_iqr, policy_match_vi

**Output → `artifacts/metadata/phase2.json`**

---

### Phase 3 — Model-Based: VI & PI on CartPole

**Goal:** Assess how discretization coarseness interacts with DP convergence; disentangle binning quality from model quality.

**Input contract:** None — Phase 3 is self-contained. It builds independent T/R models for every grid (coarse, default, fine) from fresh rollouts and does not load or depend on Phase 1 artifacts. Any reuse of Phase 1 model data would be optional caching, not a phase contract requirement. This design keeps all three grids symmetric — no grid receives special treatment, coverage diagnostics are directly comparable, and the phase reruns correctly without Phase 1 artifacts present.

**Seed usage:** One planning run per algorithm per grid. Policy quality evaluated over 5 seeds × 100 episodes each.

**Tasks:**
- Ablate grids: coarse `(1,1,6,12)` → default `(3,3,8,12)` → fine `(5,5,10,16)`
  - Rebuild T/R model for each grid; record per-grid coverage diagnostics
  - Run **both VI and PI** on each grid; evaluate each policy with 5 seeds
  - Record per-grid policy agreement: action-agreement % across all discrete states + exact match flag
- Separate coverage diagnostics per grid so weak performance can be attributed correctly (binning vs model sparsity)

**Figures → `artifacts/figures/phase3_vi_pi_cartpole/`:**
- `cartpole_vi_convergence.png` — VI residual vs iteration using one representative curve; annotate that coarse/default/fine traces overlap and all converge in 1376 sweeps
- `cartpole_pi_convergence.png` — PI policy changes vs iteration, one curve per grid with stable-iteration markers
- `cartpole_mean_episode_length_by_grid.png` — grouped bar chart: mean episode length by grid, VI vs PI
- `cartpole_wall_clock_by_grid.png` — grouped bar chart: planning wall-clock by grid, VI vs PI
- `cartpole_model_coverage_by_grid.png` — coverage % by grid using the standard grid palette: coarse gray, default teal, fine mauve
- `cartpole_policy_slice.png` — optional qualitative figure if the report discusses where VI and PI differ in state space

**Report comparison note:** use the separate convergence figures for algorithmic behavior, use the three standalone grid charts for discretization tradeoffs, and use a compact summary table for exact wall-clock, final-performance, and policy-agreement values.

**Metrics → `artifacts/metrics/phase3_vi_pi_cartpole/`:**
- `vi_convergence.csv` — grid, iteration, delta_v, wall_clock_s
- `pi_convergence.csv` — grid, iteration, delta_v, policy_changes, wall_clock_s
- `policy_eval_per_seed.csv` — grid, algorithm, seed, mean_episode_len
- `policy_eval_aggregate.csv` — grid, algorithm, mean_episode_len, eval_episode_len_iqr (IQR across 5 evaluation seeds)
- `policy_agreement.csv` — grid, action_agreement_pct, exact_match (VI vs PI per grid)
- `discretization_study.csv` — grid, algorithm, mean_episode_len, eval_episode_len_iqr, iterations_to_conv, wall_clock_s, coverage_pct, smoothed_pct, rollout_steps

**Output → `artifacts/metadata/phase3.json`**

---

### Phase 4 — Model-Free: SARSA & Q-Learning on Blackjack

**Goal:** Compare on-policy vs off-policy learning on the stochastic discrete MDP.

**Seed usage:** Full 5-seed training for all HP search stages and final evaluation.

**Tasks:**
- Implement `src/algorithms/sarsa.py`, `src/algorithms/q_learning.py`
- Staged HP search — **every stage scored over all 5 seeds**:
  - Stage 1: 24 random configs, 200-episode pilot → keep top 8 by mean return across seeds
  - Stage 2: 400 episodes → keep top 3
  - Stage 3: local refinement ±2× α, ±25% decay horizon → champion per algorithm
- Validate ≥2 hyperparameters per algorithm: α schedule, ε schedule (+ γ and Q₀ reported)
- **Primary comparison (controlled):** run both algorithms under the **same fixed baseline exploration schedule**, defined before any tuning: `ε: 1.0 → 0.01 over 10k steps, floor 0.01` (FAQ quick-start defaults). Pre-specified, not derived from any algorithm's search results — ensures a truly neutral comparison that isolates the on-policy vs off-policy effect.
- **Follow-up sensitivity:** run each algorithm under its own separately tuned schedule; compare to controlled result to quantify exploration sensitivity
- Primary learning curve x-axis: **episodes**. Cumulative steps also logged in CSV.

**Figures → `artifacts/figures/phase4_model_free_blackjack/`:**
- `blackjack_sarsa_learning_curve.png` — mean return ± IQR vs episodes (5 seeds, own schedule)
- `blackjack_qlearning_learning_curve.png` — same
- `blackjack_sarsa_vs_qlearning_controlled.png` — overlay under shared exploration schedule
- `blackjack_sarsa_vs_qlearning_tuned.png` — overlay under per-algorithm tuned schedule
- `blackjack_hyperparam_sensitivity.png` — metric vs α / ε-decay, top configs

**Metrics → `artifacts/metrics/phase4_model_free_blackjack/`:**
- `sarsa_curves.csv` — regime, seed, episode, cumulative_steps, mean_return, epsilon (`regime`: `controlled` or `tuned`)
- `qlearning_curves.csv` — same schema
- `hyperparam_search.csv` — stage, config_id, alpha_start, alpha_end, gamma, eps_start, eps_floor, decay_horizon, mean_return_5seeds, std_return_5seeds
- `summary_per_seed.csv` — regime, algorithm, seed, final_mean_return, convergence_episode
- `summary_aggregate.csv` — regime, algorithm, mean_final_return, final_window_iqr, mean_convergence_episode, convergence_episode_iqr

`final_window_iqr`: IQR of mean return over the last 10% of episodes across seeds (stability of achieved performance).
`convergence_episode_iqr`: IQR of convergence episode across seeds (stability of learning speed).

**Output → `artifacts/metadata/phase4.json`**

---

### Phase 5 — Model-Free: SARSA & Q-Learning on CartPole

**Goal:** Same model-free analysis on discretized CartPole; study discretization interaction.

**Tasks:**
- Default grid `(3,3,8,12)` for main comparison
- Re-run staged HP search (all-5-seed scored) for CartPole-specific scales
- **Primary comparison (controlled):** both algorithms under same fixed baseline schedule (`ε: 1.0 → 0.01 over 10k steps, floor 0.01`) — same schedule as Phase 4 controlled runs
- **Follow-up:** per-algorithm tuned schedules; compare to controlled result
- Discretization interaction: **tuned regime only** — champion configs (per-algorithm tuned schedule) under coarse/default/fine grids. Purpose is best-achievable performance per grid; controlled regime would conflate exploration adequacy with binning quality.
- Primary x-axis: **episodes**. Cumulative steps logged.

**Figures → `artifacts/figures/phase5_model_free_cartpole/`:**
- `cartpole_sarsa_learning_curve.png` — mean episode length ± IQR vs episodes
- `cartpole_qlearning_learning_curve.png` — same
- `cartpole_sarsa_vs_qlearning_controlled.png` — overlay under shared exploration schedule
- `cartpole_sarsa_vs_qlearning_tuned.png` — overlay under per-algorithm tuned schedule
- `cartpole_model_free_discretization.png` — grid vs final mean episode length

**Metrics → `artifacts/metrics/phase5_model_free_cartpole/`:**
- `sarsa_curves.csv`, `qlearning_curves.csv` — regime, seed, episode, cumulative_steps, mean_episode_len, epsilon (`regime`: `controlled` or `tuned`)
- `discretization_study.csv` — regime, grid, algorithm, seed, final_mean_len, convergence_episode (`regime` is always `tuned` for this study)
- `summary_per_seed.csv` — regime, algorithm, seed, final_mean_len, convergence_episode
- `summary_aggregate.csv` — regime, algorithm, mean_final_len, final_window_iqr, mean_convergence_episode, convergence_episode_iqr

**Output → `artifacts/metadata/phase5.json`**

---

### Phase 6 — Cross-Method Comparison

**Goal:** Honest side-by-side using correct axes per method type — five separate outputs with fixed regime assignments.

**Tasks:**
- Load phase 2–5 metadata JSONs
- Produce five independent comparison outputs with explicit regime assignments:
  1. **Planning efficiency** — VI vs PI: iterations to convergence, per MDP (bar chart)
  2. **Learning efficiency** — SARSA vs Q-Learning: episodes to convergence, **controlled regime**, per MDP (bar chart) — isolates algorithmic difference
  3. **Stability** — SARSA vs Q-Learning: `final_window_iqr` + `convergence_episode_iqr`, **controlled regime**, per MDP (bar chart)
  4. **Wall-clock** — all methods, per MDP; CartPole DP includes model-build as separate stack
  5. **Final performance** — all methods, **tuned regime** for model-free (best achieved performance), per MDP (bar chart)

Regime assignment is fixed and must not be mixed across charts:
- Controlled regime → charts 2 and 3 only
- Tuned regime → chart 5 only
- DP methods have no regime distinction (single deterministic planning run)

**CartPole grid contract:** all Phase 6 CartPole cross-method outputs use the **default grid `(3,3,8,12)` only**. The coarse/fine ablations are scoped to Phases 3 and 5. This ensures all CartPole bars in Phase 6 refer to one consistent setup.

**Figures → `artifacts/figures/phase6_comparison/`:**
- `planning_efficiency_comparison.png` — VI vs PI iterations, both MDPs
- `learning_efficiency_comparison.png` — SARSA vs Q-Learning episodes (controlled), both MDPs
- `stability_comparison.png` — final-window IQR + convergence-episode IQR (controlled), both MDPs
- `wall_clock_comparison.png` — all methods (CartPole DP stacked: model-build + planning)
- `final_performance_comparison.png` — all methods, model-free from tuned regime, both MDPs

**Output → `artifacts/metadata/phase6.json`**

---

### Phase 7 (Optional EC) — Rainbow DQN Ablation on CartPole

**Goal:** Vanilla DQN + Double DQN ablation; compare on episode axis to tabular methods.

**Tasks:**
- `src/algorithms/dqn.py` — experience replay + target network
- Double DQN variant; 5 seeds × both variants
- Primary x-axis: episodes; cumulative steps logged

**Figures → `artifacts/figures/phase7_dqn_ec/`:**
- `cartpole_dqn_vs_double_dqn.png` — mean return ± IQR, 5 seeds
- `cartpole_dqn_vs_tabular.png` — DQN vs tabular Q-Learning, episode axis

**Output → `artifacts/metadata/phase7.json`**

---

### Phase 8 — Report Tables, Repro Artifacts & Submission Checklist

**Goal:** Read all `phase{N}.json`, emit `.tex` files, generate `report_numbers.tex`, and produce all submission artifacts.

**LaTeX output → `artifacts/tables/`:**
- `tab_phase2_vi_pi_blackjack.tex`
- `tab_phase3_vi_pi_cartpole.tex`
- `tab_phase4_model_free_blackjack.tex`
- `tab_phase5_model_free_cartpole.tex`
- `tab_phase6_comparison.tex`
- `tab_hyperparams.tex` — ≥2 validated HPs per model, all 4 algorithms
- `report_numbers.tex` — `\newcommand` macros for every inline number, including:
  - `\SeedList` — exact seed list `[42, 43, 44, 45, 46]`
  - `\WallClockBlackjackVI`, `\WallClockBlackjackPI`, `\WallClockBlackjackSARSA`, `\WallClockBlackjackQL`
  - `\WallClockCartPoleVI`, `\WallClockCartPolePI`, `\WallClockCartPoleSARSA`, `\WallClockCartPoleQL`
  - `\WallClockCartPoleModel` — CartPole model-build rollout cost (reported as fixed overhead)
  - `\WallClockTotal` — sum of all above; satisfies FAQ requirement to report total experiment wall-clock

**Repro output → `artifacts/repro/`:**

| File | Contents | Owner |
|---|---|---|
| `runbook.md` | Exact Linux commands: `make dev`, `make pipeline`, seeds, expected outputs, figure paths. Also states explicitly: **no external data files** — both environments (`Blackjack-v1`, `CartPole-v1`) are installed via `gymnasium`; all artifacts are generated by the pipeline under `artifacts/`. | Generated by Phase 8 script |
| `ai_use_statement.md` | Draft AI-use statement for insertion into report | Manually authored, stored here before copy into Overleaf |
| `overleaf_link.md` | READ-ONLY Overleaf project URL + confirmed accessible date | Manually recorded |
| `submission_checklist.md` | Tracks all deliverables below | Generated by Phase 8, manually checked off |

**Submission checklist (in `submission_checklist.md`):**
- [ ] `RL_Report_{GTusername}.pdf` — compiled from Overleaf
- [ ] `REPRO_RL_{GTusername}.pdf` — compiled from `artifacts/repro/` contents (runbook + overleaf link + commit SHA)
- [ ] READ-ONLY Overleaf link recorded in `overleaf_link.md`
- [ ] GitHub commit SHA (final push) recorded in `submission_checklist.md`
- [ ] AI-use statement in `ai_use_statement.md` copied into report
- [ ] Mandatory citations present: Sutton & Barto 2018 (Blackjack), Barto et al. 1983 (CartPole)
- [ ] Report ≤ 8 pages including references

---

## Report Outline (section → figure/table map)

| Report Section | Required by assignment | Figures | Tables |
|---|---|---|---|
| MDP descriptions + hypothesis | §3.1, §4 | — | — |
| Methods: VI, PI, SARSA, Q-Learning | §4 | — | — |
| CartPole discretization strategy | §4 | `cartpole_mean_episode_length_by_grid.png`, `cartpole_wall_clock_by_grid.png`, `cartpole_model_coverage_by_grid.png` | bin edges in `tab_hyperparams` |
| VI vs PI results | §4 analysis | Blackjack convergence + Blackjack policy heatmap + `cartpole_vi_convergence.png` + `cartpole_pi_convergence.png` | `tab_phase2`, `tab_phase3` |
| SARSA vs Q-Learning results | §4 analysis | learning curves (controlled + tuned overlays), hyperparam sensitivity | `tab_phase4`, `tab_phase5` |
| Cross-method comparison | §4 analysis | planning efficiency, learning efficiency (controlled), stability (controlled), wall-clock, final performance (tuned) | `tab_phase6` |
| Hyperparameter validation | FAQ req | hyperparam sensitivity | `tab_hyperparams` |
| Extra credit (if done) | §3.1 EC | DQN figures | from `phase7.json` |
| Conclusion | §4 | — | — |
| AI Use Statement | §6 | — | — |
| References | §6 | — | — |

8-page budget: ~1p MDP descriptions, ~1p methods, ~4p results (figures-heavy), ~1p comparison + conclusion, ~0.5p references + AI statement.

---

## Metrics Checklist

### Per algorithm × environment:
- [ ] Iterations (VI/PI) or episodes (SARSA/Q-L) to convergence
- [ ] Final performance: mean return (Blackjack) or mean episode length (CartPole)
- [ ] VI/PI: `mean_eval_return` + `eval_return_iqr` from `policy_eval_aggregate.csv` (IQR across 5 evaluation seeds)
- [ ] SARSA/Q-L: `summary_per_seed.csv` + `summary_aggregate.csv` (per-seed rows and aggregate rows are separate files)
- [ ] **Stability — model-free only:** `final_window_iqr` + `convergence_episode_iqr` in `summary_aggregate.csv`
- [ ] `regime` column in all model-free CSVs: `controlled` (fixed baseline ε schedule) or `tuned` (per-algorithm)
- [ ] Phase 6 regime assignment: controlled → learning efficiency + stability charts; tuned → final performance chart
- [ ] Wall-clock time
- [ ] Convergence diagnostic: ΔV/iteration for VI/PI; running mean return for model-free
- [ ] Cumulative steps logged in all model-free CSVs (secondary column)

### Hyperparameter validation (≥2 per model):
- [ ] VI: δ, γ
- [ ] PI: δ, γ
- [ ] SARSA: α schedule, ε schedule
- [ ] Q-Learning: α schedule, ε schedule, γ, Q₀ init

### CartPole specific:
- [ ] Per-grid coverage %: default + coarse + fine grids
- [ ] Per-grid smoothed % and rollout steps (in `discretization_study.csv`)
- [ ] Per-grid VI vs PI policy agreement: action-agreement % + exact match flag (in `policy_agreement.csv`)
- [ ] Bin edges documented verbatim in `phase1.json` and report
- [ ] Model-build cost reported as fixed overhead (separate stack in wall-clock chart)

### Submission artifacts:
- [ ] Overleaf READ-ONLY link in `artifacts/repro/overleaf_link.md`
- [ ] GitHub commit SHA in `artifacts/repro/submission_checklist.md`
- [ ] Linux runbook in `artifacts/repro/runbook.md` (generated by Phase 8)
- [ ] AI-use statement in `artifacts/repro/ai_use_statement.md`
- [ ] Two mandatory citations: Sutton & Barto 2018, Barto et al. 1983

---

## Figures Checklist

### Blackjack:
- [ ] VI: ΔV vs iteration (single planning trace)
- [ ] PI: ΔV + policy-change count vs iteration (single trace)
- [ ] Optimal policy heatmap (usable ace / no ace panels)
- [ ] Value function heatmap
- [ ] SARSA: mean return ± IQR vs episodes (5 seeds)
- [ ] Q-Learning: mean return ± IQR vs episodes (5 seeds)
- [ ] SARSA vs Q-Learning — controlled (shared exploration schedule)
- [ ] SARSA vs Q-Learning — tuned (per-algorithm schedule)
- [ ] Hyperparameter sensitivity

### CartPole:
- [ ] VI convergence: representative ΔV vs iteration curve with note that all grids overlap and share the same 1376-sweep convergence
- [ ] PI convergence: policy changes vs iteration per grid with stable-iteration markers
- [ ] Mean episode length by grid: grouped bar chart (VI vs PI)
- [ ] Planning wall-clock by grid: grouped bar chart (VI vs PI)
- [ ] Model coverage by grid: single bar chart with the standard coarse/default/fine grid palette
- [ ] Policy slice (optional): qualitative VI vs PI state-space view
- [ ] SARSA: episode length mean ± IQR vs episodes
- [ ] Q-Learning: episode length mean ± IQR vs episodes
- [ ] SARSA vs Q-Learning — controlled (shared exploration schedule)
- [ ] SARSA vs Q-Learning — tuned (per-algorithm schedule)
- [ ] Model-free discretization interaction: grid vs final mean episode length

### Cross-method (5 separate outputs — regime fixed per chart):
- [ ] Planning efficiency: VI vs PI iterations, both MDPs
- [ ] Learning efficiency: SARSA vs Q-Learning episodes, **controlled**, both MDPs
- [ ] Stability: final-window IQR + convergence-episode IQR, **controlled**, both MDPs
- [ ] Wall-clock: all methods bar chart, CartPole DP stacked (model-build + planning)
- [ ] Final performance: all methods, model-free **tuned**, both MDPs

### Extra Credit (if attempted):
- [ ] DQN vs Double DQN (5 seeds, episode axis)
- [ ] DQN vs tabular Q-Learning overlay

---

## Config Constants (`src/config.py`)

```python
from pathlib import Path

# Seeds
SEEDS: list[int] = [42, 43, 44, 45, 46]  # 5 per FAQ

# VI / PI
VI_GAMMA: float = 0.99
VI_DELTA: float = 1e-6
PI_GAMMA: float = 0.99
PI_DELTA: float = 1e-6
VI_PI_CONSEC_SWEEPS: int = 1

# CartPole discretization — default grid
CARTPOLE_BINS: tuple[int, ...] = (3, 3, 8, 12)
CARTPOLE_CLAMPS: dict = {
    "x":        (-2.4, 2.4),
    "xdot":     (-3.0, 3.0),
    "theta":    (-0.2, 0.2),
    "thetadot": (-3.5, 3.5),
}
# Non-uniform bin edges populated by cartpole_discretizer.py; stored here for report reproducibility

# CartPole model estimation
CARTPOLE_MODEL_ROLLOUT_STEPS: int = 500_000
CARTPOLE_MODEL_MIN_VISITS: int = 5

# Model-free convergence criterion (running-mean plateau)
RL_CONVERGENCE_WINDOW: int = 100       # W: window size in episodes
RL_CONVERGENCE_DELTA: float = 0.01    # minimum improvement between consecutive windows
RL_CONVERGENCE_M: int = 3             # number of consecutive window-pairs below delta

# Model-free defaults (starting point for staged HP search)
RL_GAMMA: float = 0.99
RL_ALPHA_START: float = 0.5
RL_ALPHA_END: float = 0.1
RL_EPS_START: float = 1.0
RL_EPS_END: float = 0.01
RL_EPS_DECAY_STEPS: int = 10_000

# Paths
DATA_DIR: Path = Path("data")
ARTIFACTS_DIR: Path = Path("artifacts")
```

---

## Makefile Targets to Add

```makefile
phase1     # env setup + CartPole model estimation
phase2     # VI/PI on Blackjack
phase3     # VI/PI on CartPole + discretization study
phase4     # SARSA/Q-Learning on Blackjack (HP search + final eval)
phase5     # SARSA/Q-Learning on CartPole
phase6     # cross-method comparison
phase7     # DQN EC (optional)
phase8     # report tables + repro artifacts

gate1      # test_envs — run after Phase 1
gate2      # test_algorithms — run after Phase 2
gates      # gate1 + gate2

pipeline   # phase1 → phase2 → phase3 → phase4 → phase5 → phase6 → phase8
overnight  # pipeline in detached tmux/screen session
```
