---
inclusion: always
---

# Tech — CS7641 RL Spring 2026

## Stack

- **Python:** 3.13
- **Package manager:** uv
- **RL environments:** `gymnasium` (`Blackjack-v1`, `CartPole-v1`)
- **MDP utilities:** `bettermdptools` (T/R matrix construction, VI/PI wrappers), `pymdptoolbox`
- **Data:** pandas, numpy
- **Viz:** matplotlib
- **Testing:** pytest
- **Linting:** ruff

## Seeds

| Constant | Value | Use |
|----------|-------|-----|
| `SEEDS` | `[42, 43, 44, 45, 46]` | 5 seeds per FAQ |

Seed semantics differ by method type:
- **VI / PI:** planning is deterministic on a fixed T/R model — one planning run. Seeds apply only to **policy evaluation** rollouts (5 × N episodes against the greedy policy).
- **SARSA / Q-Learning:** full training is stochastic — 5 seeds vary the entire training loop.

No single-seed results submitted. All reported performance metrics are mean ± IQR across the 5 seeds.

## Reporting Conventions

**Output schema — model-free phases:**
- `*_curves.csv` — per-seed, per-episode rows with a `regime` column (`controlled` or `tuned`)
- `summary_per_seed.csv` — one row per (regime, algorithm, seed)
- `summary_aggregate.csv` — one row per (regime, algorithm); includes stability metrics

**Stability metrics (model-free only):**
- `final_window_iqr` — IQR of mean return/episode-length over the last 10% of episodes, across 5 seeds
- `convergence_episode_iqr` — IQR of `convergence_episode` across 5 seeds

**Model-free convergence rule (`convergence_episode`):**
Running-mean plateau: first episode E where `|mean_return(E-W:E) − mean_return(E-2W:E-W)| < RL_CONVERGENCE_DELTA` for `RL_CONVERGENCE_M` consecutive window-pairs.
Constants: `W=100`, `RL_CONVERGENCE_DELTA=0.01`, `RL_CONVERGENCE_M=3` (all in `config.py`).

**Wall-clock accounting:**
- Model-free bars in the wall-clock chart = controlled-regime final reporting run (5 seeds × full episode budget). HP search cost noted separately as overhead.
- CartPole DP bar = model-build rollout cost (stacked) + planning run.

**DP evaluation outputs:**
- `policy_eval_per_seed.csv` — one row per (algorithm, seed)
- `policy_eval_aggregate.csv` — one row per `(algorithm)` for Blackjack; one row per `(grid, algorithm)` for CartPole; includes `eval_return_iqr` or `eval_episode_len_iqr`

## DP Model Construction

| Environment | Method |
|---|---|
| `Blackjack-v1` | Use `bettermdptools` built-in `Blackjack` wrapper — provides exact T and R matrices by analytic enumeration |
| `CartPole-v1` | Estimate T and R matrices by rolling out the gym env under a uniform random policy; bin observations using the `CartPoleDiscretizer`; accumulate empirical transition counts and reward averages |

## CartPole Discretization

Default grid: `(3, 3, 8, 12)` for `(x, ẋ, θ, θ̇)`.

Clamps before binning:

| Feature | Clamp |
|---|---|
| Cart position x | `[-2.4, 2.4]` |
| Cart velocity ẋ | `[-3.0, 3.0]` |
| Pole angle θ | `[-0.2, 0.2]` |
| Pole angular velocity θ̇ | `[-3.5, 3.5]` |

Bin edges are **non-uniform**: angle and angular velocity use finer resolution near zero. Exact edges stored in `config.py` and reported verbatim in the reproducibility sheet.

Test: `test_discretizer_coverage` — verifies every binned state index is in `[0, n_states)` and that the full state space is enumerable (not invertibility, which is undefined for a many-to-one map).

## Hyperparameter Search Protocol (FAQ-compliant)

Staged search, each stage scored over **all 5 seeds**:

| Stage | Configs | Episodes/seed | Keep |
|---|---|---|---|
| 1 (coarse random) | 24 | 200 | top 8 by mean return |
| 2 (promotion) | 8 | 400 | top 3 |
| 3 (local refinement) | ±2× α, ±25% decay | 1000 | champion |

Champion evaluated over 5 seeds for full episode budget. Report search protocol, ranges, budgets, and sensitivity (which hyperparameters moved the metric vs. noise).

Validated hyperparameters (≥2 per model):
- **VI/PI:** δ (convergence threshold), γ (discount)
- **SARSA:** α schedule (start, end, decay law), ε schedule (start, floor, decay horizon)
- **Q-Learning:** α schedule, ε schedule, γ, Q₀ initialisation strategy

## Logging Standard

Every phase script calls `configure_logger(run_id)` from `src/utils/logger.py`.
`run_id` pattern: `phase{N}` (e.g. `phase2`). No bare `print()`. tqdm exempt.

## Build Commands

```
make dev                                    # venv + all deps (incl. dev)
make test                                   # full pytest
make lint / make format                     # ruff
uv run python scripts/run_phase_{N}_{slug}.py   # phase entrypoints
bash ml_run.sh "make <target>"             # with sleep prevention (inline)
bash ml_run.sh --detach "make <target>"    # background tmux/screen session
```
