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

No single-seed results submitted. Report-facing aggregate statistics use the documented variability convention for the relevant output type, always over the 5 seeds.

## Preferred Phase Lifecycle (go-forward contract)

All phases implemented or refactored going forward should expose the same lightweight lifecycle:

- `run() -> Path`: compute results, write artifacts, and return the checkpoint JSON path
- `visualize(checkpoint_path: Path) -> list[Path]`: reload artifacts from disk only and render all figures for that phase

Preferred orchestration is decoupled:

- phase `__main__` entrypoints should focus on computation and artifact writing,
- the standalone renderer `scripts/visualize_all.py` should load saved checkpoints and call `visualize(checkpoint_path)` in a fresh process,
- `scripts/visualize_all.py --phase phase3` is the target single-phase rerender interface,
- running the renderer without a phase filter should rerender every phase whose checkpoint exists,
- `make viz` is the repository-wide rerender command and may clear `artifacts/figures/` first.

This avoids log leakage between long-running experiments and figure generation, while preserving the per-phase `visualize()` contract. A phase may still call `visualize()` inline during migration or local debugging, but that is no longer the preferred default.

The same isolation principle applies to experimentation vs validation runs:

- canonical phase logs under `artifacts/logs/phase{N}.log` are for real workspace experiment runs,
- temporary tests, smoke runs, and debug runs should write to isolated artifact locations or use distinct run IDs,
- test/debug artifacts must never be allowed to look like the authoritative output of a completed project phase.

This is a function contract, not a requirement for a heavy class hierarchy.

## Artifact Boundary Rules

- A phase may declare zero or more upstream artifact inputs.
- A phase never consumes another phase's Python runtime variables.
- Plotting never consumes a phase's live Python runtime variables.
- When CSV/JSON are insufficient for plotting, persist a small `plot_*.npz` support artifact and reload it inside `visualize()`.
- Shared downstream model artifacts belong in `artifacts/metadata/`; phase-local plot-support artifacts belong next to metrics.

## Target Checkpoint Schema (for refactored phases 1–3 and future phases)

The checkpoint JSON remains human-readable, but it should also serve as a machine-readable manifest for `visualize()` and downstream phases.

Target top-level fields:

- `schema_version`
- `phase_id`
- `slug`
- `upstream_inputs`
- `outputs`
- `config_snapshot`
- `summary`

The `outputs` section should list the metrics directory, figure directory, and any plot-support or shared model artifacts needed to reproduce figures or feed downstream phases.

## Plotting Contract

Plotting should be dynamic from saved metrics and metadata wherever possible:

- curve extents
- legend entries
- titles and annotations derived from saved run context
- error bars and value ranges

The plotting layer should keep only fixed semantic defaults centralized in `src/utils/plotting.py`:

- algorithm colors
- grid colors
- action colors
- reference and annotation colors
- environment axis ordering
- DPI and readable font defaults
- environment-defined markers such as CartPole terminal thresholds

### Color Semantics

Color should represent one primary semantic layer at a time. The plotting layer should keep that contract explicit with named constants.

Recommended constants in `src/utils/plotting.py`:

- `ALGO_COLORS = {"VI": "#4C72B0", "PI": "#8C564B"}`
- `CP_GRID_COLORS = {"coarse": "#BAB0AC", "default": "#76B7B2", "fine": "#B07AA1"}`
- `BJ_ACTION_COLORS` and `CP_ACTION_COLORS` as action-specific palettes separate from algorithms and grids
- `REFERENCE_COLORS` for thresholds and convergence markers

Rules:

- Reserve blue and brown for VI and PI across the repository.
- Use `CP_GRID_COLORS` whenever marks represent `coarse`, `default`, and `fine`, including coverage-by-grid figures.
- Use action colors only for policy maps and action-slice figures; action colors must not imply algorithm identity.
- Threshold lines, stable-iteration markers, and other annotations should use muted reference colors rather than algorithm or grid colors.

## Report-Facing Figure Design

Report-facing figures should maximize explanatory value, not maximize the amount of data forced into one image.

Guidelines:

- Convergence figures are diagnostic evidence; they do not need to carry every comparison result by themselves.
- When two algorithms converge in different natural objects, prefer separate figures over forced composite comparisons.
- When multiple traces are numerically or visually indistinguishable, prefer a representative curve plus an explicit annotation over a low-value overlay.
- Use standalone tradeoff charts or compact tables for exact wall-clock, final-performance, and agreement comparisons that are clearer outside the convergence figure.

Applied example for DP phases:

- VI may be best shown with `delta_v` vs iteration.
- PI may be best shown with policy changes vs iteration.
- Grid tradeoffs such as episode length, wall-clock, and model coverage may be clearer as separate categorical charts than as one composite panel.

## Reporting Conventions

**Output schema — model-free phases:**
- `mf_learning_curves.csv` — algorithm, seed, regime, episode, window_mean (one row per window checkpoint per seed)
- `mf_hp_search.csv` — per-config HP search results; ranked by `mean_return` (= win_rate − loss_rate for Blackjack) or `mean_episode_len` (CartPole)
- `mf_eval_per_seed.csv` — one row per (algorithm, seed, regime); includes mean_return, final_window_return, convergence_episode
- `mf_eval_summary.csv` — long-format aggregate: one row per (algorithm, regime, metric) with mean/std/iqr columns

**Stability metrics (model-free only):**
- `final_window_iqr` — IQR of `final_window_return` (last window-mean from training curve) across 5 seeds; stored in `phase{N}.json` checkpoint summary per (regime, algorithm)
- `convergence_episode_iqr` — IQR of `convergence_episode` across 5 seeds; stored in `phase{N}.json` checkpoint summary

**Model-free convergence rule (`convergence_episode`):**
Running-mean plateau: first episode E where `|mean_return(E-W:E) − mean_return(E-2W:E-W)| < delta` for `RL_CONVERGENCE_M` consecutive window-pairs.
Constants: `W=100`, `RL_CONVERGENCE_M=3` (in `config.py`).
Delta is signal-scaled per environment: `RL_CONVERGENCE_DELTA=0.01` for Blackjack (return in [-1,1]); `CP_RL_CONVERGENCE_DELTA=10.0` for CartPole (episode length in [1,500], ~2% of range).

**Variability convention for model-free plots:**
- Learning curves (line plots): mean ± std band across seeds (smooth; std is appropriate for window-averaged time series).
- Bar charts (comparison, discretization): mean with ±IQR/2 whiskers (robust to seed outliers; IQR value shown in legend label).

**Wall-clock accounting:**
- `train_wall_clock_s` in checkpoint = controlled+tuned final training only (timed from job submission to collection, before discretization study).
- `disc_wall_clock_s` = discretization study separately timed.
- Model-free bars in the wall-clock chart = controlled-regime `train_wall_clock_s`. HP search cost noted separately as overhead.
- CartPole DP bar = model-build rollout cost (stacked) + planning run.

**DP evaluation outputs:**
- `policy_eval_per_seed.csv` — one row per (algorithm, seed)
- `policy_eval_aggregate.csv` — one row per `(algorithm)` for Blackjack; one row per `(grid, algorithm)` for CartPole; includes `eval_return_iqr` or `eval_episode_len_iqr`
- Optional `plot_*.npz` artifacts may be written when figures need arrays or decoded grids that are not recoverable from summary CSVs alone.

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

| Stage | Configs | Episodes/seed (Blackjack) | Episodes/seed (CartPole) | Keep |
|---|---|---|---|---|
| 1 (coarse random) | 24 | 20,000 | 2,000 | top 8 by mean return / mean episode len |
| 2 (promotion) | 8 | 50,000 | 5,000 | top 3 |
| 3 (local refinement) | ±2× α, ±25% decay | 100,000 | 10,000 | champion |

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
uv run python scripts/visualize_all.py          # standalone rerender of all available phases
bash ml_run.sh "make <target>"             # with sleep prevention (inline)
bash ml_run.sh --detach "make <target>"    # background tmux/screen session
make viz                                    # wipe figures and rerender all available phases
```

Target filtered visualization interface:

```
uv run python scripts/visualize_all.py --phase phase3
```
