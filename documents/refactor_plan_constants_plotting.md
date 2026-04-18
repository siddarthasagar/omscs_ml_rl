# Refactor Plan — Constants, Evaluation Budgets, and Dynamic Plotting

## Objective

This document describes a focused refactor plan for two related goals:

1. Make every important constant easy to locate, explain, and defend in the report.
2. Move evaluation budgets and plotting style out of phase scripts into a cleaner configuration and plotting layer.

The refactor is intentionally scoped to changes that add direct value for report generation, reproducibility, and future iteration safety. It is not a general architecture rewrite.

---

## Why This Refactor Is Worth Doing

The current project already centralizes many experiment constants in `src/config.py`, but there are still three pain points:

1. Some constants that affect reported results still live inside phase scripts.
2. Plotting code mixes scientific semantics, report styling, and local script defaults.
3. It is harder than it should be to explain, in one place, why each constant exists and how it was chosen.

For the report, the most important outcome is not "fewer constants." The most important outcome is:

- each reported number can be traced to a documented configuration choice,
- each configuration choice can be described honestly as requirement-driven, environment-driven, budget-driven, or style-driven,
- plots can be regenerated from saved artifacts without hand-maintaining plot-specific logic in every phase script.

---

## Scope

### In Scope

- Reorganize constants into explicit sections with developer comments that explain why they exist.
- Move evaluation budgets out of phase scripts and into `src/config.py`.
- Move plotting style defaults and common plotting semantics into `src/utils/plotting.py`.
- Refactor Phase 2 and Phase 3 plotting so figures are generated from written CSV/JSON artifacts rather than hand-built from in-memory local variables only.
- Keep plots as data-driven as practical so iterative code changes are less likely to break figures.

### Out of Scope

- Rewriting algorithm implementations.
- Changing the phase-gate workflow.
- Introducing external configuration systems such as YAML, Hydra, or Pydantic.
- Making every style choice runtime-configurable.
- Refactoring phases 4-8 before those scripts exist.

---

## Design Principles

### 1. Experiment protocol should be fixed and documented

Constants that define the experiment contract should stay explicit and stable.

Examples:

- seeds,
- default gamma and delta,
- CartPole discretization edges,
- hyperparameter sweep grids,
- evaluation episode budgets.

These should not be inferred from whatever happened to run last.

### 2. Plot generation should be data-driven

Plots should derive their curve extents, panel contents, labels, and titles from saved metrics and metadata.

Important note: metadata is also data. A plot title such as `gamma=0.99, delta=1e-6` should be read from the saved phase metadata, not repeated manually inside the plotting function.

### 3. Semantic display choices should remain centralized

Some plot choices should stay fixed for readability and cross-run comparability.

Examples:

- VI is always blue, PI is always orange or green according to the chosen palette,
- Blackjack actions use the same Stick/Hit color mapping across figures,
- grid order is always coarse → default → fine,
- terminal angle markers stay tied to the CartPole environment definition.

These are not fragile "hardcodes." They are stable semantics and should live in the plotting layer, not in each phase script.

### 4. Local script constants should be minimized

Phase scripts should mainly do four things:

1. load config,
2. run computation,
3. write metrics and metadata,
4. call centralized plot functions.

---

## Constant Taxonomy

The project should explicitly classify constants into the following sections.

### A. Reproducibility and Assignment Contract

Purpose:

- constants required by the assignment or by the repo's reproducibility contract.

Examples:

- `SEEDS`
- output directories
- phase artifact naming

How to defend in the report:

- "We used five seeds to match the assignment FAQ guidance and aggregated results with mean and IQR."

### B. DP Planning Defaults

Purpose:

- reference choices used for the main VI/PI runs.

Examples:

- `VI_GAMMA`, `PI_GAMMA`
- `VI_DELTA`, `PI_DELTA`
- `VI_PI_CONSEC_SWEEPS`

How to defend in the report:

- "We used one reference planning configuration for the main results and then validated sensitivity to both `gamma` and `delta` in a separate sweep."

### C. Environment Abstraction Constants

Purpose:

- encode domain assumptions about discretization, clamps, state ordering, and environment-specific limits.

Examples:

- `CARTPOLE_BINS`
- `CARTPOLE_CLAMPS`
- explicit CartPole bin edges
- `CARTPOLE_GRID_CONFIGS`

How to defend in the report:

- "These values define the state abstraction, not a tuning result. They were chosen to prioritize pole angle and angular velocity resolution while keeping the state space tractable."

### D. Model-Build Budgets

Purpose:

- control how empirical models are constructed.

Examples:

- `CARTPOLE_MODEL_ROLLOUT_STEPS`
- per-grid rollout steps inside `CARTPOLE_GRID_CONFIGS`
- `CARTPOLE_MODEL_MIN_VISITS`
- `CARTPOLE_MODEL_SEED`

How to defend in the report:

- "These values were chosen as model-quality budgets: large enough to report coverage honestly, but still feasible within assignment compute limits."

### E. Hyperparameter Validation Ranges

Purpose:

- define the search space used to satisfy the assignment requirement to validate at least two hyperparameters per model.

Examples:

- `VI_PI_HP_GAMMA_VALUES`
- `VI_PI_HP_DELTA_VALUES`
- model-free schedule defaults and search anchors

How to defend in the report:

- "These were validation ranges, not hand-picked winning values. The final chosen setting was selected after evaluating sensitivity within these predefined ranges."

### F. Evaluation Budgets

Purpose:

- define how final reported metrics and lighter validation sweeps are evaluated.

Examples to add or centralize:

- `BJ_EVAL_EPISODES_MAIN`
- `BJ_EVAL_EPISODES_HP`
- `CP_EVAL_EPISODES_MAIN`
- `CP_EVAL_EPISODES_HP`

How to defend in the report:

- "Main tables use the full evaluation budget; hyperparameter sweeps use a lighter budget for screening and are interpreted directionally rather than as exact replicas of the final numbers."

### G. Plotting Semantics and Style Defaults

Purpose:

- centralize visual semantics without hiding them in phase scripts.

Examples:

- algorithm colors
- action colors
- default DPI
- figure width presets
- legend placement defaults
- standard panel titles

How to defend in the report:

- these do not need scientific defense; they need consistency and readability.

---

## Developer Comment Standard

Every nontrivial constant in `src/config.py` should have either an inline comment or a short block comment that answers these questions:

1. What role does this constant play?
2. Was it chosen because of assignment guidance, environment structure, budget, or style?
3. Should it be interpreted as fixed protocol, search range, or a display default?

### Proposed Style

```python
# Assignment contract: FAQ asks for around 5 independent seeds per compared model.
# This is a reproducibility constant, not a tuned hyperparameter.
SEEDS: list[int] = [42, 43, 44, 45, 46]

# Main Blackjack reporting budget.
# Chosen to reduce evaluation noise for the final table/figure values.
# HP sweeps use a separate lighter budget for fast screening.
BJ_EVAL_EPISODES_MAIN: int = 1000

# CartPole default grid used for the main DP comparison.
# Chosen from the assignment-guided non-uniform binning strategy; coarse/fine are used only in the ablation study.
CARTPOLE_BINS: tuple[int, ...] = (3, 3, 8, 12)
```

### What Not To Do

Do not write comments that overclaim provenance, such as:

- "optimal value"
- "best setting"
- "chosen empirically" if no real sweep supports that statement

Be explicit and honest:

- assignment-guided
- chosen as default reference
- chosen as compute budget
- chosen as display convention

---

## Target File Layout After Refactor

### `src/config.py`

Keep this as the primary home for experiment constants, but reorganize it into explicit report-friendly sections:

1. Reproducibility and paths
2. DP planning defaults
3. CartPole discretization and model construction
4. Hyperparameter validation ranges
5. Evaluation budgets
6. Model-free defaults

This file should answer the question: "What settings define the experiment?"

### `src/utils/plotting.py`

Turn this from a stub into the home for:

- plot style defaults,
- semantic mappings,
- figure-size presets,
- shared helpers for reading artifacts,
- plot constructors for phase figures.

This file should answer the question: "How are saved artifacts rendered into stable report figures?"

### Phase Scripts

Phase scripts should keep only:

- phase identifier,
- orchestration logic,
- data writing,
- one call into the plotting layer.

This means the phase scripts become easier to read and easier to defend: they describe the workflow, not the chart styling.

---

## Dynamic Plotting Philosophy

The preferred strategy is: plots should be dynamic from saved data and metadata, not from hand-maintained local plotting constants.

### Dynamic From Data

The following should be inferred from CSV/JSON artifacts:

- number of iterations actually plotted,
- algorithms present,
- grids present,
- legend contents,
- final convergence iteration values shown in titles or annotations,
- y-error bars,
- histogram bin count when a count-derived heuristic is appropriate,
- value-function color scale bounds,
- policy agreement / coverage values shown as labels.

### Fixed Semantic Defaults

The following should remain centralized but fixed:

- algorithm color map,
- action color map,
- known environment axis order,
- readable figure sizes and DPI,
- CartPole terminal-angle markers,
- known grid display order,
- known Blackjack hand label order.

This gives the safety of data-driven plotting without sacrificing stable semantics.

---

## Concrete Refactor Plan

### Step 1. Reorganize `src/config.py`

Add explicit section headers and developer comments for all report-relevant constants.

Changes:

- keep existing constants,
- rename evaluation budgets from script-local names into config-level names,
- keep path constants unchanged,
- add short rationale comments to each constant group.

Deliverable:

- one place to inspect all experiment-defining settings,
- direct material for the reproducibility sheet and report methods section.

### Step 2. Move evaluation budgets out of phase scripts

Current script-local constants such as `N_EVAL_EPISODES` and `HP_EVAL_EPISODES` should be moved into `src/config.py`.

Suggested names:

- `BJ_EVAL_EPISODES_MAIN`
- `BJ_EVAL_EPISODES_HP`
- `CP_EVAL_EPISODES_MAIN`
- `CP_EVAL_EPISODES_HP`

Why this adds value:

- final and sweep budgets become explicit protocol choices,
- differences between main and sweep values become easier to explain in the report,
- fewer opportunities for silent divergence across phases.

### Step 3. Expand phase metadata so plots can be rendered from artifacts

Each phase metadata JSON should include the plot-relevant run context.

Examples:

- selected gamma and delta,
- evaluation budgets used,
- grid names and state counts,
- stop reason / convergence iteration,
- semantic labels needed for the figure caption.

This allows plots to read titles, thresholds, and annotations from metadata rather than duplicating those choices in script code.

### Step 4. Implement a real plotting layer in `src/utils/plotting.py`

The plotting module should provide:

- shared style defaults,
- shared semantic mappings,
- small helpers for reading metrics and metadata,
- environment-specific plotting functions.

Recommended structure:

```python
# style registry
DEFAULT_DPI = 150
ALGO_COLORS = {...}
ACTION_COLORS = {...}

def load_phase_artifacts(metrics_dir: Path, metadata_path: Path) -> tuple[...]:
    ...

def plot_blackjack_convergence(...):
    ...

def plot_blackjack_policy_map(...):
    ...

def plot_blackjack_value_surface(...):
    ...

def plot_cartpole_convergence(...):
    ...

def plot_cartpole_discretization(...):
    ...
```

### Step 5. Refactor Phase 2 plotting to artifact-driven rendering

Current state:

- Phase 2 still defines plot layout and styling inside the script.

Target state:

- Phase 2 writes all CSVs and metadata first,
- then calls centralized plotting functions that read those artifacts.

High-value changes only:

- move convergence plotting to `src/utils/plotting.py`,
- move Blackjack policy-map plotting to `src/utils/plotting.py`,
- move Blackjack value-surface plotting to `src/utils/plotting.py`,
- keep Blackjack grid decoding in either a small plotting helper or a domain-specific helper if reused.

Why Phase 2 first:

- it is smaller,
- it directly supports the report,
- it gives a template for later phases.

### Step 6. Refactor Phase 3 plotting to artifact-driven rendering

Apply the same pattern to:

- convergence curves,
- discretization-study plot,
- policy-slice plot.

Important rule:

- plot panels and labels should be discovered from saved data where possible,
- but grid order should still come from a centralized semantic order.

### Step 7. Optional cleanup for Phase 1

Phase 1 figures are less critical for the final report, so Phase 1 should only be cleaned up after Phases 2 and 3 are done.

Keep this limited to:

- moving color defaults and DPI into the plotting layer,
- leaving phase-local logic if the function is highly specific and not reused.

---

## What Should Become Dynamic vs What Should Stay Fixed

| Item | Dynamic from artifacts? | Central fixed default? | Reason |
|---|---|---|---|
| Iteration extents | Yes | No | Prevent stale hardcoded x-limits |
| Legend entries | Yes | No | Determined by available data |
| Error bars | Yes | No | Derived from saved metrics |
| Value color scale bounds | Yes | No | Should reflect actual saved values |
| Grid names present in a figure | Yes | Partly | Read from data, ordered by semantic mapping |
| Algorithm colors | No | Yes | Cross-figure consistency |
| Action colors | No | Yes | Cross-figure consistency |
| Figure DPI | No | Yes | Stable report quality |
| Font sizes | No | Yes | Legibility and layout consistency |
| Blackjack axis ordering | No | Yes | Semantic order is part of the environment |
| CartPole terminal threshold markers | No | Yes | Environment-defined constant |

---

## File-by-File Implementation Sequence

### Pass 1 — Constant and metadata cleanup

- `src/config.py`
  - reorganize sections
  - add evaluation budgets
  - add developer comments
- `scripts/run_phase_2_vi_pi_blackjack.py`
  - import evaluation budgets from config
  - stop declaring them locally
- `scripts/run_phase_3_vi_pi_cartpole.py`
  - import evaluation budgets from config
  - stop declaring them locally
- `artifacts/metadata/phase2.json` schema writer
  - include plot-relevant run settings
- `artifacts/metadata/phase3.json` schema writer
  - include plot-relevant run settings

### Pass 2 — Plotting layer extraction

- `src/utils/plotting.py`
  - add style defaults
  - add artifact readers
  - add phase-specific plotting functions
- `scripts/run_phase_2_vi_pi_blackjack.py`
  - replace local plot helpers with calls into plotting utils
- `scripts/run_phase_3_vi_pi_cartpole.py`
  - replace local plot helpers with calls into plotting utils

### Pass 3 — Validation and report support

- `tests/`
  - add smoke tests that plot functions can render from saved fixture-like CSV/JSON inputs
- documentation
  - update review / reproducibility guidance if naming changes

---

## Acceptance Criteria

The refactor is successful when all of the following are true:

1. No phase script contains local evaluation budget constants.
2. Every report-relevant constant is either in `src/config.py` or explicitly justified as a local plotting semantic in `src/utils/plotting.py`.
3. Phase 2 and Phase 3 figures can be regenerated from saved metrics plus metadata.
4. Plot titles and annotations no longer rely on duplicated hardcoded run settings in the phase scripts.
5. The report can explain every important constant using one of four categories:
   - assignment-driven
   - environment-driven
   - budget-driven
   - style-driven
6. No algorithm outputs change except where plot generation or metadata labeling is intentionally improved.

---

## Guardrails to Avoid Buggy Refactoring

### Keep the refactor shallow

Do not combine this cleanup with algorithm changes.

### Preserve artifact schemas where possible

If a CSV schema must change, update plotting and any dependent review/report code in the same change set.

### Treat metadata as an explicit contract

If a plot needs a run parameter, add it to metadata rather than re-encoding it in the plot function.

### Prefer data-driven extents over hand-maintained ones

This matches the goal of making iterative development safer.

### Do not over-generalize plotting too early

The plotting layer should centralize style and artifact rendering, not try to create one giant universal plot factory.

---

## Recommended First Implementation Slice

If this refactor is executed incrementally, the highest-value first slice is:

1. reorganize `src/config.py`,
2. move all evaluation budgets into config,
3. add plot-relevant fields to `phase2.json` and `phase3.json`,
4. implement Phase 2 plotting in `src/utils/plotting.py`,
5. leave Phase 3 plotting for the next pass.

This yields immediate report value with limited code churn.

---

## Suggested Report Language

After the refactor, the report can describe constants in a simple, defensible way:

> We organized constants into four categories: assignment-driven reproducibility settings, environment-driven modeling assumptions, budget-driven evaluation choices, and style-driven visualization defaults. Main experiment settings were centralized in `src/config.py`, while plotting semantics were centralized in the shared plotting layer. This separation made it possible to regenerate figures from saved metrics and metadata while keeping the scientific assumptions explicit and stable.

That is the key benefit of the refactor: not fewer constants, but clearer provenance for each one.