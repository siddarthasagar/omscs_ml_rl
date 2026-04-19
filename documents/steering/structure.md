---
inclusion: always
---

# Structure — CS7641 RL

## Repository Layout

```
omscs_ml_rl/
├── data/                              # Raw datasets (git-tracked)
├── documents/
│   ├── steering/                      # Always-loaded project context
│   └── canvas/                        # Assignment docs (read-only reference)
├── src/
│   ├── config.py                      # Centralized constants (seeds, paths, hyperparams)
│   ├── envs/                          # MDP environment wrappers
│   ├── algorithms/                    # RL algorithm implementations
│   └── utils/
│       ├── logger.py                  # configure_logger(run_id) → Logger
│       └── plotting.py                # All plot_* functions
├── scripts/
│   ├── run_phase_{N}_{slug}.py        # One script per phase
│   ├── visualize_all.py               # Standalone figure renderer; supports all-phase or filtered rerenders
│   └── run_phase_8_report_tables.py     # Generates LaTeX tables, report_numbers.tex, and repro artifacts
├── tests/
└── artifacts/                         # git-ignored runtime outputs
    ├── logs/                          # phase{N}.log per run
    ├── metrics/phase{N}_{slug}/       # CSVs per phase
    ├── figures/phase{N}_{slug}/       # PNGs per phase
    ├── metadata/                      # phase{N}.json — human-checkpointable JSON per phase
    ├── tables/                        # .tex files imported by the report
    └── repro/                         # runbook.md + submission_checklist.md (auto-generated, Phase 8); overleaf_link.md + ai_use_statement.md (manually authored)
```

## Core Design Principles

### 1. Phase-gate execution with declared artifact contracts

Each phase script owns a declared artifact contract:
- optional upstream artifact inputs (zero or more files)
- raw metrics written to `artifacts/metrics/phase{N}_{slug}/`
- a **`artifacts/metadata/phase{N}.json`** checkpoint summarising key results and runtime context
- optional plot-support artifacts when CSV/JSON alone are insufficient for visualization
- rendered figures written to `artifacts/figures/phase{N}_{slug}/`

A phase may have **no upstream phase dependency** if it is self-contained (for example, a phase built on the exact analytic Blackjack model). When a phase does depend on earlier work, it consumes only declared artifact files — never another phase's in-memory Python state.

This enables human evaluation at each gate: review the checkpoint JSON, metrics, and figures before running the next dependent phase. Nothing downstream should rely on implicit runtime coupling.

### 2. Two-step lifecycle with standalone rendering

Preferred go-forward contract for every phase module:
- `run() -> Path`: execute computation, write all required artifacts, and return the checkpoint path
- `visualize(checkpoint_path: Path) -> list[Path]`: reload artifacts from disk only and render the phase figures

Preferred orchestration:

- phase entrypoints compute artifacts first and do not need to inline matplotlib rendering,
- `scripts/visualize_all.py` is the standalone renderer that reloads checkpoints and dispatches phase `visualize()` functions in a fresh process,
- an optional phase filter such as `--phase phase3` should rerender one phase only,
- no filter should rerender all phases whose checkpoints already exist,
- `make viz` remains the repo-level full rerender target and may wipe `artifacts/figures/` first.

This keeps experiment execution and visualization coupled through explicit artifact contracts, while isolating logs and avoiding long experiment runs being cluttered by rendering output. A phase script may still call `visualize()` inline for local debugging, but the preferred default is compute-only phase entrypoints plus standalone rendering.

### 3. Artifact categories and boundaries

- `artifacts/metrics/phase{N}_{slug}/`: tabular outputs and optional plot-support artifacts such as `plot_*.npz`
- `artifacts/metadata/`: checkpoint JSONs and shared model artifacts consumed by downstream phases
- `artifacts/figures/phase{N}_{slug}/`: rendered PNGs only

If CSV/JSON are not sufficient to reconstruct a figure, the phase should persist a small plot-support artifact and reload it inside `visualize()`. Plot generation should never depend on transient runtime objects from `run()`.

### 4. LaTeX tables and repro artifacts generated from code — never typed by hand

The final phase script reads all `phase{N}.json` files and writes:
- `artifacts/tables/tab_phase{N}_*.tex` — tabular bodies (no `\begin{table}` wrapper)
- `artifacts/tables/report_numbers.tex` — `\newcommand` macros for every number cited inline
- `artifacts/repro/runbook.md` — **auto-generated**: exact Linux reproduction commands (`make dev`, `make pipeline`), seed list, expected outputs, figure paths; explicitly states no external data files (both environments from `gymnasium`, all artifacts generated under `artifacts/`)
- `artifacts/repro/submission_checklist.md` — **auto-generated**: deliverable tracker

Two repro files are **manually authored** (not generated by code):
- `artifacts/repro/overleaf_link.md` — READ-ONLY Overleaf URL, recorded manually after project is created
- `artifacts/repro/ai_use_statement.md` — AI-use disclosure, written manually before submission

`report_numbers.tex` includes `\SeedList`, per-method/per-MDP wall-clock macros, and `\WallClockTotal` to satisfy the FAQ seed-list and total wall-clock reporting requirements.

The report does `\input{tables/report_numbers}` in its preamble and `\input{tables/tab_*}` at each table site. Numbers in prose and in tables are always in sync with the actual experiment output.

### 5. Structured logging — no bare print()

Every phase script calls `configure_logger(run_id)` from `src/utils/logger.py`.
- `run_id` pattern: `phase{N}`
- Writes to `artifacts/logs/{run_id}.log` **and** stdout simultaneously
- No bare `print()` in phase scripts (tqdm progress bars are exempt)

Logging isolation is a universal project rule:
- Canonical phase logs such as `artifacts/logs/phase5.log` are reserved for real workspace experiment runs only.
- Tests, smoke runs, temporary-budget validations, and debug runs must not overwrite or interleave with canonical experiment logs.
- Non-canonical runs should use an isolated artifact root, temporary workspace, or distinct `run_id` so their logs remain separate from report-facing experiment evidence.
- The same separation rule applies to metrics, figures, and checkpoint JSONs: test/debug artifacts must not be mistaken for final phase outputs.

## Naming Conventions

- Phase scripts: `run_phase_{N}_{slug}.py`
- Artifact dirs: `artifacts/{metrics,figures}/phase{N}_{slug}/`
- Log files: `artifacts/logs/phase{N}.log`
- Metadata: `artifacts/metadata/phase{N}.json`
- Optional plot-support artifacts: `artifacts/metrics/phase{N}_{slug}/plot_*.npz`
