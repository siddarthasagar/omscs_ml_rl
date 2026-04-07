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
│   └── run_phase_last_report_tables.py  # Generates LaTeX tables, report_numbers.tex, and repro artifacts
├── tests/
└── artifacts/                         # git-ignored runtime outputs
    ├── logs/                          # phase{N}.log per run
    ├── metrics/phase{N}_{slug}/       # CSVs per phase
    ├── figures/phase{N}_{slug}/       # PNGs per phase
    ├── metadata/                      # phase{N}.json — human-checkpointable JSON per phase
    ├── tables/                        # .tex files imported by the report
    └── repro/                         # runbook.md, submission_checklist.md, overleaf_link.md, ai_use_statement.md (Phase 8)
```

## Core Design Principles

### 1. Phase-gate execution with JSON checkpoints

Each phase script:
- Runs its computation and saves raw metrics as CSVs to `artifacts/metrics/phase{N}_{slug}/`
- Saves a **`artifacts/metadata/phase{N}.json`** summarising the key results for that phase
- The next phase reads from the prior phase's JSON — never re-derives from CSVs itself

This enables human evaluation at each gate: review the JSON (and figures) before running the next phase. Nothing downstream runs until you're satisfied with the checkpoint.

### 2. LaTeX tables and repro artifacts generated from code — never typed by hand

The final phase script reads all `phase{N}.json` files and writes:
- `artifacts/tables/tab_phase{N}_*.tex` — tabular bodies (no `\begin{table}` wrapper)
- `artifacts/tables/report_numbers.tex` — `\newcommand` macros for every number cited inline
- `artifacts/repro/runbook.md` — exact Linux reproduction instructions
- `artifacts/repro/submission_checklist.md` — deliverable tracker

The report does `\input{tables/report_numbers}` in its preamble and `\input{tables/tab_*}` at each table site. Numbers in prose and in tables are always in sync with the actual experiment output.

### 3. Structured logging — no bare print()

Every phase script calls `configure_logger(run_id)` from `src/utils/logger.py`.
- `run_id` pattern: `phase{N}`
- Writes to `artifacts/logs/{run_id}.log` **and** stdout simultaneously
- No bare `print()` in phase scripts (tqdm progress bars are exempt)

## Naming Conventions

- Phase scripts: `run_phase_{N}_{slug}.py`
- Artifact dirs: `artifacts/{metrics,figures}/phase{N}_{slug}/`
- Log files: `artifacts/logs/phase{N}.log`
- Metadata: `artifacts/metadata/phase{N}.json`
