---
inclusion: always
---

# Tech — CS7641 RL

## Stack

- **Python:** 3.13
- **Package manager:** uv
- **Data:** pandas, numpy
- **Viz:** matplotlib
- **Testing:** pytest
- **Linting:** ruff

> ML/RL-specific dependencies will be added here once the assignment scope is confirmed.

## Seeds

| Constant | Value | Use |
|----------|-------|-----|
| `SEED` | 42 | All runs (expand to `SEEDS_REPORT` list if multi-seed averaging is needed) |

## Logging Standard

Every phase script calls `configure_logger(run_id)` from `src/utils/logger.py`.
`run_id` pattern: `phase{N}_{YYYYMMDDTHHMMSS}` (e.g. `phase2_20260405T143000`).
No bare `print()` in phase scripts. tqdm progress bars are exempt.

## Build Commands

```
make dev                                    # venv + all deps (incl. dev)
make test                                   # full pytest
make lint / make format                     # ruff
uv run python scripts/run_phase_N_*.py     # phase entrypoints
bash ml_run.sh "make <target>"             # with sleep prevention (inline)
bash ml_run.sh --detach "make <target>"    # background tmux/screen session
```
