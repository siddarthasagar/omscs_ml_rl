# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

OMSCS ML — Reinforcement Learning coursework. Modeled after the sibling `omscs_ml_ul` project.

## Commands

```bash
make dev        # create .venv and install all deps (including dev)
make setup      # create .venv and install prod deps only
make upgrade    # upgrade lock file and sync

make lint       # ruff check
make format     # ruff check --fix + ruff format
make test       # pytest
make clean      # remove .venv, caches, __pycache__
make run        # python main.py
```

Run a long job with sleep prevention (inline or detached):
```bash
bash ml_run.sh "make <target>"
bash ml_run.sh --detach "make <target>" [session_name]
```

## Structure

```
src/            # importable library code
scripts/        # one-off runner scripts
tests/          # pytest suite
artifacts/      # generated outputs (gitignored): figures, logs, metrics, tables, metadata
data/           # raw datasets (gitignored contents)
documents/      # specs, ADRs, course materials
tmp/            # scratch space (gitignored)
```

## Tooling

- **uv** for dependency management (`uv sync`, `uv add`)
- **ruff** for lint + format (configured in `pyproject.toml`, ignores F403/F405)
- **pytest** with debug-level CLI logging
- **ml_run.sh** wraps any command with `caffeinate` (macOS) or `systemd-inhibit` (Linux) to prevent sleep during long runs
