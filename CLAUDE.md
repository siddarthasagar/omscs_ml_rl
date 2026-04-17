# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

OMSCS ML — Reinforcement Learning coursework (CS7641). Modeled after the sibling `omscs_ml_ul` project.

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

Run experiment phases (always via ml_run.sh — never direct uv/python):
```bash
make phase1                                         # inline, sleep-prevention active
bash ml_run.sh --detach "make phase1" [session]     # background tmux/screen session
```

All `make phase{N}` targets wrap the script in `bash ml_run.sh` automatically.
Use `--detach` for anything expected to run longer than a few minutes.

## Design Principles

See `documents/steering/` — `structure.md`, `tech.md`, and `product.md` are the authoritative source for all design decisions, naming conventions, and patterns.

## Structure

```
src/            # importable library (config, algorithms, envs, utils)
scripts/        # one script per phase + final report-table generator
tests/          # pytest gates (run before advancing phases)
artifacts/      # gitignored: logs/ metrics/ figures/ metadata/ tables/
data/           # raw datasets
documents/
  steering/     # always-loaded context: product.md, structure.md, tech.md
  canvas/       # assignment brief (read-only)
```
