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

Run a long job with sleep prevention:
```bash
bash ml_run.sh "make <target>"                      # inline with caffeinate/systemd-inhibit
bash ml_run.sh --detach "make <target>" [session]   # background tmux/screen
```

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
