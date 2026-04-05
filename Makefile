PY=uv run python

.DEFAULT_GOAL := help

.PHONY: help setup dev upgrade run lint format test clean

help: ## Show available targets
	@grep -E '^[a-zA-Z0-9_\-]+:.*?##' Makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup: ## Create venv and sync prod dependencies
	uv venv --clear
	uv sync

dev: ## Create venv and sync dev dependencies
	uv venv --clear
	uv sync --dev --all-extras

upgrade: ## Upgrade locked dependencies and sync
	uv lock --upgrade
	uv sync --dev --all-extras

run: ## Run the project entrypoint
	$(PY) main.py

lint: ## Run Ruff checks
	uv run ruff check .

format: ## Run Ruff fix + format
	uv run ruff check --fix .
	uv run ruff format .

test: ## Run the test suite
	uv run pytest

clean: ## Remove build artifacts and caches
	rm -rf .pytest_cache .ruff_cache .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +
