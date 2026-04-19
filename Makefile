PY=uv run python

.DEFAULT_GOAL := help

.PHONY: help setup dev upgrade run lint format test clean \
        phase1 phase2 phase3 phase4 phase5 phase6 phase7 phase8 \
        viz \
        gate1 gate2 gates pipeline overnight

help: ## Show available targets
	@grep -E '^[a-zA-Z0-9_\-]+:.*?##' Makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup: ## Create venv and sync prod dependencies
	uv venv --clear
	uv sync

add-deps: ## Add dependencies from requirements.txt via uv when present
	@if [ -f requirements.txt ]; then uv add -r requirements.txt; else echo "requirements.txt not found"; fi	

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

# ── Experiment phases ──────────────────────────────────────────────────────────

phase1: ## Env setup + CartPole model estimation
	bash ml_run.sh "uv run python scripts/run_phase_1_env_setup.py"

phase2: ## VI/PI on Blackjack
	@test -f scripts/run_phase_2_vi_pi_blackjack.py || \
		{ echo "ERROR: scripts/run_phase_2_vi_pi_blackjack.py not implemented yet"; exit 1; }
	bash ml_run.sh "uv run python scripts/run_phase_2_vi_pi_blackjack.py"

phase3: ## VI/PI on CartPole + discretization study
	@test -f scripts/run_phase_3_vi_pi_cartpole.py || \
		{ echo "ERROR: scripts/run_phase_3_vi_pi_cartpole.py not implemented yet"; exit 1; }
	bash ml_run.sh "uv run python scripts/run_phase_3_vi_pi_cartpole.py"

phase4: ## SARSA/Q-Learning on Blackjack (HP search + final eval)
	@test -f scripts/run_phase_4_model_free_blackjack.py || \
		{ echo "ERROR: scripts/run_phase_4_model_free_blackjack.py not implemented yet"; exit 1; }
	bash ml_run.sh "uv run python scripts/run_phase_4_model_free_blackjack.py"

phase5: ## SARSA/Q-Learning on CartPole
	@test -f scripts/run_phase_5_model_free_cartpole.py || \
		{ echo "ERROR: scripts/run_phase_5_model_free_cartpole.py not implemented yet"; exit 1; }
	bash ml_run.sh "uv run python scripts/run_phase_5_model_free_cartpole.py"

phase6: ## Cross-method comparison
	@test -f scripts/run_phase_6_comparison.py || \
		{ echo "ERROR: scripts/run_phase_6_comparison.py not implemented yet"; exit 1; }
	bash ml_run.sh "uv run python scripts/run_phase_6_comparison.py"

phase7: ## DQN extra credit (optional)
	@test -f scripts/run_phase_7_dqn_ec.py || \
		{ echo "ERROR: scripts/run_phase_7_dqn_ec.py not implemented yet"; exit 1; }
	bash ml_run.sh "uv run python scripts/run_phase_7_dqn_ec.py"

phase8: ## Report tables + repro artifacts
	@test -f scripts/run_phase_8_report_tables.py || \
		{ echo "ERROR: scripts/run_phase_8_report_tables.py not implemented yet"; exit 1; }
	bash ml_run.sh "uv run python scripts/run_phase_8_report_tables.py"

# ── Visualization-only (no recomputation) ────────────────────────────────────

viz: ## Re-render all phase figures (wipes figures dir first)
	rm -rf artifacts/figures
	$(PY) scripts/visualize_all.py

# ── Gates (pytest) ────────────────────────────────────────────────────────────

gate1: ## Gate 1 — env tests (run after Phase 1)
	uv run pytest tests/test_envs.py -v

gate2: ## Gate 2 — algorithm tests (run after Phase 2)
	@test -f tests/test_algorithms.py || \
		{ echo "ERROR: tests/test_algorithms.py not implemented yet"; exit 1; }
	uv run pytest tests/test_algorithms.py -v

gates: gate1 gate2 ## Run all gates

# ── Pipeline ──────────────────────────────────────────────────────────────────

pipeline: phase1 gate1 phase2 phase3 phase4 phase5 phase6 phase8 ## Full pipeline

overnight: ## Run full pipeline in a detached background session
	bash ml_run.sh --detach "make pipeline" overnight
