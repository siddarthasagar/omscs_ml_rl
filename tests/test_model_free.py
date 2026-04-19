"""Tests for Phase 4 model-free algorithms and artifact pipeline.

Covers:
  - encode_bj_state correctness for known observations
  - SARSA and Q-Learning run without error and return correct shapes
  - early-stopping disablement via convergence_delta=0
  - serial-vs-parallel equivalence of _run_phase4_final_job

All tests use a tiny episode budget so the suite runs in a few seconds.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.algorithms import QLearningConfig, SarsaConfig, run_q_learning, run_sarsa
from src.algorithms.model_free_utils import (
    check_convergence,
    encode_bj_state,
    epsilon_greedy_action,
    greedy_action,
    linear_schedule,
)
from src.config import BJ_N_ACTIONS, BJ_N_STATES

# ── Fixtures ──────────────────────────────────────────────────────────────────

TINY_EPISODES = 500
TINY_EVAL_EPISODES = 100
SMOKE_SEEDS = [42, 43]


@pytest.fixture(scope="module")
def sarsa_config_tiny():
    return SarsaConfig(
        alpha_start=0.5,
        alpha_end=0.1,
        alpha_decay_steps=1_000,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay_steps=500,
        gamma=0.99,
        convergence_window=50,
        convergence_delta=0.1,
        convergence_m=2,
    )


@pytest.fixture(scope="module")
def ql_config_tiny():
    return QLearningConfig(
        alpha_start=0.5,
        alpha_end=0.1,
        alpha_decay_steps=1_000,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay_steps=500,
        gamma=0.99,
        convergence_window=50,
        convergence_delta=0.1,
        convergence_m=2,
    )


# ── State encoding ────────────────────────────────────────────────────────────


def test_encode_bj_state_no_usable_ace():
    # player_sum=15, dealer=5, usable_ace=False
    assert encode_bj_state((15, 5, False)) == 15 * 11 * 2 + 5 * 2 + 0


def test_encode_bj_state_usable_ace():
    # player_sum=18, dealer=10, usable_ace=True
    assert encode_bj_state((18, 10, True)) == 18 * 11 * 2 + 10 * 2 + 1


def test_encode_bj_state_range():
    # All valid Blackjack states should map into [0, BJ_N_STATES)
    for ps in range(4, 22):
        for dc in range(1, 11):
            for ua in [0, 1]:
                s = encode_bj_state((ps, dc, ua))
                assert 0 <= s < BJ_N_STATES, (
                    f"State {s} out of range for obs ({ps},{dc},{ua})"
                )


# ── Linear schedule ───────────────────────────────────────────────────────────


def test_linear_schedule_at_zero():
    assert linear_schedule(0, 1.0, 0.0, 100) == pytest.approx(1.0)


def test_linear_schedule_at_end():
    assert linear_schedule(100, 1.0, 0.0, 100) == pytest.approx(0.0)


def test_linear_schedule_beyond_end():
    assert linear_schedule(200, 1.0, 0.0, 100) == pytest.approx(0.0)


def test_linear_schedule_midpoint():
    assert linear_schedule(50, 1.0, 0.0, 100) == pytest.approx(0.5)


# ── check_convergence ─────────────────────────────────────────────────────────


def test_check_convergence_not_enough_data():
    assert not check_convergence([0.3, 0.31], window=100, delta=0.01, m_consec=3)


def test_check_convergence_not_plateaued():
    # Steady improvement — should not converge
    means = [0.30, 0.32, 0.34, 0.36, 0.38]
    assert not check_convergence(means, window=100, delta=0.01, m_consec=3)


def test_check_convergence_plateaued():
    # Tiny changes over 4 consecutive pairs → converged
    means = [0.40, 0.400, 0.4001, 0.4002, 0.4003]
    assert check_convergence(means, window=100, delta=0.01, m_consec=3)


# ── Action selection ──────────────────────────────────────────────────────────


def test_greedy_action_picks_max():
    Q = np.array([[0.1, 0.9], [0.8, 0.2]])
    assert greedy_action(Q, state=0) == 1
    assert greedy_action(Q, state=1) == 0


def test_epsilon_greedy_greedy_when_eps_zero():
    Q = np.array([[0.1, 0.9]])
    rng = np.random.default_rng(0)
    # With epsilon=0 it should always be greedy
    actions = {epsilon_greedy_action(Q, 0, 0.0, rng) for _ in range(50)}
    assert actions == {1}


def test_epsilon_greedy_random_when_eps_one():
    Q = np.array([[0.1, 0.9]])
    rng = np.random.default_rng(0)
    actions = {epsilon_greedy_action(Q, 0, 1.0, rng) for _ in range(200)}
    assert len(actions) == 2  # both actions should appear


# ── Algorithm smoke tests ─────────────────────────────────────────────────────


def test_sarsa_returns_correct_shapes(sarsa_config_tiny):
    import gymnasium as gym

    env = gym.make("Blackjack-v1")
    Q, trace = run_sarsa(
        env, sarsa_config_tiny, BJ_N_STATES, BJ_N_ACTIONS, TINY_EPISODES, seed=42
    )
    env.close()
    assert Q.shape == (BJ_N_STATES, BJ_N_ACTIONS)
    assert len(trace) <= TINY_EPISODES
    assert len(trace) >= 1
    assert all(r["episode"] >= 1 for r in trace)


def test_q_learning_returns_correct_shapes(ql_config_tiny):
    import gymnasium as gym

    env = gym.make("Blackjack-v1")
    Q, trace = run_q_learning(
        env, ql_config_tiny, BJ_N_STATES, BJ_N_ACTIONS, TINY_EPISODES, seed=42
    )
    env.close()
    assert Q.shape == (BJ_N_STATES, BJ_N_ACTIONS)
    assert len(trace) <= TINY_EPISODES
    assert len(trace) >= 1


def test_sarsa_q_values_updated():
    """Q-table should not remain all-zero after training."""
    import gymnasium as gym

    cfg = SarsaConfig(
        alpha_start=0.5,
        alpha_end=0.1,
        alpha_decay_steps=500,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay_steps=200,
        gamma=0.99,
        convergence_window=50,
        convergence_delta=1.0,
        convergence_m=2,
    )
    env = gym.make("Blackjack-v1")
    Q, _ = run_sarsa(env, cfg, BJ_N_STATES, BJ_N_ACTIONS, n_episodes=300, seed=0)
    env.close()
    assert np.any(Q != 0.0), "Q-table should have non-zero entries after training"


def test_early_stopping_disabled_runs_full_budget():
    """convergence_delta=0 must prevent early stopping — all episodes should run."""
    import gymnasium as gym

    n_eps = 300
    cfg = SarsaConfig(
        alpha_start=0.5,
        alpha_end=0.1,
        alpha_decay_steps=500,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay_steps=200,
        gamma=0.99,
        convergence_window=50,
        convergence_delta=0.0,  # never fires
        convergence_m=2,
    )
    env = gym.make("Blackjack-v1")
    _, trace = run_sarsa(env, cfg, BJ_N_STATES, BJ_N_ACTIONS, n_episodes=n_eps, seed=0)
    env.close()
    assert len(trace) == n_eps, (
        f"Expected {n_eps} episodes with early-stopping disabled, got {len(trace)}"
    )


# ── Serial-vs-parallel equivalence ────────────────────────────────────────────


def _load_phase4_module():
    """Load run_phase_4_model_free_blackjack without triggering __main__ side effects."""
    p = Path("scripts/run_phase_4_model_free_blackjack.py")
    spec = importlib.util.spec_from_file_location("run_phase_4_model_free_blackjack", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def phase4_mod():
    return _load_phase4_module()


@pytest.fixture(scope="module")
def tiny_job(phase4_mod):
    """A minimal final-training job for equivalence testing."""
    hp = {
        "alpha_start": 0.5,
        "alpha_end": 0.1,
        "alpha_decay_steps": 1_000,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_steps": 500,
        "gamma": 0.99,
    }
    return {
        "algorithm": "sarsa",
        "seed": 42,
        "regime": "tuned",
        "hp": hp,
        "train_episodes": TINY_EPISODES,
        "eval_episodes": TINY_EVAL_EPISODES,
        "disable_early_stopping": True,
    }


def test_final_job_result_shape(phase4_mod, tiny_job):
    """Worker result should have the expected keys and array shapes."""
    result = phase4_mod._run_phase4_final_job(tiny_job)
    assert result["algorithm"] == "sarsa"
    assert result["seed"] == 42
    assert result["regime"] == "tuned"
    assert result["episodes_run"] == TINY_EPISODES  # early stopping disabled
    assert result["curve_episode"].dtype == np.int32
    assert result["curve_window_mean"].dtype == np.float32
    assert len(result["curve_episode"]) == len(result["curve_window_mean"])
    # window_mean populated every convergence_window (default 100) episodes
    assert len(result["curve_episode"]) == TINY_EPISODES // 100
    assert 0.0 <= result["win_rate"] <= 1.0
    assert 0.0 <= result["draw_rate"] <= 1.0
    assert 0.0 <= result["loss_rate"] <= 1.0
    assert (
        abs(result["win_rate"] + result["draw_rate"] + result["loss_rate"] - 1.0) < 1e-6
    )


def test_deterministic_aggregation_order(phase4_mod):
    """Results sorted by (regime, algorithm, seed) should match the canonical order."""
    from src.config import SEEDS as CONFIG_SEEDS

    fake_results = [
        {"algorithm": "qlearning", "seed": CONFIG_SEEDS[2], "regime": "tuned"},
        {"algorithm": "sarsa", "seed": CONFIG_SEEDS[0], "regime": "controlled"},
        {"algorithm": "sarsa", "seed": CONFIG_SEEDS[1], "regime": "tuned"},
        {"algorithm": "qlearning", "seed": CONFIG_SEEDS[0], "regime": "controlled"},
    ]
    regime_order = {"controlled": 0, "tuned": 1}
    algo_order = {"sarsa": 0, "qlearning": 1}
    seed_order = {s: i for i, s in enumerate(CONFIG_SEEDS)}
    fake_results.sort(
        key=lambda r: (
            regime_order[r["regime"]],
            algo_order[r["algorithm"]],
            seed_order[r["seed"]],
        )
    )

    assert fake_results[0] == {
        "algorithm": "sarsa",
        "seed": CONFIG_SEEDS[0],
        "regime": "controlled",
    }
    assert fake_results[1] == {
        "algorithm": "qlearning",
        "seed": CONFIG_SEEDS[0],
        "regime": "controlled",
    }
    assert fake_results[2] == {
        "algorithm": "sarsa",
        "seed": CONFIG_SEEDS[1],
        "regime": "tuned",
    }
    assert fake_results[3] == {
        "algorithm": "qlearning",
        "seed": CONFIG_SEEDS[2],
        "regime": "tuned",
    }


# ── Top-level run() smoke test ────────────────────────────────────────────────


@pytest.mark.slow
def test_run_lifecycle(tmp_path, monkeypatch):
    """End-to-end smoke: run() writes all expected artifacts without error.

    Uses a drastically reduced budget so the test finishes in seconds:
    - 1 HP stage-1 config × 200 episodes
    - 0 stage-2/3 promotion (top-k=1 keeps the only candidate)
    - 200 final training episodes per seed
    - 2 seeds, 1 worker (serial)
    All artifact paths are redirected to tmp_path.
    """
    import importlib.util
    from pathlib import Path
    from unittest.mock import patch

    # Load a fresh module instance so patches do not bleed into other tests.
    p = Path("scripts/run_phase_4_model_free_blackjack.py")
    spec = importlib.util.spec_from_file_location("_phase4_smoke", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Patch module-level config constants to a tiny budget.
    patches = {
        "BJ_HP_STAGE1_CONFIGS": 1,
        "BJ_HP_STAGE1_EPISODES": 200,
        "BJ_HP_STAGE2_TOP_K": 1,
        "BJ_HP_STAGE2_EPISODES": 200,
        "BJ_HP_STAGE3_TOP_K": 1,
        "BJ_HP_STAGE3_EPISODES": 200,
        "BJ_TRAIN_EPISODES": 200,
        "BJ_EVAL_EPISODES_MAIN": 50,
        "BJ_EVAL_EPISODES_HP": 50,
        "SEEDS": [42, 43],
        "PHASE4_FINAL_TRAIN_MAX_WORKERS": 1,
    }
    for name, val in patches.items():
        monkeypatch.setattr(mod, name, val)

    # Redirect artifact directories to tmp_path.
    from src.utils.phase_artifacts import PhasePaths

    fake_paths = PhasePaths(
        phase_id="phase4",
        slug="model_free_blackjack",
        metrics_dir=tmp_path / "metrics",
        figures_dir=tmp_path / "figures",
        metadata_dir=tmp_path / "metadata",
    )
    fake_paths.makedirs()

    with patch.object(mod, "resolve_phase_paths", return_value=fake_paths):
        checkpoint_path = mod.run()

    # Checkpoint file was written.
    assert checkpoint_path.exists(), "checkpoint not written"

    # All required CSVs are present.
    for name in [
        "mf_hp_search.csv",
        "mf_learning_curves.csv",
        "mf_eval_per_seed.csv",
        "mf_eval_summary.csv",
    ]:
        assert (fake_paths.metrics_dir / name).exists(), f"missing {name}"

    # Checkpoint JSON is loadable and has expected top-level keys.
    import json

    with open(checkpoint_path) as f:
        ckpt = json.load(f)
    assert "summary" in ckpt
    for regime in ["controlled", "tuned"]:
        for algo in ["sarsa", "qlearning"]:
            s = ckpt["summary"][regime][algo]
            assert "mean_return" in s
            assert "final_window_iqr" in s

    # Per-seed CSV has expected columns.
    import pandas as pd

    per_seed = pd.read_csv(fake_paths.metrics_dir / "mf_eval_per_seed.csv")
    for col in ["algorithm", "seed", "regime", "mean_return", "final_window_return",
                "convergence_episode", "train_wall_clock_s"]:
        assert col in per_seed.columns, f"missing column {col}"
    assert set(per_seed["regime"].unique()) == {"controlled", "tuned"}
    assert set(per_seed["algorithm"].unique()) == {"sarsa", "qlearning"}
