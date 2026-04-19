"""Tests for Phase 5 model-free CartPole pipeline.

Covers:
  - CartPoleDiscreteWrapper: obs → integer state, observation_space shape
  - _eval_cp_policy: returns finite mean_episode_len in valid range
  - _run_disc_job: correct result keys and convergence_episode semantics
  - _run_phase5_final_job: correct result keys, array shapes, episode count
  - run() lifecycle smoke test (slow)

All non-slow tests use tiny episode budgets and run in a few seconds.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.algorithms import SarsaConfig, run_sarsa
from src.config import CP_N_ACTIONS
from src.envs.cartpole_discretizer import CartPoleDiscretizer

# ── Fixtures ──────────────────────────────────────────────────────────────────

TINY_EPISODES = 200
TINY_EVAL_EPISODES = 50
SMOKE_SEEDS = [42, 43]


def _load_phase5_module():
    p = Path("scripts/run_phase_5_model_free_cartpole.py")
    spec = importlib.util.spec_from_file_location("run_phase_5_model_free_cartpole", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def phase5_mod():
    return _load_phase5_module()


@pytest.fixture(scope="module")
def default_discretizer():
    return CartPoleDiscretizer()


# ── CartPoleDiscreteWrapper ───────────────────────────────────────────────────


def test_wrapper_obs_is_integer(phase5_mod, default_discretizer):
    """Wrapper reset/step should return integer observations."""
    env = phase5_mod._CartPoleDiscreteWrapper(default_discretizer)
    obs, _ = env.reset(seed=0)
    assert isinstance(obs, (int, np.integer)), f"Expected int, got {type(obs)}"
    assert 0 <= obs < default_discretizer.n_states
    obs2, _, _, _, _ = env.step(0)
    assert isinstance(obs2, (int, np.integer))
    assert 0 <= obs2 < default_discretizer.n_states
    env.close()


def test_wrapper_observation_space(phase5_mod, default_discretizer):
    env = phase5_mod._CartPoleDiscreteWrapper(default_discretizer)
    assert env.observation_space.n == default_discretizer.n_states
    env.close()


def test_wrapper_custom_grid(phase5_mod):
    """Wrapper should use n_states from a custom grid config."""
    from src.config import CARTPOLE_GRID_CONFIGS

    coarse_disc = CartPoleDiscretizer(CARTPOLE_GRID_CONFIGS["coarse"])
    env = phase5_mod._CartPoleDiscreteWrapper(coarse_disc)
    assert env.observation_space.n == coarse_disc.n_states
    obs, _ = env.reset(seed=0)
    assert 0 <= obs < coarse_disc.n_states
    env.close()


# ── _eval_cp_policy ───────────────────────────────────────────────────────────


def test_eval_cp_policy_range(phase5_mod, default_discretizer):
    """Greedy policy evaluation should return finite mean_episode_len in [1, 500]."""
    Q = np.zeros((default_discretizer.n_states, CP_N_ACTIONS))
    stats = phase5_mod._eval_cp_policy(Q, default_discretizer, n_episodes=20, seed=42)
    assert "mean_episode_len" in stats
    assert 1.0 <= stats["mean_episode_len"] <= 500.0


def test_eval_cp_policy_random_vs_zero_q(phase5_mod, default_discretizer):
    """A learned Q-table (non-zero) should not necessarily differ from zero Q,
    but the function must run without error and return a positive value."""
    cfg = SarsaConfig(
        alpha_start=0.5,
        alpha_end=0.1,
        alpha_decay_steps=500,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay_steps=200,
        gamma=0.99,
        convergence_window=50,
        convergence_delta=0.0,
        convergence_m=2,
    )

    env = phase5_mod._CartPoleDiscreteWrapper(default_discretizer)
    Q, _ = run_sarsa(
        env,
        cfg,
        default_discretizer.n_states,
        CP_N_ACTIONS,
        n_episodes=100,
        seed=0,
    )
    env.close()

    stats = phase5_mod._eval_cp_policy(Q, default_discretizer, n_episodes=20, seed=0)
    assert stats["mean_episode_len"] > 0


# ── _run_phase5_final_job ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tiny_cp_job(phase5_mod):
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
        "grid_config": None,
        "train_episodes": TINY_EPISODES,
        "eval_episodes": TINY_EVAL_EPISODES,
        "disable_early_stopping": True,
    }


def test_final_job_result_keys(phase5_mod, tiny_cp_job):
    result = phase5_mod._run_phase5_final_job(tiny_cp_job)
    for key in [
        "algorithm",
        "seed",
        "regime",
        "episodes_run",
        "train_wall_clock_s",
        "curve_episode",
        "curve_window_mean",
        "mean_episode_len",
    ]:
        assert key in result, f"missing key: {key}"


def test_final_job_episodes_run(phase5_mod, tiny_cp_job):
    result = phase5_mod._run_phase5_final_job(tiny_cp_job)
    assert result["episodes_run"] == TINY_EPISODES  # early stopping disabled


def test_final_job_curve_shapes(phase5_mod, tiny_cp_job):
    result = phase5_mod._run_phase5_final_job(tiny_cp_job)
    assert result["curve_episode"].dtype == np.int32
    assert result["curve_window_mean"].dtype == np.float32
    assert len(result["curve_episode"]) == len(result["curve_window_mean"])
    assert len(result["curve_episode"]) == TINY_EPISODES // 100


def test_final_job_mean_episode_len_range(phase5_mod, tiny_cp_job):
    result = phase5_mod._run_phase5_final_job(tiny_cp_job)
    assert 1.0 <= result["mean_episode_len"] <= 500.0


# ── _run_disc_job ─────────────────────────────────────────────────────────────


def test_disc_job_result_keys(phase5_mod):
    from src.config import CARTPOLE_GRID_CONFIGS

    hp = {
        "alpha_start": 0.5,
        "alpha_end": 0.1,
        "alpha_decay_steps": 500,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_steps": 200,
        "gamma": 0.99,
    }
    job = {
        "algorithm": "qlearning",
        "seed": 42,
        "grid_name": "coarse",
        "hp": hp,
        "grid_config": CARTPOLE_GRID_CONFIGS["coarse"],
        "train_episodes": TINY_EPISODES,
        "eval_episodes": TINY_EVAL_EPISODES,
    }
    result = phase5_mod._run_disc_job(job)
    for key in [
        "algorithm",
        "seed",
        "grid_name",
        "episodes_run",
        "train_wall_clock_s",
        "mean_episode_len",
        "convergence_episode",
    ]:
        assert key in result, f"missing key: {key}"
    assert result["grid_name"] == "coarse"
    assert result["mean_episode_len"] >= 1.0


# ── run() smoke test ──────────────────────────────────────────────────────────


@pytest.mark.slow
def test_run_lifecycle(tmp_path, monkeypatch):
    """End-to-end smoke: run() writes all expected Phase 5 artifacts without error.

    Uses a drastically reduced budget:
    - 1 HP stage-1 config × 100 episodes
    - top-k=1 for stages 2 and 3
    - 100 final training episodes × 2 seeds
    - serial execution
    """
    import json
    from unittest.mock import patch

    p = Path("scripts/run_phase_5_model_free_cartpole.py")
    spec = importlib.util.spec_from_file_location("_phase5_smoke", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    patches = {
        "CP_HP_STAGE1_CONFIGS": 1,
        "CP_HP_STAGE1_EPISODES": 100,
        "CP_HP_STAGE2_TOP_K": 1,
        "CP_HP_STAGE2_EPISODES": 100,
        "CP_HP_STAGE3_TOP_K": 1,
        "CP_HP_STAGE3_EPISODES": 100,
        "CP_TRAIN_EPISODES": 100,
        "CP_DISC_TRAIN_EPISODES": 100,
        "CP_EVAL_EPISODES_MAIN": 20,
        "CP_EVAL_EPISODES_HP": 20,
        "SEEDS": [42, 43],
        "PHASE5_FINAL_TRAIN_MAX_WORKERS": 1,
        "PHASE5_HP_SEARCH_MAX_WORKERS": 1,
        # Only run one grid to keep the test fast
        "CARTPOLE_GRID_NAMES": ["default"],
    }
    for name, val in patches.items():
        monkeypatch.setattr(mod, name, val)

    from src.utils.phase_artifacts import PhasePaths

    fake_paths = PhasePaths(
        phase_id="phase5",
        slug="model_free_cartpole",
        metrics_dir=tmp_path / "metrics",
        figures_dir=tmp_path / "figures",
        metadata_dir=tmp_path / "metadata",
        logs_dir=Path("tmp") / "smoke_logs",
    )
    fake_paths.makedirs()

    with patch.object(mod, "resolve_phase_paths", return_value=fake_paths):
        checkpoint_path = mod.run()

    assert checkpoint_path.exists(), "checkpoint not written"

    for name in [
        "mf_hp_search.csv",
        "mf_learning_curves.csv",
        "mf_eval_per_seed.csv",
        "mf_eval_summary.csv",
        "mf_discretization.csv",
    ]:
        assert (fake_paths.metrics_dir / name).exists(), f"missing {name}"

    with open(checkpoint_path) as f:
        ckpt = json.load(f)
    assert "summary" in ckpt
    for regime in ["controlled", "tuned"]:
        for algo in ["sarsa", "qlearning"]:
            s = ckpt["summary"][regime][algo]
            assert "mean_episode_len" in s
            assert "final_window_iqr" in s

    import pandas as pd

    per_seed = pd.read_csv(fake_paths.metrics_dir / "mf_eval_per_seed.csv")
    for col in [
        "algorithm",
        "seed",
        "regime",
        "mean_episode_len",
        "final_window_return",
        "convergence_episode",
        "train_wall_clock_s",
    ]:
        assert col in per_seed.columns, f"missing column: {col}"
    assert set(per_seed["regime"].unique()) == {"controlled", "tuned"}
    assert set(per_seed["algorithm"].unique()) == {"sarsa", "qlearning"}

    disc = pd.read_csv(fake_paths.metrics_dir / "mf_discretization.csv")
    for col in ["grid", "algorithm", "seed", "final_mean_len", "convergence_episode"]:
        assert col in disc.columns, f"missing disc column: {col}"
