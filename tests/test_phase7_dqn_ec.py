"""Tests for Phase 7 DQN extra-credit pipeline.

Covers:
  - run() lifecycle: trains both variants, writes CSVs + phase7.json + 2 figures
  - checkpoint schema: all required keys, variant sections, greedy eval metrics
  - visualize() can re-render both figures from checkpoint alone
  - DQN primitive units: forward shape, train_step, target copy, replay buffer
  - evaluate_dqn_greedy: returns correct number of episodes

All compute-heavy tests mock _run_phase7_job (the parallel worker) so Phase 5
completion and real DQN training are not required.
"""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest


# ── Shared fixture helpers ────────────────────────────────────────────────────

_N_EVAL = 5  # episodes returned by the fake greedy eval


def _fake_wm_trace(n_windows: int = 5, conv_window: int = 100) -> list[dict]:
    """Minimal window-checkpoint trace (one entry per 100 episodes)."""
    return [
        {
            "episode": i * conv_window,
            "window_mean": float(150 + i * 20),
        }
        for i in range(1, n_windows + 1)
    ]


def _fake_job_result(variant: str, seed: int) -> dict:
    """Pre-fabricated result from _run_phase7_job for one (variant, seed)."""
    wm_trace = _fake_wm_trace()
    return {
        "variant": variant,
        "seed": seed,
        "wm_trace": wm_trace,
        "all_ep_lens": [120.0] * 500,
        "eval_ep_lens": [180.0 + i for i in range(_N_EVAL)],
        "final_window_return": wm_trace[-1]["window_mean"],
        "convergence_episode": 500,
        "train_wall_clock_s": 30.0 + seed * 0.1,
    }


def _minimal_p5() -> dict:
    return {
        "summary": {
            "tuned": {
                "sarsa": {"mean_episode_len": 325.3, "iqr_episode_len": 311.1},
                "qlearning": {"mean_episode_len": 376.8, "iqr_episode_len": 111.0},
            }
        }
    }


# ── Module loader ─────────────────────────────────────────────────────────────


def _load_phase7_module():
    p = Path("scripts/run_phase_7_dqn_ec.py")
    spec = importlib.util.spec_from_file_location("_phase7_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── run() + visualize() lifecycle ─────────────────────────────────────────────


@pytest.mark.slow
def test_run_lifecycle(tmp_path):
    """run() writes phase7.json + CSVs; visualize() renders both figures."""
    from src.utils.phase_artifacts import PhasePaths

    mod = _load_phase7_module()

    fake_paths = PhasePaths(
        phase_id="phase7",
        slug="dqn_ec",
        metrics_dir=tmp_path / "metrics",
        figures_dir=tmp_path / "figures",
        metadata_dir=tmp_path / "metadata",
        logs_dir=tmp_path / "logs",
    )
    fake_paths.makedirs()

    # Build fake results for all (variant, seed) combos
    from src.config import SEEDS

    fake_results = [
        _fake_job_result(v, s) for v in ["vanilla_dqn", "double_dqn"] for s in SEEDS
    ]

    # Capture job submissions to return fake results (serial path: n_workers=1)
    with (
        patch.object(mod, "resolve_phase_paths", return_value=fake_paths),
        patch.object(mod, "load_checkpoint_json", return_value=_minimal_p5()),
        patch.object(mod, "PHASE7_MAX_WORKERS", 1),
        patch.object(mod, "_run_phase7_job", side_effect=fake_results),
    ):
        checkpoint_path = mod.run()

    assert checkpoint_path.exists(), "phase7.json not written"

    with open(checkpoint_path) as f:
        ckpt = json.load(f)

    # Top-level schema
    for key in [
        "schema_version",
        "phase_id",
        "slug",
        "upstream_inputs",
        "outputs",
        "config_snapshot",
        "summary",
    ]:
        assert key in ckpt, f"missing top-level key: {key}"

    summary = ckpt["summary"]
    assert "variants" in summary
    assert "learning_curves" in summary
    assert "tabular_comparison" in summary

    # Both variants present with greedy eval keys
    for variant in ["vanilla_dqn", "double_dqn"]:
        assert variant in summary["variants"], f"missing variant: {variant}"
        v = summary["variants"][variant]
        assert "mean_final_ep_len" in v
        assert "final_ep_len_iqr" in v
        assert "mean_eval_ep_len" in v, "greedy eval metric missing"
        assert "eval_ep_len_iqr" in v, "greedy eval IQR missing"
        lc = summary["learning_curves"][variant]
        for k in ["episodes", "mean", "q25", "q75"]:
            assert k in lc, f"missing learning_curve key: {k}"

    # Tabular comparison keys
    for key in [
        "sarsa_tuned_mean_ep_len",
        "sarsa_tuned_ep_len_iqr",
        "qlearning_tuned_mean_ep_len",
        "qlearning_tuned_ep_len_iqr",
    ]:
        assert key in summary["tabular_comparison"], f"missing tabular key: {key}"

    # CSVs written
    assert (fake_paths.metrics_dir / "dqn_learning_curves.csv").exists()
    eval_df_path = fake_paths.metrics_dir / "dqn_eval_per_seed.csv"
    assert eval_df_path.exists()

    import pandas as pd

    eval_df = pd.read_csv(eval_df_path)
    assert "mean_eval_ep_len" in eval_df.columns, "greedy eval column missing from CSV"

    # Figures written
    mod.visualize(checkpoint_path)
    for fname in ["cartpole_dqn_vs_double_dqn.png", "cartpole_dqn_vs_tabular.png"]:
        assert (fake_paths.figures_dir / fname).exists(), f"missing figure: {fname}"


# ── visualize() from checkpoint ───────────────────────────────────────────────


@pytest.mark.slow
def test_visualize_from_checkpoint(tmp_path):
    """visualize() re-renders both figures from a saved checkpoint alone."""
    mod = _load_phase7_module()
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()

    ckpt = {
        "schema_version": "1.0",
        "phase_id": "phase7",
        "slug": "dqn_ec",
        "upstream_inputs": ["artifacts/metadata/phase5.json"],
        "outputs": {
            "figures_dir": str(figures_dir),
            "metrics_dir": str(tmp_path / "metrics"),
        },
        "config_snapshot": {},
        "summary": {
            "variants": {
                "vanilla_dqn": {
                    "mean_final_ep_len": 362.6,
                    "final_ep_len_iqr": 81.1,
                    "mean_eval_ep_len": 420.5,
                    "eval_ep_len_iqr": 60.0,
                    "mean_convergence_episode": None,
                    "convergence_episode_iqr": None,
                },
                "double_dqn": {
                    "mean_final_ep_len": 364.0,
                    "final_ep_len_iqr": 87.5,
                    "mean_eval_ep_len": 450.2,
                    "eval_ep_len_iqr": 40.0,
                    "mean_convergence_episode": None,
                    "convergence_episode_iqr": None,
                },
            },
            "learning_curves": {
                "vanilla_dqn": {
                    "episodes": [100, 200, 300, 400, 500],
                    "mean": [150.0, 250.0, 350.0, 410.0, 420.0],
                    "q25": [120.0, 200.0, 300.0, 380.0, 400.0],
                    "q75": [180.0, 300.0, 400.0, 440.0, 440.0],
                },
                "double_dqn": {
                    "episodes": [100, 200, 300, 400, 500],
                    "mean": [160.0, 270.0, 370.0, 430.0, 450.0],
                    "q25": [130.0, 220.0, 320.0, 400.0, 430.0],
                    "q75": [190.0, 320.0, 420.0, 460.0, 470.0],
                },
            },
            "tabular_comparison": {
                "sarsa_tuned_mean_ep_len": 325.3,
                "sarsa_tuned_ep_len_iqr": 311.1,
                "qlearning_tuned_mean_ep_len": 376.8,
                "qlearning_tuned_ep_len_iqr": 111.0,
            },
        },
    }

    ckpt_path = tmp_path / "phase7.json"
    with open(ckpt_path, "w") as f:
        json.dump(ckpt, f)

    with patch.object(mod, "load_checkpoint_json", return_value=ckpt):
        figs = mod.visualize(ckpt_path)

    assert len(figs) == 2
    for fig in figs:
        assert fig.exists(), f"figure not written: {fig}"


# ── DQN primitive units ───────────────────────────────────────────────────────


def test_dqn_forward_shape():
    """_MLP forward pass returns correct (batch, n_actions) shape."""
    import numpy as np
    from src.algorithms.dqn import _MLP

    rng = np.random.default_rng(0)
    net = _MLP(obs_dim=4, hidden_dim=32, n_actions=2, rng=rng)
    x = rng.standard_normal((8, 4))
    out = net.forward(x)
    assert out.shape == (8, 2)


def test_dqn_train_step_runs():
    """train_step executes without error and returns a finite loss."""
    import numpy as np
    from src.algorithms.dqn import _MLP

    rng = np.random.default_rng(1)
    net = _MLP(obs_dim=4, hidden_dim=32, n_actions=2, rng=rng)
    obs = rng.standard_normal((16, 4))
    actions = rng.integers(0, 2, size=16)
    targets = rng.standard_normal(16) * 100.0
    loss = net.train_step(obs, actions, targets, lr=1e-3)
    assert isinstance(loss, float)
    assert loss == loss  # not NaN


def test_target_network_copy():
    """copy_weights_from produces parameter-identical networks."""
    import numpy as np
    from src.algorithms.dqn import _MLP

    rng = np.random.default_rng(2)
    online = _MLP(obs_dim=4, hidden_dim=32, n_actions=2, rng=rng)
    target = _MLP(obs_dim=4, hidden_dim=32, n_actions=2, rng=rng)
    target.copy_weights_from(online)

    x = rng.standard_normal((4, 4))
    np.testing.assert_array_equal(online.forward(x), target.forward(x))


def test_replay_buffer_sample():
    """ReplayBuffer fills and samples without error."""
    import numpy as np
    from src.algorithms.dqn import _ReplayBuffer

    buf = _ReplayBuffer(capacity=100, obs_dim=4)
    rng = np.random.default_rng(3)
    for _ in range(50):
        obs = rng.standard_normal(4)
        buf.add(obs, int(rng.integers(0, 2)), 1.0, rng.standard_normal(4), False)

    assert len(buf) == 50
    s, a, r, ns, d = buf.sample(16, rng)
    assert s.shape == (16, 4)
    assert a.shape == (16,)


def test_evaluate_dqn_greedy():
    """evaluate_dqn_greedy returns exactly n_episodes episode lengths."""
    import numpy as np
    from src.algorithms.dqn import DQNConfig, _MLP, evaluate_dqn_greedy

    rng = np.random.default_rng(4)
    config = DQNConfig()
    net = _MLP(config.obs_dim, config.hidden_dim, config.n_actions, rng)
    ep_lens = evaluate_dqn_greedy(net, config, n_episodes=10, seed=99)
    assert len(ep_lens) == 10
    assert all(isinstance(v, float) and v >= 1.0 for v in ep_lens)
