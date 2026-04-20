"""Tests for Phase 6 cross-method comparison pipeline.

Covers:
  - run() lifecycle: loads upstream checkpoints, writes phase6.json + 5 figures
  - checkpoint schema: all five comparison sections present with expected keys
  - visualize() can re-render from checkpoint alone

All tests use a mock upstream bundle so Phase 5 completion is not required.
"""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest


# ── Shared fixture helpers ────────────────────────────────────────────────────


def _minimal_p1():
    return {"cartpole": {"wall_clock_s": 36.56}}


def _minimal_p2():
    return {
        "summary": {
            "vi": {
                "convergence_iter": 13,
                "wall_clock_s": 0.001,
                "mean_eval_return": 0.08,
                "eval_return_iqr": 0.02,
            },
            "pi": {
                "stable_iter": 2,
                "wall_clock_s": 0.001,
                "mean_eval_return": 0.09,
                "eval_return_iqr": 0.01,
            },
        }
    }


def _minimal_p3():
    return {
        "summary": {
            "default": {
                "vi": {
                    "iterations": 94,
                    "wall_clock_s": 0.59,
                    "mean_episode_len": 320.0,
                    "eval_episode_len_iqr": 40.0,
                },
                "pi": {
                    "iterations": 8,
                    "wall_clock_s": 0.05,
                    "mean_episode_len": 310.0,
                    "eval_episode_len_iqr": 50.0,
                    "policy_changes_at_convergence": 0,
                    "stop_reason": "stable",
                },
            }
        }
    }


def _minimal_p4():
    def _algo(mean_conv, conv_iqr, fw_iqr):
        return {
            "mean_return": 0.08,
            "std_return": 0.01,
            "iqr_return": 0.02,
            "mean_final_return": 0.07,
            "final_window_iqr": fw_iqr,
            "mean_win_rate": 0.40,
            "mean_draw_rate": 0.08,
            "mean_loss_rate": 0.32,
            "mean_convergence_episode": mean_conv,
            "convergence_episode_iqr": conv_iqr,
        }

    return {
        "summary": {
            "controlled": {
                "sarsa": _algo(100_000, 50_000, 0.03),
                "qlearning": _algo(80_000, 40_000, 0.02),
            },
            "tuned": {
                "sarsa": _algo(90_000, 45_000, 0.025),
                "qlearning": _algo(70_000, 35_000, 0.018),
            },
            "final_training": {"n_workers": 9, "train_wall_clock_s": 84.7},
        }
    }


def _minimal_p5():
    def _algo(mean_conv, conv_iqr, fw_iqr, mean_len, iqr_len):
        return {
            "mean_episode_len": mean_len,
            "std_episode_len": 60.0,
            "iqr_episode_len": iqr_len,
            "mean_final_return": mean_len,
            "final_window_iqr": fw_iqr,
            "mean_convergence_episode": mean_conv,
            "convergence_episode_iqr": conv_iqr,
        }

    return {
        "summary": {
            "controlled": {
                "sarsa": _algo(1160, 600, 207.6, 275.6, 325.7),
                "qlearning": _algo(1140, 600, 290.6, 326.8, 309.1),
            },
            "tuned": {
                "sarsa": _algo(1860, 2400, 171.8, 338.7, 332.9),
                "qlearning": _algo(3300, 400, 92.2, 360.9, 104.8),
            },
        }
    }


def _minimal_bj_per_seed():
    import pandas as pd

    rows = []
    for algo in ["sarsa", "qlearning"]:
        for seed in [42, 43, 44, 45, 46]:
            rows.append(
                {
                    "algorithm": algo,
                    "seed": seed,
                    "regime": "controlled",
                    "train_wall_clock_s": 30.0 + seed * 0.1,
                    "mean_return": 0.08,
                    "final_window_return": 0.07,
                    "convergence_episode": 100_000,
                }
            )
    return pd.DataFrame(rows)


def _minimal_cp_per_seed():
    import pandas as pd

    rows = []
    for algo in ["sarsa", "qlearning"]:
        for seed in [42, 43, 44, 45, 46]:
            rows.append(
                {
                    "algorithm": algo,
                    "seed": seed,
                    "regime": "controlled",
                    "train_wall_clock_s": 70.0 + seed * 0.5,
                    "mean_episode_len": 275.0,
                    "final_window_return": 270.0,
                    "convergence_episode": 1160,
                }
            )
    return pd.DataFrame(rows)


# ── Module loader ─────────────────────────────────────────────────────────────


def _load_phase6_module():
    p = Path("scripts/run_phase_6_comparison.py")
    spec = importlib.util.spec_from_file_location("_phase6_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── run() lifecycle smoke test ────────────────────────────────────────────────


@pytest.mark.slow
def test_run_lifecycle(tmp_path):
    """run() writes phase6.json; visualize() renders all 5 figures from disk."""

    from src.utils.phase_artifacts import PhasePaths

    mod = _load_phase6_module()

    fake_paths = PhasePaths(
        phase_id="phase6",
        slug="comparison",
        metrics_dir=tmp_path / "metrics",
        figures_dir=tmp_path / "figures",
        metadata_dir=tmp_path / "metadata",
        logs_dir=tmp_path / "logs",
    )
    fake_paths.makedirs()

    with (
        patch.object(mod, "resolve_phase_paths", return_value=fake_paths),
        patch.object(
            mod,
            "load_checkpoint_json",
            side_effect=[
                _minimal_p1(),
                _minimal_p2(),
                _minimal_p3(),
                _minimal_p4(),
                _minimal_p5(),
            ],
        ),
        patch(
            "pandas.read_csv",
            side_effect=[_minimal_bj_per_seed(), _minimal_cp_per_seed()],
        ),
    ):
        checkpoint_path = mod.run()

    mod.visualize(checkpoint_path)

    assert checkpoint_path.exists(), "phase6.json not written"

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

    # All five sections present
    for section in [
        "planning_efficiency",
        "learning_efficiency",
        "stability",
        "wall_clock",
        "final_performance",
    ]:
        assert section in summary, f"missing summary section: {section}"

    # planning_efficiency structure
    pe = summary["planning_efficiency"]
    for env in ["blackjack", "cartpole"]:
        for method in ["vi", "pi"]:
            assert "iterations" in pe[env][method]

    # learning_efficiency structure
    le = summary["learning_efficiency"]
    for env in ["blackjack", "cartpole"]:
        for algo in ["sarsa", "qlearning"]:
            assert "mean_convergence_episode" in le[env][algo]
            assert "convergence_episode_iqr" in le[env][algo]

    # stability structure
    st = summary["stability"]
    for env in ["blackjack", "cartpole"]:
        for algo in ["sarsa", "qlearning"]:
            assert "final_window_iqr" in st[env][algo]
            assert "convergence_episode_iqr" in st[env][algo]

    # wall_clock structure
    wc = summary["wall_clock"]
    for method in ["vi", "pi", "sarsa", "qlearning"]:
        assert method in wc["blackjack"]
        assert method in wc["cartpole"]
    assert "model_build_s" in wc["cartpole"]

    # final_performance structure
    fp = summary["final_performance"]
    assert "mean_eval_return" in fp["blackjack"]["vi"]
    assert "mean_episode_len" in fp["cartpole"]["vi"]
    assert "mean_return" in fp["blackjack"]["sarsa"]
    assert "mean_episode_len" in fp["cartpole"]["sarsa"]

    # Figures written
    expected_figs = [
        "planning_efficiency_comparison.png",
        "learning_efficiency_comparison.png",
        "stability_comparison.png",
        "wall_clock_comparison.png",
        "final_performance_comparison.png",
    ]
    for fname in expected_figs:
        assert (fake_paths.figures_dir / fname).exists(), f"missing figure: {fname}"


@pytest.mark.slow
def test_visualize_from_checkpoint(tmp_path):
    """visualize() re-renders all 5 figures from a saved checkpoint alone."""
    mod = _load_phase6_module()
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()

    ckpt = {
        "schema_version": "1.0",
        "phase_id": "phase6",
        "slug": "comparison",
        "upstream_inputs": [],
        "outputs": {"figures_dir": str(figures_dir)},
        "config_snapshot": {},
        "summary": {
            "planning_efficiency": {
                "blackjack": {"vi": {"iterations": 13}, "pi": {"iterations": 2}},
                "cartpole": {"vi": {"iterations": 94}, "pi": {"iterations": 8}},
            },
            "learning_efficiency": {
                "blackjack": {
                    "sarsa": {
                        "mean_convergence_episode": 100_000,
                        "convergence_episode_iqr": 50_000,
                    },
                    "qlearning": {
                        "mean_convergence_episode": 80_000,
                        "convergence_episode_iqr": 40_000,
                    },
                },
                "cartpole": {
                    "sarsa": {
                        "mean_convergence_episode": 1160,
                        "convergence_episode_iqr": 600,
                    },
                    "qlearning": {
                        "mean_convergence_episode": 1140,
                        "convergence_episode_iqr": 600,
                    },
                },
            },
            "stability": {
                "blackjack": {
                    "sarsa": {
                        "final_window_iqr": 0.03,
                        "convergence_episode_iqr": 50_000,
                    },
                    "qlearning": {
                        "final_window_iqr": 0.02,
                        "convergence_episode_iqr": 40_000,
                    },
                },
                "cartpole": {
                    "sarsa": {
                        "final_window_iqr": 207.6,
                        "convergence_episode_iqr": 600,
                    },
                    "qlearning": {
                        "final_window_iqr": 290.6,
                        "convergence_episode_iqr": 600,
                    },
                },
            },
            "wall_clock": {
                "blackjack": {
                    "vi": 0.001,
                    "pi": 0.001,
                    "sarsa": 30.5,
                    "qlearning": 31.0,
                },
                "cartpole": {
                    "model_build_s": 36.56,
                    "vi_planning_s": 0.59,
                    "pi_planning_s": 0.05,
                    "vi": 37.15,
                    "pi": 36.61,
                    "sarsa": 72.5,
                    "qlearning": 74.3,
                },
            },
            "final_performance": {
                "blackjack": {
                    "vi": {"mean_eval_return": 0.08, "eval_return_iqr": 0.02},
                    "pi": {"mean_eval_return": 0.09, "eval_return_iqr": 0.01},
                    "sarsa": {"mean_return": 0.07, "iqr_return": 0.03},
                    "qlearning": {"mean_return": 0.08, "iqr_return": 0.02},
                },
                "cartpole": {
                    "vi": {"mean_episode_len": 320.0, "eval_episode_len_iqr": 40.0},
                    "pi": {"mean_episode_len": 310.0, "eval_episode_len_iqr": 50.0},
                    "sarsa": {"mean_episode_len": 338.7, "iqr_episode_len": 332.9},
                    "qlearning": {"mean_episode_len": 360.9, "iqr_episode_len": 104.8},
                },
            },
        },
    }
    ckpt_path = tmp_path / "phase6.json"
    with open(ckpt_path, "w") as f:
        json.dump(ckpt, f)

    with patch.object(mod, "load_checkpoint_json", return_value=ckpt):
        figs = mod.visualize(ckpt_path)

    assert len(figs) == 5
    for fig in figs:
        assert fig.exists(), f"figure not written: {fig}"


@pytest.mark.slow
def test_visualize_none_convergence(tmp_path):
    """visualize() renders without error when convergence values are None (non-converged run)."""
    from src.utils.plotting import plot_p6_learning_efficiency, plot_p6_stability

    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()

    learning_data = {
        "learning_efficiency": {
            "blackjack": {
                "sarsa": {
                    "mean_convergence_episode": None,
                    "convergence_episode_iqr": None,
                },
                "qlearning": {
                    "mean_convergence_episode": 80_000,
                    "convergence_episode_iqr": 40_000,
                },
            },
            "cartpole": {
                "sarsa": {
                    "mean_convergence_episode": None,
                    "convergence_episode_iqr": None,
                },
                "qlearning": {
                    "mean_convergence_episode": 1140,
                    "convergence_episode_iqr": 600,
                },
            },
        }
    }
    fig_path = plot_p6_learning_efficiency(learning_data, figures_dir)
    assert fig_path.exists(), (
        "learning_efficiency figure not written with None convergence"
    )

    stability_data = {
        "stability": {
            "blackjack": {
                "sarsa": {"final_window_iqr": None, "convergence_episode_iqr": None},
                "qlearning": {
                    "final_window_iqr": 0.02,
                    "convergence_episode_iqr": 40_000,
                },
            },
            "cartpole": {
                "sarsa": {"final_window_iqr": 207.6, "convergence_episode_iqr": None},
                "qlearning": {
                    "final_window_iqr": 290.6,
                    "convergence_episode_iqr": 600,
                },
            },
        }
    }
    fig_path = plot_p6_stability(stability_data, figures_dir)
    assert fig_path.exists(), "stability figure not written with None convergence"
