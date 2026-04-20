"""Phase 6 — Cross-Method Comparison.

Lifecycle:
    run()               → load upstream checkpoints, assemble comparison data,
                          write phase6.json, return checkpoint path
    visualize(path)     → reload from phase6.json only, render all 5 figures

Five independent comparison outputs (regime assignment is fixed):
  1. planning_efficiency_comparison.png   — VI vs PI iterations, both MDPs
  2. learning_efficiency_comparison.png   — SARSA vs Q-Learning episodes to
                                            convergence, controlled, both MDPs
  3. stability_comparison.png             — final_window_iqr + convergence_
                                            episode_iqr, controlled, both MDPs
  4. wall_clock_comparison.png            — all methods; CartPole DP stacked
                                            (model-build + planning)
  5. final_performance_comparison.png     — all methods; model-free from tuned

Upstream inputs:
    artifacts/metadata/phase1.json  — CartPole model-build wall-clock
    artifacts/metadata/phase2.json  — Blackjack VI/PI
    artifacts/metadata/phase3.json  — CartPole VI/PI (default grid only)
    artifacts/metadata/phase4.json  — Blackjack SARSA/Q-Learning
    artifacts/metadata/phase5.json  — CartPole SARSA/Q-Learning
    artifacts/metrics/phase4_model_free_blackjack/mf_eval_per_seed.csv
    artifacts/metrics/phase5_model_free_cartpole/mf_eval_per_seed.csv

Outputs:
    artifacts/figures/phase6_comparison/  (5 PNG files)
    artifacts/metadata/phase6.json

Usage:
    make phase6
"""

from pathlib import Path

from src.utils.phase_artifacts import (
    SCHEMA_VERSION,
    load_checkpoint_json,
    resolve_phase_paths,
    write_checkpoint_json,
)
from src.utils.plotting import (
    plot_p6_final_performance,
    plot_p6_learning_efficiency,
    plot_p6_planning_efficiency,
    plot_p6_stability,
    plot_p6_wall_clock,
)

_PHASE_ID = "phase6"
_SLUG = "comparison"


def run() -> Path:
    """Load all upstream checkpoints, assemble comparison data, write phase6.json."""
    import pandas as pd

    from src.utils.logger import configure_logger

    paths = resolve_phase_paths(_PHASE_ID, _SLUG)
    paths.makedirs()
    log = configure_logger("phase6", log_dir=paths.logs_dir)

    log.info("=== Phase 6 Cross-Method Comparison ===")

    # ── Load upstream checkpoints ─────────────────────────────────────────────
    meta_dir = Path("artifacts/metadata")
    p1 = load_checkpoint_json(meta_dir / "phase1.json")
    p2 = load_checkpoint_json(meta_dir / "phase2.json")
    p3 = load_checkpoint_json(meta_dir / "phase3.json")
    p4 = load_checkpoint_json(meta_dir / "phase4.json")
    p5 = load_checkpoint_json(meta_dir / "phase5.json")

    # ── Load per-seed CSVs for wall-clock (per-algo controlled breakdown) ─────
    bj_per_seed = pd.read_csv(
        "artifacts/metrics/phase4_model_free_blackjack/mf_eval_per_seed.csv"
    )
    cp_per_seed = pd.read_csv(
        "artifacts/metrics/phase5_model_free_cartpole/mf_eval_per_seed.csv"
    )

    def _mf_controlled_wall_clock(df: pd.DataFrame, algo: str) -> float:
        """Approximate parallel wall-clock: max seed time for (algo, controlled)."""
        sub = df[(df["algorithm"] == algo) & (df["regime"] == "controlled")]
        return round(float(sub["train_wall_clock_s"].max()), 2)

    # ── 1. Planning efficiency ─────────────────────────────────────────────────
    log.info("Assembling planning efficiency data …")
    planning = {
        "blackjack": {
            "vi": {"iterations": p2["summary"]["vi"]["convergence_iter"]},
            "pi": {"iterations": p2["summary"]["pi"]["stable_iter"]},
        },
        "cartpole": {
            "vi": {"iterations": p3["summary"]["default"]["vi"]["iterations"]},
            "pi": {"iterations": p3["summary"]["default"]["pi"]["iterations"]},
        },
    }

    # ── 2. Learning efficiency (controlled regime) ────────────────────────────
    log.info("Assembling learning efficiency data …")
    learning_efficiency = {
        "blackjack": {
            algo: {
                "mean_convergence_episode": p4["summary"]["controlled"][algo][
                    "mean_convergence_episode"
                ],
                "convergence_episode_iqr": p4["summary"]["controlled"][algo][
                    "convergence_episode_iqr"
                ],
            }
            for algo in ["sarsa", "qlearning"]
        },
        "cartpole": {
            algo: {
                "mean_convergence_episode": p5["summary"]["controlled"][algo][
                    "mean_convergence_episode"
                ],
                "convergence_episode_iqr": p5["summary"]["controlled"][algo][
                    "convergence_episode_iqr"
                ],
            }
            for algo in ["sarsa", "qlearning"]
        },
    }

    # ── 3. Stability (controlled regime) ─────────────────────────────────────
    log.info("Assembling stability data …")
    stability = {
        "blackjack": {
            algo: {
                "final_window_iqr": p4["summary"]["controlled"][algo][
                    "final_window_iqr"
                ],
                "convergence_episode_iqr": p4["summary"]["controlled"][algo][
                    "convergence_episode_iqr"
                ],
            }
            for algo in ["sarsa", "qlearning"]
        },
        "cartpole": {
            algo: {
                "final_window_iqr": p5["summary"]["controlled"][algo][
                    "final_window_iqr"
                ],
                "convergence_episode_iqr": p5["summary"]["controlled"][algo][
                    "convergence_episode_iqr"
                ],
            }
            for algo in ["sarsa", "qlearning"]
        },
    }

    # ── 4. Wall-clock ─────────────────────────────────────────────────────────
    log.info("Assembling wall-clock data …")
    cp_model_build_s = p1["cartpole"]["wall_clock_s"]
    wall_clock = {
        "blackjack": {
            "vi": p2["summary"]["vi"]["wall_clock_s"],
            "pi": p2["summary"]["pi"]["wall_clock_s"],
            "sarsa": _mf_controlled_wall_clock(bj_per_seed, "sarsa"),
            "qlearning": _mf_controlled_wall_clock(bj_per_seed, "qlearning"),
        },
        "cartpole": {
            "model_build_s": cp_model_build_s,
            "vi_planning_s": p3["summary"]["default"]["vi"]["wall_clock_s"],
            "pi_planning_s": p3["summary"]["default"]["pi"]["wall_clock_s"],
            "vi": round(
                cp_model_build_s + p3["summary"]["default"]["vi"]["wall_clock_s"], 3
            ),
            "pi": round(
                cp_model_build_s + p3["summary"]["default"]["pi"]["wall_clock_s"], 3
            ),
            "sarsa": _mf_controlled_wall_clock(cp_per_seed, "sarsa"),
            "qlearning": _mf_controlled_wall_clock(cp_per_seed, "qlearning"),
        },
    }

    # ── 5. Final performance (tuned for model-free, single run for DP) ────────
    log.info("Assembling final performance data …")
    final_performance = {
        "blackjack": {
            "vi": {
                "mean_eval_return": p2["summary"]["vi"]["mean_eval_return"],
                "eval_return_iqr": p2["summary"]["vi"]["eval_return_iqr"],
            },
            "pi": {
                "mean_eval_return": p2["summary"]["pi"]["mean_eval_return"],
                "eval_return_iqr": p2["summary"]["pi"]["eval_return_iqr"],
            },
            "sarsa": {
                "mean_return": p4["summary"]["tuned"]["sarsa"]["mean_return"],
                "iqr_return": p4["summary"]["tuned"]["sarsa"]["iqr_return"],
            },
            "qlearning": {
                "mean_return": p4["summary"]["tuned"]["qlearning"]["mean_return"],
                "iqr_return": p4["summary"]["tuned"]["qlearning"]["iqr_return"],
            },
        },
        "cartpole": {
            "vi": {
                "mean_episode_len": p3["summary"]["default"]["vi"]["mean_episode_len"],
                "eval_episode_len_iqr": p3["summary"]["default"]["vi"][
                    "eval_episode_len_iqr"
                ],
            },
            "pi": {
                "mean_episode_len": p3["summary"]["default"]["pi"]["mean_episode_len"],
                "eval_episode_len_iqr": p3["summary"]["default"]["pi"][
                    "eval_episode_len_iqr"
                ],
            },
            "sarsa": {
                "mean_episode_len": p5["summary"]["tuned"]["sarsa"]["mean_episode_len"],
                "iqr_episode_len": p5["summary"]["tuned"]["sarsa"]["iqr_episode_len"],
            },
            "qlearning": {
                "mean_episode_len": p5["summary"]["tuned"]["qlearning"][
                    "mean_episode_len"
                ],
                "iqr_episode_len": p5["summary"]["tuned"]["qlearning"][
                    "iqr_episode_len"
                ],
            },
        },
    }

    log.info("Planning efficiency assembled.")
    log.info(
        "  Blackjack — VI: %d iters, PI: %d iters",
        planning["blackjack"]["vi"]["iterations"],
        planning["blackjack"]["pi"]["iterations"],
    )
    log.info(
        "  CartPole  — VI: %d iters, PI: %d iters",
        planning["cartpole"]["vi"]["iterations"],
        planning["cartpole"]["pi"]["iterations"],
    )
    log.info("Learning efficiency assembled (controlled).")
    for env, env_key in [("Blackjack", "blackjack"), ("CartPole", "cartpole")]:
        for algo in ["sarsa", "qlearning"]:
            d = learning_efficiency[env_key][algo]
            log.info(
                "  %s %s: conv_ep=%.0f iqr=%.0f",
                env,
                algo,
                d["mean_convergence_episode"] or float("nan"),
                d["convergence_episode_iqr"] or float("nan"),
            )

    # ── Write checkpoint ──────────────────────────────────────────────────────
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": _PHASE_ID,
        "slug": _SLUG,
        "upstream_inputs": [
            *[str(meta_dir / f"phase{n}.json") for n in [1, 2, 3, 4, 5]],
            "artifacts/metrics/phase4_model_free_blackjack/mf_eval_per_seed.csv",
            "artifacts/metrics/phase5_model_free_cartpole/mf_eval_per_seed.csv",
        ],
        "outputs": {
            "figures_dir": str(paths.figures_dir),
        },
        "config_snapshot": {
            "cartpole_grid": "default",
            "mf_regime_efficiency_stability": "controlled",
            "mf_regime_final_performance": "tuned",
        },
        "summary": {
            "planning_efficiency": planning,
            "learning_efficiency": learning_efficiency,
            "stability": stability,
            "wall_clock": wall_clock,
            "final_performance": final_performance,
        },
    }
    write_checkpoint_json(checkpoint, paths.checkpoint_path)
    log.info("Phase 6 checkpoint saved → %s", paths.checkpoint_path)

    return paths.checkpoint_path


def visualize(checkpoint_path: Path) -> list[Path]:
    """Render Phase 6 figures from saved checkpoint. No live computation."""
    from src.utils.logger import configure_logger

    log = configure_logger("phase6")
    checkpoint = load_checkpoint_json(checkpoint_path)
    figures_dir = Path(checkpoint["outputs"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    log.info("Rendering Phase 6 figures → %s", figures_dir)
    figs: list[Path] = []

    figs.append(plot_p6_planning_efficiency(checkpoint["summary"], figures_dir))
    log.info("  planning efficiency → %s", figs[-1])

    figs.append(plot_p6_learning_efficiency(checkpoint["summary"], figures_dir))
    log.info("  learning efficiency → %s", figs[-1])

    figs.append(plot_p6_stability(checkpoint["summary"], figures_dir))
    log.info("  stability → %s", figs[-1])

    figs.append(plot_p6_wall_clock(checkpoint["summary"], figures_dir))
    log.info("  wall clock → %s", figs[-1])

    figs.append(plot_p6_final_performance(checkpoint["summary"], figures_dir))
    log.info("  final performance → %s", figs[-1])

    log.info("Phase 6 figures done (%d files)", len(figs))
    return figs


if __name__ == "__main__":
    checkpoint_path = run()
    visualize(checkpoint_path)
