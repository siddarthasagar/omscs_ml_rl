"""Phase 3 — Model-Based: VI & PI on CartPole (discretization ablation study).

Builds T/R models for coarse/default/fine grids, runs VI and PI on each, and
evaluates the resulting policies.  Outputs per-grid convergence curves,
a discretization study figure, and all metric CSVs.

Outputs:
  artifacts/metrics/phase3_vi_pi_cartpole/
    vi_convergence.csv, pi_convergence.csv
    policy_eval_per_seed.csv, policy_eval_aggregate.csv
    policy_agreement.csv, discretization_study.csv
  artifacts/figures/phase3_vi_pi_cartpole/
    cartpole_vi_convergence.png, cartpole_pi_convergence.png
    cartpole_discretization_study.png
  artifacts/metadata/phase3.json
  artifacts/logs/phase3.log

Usage:
  uv run python scripts/run_phase_3_vi_pi_cartpole.py
"""

import json
import time

import numpy as np
import pandas as pd

from src.algorithms import eval_cartpole_policy, run_pi, run_vi
from src.config import (
    CARTPOLE_GRID_CONFIGS,
    CARTPOLE_MODEL_MIN_VISITS,
    CARTPOLE_MODEL_ROLLOUT_STEPS,
    CARTPOLE_MODEL_SEED,
    FIGURES_DIR,
    METADATA_DIR,
    METRICS_DIR,
    PI_DELTA,
    PI_GAMMA,
    SEEDS,
    VI_DELTA,
    VI_GAMMA,
    VI_PI_CONSEC_SWEEPS,
    VI_PI_HP_DELTA_VALUES,
    VI_PI_HP_GAMMA_VALUES,
)
from src.envs.cartpole_discretizer import CartPoleDiscretizer
from src.envs.cartpole_model import build_cartpole_model
from src.utils.logger import configure_logger

logger = configure_logger("phase3")

N_EVAL_EPISODES = 100
PHASE_DIR = "phase3_vi_pi_cartpole"
GRID_NAMES = ["coarse", "default", "fine"]
GRID_COLORS = {"coarse": "#DD8452", "default": "#4C72B0", "fine": "#55A868"}


# ── Per-grid runner ───────────────────────────────────────────────────────────


def _run_grid(grid_name: str) -> dict:
    """Build model, run VI and PI, evaluate policies for one grid config."""
    cfg = CARTPOLE_GRID_CONFIGS[grid_name]
    disc = CartPoleDiscretizer(grid_config=cfg)
    logger.info(
        "=== Grid '%s': bins=%s, n_states=%d ===",
        grid_name,
        list(cfg["bins"]),
        disc.n_states,
    )

    # Use per-grid rollout budget if specified; fall back to global constant.
    grid_rollout_steps = cfg.get("rollout_steps", CARTPOLE_MODEL_ROLLOUT_STEPS)
    logger.info("Grid '%s': using rollout_steps=%d", grid_name, grid_rollout_steps)

    # Build T/R model for this grid
    cp_model = build_cartpole_model(
        discretizer=disc,
        rollout_steps=grid_rollout_steps,
        min_visits=CARTPOLE_MODEL_MIN_VISITS,
        seed=CARTPOLE_MODEL_SEED,
        logger=logger,
    )
    T, R = cp_model["T"], cp_model["R"]

    # VI
    logger.info("Running VI on grid '%s'...", grid_name)
    t0 = time.perf_counter()
    V_vi, policy_vi, trace_vi = run_vi(
        T, R, VI_GAMMA, VI_DELTA, m_consec=VI_PI_CONSEC_SWEEPS, logger=logger
    )
    vi_wall = time.perf_counter() - t0
    logger.info("VI: %d iters, %.3fs", len(trace_vi), vi_wall)

    # PI
    logger.info("Running PI on grid '%s'...", grid_name)
    t0 = time.perf_counter()
    V_pi, policy_pi, trace_pi = run_pi(
        T, R, PI_GAMMA, PI_DELTA, m_consec=VI_PI_CONSEC_SWEEPS, logger=logger
    )
    pi_wall = time.perf_counter() - t0
    logger.info("PI: %d iters, %.3fs", len(trace_pi), pi_wall)

    # Policy agreement (non-absorbing states only)
    n = disc.n_states
    agreement = float((policy_vi[:n] == policy_pi[:n]).mean())
    logger.info("Grid '%s' VI vs PI agreement: %.1f%%", grid_name, agreement * 100)

    # Evaluate VI policy
    logger.info(
        "Evaluating VI on '%s' (%d seeds × %d eps)...",
        grid_name,
        len(SEEDS),
        N_EVAL_EPISODES,
    )
    vi_eval = eval_cartpole_policy(
        policy_vi, disc, seeds=SEEDS, n_episodes=N_EVAL_EPISODES
    )
    vi_lens = [ep_len for _, ep_len in vi_eval]
    vi_mean = float(np.mean(vi_lens))
    vi_iqr = float(np.percentile(vi_lens, 75) - np.percentile(vi_lens, 25))

    # Evaluate PI policy
    logger.info("Evaluating PI on '%s'...", grid_name)
    pi_eval = eval_cartpole_policy(
        policy_pi, disc, seeds=SEEDS, n_episodes=N_EVAL_EPISODES
    )
    pi_lens = [ep_len for _, ep_len in pi_eval]
    pi_mean = float(np.mean(pi_lens))
    pi_iqr = float(np.percentile(pi_lens, 75) - np.percentile(pi_lens, 25))

    logger.info(
        "Grid '%s': VI mean_len=%.1f, PI mean_len=%.1f",
        grid_name,
        vi_mean,
        pi_mean,
    )

    return {
        "grid_name": grid_name,
        "bins": list(cfg["bins"]),
        "n_states": disc.n_states,
        "coverage_pct": cp_model["coverage_pct"],
        "smoothed_pct": cp_model["smoothed_pct"],
        "rollout_steps": grid_rollout_steps,
        "T": T,
        "R": R,
        "disc": disc,
        "vi": {
            "iters": len(trace_vi),
            "wall_clock_s": round(vi_wall, 3),
            "mean_episode_len": vi_mean,
            "iqr": vi_iqr,
            "trace": trace_vi,
            "eval": vi_eval,
        },
        "pi": {
            "iters": len(trace_pi),
            "wall_clock_s": round(pi_wall, 3),
            "mean_episode_len": pi_mean,
            "iqr": pi_iqr,
            "trace": trace_pi,
            "eval": pi_eval,
        },
        "policy_vi": policy_vi,
        "policy_pi": policy_pi,
        "policy_agreement": agreement,
    }


# ── Figure helpers ────────────────────────────────────────────────────────────


def _plot_convergence_curves(
    grid_results: dict, algo_key: str, fig_dir, filename: str, title: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in GRID_NAMES:
        res = grid_results[name]
        trace = res[algo_key]["trace"]
        iters = [t["iteration"] for t in trace]
        dvs = [t["delta_v"] for t in trace]
        ax.semilogy(
            iters,
            dvs,
            label=f"{name} ({res['n_states']} states)",
            color=GRID_COLORS[name],
            linewidth=1.5,
        )

    delta = VI_DELTA if algo_key == "vi" else PI_DELTA
    ax.axhline(
        delta, color="red", linewidth=1, linestyle="--", label=f"δ = {delta:.0e}"
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("max ΔV (log scale)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = fig_dir / filename
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out)


def _plot_discretization_study(grid_results: dict, fig_dir) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = list(range(len(GRID_NAMES)))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel 1: mean episode length (policy quality)
    for algo, alg_key in [("VI", "vi"), ("PI", "pi")]:
        means = [grid_results[g][alg_key]["mean_episode_len"] for g in GRID_NAMES]
        iqrs = [grid_results[g][alg_key]["iqr"] for g in GRID_NAMES]
        axes[0].errorbar(
            x, means, yerr=iqrs, label=algo, marker="o", capsize=4, linewidth=1.5
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(GRID_NAMES)
    axes[0].set_xlabel("Grid")
    axes[0].set_ylabel("Mean episode length")
    axes[0].set_title("Policy quality vs grid")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: planning wall-clock time (model build included for CartPole?)
    for algo, alg_key in [("VI", "vi"), ("PI", "pi")]:
        walls = [grid_results[g][alg_key]["wall_clock_s"] for g in GRID_NAMES]
        axes[1].plot(x, walls, label=algo, marker="o", linewidth=1.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(GRID_NAMES)
    axes[1].set_xlabel("Grid")
    axes[1].set_ylabel("Planning wall-clock (s)")
    axes[1].set_title("Planning time vs grid")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: model coverage %
    coverages = [grid_results[g]["coverage_pct"] * 100 for g in GRID_NAMES]
    bars = axes[2].bar(
        x,
        coverages,
        color=[GRID_COLORS[g] for g in GRID_NAMES],
        alpha=0.85,
        edgecolor="white",
    )
    for bar, val in zip(bars, coverages):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(GRID_NAMES)
    axes[2].set_xlabel("Grid")
    axes[2].set_ylabel("Coverage (%)")
    axes[2].set_title("Model coverage vs grid")
    axes[2].set_ylim(0, 110)
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle("CartPole Discretization Study", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = fig_dir / "cartpole_discretization_study.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out)


def _plot_policy_slice(grid_results: dict, fig_dir) -> None:
    """Decision-boundary slice in (theta, thetadot) space at x=0, xdot=0.

    One column per grid (coarse/default/fine), two rows (VI / PI).
    Action 0 = push left (blue), Action 1 = push right (orange).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    ACTION_COLORS = ["#4C72B0", "#DD8452"]  # 0=left, 1=right
    cmap = mcolors.ListedColormap(ACTION_COLORS)

    n_grids = len(GRID_NAMES)
    fig, axes = plt.subplots(2, n_grids, figsize=(4 * n_grids, 7))

    for col, grid_name in enumerate(GRID_NAMES):
        res = grid_results[grid_name]
        disc = res["disc"]

        # Build a fine meshgrid over (theta, thetadot) at x=0, xdot=0
        theta_lim = 0.20  # rad — CartPole terminal threshold is ±0.2095
        thetadot_lim = 3.0
        N = 120
        thetas = np.linspace(-theta_lim, theta_lim, N)
        thetadots = np.linspace(-thetadot_lim, thetadot_lim, N)
        TH, TD = np.meshgrid(thetas, thetadots)

        obs_grid = np.stack(
            [
                np.zeros_like(TH.ravel()),  # x = 0
                np.zeros_like(TH.ravel()),  # xdot = 0
                TH.ravel(),
                TD.ravel(),
            ],
            axis=1,
        )

        states = np.array([disc.obs_to_state(o) for o in obs_grid])

        for row, (algo, policy_key) in enumerate(
            [("VI", "policy_vi"), ("PI", "policy_pi")]
        ):
            policy = res[policy_key]
            actions = policy[states].reshape(N, N)

            ax = axes[row, col]
            ax.imshow(
                actions,
                origin="lower",
                aspect="auto",
                extent=[-theta_lim, theta_lim, -thetadot_lim, thetadot_lim],
                cmap=cmap,
                vmin=-0.5,
                vmax=1.5,
                interpolation="nearest",
            )
            # Mark the terminal angle boundary
            ax.axvline(0.2095, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
            ax.axvline(-0.2095, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
            ax.axhline(0, color="white", linewidth=0.5, linestyle=":", alpha=0.5)
            ax.axvline(0, color="white", linewidth=0.5, linestyle=":", alpha=0.5)

            if row == 0:
                ax.set_title(f"{grid_name}\n({disc.n_states} states)", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{algo}\nθ̇ (rad/s)", fontsize=9)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            if row == 1:
                ax.set_xlabel("θ (rad)", fontsize=9)
            else:
                ax.set_xticklabels([])

    legend_patches = [
        mpatches.Patch(color=ACTION_COLORS[0], label="Push left (0)"),
        mpatches.Patch(color=ACTION_COLORS[1], label="Push right (1)"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=2,
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "CartPole Policy Slice: (θ, θ̇) at x=0, ẋ=0",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    out = fig_dir / "cartpole_policy_slice.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out)


HP_EVAL_EPISODES = 50  # lighter budget for the HP sweep


def _hp_sweep_cartpole(
    T: np.ndarray, R: np.ndarray, disc, grid_name: str = "default"
) -> list[dict]:
    """Sweep gamma and delta for VI and PI on the given CartPole grid model.

    Gamma sweep: vary gamma, hold delta at reference.
    Delta sweep: vary delta, hold gamma at 0.99.

    Returns list of row dicts for hp_validation.csv.
    """
    rows: list[dict] = []

    def _eval(policy):
        results = eval_cartpole_policy(
            policy, disc, seeds=SEEDS, n_episodes=HP_EVAL_EPISODES
        )
        lens = [ep_len for _, ep_len in results]
        return float(np.mean(lens)), float(
            np.percentile(lens, 75) - np.percentile(lens, 25)
        )

    # ── gamma sweep ───────────────────────────────────────────────────────────
    for gamma in VI_PI_HP_GAMMA_VALUES:
        for algo_label, run_fn, ref_delta in [
            ("VI", run_vi, VI_DELTA),
            ("PI", run_pi, PI_DELTA),
        ]:
            t0 = time.perf_counter()
            if algo_label == "VI":
                _, policy, trace = run_fn(
                    T, R, gamma, ref_delta, m_consec=VI_PI_CONSEC_SWEEPS, logger=logger
                )
            else:
                _, policy, trace = run_fn(T, R, gamma, ref_delta, logger=logger)
            wall = time.perf_counter() - t0
            mean_len, iqr = _eval(policy)
            rows.append(
                {
                    "algorithm": algo_label,
                    "grid": grid_name,
                    "sweep_param": "gamma",
                    "gamma": gamma,
                    "delta": ref_delta,
                    "iterations": len(trace),
                    "wall_clock_s": round(wall, 3),
                    "mean_episode_len": round(mean_len, 2),
                    "episode_len_iqr": round(iqr, 2),
                }
            )
            logger.info(
                "HP gamma sweep [%s/%s] gamma=%.2f delta=%.0e → %d iters, "
                "mean_len=%.1f, %.3fs",
                algo_label,
                grid_name,
                gamma,
                ref_delta,
                len(trace),
                mean_len,
                wall,
            )

    # ── delta sweep ───────────────────────────────────────────────────────────
    ref_gamma = 0.99
    for delta in VI_PI_HP_DELTA_VALUES:
        for algo_label, run_fn in [("VI", run_vi), ("PI", run_pi)]:
            t0 = time.perf_counter()
            if algo_label == "VI":
                _, policy, trace = run_fn(
                    T, R, ref_gamma, delta, m_consec=VI_PI_CONSEC_SWEEPS, logger=logger
                )
            else:
                _, policy, trace = run_fn(T, R, ref_gamma, delta, logger=logger)
            wall = time.perf_counter() - t0
            mean_len, iqr = _eval(policy)
            rows.append(
                {
                    "algorithm": algo_label,
                    "grid": grid_name,
                    "sweep_param": "delta",
                    "gamma": ref_gamma,
                    "delta": delta,
                    "iterations": len(trace),
                    "wall_clock_s": round(wall, 3),
                    "mean_episode_len": round(mean_len, 2),
                    "episode_len_iqr": round(iqr, 2),
                }
            )
            logger.info(
                "HP delta sweep [%s/%s] gamma=%.2f delta=%.0e → %d iters, "
                "mean_len=%.1f, %.3fs",
                algo_label,
                grid_name,
                delta,
                ref_gamma,
                len(trace),
                mean_len,
                wall,
            )

    return rows


def _save_figures(grid_results: dict) -> None:
    fig_dir = FIGURES_DIR / PHASE_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== Phase 3 Figures ===")
    _plot_convergence_curves(
        grid_results,
        "vi",
        fig_dir,
        "cartpole_vi_convergence.png",
        "Value Iteration Convergence — CartPole (by grid)",
    )
    _plot_convergence_curves(
        grid_results,
        "pi",
        fig_dir,
        "cartpole_pi_convergence.png",
        "Policy Iteration Convergence — CartPole (by grid)",
    )
    _plot_discretization_study(grid_results, fig_dir)
    _plot_policy_slice(grid_results, fig_dir)


# ── Main ──────────────────────────────────────────────────────────────────────


def run() -> None:
    metrics_dir = METRICS_DIR / PHASE_DIR
    metrics_dir.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== Phase 3: VI & PI on CartPole (grid ablation) ===")
    grid_results = {name: _run_grid(name) for name in GRID_NAMES}

    # ── CSVs ──────────────────────────────────────────────────────────────────
    vi_conv, pi_conv = [], []
    eval_per_seed, eval_aggregate = [], []
    policy_agreement_rows, study_rows = [], []

    for name, res in grid_results.items():
        for t in res["vi"]["trace"]:
            vi_conv.append({"grid": name, **t})
        for t in res["pi"]["trace"]:
            pi_conv.append({"grid": name, **t})

        for seed, val in res["vi"]["eval"]:
            eval_per_seed.append(
                {
                    "grid": name,
                    "algorithm": "VI",
                    "seed": seed,
                    "mean_episode_len": val,
                }
            )
        for seed, val in res["pi"]["eval"]:
            eval_per_seed.append(
                {
                    "grid": name,
                    "algorithm": "PI",
                    "seed": seed,
                    "mean_episode_len": val,
                }
            )

        for algo, alg_key in [("VI", "vi"), ("PI", "pi")]:
            eval_aggregate.append(
                {
                    "grid": name,
                    "algorithm": algo,
                    "mean_episode_len": res[alg_key]["mean_episode_len"],
                    "eval_episode_len_iqr": res[alg_key]["iqr"],
                }
            )
            study_rows.append(
                {
                    "grid": name,
                    "algorithm": algo,
                    "mean_episode_len": res[alg_key]["mean_episode_len"],
                    "eval_episode_len_iqr": res[alg_key]["iqr"],
                    "iterations_to_conv": res[alg_key]["iters"],
                    "wall_clock_s": res[alg_key]["wall_clock_s"],
                    "coverage_pct": round(res["coverage_pct"], 4),
                    "smoothed_pct": round(res["smoothed_pct"], 4),
                    "rollout_steps": res["rollout_steps"],
                }
            )

        policy_agreement_rows.append(
            {
                "grid": name,
                "action_agreement_pct": round(res["policy_agreement"] * 100, 2),
                "exact_match": res["policy_agreement"] == 1.0,
            }
        )

    pd.DataFrame(vi_conv).to_csv(metrics_dir / "vi_convergence.csv", index=False)
    pd.DataFrame(pi_conv).to_csv(metrics_dir / "pi_convergence.csv", index=False)
    pd.DataFrame(eval_per_seed).to_csv(
        metrics_dir / "policy_eval_per_seed.csv", index=False
    )
    pd.DataFrame(eval_aggregate).to_csv(
        metrics_dir / "policy_eval_aggregate.csv", index=False
    )
    pd.DataFrame(policy_agreement_rows).to_csv(
        metrics_dir / "policy_agreement.csv", index=False
    )
    pd.DataFrame(study_rows).to_csv(
        metrics_dir / "discretization_study.csv", index=False
    )
    logger.info("Metrics saved → %s", metrics_dir)

    # ── Hyperparameter validation sweep (default grid only) ───────────────────
    logger.info("=== VI/PI Hyperparameter Validation Sweep (default grid) ===")
    default_res = grid_results["default"]
    hp_rows = _hp_sweep_cartpole(
        default_res["T"], default_res["R"], default_res["disc"]
    )
    hp_df = pd.DataFrame(hp_rows)
    hp_df.to_csv(metrics_dir / "hp_validation.csv", index=False)
    logger.info("HP validation saved → %s", metrics_dir / "hp_validation.csv")

    # ── Figures ───────────────────────────────────────────────────────────────
    _save_figures(grid_results)

    # ── Checkpoint JSON ───────────────────────────────────────────────────────
    checkpoint = {
        name: {
            "bins": res["bins"],
            "n_states": res["n_states"],
            "rollout_steps": res["rollout_steps"],
            "coverage_pct": round(res["coverage_pct"], 4),
            "smoothed_pct": round(res["smoothed_pct"], 4),
            "vi": {
                "iterations": res["vi"]["iters"],
                "wall_clock_s": res["vi"]["wall_clock_s"],
                "mean_episode_len": round(res["vi"]["mean_episode_len"], 2),
                "eval_episode_len_iqr": round(res["vi"]["iqr"], 2),
            },
            "pi": {
                "iterations": res["pi"]["iters"],
                "wall_clock_s": res["pi"]["wall_clock_s"],
                "mean_episode_len": round(res["pi"]["mean_episode_len"], 2),
                "eval_episode_len_iqr": round(res["pi"]["iqr"], 2),
                "policy_changes_at_convergence": res["pi"]["trace"][-1][
                    "policy_changes"
                ],
                "stop_reason": res["pi"]["trace"][-1].get(
                    "stop_reason", "policy_stable"
                ),
            },
            "policy_agreement_pct": round(res["policy_agreement"] * 100, 2),
        }
        for name, res in grid_results.items()
    }
    hp_vi_gamma = hp_df[(hp_df.algorithm == "VI") & (hp_df.sweep_param == "gamma")]
    hp_pi_gamma = hp_df[(hp_df.algorithm == "PI") & (hp_df.sweep_param == "gamma")]
    checkpoint["hp_validation"] = {
        "grid": "default",
        "validated_hyperparameters": ["gamma", "delta"],
        "gamma_sweep_gammas": list(VI_PI_HP_GAMMA_VALUES),
        "delta_sweep_deltas": list(VI_PI_HP_DELTA_VALUES),
        "vi_gamma_episode_len_range": [
            round(float(hp_vi_gamma.mean_episode_len.min()), 2),
            round(float(hp_vi_gamma.mean_episode_len.max()), 2),
        ],
        "pi_gamma_episode_len_range": [
            round(float(hp_pi_gamma.mean_episode_len.min()), 2),
            round(float(hp_pi_gamma.mean_episode_len.max()), 2),
        ],
        "note": (
            "gamma materially impacts policy quality on CartPole (lower gamma → "
            "myopic policy → shorter episodes); delta affects convergence speed "
            "but not final policy quality for values ≤1e-3."
        ),
    }

    phase3_path = METADATA_DIR / "phase3.json"
    with open(phase3_path, "w") as f:
        json.dump(checkpoint, f, indent=2)
    logger.info("Phase 3 checkpoint saved → %s", phase3_path)

    logger.info("=== Phase 3 Summary ===")
    for name in GRID_NAMES:
        res = grid_results[name]
        logger.info(
            "Grid %-8s | VI: %3d iters, len=%.1f | PI: %3d iters, len=%.1f | "
            "coverage=%.1f%% | agreement=%.1f%%",
            name,
            res["vi"]["iters"],
            res["vi"]["mean_episode_len"],
            res["pi"]["iters"],
            res["pi"]["mean_episode_len"],
            res["coverage_pct"] * 100,
            res["policy_agreement"] * 100,
        )


if __name__ == "__main__":
    run()
