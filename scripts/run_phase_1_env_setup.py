"""Phase 1 — Environment Setup, MDP Characterization & CartPole Model.

Outputs:
  artifacts/metadata/phase1.json                    — checkpoint for human review
  artifacts/metadata/cartpole_model.npz             — T, R, absorbing_state_index
  artifacts/figures/phase1/bin_edges.png            — bin edge diagram (4 dimensions)
  artifacts/figures/phase1/coverage_heatmap.png     — θ × θ̇ coverage heatmap
  artifacts/figures/phase1/visit_histogram.png      — visit count distribution
  artifacts/logs/phase1.log

Usage:
  uv run python scripts/run_phase_1_env_setup.py
"""

import json
import time
from pathlib import Path

import numpy as np

from src.config import (
    CARTPOLE_BINS,
    CARTPOLE_CLAMPS,
    CARTPOLE_MODEL_MIN_VISITS,
    CARTPOLE_MODEL_ROLLOUT_STEPS,
    CARTPOLE_MODEL_SEED,
    CARTPOLE_THETA_EDGES,
    CARTPOLE_THETADOT_EDGES,
    CARTPOLE_X_EDGES,
    CARTPOLE_XDOT_EDGES,
    FIGURES_DIR,
    METADATA_DIR,
)
from src.envs.blackjack_env import get_blackjack_model
from src.envs.cartpole_discretizer import CartPoleDiscretizer
from src.envs.cartpole_model import build_cartpole_model
from src.utils.logger import configure_logger

logger = configure_logger("phase1")


# ── Figure helpers ────────────────────────────────────────────────────────────


def _plot_bin_edges(disc, fig_dir: Path) -> None:
    """4-panel diagram showing bin boundaries for each CartPole dimension."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dim_labels = [
        "x — cart position (m)",
        "ẋ — cart velocity (m/s)",
        "θ — pole angle (rad)",
        "θ̇ — angular velocity (rad/s)",
    ]
    is_uniform = [True, True, False, False]
    colors_even = ["#4C72B0", "#4C72B0", "#DD8452", "#DD8452"]
    colors_odd = ["#A8C4E0", "#A8C4E0", "#F5C89A", "#F5C89A"]

    fig, axes = plt.subplots(4, 1, figsize=(10, 5))

    for i, ax in enumerate(axes):
        edges = disc._edges[i]
        for j in range(len(edges) - 1):
            lo, hi = edges[j], edges[j + 1]
            color = colors_even[i] if j % 2 == 0 else colors_odd[i]
            ax.axvspan(lo, hi, alpha=0.85, color=color)

        for e in edges:
            ax.axvline(e, color="black", linewidth=0.8)

        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks(edges)
        ax.set_xticklabels([f"{e:.3f}" for e in edges], fontsize=6, rotation=45)
        spacing = "uniform" if is_uniform[i] else "non-uniform (finer near 0)"
        n_bins = len(edges) - 1
        ax.set_title(
            f"{dim_labels[i]}  —  {n_bins} bins, {spacing}", fontsize=8, loc="left"
        )

    fig.suptitle("CartPole Discretization: Bin Edges", fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = fig_dir / "bin_edges.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out)


def _plot_coverage_heatmap(
    disc, visit_counts_sa: np.ndarray, min_visits: int, fig_dir: Path
) -> None:
    """θ × θ̇ coverage heatmap, marginalized over x and ẋ."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = disc._bins  # (3, 3, 8, 12)
    # Reshape to (x, xdot, theta, thetadot, actions)
    vc = visit_counts_sa.reshape(bins[0], bins[1], bins[2], bins[3], 2)

    # Coverage fraction: for each (theta, thetadot), what fraction of
    # (x_bin, xdot_bin, action) triples has >= min_visits?
    covered = (vc >= min_visits).astype(float)  # (3,3,8,12,2)
    cov_frac = covered.sum(axis=(0, 1, 4)) / (bins[0] * bins[1] * 2)  # (8,12)

    # Mean log10 visits for visited cells (for second panel)
    with np.errstate(divide="ignore"):
        log_visits = np.where(vc > 0, np.log10(vc.astype(float) + 1), 0.0)
    mean_log = log_visits.mean(axis=(0, 1, 4))  # (8,12)

    theta_centers = (np.array(disc._edges[2])[:-1] + np.array(disc._edges[2])[1:]) / 2
    thetadot_centers = (
        np.array(disc._edges[3])[:-1] + np.array(disc._edges[3])[1:]
    ) / 2

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1 — binary coverage fraction
    im1 = axes[0].imshow(
        cov_frac, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="Blues"
    )
    plt.colorbar(im1, ax=axes[0], label=f"Fraction covered (≥{min_visits} visits)")
    axes[0].set_title(
        f"Coverage fraction (≥{min_visits} visits)\nmarginalized over x, ẋ, actions"
    )
    axes[0].set_xlabel("θ̇ bin (rad/s)")
    axes[0].set_ylabel("θ bin (rad)")
    axes[0].set_xticks(range(bins[3]))
    axes[0].set_xticklabels(
        [f"{c:.2f}" for c in thetadot_centers], fontsize=6, rotation=45
    )
    axes[0].set_yticks(range(bins[2]))
    axes[0].set_yticklabels([f"{c:.4f}" for c in theta_centers], fontsize=7)

    # Panel 2 — mean log10 visit count
    im2 = axes[1].imshow(mean_log, aspect="auto", origin="lower", cmap="YlOrRd")
    plt.colorbar(im2, ax=axes[1], label="Mean log₁₀(visits+1)")
    axes[1].set_title("Mean log₁₀(visits+1)\nmarginalized over x, ẋ, actions")
    axes[1].set_xlabel("θ̇ bin (rad/s)")
    axes[1].set_ylabel("θ bin (rad)")
    axes[1].set_xticks(range(bins[3]))
    axes[1].set_xticklabels(
        [f"{c:.2f}" for c in thetadot_centers], fontsize=6, rotation=45
    )
    axes[1].set_yticks(range(bins[2]))
    axes[1].set_yticklabels([f"{c:.4f}" for c in theta_centers], fontsize=7)

    fig.suptitle(
        "CartPole Model: State Space Coverage (θ × θ̇)", fontsize=10, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = fig_dir / "coverage_heatmap.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out)


def _plot_visit_histogram(
    visit_counts_sa: np.ndarray, min_visits: int, fig_dir: Path
) -> None:
    """Distribution of visit counts for visited (s, a) pairs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flat = visit_counts_sa.flatten()
    total_sa = len(flat)
    visited = flat[flat > 0]
    covered = flat[flat >= min_visits]

    fig, ax = plt.subplots(figsize=(8, 4))

    n_bins = min(60, int(visited.max()) if len(visited) > 0 else 10)
    ax.hist(
        visited,
        bins=n_bins,
        color="#4C72B0",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(
        min_visits,
        color="#D62728",
        linewidth=2,
        linestyle="--",
        label=f"min_visits={min_visits} (coverage threshold)",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Visit count per (s, a) pair")
    ax.set_ylabel("Number of (s, a) pairs (log scale)")
    ax.set_title(
        f"CartPole Model: Visit Count Distribution\n"
        f"{len(visited):,}/{total_sa:,} (s,a) pairs visited  |  "
        f"{len(covered):,}/{total_sa:,} covered (≥{min_visits})"
    )
    ax.legend()
    plt.tight_layout()
    out = fig_dir / "visit_histogram.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out)


def _save_figures(disc, cp_model: dict) -> None:
    fig_dir = FIGURES_DIR / "phase1"
    fig_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== Phase 1 Figures ===")
    _plot_bin_edges(disc, fig_dir)
    _plot_coverage_heatmap(
        disc, cp_model["visit_counts"], CARTPOLE_MODEL_MIN_VISITS, fig_dir
    )
    _plot_visit_histogram(cp_model["visit_counts"], CARTPOLE_MODEL_MIN_VISITS, fig_dir)


def run() -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Blackjack ─────────────────────────────────────────────────────────────
    logger.info("=== Blackjack-v1 model ===")
    t0 = time.perf_counter()
    T_bj, R_bj, n_states_bj, n_actions_bj = get_blackjack_model()
    bj_time = time.perf_counter() - t0

    logger.info(
        "Blackjack: n_states=%d, n_actions=%d (%.2fs)",
        n_states_bj,
        n_actions_bj,
        bj_time,
    )

    # Sanity checks
    row_sums = T_bj.sum(axis=2)
    max_dev = float(np.abs(row_sums - 1.0).max())
    logger.info("Blackjack T row-stochasticity max deviation: %.2e", max_dev)
    logger.info("Blackjack R range: [%.3f, %.3f]", float(R_bj.min()), float(R_bj.max()))

    # ── CartPole discretizer ──────────────────────────────────────────────────
    logger.info("=== CartPole-v1 discretizer ===")
    disc = CartPoleDiscretizer()
    logger.info("CartPole bins: %s → n_states=%d", list(CARTPOLE_BINS), disc.n_states)

    # ── CartPole T/R model ────────────────────────────────────────────────────
    logger.info("=== CartPole-v1 T/R model estimation ===")
    t0 = time.perf_counter()
    cp_model = build_cartpole_model(
        discretizer=disc,
        rollout_steps=CARTPOLE_MODEL_ROLLOUT_STEPS,
        min_visits=CARTPOLE_MODEL_MIN_VISITS,
        seed=CARTPOLE_MODEL_SEED,
        logger=logger,
    )
    cp_time = time.perf_counter() - t0
    logger.info("CartPole model built in %.1fs", cp_time)

    # Save model arrays (include seed so the exact model can be recreated)
    model_path = METADATA_DIR / "cartpole_model.npz"
    np.savez_compressed(
        model_path,
        T=cp_model["T"],
        R=cp_model["R"],
        absorbing_state_index=np.array(cp_model["absorbing_state_index"]),
        model_seed=np.array(CARTPOLE_MODEL_SEED),
    )
    logger.info("CartPole model saved → %s", model_path)

    # ── Diagnostic figures ────────────────────────────────────────────────────
    _save_figures(disc, cp_model)

    # ── Write phase1.json ─────────────────────────────────────────────────────
    checkpoint = {
        "blackjack": {
            "n_states": n_states_bj,
            "n_actions": n_actions_bj,
            "model_source": "bettermdptools_analytic",
            "row_stochasticity_max_dev": max_dev,
            "reward_min": float(R_bj.min()),
            "reward_max": float(R_bj.max()),
            "wall_clock_s": round(bj_time, 4),
        },
        "cartpole": {
            "bins": list(CARTPOLE_BINS),
            "n_states": disc.n_states,
            "clamps": {k: list(v) for k, v in CARTPOLE_CLAMPS.items()},
            "bin_edges": {
                "x": CARTPOLE_X_EDGES,
                "xdot": CARTPOLE_XDOT_EDGES,
                "theta": CARTPOLE_THETA_EDGES,
                "thetadot": CARTPOLE_THETADOT_EDGES,
            },
            "model_rollout_steps": CARTPOLE_MODEL_ROLLOUT_STEPS,
            "model_seed": CARTPOLE_MODEL_SEED,
            "model_source": "rollout_estimation_laplace_smoothed",
            "coverage_pct": round(cp_model["coverage_pct"], 4),
            "smoothed_pct": round(cp_model["smoothed_pct"], 4),
            "mean_visits_covered": round(cp_model["mean_visits_covered"], 2),
            "min_visits_threshold": CARTPOLE_MODEL_MIN_VISITS,
            "absorbing_state_index": cp_model["absorbing_state_index"],
            "wall_clock_s": round(cp_time, 2),
        },
    }

    phase1_path = METADATA_DIR / "phase1.json"
    with open(phase1_path, "w") as f:
        json.dump(checkpoint, f, indent=2)
    logger.info("Phase 1 checkpoint saved → %s", phase1_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=== Phase 1 Summary ===")
    logger.info("Blackjack: %d states, %d actions", n_states_bj, n_actions_bj)
    logger.info(
        "CartPole: %d states (%s), coverage=%.1f%%, smoothed=%.1f%%",
        disc.n_states,
        "×".join(str(b) for b in CARTPOLE_BINS),
        cp_model["coverage_pct"] * 100,
        cp_model["smoothed_pct"] * 100,
    )
    if cp_model["coverage_pct"] < 0.80:
        logger.warning(
            "CartPole coverage %.1f%% is below the 80%% target — "
            "consider increasing CARTPOLE_MODEL_ROLLOUT_STEPS",
            cp_model["coverage_pct"] * 100,
        )
    else:
        logger.info("CartPole coverage target (>=80%%) met.")


if __name__ == "__main__":
    run()
