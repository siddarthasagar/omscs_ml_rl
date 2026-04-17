"""Phase 2 — Model-Based: VI & PI on Blackjack.

Outputs:
  artifacts/metrics/phase2_vi_pi_blackjack/
    vi_convergence.csv, pi_convergence.csv
    policy_eval_per_seed.csv, policy_eval_aggregate.csv, summary.csv
  artifacts/figures/phase2_vi_pi_blackjack/
    blackjack_vi_convergence.png, blackjack_pi_convergence.png
    blackjack_vi_policy_heatmap.png, blackjack_vi_value_heatmap.png
  artifacts/metadata/phase2.json
  artifacts/logs/phase2.log

Usage:
  uv run python scripts/run_phase_2_vi_pi_blackjack.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.algorithms import eval_blackjack_policy, run_pi, run_vi
from src.config import (
    FIGURES_DIR,
    METADATA_DIR,
    METRICS_DIR,
    PI_DELTA,
    PI_GAMMA,
    SEEDS,
    VI_DELTA,
    VI_GAMMA,
    VI_PI_CONSEC_SWEEPS,
)
from src.envs.blackjack_env import get_blackjack_model
from src.utils.logger import configure_logger

logger = configure_logger("phase2")

N_EVAL_EPISODES = 1000
PHASE_DIR = "phase2_vi_pi_blackjack"


# ── Blackjack state decoding ──────────────────────────────────────────────────


def _decode_bj_grids(policy: np.ndarray, V: np.ndarray, n_states: int):
    """Decode bettermdptools Blackjack states into 2-D arrays for heatmaps.

    State encoding (from bettermdptools BlackjackWrapper):
      state = hand_idx * 10 + dealer_idx
      hand_idx 0-17 → hard hands H4-H21 (player_sum = hand_idx + 4)
      hand_idx 18-26 → soft hands S12-S20 (player_sum = hand_idx - 6)
      hand_idx 27    → S21
      hand_idx 28    → BJ
      dealer_idx 0-9 → dealer cards 2,3,...,9,T,A

    Returns:
        hard_policy (18, 10), soft_policy (11, 10),
        hard_V (18, 10),      soft_V (11, 10)
    """
    hard_policy = np.full((18, 10), np.nan)
    soft_policy = np.full((11, 10), np.nan)
    hard_V = np.full((18, 10), np.nan)
    soft_V = np.full((11, 10), np.nan)

    for state in range(n_states):
        hand_idx = state // 10
        dealer_idx = state % 10
        act = int(policy[state])
        v = float(V[state])
        if hand_idx <= 17:
            hard_policy[hand_idx, dealer_idx] = act
            hard_V[hand_idx, dealer_idx] = v
        else:
            soft_idx = hand_idx - 18
            if 0 <= soft_idx <= 10:
                soft_policy[soft_idx, dealer_idx] = act
                soft_V[soft_idx, dealer_idx] = v

    return hard_policy, soft_policy, hard_V, soft_V


# ── Figure helpers ────────────────────────────────────────────────────────────


def _plot_vi_convergence(trace: list[dict], fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iters = [t["iteration"] for t in trace]
    dvs = [t["delta_v"] for t in trace]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(iters, dvs, color="#4C72B0", linewidth=1.5)
    ax.axhline(
        VI_DELTA, color="red", linewidth=1, linestyle="--", label=f"δ = {VI_DELTA:.0e}"
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("max ΔV (log scale)")
    ax.set_title(
        f"Value Iteration Convergence — Blackjack\n"
        f"γ={VI_GAMMA}, δ={VI_DELTA:.0e}, converged at iter {iters[-1]}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = fig_dir / "blackjack_vi_convergence.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out)


def _plot_pi_convergence(trace: list[dict], fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iters = [t["iteration"] for t in trace]
    dvs = [t["delta_v"] for t in trace]
    changes = [t["policy_changes"] for t in trace]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.semilogy(iters, dvs, color="#4C72B0", linewidth=1.5, label="max ΔV")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("max ΔV (log scale)", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")

    ax2 = ax1.twinx()
    ax2.bar(iters, changes, color="#DD8452", alpha=0.6, label="Policy changes")
    ax2.set_ylabel("Policy changes", color="#DD8452")
    ax2.tick_params(axis="y", labelcolor="#DD8452")

    ax1.set_title(
        f"Policy Iteration Convergence — Blackjack\n"
        f"γ={PI_GAMMA}, δ={PI_DELTA:.0e}, converged at iter {iters[-1]}"
    )
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.tight_layout()
    out = fig_dir / "blackjack_pi_convergence.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out)


def _plot_policy_heatmap(
    hard_policy: np.ndarray, soft_policy: np.ndarray, fig_dir: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    dealer_labels = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"]
    hard_labels = [str(i) for i in range(4, 22)]
    soft_labels = [
        "S12",
        "S13",
        "S14",
        "S15",
        "S16",
        "S17",
        "S18",
        "S19",
        "S20",
        "S21",
        "BJ",
    ]
    cmap = mcolors.ListedColormap(["#4C72B0", "#DD8452"])  # 0=stick, 1=hit

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, grid, ylabels, title in [
        (axes[0], hard_policy, hard_labels, "Hard hands (no usable ace)"),
        (axes[1], soft_policy, soft_labels, "Soft hands / BJ (usable ace)"),
    ]:
        im = ax.imshow(
            grid, aspect="auto", origin="lower", cmap=cmap, vmin=-0.5, vmax=1.5
        )
        ax.set_xticks(range(10))
        ax.set_xticklabels(dealer_labels)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_xlabel("Dealer card")
        ax.set_ylabel("Player hand")
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=axes.tolist(), ticks=[0, 1], fraction=0.02, pad=0.04)
    cbar.set_ticklabels(["Stick (0)", "Hit (1)"])
    fig.suptitle("VI Optimal Policy — Blackjack", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.93, 0.96])
    out = fig_dir / "blackjack_vi_policy_heatmap.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out)


def _plot_value_heatmap(hard_V: np.ndarray, soft_V: np.ndarray, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dealer_labels = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"]
    hard_labels = [str(i) for i in range(4, 22)]
    soft_labels = [
        "S12",
        "S13",
        "S14",
        "S15",
        "S16",
        "S17",
        "S18",
        "S19",
        "S20",
        "S21",
        "BJ",
    ]

    vmin = float(min(np.nanmin(hard_V), np.nanmin(soft_V)))
    vmax = float(max(np.nanmax(hard_V), np.nanmax(soft_V)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, grid, ylabels, title in [
        (axes[0], hard_V, hard_labels, "Hard hands (no usable ace)"),
        (axes[1], soft_V, soft_labels, "Soft hands / BJ (usable ace)"),
    ]:
        im = ax.imshow(
            grid, aspect="auto", origin="lower", cmap="RdYlGn", vmin=vmin, vmax=vmax
        )
        ax.set_xticks(range(10))
        ax.set_xticklabels(dealer_labels)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_xlabel("Dealer card")
        ax.set_ylabel("Player hand")
        ax.set_title(title)

    fig.colorbar(im, ax=axes.tolist(), label="V(s)", fraction=0.02, pad=0.04)
    fig.suptitle("VI Value Function — Blackjack", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.93, 0.96])
    out = fig_dir / "blackjack_vi_value_heatmap.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out)


def _save_figures(
    trace_vi: list[dict],
    trace_pi: list[dict],
    V_vi: np.ndarray,
    policy_vi: np.ndarray,
    n_states: int,
) -> None:
    fig_dir = FIGURES_DIR / PHASE_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== Phase 2 Figures ===")

    hard_policy, soft_policy, hard_V, soft_V = _decode_bj_grids(
        policy_vi, V_vi, n_states
    )
    _plot_vi_convergence(trace_vi, fig_dir)
    _plot_pi_convergence(trace_pi, fig_dir)
    _plot_policy_heatmap(hard_policy, soft_policy, fig_dir)
    _plot_value_heatmap(hard_V, soft_V, fig_dir)


# ── Main ──────────────────────────────────────────────────────────────────────


def run() -> None:
    metrics_dir = METRICS_DIR / PHASE_DIR
    metrics_dir.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info("=== Loading Blackjack T/R model ===")
    T, R, n_states, n_actions = get_blackjack_model()
    logger.info("Blackjack: %d states, %d actions", n_states, n_actions)

    # ── Value Iteration ───────────────────────────────────────────────────────
    logger.info("=== Running Value Iteration ===")
    t0 = time.perf_counter()
    V_vi, policy_vi, trace_vi = run_vi(
        T, R, VI_GAMMA, VI_DELTA, m_consec=VI_PI_CONSEC_SWEEPS, logger=logger
    )
    vi_wall = time.perf_counter() - t0
    vi_iters = len(trace_vi)
    vi_final_dv = trace_vi[-1]["delta_v"]
    logger.info("VI: %d iters, ΔV=%.2e, wall=%.2fs", vi_iters, vi_final_dv, vi_wall)

    # ── Policy Iteration ──────────────────────────────────────────────────────
    logger.info("=== Running Policy Iteration ===")
    t0 = time.perf_counter()
    V_pi, policy_pi, trace_pi = run_pi(
        T, R, PI_GAMMA, PI_DELTA, m_consec=VI_PI_CONSEC_SWEEPS, logger=logger
    )
    pi_wall = time.perf_counter() - t0
    pi_iters = len(trace_pi)
    pi_final_dv = trace_pi[-1]["delta_v"]
    logger.info("PI: %d iters, ΔV=%.2e, wall=%.2fs", pi_iters, pi_final_dv, pi_wall)

    agreement = float((policy_vi[:n_states] == policy_pi[:n_states]).mean())
    logger.info("VI vs PI policy agreement: %.1f%%", agreement * 100)

    # ── Policy evaluation ─────────────────────────────────────────────────────
    logger.info("=== Evaluating policies (5 seeds × %d episodes) ===", N_EVAL_EPISODES)
    vi_eval = eval_blackjack_policy(policy_vi, seeds=SEEDS, n_episodes=N_EVAL_EPISODES)
    vi_returns = [r for _, r in vi_eval]
    vi_mean = float(np.mean(vi_returns))
    vi_iqr = float(np.percentile(vi_returns, 75) - np.percentile(vi_returns, 25))
    logger.info("VI  eval: mean_return=%.4f, IQR=%.4f", vi_mean, vi_iqr)

    pi_eval = eval_blackjack_policy(policy_pi, seeds=SEEDS, n_episodes=N_EVAL_EPISODES)
    pi_returns = [r for _, r in pi_eval]
    pi_mean = float(np.mean(pi_returns))
    pi_iqr = float(np.percentile(pi_returns, 75) - np.percentile(pi_returns, 25))
    logger.info("PI  eval: mean_return=%.4f, IQR=%.4f", pi_mean, pi_iqr)

    # ── CSVs ──────────────────────────────────────────────────────────────────
    pd.DataFrame(trace_vi).to_csv(metrics_dir / "vi_convergence.csv", index=False)
    pd.DataFrame(trace_pi).to_csv(metrics_dir / "pi_convergence.csv", index=False)

    pd.DataFrame(
        [{"algorithm": "VI", "seed": s, "mean_return": r} for s, r in vi_eval]
        + [{"algorithm": "PI", "seed": s, "mean_return": r} for s, r in pi_eval]
    ).to_csv(metrics_dir / "policy_eval_per_seed.csv", index=False)

    pd.DataFrame(
        [
            {"algorithm": "VI", "mean_return": vi_mean, "eval_return_iqr": vi_iqr},
            {"algorithm": "PI", "mean_return": pi_mean, "eval_return_iqr": pi_iqr},
        ]
    ).to_csv(metrics_dir / "policy_eval_aggregate.csv", index=False)

    pd.DataFrame(
        [
            {
                "algorithm": "VI",
                "iterations_to_convergence": vi_iters,
                "wall_clock_s": round(vi_wall, 3),
                "mean_eval_return": vi_mean,
                "eval_return_iqr": vi_iqr,
                "policy_match_vi": 1.0,
            },
            {
                "algorithm": "PI",
                "iterations_to_convergence": pi_iters,
                "wall_clock_s": round(pi_wall, 3),
                "mean_eval_return": pi_mean,
                "eval_return_iqr": pi_iqr,
                "policy_match_vi": round(agreement, 4),
            },
        ]
    ).to_csv(metrics_dir / "summary.csv", index=False)
    logger.info("Metrics saved → %s", metrics_dir)

    # ── Figures ───────────────────────────────────────────────────────────────
    _save_figures(trace_vi, trace_pi, V_vi, policy_vi, n_states)

    # ── Checkpoint JSON ───────────────────────────────────────────────────────
    checkpoint = {
        "vi": {
            "iterations": vi_iters,
            "wall_clock_s": round(vi_wall, 3),
            "final_delta_v": round(vi_final_dv, 10),
            "mean_eval_return": round(vi_mean, 4),
            "eval_return_iqr": round(vi_iqr, 4),
        },
        "pi": {
            "iterations": pi_iters,
            "wall_clock_s": round(pi_wall, 3),
            "final_delta_v": round(pi_final_dv, 10),
            "policy_changes_at_convergence": trace_pi[-1]["policy_changes"],
            "mean_eval_return": round(pi_mean, 4),
            "eval_return_iqr": round(pi_iqr, 4),
            "policy_match_vi": round(agreement, 4),
        },
    }
    phase2_path = METADATA_DIR / "phase2.json"
    with open(phase2_path, "w") as f:
        json.dump(checkpoint, f, indent=2)
    logger.info("Phase 2 checkpoint saved → %s", phase2_path)

    logger.info("=== Phase 2 Summary ===")
    logger.info("VI: %d iters, %.3fs, mean_return=%.4f", vi_iters, vi_wall, vi_mean)
    logger.info("PI: %d iters, %.3fs, mean_return=%.4f", pi_iters, pi_wall, pi_mean)
    logger.info("Policy agreement VI vs PI: %.1f%%", agreement * 100)


if __name__ == "__main__":
    run()
