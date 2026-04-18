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
    VI_PI_HP_DELTA_VALUES,
    VI_PI_HP_GAMMA_VALUES,
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


def _plot_convergence_comparison(
    trace_vi: list[dict], trace_pi: list[dict], fig_dir: Path
) -> None:
    """Single figure: VI and PI convergence shown differently.

    Left (VI):  ΔV curve on log scale — stops when ΔV < δ.
    Right (PI): policy-changes bars on linear scale — stops when changes = 0.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    vi_iters = [t["iteration"] for t in trace_vi]
    vi_dvs = [t["delta_v"] for t in trace_vi]
    pi_iters = [t["iteration"] for t in trace_pi]
    pi_changes = [t["policy_changes"] for t in trace_pi]

    iter_max = max(vi_iters[-1], pi_iters[-1])

    fig, (ax_vi, ax_pi) = plt.subplots(1, 2, figsize=(12, 4))

    # ── Left: VI — value convergence ──────────────────────────────────────────
    ax_vi.semilogy(vi_iters, vi_dvs, color="#4C72B0", linewidth=1.5, label="max ΔV")
    ax_vi.axhline(
        VI_DELTA, color="red", linewidth=1, linestyle="--", label=f"δ = {VI_DELTA:.0e}"
    )
    ax_vi.set_xlim(0, iter_max + 1)
    ax_vi.set_xlabel("Iteration")
    ax_vi.set_ylabel("max |ΔV| (log scale)")
    ax_vi.set_title(
        f"Value Iteration  (γ={VI_GAMMA})\n"
        f"stops when max |ΔV| < δ — converged iter {vi_iters[-1]}"
    )
    ax_vi.legend(fontsize=9)
    ax_vi.grid(True, alpha=0.3)
    ax_vi.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ── Right: PI — policy stability ──────────────────────────────────────────
    color_bar = "#DD8452"
    color_stable = "#2ca02c"

    # Only plot non-zero bars — zero means stable, shown by the vline annotation instead
    nonzero = [(i, c) for i, c in zip(pi_iters, pi_changes) if c > 0]
    if nonzero:
        nz_iters, nz_changes = zip(*nonzero)
        ax_pi.bar(
            nz_iters, nz_changes, color=color_bar, alpha=0.75, label="Policy changes"
        )
    ax_pi.set_yscale("log")
    stable_iter = pi_iters[-1]
    ax_pi.axvline(
        stable_iter,
        color=color_stable,
        linewidth=1.2,
        linestyle=":",
        label=f"Policy stable (iter {stable_iter})",
    )
    ax_pi.annotate(
        "Optimal policy found",
        xy=(stable_iter, 0),
        xycoords=("data", "axes fraction"),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=8,
        color=color_stable,
        arrowprops=dict(arrowstyle="-", color=color_stable, lw=0.8),
    )
    ax_pi.set_xlim(0, iter_max + 1)
    ax_pi.set_xlabel("Iteration")
    ax_pi.set_ylabel("# policy changes (log scale)")
    ax_pi.set_title(
        f"Policy Iteration  (γ={PI_GAMMA})\n"
        f"stops when policy changes = 0 — stable iter {stable_iter}"
    )
    ax_pi.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_pi.legend(fontsize=9)
    ax_pi.grid(True, alpha=0.3)

    fig.suptitle("VI vs PI Convergence — Blackjack", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = fig_dir / "blackjack_convergence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out)


def _plot_policy_heatmap(
    hard_policy: np.ndarray, soft_policy: np.ndarray, fig_dir: Path
) -> None:
    """Crisp decision-region map: solid colors, sharp action boundaries, cell grid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
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

    STICK_COLOR = "#4C72B0"  # blue
    HIT_COLOR = "#DD8452"  # orange
    cmap = mcolors.ListedColormap([STICK_COLOR, HIT_COLOR])

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, grid, ylabels, title in [
        (axes[0], hard_policy, hard_labels, "Hard hands (no usable ace)"),
        (axes[1], soft_policy, soft_labels, "Soft hands / usable ace"),
    ]:
        n_rows, n_cols = grid.shape
        ax.imshow(
            grid,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=-0.5,
            vmax=1.5,
            interpolation="nearest",
        )
        # Draw sharp cell boundaries
        for x in range(n_cols + 1):
            ax.axvline(x - 0.5, color="white", linewidth=0.6, alpha=0.7)
        for y in range(n_rows + 1):
            ax.axhline(y - 0.5, color="white", linewidth=0.6, alpha=0.7)
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(dealer_labels, fontsize=9)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_xlabel("Dealer card", fontsize=10)
        ax.set_ylabel("Player hand", fontsize=10)
        ax.set_title(title, fontsize=10)

    legend_patches = [
        mpatches.Patch(color=STICK_COLOR, label="Stick"),
        mpatches.Patch(color=HIT_COLOR, label="Hit"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=2,
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle("VI Optimal Policy — Blackjack", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = fig_dir / "blackjack_vi_policy_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out)


def _plot_value_heatmap(hard_V: np.ndarray, soft_V: np.ndarray, fig_dir: Path) -> None:
    """3D value-function surface (Sutton-and-Barto style): dealer × player → V(s)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dealer_ticks = np.arange(1, 11)  # 1..10 → dealer cards 2..A
    hard_ticks = np.arange(4, 22)  # hard hand player sums
    soft_ticks = np.arange(12, 23)  # soft hand player sums (S12–S21 + BJ≈22)

    dealer_grid_hard, hand_grid_hard = np.meshgrid(dealer_ticks, hard_ticks)
    dealer_grid_soft, hand_grid_soft = np.meshgrid(dealer_ticks, soft_ticks)

    vmin = float(min(np.nanmin(hard_V), np.nanmin(soft_V)))
    vmax = float(max(np.nanmax(hard_V), np.nanmax(soft_V)))

    fig = plt.figure(figsize=(14, 5))
    for idx, (Z, dealer_grid, hand_grid, hand_ticks, title) in enumerate(
        [
            (hard_V, dealer_grid_hard, hand_grid_hard, hard_ticks, "No usable ace"),
            (soft_V, dealer_grid_soft, hand_grid_soft, soft_ticks, "Usable ace"),
        ],
        start=1,
    ):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        surf = ax.plot_surface(
            dealer_grid,
            hand_grid,
            Z,
            cmap="RdYlGn",
            vmin=vmin,
            vmax=vmax,
            edgecolor="none",
            alpha=0.92,
        )
        ax.set_xlabel("Dealer card", fontsize=8, labelpad=4)
        ax.set_ylabel("Player sum", fontsize=8, labelpad=4)
        ax.set_zlabel("V(s)", fontsize=8, labelpad=2)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(dealer_ticks[::2])
        ax.set_xticklabels(["2", "4", "6", "8", "10"], fontsize=7)
        ax.set_yticks(hand_ticks[::3])
        ax.set_yticklabels([str(t) for t in hand_ticks[::3]], fontsize=7)
        ax.tick_params(axis="z", labelsize=7)
        ax.view_init(elev=28, azim=-55)

    fig.colorbar(surf, ax=fig.axes, label="V(s)", shrink=0.5, aspect=12, pad=0.05)
    fig.suptitle("VI Value Function — Blackjack", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = fig_dir / "blackjack_vi_value_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
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
    _plot_convergence_comparison(trace_vi, trace_pi, fig_dir)
    _plot_policy_heatmap(hard_policy, soft_policy, fig_dir)
    _plot_value_heatmap(hard_V, soft_V, fig_dir)


# ── Hyperparameter validation ─────────────────────────────────────────────────

HP_EVAL_EPISODES = 500  # lighter than full eval to keep sweep fast


def _hp_sweep_blackjack(T: np.ndarray, R: np.ndarray) -> list[dict]:
    """Sweep gamma and delta for VI and PI on Blackjack.

    Gamma sweep: vary gamma in VI_PI_HP_GAMMA_VALUES, hold delta at reference.
    Delta sweep: vary delta in VI_PI_HP_DELTA_VALUES, hold gamma at 0.99.

    Returns list of row dicts for hp_validation.csv.
    """
    rows: list[dict] = []

    def _eval(policy):
        results = eval_blackjack_policy(
            policy, seeds=SEEDS, n_episodes=HP_EVAL_EPISODES
        )
        vals = [r for _, r in results]
        return float(np.mean(vals)), float(
            np.percentile(vals, 75) - np.percentile(vals, 25)
        )

    # ── gamma sweep (delta fixed at reference) ────────────────────────────────
    for gamma in VI_PI_HP_GAMMA_VALUES:
        for algo_label, run_fn, ref_delta in [
            ("VI", run_vi, VI_DELTA),
            ("PI", run_pi, PI_DELTA),
        ]:
            t0 = time.perf_counter()
            if algo_label == "VI":
                _, policy, trace = run_fn(
                    T, R, gamma, ref_delta, m_consec=VI_PI_CONSEC_SWEEPS
                )
            else:
                _, policy, trace = run_fn(T, R, gamma, ref_delta)
            wall = time.perf_counter() - t0
            mean_ret, iqr = _eval(policy)
            rows.append(
                {
                    "algorithm": algo_label,
                    "sweep_param": "gamma",
                    "gamma": gamma,
                    "delta": ref_delta,
                    "iterations": len(trace),
                    "wall_clock_s": round(wall, 3),
                    "mean_eval_return": round(mean_ret, 4),
                    "eval_return_iqr": round(iqr, 4),
                }
            )
            logger.info(
                "HP gamma sweep [%s] gamma=%.2f delta=%.0e → %d iters, "
                "mean_return=%.4f, %.3fs",
                algo_label,
                gamma,
                ref_delta,
                len(trace),
                mean_ret,
                wall,
            )

    # ── delta sweep (gamma fixed at 0.99) ─────────────────────────────────────
    ref_gamma = 0.99
    for delta in VI_PI_HP_DELTA_VALUES:
        for algo_label, run_fn in [("VI", run_vi), ("PI", run_pi)]:
            t0 = time.perf_counter()
            if algo_label == "VI":
                _, policy, trace = run_fn(
                    T, R, ref_gamma, delta, m_consec=VI_PI_CONSEC_SWEEPS
                )
            else:
                _, policy, trace = run_fn(T, R, ref_gamma, delta)
            wall = time.perf_counter() - t0
            mean_ret, iqr = _eval(policy)
            rows.append(
                {
                    "algorithm": algo_label,
                    "sweep_param": "delta",
                    "gamma": ref_gamma,
                    "delta": delta,
                    "iterations": len(trace),
                    "wall_clock_s": round(wall, 3),
                    "mean_eval_return": round(mean_ret, 4),
                    "eval_return_iqr": round(iqr, 4),
                }
            )
            logger.info(
                "HP delta sweep [%s] gamma=%.2f delta=%.0e → %d iters, "
                "mean_return=%.4f, %.3fs",
                algo_label,
                ref_gamma,
                delta,
                len(trace),
                mean_ret,
                wall,
            )

    return rows


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

    # ── Hyperparameter validation sweep ──────────────────────────────────────
    logger.info("=== VI/PI Hyperparameter Validation Sweep ===")
    hp_rows = _hp_sweep_blackjack(T, R)
    hp_df = pd.DataFrame(hp_rows)
    hp_df.to_csv(metrics_dir / "hp_validation.csv", index=False)
    logger.info("HP validation saved → %s", metrics_dir / "hp_validation.csv")

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
            "stop_reason": trace_pi[-1].get("stop_reason", "policy_stable"),
            "mean_eval_return": round(pi_mean, 4),
            "eval_return_iqr": round(pi_iqr, 4),
            "policy_match_vi": round(agreement, 4),
        },
    }
    # Summarise HP sweep: list validated hyperparameters and sensitivity verdict
    hp_vi_gamma = hp_df[(hp_df.algorithm == "VI") & (hp_df.sweep_param == "gamma")]
    hp_pi_gamma = hp_df[(hp_df.algorithm == "PI") & (hp_df.sweep_param == "gamma")]
    checkpoint["hp_validation"] = {
        "validated_hyperparameters": ["gamma", "delta"],
        "gamma_sweep_gammas": list(VI_PI_HP_GAMMA_VALUES),
        "delta_sweep_deltas": list(VI_PI_HP_DELTA_VALUES),
        "vi_gamma_return_range": [
            round(float(hp_vi_gamma.mean_eval_return.min()), 4),
            round(float(hp_vi_gamma.mean_eval_return.max()), 4),
        ],
        "pi_gamma_return_range": [
            round(float(hp_pi_gamma.mean_eval_return.min()), 4),
            round(float(hp_pi_gamma.mean_eval_return.max()), 4),
        ],
        "note": (
            "gamma materially impacts policy quality (lower gamma → shorter horizon → "
            "worse expected return); delta affects convergence speed but not final policy "
            "for values ≤1e-3 on this MDP."
        ),
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
