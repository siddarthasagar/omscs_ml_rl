"""Centralized plotting layer for RL experiment phases.

Style-driven constants live here so figures stay consistent across phases.
Plot functions read from saved CSV/JSON artifacts where possible so figures
can be regenerated without re-running experiments.

No bare plt.show() calls — all figures are saved to artifacts/figures/.
"""

from pathlib import Path

import numpy as np

# ── G. Style Defaults (display conventions, not scientific choices) ───────────

DEFAULT_DPI: int = 150

# Algorithm color palette — fixed so VI/PI look the same across all figures.
ALGO_COLORS: dict[str, str] = {
    "VI": "#4C72B0",  # blue
    "PI": "#DD8452",  # orange
}

# Blackjack action colors — fixed semantic mapping across policy figures.
BJ_ACTION_COLORS: dict[str, str] = {
    "Stick": "#4C72B0",  # blue
    "Hit": "#DD8452",  # orange
}

# CartPole terminal-angle marker — environment-defined constant (±12°).
CARTPOLE_TERMINAL_ANGLE_RAD: float = 12 * np.pi / 180  # ≈ 0.2094 rad

# Convergence annotation colors
COLOR_DELTA_LINE: str = "red"
COLOR_STABLE_LINE: str = "#2ca02c"  # green


# ── Blackjack domain helper ───────────────────────────────────────────────────


def decode_bj_grids(
    policy: np.ndarray, V: np.ndarray, n_states: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode bettermdptools Blackjack states into 2-D grids for heatmaps.

    State encoding (from bettermdptools BlackjackWrapper):
      state = hand_idx * 10 + dealer_idx
      hand_idx 0-17  → hard hands H4–H21
      hand_idx 18-28 → soft hands S12–S21, BJ
      dealer_idx 0-9 → dealer cards 2,3,...,9,T,A

    Returns:
        hard_policy (18, 10), soft_policy (11, 10),
        hard_V      (18, 10), soft_V      (11, 10)
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


# ── Phase 2: Blackjack figures ────────────────────────────────────────────────


def plot_bj_convergence(metrics_dir: Path, metadata: dict, fig_dir: Path) -> Path:
    """VI vs PI convergence figure read from saved CSVs + metadata.

    Left  (VI): ΔV on log scale with δ threshold — stops when ΔV < δ.
    Right (PI): policy-changes bars on log scale  — stops when changes = 0.

    Args:
        metrics_dir: directory containing vi_convergence.csv + pi_convergence.csv
        metadata:    phase2 metadata dict (must include vi.gamma, vi.delta,
                     vi.convergence_iter, pi.gamma, pi.stable_iter)
        fig_dir:     output directory
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.ticker import MaxNLocator

    trace_vi = pd.read_csv(metrics_dir / "vi_convergence.csv").to_dict("records")
    trace_pi = pd.read_csv(metrics_dir / "pi_convergence.csv").to_dict("records")

    vi_iters = [t["iteration"] for t in trace_vi]
    vi_dvs = [t["delta_v"] for t in trace_vi]
    pi_iters = [t["iteration"] for t in trace_pi]
    pi_changes = [t["policy_changes"] for t in trace_pi]

    vi_gamma = metadata["vi"]["gamma"]
    vi_delta = metadata["vi"]["delta"]
    vi_conv_iter = metadata["vi"]["convergence_iter"]
    pi_gamma = metadata["pi"]["gamma"]
    pi_stable_iter = metadata["pi"]["stable_iter"]

    iter_max = max(vi_iters[-1], pi_iters[-1])

    fig, (ax_vi, ax_pi) = plt.subplots(1, 2, figsize=(12, 4))

    # ── Left: VI — value convergence ─────────────────────────────────────────
    ax_vi.semilogy(
        vi_iters, vi_dvs, color=ALGO_COLORS["VI"], linewidth=1.5, label="max ΔV"
    )
    ax_vi.axhline(
        vi_delta,
        color=COLOR_DELTA_LINE,
        linewidth=1,
        linestyle="--",
        label=f"δ = {vi_delta:.0e}",
    )
    ax_vi.set_xlim(0, iter_max + 1)
    ax_vi.set_xlabel("Iteration")
    ax_vi.set_ylabel("max |ΔV| (log scale)")
    ax_vi.set_title(
        f"Value Iteration  (γ={vi_gamma})\n"
        f"stops when max |ΔV| < δ — converged iter {vi_conv_iter}"
    )
    ax_vi.legend(fontsize=9)
    ax_vi.grid(True, alpha=0.3)
    ax_vi.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ── Right: PI — policy stability ─────────────────────────────────────────
    # Only plot non-zero bars — zero means stable, annotated by vline instead
    nonzero = [(i, c) for i, c in zip(pi_iters, pi_changes) if c > 0]
    if nonzero:
        nz_iters, nz_changes = zip(*nonzero)
        ax_pi.bar(
            nz_iters,
            nz_changes,
            color=ALGO_COLORS["PI"],
            alpha=0.75,
            label="Policy changes",
        )
    ax_pi.set_yscale("log")
    ax_pi.axvline(
        pi_stable_iter,
        color=COLOR_STABLE_LINE,
        linewidth=1.2,
        linestyle=":",
        label=f"Policy stable (iter {pi_stable_iter})",
    )
    ax_pi.annotate(
        "Optimal policy found",
        xy=(pi_stable_iter, 0),
        xycoords=("data", "axes fraction"),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=8,
        color=COLOR_STABLE_LINE,
        arrowprops=dict(arrowstyle="-", color=COLOR_STABLE_LINE, lw=0.8),
    )
    ax_pi.set_xlim(0, iter_max + 1)
    ax_pi.set_xlabel("Iteration")
    ax_pi.set_ylabel("# policy changes (log scale)")
    ax_pi.set_title(
        f"Policy Iteration  (γ={pi_gamma})\n"
        f"stops when policy changes = 0 — stable iter {pi_stable_iter}"
    )
    ax_pi.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_pi.legend(fontsize=9)
    ax_pi.grid(True, alpha=0.3)

    fig.suptitle("VI vs PI Convergence — Blackjack", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = fig_dir / "blackjack_convergence.png"
    fig.savefig(out, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_bj_policy_map(
    hard_policy: np.ndarray, soft_policy: np.ndarray, fig_dir: Path
) -> Path:
    """Crisp Blackjack decision-region heatmap: solid colors, sharp boundaries."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    # Fixed semantic axis labels (environment-driven, not style choices)
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

    cmap = mcolors.ListedColormap([BJ_ACTION_COLORS["Stick"], BJ_ACTION_COLORS["Hit"]])

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
        mpatches.Patch(color=BJ_ACTION_COLORS["Stick"], label="Stick"),
        mpatches.Patch(color=BJ_ACTION_COLORS["Hit"], label="Hit"),
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
    fig.savefig(out, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_bj_value_surface(
    hard_V: np.ndarray, soft_V: np.ndarray, fig_dir: Path
) -> Path:
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
    for idx, (Z, dealer_grid, hand_grid, h_ticks, title) in enumerate(
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
        ax.set_yticks(h_ticks[::3])
        ax.set_yticklabels([str(t) for t in h_ticks[::3]], fontsize=7)
        ax.tick_params(axis="z", labelsize=7)
        ax.view_init(elev=28, azim=-55)

    fig.colorbar(surf, ax=fig.axes, label="V(s)", shrink=0.5, aspect=12, pad=0.05)
    fig.suptitle("VI Value Function — Blackjack", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = fig_dir / "blackjack_vi_value_heatmap.png"
    fig.savefig(out, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out
