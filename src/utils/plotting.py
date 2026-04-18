"""Centralized plotting layer for RL experiment phases.

Style-driven constants live here so figures stay consistent across phases.
Plot functions read from saved CSV/JSON/NPZ artifacts so figures can be
regenerated without re-running experiments.

No bare plt.show() calls — all figures saved to artifacts/figures/.
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
CP_GRID_COLORS: dict[str, str] = {
    "coarse": "#DD8452",  # orange
    "default": "#4C72B0",  # blue
    "fine": "#55A868",  # green
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
        metadata:    summary dict with vi.{gamma,delta,convergence_iter}
                     and pi.{gamma,stable_iter}
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
    fig.suptitle("Optimal Policy — Blackjack (VI = PI)", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = fig_dir / "blackjack_policy_heatmap.png"
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
    fig.suptitle("Blackjack Value Surface (VI)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = fig_dir / "blackjack_value_surface_3d.png"
    fig.savefig(out, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Phase 3: CartPole figures ─────────────────────────────────────────────────


def plot_cp_convergence(
    metrics_dir: Path,
    algo: str,
    title: str,
    delta: float,
    grid_n_states: dict[str, int],
    grid_names: list[str],
    fig_dir: Path,
) -> Path:
    """ΔV vs iteration for one algorithm, one curve per grid. Reads from CSV.

    Args:
        metrics_dir:    directory containing vi_convergence.csv / pi_convergence.csv
        algo:           "vi" or "pi"
        title:          figure title
        delta:          convergence threshold (VI_DELTA or PI_DELTA) for the δ line
        grid_n_states:  {"coarse": N, "default": N, "fine": N} for legend labels
        grid_names:     canonical grid order from the checkpoint (not source constant)
        fig_dir:        output directory
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(metrics_dir / f"{algo}_convergence.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    for grid_name in grid_names:
        sub = df[df["grid"] == grid_name]
        n = grid_n_states.get(grid_name, "?")
        ax.semilogy(
            sub["iteration"],
            sub["delta_v"],
            label=f"{grid_name} ({n} states)",
            color=CP_GRID_COLORS[grid_name],
            linewidth=1.5,
        )

    ax.axhline(
        delta,
        color=COLOR_DELTA_LINE,
        linewidth=1,
        linestyle="--",
        label=f"δ = {delta:.0e}",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("max |ΔV| (log scale)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f"cartpole_{algo}_convergence.png"
    out = fig_dir / filename
    fig.savefig(out, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cp_discretization_study(
    metrics_dir: Path,
    grid_names: list[str],
    fig_dir: Path,
) -> Path:
    """3-panel discretization study. Reads from discretization_study.csv.

    Panels: mean episode length vs grid, wall-clock vs grid, coverage % vs grid.

    Args:
        metrics_dir:  directory containing discretization_study.csv
        grid_names:   canonical grid order from the checkpoint (not source constant)
        fig_dir:      output directory
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(metrics_dir / "discretization_study.csv")
    x = list(range(len(grid_names)))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel 1: policy quality
    for algo in ["VI", "PI"]:
        sub = df[df["algorithm"] == algo].set_index("grid")
        means = [sub.loc[g, "mean_episode_len"] for g in grid_names]
        iqrs = [sub.loc[g, "eval_episode_len_iqr"] for g in grid_names]
        axes[0].errorbar(
            x,
            means,
            yerr=iqrs,
            label=algo,
            marker="o",
            capsize=4,
            linewidth=1.5,
            color=ALGO_COLORS[algo],
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(grid_names)
    axes[0].set_xlabel("Grid")
    axes[0].set_ylabel("Mean episode length")
    axes[0].set_title("Policy quality vs grid")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: planning wall-clock
    for algo in ["VI", "PI"]:
        sub = df[df["algorithm"] == algo].set_index("grid")
        walls = [sub.loc[g, "wall_clock_s"] for g in grid_names]
        axes[1].plot(
            x, walls, label=algo, marker="o", linewidth=1.5, color=ALGO_COLORS[algo]
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(grid_names)
    axes[1].set_xlabel("Grid")
    axes[1].set_ylabel("Planning wall-clock (s)")
    axes[1].set_title("Planning time vs grid")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: model coverage
    # Use VI rows (coverage_pct is per-grid, same for both algorithms)
    sub_vi = df[df["algorithm"] == "VI"].set_index("grid")
    coverages = [sub_vi.loc[g, "coverage_pct"] * 100 for g in grid_names]
    bars = axes[2].bar(
        x,
        coverages,
        color=[CP_GRID_COLORS[g] for g in grid_names],
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
    axes[2].set_xticklabels(grid_names)
    axes[2].set_xlabel("Grid")
    axes[2].set_ylabel("Coverage (%)")
    axes[2].set_title("Model coverage vs grid")
    axes[2].set_ylim(0, 110)
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle("CartPole Discretization Study", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = fig_dir / "cartpole_discretization_study.png"
    fig.savefig(out, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cp_policy_slice(
    npz_path: Path,
    grid_n_states: dict[str, int],
    grid_configs: dict[str, dict],
    grid_names: list[str],
    fig_dir: Path,
) -> Path:
    """Decision-boundary slice in (θ, θ̇) at x=0, ẋ=0. Loads from NPZ.

    Policies are loaded from the plot-support NPZ. Discretizers are
    reconstructed from the saved grid_configs (from checkpoint config_snapshot),
    not from the current source config, so rerenders remain reproducible.

    Args:
        npz_path:       path to plot_cp_grids.npz
        grid_n_states:  {"coarse": N, "default": N, "fine": N} for subplot titles
        grid_configs:   per-grid discretizer configs, as stored in the checkpoint
        grid_names:     canonical grid order from the checkpoint (not source constant)
        fig_dir:        output directory
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    from src.envs.cartpole_discretizer import CartPoleDiscretizer

    grids = np.load(npz_path)

    theta_lim = 0.20
    thetadot_lim = 3.0
    N = 120
    thetas = np.linspace(-theta_lim, theta_lim, N)
    thetadots = np.linspace(-thetadot_lim, thetadot_lim, N)
    TH, TD = np.meshgrid(thetas, thetadots)
    obs_grid = np.stack(
        [np.zeros_like(TH.ravel()), np.zeros_like(TH.ravel()), TH.ravel(), TD.ravel()],
        axis=1,
    )

    cmap = mcolors.ListedColormap([ALGO_COLORS["VI"], ALGO_COLORS["PI"]])

    n_grids = len(grid_names)
    fig, axes = plt.subplots(2, n_grids, figsize=(4 * n_grids, 7))

    for col, grid_name in enumerate(grid_names):
        disc = CartPoleDiscretizer(grid_config=grid_configs[grid_name])
        states = np.array([disc.obs_to_state(o) for o in obs_grid])
        n = grid_n_states.get(grid_name, "?")

        for row, (algo, policy_key) in enumerate(
            [("VI", f"policy_vi_{grid_name}"), ("PI", f"policy_pi_{grid_name}")]
        ):
            policy = grids[policy_key]
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
            ax.axvline(
                CARTPOLE_TERMINAL_ANGLE_RAD,
                color="red",
                linewidth=0.8,
                linestyle="--",
                alpha=0.7,
            )
            ax.axvline(
                -CARTPOLE_TERMINAL_ANGLE_RAD,
                color="red",
                linewidth=0.8,
                linestyle="--",
                alpha=0.7,
            )
            ax.axhline(0, color="white", linewidth=0.5, linestyle=":", alpha=0.5)
            ax.axvline(0, color="white", linewidth=0.5, linestyle=":", alpha=0.5)

            if row == 0:
                ax.set_title(f"{grid_name}\n({n} states)", fontsize=9)
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
        mpatches.Patch(color=ALGO_COLORS["VI"], label="Push left (0)"),
        mpatches.Patch(color=ALGO_COLORS["PI"], label="Push right (1)"),
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
        "CartPole Policy Slice: (θ, θ̇) at x=0, ẋ=0", fontsize=12, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    out = fig_dir / "cartpole_policy_slice.png"
    fig.savefig(out, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out
