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

# Algorithm colors — blue for VI, brown for PI; fixed across all figures.
# No other constant in this file should reuse these hues.
ALGO_COLORS: dict[str, str] = {
    "VI": "#4C72B0",  # blue
    "PI": "#8C564B",  # brown
}

# Grid colors — used whenever marks represent coarse / default / fine grids,
# including convergence curves, bar charts, and coverage figures.
CP_GRID_COLORS: dict[str, str] = {
    "coarse": "#BAB0AC",  # muted gray
    "default": "#76B7B2",  # teal
    "fine": "#B07AA1",  # purple
}

# Action colors — semantically distinct from algorithm and grid colors.
# Used only in policy-map and action-slice figures; must not imply algorithm identity.
BJ_ACTION_COLORS: dict[str, str] = {
    "Stick": "#59A14F",  # green
    "Hit": "#F28E2B",  # orange
}
CP_ACTION_COLORS: dict[str, str] = {
    "left": "#F28E2B",  # orange — action 0, push left
    "right": "#59A14F",  # green — action 1, push right
}

# Reference colors — threshold lines, stable-iteration markers, and annotations.
# Muted so they do not dominate the primary data ink.
REFERENCE_COLORS: dict[str, str] = {
    "threshold": "#E15759",  # muted red — convergence threshold (δ lines)
    "stable": "#499894",  # muted teal — stable-iteration markers
}

# CartPole terminal-angle marker — environment-defined constant (±12°).
CARTPOLE_TERMINAL_ANGLE_RAD: float = 12 * np.pi / 180  # ≈ 0.2094 rad


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

    fig, (ax_vi, ax_pi) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    # ── Left: VI — value convergence ─────────────────────────────────────────
    ax_vi.semilogy(
        vi_iters, vi_dvs, color=ALGO_COLORS["VI"], linewidth=1.5, label="max ΔV"
    )
    ax_vi.axhline(
        vi_delta,
        color=REFERENCE_COLORS["threshold"],
        linewidth=1,
        linestyle="--",
        label=f"δ = {vi_delta:.0e}",
    )
    ax_vi.set_xlim(0, iter_max + 1)
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
        color=REFERENCE_COLORS["stable"],
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
        color=REFERENCE_COLORS["stable"],
        arrowprops=dict(arrowstyle="-", color=REFERENCE_COLORS["stable"], lw=0.8),
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
    plt.tight_layout(rect=[0, 0, 1, 0.97])
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

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
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


def plot_cp_vi_convergence(
    metrics_dir: Path,
    grid_n_states: dict[str, int],
    grid_names: list[str],
    fig_dir: Path,
) -> Path:
    """VI residual vs iteration. All grids overlap — one representative curve shown
    with an annotation confirming identical sweep counts across grids.

    Args:
        metrics_dir:   directory containing vi_convergence.csv
        grid_n_states: {"coarse": N, ...} for the annotation
        grid_names:    canonical grid order from the checkpoint
        fig_dir:       output directory
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    vi_df = pd.read_csv(metrics_dir / "vi_convergence.csv")

    # Use "default" as the representative; fall back to first grid if absent.
    rep = "default" if "default" in grid_names else grid_names[0]
    sub = vi_df[vi_df["grid"] == rep]
    sweep_count = int(sub["iteration"].iloc[-1])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(
        sub["iteration"],
        sub["delta_v"],
        color=CP_GRID_COLORS[rep],
        linewidth=2,
        label=f"{rep} ({grid_n_states.get(rep, '?')} states)",
    )

    # Annotation: all grids produce the same convergence curve
    state_counts = ", ".join(f"{g}: {grid_n_states.get(g, '?')}" for g in grid_names)
    ax.text(
        0.97,
        0.97,
        f"All grids converge in {sweep_count} sweeps\n({state_counts})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("max |ΔV| (log scale)")
    ax.set_title("Value Iteration Convergence — CartPole", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = fig_dir / "cartpole_vi_convergence.png"
    fig.savefig(out, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cp_pi_convergence(
    metrics_dir: Path,
    grid_n_states: dict[str, int],
    grid_names: list[str],
    fig_dir: Path,
) -> Path:
    """PI policy changes vs iteration, one curve per grid.

    Non-zero policy-change iterations plotted on log y; dashed vline marks the
    stable iteration (policy_changes == 0) for each grid.

    Args:
        metrics_dir:   directory containing pi_convergence.csv
        grid_n_states: {"coarse": N, ...} for legend labels
        grid_names:    canonical grid order from the checkpoint
        fig_dir:       output directory
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    pi_df = pd.read_csv(metrics_dir / "pi_convergence.csv")

    fig, ax = plt.subplots(figsize=(7, 5))
    for grid_name in grid_names:
        sub = pi_df[pi_df["grid"] == grid_name].copy()
        n = grid_n_states.get(grid_name, "?")
        color = CP_GRID_COLORS[grid_name]
        non_zero = sub[sub["policy_changes"] > 0]
        ax.semilogy(
            non_zero["iteration"],
            non_zero["policy_changes"],
            label=f"{grid_name} ({n} states)",
            color=color,
            linewidth=1.5,
            marker="o",
            markersize=4,
        )
        stable_iter = int(sub["iteration"].iloc[-1])
        ax.axvline(
            stable_iter,
            color=color,
            linewidth=0.9,
            linestyle="--",
            alpha=0.75,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Policy changes (log scale)")
    ax.set_title("Policy Iteration Convergence — CartPole", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = fig_dir / "cartpole_pi_convergence.png"
    fig.savefig(out, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cp_mean_episode_length(
    metrics_dir: Path,
    grid_names: list[str],
    fig_dir: Path,
) -> Path:
    """Grouped bar chart: mean episode length by grid, VI vs PI (± IQR).

    Args:
        metrics_dir:  directory containing discretization_study.csv
        grid_names:   canonical grid order from the checkpoint
        fig_dir:      output directory
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = pd.read_csv(metrics_dir / "discretization_study.csv")
    x = np.arange(len(grid_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    for offset, algo in zip([-width / 2, width / 2], ["VI", "PI"]):
        sub = df[df["algorithm"] == algo].set_index("grid")
        means = [sub.loc[g, "mean_episode_len"] for g in grid_names]
        iqrs = [sub.loc[g, "eval_episode_len_iqr"] for g in grid_names]
        ax.bar(
            x + offset,
            means,
            width,
            yerr=iqrs,
            label=algo,
            color=ALGO_COLORS[algo],
            alpha=0.85,
            capsize=4,
            error_kw={"elinewidth": 1.2},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(grid_names)
    ax.set_xlabel("Grid")
    ax.set_ylabel("Mean episode length")
    ax.set_title("Mean Episode Length by Grid — CartPole", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = fig_dir / "cartpole_mean_episode_length_by_grid.png"
    fig.savefig(out, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cp_wall_clock(
    metrics_dir: Path,
    grid_names: list[str],
    fig_dir: Path,
) -> Path:
    """Grouped bar chart: planning wall-clock by grid, VI vs PI.

    Args:
        metrics_dir:  directory containing discretization_study.csv
        grid_names:   canonical grid order from the checkpoint
        fig_dir:      output directory
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = pd.read_csv(metrics_dir / "discretization_study.csv")
    x = np.arange(len(grid_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    for offset, algo in zip([-width / 2, width / 2], ["VI", "PI"]):
        sub = df[df["algorithm"] == algo].set_index("grid")
        walls = [sub.loc[g, "wall_clock_s"] for g in grid_names]
        ax.bar(
            x + offset,
            walls,
            width,
            label=algo,
            color=ALGO_COLORS[algo],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(grid_names)
    ax.set_xlabel("Grid")
    ax.set_ylabel("Planning wall-clock (s)")
    ax.set_title("Planning Wall-Clock by Grid — CartPole", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = fig_dir / "cartpole_wall_clock_by_grid.png"
    fig.savefig(out, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cp_model_coverage(
    metrics_dir: Path,
    grid_names: list[str],
    fig_dir: Path,
) -> Path:
    """Coverage % by grid — single bar chart with a coverage-specific palette
    distinct from VI/PI algorithm colors and CP_GRID_COLORS.

    Args:
        metrics_dir:  directory containing discretization_study.csv
        grid_names:   canonical grid order from the checkpoint
        fig_dir:      output directory
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = pd.read_csv(metrics_dir / "discretization_study.csv")
    # coverage_pct is per-grid (same for VI and PI); use VI rows
    sub = df[df["algorithm"] == "VI"].set_index("grid")
    coverages = [sub.loc[g, "coverage_pct"] * 100 for g in grid_names]
    x = np.arange(len(grid_names))

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        x,
        coverages,
        color=[CP_GRID_COLORS[g] for g in grid_names],
        alpha=0.9,
        edgecolor="white",
    )
    for bar, val in zip(bars, coverages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(grid_names)
    ax.set_xlabel("Grid")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Model Coverage by Grid — CartPole", fontweight="bold")
    ax.set_ylim(0, 115)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = fig_dir / "cartpole_model_coverage_by_grid.png"
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

    cmap = mcolors.ListedColormap([CP_ACTION_COLORS["left"], CP_ACTION_COLORS["right"]])

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
        mpatches.Patch(color=CP_ACTION_COLORS["left"], label="Push left (0)"),
        mpatches.Patch(color=CP_ACTION_COLORS["right"], label="Push right (1)"),
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


# ── Phase 4 — Model-Free (SARSA / Q-Learning) ─────────────────────────────────

# Algorithm colors for model-free methods — distinct from VI/PI ALGO_COLORS.
MF_ALGO_COLORS: dict[str, str] = {
    "sarsa": "#E377C2",  # pink
    "qlearning": "#17BECF",  # cyan
}


def plot_mf_learning_curve(
    metrics_dir: Path,
    fig_dir: Path,
) -> Path:
    """Learning curves for SARSA and Q-Learning, split by regime.

    Reads ``mf_learning_curves.csv``
    (columns: algorithm, seed, regime, episode, window_mean).

    Two subplots side-by-side: controlled (left) and tuned (right).
    Within each subplot, one curve per algorithm with ±1 std band over seeds.
    Line color = MF_ALGO_COLORS; data is already window-smoothed.

    Args:
        metrics_dir: Directory containing mf_learning_curves.csv.
        fig_dir:     Output directory.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(metrics_dir / "mf_learning_curves.csv")
    regimes = ["controlled", "tuned"]
    # Gracefully handle single-regime CSVs (e.g. from smoke tests)
    if "regime" not in df.columns:
        df["regime"] = "tuned"
    available_regimes = [r for r in regimes if r in df["regime"].unique()]

    fig, axes = plt.subplots(
        len(available_regimes), 1, figsize=(7, 5 * len(available_regimes)), sharey=True
    )
    if len(available_regimes) == 1:
        axes = [axes]

    for ax, regime in zip(axes, available_regimes):
        regime_df = df[df["regime"] == regime]
        for algo in ["sarsa", "qlearning"]:
            sub = regime_df[regime_df["algorithm"] == algo]
            if sub.empty:
                continue
            grouped = (
                sub.groupby("episode")["window_mean"].agg(["mean", "std"]).reset_index()
            )
            color = MF_ALGO_COLORS[algo]
            label = "SARSA" if algo == "sarsa" else "Q-Learning"
            ax.plot(
                grouped["episode"],
                grouped["mean"],
                label=label,
                color=color,
                linewidth=1.5,
            )
            ax.fill_between(
                grouped["episode"],
                grouped["mean"] - grouped["std"].fillna(0),
                grouped["mean"] + grouped["std"].fillna(0),
                alpha=0.15,
                color=color,
            )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Window-mean return (100-ep window)")
        ax.set_title(f"{regime.capitalize()} schedule", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Blackjack Learning Curves — SARSA vs Q-Learning",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    out = fig_dir / "blackjack_mf_learning_curves.png"
    fig.savefig(out, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_mf_comparison(
    metrics_dir: Path,
    fig_dir: Path,
) -> Path:
    """Grouped bar chart: final win-rate comparison, controlled vs tuned.

    Reads ``mf_eval_summary.csv``
    (columns: algorithm, regime, metric, mean, std, iqr).

    Two subplots: controlled (left) and tuned (right).  Within each subplot,
    grouped bars for SARSA and Q-Learning across win/draw/loss rates.
    Error bars show std across seeds; IQR shown as secondary annotation.

    Args:
        metrics_dir: Directory containing mf_eval_summary.csv.
        fig_dir:     Output directory.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = pd.read_csv(metrics_dir / "mf_eval_summary.csv")
    if "regime" not in df.columns:
        df["regime"] = "tuned"
    available_regimes = [
        r for r in ["controlled", "tuned"] if r in df["regime"].unique()
    ]

    metrics = ["win_rate", "draw_rate", "loss_rate"]
    labels = ["Win", "Draw", "Loss"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, axes = plt.subplots(
        len(available_regimes), 1, figsize=(7, 5 * len(available_regimes)), sharey=True
    )
    if len(available_regimes) == 1:
        axes = [axes]

    for ax, regime in zip(axes, available_regimes):
        regime_df = df[df["regime"] == regime]
        for offset, algo, display in zip(
            [-width / 2, width / 2],
            ["sarsa", "qlearning"],
            ["SARSA", "Q-Learning"],
        ):
            sub = regime_df[regime_df["algorithm"] == algo].set_index("metric")
            if sub.empty:
                continue
            means = [float(sub.loc[m, "mean"]) for m in metrics]
            stds = [float(sub.loc[m, "std"]) for m in metrics]
            ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                label=display,
                color=MF_ALGO_COLORS[algo],
                alpha=0.85,
                capsize=4,
                error_kw={"elinewidth": 1.2},
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Outcome")
        ax.set_ylabel("Rate")
        ax.set_title(f"{regime.capitalize()} schedule", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(
        "Blackjack Final Eval — SARSA vs Q-Learning", fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out = fig_dir / "blackjack_mf_comparison.png"
    fig.savefig(out, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_mf_hp_sensitivity(
    metrics_dir: Path,
    fig_dir: Path,
) -> Path:
    """HP sensitivity scatter: mean_return vs α_start, coloured by ε_decay_steps.

    Reads ``mf_hp_search.csv``
    (columns: algorithm, alpha_start, eps_decay_steps, mean_return).

    Args:
        metrics_dir: Directory containing mf_hp_search.csv.
        fig_dir:     Output directory.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(metrics_dir / "mf_hp_search.csv")
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), constrained_layout=True)

    for ax, algo, display in zip(
        axes,
        ["sarsa", "qlearning"],
        ["SARSA", "Q-Learning"],
    ):
        sub = df[df["algorithm"] == algo]
        sc = ax.scatter(
            sub["alpha_start"],
            sub["mean_return"],
            c=sub["eps_decay_steps"],
            cmap="viridis",
            alpha=0.8,
            edgecolors="white",
            linewidths=0.4,
            s=60,
        )
        ax.set_xlabel("α start")
        ax.set_ylabel("Mean return (win rate − loss rate)")
        ax.set_title(display, fontweight="bold")
        ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.8)
    cbar.set_label("ε decay steps")
    fig.suptitle(
        "HP Sensitivity — Blackjack Model-Free", fontsize=12, fontweight="bold"
    )
    out = fig_dir / "blackjack_mf_hp_sensitivity.png"
    fig.savefig(out, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Phase 5 — CartPole model-free ─────────────────────────────────────────────


def plot_cp_mf_learning_curve(
    metrics_dir: Path,
    fig_dir: Path,
) -> Path:
    """Learning curves for CartPole SARSA and Q-Learning, split by regime.

    Reads ``mf_learning_curves.csv``
    (columns: algorithm, seed, regime, episode, window_mean).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(metrics_dir / "mf_learning_curves.csv")
    if "regime" not in df.columns:
        df["regime"] = "tuned"
    available_regimes = [
        r for r in ["controlled", "tuned"] if r in df["regime"].unique()
    ]

    fig, axes = plt.subplots(
        len(available_regimes), 1, figsize=(7, 5 * len(available_regimes)), sharey=True
    )
    if len(available_regimes) == 1:
        axes = [axes]

    for ax, regime in zip(axes, available_regimes):
        regime_df = df[df["regime"] == regime]
        for algo in ["sarsa", "qlearning"]:
            sub = regime_df[regime_df["algorithm"] == algo]
            if sub.empty:
                continue
            grouped = (
                sub.groupby("episode")["window_mean"].agg(["mean", "std"]).reset_index()
            )
            color = MF_ALGO_COLORS[algo]
            label = "SARSA" if algo == "sarsa" else "Q-Learning"
            ax.plot(
                grouped["episode"],
                grouped["mean"],
                label=label,
                color=color,
                linewidth=1.5,
            )
            ax.fill_between(
                grouped["episode"],
                grouped["mean"] - grouped["std"].fillna(0),
                grouped["mean"] + grouped["std"].fillna(0),
                alpha=0.15,
                color=color,
            )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Window-mean episode length (100-ep window)")
        ax.set_title(f"{regime.capitalize()} schedule", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "CartPole Learning Curves — SARSA vs Q-Learning", fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out = fig_dir / "cartpole_mf_learning_curves.png"
    fig.savefig(out, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cp_mf_comparison(
    metrics_dir: Path,
    fig_dir: Path,
) -> Path:
    """Grouped bar chart: mean episode length comparison, controlled vs tuned.

    Reads ``mf_eval_summary.csv``
    (columns: algorithm, regime, metric, mean, std, iqr).

    Error whiskers = IQR/2 (half-IQR symmetric, robust to seed outliers).
    IQR value shown in legend label.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = pd.read_csv(metrics_dir / "mf_eval_summary.csv")
    if "regime" not in df.columns:
        df["regime"] = "tuned"
    available_regimes = [
        r for r in ["controlled", "tuned"] if r in df["regime"].unique()
    ]

    x = np.array([0])
    width = 0.35

    fig, axes = plt.subplots(
        len(available_regimes), 1, figsize=(7, 5 * len(available_regimes)), sharey=True
    )
    if len(available_regimes) == 1:
        axes = [axes]

    for ax, regime in zip(axes, available_regimes):
        regime_df = df[(df["regime"] == regime) & (df["metric"] == "mean_episode_len")]
        for offset, algo, display in zip(
            [-width / 2, width / 2],
            ["sarsa", "qlearning"],
            ["SARSA", "Q-Learning"],
        ):
            sub = regime_df[regime_df["algorithm"] == algo]
            if sub.empty:
                continue
            mean_val = float(sub["mean"].iloc[0])
            iqr_val = float(sub["iqr"].iloc[0])
            ax.bar(
                x + offset,
                [mean_val],
                width,
                yerr=[iqr_val / 2],
                label=f"{display} (IQR={iqr_val:.1f})",
                color=MF_ALGO_COLORS[algo],
                alpha=0.85,
                capsize=5,
                error_kw={"elinewidth": 1.2},
            )
        ax.set_xticks(x)
        ax.set_xticklabels(["Mean episode length"])
        ax.set_ylabel("Mean episode length (steps)")
        ax.set_title(f"{regime.capitalize()} schedule", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(
        "CartPole Final Eval — SARSA vs Q-Learning", fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out = fig_dir / "cartpole_mf_comparison.png"
    fig.savefig(out, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cp_mf_hp_sensitivity(
    metrics_dir: Path,
    fig_dir: Path,
) -> Path:
    """HP sensitivity scatter: mean_episode_len vs α_start, coloured by ε_decay_steps.

    Reads ``mf_hp_search.csv``
    (columns: algorithm, alpha_start, eps_decay_steps, mean_episode_len).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(metrics_dir / "mf_hp_search.csv")
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), constrained_layout=True)

    for ax, algo, display in zip(axes, ["sarsa", "qlearning"], ["SARSA", "Q-Learning"]):
        sub = df[df["algorithm"] == algo]
        sc = ax.scatter(
            sub["alpha_start"],
            sub["mean_episode_len"],
            c=sub["eps_decay_steps"],
            cmap="viridis",
            alpha=0.8,
            edgecolors="white",
            linewidths=0.4,
            s=60,
        )
        ax.set_xlabel("α start")
        ax.set_ylabel("Mean episode length")
        ax.set_title(display, fontweight="bold")
        ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.8)
    cbar.set_label("ε decay steps")
    fig.suptitle("HP Sensitivity — CartPole Model-Free", fontsize=12, fontweight="bold")
    out = fig_dir / "cartpole_mf_hp_sensitivity.png"
    fig.savefig(out, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cp_mf_discretization(
    metrics_dir: Path,
    fig_dir: Path,
) -> Path:
    """Grouped bar chart: final mean episode length vs discretization grid.

    Reads ``mf_discretization.csv``
    (columns: grid, algorithm, seed, final_mean_len, convergence_episode).
    One group per grid (coarse / default / fine), bars for SARSA and Q-Learning.
    Error whiskers = IQR/2 (half-IQR, robust to seed outliers).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = pd.read_csv(metrics_dir / "mf_discretization.csv")
    grid_names = [g for g in ["coarse", "default", "fine"] if g in df["grid"].unique()]
    x = np.arange(len(grid_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    for offset, algo, display in zip(
        [-width / 2, width / 2], ["sarsa", "qlearning"], ["SARSA", "Q-Learning"]
    ):
        means, iqr_halves = [], []
        for grid in grid_names:
            sub = df[(df["grid"] == grid) & (df["algorithm"] == algo)]["final_mean_len"]
            means.append(float(sub.mean()))
            iqr = float(np.percentile(sub, 75) - np.percentile(sub, 25))
            iqr_halves.append(iqr / 2)
        ax.bar(
            x + offset,
            means,
            width,
            yerr=iqr_halves,
            label=display,
            color=MF_ALGO_COLORS[algo],
            alpha=0.85,
            capsize=4,
            error_kw={"elinewidth": 1.2},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in grid_names])
    ax.set_xlabel("Discretization grid")
    ax.set_ylabel("Mean episode length (steps)")
    ax.set_title("CartPole Discretization Study — Tuned Regime", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = fig_dir / "cartpole_mf_discretization.png"
    fig.savefig(out, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Phase 6: Cross-Method Comparison figures ──────────────────────────────────

# Model-free algorithm colors (extend ALGO_COLORS for MF methods)
_MF_COLORS: dict[str, str] = {
    "sarsa": "#E377C2",  # pink
    "qlearning": "#BCBD22",  # yellow-green
}
_ALL_METHOD_COLORS: dict[str, str] = {**ALGO_COLORS, **_MF_COLORS}
_MF_LABELS: dict[str, str] = {"sarsa": "SARSA", "qlearning": "Q-Learning"}
_ENV_LABELS = {"blackjack": "Blackjack", "cartpole": "CartPole"}


def _grouped_bars(
    ax,
    groups: list[str],
    series: dict[str, list[float]],
    errors: dict[str, list[float]] | None = None,
    colors: dict[str, str] | None = None,
    width: float = 0.35,
) -> None:
    """Draw grouped bar chart on *ax*.

    Args:
        groups:  x-axis group labels (e.g. ["Blackjack", "CartPole"])
        series:  {label: [value_per_group]}
        errors:  {label: [error_per_group]} — drawn as ±error whiskers
        colors:  {label: hex} — falls back to matplotlib default cycle
        width:   total width budget per group (split across series)
    """

    n_groups = len(groups)
    n_series = len(series)
    bar_w = width / n_series
    x = np.arange(n_groups)
    offsets = np.linspace(-(width - bar_w) / 2, (width - bar_w) / 2, n_series)

    for offset, (label, vals) in zip(offsets, series.items()):
        errs = errors.get(label) if errors else None
        color = (colors or {}).get(label)
        ax.bar(
            x + offset,
            vals,
            bar_w,
            label=label,
            color=color,
            yerr=errs,
            capsize=4,
            error_kw={"elinewidth": 1.2, "ecolor": "#555555"},
        )
    ax.set_xticks(x)
    ax.set_xticklabels(groups)


def plot_p6_planning_efficiency(summary: dict, fig_dir: Path) -> Path:
    """VI vs PI iterations to convergence, both MDPs (two-panel bar chart)."""
    import matplotlib.pyplot as plt

    data = summary["planning_efficiency"]
    envs = ["blackjack", "cartpole"]

    fig, axes = plt.subplots(2, 1, figsize=(7, 8))
    for ax, env in zip(axes, envs):
        vi_iters = data[env]["vi"]["iterations"]
        pi_iters = data[env]["pi"]["iterations"]
        bars = ax.bar(
            ["VI", "PI"],
            [vi_iters, pi_iters],
            color=[ALGO_COLORS["VI"], ALGO_COLORS["PI"]],
            width=0.5,
        )
        for bar, val in zip(bars, [vi_iters, pi_iters]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vi_iters, pi_iters) * 0.02,
                str(int(val)),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title(_ENV_LABELS[env])
        ax.set_ylabel("Iterations to convergence")
        ax.set_ylim(0, max(vi_iters, pi_iters) * 1.25)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Planning Efficiency — VI vs PI", fontsize=12)
    plt.tight_layout()
    out = fig_dir / "planning_efficiency_comparison.png"
    fig.savefig(out, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_p6_learning_efficiency(summary: dict, fig_dir: Path) -> Path:
    """SARSA vs Q-Learning episodes to convergence, controlled regime."""
    import matplotlib.pyplot as plt

    data = summary["learning_efficiency"]
    envs = ["blackjack", "cartpole"]

    fig, axes = plt.subplots(2, 1, figsize=(7, 8))
    for ax, env in zip(axes, envs):
        algos = ["sarsa", "qlearning"]
        means = [
            v
            if (v := data[env][a]["mean_convergence_episode"]) is not None
            else float("nan")
            for a in algos
        ]
        iqrs = [
            (v / 2)
            if (v := data[env][a]["convergence_episode_iqr"]) is not None
            else float("nan")
            for a in algos
        ]
        labels = [_MF_LABELS[a] for a in algos]
        colors = [_MF_COLORS[a] for a in algos]

        bars = ax.bar(
            labels,
            means,
            color=colors,
            width=0.5,
            yerr=iqrs,
            capsize=5,
            error_kw={"elinewidth": 1.2, "ecolor": "#555555"},
        )
        finite_means = [v for v in means if v == v]
        max_mean = max(finite_means) if finite_means else 0
        for bar, val in zip(bars, means):
            if val == val and val:  # skip NaN and zero
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_mean * 0.02,
                    f"{int(val):,}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_title(_ENV_LABELS[env])
        ax.set_ylabel("Episodes to convergence (controlled)")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Learning Efficiency — Controlled Regime", fontsize=12)
    plt.tight_layout()
    out = fig_dir / "learning_efficiency_comparison.png"
    fig.savefig(out, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_p6_stability(summary: dict, fig_dir: Path) -> Path:
    """SARSA vs Q-Learning stability metrics, controlled regime.

    Two subplots per MDP: final_window_iqr and convergence_episode_iqr.
    """
    import matplotlib.pyplot as plt

    data = summary["stability"]
    envs = ["blackjack", "cartpole"]
    algos = ["sarsa", "qlearning"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    metric_labels = {
        "final_window_iqr": "Final-window IQR\n(return stability)",
        "convergence_episode_iqr": "Convergence-episode IQR\n(speed stability)",
    }

    for col, env in enumerate(envs):
        for row, (metric, ylabel) in enumerate(metric_labels.items()):
            ax = axes[row][col]
            vals = [
                v if (v := data[env][a][metric]) is not None else float("nan")
                for a in algos
            ]
            labels = [_MF_LABELS[a] for a in algos]
            colors = [_MF_COLORS[a] for a in algos]
            bars = ax.bar(labels, vals, color=colors, width=0.5)
            finite_vals = [v for v in vals if v == v]
            max_val = max(finite_vals) if finite_vals else 0
            for bar, val in zip(bars, vals):
                if val == val and val:  # skip NaN and zero
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max_val * 0.02 if max_val else 0.01,
                        f"{val:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
            if row == 0:
                ax.set_title(_ENV_LABELS[env])
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Stability — Controlled Regime (lower = more stable)", fontsize=12)
    plt.tight_layout()
    out = fig_dir / "stability_comparison.png"
    fig.savefig(out, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_p6_wall_clock(summary: dict, fig_dir: Path) -> Path:
    """Wall-clock comparison — all methods; CartPole DP stacked model-build + planning."""
    import matplotlib.pyplot as plt

    wc = summary["wall_clock"]
    methods = ["VI", "PI", "SARSA", "Q-Learning"]
    colors = [
        ALGO_COLORS["VI"],
        ALGO_COLORS["PI"],
        _MF_COLORS["sarsa"],
        _MF_COLORS["qlearning"],
    ]

    bj_times = [
        wc["blackjack"]["vi"],
        wc["blackjack"]["pi"],
        wc["blackjack"]["sarsa"],
        wc["blackjack"]["qlearning"],
    ]
    cp_planning = [
        wc["cartpole"]["vi_planning_s"],
        wc["cartpole"]["pi_planning_s"],
        wc["cartpole"]["sarsa"],
        wc["cartpole"]["qlearning"],
    ]
    # Model-build overhead stacked only on DP bars
    cp_model_build = [
        wc["cartpole"]["model_build_s"],
        wc["cartpole"]["model_build_s"],
        0.0,
        0.0,
    ]

    x = np.arange(len(methods))
    w = 0.35
    fig, axes = plt.subplots(2, 1, figsize=(7, 8))

    # Blackjack
    for i, (m, t, c) in enumerate(zip(methods, bj_times, colors)):
        axes[0].bar(x[i], t, w, color=c, label=m)
        axes[0].text(
            x[i],
            t + max(bj_times) * 0.02,
            f"{t:.2f}s",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_title("Blackjack")
    axes[0].set_ylabel("Wall-clock (s)")
    axes[0].grid(True, alpha=0.3, axis="y")

    # CartPole — stacked for DP
    max_cp = max(p + m for p, m in zip(cp_planning, cp_model_build))
    for i, (m, plan, mbuild, c) in enumerate(
        zip(methods, cp_planning, cp_model_build, colors)
    ):
        axes[1].bar(
            x[i], mbuild, w, color="#CCCCCC", label="Model build" if i == 0 else None
        )
        axes[1].bar(x[i], plan, w, bottom=mbuild, color=c)
        total = plan + mbuild
        axes[1].text(
            x[i],
            total + max_cp * 0.02,
            f"{total:.1f}s",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].set_title("CartPole (default grid)")
    axes[1].set_ylabel("Wall-clock (s)")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].legend(fontsize=8)

    fig.suptitle(
        "Wall-Clock — Controlled Regime (model-free) / Planning Run (DP)", fontsize=11
    )
    plt.tight_layout()
    out = fig_dir / "wall_clock_comparison.png"
    fig.savefig(out, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_p6_final_performance(summary: dict, fig_dir: Path) -> Path:
    """Final performance — all methods; model-free from tuned regime."""
    import matplotlib.pyplot as plt

    fp = summary["final_performance"]

    fig, axes = plt.subplots(2, 1, figsize=(7, 8))

    # ── Blackjack (mean return, ±IQR/2) ──────────────────────────────────────
    ax = axes[0]
    bj_methods = ["VI", "PI", "SARSA", "Q-Learning"]
    bj_vals = [
        fp["blackjack"]["vi"]["mean_eval_return"],
        fp["blackjack"]["pi"]["mean_eval_return"],
        fp["blackjack"]["sarsa"]["mean_return"],
        fp["blackjack"]["qlearning"]["mean_return"],
    ]
    bj_errs = [
        fp["blackjack"]["vi"]["eval_return_iqr"] / 2,
        fp["blackjack"]["pi"]["eval_return_iqr"] / 2,
        fp["blackjack"]["sarsa"]["iqr_return"] / 2,
        fp["blackjack"]["qlearning"]["iqr_return"] / 2,
    ]
    bj_colors = [
        ALGO_COLORS["VI"],
        ALGO_COLORS["PI"],
        _MF_COLORS["sarsa"],
        _MF_COLORS["qlearning"],
    ]
    x = np.arange(len(bj_methods))
    bars = ax.bar(
        x,
        bj_vals,
        0.5,
        color=bj_colors,
        yerr=bj_errs,
        capsize=5,
        error_kw={"elinewidth": 1.2, "ecolor": "#555555"},
    )
    for bar, val in zip(bars, bj_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(bj_methods)
    ax.set_title("Blackjack")
    ax.set_ylabel("Mean return ±IQR/2\n(model-free: tuned regime)")
    ax.grid(True, alpha=0.3, axis="y")

    # ── CartPole (mean episode length, ±IQR/2) ────────────────────────────────
    ax = axes[1]
    cp_methods = ["VI", "PI", "SARSA", "Q-Learning"]
    cp_vals = [
        fp["cartpole"]["vi"]["mean_episode_len"],
        fp["cartpole"]["pi"]["mean_episode_len"],
        fp["cartpole"]["sarsa"]["mean_episode_len"],
        fp["cartpole"]["qlearning"]["mean_episode_len"],
    ]
    cp_errs = [
        fp["cartpole"]["vi"]["eval_episode_len_iqr"] / 2,
        fp["cartpole"]["pi"]["eval_episode_len_iqr"] / 2,
        fp["cartpole"]["sarsa"]["iqr_episode_len"] / 2,
        fp["cartpole"]["qlearning"]["iqr_episode_len"] / 2,
    ]
    cp_colors = [
        ALGO_COLORS["VI"],
        ALGO_COLORS["PI"],
        _MF_COLORS["sarsa"],
        _MF_COLORS["qlearning"],
    ]
    x = np.arange(len(cp_methods))
    bars = ax.bar(
        x,
        cp_vals,
        0.5,
        color=cp_colors,
        yerr=cp_errs,
        capsize=5,
        error_kw={"elinewidth": 1.2, "ecolor": "#555555"},
    )
    for bar, val in zip(bars, cp_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(cp_vals) * 0.02,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(cp_methods)
    ax.set_title("CartPole (default grid)")
    ax.set_ylabel("Mean episode length ±IQR/2\n(model-free: tuned regime)")
    ax.axhline(
        500,
        color=REFERENCE_COLORS["threshold"],
        linewidth=1,
        linestyle="--",
        label="Max (500)",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Final Performance — DP: single run · Model-Free: tuned regime", fontsize=11
    )
    plt.tight_layout()
    out = fig_dir / "final_performance_comparison.png"
    fig.savefig(out, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Phase 7 — DQN Extra Credit ────────────────────────────────────────────────

# DQN variant colors — distinct from VI/PI and model-free tabular palettes
_DQN_COLORS: dict[str, str] = {
    "vanilla_dqn": "#1F77B4",  # medium blue
    "double_dqn": "#FF7F0E",  # orange
}
_DQN_LABELS: dict[str, str] = {
    "vanilla_dqn": "Vanilla DQN",
    "double_dqn": "Double DQN",
}


def plot_p7_dqn_vs_double_dqn(summary: dict, fig_dir: Path) -> Path:
    """Learning curves: mean episode length ± IQR vs episodes for both DQN variants.

    Uses the aggregated per-seed learning curves stored in the Phase 7 checkpoint.
    Seeds that converge early are padded with their final window_mean.
    """
    import matplotlib.pyplot as plt

    curves = summary["learning_curves"]

    fig, ax = plt.subplots(figsize=(8, 4))

    for variant in ["vanilla_dqn", "double_dqn"]:
        vc = curves[variant]
        eps = np.array(vc["episodes"])
        mn = np.array(vc["mean"])
        q25 = np.array(vc["q25"])
        q75 = np.array(vc["q75"])
        color = _DQN_COLORS[variant]
        label = _DQN_LABELS[variant]

        ax.plot(eps, mn, color=color, linewidth=1.8, label=label)
        ax.fill_between(eps, q25, q75, color=color, alpha=0.2)

    ax.axhline(
        500,
        color=REFERENCE_COLORS["threshold"],
        linewidth=1,
        linestyle="--",
        label="Max (500)",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean episode length (window=100)")
    ax.set_title("DQN vs Double DQN — CartPole (5 seeds, mean ± IQR)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = fig_dir / "cartpole_dqn_vs_double_dqn.png"
    fig.savefig(out, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_p7_dqn_vs_tabular(summary: dict, fig_dir: Path) -> Path:
    """Final performance bar chart: DQN variants vs tabular SARSA/Q-Learning (tuned).

    DQN bars use the greedy evaluation metric (mean_eval_ep_len) so the
    comparison is apples-to-apples with Phase 5 post-training greedy eval.
    """
    import matplotlib.pyplot as plt

    vs = summary["variants"]
    tc = summary["tabular_comparison"]

    methods = ["Vanilla\nDQN", "Double\nDQN", "SARSA\n(tuned)", "Q-Learning\n(tuned)"]
    vals = [
        vs["vanilla_dqn"]["mean_eval_ep_len"],
        vs["double_dqn"]["mean_eval_ep_len"],
        tc["sarsa_tuned_mean_ep_len"],
        tc["qlearning_tuned_mean_ep_len"],
    ]
    # IQR/2 for error bars
    errs = [
        vs["vanilla_dqn"]["eval_ep_len_iqr"] / 2,
        vs["double_dqn"]["eval_ep_len_iqr"] / 2,
        tc["sarsa_tuned_ep_len_iqr"] / 2,
        tc["qlearning_tuned_ep_len_iqr"] / 2,
    ]
    colors = [
        _DQN_COLORS["vanilla_dqn"],
        _DQN_COLORS["double_dqn"],
        _MF_COLORS["sarsa"],
        _MF_COLORS["qlearning"],
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(methods))
    bars = ax.bar(
        x,
        vals,
        0.5,
        color=colors,
        yerr=errs,
        capsize=5,
        error_kw={"elinewidth": 1.2, "ecolor": "#555555"},
    )
    finite_vals = [v for v in vals if v == v]
    max_val = max(finite_vals) if finite_vals else 0
    for bar, val in zip(bars, vals):
        if val == val:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_val * 0.02,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.axhline(
        500,
        color=REFERENCE_COLORS["threshold"],
        linewidth=1,
        linestyle="--",
        label="Max (500)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel(
        "Mean episode length ± IQR/2\n(DQN: greedy eval · tabular: tuned eval)"
    )
    ax.set_title("CartPole Final Performance — DQN vs Tabular Methods")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = fig_dir / "cartpole_dqn_vs_tabular.png"
    fig.savefig(out, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out
