"""Phase 7 (EC) — DQN and Double DQN on CartPole.

Lifecycle:
    run()               → train 5 seeds × 2 variants, evaluate greedy policies,
                          write CSVs + phase7.json
    visualize(path)     → reload checkpoint, render 2 figures

Two variants:
  - Vanilla DQN (Mnih et al., 2015): experience replay + target network
  - Double DQN (van Hasselt et al., 2016): online net selects action,
    target net evaluates Q-value (reduces maximisation bias)

Comparison axes:
  1. cartpole_dqn_vs_double_dqn.png  — learning curves (mean ep-len ± IQR), 5 seeds
  2. cartpole_dqn_vs_tabular.png     — final performance bar chart vs tabular methods
                                       (DQN: greedy eval; tabular: Phase 5 eval metric)

Training is embarrassingly parallel — all (variant, seed) jobs run concurrently.

Upstream inputs:
    artifacts/metadata/phase5.json   — tabular SARSA / Q-Learning tuned performance

Outputs:
    artifacts/metrics/phase7_dqn_ec/
        dqn_learning_curves.csv
        dqn_eval_per_seed.csv
    artifacts/figures/phase7_dqn_ec/
        cartpole_dqn_vs_double_dqn.png
        cartpole_dqn_vs_tabular.png
    artifacts/metadata/phase7.json

Usage:
    make phase7
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from src.algorithms.dqn import DQNConfig, evaluate_dqn_greedy, run_dqn
from src.config import (
    CP_N_ACTIONS,
    DQN_BATCH_SIZE,
    DQN_EVAL_EPISODES,
    DQN_EVAL_SEED_OFFSET,
    DQN_EPS_DECAY_STEPS,
    DQN_EPS_END,
    DQN_EPS_START,
    DQN_GAMMA,
    DQN_HIDDEN_DIM,
    DQN_LR,
    DQN_MAX_GRAD_NORM,
    DQN_REPLAY_SIZE,
    DQN_TARGET_UPDATE_STEPS,
    DQN_TRAIN_EPISODES,
    DQN_TRAIN_START,
    DQN_UPDATE_EVERY,
    PHASE7_MAX_WORKERS,
    SEEDS,
)
from src.utils.phase_artifacts import (
    SCHEMA_VERSION,
    load_checkpoint_json,
    resolve_phase_paths,
    write_checkpoint_json,
)
from src.utils.plotting import plot_p7_dqn_vs_double_dqn, plot_p7_dqn_vs_tabular

_PHASE_ID = "phase7"
_SLUG = "dqn_ec"
_VARIANTS = ["vanilla_dqn", "double_dqn"]


def _build_config() -> DQNConfig:
    return DQNConfig(
        obs_dim=4,
        hidden_dim=DQN_HIDDEN_DIM,
        n_actions=CP_N_ACTIONS,
        replay_size=DQN_REPLAY_SIZE,
        batch_size=DQN_BATCH_SIZE,
        train_start=DQN_TRAIN_START,
        update_every=DQN_UPDATE_EVERY,
        target_update_steps=DQN_TARGET_UPDATE_STEPS,
        lr=DQN_LR,
        gamma=DQN_GAMMA,
        max_grad_norm=DQN_MAX_GRAD_NORM,
        eps_start=DQN_EPS_START,
        eps_end=DQN_EPS_END,
        eps_decay_steps=DQN_EPS_DECAY_STEPS,
    )


# ── Top-level worker (must be picklable — no logger, no shared state) ─────────


def _run_phase7_job(job: dict) -> dict:
    """Worker: train one (variant, seed) and evaluate the greedy policy.

    Must be a top-level function so ProcessPoolExecutor can pickle it on macOS
    (spawn start method).  No logger; returns a plain dict.

    Args:
        job: Plain dict with keys:
            variant, seed, config_fields, n_episodes, n_eval_episodes.

    Returns:
        Plain dict with keys:
            variant, seed, wm_trace, all_ep_lens, eval_ep_lens,
            final_window_return, convergence_episode, train_wall_clock_s.
    """
    import gymnasium as gym
    from src.algorithms.dqn import DQNConfig

    variant: str = job["variant"]
    seed: int = job["seed"]
    n_episodes: int = job["n_episodes"]
    n_eval: int = job["n_eval_episodes"]
    eval_seed_offset: int = job["eval_seed_offset"]

    config = DQNConfig(**job["config_fields"])
    double = variant == "double_dqn"

    env = gym.make("CartPole-v1")
    trace, net = run_dqn(
        env,
        config,
        n_episodes=n_episodes,
        seed=seed,
        double_dqn=double,
        log_interval=n_episodes + 1,  # suppress worker-side logging
    )
    env.close()
    train_wall = trace[-1]["wall_clock_s"] if trace else 0.0

    eval_ep_lens = evaluate_dqn_greedy(net, config, n_eval, seed + eval_seed_offset)

    # Keep only window records to reduce IPC payload
    wm_trace = [r for r in trace if r["window_mean"] is not None]

    last_ep = trace[-1]["episode"] if trace else 0
    conv_ep: int | None = last_ep if last_ep < n_episodes else None

    return {
        "variant": variant,
        "seed": seed,
        "wm_trace": wm_trace,
        "all_ep_lens": [r["ep_len"] for r in trace],
        "eval_ep_lens": eval_ep_lens,
        "final_window_return": wm_trace[-1]["window_mean"]
        if wm_trace
        else float("nan"),
        "convergence_episode": conv_ep,
        "train_wall_clock_s": train_wall,
    }


# ── Aggregation helpers ───────────────────────────────────────────────────────


def _aggregate_curves(
    per_seed_wm_traces: list[list[dict]],
    convergence_window: int,
) -> dict:
    """Aggregate per-seed window_mean curves → mean / q25 / q75.

    Seeds that converge early are padded with their final window_mean so all
    curves share a common episode axis.
    """
    seed_curves: list[dict[int, float]] = []
    for wm_trace in per_seed_wm_traces:
        seed_curves.append({r["episode"]: r["window_mean"] for r in wm_trace})

    max_ep = max(
        (max(d.keys()) for d in seed_curves if d),
        default=convergence_window,
    )
    episodes = list(range(convergence_window, max_ep + 1, convergence_window))

    means, q25s, q75s = [], [], []
    for ep in episodes:
        vals = []
        for d in seed_curves:
            if ep in d:
                vals.append(d[ep])
            else:
                prev = [wm for e, wm in d.items() if e <= ep]
                if prev:
                    vals.append(prev[-1])
        if vals:
            means.append(float(np.mean(vals)))
            q25s.append(float(np.percentile(vals, 25)))
            q75s.append(float(np.percentile(vals, 75)))
        else:
            means.append(float("nan"))
            q25s.append(float("nan"))
            q75s.append(float("nan"))

    return {"episodes": episodes, "mean": means, "q25": q25s, "q75": q75s}


# ── Main lifecycle ────────────────────────────────────────────────────────────


def run() -> Path:
    """Train DQN and Double DQN, evaluate policies, write artifacts."""
    from src.utils.logger import configure_logger

    paths = resolve_phase_paths(_PHASE_ID, _SLUG)
    paths.makedirs()
    log = configure_logger("phase7", log_dir=paths.logs_dir)

    log.info("=== Phase 7 DQN Extra Credit ===")
    config = _build_config()

    # ── Load Phase 5 tabular comparison baseline ──────────────────────────────
    p5 = load_checkpoint_json(Path("artifacts/metadata/phase5.json"))
    tabular_comparison = {
        "sarsa_tuned_mean_ep_len": p5["summary"]["tuned"]["sarsa"]["mean_episode_len"],
        "sarsa_tuned_ep_len_iqr": p5["summary"]["tuned"]["sarsa"]["iqr_episode_len"],
        "qlearning_tuned_mean_ep_len": p5["summary"]["tuned"]["qlearning"][
            "mean_episode_len"
        ],
        "qlearning_tuned_ep_len_iqr": p5["summary"]["tuned"]["qlearning"][
            "iqr_episode_len"
        ],
    }
    log.info(
        "Tabular baseline — SARSA tuned: %.1f, Q-Learning tuned: %.1f",
        tabular_comparison["sarsa_tuned_mean_ep_len"],
        tabular_comparison["qlearning_tuned_mean_ep_len"],
    )

    # ── Build jobs ────────────────────────────────────────────────────────────
    config_fields = {f: getattr(config, f) for f in config.__dataclass_fields__}
    jobs = [
        {
            "variant": variant,
            "seed": seed,
            "config_fields": config_fields,
            "n_episodes": DQN_TRAIN_EPISODES,
            "n_eval_episodes": DQN_EVAL_EPISODES,
            "eval_seed_offset": DQN_EVAL_SEED_OFFSET,
        }
        for variant in _VARIANTS
        for seed in SEEDS
    ]
    n_jobs = len(jobs)

    # ── Determine worker count ────────────────────────────────────────────────
    if PHASE7_MAX_WORKERS is None:
        cpu = os.cpu_count() or 1
        n_workers = min(n_jobs, max(1, cpu - 1))
    else:
        n_workers = PHASE7_MAX_WORKERS

    log.info(
        "Dispatching %d jobs (%d variants × %d seeds × %d episodes) → %d workers",
        n_jobs,
        len(_VARIANTS),
        len(SEEDS),
        DQN_TRAIN_EPISODES,
        n_workers,
    )

    # ── Execute (serial fallback for n_workers == 1) ──────────────────────────
    results: list[dict] = []
    t_dispatch = time.perf_counter()

    if n_workers == 1:
        for job in jobs:
            res = _run_phase7_job(job)
            log.info(
                "  %s seed=%d  final_window=%.1f  eval_mean=%.1f  conv_ep=%s  t=%.1fs",
                res["variant"],
                res["seed"],
                res["final_window_return"],
                float(np.mean(res["eval_ep_lens"])),
                str(res["convergence_episode"])
                if res["convergence_episode"]
                else "none",
                res["train_wall_clock_s"],
            )
            results.append(res)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_phase7_job, job): job for job in jobs}
            for fut in as_completed(futures):
                res = fut.result()
                log.info(
                    "  %s seed=%d  final_window=%.1f  eval_mean=%.1f  conv_ep=%s  t=%.1fs",
                    res["variant"],
                    res["seed"],
                    res["final_window_return"],
                    float(np.mean(res["eval_ep_lens"])),
                    str(res["convergence_episode"])
                    if res["convergence_episode"]
                    else "none",
                    res["train_wall_clock_s"],
                )
                results.append(res)

    log.info("All jobs done (wall-clock: %.1fs)", time.perf_counter() - t_dispatch)

    # ── Build CSV rows ────────────────────────────────────────────────────────
    curve_rows: list[dict] = []
    eval_rows: list[dict] = []

    for res in results:
        variant = res["variant"]
        seed = res["seed"]

        for r in res["wm_trace"]:
            curve_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "episode": r["episode"],
                    "window_mean": r["window_mean"],
                }
            )

        eval_rows.append(
            {
                "variant": variant,
                "seed": seed,
                "mean_episode_len": round(float(np.mean(res["all_ep_lens"])), 2),
                "final_window_return": round(res["final_window_return"], 2),
                "mean_eval_ep_len": round(float(np.mean(res["eval_ep_lens"])), 2),
                "convergence_episode": res["convergence_episode"],
                "train_wall_clock_s": round(res["train_wall_clock_s"], 2),
            }
        )

    curves_csv = paths.metrics_dir / "dqn_learning_curves.csv"
    pd.DataFrame(curve_rows).to_csv(curves_csv, index=False)
    log.info("Learning curves → %s", curves_csv)

    eval_csv = paths.metrics_dir / "dqn_eval_per_seed.csv"
    pd.DataFrame(eval_rows).to_csv(eval_csv, index=False)
    log.info("Eval per-seed → %s", eval_csv)

    # ── Assemble summary ──────────────────────────────────────────────────────
    variants_summary: dict = {}
    learning_curves: dict = {}

    for variant in _VARIANTS:
        rows = [r for r in eval_rows if r["variant"] == variant]
        final_wm_vals = [r["final_window_return"] for r in rows]
        eval_vals = [r["mean_eval_ep_len"] for r in rows]
        conv_eps = [
            r["convergence_episode"]
            for r in rows
            if r["convergence_episode"] is not None
        ]

        variants_summary[variant] = {
            # Training-window metric (ε-greedy): kept for transparency
            "mean_final_ep_len": round(float(np.mean(final_wm_vals)), 2),
            "final_ep_len_iqr": round(
                float(
                    np.percentile(final_wm_vals, 75) - np.percentile(final_wm_vals, 25)
                ),
                2,
            ),
            # Greedy evaluation metric: comparable to Phase 5 eval
            "mean_eval_ep_len": round(float(np.mean(eval_vals)), 2),
            "eval_ep_len_iqr": round(
                float(np.percentile(eval_vals, 75) - np.percentile(eval_vals, 25)), 2
            ),
            "mean_convergence_episode": round(float(np.mean(conv_eps)), 1)
            if conv_eps
            else None,
            "convergence_episode_iqr": round(
                float(np.percentile(conv_eps, 75) - np.percentile(conv_eps, 25)), 1
            )
            if len(conv_eps) >= 2
            else None,
        }

        per_seed_wm_traces = [
            res["wm_trace"] for res in results if res["variant"] == variant
        ]
        learning_curves[variant] = _aggregate_curves(
            per_seed_wm_traces, config.convergence_window
        )

        log.info(
            "%s — mean_final=%.1f  mean_eval=%.1f  eval_iqr=%.1f  conv_ep=%s",
            variant,
            variants_summary[variant]["mean_final_ep_len"],
            variants_summary[variant]["mean_eval_ep_len"],
            variants_summary[variant]["eval_ep_len_iqr"],
            str(variants_summary[variant]["mean_convergence_episode"]),
        )

    # ── Write checkpoint ──────────────────────────────────────────────────────
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": _PHASE_ID,
        "slug": _SLUG,
        "upstream_inputs": ["artifacts/metadata/phase5.json"],
        "outputs": {
            "figures_dir": str(paths.figures_dir),
            "metrics_dir": str(paths.metrics_dir),
        },
        "config_snapshot": {
            "n_episodes": DQN_TRAIN_EPISODES,
            "n_eval_episodes": DQN_EVAL_EPISODES,
            "hidden_dim": DQN_HIDDEN_DIM,
            "lr": DQN_LR,
            "gamma": DQN_GAMMA,
            "replay_size": DQN_REPLAY_SIZE,
            "batch_size": DQN_BATCH_SIZE,
            "target_update_steps": DQN_TARGET_UPDATE_STEPS,
            "eps_decay_steps": DQN_EPS_DECAY_STEPS,
            "max_grad_norm": DQN_MAX_GRAD_NORM,
            "n_workers": n_workers,
        },
        "summary": {
            "variants": variants_summary,
            "learning_curves": learning_curves,
            "tabular_comparison": tabular_comparison,
        },
    }
    write_checkpoint_json(checkpoint, paths.checkpoint_path)
    log.info("Phase 7 checkpoint → %s", paths.checkpoint_path)

    return paths.checkpoint_path


def visualize(checkpoint_path: Path) -> list[Path]:
    """Render Phase 7 figures from saved checkpoint. No live computation."""
    from src.utils.logger import configure_logger

    log = configure_logger("phase7")
    checkpoint = load_checkpoint_json(checkpoint_path)
    figures_dir = Path(checkpoint["outputs"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    log.info("Rendering Phase 7 figures → %s", figures_dir)
    figs: list[Path] = []

    figs.append(plot_p7_dqn_vs_double_dqn(checkpoint["summary"], figures_dir))
    log.info("  DQN vs Double DQN → %s", figs[-1])

    figs.append(plot_p7_dqn_vs_tabular(checkpoint["summary"], figures_dir))
    log.info("  DQN vs tabular → %s", figs[-1])

    log.info("Phase 7 figures done (%d files)", len(figs))
    return figs


if __name__ == "__main__":
    checkpoint_path = run()
    visualize(checkpoint_path)
