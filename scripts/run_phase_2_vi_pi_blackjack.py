"""Phase 2 — Model-Based: VI & PI on Blackjack.

Outputs:
  artifacts/metrics/phase2_vi_pi_blackjack/
    vi_convergence.csv, pi_convergence.csv
    policy_eval_per_seed.csv, policy_eval_aggregate.csv, summary.csv
    hp_validation.csv
  artifacts/figures/phase2_vi_pi_blackjack/
    blackjack_convergence.png
    blackjack_vi_policy_heatmap.png
    blackjack_vi_value_heatmap.png
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
    BJ_EVAL_EPISODES_HP,
    BJ_EVAL_EPISODES_MAIN,
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
from src.utils.plotting import (
    decode_bj_grids,
    plot_bj_convergence,
    plot_bj_policy_map,
    plot_bj_value_surface,
)

logger = configure_logger("phase2")

PHASE_DIR = "phase2_vi_pi_blackjack"


# ── Figures ───────────────────────────────────────────────────────────────────


def _save_figures(
    metrics_dir: Path,
    metadata: dict,
    V_vi: np.ndarray,
    policy_vi: np.ndarray,
    n_states: int,
) -> None:
    fig_dir = FIGURES_DIR / PHASE_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== Phase 2 Figures ===")

    hard_policy, soft_policy, hard_V, soft_V = decode_bj_grids(
        policy_vi, V_vi, n_states
    )
    out = plot_bj_convergence(metrics_dir, metadata, fig_dir)
    logger.info("Saved → %s", out)
    out = plot_bj_policy_map(hard_policy, soft_policy, fig_dir)
    logger.info("Saved → %s", out)
    out = plot_bj_value_surface(hard_V, soft_V, fig_dir)
    logger.info("Saved → %s", out)


# ── Hyperparameter validation ─────────────────────────────────────────────────


def _hp_sweep_blackjack(T: np.ndarray, R: np.ndarray) -> list[dict]:
    """Sweep gamma and delta for VI and PI on Blackjack.

    Gamma sweep: vary gamma in VI_PI_HP_GAMMA_VALUES, hold delta at reference.
    Delta sweep: vary delta in VI_PI_HP_DELTA_VALUES, hold gamma at 0.99.

    Returns list of row dicts for hp_validation.csv.
    """
    rows: list[dict] = []

    def _eval(policy):
        results = eval_blackjack_policy(
            policy, seeds=SEEDS, n_episodes=BJ_EVAL_EPISODES_HP
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

    # ── delta sweep (gamma fixed at reference) ────────────────────────────────
    ref_gamma = VI_GAMMA
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
    logger.info(
        "=== Evaluating policies (5 seeds × %d episodes) ===",
        BJ_EVAL_EPISODES_MAIN,
    )
    vi_eval = eval_blackjack_policy(
        policy_vi, seeds=SEEDS, n_episodes=BJ_EVAL_EPISODES_MAIN
    )
    vi_returns = [r for _, r in vi_eval]
    vi_mean = float(np.mean(vi_returns))
    vi_iqr = float(np.percentile(vi_returns, 75) - np.percentile(vi_returns, 25))
    logger.info("VI  eval: mean_return=%.4f, IQR=%.4f", vi_mean, vi_iqr)

    pi_eval = eval_blackjack_policy(
        policy_pi, seeds=SEEDS, n_episodes=BJ_EVAL_EPISODES_MAIN
    )
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

    # ── Checkpoint JSON ───────────────────────────────────────────────────────
    # Includes plot-relevant run settings so figures can be regenerated from
    # artifacts without re-running the experiment.
    checkpoint = {
        "vi": {
            "gamma": VI_GAMMA,
            "delta": VI_DELTA,
            "convergence_iter": vi_iters,
            "wall_clock_s": round(vi_wall, 3),
            "final_delta_v": round(vi_final_dv, 10),
            "mean_eval_return": round(vi_mean, 4),
            "eval_return_iqr": round(vi_iqr, 4),
            "eval_episodes": BJ_EVAL_EPISODES_MAIN,
        },
        "pi": {
            "gamma": PI_GAMMA,
            "delta": PI_DELTA,
            "iterations": pi_iters,
            "stable_iter": pi_iters,
            "wall_clock_s": round(pi_wall, 3),
            "final_delta_v": round(pi_final_dv, 10),
            "policy_changes_at_convergence": trace_pi[-1]["policy_changes"],
            "stop_reason": trace_pi[-1].get("stop_reason", "policy_stable"),
            "mean_eval_return": round(pi_mean, 4),
            "eval_return_iqr": round(pi_iqr, 4),
            "policy_match_vi": round(agreement, 4),
            "eval_episodes": BJ_EVAL_EPISODES_MAIN,
        },
    }
    hp_vi_gamma = hp_df[(hp_df.algorithm == "VI") & (hp_df.sweep_param == "gamma")]
    hp_pi_gamma = hp_df[(hp_df.algorithm == "PI") & (hp_df.sweep_param == "gamma")]
    checkpoint["hp_validation"] = {
        "validated_hyperparameters": ["gamma", "delta"],
        "eval_episodes_per_setting": BJ_EVAL_EPISODES_HP,
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

    # ── Figures (read from saved CSVs + metadata) ─────────────────────────────
    _save_figures(metrics_dir, checkpoint, V_vi, policy_vi, n_states)

    logger.info("=== Phase 2 Summary ===")
    logger.info("VI: %d iters, %.3fs, mean_return=%.4f", vi_iters, vi_wall, vi_mean)
    logger.info("PI: %d iters, %.3fs, mean_return=%.4f", pi_iters, pi_wall, pi_mean)
    logger.info("Policy agreement VI vs PI: %.1f%%", agreement * 100)


if __name__ == "__main__":
    run()
