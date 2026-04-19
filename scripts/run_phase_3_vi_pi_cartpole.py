"""Phase 3 — Model-Based: VI & PI on CartPole (discretization ablation study).

Lifecycle:
    run()               → build models, run algorithms, write all artifacts, return checkpoint path
    visualize(path)     → reload from disk only, render all figures

Outputs:
  artifacts/metrics/phase3_vi_pi_cartpole/
    vi_convergence.csv, pi_convergence.csv
    policy_eval_per_seed.csv, policy_eval_aggregate.csv
    policy_agreement.csv, discretization_study.csv
    hp_validation.csv
    plot_cp_grids.npz          ← plot-support: per-grid policy arrays
  artifacts/figures/phase3_vi_pi_cartpole/
    cartpole_vi_convergence.png              ← VI ΔV vs iteration (representative curve)
    cartpole_pi_convergence.png              ← PI policy changes vs iteration, per grid
    cartpole_mean_episode_length_by_grid.png ← grouped bar: mean episode length, VI vs PI
    cartpole_wall_clock_by_grid.png          ← grouped bar: planning wall-clock, VI vs PI
    cartpole_model_coverage_by_grid.png      ← coverage % by grid
    cartpole_policy_slice.png                ← optional: (θ, θ̇) decision-boundary slice
  artifacts/metadata/phase3.json
  artifacts/logs/phase3.log

Usage:
  uv run python scripts/run_phase_3_vi_pi_cartpole.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.algorithms import eval_cartpole_policy, run_pi, run_vi
from src.config import (
    CARTPOLE_GRID_CONFIGS,
    CARTPOLE_GRID_NAMES,
    CARTPOLE_MODEL_MIN_VISITS,
    CARTPOLE_MODEL_ROLLOUT_STEPS,
    CARTPOLE_MODEL_SEED,
    CP_EVAL_EPISODES_HP,
    CP_EVAL_EPISODES_MAIN,
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
from src.utils.phase_artifacts import (
    SCHEMA_VERSION,
    resolve_phase_paths,
    load_checkpoint_json,
    validate_required_outputs,
    write_checkpoint_json,
)
from src.utils.plotting import (
    plot_cp_mean_episode_length,
    plot_cp_model_coverage,
    plot_cp_pi_convergence,
    plot_cp_policy_slice,
    plot_cp_vi_convergence,
    plot_cp_wall_clock,
)

logger = configure_logger("phase3")

_PHASE_ID = "phase3"
_SLUG = "vi_pi_cartpole"


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

    grid_rollout_steps = cfg.get("rollout_steps", CARTPOLE_MODEL_ROLLOUT_STEPS)
    logger.info("Grid '%s': using rollout_steps=%d", grid_name, grid_rollout_steps)

    cp_model = build_cartpole_model(
        discretizer=disc,
        rollout_steps=grid_rollout_steps,
        min_visits=CARTPOLE_MODEL_MIN_VISITS,
        seed=CARTPOLE_MODEL_SEED,
        logger=logger,
    )
    T, R = cp_model["T"], cp_model["R"]

    logger.info("Running VI on grid '%s'...", grid_name)
    t0 = time.perf_counter()
    V_vi, policy_vi, trace_vi = run_vi(
        T, R, VI_GAMMA, VI_DELTA, m_consec=VI_PI_CONSEC_SWEEPS, logger=logger
    )
    vi_wall = time.perf_counter() - t0
    logger.info("VI: %d iters, %.3fs", len(trace_vi), vi_wall)

    logger.info("Running PI on grid '%s'...", grid_name)
    t0 = time.perf_counter()
    _, policy_pi, trace_pi = run_pi(
        T, R, PI_GAMMA, PI_DELTA, m_consec=VI_PI_CONSEC_SWEEPS, logger=logger
    )
    pi_wall = time.perf_counter() - t0
    logger.info("PI: %d iters, %.3fs", len(trace_pi), pi_wall)

    n = disc.n_states
    agreement = float((policy_vi[:n] == policy_pi[:n]).mean())
    logger.info("Grid '%s' VI vs PI agreement: %.1f%%", grid_name, agreement * 100)

    logger.info(
        "Evaluating VI on '%s' (%d seeds × %d eps)...",
        grid_name,
        len(SEEDS),
        CP_EVAL_EPISODES_MAIN,
    )
    vi_eval = eval_cartpole_policy(
        policy_vi, disc, seeds=SEEDS, n_episodes=CP_EVAL_EPISODES_MAIN
    )
    vi_lens = [ep_len for _, ep_len in vi_eval]
    vi_mean = float(np.mean(vi_lens))
    vi_iqr = float(np.percentile(vi_lens, 75) - np.percentile(vi_lens, 25))

    logger.info("Evaluating PI on '%s'...", grid_name)
    pi_eval = eval_cartpole_policy(
        policy_pi, disc, seeds=SEEDS, n_episodes=CP_EVAL_EPISODES_MAIN
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
        # T and R kept for HP sweep — not persisted beyond run()
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


# ── HP sweep ──────────────────────────────────────────────────────────────────


def _hp_sweep_cartpole(
    T: np.ndarray,
    R: np.ndarray,
    disc: CartPoleDiscretizer,
    grid_name: str = "default",
) -> list[dict]:
    """Sweep gamma and delta for VI and PI on the given CartPole grid."""
    rows: list[dict] = []

    def _eval(policy):
        results = eval_cartpole_policy(
            policy, disc, seeds=SEEDS, n_episodes=CP_EVAL_EPISODES_HP
        )
        lens = [ep_len for _, ep_len in results]
        return float(np.mean(lens)), float(
            np.percentile(lens, 75) - np.percentile(lens, 25)
        )

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

    ref_gamma = VI_GAMMA
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


# ── Lifecycle ─────────────────────────────────────────────────────────────────


def run() -> Path:
    """Execute Phase 3 computation, write all artifacts, return checkpoint path."""
    paths = resolve_phase_paths(_PHASE_ID, _SLUG)
    paths.makedirs()

    logger.info("=== Phase 3: VI & PI on CartPole (grid ablation) ===")
    grid_results = {name: _run_grid(name) for name in CARTPOLE_GRID_NAMES}

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

    pd.DataFrame(vi_conv).to_csv(paths.metrics_dir / "vi_convergence.csv", index=False)
    pd.DataFrame(pi_conv).to_csv(paths.metrics_dir / "pi_convergence.csv", index=False)
    pd.DataFrame(eval_per_seed).to_csv(
        paths.metrics_dir / "policy_eval_per_seed.csv", index=False
    )
    pd.DataFrame(eval_aggregate).to_csv(
        paths.metrics_dir / "policy_eval_aggregate.csv", index=False
    )
    pd.DataFrame(policy_agreement_rows).to_csv(
        paths.metrics_dir / "policy_agreement.csv", index=False
    )
    pd.DataFrame(study_rows).to_csv(
        paths.metrics_dir / "discretization_study.csv", index=False
    )
    logger.info("Metrics saved → %s", paths.metrics_dir)

    # ── HP sweep (default grid) ───────────────────────────────────────────────
    logger.info("=== VI/PI Hyperparameter Validation Sweep (default grid) ===")
    default_res = grid_results["default"]
    hp_rows = _hp_sweep_cartpole(
        default_res["T"], default_res["R"], default_res["disc"]
    )
    hp_df = pd.DataFrame(hp_rows)
    hp_df.to_csv(paths.metrics_dir / "hp_validation.csv", index=False)
    logger.info("HP validation saved → %s", paths.metrics_dir / "hp_validation.csv")

    # ── Plot-support NPZ ──────────────────────────────────────────────────────
    # Persist per-grid policies so visualize() can render the policy-slice
    # figure without re-running model building or planning.
    npz_data = {}
    for grid_name in CARTPOLE_GRID_NAMES:
        res = grid_results[grid_name]
        npz_data[f"policy_vi_{grid_name}"] = res["policy_vi"]
        npz_data[f"policy_pi_{grid_name}"] = res["policy_pi"]
    npz_path = paths.metrics_dir / "plot_cp_grids.npz"
    np.savez_compressed(npz_path, **npz_data)
    logger.info("Plot-support NPZ saved → %s", npz_path)

    # ── Checkpoint JSON ───────────────────────────────────────────────────────
    hp_vi_gamma = hp_df[(hp_df.algorithm == "VI") & (hp_df.sweep_param == "gamma")]
    hp_pi_gamma = hp_df[(hp_df.algorithm == "PI") & (hp_df.sweep_param == "gamma")]

    grid_summary = {
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

    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": _PHASE_ID,
        "slug": _SLUG,
        # Intentionally empty: Phase 3 builds all three grids independently
        # (each grid constructs its own CartPole model via fresh rollouts).
        # It does NOT depend on the Phase 1 default-grid model artifact.
        "upstream_inputs": [],
        "outputs": {
            "metrics_dir": str(paths.metrics_dir),
            "figures_dir": str(paths.figures_dir),
            "plot_support": [str(npz_path)],
        },
        "config_snapshot": {
            "vi_gamma": VI_GAMMA,
            "vi_delta": VI_DELTA,
            "pi_gamma": PI_GAMMA,
            "pi_delta": PI_DELTA,
            "vi_pi_consec_sweeps": VI_PI_CONSEC_SWEEPS,
            "seeds": SEEDS,
            "eval_episodes_main": CP_EVAL_EPISODES_MAIN,
            "eval_episodes_hp": CP_EVAL_EPISODES_HP,
            "grid_names": CARTPOLE_GRID_NAMES,
            # Serialise tuples → lists for JSON compatibility.
            # Saved here so visualize() can reconstruct discretizers from the
            # checkpoint rather than from the current source config — rerenders
            # of old checkpoints stay reproducible even if configs change later.
            "grid_configs": {
                name: {
                    k: list(v) if isinstance(v, tuple) else v
                    for k, v in CARTPOLE_GRID_CONFIGS[name].items()
                }
                for name in CARTPOLE_GRID_NAMES
            },
        },
        "summary": {
            **grid_summary,
            "hp_validation": {
                "grid": "default",
                "validated_hyperparameters": ["gamma", "delta"],
                "eval_episodes_per_setting": CP_EVAL_EPISODES_HP,
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
            },
        },
    }

    write_checkpoint_json(checkpoint, paths.checkpoint_path)
    logger.info("Phase 3 checkpoint saved → %s", paths.checkpoint_path)

    # ── Validate all required outputs exist ───────────────────────────────────
    validate_required_outputs(
        [
            paths.metrics_dir / "vi_convergence.csv",
            paths.metrics_dir / "pi_convergence.csv",
            paths.metrics_dir / "policy_eval_per_seed.csv",
            paths.metrics_dir / "policy_eval_aggregate.csv",
            paths.metrics_dir / "policy_agreement.csv",
            paths.metrics_dir / "discretization_study.csv",
            paths.metrics_dir / "hp_validation.csv",
            npz_path,
            paths.checkpoint_path,
        ]
    )

    logger.info("=== Phase 3 Summary ===")
    for name in CARTPOLE_GRID_NAMES:
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

    return paths.checkpoint_path


def visualize(checkpoint_path: Path) -> list[Path]:
    """Render Phase 3 figures from saved artifacts. No live computation."""
    checkpoint = load_checkpoint_json(checkpoint_path)

    metrics_dir = Path(checkpoint["outputs"]["metrics_dir"])
    figures_dir = Path(checkpoint["outputs"]["figures_dir"])
    npz_path = Path(checkpoint["outputs"]["plot_support"][0])
    summary = checkpoint["summary"]
    cfg_snap = checkpoint["config_snapshot"]

    figures_dir.mkdir(parents=True, exist_ok=True)

    grid_names: list[str] = cfg_snap["grid_names"]
    grid_configs = cfg_snap["grid_configs"]

    # Validate checkpoint integrity before touching any plotting code.
    grids_npz = np.load(npz_path)
    missing: list[str] = []
    for g in grid_names:
        if g not in summary:
            missing.append(f"summary[{g!r}]")
        if g not in grid_configs:
            missing.append(f"grid_configs[{g!r}]")
        for algo in ("vi", "pi"):
            key = f"policy_{algo}_{g}"
            if key not in grids_npz:
                missing.append(f"NPZ key {key!r}")
    if missing:
        raise KeyError(
            "Checkpoint/NPZ integrity check failed — missing keys:\n"
            + "\n".join(f"  {m}" for m in missing)
        )

    grid_n_states = {g: summary[g]["n_states"] for g in grid_names}

    logger.info("=== Phase 3 Figures ===")
    figs: list[Path] = []

    out = plot_cp_vi_convergence(metrics_dir, grid_n_states, grid_names, figures_dir)
    logger.info("Saved → %s", out)
    figs.append(out)

    out = plot_cp_pi_convergence(metrics_dir, grid_n_states, grid_names, figures_dir)
    logger.info("Saved → %s", out)
    figs.append(out)

    out = plot_cp_mean_episode_length(metrics_dir, grid_names, figures_dir)
    logger.info("Saved → %s", out)
    figs.append(out)

    out = plot_cp_wall_clock(metrics_dir, grid_names, figures_dir)
    logger.info("Saved → %s", out)
    figs.append(out)

    out = plot_cp_model_coverage(metrics_dir, grid_names, figures_dir)
    logger.info("Saved → %s", out)
    figs.append(out)

    out = plot_cp_policy_slice(
        npz_path, grid_n_states, grid_configs, grid_names, figures_dir
    )
    logger.info("Saved → %s", out)
    figs.append(out)

    return figs


if __name__ == "__main__":
    checkpoint_path = run()
    visualize(checkpoint_path)
