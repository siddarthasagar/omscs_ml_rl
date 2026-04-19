"""Phase 5 — Model-Free: SARSA & Q-Learning on CartPole.

Lifecycle:
    run()               → execute computation, write all artifacts, return checkpoint path
    visualize(path)     → reload from disk only, render all figures

HP search: 3-stage progressive narrowing, each stage scored over all SEEDS.
Final training: parallel via ProcessPoolExecutor, two regimes per algorithm.
  - controlled: both algorithms use the same fixed baseline schedule (fair comparison).
  - tuned:      each algorithm uses its own HP-search winner.
Discretization study: tuned HP winner per algorithm × coarse/default/fine grids × 5 seeds.

Outputs:
  artifacts/metrics/phase5_model_free_cartpole/
    mf_hp_search.csv            ← per-config mean_episode_len (avg over SEEDS) for all stages
    mf_learning_curves.csv      ← window-mean returns per (algorithm, seed, regime)
    mf_eval_per_seed.csv        ← mean_episode_len + wall-clock per (algorithm, seed, regime)
    mf_eval_summary.csv         ← mean / std / IQR across seeds per (algorithm, regime, metric)
    mf_discretization.csv       ← grid, algorithm, seed, final_mean_len, convergence_episode
  artifacts/figures/phase5_model_free_cartpole/
    cartpole_mf_learning_curves.png
    cartpole_mf_comparison.png
    cartpole_mf_hp_sensitivity.png
    cartpole_mf_discretization.png
  artifacts/metadata/phase5.json
  artifacts/logs/phase5.log

Usage:
  make phase5
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

from src.algorithms import QLearningConfig, SarsaConfig, run_q_learning, run_sarsa
from src.algorithms.model_free_utils import check_convergence, greedy_action
from src.config import (
    CARTPOLE_GRID_CONFIGS,
    CARTPOLE_GRID_NAMES,
    CP_BASELINE_ALPHA_DECAY_STEPS,
    CP_BASELINE_ALPHA_END,
    CP_BASELINE_ALPHA_START,
    CP_BASELINE_EPS_DECAY_STEPS,
    CP_BASELINE_EPS_END,
    CP_BASELINE_EPS_START,
    CP_BASELINE_GAMMA,
    CP_DISC_TRAIN_EPISODES,
    CP_EVAL_EPISODES_HP,
    CP_EVAL_EPISODES_MAIN,
    CP_HP_STAGE1_CONFIGS,
    CP_HP_STAGE1_EPISODES,
    CP_HP_STAGE2_EPISODES,
    CP_HP_STAGE2_TOP_K,
    CP_HP_STAGE3_EPISODES,
    CP_HP_STAGE3_TOP_K,
    CP_N_ACTIONS,
    CP_TRAIN_EPISODES,
    PHASE5_FINAL_TRAIN_MAX_WORKERS,
    RL_CONVERGENCE_DELTA,
    RL_CONVERGENCE_M,
    RL_CONVERGENCE_WINDOW,
    SEEDS,
)
from src.envs.cartpole_discretizer import CartPoleDiscretizer
from src.utils.phase_artifacts import (
    SCHEMA_VERSION,
    load_checkpoint_json,
    resolve_phase_paths,
    validate_required_outputs,
    write_checkpoint_json,
)
from src.utils.plotting import (
    plot_cp_mf_comparison,
    plot_cp_mf_discretization,
    plot_cp_mf_hp_sensitivity,
    plot_cp_mf_learning_curve,
)

# No module-level logger — configure_logger is called inside run() only.
# Workers spawned by ProcessPoolExecutor must not inherit file handlers.
_logger = None  # set by run(); accessed via _get_logger() in HP-search helpers

_PHASE_ID = "phase5"
_SLUG = "model_free_cartpole"

# HP search space (shared across SARSA and Q-Learning)
_HP_ALPHA_STARTS = [0.1, 0.3, 0.5, 0.7, 0.9]
_HP_ALPHA_ENDS = [0.001, 0.01, 0.05]
_HP_ALPHA_DECAY_STEPS = [10_000, 30_000, 50_000, 100_000]
_HP_EPS_DECAY_STEPS = [2_000, 5_000, 10_000, 20_000]
_HP_GAMMA = 0.99

# HP search seed scheme: SEEDS values shifted by config_idx * stride.
_HP_SEED_CONFIG_STRIDE: int = 100_000
_HP_EVAL_SEED_OFFSET: int = 500_000


def _get_logger():
    return _logger


# ── CartPole discrete wrapper ─────────────────────────────────────────────────


class _CartPoleDiscreteWrapper(gym.ObservationWrapper):
    """Wraps CartPole-v1 to return integer state indices via CartPoleDiscretizer.

    Allows run_sarsa / run_q_learning (which call int(obs) for non-tuple
    observations) to work directly with CartPole's continuous observations.
    """

    def __init__(self, discretizer: CartPoleDiscretizer) -> None:
        super().__init__(gym.make("CartPole-v1"))
        self._disc = discretizer
        self.observation_space = gym.spaces.Discrete(discretizer.n_states)

    def observation(self, obs: np.ndarray) -> int:
        return self._disc.obs_to_state(obs)


# ── Config builder ────────────────────────────────────────────────────────────


def _build_mf_config(
    algo_label: str,
    hp: dict,
    *,
    disable_early_stopping: bool = False,
) -> SarsaConfig | QLearningConfig:
    kwargs = dict(
        alpha_start=hp["alpha_start"],
        alpha_end=hp["alpha_end"],
        alpha_decay_steps=hp["alpha_decay_steps"],
        eps_start=hp.get("eps_start", 1.0),
        eps_end=hp.get("eps_end", 0.01),
        eps_decay_steps=hp["eps_decay_steps"],
        gamma=hp.get("gamma", _HP_GAMMA),
        convergence_window=RL_CONVERGENCE_WINDOW,
        convergence_delta=0.0 if disable_early_stopping else RL_CONVERGENCE_DELTA,
        convergence_m=RL_CONVERGENCE_M,
    )
    return SarsaConfig(**kwargs) if algo_label == "sarsa" else QLearningConfig(**kwargs)


# ── Evaluation helper ─────────────────────────────────────────────────────────


def _eval_cp_policy(
    Q: np.ndarray,
    discretizer: CartPoleDiscretizer,
    n_episodes: int,
    seed: int,
) -> dict[str, float]:
    """Greedy rollout of a Q-table on CartPole-v1.

    Returns mean_episode_len (primary ranking signal; higher is better).
    For CartPole, episodic return == episode length since reward=+1 per step.
    """
    rng = np.random.default_rng(seed)
    env = _CartPoleDiscreteWrapper(discretizer)
    total_len = 0
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
    for _ in range(n_episodes):
        state = int(obs)
        done = False
        ep_len = 0
        while not done:
            action = greedy_action(Q, state)
            obs, _reward, terminated, truncated, _ = env.step(action)
            state = int(obs)
            ep_len += 1
            done = terminated or truncated
        total_len += ep_len
        obs, _ = env.reset()
    env.close()
    return {"mean_episode_len": total_len / n_episodes}


# ── Top-level worker (must be picklable — no logger, no shared state) ─────────


def _run_phase5_final_job(job: dict) -> dict:
    """Worker for one final-training job: (algorithm, seed, regime, grid_config).

    Args:
        job: Plain dict with keys:
            algorithm, seed, regime, hp, grid_config,
            train_episodes, eval_episodes, disable_early_stopping.

    Returns:
        Compact result dict with keys:
            algorithm, seed, regime, episodes_run, train_wall_clock_s,
            curve_episode (ndarray), curve_window_mean (ndarray),
            mean_episode_len.
    """
    algo_label: str = job["algorithm"]
    seed: int = job["seed"]
    regime: str = job["regime"]
    hp: dict = job["hp"]
    grid_config: dict | None = job.get("grid_config")
    train_episodes: int = job["train_episodes"]
    eval_episodes: int = job["eval_episodes"]
    disable_es: bool = job.get("disable_early_stopping", True)

    discretizer = CartPoleDiscretizer(grid_config)
    env = _CartPoleDiscreteWrapper(discretizer)
    cfg = _build_mf_config(algo_label, hp, disable_early_stopping=disable_es)

    t_train = time.perf_counter()
    if algo_label == "sarsa":
        Q, trace = run_sarsa(
            env,
            cfg,
            discretizer.n_states,
            CP_N_ACTIONS,
            train_episodes,
            seed=seed,
            log_interval=train_episodes + 1,
        )
    else:
        Q, trace = run_q_learning(
            env,
            cfg,
            discretizer.n_states,
            CP_N_ACTIONS,
            train_episodes,
            seed=seed,
            log_interval=train_episodes + 1,
        )
    env.close()
    train_wall = time.perf_counter() - t_train

    stats = _eval_cp_policy(Q, discretizer, eval_episodes, seed=seed)

    window_records = [
        (r["episode"], r["window_mean"])
        for r in trace
        if r.get("window_mean") is not None
    ]
    episodes_arr = np.array([e for e, _ in window_records], dtype=np.int32)
    wmeans_arr = np.array([w for _, w in window_records], dtype=np.float32)

    return {
        "algorithm": algo_label,
        "seed": seed,
        "regime": regime,
        "episodes_run": len(trace),
        "train_wall_clock_s": round(train_wall, 3),
        "curve_episode": episodes_arr,
        "curve_window_mean": wmeans_arr,
        "mean_episode_len": stats["mean_episode_len"],
    }


def _run_disc_job(job: dict) -> dict:
    """Worker for one discretization-study job: (algorithm, seed, grid_name)."""
    algo_label: str = job["algorithm"]
    seed: int = job["seed"]
    grid_name: str = job["grid_name"]
    hp: dict = job["hp"]
    grid_config: dict = job["grid_config"]
    train_episodes: int = job["train_episodes"]
    eval_episodes: int = job["eval_episodes"]

    discretizer = CartPoleDiscretizer(grid_config)
    env = _CartPoleDiscreteWrapper(discretizer)
    # Early stopping enabled for discretization study (natural plateau per grid).
    cfg = _build_mf_config(algo_label, hp, disable_early_stopping=False)

    t_train = time.perf_counter()
    if algo_label == "sarsa":
        Q, trace = run_sarsa(
            env,
            cfg,
            discretizer.n_states,
            CP_N_ACTIONS,
            train_episodes,
            seed=seed,
            log_interval=train_episodes + 1,
        )
    else:
        Q, trace = run_q_learning(
            env,
            cfg,
            discretizer.n_states,
            CP_N_ACTIONS,
            train_episodes,
            seed=seed,
            log_interval=train_episodes + 1,
        )
    env.close()
    train_wall = time.perf_counter() - t_train

    stats = _eval_cp_policy(Q, discretizer, eval_episodes, seed=seed)

    # Retroactively compute convergence episode.
    wm_list = [
        float(r["window_mean"]) for r in trace if r.get("window_mean") is not None
    ]
    conv_ep: int | None = None
    for k in range(len(wm_list)):
        if check_convergence(
            wm_list[: k + 1],
            window=RL_CONVERGENCE_WINDOW,
            delta=RL_CONVERGENCE_DELTA,
            m_consec=RL_CONVERGENCE_M,
        ):
            window_eps = [
                r["episode"] for r in trace if r.get("window_mean") is not None
            ]
            conv_ep = int(window_eps[k])
            break

    return {
        "algorithm": algo_label,
        "seed": seed,
        "grid_name": grid_name,
        "episodes_run": len(trace),
        "train_wall_clock_s": round(train_wall, 3),
        "mean_episode_len": stats["mean_episode_len"],
        "convergence_episode": conv_ep,
    }


# ── HP Search (multi-seed scoring) ────────────────────────────────────────────


def _make_random_hp_configs(n: int, rng_seed: int) -> list[dict]:
    rng = np.random.default_rng(rng_seed)
    return [
        {
            "alpha_start": float(rng.choice(_HP_ALPHA_STARTS)),
            "alpha_end": float(rng.choice(_HP_ALPHA_ENDS)),
            "alpha_decay_steps": int(rng.choice(_HP_ALPHA_DECAY_STEPS)),
            "eps_decay_steps": int(rng.choice(_HP_EPS_DECAY_STEPS)),
            "gamma": _HP_GAMMA,
        }
        for _ in range(n)
    ]


def _run_hp_stage(
    algo_label: str,
    hp_configs: list[dict],
    n_episodes: int,
    stage: int,
    *,
    grid_config: dict | None = None,
) -> list[dict]:
    """Run one HP search stage; scores each config over all SEEDS.

    Promotion criterion: mean_episode_len (higher is better).
    """
    log = _get_logger()
    discretizer = CartPoleDiscretizer(grid_config)
    rows: list[dict] = []

    for i, hp in enumerate(hp_configs):
        seed_lens: list[float] = []
        for seed in SEEDS:
            train_seed = seed + i * _HP_SEED_CONFIG_STRIDE
            env = _CartPoleDiscreteWrapper(discretizer)
            cfg = _build_mf_config(algo_label, hp)
            if algo_label == "sarsa":
                Q, _ = run_sarsa(
                    env,
                    cfg,
                    discretizer.n_states,
                    CP_N_ACTIONS,
                    n_episodes,
                    seed=train_seed,
                    log_interval=n_episodes + 1,
                )
            else:
                Q, _ = run_q_learning(
                    env,
                    cfg,
                    discretizer.n_states,
                    CP_N_ACTIONS,
                    n_episodes,
                    seed=train_seed,
                    log_interval=n_episodes + 1,
                )
            env.close()
            eval_stats = _eval_cp_policy(
                Q,
                discretizer,
                CP_EVAL_EPISODES_HP,
                seed=train_seed + _HP_EVAL_SEED_OFFSET,
            )
            seed_lens.append(eval_stats["mean_episode_len"])

        mean_len = float(np.mean(seed_lens))
        std_len = float(np.std(seed_lens))
        row = {
            "algorithm": algo_label,
            "stage": stage,
            **hp,
            "mean_episode_len": mean_len,
            "mean_episode_len_std": std_len,
        }
        rows.append(row)
        if log:
            log.info(
                "  [%s] stage%d cfg%d: α=%.2f→%.3f eps_decay=%d  len=%.1f±%.1f",
                algo_label,
                stage,
                i,
                hp["alpha_start"],
                hp["alpha_end"],
                hp["eps_decay_steps"],
                mean_len,
                std_len,
            )
    return rows


def _hp_search(
    algo_label: str,
    *,
    grid_config: dict | None = None,
) -> tuple[dict, list[dict]]:
    """Run 3-stage HP search for *algo_label*. Returns (best_hp, all_rows)."""
    log = _get_logger()
    rng_seed = 0 if algo_label == "sarsa" else 1

    if log:
        log.info(
            "=== [%s] HP Stage 1 — %d configs × %d episodes × %d seeds ===",
            algo_label,
            CP_HP_STAGE1_CONFIGS,
            CP_HP_STAGE1_EPISODES,
            len(SEEDS),
        )
    stage1_configs = _make_random_hp_configs(CP_HP_STAGE1_CONFIGS, rng_seed=rng_seed)
    rows1 = _run_hp_stage(
        algo_label,
        stage1_configs,
        CP_HP_STAGE1_EPISODES,
        stage=1,
        grid_config=grid_config,
    )

    top_k_idx = sorted(
        range(len(rows1)), key=lambda i: rows1[i]["mean_episode_len"], reverse=True
    )[:CP_HP_STAGE2_TOP_K]
    stage2_configs = [stage1_configs[i] for i in top_k_idx]

    if log:
        log.info(
            "=== [%s] HP Stage 2 — top %d × %d episodes × %d seeds ===",
            algo_label,
            CP_HP_STAGE2_TOP_K,
            CP_HP_STAGE2_EPISODES,
            len(SEEDS),
        )
    rows2 = _run_hp_stage(
        algo_label,
        stage2_configs,
        CP_HP_STAGE2_EPISODES,
        stage=2,
        grid_config=grid_config,
    )

    top3_idx = sorted(
        range(len(rows2)), key=lambda i: rows2[i]["mean_episode_len"], reverse=True
    )[:CP_HP_STAGE3_TOP_K]
    top3_configs = [stage2_configs[i] for i in top3_idx]

    stage3_configs: list[dict] = []
    for hp in top3_configs:
        for alpha_mult in [0.5, 1.0, 2.0]:
            for decay_mult in [0.75, 1.0, 1.25]:
                new_hp = dict(hp)
                new_hp["alpha_start"] = min(float(hp["alpha_start"]) * alpha_mult, 0.99)
                new_hp["eps_decay_steps"] = int(hp["eps_decay_steps"] * decay_mult)
                stage3_configs.append(new_hp)

    if log:
        log.info(
            "=== [%s] HP Stage 3 — %d perturbation configs × %d episodes × %d seeds ===",
            algo_label,
            len(stage3_configs),
            CP_HP_STAGE3_EPISODES,
            len(SEEDS),
        )
    rows3 = _run_hp_stage(
        algo_label,
        stage3_configs,
        CP_HP_STAGE3_EPISODES,
        stage=3,
        grid_config=grid_config,
    )

    best_row = max(rows3, key=lambda r: r["mean_episode_len"])
    best_hp = {
        "alpha_start": best_row["alpha_start"],
        "alpha_end": best_row["alpha_end"],
        "alpha_decay_steps": int(best_row["alpha_decay_steps"]),
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_steps": int(best_row["eps_decay_steps"]),
        "gamma": best_row["gamma"],
    }
    if log:
        log.info(
            "  [%s] Best HP: α=%.2f→%.3f α_decay=%d eps_decay=%d len=%.1f±%.1f",
            algo_label,
            best_hp["alpha_start"],
            best_hp["alpha_end"],
            best_hp["alpha_decay_steps"],
            best_hp["eps_decay_steps"],
            best_row["mean_episode_len"],
            best_row["mean_episode_len_std"],
        )
    return best_hp, rows1 + rows2 + rows3


# ── Lifecycle ─────────────────────────────────────────────────────────────────


def run() -> Path:
    """Execute Phase 5 computation, write all artifacts, return checkpoint path."""
    global _logger
    from src.utils.logger import configure_logger

    _logger = configure_logger("phase5")
    log = _logger

    paths = resolve_phase_paths(_PHASE_ID, _SLUG)
    paths.makedirs()

    # Default grid config (None → CartPoleDiscretizer uses CARTPOLE_* defaults)
    default_grid_config = None

    # ── HP Search ─────────────────────────────────────────────────────────────
    log.info("=== Phase 5 HP Search (multi-seed scoring over %d seeds) ===", len(SEEDS))
    t_hp_start = time.perf_counter()

    sarsa_best_hp, sarsa_hp_rows = _hp_search("sarsa", grid_config=default_grid_config)
    ql_best_hp, ql_hp_rows = _hp_search("qlearning", grid_config=default_grid_config)

    hp_df = pd.DataFrame(sarsa_hp_rows + ql_hp_rows)
    hp_df.to_csv(paths.metrics_dir / "mf_hp_search.csv", index=False)
    log.info(
        "HP search done (%.1fs) → %s",
        time.perf_counter() - t_hp_start,
        paths.metrics_dir / "mf_hp_search.csv",
    )

    # ── Final Training — two regimes ─────────────────────────────────────────
    baseline_hp = {
        "alpha_start": CP_BASELINE_ALPHA_START,
        "alpha_end": CP_BASELINE_ALPHA_END,
        "alpha_decay_steps": CP_BASELINE_ALPHA_DECAY_STEPS,
        "eps_start": CP_BASELINE_EPS_START,
        "eps_end": CP_BASELINE_EPS_END,
        "eps_decay_steps": CP_BASELINE_EPS_DECAY_STEPS,
        "gamma": CP_BASELINE_GAMMA,
    }
    tuned_hps = {"sarsa": sarsa_best_hp, "qlearning": ql_best_hp}

    jobs = [
        {
            "algorithm": algo,
            "seed": seed,
            "regime": regime,
            "hp": tuned_hps[algo] if regime == "tuned" else baseline_hp,
            "grid_config": default_grid_config,
            "train_episodes": CP_TRAIN_EPISODES,
            "eval_episodes": CP_EVAL_EPISODES_MAIN,
            "disable_early_stopping": True,
        }
        for regime in ["controlled", "tuned"]
        for algo in ["sarsa", "qlearning"]
        for seed in SEEDS
    ]

    n_workers: int | None = PHASE5_FINAL_TRAIN_MAX_WORKERS
    if n_workers is None:
        cpu = os.cpu_count() or 1
        n_workers = min(len(jobs), max(1, cpu - 1))

    log.info(
        "=== Phase 5 Final Training — %d jobs, %d workers "
        "(2 regimes × 2 algos × %d seeds × %d eps) ===",
        len(jobs),
        n_workers,
        len(SEEDS),
        CP_TRAIN_EPISODES,
    )
    t_train_start = time.perf_counter()
    results: list[dict] = []

    if n_workers == 1:
        for job in jobs:
            log.info(
                "  [serial] starting %s/%s seed=%d",
                job["regime"],
                job["algorithm"],
                job["seed"],
            )
            result = _run_phase5_final_job(job)
            log.info(
                "  [serial] done %s/%s seed=%d  episodes=%d  len=%.1f  %.1fs",
                result["regime"],
                result["algorithm"],
                result["seed"],
                result["episodes_run"],
                result["mean_episode_len"],
                result["train_wall_clock_s"],
            )
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_phase5_final_job, job): job for job in jobs}
            for fut in as_completed(futures):
                result = fut.result()
                log.info(
                    "  done %s/%s seed=%d  episodes=%d  len=%.1f  train=%.1fs",
                    result["regime"],
                    result["algorithm"],
                    result["seed"],
                    result["episodes_run"],
                    result["mean_episode_len"],
                    result["train_wall_clock_s"],
                )
                results.append(result)

    t_train_end = time.perf_counter()
    log.info("Final training done (%.1fs)", t_train_end - t_train_start)

    # Deterministic sort
    regime_order = {"controlled": 0, "tuned": 1}
    algo_order = {"sarsa": 0, "qlearning": 1}
    seed_order = {s: i for i, s in enumerate(SEEDS)}
    results.sort(
        key=lambda r: (
            regime_order[r["regime"]],
            algo_order[r["algorithm"]],
            seed_order[r["seed"]],
        )
    )

    # ── Write CSVs — final training ───────────────────────────────────────────
    curve_rows: list[dict] = []
    eval_rows: list[dict] = []

    for result in results:
        for ep, wm in zip(result["curve_episode"], result["curve_window_mean"]):
            curve_rows.append(
                {
                    "algorithm": result["algorithm"],
                    "seed": result["seed"],
                    "regime": result["regime"],
                    "episode": int(ep),
                    "window_mean": float(wm),
                }
            )
        wm_list = [float(v) for v in result["curve_window_mean"]]
        conv_ep: int | None = None
        for k in range(len(wm_list)):
            if check_convergence(
                wm_list[: k + 1],
                window=RL_CONVERGENCE_WINDOW,
                delta=RL_CONVERGENCE_DELTA,
                m_consec=RL_CONVERGENCE_M,
            ):
                conv_ep = int(result["curve_episode"][k])
                break

        final_window_return = float(wm_list[-1]) if wm_list else float("nan")
        eval_rows.append(
            {
                "algorithm": result["algorithm"],
                "seed": result["seed"],
                "regime": result["regime"],
                "mean_episode_len": result["mean_episode_len"],
                "final_window_return": final_window_return,
                "convergence_episode": conv_ep,
                "train_wall_clock_s": result["train_wall_clock_s"],
            }
        )

    pd.DataFrame(curve_rows).to_csv(
        paths.metrics_dir / "mf_learning_curves.csv", index=False
    )
    eval_per_seed_df = pd.DataFrame(eval_rows)
    eval_per_seed_df.to_csv(paths.metrics_dir / "mf_eval_per_seed.csv", index=False)

    summary_rows: list[dict] = []
    for regime in ["controlled", "tuned"]:
        for algo in ["sarsa", "qlearning"]:
            sub = eval_per_seed_df[
                (eval_per_seed_df["algorithm"] == algo)
                & (eval_per_seed_df["regime"] == regime)
            ]
            for metric in ["mean_episode_len", "final_window_return"]:
                vals = sub[metric].values
                summary_rows.append(
                    {
                        "algorithm": algo,
                        "regime": regime,
                        "metric": metric,
                        "mean": round(float(np.mean(vals)), 4),
                        "std": round(float(np.std(vals)), 4),
                        "iqr": round(
                            float(np.percentile(vals, 75) - np.percentile(vals, 25)), 4
                        ),
                    }
                )
    pd.DataFrame(summary_rows).to_csv(
        paths.metrics_dir / "mf_eval_summary.csv", index=False
    )
    log.info("Metrics saved → %s", paths.metrics_dir)

    # ── Discretization study ──────────────────────────────────────────────────
    log.info(
        "=== Phase 5 Discretization Study (%d grids × 2 algos × %d seeds) ===",
        len(CARTPOLE_GRID_NAMES),
        len(SEEDS),
    )
    t_disc_start = time.perf_counter()

    disc_jobs = [
        {
            "algorithm": algo,
            "seed": seed,
            "grid_name": grid_name,
            "hp": tuned_hps[algo],
            "grid_config": CARTPOLE_GRID_CONFIGS[grid_name],
            "train_episodes": CP_DISC_TRAIN_EPISODES,
            "eval_episodes": CP_EVAL_EPISODES_MAIN,
        }
        for grid_name in CARTPOLE_GRID_NAMES
        for algo in ["sarsa", "qlearning"]
        for seed in SEEDS
    ]

    disc_results: list[dict] = []
    if n_workers == 1:
        for job in disc_jobs:
            log.info(
                "  [serial] disc %s/%s grid=%s",
                job["algorithm"],
                str(job["seed"]),
                job["grid_name"],
            )
            disc_results.append(_run_disc_job(job))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_disc_job, job): job for job in disc_jobs}
            for fut in as_completed(futures):
                r = fut.result()
                log.info(
                    "  done disc %s/%s grid=%s  len=%.1f",
                    r["algorithm"],
                    str(r["seed"]),
                    r["grid_name"],
                    r["mean_episode_len"],
                )
                disc_results.append(r)

    log.info("Discretization study done (%.1fs)", time.perf_counter() - t_disc_start)

    grid_order = {g: i for i, g in enumerate(CARTPOLE_GRID_NAMES)}
    disc_results.sort(
        key=lambda r: (
            grid_order[r["grid_name"]],
            algo_order[r["algorithm"]],
            seed_order[r["seed"]],
        )
    )
    disc_df = pd.DataFrame(
        [
            {
                "grid": r["grid_name"],
                "algorithm": r["algorithm"],
                "seed": r["seed"],
                "final_mean_len": r["mean_episode_len"],
                "convergence_episode": r["convergence_episode"],
            }
            for r in disc_results
        ]
    )
    disc_df.to_csv(paths.metrics_dir / "mf_discretization.csv", index=False)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    def _regime_algo_summary(algo: str, regime: str) -> dict:
        sub = eval_per_seed_df[
            (eval_per_seed_df["algorithm"] == algo)
            & (eval_per_seed_df["regime"] == regime)
        ]
        lens = sub["mean_episode_len"].values
        fwr = sub["final_window_return"].values
        conv = sub["convergence_episode"].dropna()
        return {
            "mean_episode_len": round(float(np.mean(lens)), 2),
            "std_episode_len": round(float(np.std(lens)), 2),
            "iqr_episode_len": round(
                float(np.percentile(lens, 75) - np.percentile(lens, 25)), 2
            ),
            "mean_final_return": round(float(np.mean(fwr)), 2),
            "final_window_iqr": round(
                float(np.percentile(fwr, 75) - np.percentile(fwr, 25)), 2
            ),
            "mean_convergence_episode": round(float(conv.mean()), 1)
            if len(conv)
            else None,
            "convergence_episode_iqr": round(
                float(np.percentile(conv, 75) - np.percentile(conv, 25)), 1
            )
            if len(conv)
            else None,
        }

    def _disc_grid_summary(grid_name: str) -> dict:
        sub = disc_df[disc_df["grid"] == grid_name]
        out = {}
        for algo in ["sarsa", "qlearning"]:
            algo_sub = sub[sub["algorithm"] == algo]["final_mean_len"].values
            out[algo] = {
                "mean_episode_len": round(float(np.mean(algo_sub)), 2),
                "std_episode_len": round(float(np.std(algo_sub)), 2),
            }
        return out

    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": _PHASE_ID,
        "slug": _SLUG,
        "upstream_inputs": [],
        "outputs": {
            "metrics_dir": str(paths.metrics_dir),
            "figures_dir": str(paths.figures_dir),
        },
        "config_snapshot": {
            "seeds": SEEDS,
            "n_actions": CP_N_ACTIONS,
            "default_grid": "default",
            "train_episodes": CP_TRAIN_EPISODES,
            "early_stopping_disabled_for_final_training": True,
            "baseline_schedule": baseline_hp,
            "sarsa_best_hp": sarsa_best_hp,
            "qlearning_best_hp": ql_best_hp,
        },
        "summary": {
            "controlled": {
                "sarsa": _regime_algo_summary("sarsa", "controlled"),
                "qlearning": _regime_algo_summary("qlearning", "controlled"),
            },
            "tuned": {
                "sarsa": _regime_algo_summary("sarsa", "tuned"),
                "qlearning": _regime_algo_summary("qlearning", "tuned"),
            },
            "discretization": {g: _disc_grid_summary(g) for g in CARTPOLE_GRID_NAMES},
            "hp_search": {
                "stages": 3,
                "seeds_per_config": len(SEEDS),
                "stage1_configs_per_algo": CP_HP_STAGE1_CONFIGS,
                "stage1_episodes": CP_HP_STAGE1_EPISODES,
                "stage2_top_k": CP_HP_STAGE2_TOP_K,
                "stage2_episodes": CP_HP_STAGE2_EPISODES,
                "stage3_top_k": CP_HP_STAGE3_TOP_K,
                "stage3_episodes": CP_HP_STAGE3_EPISODES,
            },
            "final_training": {
                "n_workers": n_workers,
                "train_wall_clock_s": round(t_train_end - t_train_start, 3),
                "disc_wall_clock_s": round(time.perf_counter() - t_disc_start, 3),
            },
        },
    }

    write_checkpoint_json(checkpoint, paths.checkpoint_path)
    log.info("Phase 5 checkpoint saved → %s", paths.checkpoint_path)

    validate_required_outputs(
        [
            paths.metrics_dir / "mf_hp_search.csv",
            paths.metrics_dir / "mf_learning_curves.csv",
            paths.metrics_dir / "mf_eval_per_seed.csv",
            paths.metrics_dir / "mf_eval_summary.csv",
            paths.metrics_dir / "mf_discretization.csv",
            paths.checkpoint_path,
        ]
    )

    log.info("=== Phase 5 Summary ===")
    for regime in ["controlled", "tuned"]:
        for algo in ["sarsa", "qlearning"]:
            s = checkpoint["summary"][regime][algo]
            log.info(
                "[%s] %s: len=%.1f ± %.1f (IQR=%.1f)  final_window_iqr=%.1f",
                regime,
                algo.upper(),
                s["mean_episode_len"],
                s["std_episode_len"],
                s["iqr_episode_len"],
                s["final_window_iqr"],
            )

    return paths.checkpoint_path


def visualize(checkpoint_path: Path) -> list[Path]:
    """Render Phase 5 figures from saved artifacts. No live computation."""
    from src.utils.logger import configure_logger

    log = configure_logger("phase5")
    checkpoint = load_checkpoint_json(checkpoint_path)

    metrics_dir = Path(checkpoint["outputs"]["metrics_dir"])
    figures_dir = Path(checkpoint["outputs"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    figs: list[Path] = []

    log.info("Rendering Phase 5 figures → %s", figures_dir)

    figs.append(plot_cp_mf_learning_curve(metrics_dir, figures_dir))
    log.info("  learning curves → %s", figs[-1])

    figs.append(plot_cp_mf_comparison(metrics_dir, figures_dir))
    log.info("  comparison → %s", figs[-1])

    figs.append(plot_cp_mf_hp_sensitivity(metrics_dir, figures_dir))
    log.info("  HP sensitivity → %s", figs[-1])

    figs.append(plot_cp_mf_discretization(metrics_dir, figures_dir))
    log.info("  discretization → %s", figs[-1])

    log.info("Phase 5 figures done (%d files)", len(figs))
    return figs


if __name__ == "__main__":
    checkpoint_path = run()
    visualize(checkpoint_path)
