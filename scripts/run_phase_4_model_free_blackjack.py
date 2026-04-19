"""Phase 4 — Model-Free: SARSA & Q-Learning on Blackjack.

Lifecycle:
    run()               → execute computation, write all artifacts, return checkpoint path
    visualize(path)     → reload from disk only, render all figures

HP search: 3-stage progressive narrowing, each stage scored over all SEEDS.
Final training: parallel via ProcessPoolExecutor, two regimes per algorithm.
  - controlled: both algorithms use the same fixed baseline schedule (fair comparison).
  - tuned: each algorithm uses its own HP-search winner.

Outputs:
  artifacts/metrics/phase4_model_free_blackjack/
    mf_hp_search.csv            ← per-config mean_return + win_rate (avg over SEEDS) for all stages
    mf_learning_curves.csv      ← window-mean returns per (algorithm, seed, regime)
    mf_eval_per_seed.csv        ← win/draw/loss + wall-clock per (algorithm, seed, regime)
    mf_eval_summary.csv         ← mean / std / IQR across seeds per (algorithm, regime, metric)
  artifacts/figures/phase4_model_free_blackjack/
    blackjack_mf_learning_curves.png
    blackjack_mf_comparison.png
    blackjack_mf_hp_sensitivity.png
  artifacts/metadata/phase4.json
  artifacts/logs/phase4.log

Usage:
  make phase4
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

from src.algorithms import QLearningConfig, SarsaConfig, run_q_learning, run_sarsa
from src.algorithms.model_free_utils import (
    check_convergence,
    encode_bj_state,
    greedy_action,
)
from src.config import (
    BJ_BASELINE_ALPHA_DECAY_STEPS,
    BJ_BASELINE_ALPHA_END,
    BJ_BASELINE_ALPHA_START,
    BJ_BASELINE_EPS_DECAY_STEPS,
    BJ_BASELINE_EPS_END,
    BJ_BASELINE_EPS_START,
    BJ_BASELINE_GAMMA,
    BJ_EVAL_EPISODES_HP,
    BJ_EVAL_EPISODES_MAIN,
    BJ_HP_STAGE1_CONFIGS,
    BJ_HP_STAGE1_EPISODES,
    BJ_HP_STAGE2_EPISODES,
    BJ_HP_STAGE2_TOP_K,
    BJ_HP_STAGE3_EPISODES,
    BJ_HP_STAGE3_TOP_K,
    BJ_N_ACTIONS,
    BJ_N_STATES,
    BJ_TRAIN_EPISODES,
    PHASE4_FINAL_TRAIN_MAX_WORKERS,
    RL_CONVERGENCE_DELTA,
    RL_CONVERGENCE_M,
    RL_CONVERGENCE_WINDOW,
    SEEDS,
)
from src.utils.phase_artifacts import (
    SCHEMA_VERSION,
    load_checkpoint_json,
    resolve_phase_paths,
    validate_required_outputs,
    write_checkpoint_json,
)
from src.utils.plotting import (
    plot_mf_comparison,
    plot_mf_hp_sensitivity,
    plot_mf_learning_curve,
)

# No module-level logger — configure_logger is called inside run() only.
# Workers spawned by ProcessPoolExecutor must not inherit file handlers.
_logger = None  # set by run(); accessed via _get_logger() in HP-search helpers

_PHASE_ID = "phase4"
_SLUG = "model_free_blackjack"

# HP search space (shared across SARSA and Q-Learning for fair comparison)
_HP_ALPHA_STARTS = [0.1, 0.3, 0.5, 0.7, 0.9]
_HP_ALPHA_ENDS = [0.001, 0.01, 0.05]
_HP_ALPHA_DECAY_STEPS = [50_000, 100_000, 200_000, 500_000]
_HP_EPS_DECAY_STEPS = [20_000, 50_000, 100_000, 200_000]
_HP_GAMMA = 0.99
# HP search seed scheme: actual SEEDS values shifted by config_idx * stride.
# This anchors HP-search seeds to the documented reproducibility contract while
# keeping each config's random stream distinct from its neighbours.
_HP_SEED_CONFIG_STRIDE: int = 100_000
_HP_EVAL_SEED_OFFSET: int = 500_000  # eval seed = train_seed + this offset


def _get_logger():
    """Return the main-process logger (None in worker processes)."""
    return _logger


# ── Config builder ────────────────────────────────────────────────────────────


def _build_mf_config(
    algo_label: str,
    hp: dict,
    *,
    disable_early_stopping: bool = False,
) -> SarsaConfig | QLearningConfig:
    """Build a SarsaConfig or QLearningConfig from a plain HP dict.

    Args:
        disable_early_stopping: If True, set convergence_delta=0.0 so the
            running-window plateau rule never fires.  Use this for final
            training so all seeds produce curves of equal length.
    """
    kwargs = dict(
        alpha_start=hp["alpha_start"],
        alpha_end=hp["alpha_end"],
        alpha_decay_steps=hp["alpha_decay_steps"],
        eps_start=hp.get("eps_start", 1.0),
        eps_end=hp.get("eps_end", 0.01),
        eps_decay_steps=hp["eps_decay_steps"],
        gamma=hp.get("gamma", _HP_GAMMA),
        convergence_window=RL_CONVERGENCE_WINDOW,
        # convergence_delta=0.0 → abs(diff) < 0 is always False → never converges
        convergence_delta=0.0 if disable_early_stopping else RL_CONVERGENCE_DELTA,
        convergence_m=RL_CONVERGENCE_M,
    )
    return SarsaConfig(**kwargs) if algo_label == "sarsa" else QLearningConfig(**kwargs)


# ── Evaluation helper ─────────────────────────────────────────────────────────


def _eval_mf_policy(Q: np.ndarray, n_episodes: int, seed: int) -> dict[str, float]:
    """Greedy rollout evaluation of a Q-table on Blackjack-v1.

    Returns win_rate, draw_rate, loss_rate, and mean_return.
    mean_return = win_rate − loss_rate (draw = 0 reward, win = +1, loss = −1).
    This matches the episodic return the algorithms optimize and is the primary
    ranking signal for HP search.
    """
    rng = np.random.default_rng(seed)
    env = gym.make("Blackjack-v1")
    wins = draws = losses = 0
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
    for _ in range(n_episodes):
        state = encode_bj_state(obs)
        done = False
        ep_return = 0.0
        while not done:
            action = greedy_action(Q, state)
            obs, reward, terminated, truncated, _ = env.step(action)
            state = encode_bj_state(obs)
            ep_return += float(reward)
            done = terminated or truncated
        if ep_return > 0:
            wins += 1
        elif ep_return == 0:
            draws += 1
        else:
            losses += 1
        obs, _ = env.reset()
    env.close()
    return {
        "win_rate": wins / n_episodes,
        "draw_rate": draws / n_episodes,
        "loss_rate": losses / n_episodes,
        "mean_return": (wins - losses) / n_episodes,
    }


# ── Top-level worker (must be picklable — no logger, no shared state) ─────────


def _run_phase4_final_job(job: dict) -> dict:
    """Worker for one final-training job: one (algorithm, seed, regime) triple.

    Args:
        job: Plain dict with keys:
            algorithm, seed, regime, hp, train_episodes, eval_episodes,
            disable_early_stopping.

    Returns:
        Compact result dict with keys:
            algorithm, seed, regime, episodes_run, train_wall_clock_s,
            eval_wall_clock_s, curve_episode (ndarray), curve_window_mean (ndarray),
            win_rate, draw_rate, loss_rate.
    """
    algo_label: str = job["algorithm"]
    seed: int = job["seed"]
    regime: str = job["regime"]
    hp: dict = job["hp"]
    train_episodes: int = job["train_episodes"]
    eval_episodes: int = job["eval_episodes"]
    disable_es: bool = job.get("disable_early_stopping", True)

    env = gym.make("Blackjack-v1")
    cfg = _build_mf_config(algo_label, hp, disable_early_stopping=disable_es)

    t_train = time.perf_counter()
    if algo_label == "sarsa":
        Q, trace = run_sarsa(
            env,
            cfg,
            BJ_N_STATES,
            BJ_N_ACTIONS,
            train_episodes,
            seed=seed,
            log_interval=train_episodes + 1,  # suppress worker-side logging
        )
    else:
        Q, trace = run_q_learning(
            env,
            cfg,
            BJ_N_STATES,
            BJ_N_ACTIONS,
            train_episodes,
            seed=seed,
            log_interval=train_episodes + 1,
        )
    env.close()
    train_wall = time.perf_counter() - t_train

    t_eval = time.perf_counter()
    stats = _eval_mf_policy(Q, eval_episodes, seed=seed)
    eval_wall = time.perf_counter() - t_eval

    # Subsample: keep only window records (every convergence_window episodes).
    # This avoids IPC overhead from 500k-row traces while preserving the
    # smoothed learning curve needed for plots.
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
        "eval_wall_clock_s": round(eval_wall, 3),
        "curve_episode": episodes_arr,
        "curve_window_mean": wmeans_arr,
        "win_rate": stats["win_rate"],
        "draw_rate": stats["draw_rate"],
        "loss_rate": stats["loss_rate"],
        "mean_return": stats["mean_return"],
    }


# ── HP Search (multi-seed scoring) ────────────────────────────────────────────


def _make_random_hp_configs(n: int, rng_seed: int) -> list[dict]:
    """Sample *n* random HP configs from the search space."""
    rng = np.random.default_rng(rng_seed)
    return [
        {
            "alpha_start": float(rng.choice(_HP_ALPHA_STARTS)),
            "alpha_end": float(rng.choice(_HP_ALPHA_ENDS)),
            "alpha_decay_steps": int(rng.choice(_HP_ALPHA_DECAY_STEPS)),
            "eps_start": 1.0,
            "eps_end": 0.01,
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
) -> list[dict]:
    """Run one HP search stage; scores each config over all SEEDS.

    Seed scheme: train_seed = seed + config_idx * _HP_SEED_CONFIG_STRIDE.
    This keeps configs on distinct random streams while anchoring all seeds
    to the documented SEEDS values (so the scheme is reproducible from the
    checkpoint metadata alone).

    Promotion uses ``mean_return`` (= win_rate − loss_rate), which matches
    the episodic return the algorithms actually optimize.

    Args:
        algo_label:  'sarsa' or 'qlearning'.
        hp_configs:  list of HP dicts to evaluate.
        n_episodes:  training budget per (config, seed) run.
        stage:       stage number, stored verbatim in the output rows.

    Returns:
        List of one row dict per config for mf_hp_search.csv.
    """
    log = _get_logger()
    rows: list[dict] = []

    for i, hp in enumerate(hp_configs):
        seed_returns: list[float] = []
        seed_win_rates: list[float] = []
        for seed in SEEDS:
            train_seed = seed + i * _HP_SEED_CONFIG_STRIDE
            env = gym.make("Blackjack-v1")
            cfg = _build_mf_config(algo_label, hp)
            if algo_label == "sarsa":
                Q, _ = run_sarsa(
                    env, cfg, BJ_N_STATES, BJ_N_ACTIONS, n_episodes, seed=train_seed
                )
            else:
                Q, _ = run_q_learning(
                    env, cfg, BJ_N_STATES, BJ_N_ACTIONS, n_episodes, seed=train_seed
                )
            env.close()
            eval_stats = _eval_mf_policy(
                Q, BJ_EVAL_EPISODES_HP, seed=train_seed + _HP_EVAL_SEED_OFFSET
            )
            seed_returns.append(eval_stats["mean_return"])
            seed_win_rates.append(eval_stats["win_rate"])

        mean_ret = float(np.mean(seed_returns))
        std_ret = float(np.std(seed_returns))
        row = {
            "algorithm": algo_label,
            "stage": stage,
            "config_idx": i,
            "alpha_start": hp["alpha_start"],
            "alpha_end": hp["alpha_end"],
            "alpha_decay_steps": hp["alpha_decay_steps"],
            "eps_decay_steps": hp["eps_decay_steps"],
            "gamma": hp["gamma"],
            # mean_return is the primary ranking signal
            "mean_return": mean_ret,
            "mean_return_std": std_ret,
            # win_rate kept as a diagnostic
            "win_rate": float(np.mean(seed_win_rates)),
        }
        rows.append(row)
        if log:
            log.info(
                "  [%s] stage%d cfg%d: α=%.2f→%.3f eps_decay=%d  return=%.3f±%.3f",
                algo_label,
                stage,
                i,
                hp["alpha_start"],
                hp["alpha_end"],
                hp["eps_decay_steps"],
                mean_ret,
                std_ret,
            )

    return rows


def _hp_search(algo_label: str) -> tuple[dict, list[dict]]:
    """Run 3-stage HP search for *algo_label*. Returns (best_hp, all_rows)."""
    log = _get_logger()
    rng_seed = 0 if algo_label == "sarsa" else 1

    if log:
        log.info(
            "=== [%s] HP Stage 1 — %d configs × %d episodes × %d seeds ===",
            algo_label,
            BJ_HP_STAGE1_CONFIGS,
            BJ_HP_STAGE1_EPISODES,
            len(SEEDS),
        )
    stage1_configs = _make_random_hp_configs(BJ_HP_STAGE1_CONFIGS, rng_seed=rng_seed)
    rows1 = _run_hp_stage(algo_label, stage1_configs, BJ_HP_STAGE1_EPISODES, stage=1)

    top_k_idx = sorted(
        range(len(rows1)), key=lambda i: rows1[i]["mean_return"], reverse=True
    )[:BJ_HP_STAGE2_TOP_K]
    stage2_configs = [stage1_configs[i] for i in top_k_idx]

    if log:
        log.info(
            "=== [%s] HP Stage 2 — top %d × %d episodes × %d seeds ===",
            algo_label,
            BJ_HP_STAGE2_TOP_K,
            BJ_HP_STAGE2_EPISODES,
            len(SEEDS),
        )
    rows2 = _run_hp_stage(algo_label, stage2_configs, BJ_HP_STAGE2_EPISODES, stage=2)

    top3_idx = sorted(
        range(len(rows2)), key=lambda i: rows2[i]["mean_return"], reverse=True
    )[:BJ_HP_STAGE3_TOP_K]
    top3_configs = [stage2_configs[i] for i in top3_idx]

    # Stage 3: perturb top 3 — α×{0.5,1,2} and eps_decay×{0.75,1,1.25}
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
            BJ_HP_STAGE3_EPISODES,
            len(SEEDS),
        )
    rows3 = _run_hp_stage(algo_label, stage3_configs, BJ_HP_STAGE3_EPISODES, stage=3)

    best_row = max(rows3, key=lambda r: r["mean_return"])
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
            "  [%s] Best HP: α=%.2f→%.3f α_decay=%d eps_decay=%d return=%.3f±%.3f",
            algo_label,
            best_hp["alpha_start"],
            best_hp["alpha_end"],
            best_hp["alpha_decay_steps"],
            best_hp["eps_decay_steps"],
            best_row["mean_return"],
            best_row["mean_return_std"],
        )
    return best_hp, rows1 + rows2 + rows3


# ── Lifecycle ─────────────────────────────────────────────────────────────────


def run() -> Path:
    """Execute Phase 4 computation, write all artifacts, return checkpoint path."""
    global _logger
    from src.utils.logger import configure_logger

    _logger = configure_logger("phase4")
    log = _logger

    paths = resolve_phase_paths(_PHASE_ID, _SLUG)
    paths.makedirs()

    # ── HP Search (serial, multi-seed scoring) ────────────────────────────────
    log.info("=== Phase 4 HP Search (multi-seed scoring over %d seeds) ===", len(SEEDS))
    t_hp_start = time.perf_counter()

    sarsa_best_hp, sarsa_hp_rows = _hp_search("sarsa")
    ql_best_hp, ql_hp_rows = _hp_search("qlearning")

    hp_df = pd.DataFrame(sarsa_hp_rows + ql_hp_rows)
    hp_df.to_csv(paths.metrics_dir / "mf_hp_search.csv", index=False)
    log.info(
        "HP search done (%.1fs) → %s",
        time.perf_counter() - t_hp_start,
        paths.metrics_dir / "mf_hp_search.csv",
    )

    # ── Final Training — two regimes ─────────────────────────────────────────
    # controlled: both algorithms share the same fixed baseline schedule for
    #             fair head-to-head comparison.
    # tuned:      each algorithm uses its own HP-search winner.
    baseline_hp = {
        "alpha_start": BJ_BASELINE_ALPHA_START,
        "alpha_end": BJ_BASELINE_ALPHA_END,
        "alpha_decay_steps": BJ_BASELINE_ALPHA_DECAY_STEPS,
        "eps_start": BJ_BASELINE_EPS_START,
        "eps_end": BJ_BASELINE_EPS_END,
        "eps_decay_steps": BJ_BASELINE_EPS_DECAY_STEPS,
        "gamma": BJ_BASELINE_GAMMA,
    }

    tuned_hps = {"sarsa": sarsa_best_hp, "qlearning": ql_best_hp}
    jobs = [
        {
            "algorithm": algo,
            "seed": seed,
            "regime": regime,
            "hp": tuned_hps[algo] if regime == "tuned" else baseline_hp,
            "train_episodes": BJ_TRAIN_EPISODES,
            "eval_episodes": BJ_EVAL_EPISODES_MAIN,
            # disable early stopping so all seeds produce equal-length curves
            "disable_early_stopping": True,
        }
        for regime in ["controlled", "tuned"]
        for algo in ["sarsa", "qlearning"]
        for seed in SEEDS
    ]

    n_workers: int | None = PHASE4_FINAL_TRAIN_MAX_WORKERS
    if n_workers is None:
        cpu = os.cpu_count() or 1
        n_workers = min(len(jobs), max(1, cpu - 1))

    log.info(
        "=== Phase 4 Final Training — %d jobs, %d workers "
        "(2 regimes × 2 algos × %d seeds × %d eps) ===",
        len(jobs),
        n_workers,
        len(SEEDS),
        BJ_TRAIN_EPISODES,
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
            result = _run_phase4_final_job(job)
            log.info(
                "  [serial] done %s/%s seed=%d  episodes=%d  win=%.3f  %.1fs",
                result["regime"],
                result["algorithm"],
                result["seed"],
                result["episodes_run"],
                result["win_rate"],
                result["train_wall_clock_s"],
            )
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_phase4_final_job, job): job for job in jobs}
            for fut in as_completed(futures):
                result = fut.result()
                log.info(
                    "  done %s/%s seed=%d  episodes=%d  win=%.3f  train=%.1fs",
                    result["regime"],
                    result["algorithm"],
                    result["seed"],
                    result["episodes_run"],
                    result["win_rate"],
                    result["train_wall_clock_s"],
                )
                results.append(result)

    log.info("Final training done (%.1fs)", time.perf_counter() - t_train_start)

    # ── Deterministic aggregation ─────────────────────────────────────────────
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

    # ── Write CSVs ────────────────────────────────────────────────────────────
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
        # Retroactively compute convergence episode from stored window-mean curve.
        # Apply the project's plateau rule: m_consec consecutive window-pairs
        # below convergence_delta.  Returns None if the curve never plateaued.
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

        # Last window-mean return from the training curve (final learning stability proxy).
        final_window_return = float(wm_list[-1]) if wm_list else float("nan")

        eval_rows.append(
            {
                "algorithm": result["algorithm"],
                "seed": result["seed"],
                "regime": result["regime"],
                "win_rate": result["win_rate"],
                "draw_rate": result["draw_rate"],
                "loss_rate": result["loss_rate"],
                "mean_return": result["mean_return"],
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

    # Aggregate: mean / std / IQR across seeds per (algorithm, regime, metric)
    summary_rows: list[dict] = []
    for regime in ["controlled", "tuned"]:
        for algo in ["sarsa", "qlearning"]:
            sub = eval_per_seed_df[
                (eval_per_seed_df["algorithm"] == algo)
                & (eval_per_seed_df["regime"] == regime)
            ]
            for metric in [
                "win_rate",
                "draw_rate",
                "loss_rate",
                "mean_return",
                "final_window_return",
            ]:
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
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(paths.metrics_dir / "mf_eval_summary.csv", index=False)
    log.info("Metrics saved → %s", paths.metrics_dir)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    def _regime_algo_summary(algo: str, regime: str) -> dict:
        sub = eval_per_seed_df[
            (eval_per_seed_df["algorithm"] == algo)
            & (eval_per_seed_df["regime"] == regime)
        ]
        ret = sub["mean_return"].values
        win = sub["win_rate"].values
        fwr = sub["final_window_return"].values
        conv = sub["convergence_episode"].dropna()
        return {
            "mean_return": round(float(np.mean(ret)), 4),
            "std_return": round(float(np.std(ret)), 4),
            "iqr_return": round(
                float(np.percentile(ret, 75) - np.percentile(ret, 25)), 4
            ),
            "mean_final_return": round(float(np.mean(fwr)), 4),
            "final_window_iqr": round(
                float(np.percentile(fwr, 75) - np.percentile(fwr, 25)), 4
            ),
            "mean_win_rate": round(float(np.mean(win)), 4),
            "mean_draw_rate": round(float(sub["draw_rate"].mean()), 4),
            "mean_loss_rate": round(float(sub["loss_rate"].mean()), 4),
            "mean_convergence_episode": round(float(conv.mean()), 1)
            if len(conv)
            else None,
            "convergence_episode_iqr": round(
                float(np.percentile(conv, 75) - np.percentile(conv, 25)), 1
            )
            if len(conv)
            else None,
        }

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
            "n_states": BJ_N_STATES,
            "n_actions": BJ_N_ACTIONS,
            "train_episodes": BJ_TRAIN_EPISODES,
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
            "hp_search": {
                "stages": 3,
                "seeds_per_config": len(SEEDS),
                "stage1_configs_per_algo": BJ_HP_STAGE1_CONFIGS,
                "stage1_episodes": BJ_HP_STAGE1_EPISODES,
                "stage2_top_k": BJ_HP_STAGE2_TOP_K,
                "stage2_episodes": BJ_HP_STAGE2_EPISODES,
                "stage3_top_k": BJ_HP_STAGE3_TOP_K,
                "stage3_episodes": BJ_HP_STAGE3_EPISODES,
            },
            "final_training": {
                "n_workers": n_workers,
                "train_wall_clock_s": round(time.perf_counter() - t_train_start, 3),
            },
        },
    }

    write_checkpoint_json(checkpoint, paths.checkpoint_path)
    log.info("Phase 4 checkpoint saved → %s", paths.checkpoint_path)

    validate_required_outputs(
        [
            paths.metrics_dir / "mf_hp_search.csv",
            paths.metrics_dir / "mf_learning_curves.csv",
            paths.metrics_dir / "mf_eval_per_seed.csv",
            paths.metrics_dir / "mf_eval_summary.csv",
            paths.checkpoint_path,
        ]
    )

    log.info("=== Phase 4 Summary ===")
    for regime in ["controlled", "tuned"]:
        for algo in ["sarsa", "qlearning"]:
            s = checkpoint["summary"][regime][algo]
            log.info(
                "[%s] %s: return=%.4f ± %.4f (IQR=%.4f)  win=%.4f",
                regime,
                algo.upper(),
                s["mean_return"],
                s["std_return"],
                s["iqr_return"],
                s["mean_win_rate"],
            )

    return paths.checkpoint_path


def visualize(checkpoint_path: Path) -> list[Path]:
    """Render Phase 4 figures from saved artifacts. No live computation."""
    from src.utils.logger import configure_logger

    log = configure_logger("phase4")

    checkpoint = load_checkpoint_json(checkpoint_path)
    metrics_dir = Path(checkpoint["outputs"]["metrics_dir"])
    figures_dir = Path(checkpoint["outputs"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Phase 4 Figures ===")
    figs: list[Path] = []

    out = plot_mf_learning_curve(metrics_dir, figures_dir)
    log.info("Saved → %s", out)
    figs.append(out)

    out = plot_mf_comparison(metrics_dir, figures_dir)
    log.info("Saved → %s", out)
    figs.append(out)

    out = plot_mf_hp_sensitivity(metrics_dir, figures_dir)
    log.info("Saved → %s", out)
    figs.append(out)

    return figs


if __name__ == "__main__":
    checkpoint_path = run()
    visualize(checkpoint_path)
