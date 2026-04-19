"""Q-Learning (off-policy TD(0)) for tabular environments.

Update rule:
    Q(s,a) ← Q(s,a) + α [r + γ max_{a'} Q(s',a') − Q(s,a)]

α and ε are annealed linearly over steps using a step counter.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from src.algorithms.model_free_utils import (
    check_convergence,
    encode_bj_state,
    epsilon_greedy_action,
    linear_schedule,
)


@dataclass
class QLearningConfig:
    """Hyperparameters for a single Q-Learning run."""

    alpha_start: float = 0.5
    alpha_end: float = 0.1
    alpha_decay_steps: int = 100_000

    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 50_000

    gamma: float = 0.99

    # Convergence criterion (running-mean plateau)
    convergence_window: int = 100
    convergence_delta: float = 0.01
    convergence_m: int = 3


def run_q_learning(
    env: gym.Env,
    config: QLearningConfig,
    n_states: int,
    n_actions: int,
    n_episodes: int,
    seed: int,
    log_interval: int = 1_000,
    logger: logging.Logger | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Run Q-Learning on *env* and return (Q-table, trace).

    Args:
        env: A Gymnasium environment that returns flat integer states or
             Blackjack-v1 tuple observations (auto-encoded via encode_bj_state).
        config: Q-Learning hyperparameters.
        n_states: Total number of states in the tabular representation.
        n_actions: Total number of actions.
        n_episodes: Training episode budget.
        seed: RNG seed for reproducibility.
        log_interval: Emit a progress log line every this many episodes.
        logger: Optional logger; falls back to print if None.

    Returns:
        Q: Q-table of shape (n_states, n_actions).
        trace: List of dicts with keys:
            episode, ep_return, alpha, epsilon, wall_clock_s,
            window_mean (populated every convergence_window episodes).
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_states, n_actions), dtype=np.float64)

    trace: list[dict] = []
    window_returns: list[float] = []
    window_means: list[float] = []
    step = 0
    t0 = time.perf_counter()

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        state = encode_bj_state(obs) if isinstance(obs, tuple) else int(obs)

        ep_return = 0.0
        done = False

        while not done:
            alpha = linear_schedule(
                step, config.alpha_start, config.alpha_end, config.alpha_decay_steps
            )
            epsilon = linear_schedule(
                step, config.eps_start, config.eps_end, config.eps_decay_steps
            )

            action = epsilon_greedy_action(Q, state, epsilon, rng)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += float(reward)

            next_state = (
                encode_bj_state(next_obs)
                if isinstance(next_obs, tuple)
                else int(next_obs)
            )

            # Off-policy: bootstrap from greedy next action
            max_next_q = 0.0 if done else float(np.max(Q[next_state]))
            td_target = float(reward) + config.gamma * max_next_q
            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state
            step += 1

        # Snapshot schedule values at end of episode
        alpha = linear_schedule(
            step, config.alpha_start, config.alpha_end, config.alpha_decay_steps
        )
        epsilon = linear_schedule(
            step, config.eps_start, config.eps_end, config.eps_decay_steps
        )

        window_returns.append(ep_return)
        record: dict = {
            "episode": ep,
            "ep_return": ep_return,
            "alpha": alpha,
            "epsilon": epsilon,
            "wall_clock_s": round(time.perf_counter() - t0, 4),
            "window_mean": None,
        }

        if ep % config.convergence_window == 0:
            wm = float(np.mean(window_returns[-config.convergence_window :]))
            window_means.append(wm)
            record["window_mean"] = wm

            if logger and ep % log_interval == 0:
                logger.info(
                    "  Q-Learning ep %d/%d  window_mean=%.4f  α=%.4f  ε=%.4f",
                    ep,
                    n_episodes,
                    wm,
                    alpha,
                    epsilon,
                )

            if check_convergence(
                window_means,
                config.convergence_window,
                config.convergence_delta,
                config.convergence_m,
            ):
                if logger:
                    logger.info(
                        "  Q-Learning converged at episode %d (window_mean plateau)", ep
                    )
                trace.append(record)
                break
        elif logger and ep % log_interval == 0:
            logger.info(
                "  Q-Learning ep %d/%d  ep_return=%.1f  α=%.4f  ε=%.4f",
                ep,
                n_episodes,
                ep_return,
                alpha,
                epsilon,
            )

        trace.append(record)

    return Q, trace
