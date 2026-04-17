"""CartPole-v1 T/R model estimation via uniform-policy rollout.

Builds a shared tabular model used by all DP methods in Phases 2–3.
The model is estimated once and saved to artifacts/metadata/cartpole_model.npz.
"""

from __future__ import annotations

import logging
from typing import Optional

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from src.config import (
    CARTPOLE_MODEL_MIN_VISITS,
    CARTPOLE_MODEL_ROLLOUT_STEPS,
)
from src.envs.cartpole_discretizer import CartPoleDiscretizer


def build_cartpole_model(
    discretizer: Optional[CartPoleDiscretizer] = None,
    rollout_steps: int = CARTPOLE_MODEL_ROLLOUT_STEPS,
    min_visits: int = CARTPOLE_MODEL_MIN_VISITS,
    seed: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """Estimate CartPole-v1 T/R from a uniform-random-policy rollout.

    Procedure:
    1. Roll out CartPole-v1 under a uniform random policy for `rollout_steps` steps.
    2. Bin each (obs, a, obs', r, done) using the discretizer.
    3. Accumulate counts N(s, a, s') and reward sums R(s, a).
    4. Laplace smoothing (+1 pseudocount) for (s, a) pairs with zero real visits.
    5. Terminal transitions map to a dedicated absorbing state s_term.

    Args:
        discretizer: CartPoleDiscretizer instance. A default one is created if None.
        rollout_steps: Total environment steps to collect.
        min_visits: Threshold for 'covered' (s, a) pairs in diagnostics.
        seed: RNG seed for the rollout (not part of the 5-experiment seeds).
        logger: Logger for progress messages.

    Returns:
        Dict with keys:
            T         — numpy (n_actions, n_states_aug, n_states_aug) transition array
            R         — numpy (n_states_aug, n_actions) reward array
            n_states  — discretizer.n_states (excludes absorbing state)
            n_actions — 2
            absorbing_state_index — integer index of s_term
            coverage_pct — fraction of (s,a) pairs with >= min_visits real transitions
            smoothed_pct — fraction of (s,a) pairs with zero real visits (Laplace-only)
            mean_visits_covered — mean visit count for covered (s,a) pairs
    """
    if discretizer is None:
        discretizer = CartPoleDiscretizer()
    if logger is None:
        logger = logging.getLogger(__name__)

    n_states = discretizer.n_states
    n_actions = 2
    s_term = n_states  # absorbing state index
    n_states_aug = n_states + 1  # includes absorbing state

    # Accumulate raw counts and reward sums
    counts = np.zeros((n_states_aug, n_actions, n_states_aug), dtype=np.int64)
    reward_sums = np.zeros((n_states_aug, n_actions), dtype=np.float64)
    visit_counts = np.zeros((n_states_aug, n_actions), dtype=np.int64)  # N(s, a) total

    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)

    obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
    s = discretizer.obs_to_state(obs)

    steps_done = 0
    episodes = 0

    logger.info(
        "Building CartPole T/R model: %d rollout steps, seed=%s",
        rollout_steps,
        seed,
    )

    with tqdm(total=rollout_steps, unit="steps", desc="Model rollout") as pbar:
        while steps_done < rollout_steps:
            a = int(rng.integers(0, n_actions))
            obs_next, reward, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            if done:
                sp = s_term
            else:
                sp = discretizer.obs_to_state(obs_next)

            counts[s, a, sp] += 1
            reward_sums[s, a] += float(reward)
            visit_counts[s, a] += 1

            steps_done += 1
            pbar.update(1)

            if done:
                obs, _ = env.reset()
                s = discretizer.obs_to_state(obs)
                episodes += 1
            else:
                s = sp

    env.close()
    logger.info("Rollout complete: %d steps, %d episodes", steps_done, episodes)

    # ── Build T and R ─────────────────────────────────────────────────────────
    T = np.zeros((n_states_aug, n_actions, n_states_aug), dtype=np.float64)
    R = np.zeros((n_states_aug, n_actions), dtype=np.float64)

    n_sa = n_states * n_actions  # only non-absorbing states count for diagnostics
    n_covered = 0
    total_visits_covered = 0
    n_smoothed = 0

    for s_idx in range(n_states):
        for a_idx in range(n_actions):
            total = visit_counts[s_idx, a_idx]
            if total > 0:
                T[s_idx, a_idx, :] = counts[s_idx, a_idx, :] / total
                R[s_idx, a_idx] = reward_sums[s_idx, a_idx] / total
                if total >= min_visits:
                    n_covered += 1
                    total_visits_covered += total
            else:
                # Absorbing-state prior: treat unvisited (s,a) as terminal.
                # Conservative assumption — we have no evidence the agent can
                # survive from here, so we assume failure.  R=+1.0 preserves
                # the correct CartPole per-step reward for the one step taken
                # before termination.
                T[s_idx, a_idx, s_term] = 1.0
                R[s_idx, a_idx] = 1.0
                n_smoothed += 1

    # Absorbing state: self-loop with zero reward
    for a_idx in range(n_actions):
        T[s_term, a_idx, s_term] = 1.0
        R[s_term, a_idx] = 0.0

    # ── Coverage diagnostics ──────────────────────────────────────────────────
    coverage_pct = n_covered / n_sa
    smoothed_pct = n_smoothed / n_sa
    mean_visits_covered = total_visits_covered / n_covered if n_covered > 0 else 0.0

    logger.info(
        "Coverage: %.1f%% covered (>=%d visits), %.1f%% Laplace-only, mean visits=%.1f",
        coverage_pct * 100,
        min_visits,
        smoothed_pct * 100,
        mean_visits_covered,
    )

    return {
        "T": T,
        "R": R,
        "n_states": n_states,
        "n_actions": n_actions,
        "absorbing_state_index": s_term,
        "coverage_pct": float(coverage_pct),
        "smoothed_pct": float(smoothed_pct),
        "mean_visits_covered": float(mean_visits_covered),
        "visit_counts": visit_counts[
            :n_states, :
        ],  # (n_states, n_actions) — non-absorbing only
    }
