"""Policy evaluation by gymnasium rollout.

Used by Phase 2 (Blackjack) and Phase 3 (CartPole) to assess DP policy quality.
For Blackjack: metric is mean episode return.
For CartPole:  metric is mean episode length (= total reward since reward=1/step).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from bettermdptools.envs.blackjack_wrapper import BlackjackWrapper

from src.config import SEEDS
from src.envs.cartpole_discretizer import CartPoleDiscretizer


def eval_blackjack_policy(
    policy: np.ndarray,
    seeds: list[int] = SEEDS,
    n_episodes: int = 1000,
) -> list[tuple[int, float]]:
    """Evaluate a Blackjack DP policy by greedy rollouts.

    Uses BlackjackWrapper so observations are already encoded as integer state indices.

    Args:
        policy: Shape (n_states+1,) mapping integer state → action.
        seeds: RNG seeds; one independent eval run per seed.
        n_episodes: Episodes per seed.

    Returns:
        List of (seed, mean_return) pairs.
    """
    results: list[tuple[int, float]] = []
    for seed in seeds:
        env = BlackjackWrapper(gym.make("Blackjack-v1"))
        obs, _ = env.reset(seed=seed)
        ep_returns: list[float] = []
        for _ in range(n_episodes):
            done = False
            ep_return = 0.0
            while not done:
                action = int(policy[int(obs)])
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += float(reward)
                done = terminated or truncated
            ep_returns.append(ep_return)
            obs, _ = env.reset()
        env.close()
        results.append((seed, float(np.mean(ep_returns))))
    return results


def eval_cartpole_policy(
    policy: np.ndarray,
    discretizer: CartPoleDiscretizer,
    seeds: list[int] = SEEDS,
    n_episodes: int = 100,
) -> list[tuple[int, float]]:
    """Evaluate a CartPole DP policy by greedy rollouts.

    Args:
        policy: Shape (n_states+1,) mapping discrete state index → action.
        discretizer: CartPoleDiscretizer for obs→state conversion.
        seeds: RNG seeds; one independent eval run per seed.
        n_episodes: Episodes per seed.

    Returns:
        List of (seed, mean_episode_length) pairs.
    """
    results: list[tuple[int, float]] = []
    for seed in seeds:
        env = gym.make("CartPole-v1")
        obs, _ = env.reset(seed=seed)
        ep_lengths: list[int] = []
        for _ in range(n_episodes):
            done = False
            ep_len = 0
            while not done:
                state = discretizer.obs_to_state(obs)
                action = int(policy[state])
                obs, _, terminated, truncated, _ = env.step(action)
                ep_len += 1
                done = terminated or truncated
            ep_lengths.append(ep_len)
            obs, _ = env.reset()
        env.close()
        results.append((seed, float(np.mean(ep_lengths))))
    return results
