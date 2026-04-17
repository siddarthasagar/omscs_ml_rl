"""Blackjack-v1 model wrapper using bettermdptools analytic enumeration.

Returns the exact T/R matrices; no rollout or estimation required.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from bettermdptools.envs.blackjack_wrapper import BlackjackWrapper


def get_blackjack_model() -> tuple[np.ndarray, np.ndarray, int, int]:
    """Load the Blackjack-v1 T/R model via bettermdptools analytic enumeration.

    bettermdptools maps Blackjack states to integer keys 0..n_states-1.
    Terminal transitions use next_s = -1; we remap these to an absorbing state
    at index n_states so the T matrix is square over n_states + 1 states.

    Returns:
        T: Transition array of shape (n_states+1, n_actions, n_states+1).
           T[s, a, s'] = probability of transitioning from s to s' under action a.
           Index n_states is the absorbing terminal state.
        R: Reward array of shape (n_states+1, n_actions).
        n_states: Number of non-terminal states (290 for Blackjack-v1).
        n_actions: Number of actions (2: stick=0, hit=1).
    """
    gym_env = gym.make("Blackjack-v1")
    bj = BlackjackWrapper(gym_env)
    P_dict: dict = bj.P  # {int_state: {int_action: [(prob, next_s, reward, done)]}}

    n_states = len(P_dict)  # 290
    n_actions = 2
    s_term = n_states  # absorbing terminal state index
    n_states_aug = n_states + 1  # includes terminal

    T = np.zeros((n_states_aug, n_actions, n_states_aug), dtype=np.float64)
    R = np.zeros((n_states_aug, n_actions), dtype=np.float64)

    for s, actions in P_dict.items():
        for a, transitions in actions.items():
            for prob, next_s, reward, _done in transitions:
                sp = s_term if next_s < 0 else int(next_s)
                T[s, a, sp] += prob
                R[s, a] += prob * reward

    # Absorbing state: self-loop with zero reward
    for a in range(n_actions):
        T[s_term, a, s_term] = 1.0
        R[s_term, a] = 0.0

    return T, R, n_states, n_actions
