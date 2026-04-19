"""Shared utilities for tabular model-free RL (SARSA and Q-Learning).

State encoding, action selection, linear schedules, and convergence checking
are centralised here so SARSA and Q-Learning stay thin.
"""

from __future__ import annotations

import numpy as np


# ── State encoding ────────────────────────────────────────────────────────────


def encode_bj_state(obs: tuple) -> int:
    """Encode a Gymnasium Blackjack-v1 observation to a flat integer index.

    Gymnasium Blackjack-v1 observation: (player_sum, dealer_card, usable_ace)
      player_sum  ∈ [4, 21]  (18 values, shifted to 0-based via -4 … but gym
                              reports 1-based dealer so we use raw values)
      dealer_card ∈ [1, 10]  (10 values, 1 = Ace)
      usable_ace  ∈ {0, 1}

    Encoding: player_sum * 11 * 2 + dealer_card * 2 + int(usable_ace)
      → maps into [0, 703] for 704 unique states.
    """
    player_sum, dealer_card, usable_ace = obs
    return int(player_sum) * 11 * 2 + int(dealer_card) * 2 + int(usable_ace)


# ── Schedules ─────────────────────────────────────────────────────────────────


def linear_schedule(
    step: int,
    start: float,
    end: float,
    decay_steps: int,
) -> float:
    """Linear interpolation from *start* to *end* over *decay_steps*.

    Returns *end* for any step beyond decay_steps.
    """
    if step >= decay_steps:
        return end
    return start + (end - start) * step / decay_steps


# ── Action selection ──────────────────────────────────────────────────────────


def epsilon_greedy_action(
    Q: np.ndarray,
    state: int,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """ε-greedy action selection from Q[state].

    With probability ε, returns a uniformly random action; otherwise returns
    the greedy action (ties broken by argmax, i.e. lowest index wins).
    """
    if rng.random() < epsilon:
        return int(rng.integers(0, Q.shape[1]))
    return int(np.argmax(Q[state]))


def greedy_action(Q: np.ndarray, state: int) -> int:
    """Return the greedy action for *state* (no exploration)."""
    return int(np.argmax(Q[state]))


# ── Convergence ───────────────────────────────────────────────────────────────


def check_convergence(
    window_means: list[float],
    window: int,
    delta: float,
    m_consec: int,
) -> bool:
    """Check whether learning has plateaued.

    A plateau is declared when the last *m_consec* consecutive window-pairs
    each have an absolute improvement below *delta*.

    Args:
        window_means: List of per-window mean returns (one entry per window).
        window: Not used in the check logic; retained for call-site clarity.
        delta: Minimum absolute improvement between consecutive window-pairs.
        m_consec: Number of consecutive stagnant pairs required.

    Returns:
        True if converged, False otherwise.
    """
    if len(window_means) < m_consec + 1:
        return False
    recent = window_means[-(m_consec + 1) :]
    pairs_below = sum(abs(recent[i + 1] - recent[i]) < delta for i in range(m_consec))
    return pairs_below >= m_consec
