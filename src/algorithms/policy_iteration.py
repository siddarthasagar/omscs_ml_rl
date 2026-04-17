"""Policy Iteration for tabular MDPs.

Convergence rule: no policy changes after a policy improvement step.
"""

from __future__ import annotations

import logging
import time

import numpy as np


def _policy_eval(
    T: np.ndarray,
    R: np.ndarray,
    gamma: float,
    policy: np.ndarray,
    delta: float,
    max_iter: int = 100_000,
) -> np.ndarray:
    """Iterative policy evaluation for a fixed policy.

    Solves V_pi iteratively until max|V_new - V| < delta.
    """
    n_states = len(policy)
    sa_idx = np.arange(n_states)
    T_pi = T[sa_idx, policy, :]  # (n_states, n_states) transition under policy
    R_pi = R[sa_idx, policy]  # (n_states,) expected reward under policy

    V = np.zeros(n_states, dtype=np.float64)
    for _ in range(max_iter):
        V_new = R_pi + gamma * (T_pi @ V)
        # Guard against overflow / NaN from ill-conditioned rows
        np.nan_to_num(V_new, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        if float(np.abs(V_new - V).max()) < delta:
            return V_new
        V = V_new
    return V


def run_pi(
    T: np.ndarray,
    R: np.ndarray,
    gamma: float,
    delta: float,
    max_iter: int = 10_000,
    m_consec: int = 1,  # kept for API symmetry with run_vi
    logger: logging.Logger | None = None,
    log_every: int = 10,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Run Policy Iteration on a tabular MDP.

    Args:
        T: Transition matrix of shape (n_states, n_actions, n_states).
        R: Reward matrix of shape (n_states, n_actions).
        gamma: Discount factor.
        delta: Inner policy-evaluation convergence threshold.
        max_iter: Maximum number of policy improvement steps.
        m_consec: Unused; kept for symmetry with run_vi signature.
        logger: Optional logger; if provided, emits progress every log_every iters.
        log_every: Emit a progress line every this many iterations.

    Returns:
        V: Converged value function (n_states,).
        policy: Converged policy (n_states,) integer action indices.
        trace: List of per-iteration dicts: iteration, delta_v, policy_changes, wall_clock_s.
    """
    n_states = T.shape[0]
    policy = np.zeros(n_states, dtype=int)
    V = np.zeros(n_states, dtype=np.float64)
    trace: list[dict] = []
    t0 = time.perf_counter()

    for it in range(1, max_iter + 1):
        V_new = _policy_eval(T, R, gamma, policy, delta)

        # Greedy policy improvement
        Q = R + gamma * np.einsum("san,n->sa", T, V_new)
        new_policy = Q.argmax(axis=1).astype(int)

        policy_changes = int((new_policy != policy).sum())
        dv = float(np.abs(V_new - V).max())

        V = V_new
        policy = new_policy

        elapsed = round(time.perf_counter() - t0, 4)
        trace.append(
            {
                "iteration": it,
                "delta_v": dv,
                "policy_changes": policy_changes,
                "wall_clock_s": elapsed,
            }
        )

        v_converged = dv < delta
        if logger and it % log_every == 0:
            logger.info(
                "  PI iter %d/%d: delta_v=%.2e (target %.2e), policy_changes=%d, elapsed=%.1fs",
                it,
                max_iter,
                dv,
                delta,
                policy_changes,
                elapsed,
            )

        if policy_changes == 0 or v_converged:
            break

    return V, policy, trace
