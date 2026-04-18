"""Value Iteration for tabular MDPs.

Convergence rule: max_s |V_{k+1}(s) - V_k(s)| < delta for m_consec consecutive sweeps.
"""

from __future__ import annotations

import logging
import time

import numpy as np


def run_vi(
    T: np.ndarray,
    R: np.ndarray,
    gamma: float,
    delta: float,
    max_iter: int = 10_000,
    m_consec: int = 1,
    logger: logging.Logger | None = None,
    log_every: int = 200,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Run Value Iteration on a tabular MDP.

    Args:
        T: Transition matrix of shape (n_states, n_actions, n_states).
        R: Reward matrix of shape (n_states, n_actions).
        gamma: Discount factor.
        delta: Convergence threshold (max ΔV).
        max_iter: Maximum number of sweeps.
        m_consec: Consecutive sweeps below delta required to declare convergence.
        logger: Optional logger; if provided, emits progress every log_every iters.
        log_every: Emit a progress line every this many iterations.

    Returns:
        V: Optimal value function (n_states,).
        policy: Greedy policy (n_states,) integer action indices.
        trace: List of per-iteration dicts with keys: iteration, delta_v, wall_clock_s.
    """
    n_states = T.shape[0]
    V = np.zeros(n_states, dtype=np.float64)
    Q = np.empty((n_states, T.shape[1]), dtype=np.float64)
    consec = 0
    trace: list[dict] = []
    t0 = time.perf_counter()

    for it in range(1, max_iter + 1):
        # Q[s, a] = R[s, a] + gamma * sum_{s'} T[s, a, s'] * V[s']
        np.einsum("san,n->sa", T, V, out=Q)
        Q *= gamma
        Q += R

        V_new = Q.max(axis=1)
        dv = float(np.abs(V_new - V).max())
        V = V_new

        trace.append(
            {
                "iteration": it,
                "delta_v": dv,
                "wall_clock_s": round(time.perf_counter() - t0, 4),
            }
        )

        consec = consec + 1 if dv < delta else 0
        if logger and it % log_every == 0:
            logger.info("  VI iter %d: delta_v=%.2e (target %.2e)", it, dv, delta)
        if consec >= m_consec:
            break

    # Recompute Q from the final converged V to ensure policy extraction is
    # consistent with the last value function (the loop's Q may be one sweep old).
    np.einsum("san,n->sa", T, V, out=Q)
    Q *= gamma
    Q += R
    policy = Q.argmax(axis=1).astype(int)
    return V, policy, trace
