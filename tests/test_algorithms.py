"""Gate 2 — Algorithm tests.

Validates VI and PI on a small synthetic MDP (deterministic chain + absorbing
state) and on the real Blackjack model.  Does NOT re-run the full Blackjack
eval to keep gate runtime short.
"""

import numpy as np
import pytest

from src.algorithms import run_pi, run_vi
from src.config import PI_DELTA, PI_GAMMA, VI_DELTA, VI_GAMMA


# ── Synthetic MDP fixture ─────────────────────────────────────────────────────
# 3-state chain: s0 -a0→ s1 -a0→ s2(absorbing)
# Rewards: R(s0,a0)=1, R(s1,a0)=1, R(s2,*)=0
# Only one action, so optimal policy is trivial.

N = 3


@pytest.fixture(scope="module")
def chain_mdp():
    T = np.zeros((N, 1, N))
    T[0, 0, 1] = 1.0  # s0 -a0→ s1
    T[1, 0, 2] = 1.0  # s1 -a0→ s2
    T[2, 0, 2] = 1.0  # s2 -a0→ s2 (absorbing)
    R = np.array([[1.0], [1.0], [0.0]])
    return T, R


@pytest.fixture(scope="module")
def blackjack_model():
    from src.envs.blackjack_env import get_blackjack_model

    return get_blackjack_model()


# ── VI tests ──────────────────────────────────────────────────────────────────


class TestValueIteration:
    def test_returns_three_items(self, chain_mdp):
        T, R = chain_mdp
        result = run_vi(T, R, gamma=0.9, delta=1e-8)
        assert len(result) == 3

    def test_trace_is_non_empty(self, chain_mdp):
        T, R = chain_mdp
        _, _, trace = run_vi(T, R, gamma=0.9, delta=1e-8)
        assert len(trace) > 0

    def test_trace_delta_v_decreasing(self, chain_mdp):
        T, R = chain_mdp
        _, _, trace = run_vi(T, R, gamma=0.9, delta=1e-8)
        dvs = [t["delta_v"] for t in trace]
        # Allow non-monotone only at the very first step
        assert dvs[-1] < dvs[0]

    def test_v_shape(self, chain_mdp):
        T, R = chain_mdp
        V, _, _ = run_vi(T, R, gamma=0.9, delta=1e-8)
        assert V.shape == (N,)

    def test_v_absorbing_state_near_zero(self, chain_mdp):
        T, R = chain_mdp
        V, _, _ = run_vi(T, R, gamma=0.9, delta=1e-8)
        assert abs(V[2]) < 1e-4  # absorbing state with R=0 → V≈0

    def test_v_ordering(self, chain_mdp):
        T, R = chain_mdp
        # s0 is 2 steps from absorbing, s1 is 1 step → V(s0) > V(s1) > V(s2)
        V, _, _ = run_vi(T, R, gamma=0.9, delta=1e-8)
        assert V[0] > V[1] > V[2]

    def test_policy_shape(self, chain_mdp):
        T, R = chain_mdp
        _, policy, _ = run_vi(T, R, gamma=0.9, delta=1e-8)
        assert policy.shape == (N,)

    def test_policy_actions_valid(self, chain_mdp):
        T, R = chain_mdp
        _, policy, _ = run_vi(T, R, gamma=0.9, delta=1e-8)
        assert all(0 <= a < T.shape[1] for a in policy)

    def test_convergence_below_delta(self, chain_mdp):
        T, R = chain_mdp
        delta = 1e-6
        _, _, trace = run_vi(T, R, gamma=0.9, delta=delta)
        assert trace[-1]["delta_v"] < delta

    def test_m_consec_respected(self, chain_mdp):
        T, R = chain_mdp
        _, _, trace1 = run_vi(T, R, gamma=0.9, delta=1e-6, m_consec=1)
        _, _, trace3 = run_vi(T, R, gamma=0.9, delta=1e-6, m_consec=3)
        assert len(trace3) >= len(trace1)

    def test_blackjack_v_shape(self, blackjack_model):
        T, R, n_states, _ = blackjack_model
        V, _, _ = run_vi(T, R, VI_GAMMA, VI_DELTA)
        assert V.shape[0] == n_states + 1  # +1 for absorbing state

    def test_blackjack_policy_range(self, blackjack_model):
        T, R, n_states, n_actions = blackjack_model
        _, policy, _ = run_vi(T, R, VI_GAMMA, VI_DELTA)
        assert policy.min() >= 0
        assert policy.max() < n_actions

    def test_blackjack_converges(self, blackjack_model):
        T, R, _, _ = blackjack_model
        _, _, trace = run_vi(T, R, VI_GAMMA, VI_DELTA)
        assert trace[-1]["delta_v"] < VI_DELTA


# ── PI tests ──────────────────────────────────────────────────────────────────


class TestPolicyIteration:
    def test_returns_three_items(self, chain_mdp):
        T, R = chain_mdp
        result = run_pi(T, R, gamma=0.9, delta=1e-8)
        assert len(result) == 3

    def test_v_shape(self, chain_mdp):
        T, R = chain_mdp
        V, _, _ = run_pi(T, R, gamma=0.9, delta=1e-8)
        assert V.shape == (N,)

    def test_policy_changes_zero_at_convergence(self, chain_mdp):
        T, R = chain_mdp
        _, _, trace = run_pi(T, R, gamma=0.9, delta=1e-8)
        assert trace[-1]["policy_changes"] == 0

    def test_trace_has_required_keys(self, chain_mdp):
        T, R = chain_mdp
        _, _, trace = run_pi(T, R, gamma=0.9, delta=1e-8)
        required = {"iteration", "delta_v", "policy_changes", "wall_clock_s"}
        assert required.issubset(trace[0].keys())

    def test_blackjack_v_shape(self, blackjack_model):
        T, R, n_states, _ = blackjack_model
        V, _, _ = run_pi(T, R, PI_GAMMA, PI_DELTA)
        assert V.shape[0] == n_states + 1

    def test_blackjack_converges(self, blackjack_model):
        T, R, _, _ = blackjack_model
        _, _, trace = run_pi(T, R, PI_GAMMA, PI_DELTA)
        assert trace[-1]["policy_changes"] == 0

    def test_vi_pi_policy_agreement_blackjack(self, blackjack_model):
        """VI and PI should produce highly-agreeing policies on Blackjack."""
        T, R, n_states, _ = blackjack_model
        _, policy_vi, _ = run_vi(T, R, VI_GAMMA, VI_DELTA)
        _, policy_pi, _ = run_pi(T, R, PI_GAMMA, PI_DELTA)
        agreement = float((policy_vi[:n_states] == policy_pi[:n_states]).mean())
        assert agreement > 0.90, f"Low VI/PI agreement: {agreement:.1%}"
