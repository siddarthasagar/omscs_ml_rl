"""Gate 1 tests — environment setup and CartPole model validation.

Run with:  make gate1   (or: uv run pytest tests/test_envs.py)
"""

import json

import numpy as np
import pytest

from src.config import (
    CARTPOLE_BINS,
    CARTPOLE_CLAMPS,
    CARTPOLE_MODEL_MIN_VISITS,
    METADATA_DIR,
)
from src.envs.cartpole_discretizer import CartPoleDiscretizer


# ─────────────────────────── CartPole discretizer ────────────────────────────


@pytest.fixture(scope="module")
def discretizer():
    return CartPoleDiscretizer()


def test_discretizer_n_states(discretizer):
    expected = int(np.prod(CARTPOLE_BINS))
    assert discretizer.n_states == expected


def test_discretizer_obs_to_state_range(discretizer):
    """All random observations map to a valid state index in [0, n_states)."""
    rng = np.random.default_rng(0)
    clamp_vals = list(CARTPOLE_CLAMPS.values())
    for _ in range(10_000):
        obs = np.array([rng.uniform(lo, hi) for lo, hi in clamp_vals])
        idx = discretizer.obs_to_state(obs)
        assert 0 <= idx < discretizer.n_states, (
            f"State index {idx} out of range for obs={obs}"
        )


def test_discretizer_clamp_respected(discretizer):
    """Observations outside the clamp range are binned the same as the boundary."""
    # Extreme values beyond clamp should land in the boundary bins.
    extreme_low = np.array([-99.0, -99.0, -99.0, -99.0])
    extreme_high = np.array([99.0, 99.0, 99.0, 99.0])
    clamp_low = np.array([lo for lo, _ in CARTPOLE_CLAMPS.values()])
    clamp_high = np.array([hi for _, hi in CARTPOLE_CLAMPS.values()])

    assert discretizer.obs_to_state(extreme_low) == discretizer.obs_to_state(clamp_low)
    assert discretizer.obs_to_state(extreme_high) == discretizer.obs_to_state(
        clamp_high - 1e-9
    )


def test_discretizer_full_state_space_enumerable(discretizer):
    """enumerate_all_states() covers every index in [0, n_states) exactly once."""
    seen = set()
    for _bins, idx in discretizer.enumerate_all_states():
        assert 0 <= idx < discretizer.n_states
        seen.add(idx)
    assert len(seen) == discretizer.n_states, (
        f"Expected {discretizer.n_states} unique states, got {len(seen)}"
    )


# ─────────────────────────── Blackjack model ─────────────────────────────────


@pytest.fixture(scope="module")
def blackjack_model():
    from src.envs.blackjack_env import get_blackjack_model

    return get_blackjack_model()


def test_blackjack_model_shapes(blackjack_model):
    T, R, n_states, n_actions = blackjack_model
    # T includes the absorbing terminal state at index n_states
    assert T.shape == (n_states + 1, n_actions, n_states + 1), (
        f"T shape mismatch: {T.shape}"
    )
    assert R.shape == (n_states + 1, n_actions), f"R shape mismatch: {R.shape}"


def test_blackjack_row_stochastic(blackjack_model):
    T, _R, _n_states, _n_actions = blackjack_model
    row_sums = T.sum(axis=2)
    assert np.allclose(row_sums, 1.0, atol=1e-6), (
        f"T rows not stochastic; max deviation={np.abs(row_sums - 1.0).max():.2e}"
    )


def test_blackjack_reward_range(blackjack_model):
    _T, R, _n_states, _n_actions = blackjack_model
    assert R.min() >= -1.0 - 1e-9, f"R below -1: {R.min()}"
    assert R.max() <= 1.5 + 1e-9, f"R above 1.5: {R.max()}"


# ─────────────────────────── CartPole T/R model ──────────────────────────────


@pytest.fixture(scope="module")
def cartpole_model(discretizer):
    """Build a small CartPole model for fast testing (50k steps)."""
    from src.envs.cartpole_model import build_cartpole_model

    return build_cartpole_model(
        discretizer=discretizer,
        rollout_steps=50_000,
        min_visits=CARTPOLE_MODEL_MIN_VISITS,
        seed=0,
    )


def test_cartpole_model_shapes(cartpole_model, discretizer):
    T = cartpole_model["T"]
    R = cartpole_model["R"]
    n_aug = discretizer.n_states + 1
    n_actions = 2
    assert T.shape == (n_aug, n_actions, n_aug), f"T shape: {T.shape}"
    assert R.shape == (n_aug, n_actions), f"R shape: {R.shape}"


def test_cartpole_model_non_negative(cartpole_model):
    T = cartpole_model["T"]
    assert T.min() >= 0.0, f"T has negative entries: {T.min()}"


def test_cartpole_model_row_stochastic(cartpole_model):
    T = cartpole_model["T"]
    row_sums = T.sum(axis=2)
    assert np.allclose(row_sums, 1.0, atol=1e-6), (
        f"T rows not stochastic; max deviation={np.abs(row_sums - 1.0).max():.2e}"
    )


def test_cartpole_absorbing_state(cartpole_model):
    T = cartpole_model["T"]
    R = cartpole_model["R"]
    s_term = cartpole_model["absorbing_state_index"]
    n_actions = 2
    for a in range(n_actions):
        assert T[s_term, a, s_term] == pytest.approx(1.0), (
            f"Absorbing state self-loop broken for action {a}"
        )
        assert R[s_term, a] == pytest.approx(0.0), (
            f"Absorbing state reward non-zero for action {a}"
        )


def test_cartpole_coverage_threshold(cartpole_model):
    """Coverage should be > 0% even with small rollout; target >=80% for full run."""
    coverage = cartpole_model["coverage_pct"]
    # Weak check for the small test rollout (50k steps) — just verify it's computed.
    assert 0.0 <= coverage <= 1.0, f"coverage_pct out of range: {coverage}"


# ─────────────────────── Phase 1 persisted artifact validation ───────────────
# These tests load the actual outputs written by `make phase1` / run_phase_1_env_setup.py.
# They are skipped (not failed) when the artifact has not been generated yet.


@pytest.fixture(scope="module")
def phase1_json():
    path = METADATA_DIR / "phase1.json"
    if not path.exists():
        pytest.skip("phase1.json not found — run `make phase1` first")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def cartpole_npz():
    path = METADATA_DIR / "cartpole_model.npz"
    if not path.exists():
        pytest.skip("cartpole_model.npz not found — run `make phase1` first")
    return np.load(path)


def test_phase1_json_schema(phase1_json):
    """phase1.json contains the required top-level keys and sub-keys."""
    assert "blackjack" in phase1_json
    assert "cartpole" in phase1_json

    bj = phase1_json["blackjack"]
    for key in ("n_states", "n_actions", "model_source"):
        assert key in bj, f"Missing blackjack key: {key}"

    cp = phase1_json["cartpole"]
    for key in (
        "bins",
        "n_states",
        "clamps",
        "bin_edges",
        "model_rollout_steps",
        "model_seed",
        "model_source",
        "coverage_pct",
        "smoothed_pct",
        "absorbing_state_index",
    ):
        assert key in cp, f"Missing cartpole key: {key}"


def test_phase1_json_model_seed_recorded(phase1_json):
    """model_seed must be present so the artifact is reproducible."""
    seed = phase1_json["cartpole"]["model_seed"]
    assert isinstance(seed, int), f"model_seed should be int, got {type(seed)}"


def test_phase1_cartpole_npz_shapes(cartpole_npz, phase1_json):
    """Saved T and R have shapes consistent with phase1.json n_states."""
    n_states = phase1_json["cartpole"]["n_states"]
    n_aug = n_states + 1
    n_actions = 2
    T = cartpole_npz["T"]
    R = cartpole_npz["R"]
    assert T.shape == (n_aug, n_actions, n_aug), f"Saved T shape: {T.shape}"
    assert R.shape == (n_aug, n_actions), f"Saved R shape: {R.shape}"


def test_phase1_cartpole_npz_row_stochastic(cartpole_npz):
    """Saved T rows sum to 1."""
    T = cartpole_npz["T"]
    row_sums = T.sum(axis=2)
    assert np.allclose(row_sums, 1.0, atol=1e-6), (
        f"Saved T rows not stochastic; max dev={np.abs(row_sums - 1.0).max():.2e}"
    )


def test_phase1_coverage_warning(phase1_json):
    """Emit a pytest warning (not a failure) when coverage is below the 80% target."""
    coverage = phase1_json["cartpole"]["coverage_pct"]
    assert 0.0 <= coverage <= 1.0, f"coverage_pct out of range: {coverage}"
    if coverage < 0.80:
        pytest.warns(
            UserWarning,
            match="coverage",
        )
        import warnings

        warnings.warn(
            f"CartPole model coverage is {coverage * 100:.1f}% (target ≥80%). "
            "Consider increasing CARTPOLE_MODEL_ROLLOUT_STEPS.",
            UserWarning,
            stacklevel=1,
        )
