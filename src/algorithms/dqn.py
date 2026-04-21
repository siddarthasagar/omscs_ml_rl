"""DQN and Double DQN for CartPole (numpy-only implementation).

Implements vanilla DQN (Mnih et al., 2015) and the Double DQN variant
(van Hasselt et al., 2016) using a 2-hidden-layer MLP with Adam optimizer,
experience replay, and a hard-copy target network.

Double DQN ablation: online network selects the greedy action; target network
evaluates its Q-value. This decoupling reduces the maximisation bias present
in vanilla DQN.

No external deep-learning libraries required — numpy only.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass
class DQNConfig:
    """Hyperparameters for a single DQN run."""

    # Network architecture
    obs_dim: int = 4  # CartPole raw observation dim
    hidden_dim: int = 64
    n_actions: int = 2

    # Replay buffer
    replay_size: int = 10_000
    batch_size: int = 64
    train_start: int = 1_000  # minimum replay fill before training starts
    update_every: int = 4  # train the network every N environment steps

    # Target network
    target_update_steps: int = 500  # hard-copy period (environment steps)

    # Optimiser
    lr: float = 5e-4
    gamma: float = 0.99
    max_grad_norm: float = 10.0  # gradient-norm clip threshold; 0.0 = disabled

    # Exploration: linear epsilon decay over total environment steps
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 10_000

    # Convergence: plateau in running-mean episode length (same logic as tabular)
    convergence_window: int = 100  # episode window
    convergence_delta: float = 10.0  # CartPole ep-length scale (~2 % of 500)
    convergence_m: int = 3

    # State normalisation: divide each feature by its clamp magnitude
    # (x, xdot, theta, thetadot) → approximately in [-1, 1]
    obs_scale: tuple[float, ...] = (2.4, 3.0, 0.2095, 3.5)


# ── 2-layer MLP with Adam ─────────────────────────────────────────────────────


class _MLP:
    """Two-hidden-layer ReLU MLP with Adam optimiser (pure numpy)."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        n_actions: int,
        rng: np.random.Generator,
    ) -> None:
        def _w(fan_in: int, fan_out: int) -> np.ndarray:
            """He (Kaiming) normal initialisation scaled for ReLU."""
            return rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)

        self.W1 = _w(obs_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = _w(hidden_dim, hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.W3 = _w(hidden_dim, n_actions)
        self.b3 = np.zeros(n_actions)

        # Adam moments (m = first, v = second) and step counter
        shapes = [
            self.W1.shape,
            self.b1.shape,
            self.W2.shape,
            self.b2.shape,
            self.W3.shape,
            self.b3.shape,
        ]
        self._m: list[np.ndarray] = [np.zeros(s) for s in shapes]
        self._v: list[np.ndarray] = [np.zeros(s) for s in shapes]
        self._t: int = 0

    def _params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. x: (batch, obs_dim) → (batch, n_actions).

        np.errstate suppresses spurious FP exception flags that BLAS routines
        can raise in intermediate accumulators even when the final result is
        finite and correct (known numpy/BLAS behaviour).
        """
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            a1 = np.maximum(0.0, x @ self.W1 + self.b1)
            a2 = np.maximum(0.0, a1 @ self.W2 + self.b2)
            return a2 @ self.W3 + self.b3

    def train_step(
        self,
        obs: np.ndarray,  # (batch, obs_dim)
        actions: np.ndarray,  # (batch,) int
        targets: np.ndarray,  # (batch,) float — TD targets
        lr: float,
        max_grad_norm: float = 10.0,
    ) -> float:
        """MSE loss on selected (s, a) pairs; update via Adam. Returns scalar loss."""
        batch = len(obs)

        # Forward + backward pass (errstate: see forward() docstring)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            z1 = obs @ self.W1 + self.b1  # (batch, h)
            a1 = np.maximum(0.0, z1)
            z2 = a1 @ self.W2 + self.b2  # (batch, h)
            a2 = np.maximum(0.0, z2)
            q = a2 @ self.W3 + self.b3  # (batch, n_actions)

            # Error only on the chosen action
            q_taken = q[np.arange(batch), actions]  # (batch,)
            err = q_taken - targets  # (batch,)
            loss = 0.5 * float(np.mean(err**2))

            # Backprop: gradient flows only through the chosen action column
            dq = np.zeros_like(q)
            dq[np.arange(batch), actions] = err / batch  # (batch, n_actions)

            dW3 = a2.T @ dq  # (h, n_actions)
            db3 = dq.sum(axis=0)
            da2 = dq @ self.W3.T  # (batch, h)

            dz2 = da2 * (z2 > 0.0)
            dW2 = a1.T @ dz2
            db2 = dz2.sum(axis=0)
            da1 = dz2 @ self.W2.T

            dz1 = da1 * (z1 > 0.0)
            dW1 = obs.T @ dz1
            db1 = dz1.sum(axis=0)

        grads = [dW1, db1, dW2, db2, dW3, db3]

        # Gradient-norm clipping: prevents exploding gradients during early training
        if max_grad_norm > 0.0:
            total_norm = float(np.sqrt(sum(float(np.sum(g**2)) for g in grads)))
            if total_norm > max_grad_norm:
                clip = max_grad_norm / (total_norm + 1e-8)
                grads = [g * clip for g in grads]

        params = self._params()

        # Adam update (β₁=0.9, β₂=0.999, ε=1e-8)
        self._t += 1
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        bc1 = 1.0 - beta1**self._t
        bc2 = 1.0 - beta2**self._t
        for i, (g, m, v) in enumerate(zip(grads, self._m, self._v)):
            m[:] = beta1 * m + (1.0 - beta1) * g
            v[:] = beta2 * v + (1.0 - beta2) * g**2
            params[i] -= lr * (m / bc1) / (np.sqrt(v / bc2) + eps_adam)

        return loss

    def copy_weights_from(self, other: _MLP) -> None:
        """Hard-copy all weights from *other* into self (target-network update)."""
        self.W1[:] = other.W1
        self.b1[:] = other.b1
        self.W2[:] = other.W2
        self.b2[:] = other.b2
        self.W3[:] = other.W3
        self.b3[:] = other.b3


# ── Experience replay buffer ──────────────────────────────────────────────────


class _ReplayBuffer:
    """Fixed-capacity circular replay buffer (numpy arrays)."""

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._capacity = capacity
        self._size = 0
        self._ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        i = self._ptr
        self._obs[i] = obs
        self._next_obs[i] = next_obs
        self._actions[i] = action
        self._rewards[i] = reward
        self._dones[i] = float(done)
        self._ptr = (i + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = rng.integers(0, self._size, size=batch_size)
        return (
            self._obs[idx].astype(np.float64),
            self._actions[idx],
            self._rewards[idx].astype(np.float64),
            self._next_obs[idx].astype(np.float64),
            self._dones[idx].astype(np.float64),
        )

    def __len__(self) -> int:
        return self._size


# ── Helpers ───────────────────────────────────────────────────────────────────


def _normalize(obs: np.ndarray, scale: tuple[float, ...]) -> np.ndarray:
    """Divide each observation feature by its scale and clip to [-3, 3].

    CartPole velocities (cart_vel, pole_angular_vel) are unbounded — they can
    spike to very large values in the steps just before termination. Clipping
    after division bounds the network inputs while preserving all information
    within 3× the normal operating range.
    """
    return np.clip(obs / np.array(scale, dtype=np.float64), -3.0, 3.0)


def _check_convergence(
    window_means: list[float],
    delta: float,
    m: int,
) -> bool:
    """Plateau rule: last *m* consecutive window-pair differences all < *delta*."""
    if len(window_means) < m + 1:
        return False
    recent = window_means[-(m + 1) :]
    return all(abs(recent[i + 1] - recent[i]) < delta for i in range(m))


# ── Main training function ────────────────────────────────────────────────────


def run_dqn(
    env: gym.Env,
    config: DQNConfig,
    n_episodes: int,
    seed: int,
    double_dqn: bool = False,
    log_interval: int = 200,
    logger: logging.Logger | None = None,
) -> tuple[list[dict], _MLP]:
    """Train DQN or Double DQN on *env* (CartPole) and return (trace, network).

    Args:
        env: Gymnasium environment returning continuous observations.
        config: DQN hyperparameters.
        n_episodes: Episode training budget.
        seed: RNG seed for full reproducibility.
        double_dqn: If True, use the Double DQN target (reduces overestimation bias).
        log_interval: Emit a progress log line every this many episodes.
        logger: Optional logger; silent if None.

    Returns:
        trace: List of per-episode dicts with keys:
            episode, ep_len, epsilon, wall_clock_s,
            window_mean (float at multiples of convergence_window, else None).
        online_net: Trained Q-network.
    """
    variant_tag = "Double DQN" if double_dqn else "DQN"
    rng = np.random.default_rng(seed)

    online_net = _MLP(config.obs_dim, config.hidden_dim, config.n_actions, rng)
    target_net = _MLP(config.obs_dim, config.hidden_dim, config.n_actions, rng)
    target_net.copy_weights_from(online_net)

    buffer = _ReplayBuffer(config.replay_size, config.obs_dim)

    trace: list[dict] = []
    window_returns: list[float] = []
    window_means: list[float] = []
    step = 0
    t0 = time.perf_counter()

    for ep in range(1, n_episodes + 1):
        obs_raw, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        obs = _normalize(np.asarray(obs_raw, dtype=np.float64), config.obs_scale)
        done = False
        ep_len = 0

        while not done:
            # Linear epsilon decay over total environment steps
            frac = min(step / max(1, config.eps_decay_steps), 1.0)
            epsilon = config.eps_start + (config.eps_end - config.eps_start) * frac

            if rng.random() < epsilon:
                action = int(rng.integers(0, config.n_actions))
            else:
                q_vals = online_net.forward(obs[None])  # (1, n_actions)
                action = int(np.argmax(q_vals[0]))

            next_raw, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_obs = _normalize(
                np.asarray(next_raw, dtype=np.float64), config.obs_scale
            )

            buffer.add(obs, action, float(reward), next_obs, done)
            obs = next_obs
            step += 1
            ep_len += 1

            # Train every update_every steps once the buffer is warm
            if len(buffer) >= config.train_start and step % config.update_every == 0:
                s, a, r, ns, d = buffer.sample(config.batch_size, rng)

                if double_dqn:
                    # Online net selects action; target net evaluates its value
                    best_a = np.argmax(online_net.forward(ns), axis=1)
                    q_next = target_net.forward(ns)[np.arange(len(a)), best_a]
                else:
                    q_next = np.max(target_net.forward(ns), axis=1)

                td_targets = r + config.gamma * q_next * (1.0 - d)
                online_net.train_step(s, a, td_targets, config.lr, config.max_grad_norm)

            # Hard target-network update
            if step % config.target_update_steps == 0:
                target_net.copy_weights_from(online_net)

        # End-of-episode bookkeeping
        frac = min(step / max(1, config.eps_decay_steps), 1.0)
        epsilon = config.eps_start + (config.eps_end - config.eps_start) * frac

        window_returns.append(float(ep_len))
        record: dict = {
            "episode": ep,
            "ep_len": ep_len,
            "epsilon": round(epsilon, 4),
            "wall_clock_s": round(time.perf_counter() - t0, 4),
            "window_mean": None,
        }

        if ep % config.convergence_window == 0:
            wm = float(np.mean(window_returns[-config.convergence_window :]))
            window_means.append(wm)
            record["window_mean"] = wm

            if logger and ep % log_interval == 0:
                logger.info(
                    "  %s ep %d/%d  window_mean=%.1f  ε=%.4f",
                    variant_tag,
                    ep,
                    n_episodes,
                    wm,
                    epsilon,
                )

            if _check_convergence(
                window_means, config.convergence_delta, config.convergence_m
            ):
                if logger:
                    logger.info(
                        "  %s converged at episode %d (window_mean plateau)",
                        variant_tag,
                        ep,
                    )
                trace.append(record)
                break

        elif logger and ep % log_interval == 0:
            logger.info(
                "  %s ep %d/%d  ep_len=%d  ε=%.4f",
                variant_tag,
                ep,
                n_episodes,
                ep_len,
                epsilon,
            )

        trace.append(record)

    return trace, online_net


def evaluate_dqn_greedy(
    net: _MLP,
    config: DQNConfig,
    n_episodes: int,
    seed: int,
) -> list[float]:
    """Evaluate a trained DQN policy greedily (ε=0) and return episode lengths.

    Uses a fresh environment and RNG so evaluation is independent of training.
    Episode length = episode return for CartPole (reward=1 per non-terminal step).

    Args:
        net: Trained online Q-network.
        config: DQNConfig supplying obs_scale for normalisation.
        n_episodes: Number of evaluation episodes.
        seed: RNG seed for env resets (distinct from training seed).

    Returns:
        List of episode lengths, one per episode.
    """
    import gymnasium as gym

    rng = np.random.default_rng(seed)
    env = gym.make("CartPole-v1")
    ep_lens: list[float] = []

    for _ in range(n_episodes):
        obs_raw, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        obs = _normalize(np.asarray(obs_raw, dtype=np.float64), config.obs_scale)
        done = False
        ep_len = 0

        while not done:
            q_vals = net.forward(obs[None])
            action = int(np.argmax(q_vals[0]))
            next_raw, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            obs = _normalize(np.asarray(next_raw, dtype=np.float64), config.obs_scale)
            ep_len += 1

        ep_lens.append(float(ep_len))

    env.close()
    return ep_lens
