"""CartPole observation discretizer.

Maps continuous (x, xdot, theta, thetadot) observations to a single integer
state index in [0, n_states). Bin edges are defined in config.py and
reported verbatim in the reproducibility sheet.
"""

from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np

from src.config import (
    CARTPOLE_BINS,
    CARTPOLE_CLAMPS,
    CARTPOLE_THETA_EDGES,
    CARTPOLE_THETADOT_EDGES,
    CARTPOLE_X_EDGES,
    CARTPOLE_XDOT_EDGES,
)

# Canonical edge order: (x, xdot, theta, thetadot)
_EDGES: list[list[float]] = [
    CARTPOLE_X_EDGES,
    CARTPOLE_XDOT_EDGES,
    CARTPOLE_THETA_EDGES,
    CARTPOLE_THETADOT_EDGES,
]

_CLAMP_ORDER: list[str] = ["x", "xdot", "theta", "thetadot"]


def _bin_value(value: float, edges: Sequence[float]) -> int:
    """Map a clamped value to a 0-indexed bin in [0, n_bins-1].

    Uses searchsorted on the internal edges (excluding outer boundaries).
    """
    return int(np.searchsorted(edges[1:-1], value, side="right"))


class CartPoleDiscretizer:
    """Discretizes CartPole-v1 observations into integer state indices.

    Attributes:
        n_states: Total number of discrete states (product of all bin counts).
        bin_edges: Dict mapping dimension name to its edge list.
    """

    def __init__(self, grid_config: dict | None = None) -> None:
        """
        Args:
            grid_config: Optional dict with keys bins, x_edges, xdot_edges,
                         theta_edges, thetadot_edges. If None, uses the default
                         grid from config.py (CARTPOLE_BINS, CARTPOLE_*_EDGES).
        """
        if grid_config is None:
            self._bins: tuple[int, ...] = CARTPOLE_BINS
            self._edges: list[list[float]] = [list(e) for e in _EDGES]
        else:
            self._bins = tuple(grid_config["bins"])
            self._edges = [
                list(grid_config["x_edges"]),
                list(grid_config["xdot_edges"]),
                list(grid_config["theta_edges"]),
                list(grid_config["thetadot_edges"]),
            ]
        self._clamps: list[tuple[float, float]] = [
            CARTPOLE_CLAMPS[k] for k in _CLAMP_ORDER
        ]
        self.n_states: int = int(np.prod(self._bins))
        self.bin_edges: dict[str, list[float]] = dict(zip(_CLAMP_ORDER, self._edges))

    def obs_to_state(self, obs: np.ndarray) -> int:
        """Convert a continuous observation to a discrete state index.

        Args:
            obs: Array of shape (4,) — (x, xdot, theta, thetadot).

        Returns:
            Integer state index in [0, n_states).
        """
        idx = 0
        for dim, (value, edges, clamp, n_bins) in enumerate(
            zip(obs, self._edges, self._clamps, self._bins)
        ):
            clamped = float(np.clip(value, clamp[0], clamp[1]))
            bin_idx = _bin_value(clamped, edges)
            bin_idx = int(np.clip(bin_idx, 0, n_bins - 1))
            idx = idx * n_bins + bin_idx
        return idx

    def enumerate_all_states(self):
        """Yield (bin_tuple, state_index) for every possible discrete state.

        Useful for verifying full coverage of [0, n_states).
        """
        for bins in itertools.product(*[range(n) for n in self._bins]):
            idx = 0
            for b, n in zip(bins, self._bins):
                idx = idx * n + b
            yield bins, idx
