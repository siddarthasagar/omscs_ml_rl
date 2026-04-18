from pathlib import Path

# ── Seeds ─────────────────────────────────────────────────────────────────────
SEEDS: list[int] = [42, 43, 44, 45, 46]

# ── VI / PI ───────────────────────────────────────────────────────────────────
VI_GAMMA: float = 0.99
VI_DELTA: float = 1e-6
PI_GAMMA: float = 0.99
PI_DELTA: float = 1e-6
VI_PI_CONSEC_SWEEPS: int = 1  # consecutive sweeps below delta to declare convergence

# ── CartPole discretization — default grid ────────────────────────────────────
CARTPOLE_BINS: tuple[int, ...] = (3, 3, 8, 12)  # (x, xdot, theta, thetadot)

CARTPOLE_CLAMPS: dict[str, tuple[float, float]] = {
    "x": (-2.4, 2.4),
    "xdot": (-3.0, 3.0),
    "theta": (-0.2, 0.2),
    "thetadot": (-3.5, 3.5),
}

# Bin edges — x and xdot are uniform; theta and thetadot are non-uniform (finer near zero).
# Reported verbatim in the reproducibility sheet.
CARTPOLE_X_EDGES: list[float] = [-2.4, -0.8, 0.8, 2.4]  # 3 bins
CARTPOLE_XDOT_EDGES: list[float] = [-3.0, -1.0, 1.0, 3.0]  # 3 bins
CARTPOLE_THETA_EDGES: list[float] = [  # 8 bins
    -0.200,
    -0.100,
    -0.050,
    -0.025,
    0.000,
    0.025,
    0.050,
    0.100,
    0.200,
]
CARTPOLE_THETADOT_EDGES: list[float] = [  # 12 bins
    -3.50,
    -2.00,
    -1.00,
    -0.50,
    -0.25,
    -0.10,
    0.00,
    0.10,
    0.25,
    0.50,
    1.00,
    2.00,
    3.50,
]

# Grid configurations for Phase 3 ablation study (coarse → default → fine).
# x and xdot use uniform spacing; theta and thetadot are non-uniform (finer near zero).
CARTPOLE_GRID_CONFIGS: dict[str, dict] = {
    "coarse": {
        "bins": (1, 1, 6, 12),
        "x_edges": [-2.4, 2.4],
        "xdot_edges": [-3.0, 3.0],
        "theta_edges": [-0.200, -0.100, -0.025, 0.000, 0.025, 0.100, 0.200],
        "thetadot_edges": [
            -3.50,
            -2.00,
            -1.00,
            -0.50,
            -0.25,
            -0.10,
            0.00,
            0.10,
            0.25,
            0.50,
            1.00,
            2.00,
            3.50,
        ],
        # 72 SA pairs — 500 k steps is sufficient for >85 % coverage
        "rollout_steps": 500_000,
    },
    "default": {
        "bins": (3, 3, 8, 12),
        "x_edges": CARTPOLE_X_EDGES,
        "xdot_edges": CARTPOLE_XDOT_EDGES,
        "theta_edges": CARTPOLE_THETA_EDGES,
        "thetadot_edges": CARTPOLE_THETADOT_EDGES,
        # 1 728 SA pairs — 2 M steps targets ≥80 % coverage
        "rollout_steps": 2_000_000,
    },
    "fine": {
        "bins": (5, 5, 10, 16),
        "x_edges": [-2.4, -1.44, -0.48, 0.48, 1.44, 2.4],
        "xdot_edges": [-3.0, -1.8, -0.6, 0.6, 1.8, 3.0],
        "theta_edges": [
            -0.200,
            -0.100,
            -0.050,
            -0.025,
            -0.010,
            0.000,
            0.010,
            0.025,
            0.050,
            0.100,
            0.200,
        ],
        "thetadot_edges": [
            -3.50,
            -2.50,
            -1.75,
            -1.00,
            -0.50,
            -0.25,
            -0.10,
            -0.05,
            0.00,
            0.05,
            0.10,
            0.25,
            0.50,
            1.00,
            1.75,
            2.50,
            3.50,
        ],
        # 8 000 SA pairs — 5 M steps; coverage will be partial, reported explicitly
        "rollout_steps": 5_000_000,
    },
}

# ── CartPole model estimation ─────────────────────────────────────────────────
CARTPOLE_MODEL_ROLLOUT_STEPS: int = 2_000_000
CARTPOLE_MODEL_MIN_VISITS: int = (
    5  # (s, a) pairs below this count are flagged as sparse
)
CARTPOLE_MODEL_SEED: int = (
    0  # fixed seed for model-construction rollout; persisted in phase1.json
)

# ── VI / PI hyperparameter validation sweep ───────────────────────────────────
# Gamma sweep: hold delta fixed at reference, vary gamma.
# Delta sweep: hold gamma fixed at reference, vary delta.
VI_PI_HP_GAMMA_VALUES: list[float] = [0.85, 0.90, 0.95, 0.99]
VI_PI_HP_DELTA_VALUES: list[float] = [1e-2, 1e-3, 1e-4, 1e-6]

# ── Model-free convergence criterion (running-mean plateau) ───────────────────
RL_CONVERGENCE_WINDOW: int = 100  # W: window size in episodes
RL_CONVERGENCE_DELTA: float = 0.01  # minimum improvement between consecutive windows
RL_CONVERGENCE_M: int = 3  # consecutive window-pairs below delta to declare convergence

# ── Model-free defaults / HP search starting point ───────────────────────────
RL_GAMMA: float = 0.99
RL_ALPHA_START: float = 0.5
RL_ALPHA_END: float = 0.1
RL_EPS_START: float = 1.0
RL_EPS_END: float = 0.01
RL_EPS_DECAY_STEPS: int = 10_000

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR: Path = Path("data")
ARTIFACTS_DIR: Path = Path("artifacts")
METRICS_DIR: Path = ARTIFACTS_DIR / "metrics"
FIGURES_DIR: Path = ARTIFACTS_DIR / "figures"
METADATA_DIR: Path = ARTIFACTS_DIR / "metadata"
LOGS_DIR: Path = ARTIFACTS_DIR / "logs"
