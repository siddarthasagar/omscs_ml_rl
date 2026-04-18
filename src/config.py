from pathlib import Path

# ── A. Reproducibility and Assignment Contract ────────────────────────────────
# Assignment contract: FAQ asks for around 5 independent seeds per compared model.
# This is a reproducibility constant, not a tuned hyperparameter.
SEEDS: list[int] = [42, 43, 44, 45, 46]

# ── B. DP Planning Defaults ───────────────────────────────────────────────────
# Reference choices used for the main VI/PI runs.
# Chosen as default reference: high gamma (long planning horizon) + tight delta.
# Validated in a separate HP sweep (see Section E).
VI_GAMMA: float = 0.99
VI_DELTA: float = 1e-6
PI_GAMMA: float = 0.99
PI_DELTA: float = 1e-6

# Number of consecutive sweeps below delta before VI declares convergence.
# 1 = standard; increase only if oscillation is observed.
VI_PI_CONSEC_SWEEPS: int = 1

# ── C. Environment Abstraction Constants ──────────────────────────────────────
# CartPole default grid — encodes domain assumptions about discretization.
# These are environment-driven, not tuning results.
# (x, xdot, theta, thetadot): theta/thetadot use finer bins near zero where
# control matters most; x/xdot use uniform spacing.
CARTPOLE_BINS: tuple[int, ...] = (3, 3, 8, 12)

# Clamp ranges: values outside these are mapped to the boundary bin.
# Chosen to cover the observable range without bloating state count.
CARTPOLE_CLAMPS: dict[str, tuple[float, float]] = {
    "x": (-2.4, 2.4),
    "xdot": (-3.0, 3.0),
    "theta": (-0.2, 0.2),
    "thetadot": (-3.5, 3.5),
}

# Bin edges — reported verbatim in the reproducibility sheet.
# x and xdot: uniform spacing (position/velocity less critical than angle).
# theta and thetadot: non-uniform, finer near zero for better control resolution.
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

# ── D. Model-Build Budgets ────────────────────────────────────────────────────
# Control how empirical CartPole transition models are constructed.
# Chosen as compute budgets: large enough for honest coverage reporting,
# feasible within assignment limits.
CARTPOLE_MODEL_ROLLOUT_STEPS: int = 2_000_000
CARTPOLE_MODEL_MIN_VISITS: int = 5  # (s,a) pairs below this are flagged as sparse
CARTPOLE_MODEL_SEED: int = 0  # fixed seed for rollout; persisted in phase1.json

# ── E. Hyperparameter Validation Ranges ───────────────────────────────────────
# Predefined search space to satisfy the assignment requirement of validating
# at least two hyperparameters per model.
# These are validation ranges, not hand-picked winning values.
# Gamma sweep: hold delta fixed at reference, vary gamma.
# Delta sweep: hold gamma fixed at reference, vary delta.
VI_PI_HP_GAMMA_VALUES: list[float] = [0.85, 0.90, 0.95, 0.99]
VI_PI_HP_DELTA_VALUES: list[float] = [1e-2, 1e-3, 1e-4, 1e-6]

# ── F. Evaluation Budgets ─────────────────────────────────────────────────────
# Main tables use the full budget; HP sweeps use a lighter budget for fast
# screening and are interpreted directionally rather than as exact replicas.

# Blackjack: main eval budget chosen to reduce evaluation noise for final results.
BJ_EVAL_EPISODES_MAIN: int = 1_000
# Blackjack: HP sweep budget — lighter screening pass, not final quality estimate.
BJ_EVAL_EPISODES_HP: int = 500

# CartPole: main eval budget (shorter episodes; 100 is sufficient to distinguish policies).
CP_EVAL_EPISODES_MAIN: int = 100
# CartPole: HP sweep budget — fast screening.
CP_EVAL_EPISODES_HP: int = 50

# ── G. Model-Free Defaults / HP Search Starting Point ────────────────────────
# These are starting-point choices for model-free learning phases.
# Will be validated in dedicated HP sweeps (phases 4+).
RL_CONVERGENCE_WINDOW: int = 100  # W: window size in episodes
RL_CONVERGENCE_DELTA: float = 0.01  # minimum improvement between consecutive windows
RL_CONVERGENCE_M: int = 3  # consecutive window-pairs below delta → converged

RL_GAMMA: float = 0.99
RL_ALPHA_START: float = 0.5
RL_ALPHA_END: float = 0.1
RL_EPS_START: float = 1.0
RL_EPS_END: float = 0.01
RL_EPS_DECAY_STEPS: int = 10_000

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR: Path = Path("data")
ARTIFACTS_DIR: Path = Path("artifacts")
METRICS_DIR: Path = ARTIFACTS_DIR / "metrics"
FIGURES_DIR: Path = ARTIFACTS_DIR / "figures"
METADATA_DIR: Path = ARTIFACTS_DIR / "metadata"
LOGS_DIR: Path = ARTIFACTS_DIR / "logs"
