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

# Canonical grid order for the Phase 3 ablation study — drives both experiment
# execution and figure ordering. Defined here (not in plotting.py) so that a
# visual refactor cannot silently change which grids are run.
CARTPOLE_GRID_NAMES: list[str] = ["coarse", "default", "fine"]

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

# ── H. Phase 4 — Blackjack Model-Free ────────────────────────────────────────
# State space: player_sum (4–21 = 18 values) × dealer_card (1–10 = 10 values)
#              × usable_ace (0/1 = 2 values) → capped at player_sum ≤ 21.
# Encoding: player_sum * 11 * 2 + dealer_card * 2 + int(usable_ace) → 0–703.
BJ_N_STATES: int = 704  # 22 * 11 * 2 (includes unreachable states; safe upper bound)
BJ_N_ACTIONS: int = 2  # 0 = Stick, 1 = Hit

# Training budget for main final runs (per algorithm × seed).
BJ_TRAIN_EPISODES: int = 500_000
# Logging interval for training progress lines.
BJ_LOG_INTERVAL: int = 5_000

# HP search: 3-stage progressive narrowing
# Stage 1: wide random search (fast screening)
BJ_HP_STAGE1_CONFIGS: int = 24  # candidate configs
BJ_HP_STAGE1_EPISODES: int = 20_000
# Stage 2: top-k refinement (medium budget)
BJ_HP_STAGE2_TOP_K: int = 8
BJ_HP_STAGE2_EPISODES: int = 50_000
# Stage 3: fine-grained perturbation around the winner
BJ_HP_STAGE3_TOP_K: int = 3
BJ_HP_STAGE3_EPISODES: int = 100_000

# Baseline (controlled) schedule — pre-specified neutral reference used for the
# controlled-regime comparison.  Chosen before HP search so neither algorithm
# has an advantage from tuned exploration.
# Matches the FAQ quick-start schedule: ε decays over 10k steps (same as
# RL_EPS_DECAY_STEPS default), α decays moderately over half the training budget.
BJ_BASELINE_ALPHA_START: float = 0.5
BJ_BASELINE_ALPHA_END: float = 0.01
BJ_BASELINE_ALPHA_DECAY_STEPS: int = 200_000
BJ_BASELINE_EPS_START: float = 1.0
BJ_BASELINE_EPS_END: float = 0.01
BJ_BASELINE_EPS_DECAY_STEPS: int = 10_000
BJ_BASELINE_GAMMA: float = 0.99

# Final-training parallelism: None = auto (min(n_jobs, cpu_count-1)), int = explicit cap.
# Set to 1 to force serial execution for debugging or single-core environments.
PHASE4_FINAL_TRAIN_MAX_WORKERS: int | None = None

# ── I. Phase 5 — CartPole Model-Free ─────────────────────────────────────────
# State space: determined by CartPoleDiscretizer (grid-dependent).
# Default grid (3,3,8,12) → 864 states.  n_states is looked up at runtime.
CP_N_ACTIONS: int = 2  # push-left / push-right

# Training budget for main final runs (per algorithm × seed).
# CartPole episodes are long (up to 500 steps); 20k episodes is sufficient for
# plateau-based convergence at the default grid resolution.
CP_TRAIN_EPISODES: int = 20_000

# HP search: 3-stage progressive narrowing (same structure as Phase 4).
CP_HP_STAGE1_CONFIGS: int = 24
CP_HP_STAGE1_EPISODES: int = 2_000
CP_HP_STAGE2_TOP_K: int = 8
CP_HP_STAGE2_EPISODES: int = 5_000
CP_HP_STAGE3_TOP_K: int = 3
CP_HP_STAGE3_EPISODES: int = 10_000

# Baseline (controlled) schedule — same neutral FAQ quick-start schedule as BJ.
# α: moderate decay over first half of training steps (est. ~200k steps at avg 20 ep len).
# ε: 1.0 → 0.01 over 10k steps (FAQ quick-start default), then fixed.
CP_BASELINE_ALPHA_START: float = 0.5
CP_BASELINE_ALPHA_END: float = 0.01
CP_BASELINE_ALPHA_DECAY_STEPS: int = 100_000
CP_BASELINE_EPS_START: float = 1.0
CP_BASELINE_EPS_END: float = 0.01
CP_BASELINE_EPS_DECAY_STEPS: int = 10_000
CP_BASELINE_GAMMA: float = 0.99

# Discretization study: run tuned HP winner on each grid, 5 seeds each.
# Budget per (grid, algo, seed): same as main final training.
CP_DISC_TRAIN_EPISODES: int = CP_TRAIN_EPISODES

PHASE5_FINAL_TRAIN_MAX_WORKERS: int | None = None
PHASE5_HP_SEARCH_MAX_WORKERS: int | None = None

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR: Path = Path("data")
ARTIFACTS_DIR: Path = Path("artifacts")
METRICS_DIR: Path = ARTIFACTS_DIR / "metrics"
FIGURES_DIR: Path = ARTIFACTS_DIR / "figures"
METADATA_DIR: Path = ARTIFACTS_DIR / "metadata"
LOGS_DIR: Path = ARTIFACTS_DIR / "logs"
