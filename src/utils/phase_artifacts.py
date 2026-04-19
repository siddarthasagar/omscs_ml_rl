"""Shared utilities for the phase-gate artifact lifecycle.

Every phase script exposes two top-level functions:

    run() -> Path
        Execute computation, write all required artifacts, validate outputs,
        and return the checkpoint JSON path.

    visualize(checkpoint_path: Path) -> list[Path]
        Reload artifacts from disk only and render all figures for that phase.

``__main__`` calls ``run()`` then immediately ``visualize(checkpoint_path)``,
but ``visualize()`` must not consume runtime objects from ``run()``.
"""

import json
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = "1.0"


@dataclass
class PhasePaths:
    """Standard artifact paths derived from phase_id + slug."""

    phase_id: str  # e.g. "phase2"
    slug: str  # e.g. "vi_pi_blackjack"
    metrics_dir: Path
    figures_dir: Path
    metadata_dir: Path
    logs_dir: Path | None = None  # if None, configure_logger uses its own default

    @property
    def checkpoint_path(self) -> Path:
        return self.metadata_dir / f"{self.phase_id}.json"

    @property
    def phase_dir(self) -> str:
        return f"{self.phase_id}_{self.slug}"

    def makedirs(self) -> None:
        """Create metrics, figures, metadata, and (if set) logs directories."""
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        if self.logs_dir is not None:
            self.logs_dir.mkdir(parents=True, exist_ok=True)


def resolve_phase_paths(phase_id: str, slug: str) -> PhasePaths:
    """Return PhasePaths built from config path constants."""
    from src.config import FIGURES_DIR, LOGS_DIR, METADATA_DIR, METRICS_DIR

    phase_dir = f"{phase_id}_{slug}"
    return PhasePaths(
        phase_id=phase_id,
        slug=slug,
        metrics_dir=METRICS_DIR / phase_dir,
        figures_dir=FIGURES_DIR / phase_dir,
        metadata_dir=METADATA_DIR,
        logs_dir=LOGS_DIR,
    )


def write_checkpoint_json(checkpoint: dict, path: Path) -> None:
    """Write checkpoint dict to JSON, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint_json(path: Path) -> dict:
    """Load and return a checkpoint JSON as a dict."""
    with open(path) as f:
        return json.load(f)


def validate_required_outputs(paths: list[Path]) -> None:
    """Raise FileNotFoundError if any required artifact is missing post-run."""
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        lines = "\n".join(f"  {p}" for p in missing)
        raise FileNotFoundError(f"Required artifacts missing after run():\n{lines}")
