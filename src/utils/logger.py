"""Standardised per-run logging for RL experiment scripts."""

import logging
from pathlib import Path


def configure_logger(run_id: str) -> logging.Logger:
    """
    Configure a named logger that writes to artifacts/logs/{run_id}.log and stdout.

    Args:
        run_id: Phase identifier, e.g. "phase2_20260405T143000". Log is written to
                artifacts/logs/{run_id}.log and overwrites on each run.

    Returns:
        Configured Logger instance. Subsequent calls with the same run_id are
        idempotent — handlers are not duplicated.
    """
    log_dir = Path("artifacts") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(run_id)
    if logger.handlers:
        return logger  # already configured — avoid duplicate handlers

    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    fh = logging.FileHandler(log_dir / f"{run_id}.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger
