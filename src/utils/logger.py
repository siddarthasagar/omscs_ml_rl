"""Standardised per-run logging for RL experiment scripts."""

import logging
from pathlib import Path


def configure_logger(run_id: str, *, log_dir: Path | None = None) -> logging.Logger:
    """
    Configure a named logger that writes to {log_dir}/{run_id}.log and stdout.

    Args:
        run_id:  Phase identifier, e.g. "phase2". Log file is {log_dir}/{run_id}.log
                 and is overwritten on each run.
        log_dir: Directory for the log file. Defaults to ``artifacts/logs`` relative
                 to the working directory. Pass a tmp-path value in tests to keep
                 smoke-test output out of the canonical artifact tree.

    Returns:
        Configured Logger instance. Subsequent calls with the same run_id are
        idempotent — handlers are not duplicated.
    """
    if log_dir is None:
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
