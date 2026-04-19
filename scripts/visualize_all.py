"""Regenerate all phase figures from saved checkpoints.

Skips phases whose checkpoint does not yet exist — safe to run at any point
in the pipeline. Only phases that have been computed are rendered.

Usage:
    uv run python scripts/visualize_all.py              # all available phases
    uv run python scripts/visualize_all.py phase3       # single phase only
    make viz                                            # also wipes artifacts/figures first
"""

import importlib.util
import sys
from pathlib import Path

# Registry — extend with each new phase as it is implemented.
# (phase_id, checkpoint_path, script_file)
_REGISTRY: list[tuple[str, str, str]] = [
    (
        "phase2",
        "artifacts/metadata/phase2.json",
        "scripts/run_phase_2_vi_pi_blackjack.py",
    ),
    (
        "phase3",
        "artifacts/metadata/phase3.json",
        "scripts/run_phase_3_vi_pi_cartpole.py",
    ),
    (
        "phase4",
        "artifacts/metadata/phase4.json",
        "scripts/run_phase_4_model_free_blackjack.py",
    ),
]


def _load_script(script_path: str):
    """Load a phase script as a module by file path."""
    p = Path(script_path)
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    phase_filter = sys.argv[1] if len(sys.argv) > 1 else None
    registry = [r for r in _REGISTRY if phase_filter is None or r[0] == phase_filter]

    if phase_filter and not registry:
        print(
            f"ERROR: '{phase_filter}' not found in registry. Known phases: {[r[0] for r in _REGISTRY]}"
        )
        sys.exit(1)

    rendered: list[str] = []
    skipped: list[str] = []

    for phase_id, checkpoint_str, script_file in registry:
        checkpoint = Path(checkpoint_str)
        if not checkpoint.exists():
            print(f"  skip  {phase_id} — checkpoint not found ({checkpoint})")
            skipped.append(phase_id)
            continue

        print(f"\n  {phase_id} — rendering from {checkpoint}")
        mod = _load_script(script_file)
        figs = mod.visualize(checkpoint)
        for f in figs:
            print(f"    {f}")
        rendered.append(phase_id)

    print(f"\n{'─' * 50}")
    print(f"Rendered : {rendered if rendered else '(none)'}")
    if skipped:
        print(f"Skipped  : {skipped}  (run the phase first)")


if __name__ == "__main__":
    main()
