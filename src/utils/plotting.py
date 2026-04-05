"""Plotting utilities for RL experiment scripts.

Each plot_* function saves a PNG to the given out_dir and returns the path.
No bare plt.show() calls — all figures are saved to artifacts/figures/.
"""

from pathlib import Path


# Add plot_* functions here as phases are defined.
# Pattern:
#   def plot_<name>(df, ..., out_dir: Path) -> Path:
#       fig, ax = plt.subplots(...)
#       ...
#       path = out_dir / "<name>.png"
#       fig.savefig(path, dpi=150, bbox_inches="tight")
#       plt.close(fig)
#       return path
