"""Regenerate the paper's figures and tables from results/summary.json.

Reads the frozen evidence set only — runs no models, touches no harness.
    uv run python scripts/generate_figures.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from visualization import figures, tables


def main() -> int:
    figs = figures.generate_all()
    tabs = tables.generate_all()
    print(f"figures -> report/figures/: {', '.join(figs)}")
    print(f"tables  -> report/tables/:  {', '.join(t + '.md' for t in tabs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
