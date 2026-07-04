"""Publication figure primitives — style, validated palette, shared helpers.

Palette is the dataviz reference categorical set, validated colorblind-safe
(worst adjacent CVD ΔE 16.2 > 12). Aqua sits below 3:1 contrast, so every bar
carries a direct value label (the relief rule). Figures are static print PNGs;
no interaction layer applies.
"""

from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

# dataviz reference palette (light mode), assigned by role.
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e7e6e2"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"  # improvement / primary series
AQUA = "#1baf7a"  # second series
RED = "#e34948"  # regression / overfit (status-adjacent, always labelled)
ORANGE = "#eb6834"
GRAY = "#9a9992"  # reference / null
# ordinal blue ramp (light->dark = worse->better), starts >= step 250 for 2:1
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab"]

FIG_W, FIG_H, DPI = 7.4, 4.6, 200


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 11,
            "font.family": "DejaVu Sans",
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "figure.dpi": DPI,
        }
    )


def new_axes(title: str, subtitle: str | None = None) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_title(title, loc="left", pad=14 if subtitle else 8, color=INK)
    if subtitle:
        ax.text(
            0.0,
            1.015,
            subtitle,
            transform=ax.transAxes,
            fontsize=9.5,
            color=MUTED,
            ha="left",
            va="bottom",
        )
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    return fig, ax


def footnote(fig: Any, meta: dict[str, Any], extra: str = "") -> None:
    txt = (
        f"cricstate {meta['corpus_tag']} · corpus {meta['corpus_hash'][:12]}… · "
        f"seed {meta['seed']} · match-level paired bootstrap "
        f"B={meta['bootstrap']['resamples']:,} · test evaluated once"
    )
    if extra:
        txt = extra + "\n" + txt
    fig.text(0.008, 0.008, txt, fontsize=7.2, color=MUTED, ha="left", va="bottom")


def save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path)
    plt.close(fig)
