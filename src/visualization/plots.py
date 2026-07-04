"""Publication figure primitives — style, validated palette, shared helpers.

Palette matches the project's design language (pitch green / sage / amber on
paper) and is validated categorical-safe: lightness band and chroma floor
pass; worst adjacent CVD ΔE 19.6 > 12. Sage and amber sit below 3:1 contrast
against the paper surface, so every mark carries a direct value label (the
relief rule). Figures are static print PNGs; no interaction layer applies.
"""

from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

# project palette (light mode), assigned by role.
INK = "#141814"
MUTED = "#5d635b"
GRID = "#e7eae2"
LINE = "#d9ddd4"
SURFACE = "#f6f7f4"
PITCH = "#1e754a"  # primary series / state / improvement
SAGE = "#57a87e"  # second series / identity increment
AMBER = "#c98a3b"  # third series / conditions increment
BALL = "#b23a2e"  # status: regression / overfit / decision bar (always labelled)
GRAY = "#9a9992"  # reference / null
# ordinal pitch-green ramp (light->dark = worse->better), lightness-monotone
GREEN_RAMP = ["#b7d4c3", "#83b598", "#4c9370", "#1e754a"]

FIG_W, FIG_H, DPI = 7.4, 4.6, 200


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 11,
            "font.family": "DejaVu Sans",
            "axes.edgecolor": LINE,
            "axes.linewidth": 0.8,
            "axes.labelcolor": MUTED,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.titlesize": 13,
            "axes.titleweight": "normal",
            "axes.labelsize": 10,
            "figure.dpi": DPI,
        }
    )


def new_axes(title: str, subtitle: str | None = None) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_title(title, loc="left", pad=16 if subtitle else 8, color=INK)
    if subtitle:
        ax.text(
            0.0,
            1.018,
            subtitle,
            transform=ax.transAxes,
            fontsize=9.5,
            color=MUTED,
            ha="left",
            va="bottom",
        )
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
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
