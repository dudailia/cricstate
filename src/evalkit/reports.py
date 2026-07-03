"""Reporting utilities — P2 scope: reliability-diagram plotting only.

Leaderboard generation, bucket tables and bootstrap bands arrive in P4.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless + deterministic
import matplotlib.pyplot as plt

from evalkit.metrics import ReliabilityData


def plot_reliability(
    data: ReliabilityData,
    path: Path,
    title: str,
    ece_value: float | None = None,
) -> None:
    """Reliability diagram: predicted vs observed frequency over equal-mass bins."""
    fig, (ax, ax_hist) = plt.subplots(
        2,
        1,
        figsize=(5, 6),
        gridspec_kw={"height_ratios": [4, 1]},
        sharex=True,
        constrained_layout=True,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#999999", label="perfect")
    ax.plot(
        data.p_mean,
        data.y_mean,
        marker="o",
        markersize=3.5,
        linewidth=1.2,
        color="#2d5a87",
        label="model",
    )
    label = title if ece_value is None else f"{title}  (ECE={ece_value:.4f})"
    ax.set_title(label, fontsize=10)
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left")
    ax_hist.bar(data.p_mean, data.weight, width=0.02, color="#2d5a87")
    ax_hist.set_xlabel("predicted probability")
    ax_hist.set_ylabel("mass")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
