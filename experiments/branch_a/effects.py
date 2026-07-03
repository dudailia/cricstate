"""TRAIN-ONLY shrunk player effects (Branch A spec, phase A0).

Per-player log-odds offsets over the frozen K=11 alphabet, estimated by
empirical-Bayes Dirichlet shrinkage toward the population (train-marginal)
distribution:

    p_tilde_j = (c_j + lam * p_bar) / (n_j + lam)
    offset_j  = ln(p_tilde_j) - ln(p_bar)

LOGGED ASSUMPTION (minimal): the spec's "prior variance lambda" is
parameterized as prior STRENGTH in pseudo-balls — the standard equivalent
Dirichlet form, where larger lambda = smaller prior variance = harder
shrinkage. It remains the only tunable.

Estimation sees the train fold ONLY (the API takes one fold and freezes the
result). Unseen players — and null IDs from the M1 observation gap — get the
population mean, i.e. a zero offset.
"""

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PlayerEffects:
    """Frozen after fit; applying to new data never mutates it."""

    lam: float
    n_classes: int
    population_logp: FloatArray  # [K]
    offsets: dict[str, FloatArray] = field(repr=False)  # player_id -> [K]

    def offset_matrix(self, ids: pl.Series) -> FloatArray:
        """[n, K] offsets; zeros for unseen or null IDs (population mean)."""
        out = np.zeros((len(ids), self.n_classes))
        for i, pid in enumerate(ids):
            if pid is not None:
                found = self.offsets.get(pid)
                if found is not None:
                    out[i] = found
        return out

    def seen(self, ids: pl.Series) -> NDArray[np.bool_]:
        """Per-row: was this ID observed in train? (null counts as unseen)."""
        return np.array([pid is not None and pid in self.offsets for pid in ids])


def fit_effects(
    train_ids: pl.Series, train_y: NDArray[np.int64], lam: float, n_classes: int
) -> PlayerEffects:
    """Estimate shrunk per-player offsets from the TRAIN fold only.

    train_ids: player id per delivery (nullable — null rows contribute to the
    population but to no player). train_y: class index per delivery.
    """
    if lam <= 0:
        raise ValueError("lambda must be positive (pseudo-ball prior strength)")
    y = np.asarray(train_y, dtype=np.int64)
    pop_counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    # Laplace floor so log(p_bar) is finite even for classes absent in train
    p_bar = (pop_counts + 1.0) / (pop_counts.sum() + n_classes)
    log_p_bar = np.log(p_bar)

    frame = pl.DataFrame({"pid": train_ids, "y": y}).drop_nulls("pid")
    counts = (
        frame.group_by("pid", "y").len().sort("pid", "y")  # sorted → deterministic
    )
    offsets: dict[str, FloatArray] = {}
    totals: dict[str, float] = {}
    class_counts: dict[str, FloatArray] = {}
    for pid, cls, n in counts.iter_rows():
        if pid not in class_counts:
            class_counts[pid] = np.zeros(n_classes)
            totals[pid] = 0.0
        class_counts[pid][cls] += n
        totals[pid] += n
    for pid in sorted(class_counts):
        c = class_counts[pid]
        p_shrunk = (c + lam * p_bar) / (totals[pid] + lam)
        offsets[pid] = np.log(p_shrunk) - log_p_bar
    return PlayerEffects(lam=lam, n_classes=n_classes, population_logp=log_p_bar, offsets=offsets)
