"""B0 — Marginal baseline (SPEC_M2 §5).

T1: train class frequencies (per fmt, Laplace alpha = 1).
T2: constant train base rate of first-batting-team wins per fmt — a MATCH
base rate (one vote per match via match_id dedupe), not delivery-weighted.
"""

import numpy as np
import polars as pl
from numpy.typing import NDArray

from evalkit.models.base import CLASSES


class B0MarginalT1:
    name = "B0_marginal"
    version = "1.0"

    def __init__(self) -> None:
        self.probs = np.full(len(CLASSES), 1.0 / len(CLASSES))

    def fit(self, train: pl.DataFrame, val: pl.DataFrame) -> None:
        counts = np.bincount(train["y"].to_numpy(), minlength=len(CLASSES))
        self.probs = (counts + 1.0) / (counts.sum() + len(CLASSES))  # Laplace alpha=1

    def predict_proba(self, df: pl.DataFrame) -> NDArray[np.float64]:
        return np.tile(self.probs, (df.height, 1))


class B0MarginalT2:
    name = "B0_marginal"
    version = "1.0"

    def __init__(self) -> None:
        self.base_rate = 0.5

    def fit(self, train: pl.DataFrame, val: pl.DataFrame) -> None:
        per_match = train.group_by("match_id").agg(pl.col("y").first())
        rate = per_match["y"].mean()
        assert isinstance(rate, int | float)
        self.base_rate = float(rate)

    def predict_proba(self, df: pl.DataFrame) -> NDArray[np.float64]:
        return np.full(df.height, self.base_rate)
