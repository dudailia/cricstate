"""Match-level paired bootstrap (SPEC_M2 §6).

Resamples MATCHES with replacement, B = 10,000, fixed seed; percentile 95%
CIs for metrics and for paired deltas. Ball-level bootstrap is forbidden —
within-match dependence makes it fake precision. Vectorized over per-match
aggregates: metric_b = sum(n_m * l_m) / sum(n_m) over the drawn multiset, computed
via multinomial count vectors and one matmul.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

B_RESAMPLES = 10_000
BOOTSTRAP_SEED = 90210


@dataclass(frozen=True)
class CI:
    point: float
    lo: float
    hi: float

    def excludes_zero(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0


def per_match_losses(losses: FloatArray, match_ids: pl.Series) -> tuple[FloatArray, FloatArray]:
    """(sum_loss_m, n_m) per match, ordered by first appearance (stable)."""
    df = pl.DataFrame({"match_id": match_ids, "loss": losses})
    agg = df.group_by("match_id", maintain_order=True).agg(
        pl.col("loss").sum().alias("s"), pl.len().alias("n")
    )
    return agg["s"].to_numpy().astype(np.float64), agg["n"].to_numpy().astype(np.float64)


def _count_matrix(n_matches: int, rng: np.random.Generator) -> FloatArray:
    """[B, n_matches] multinomial draw counts ≡ resampling matches w/ replacement."""
    return rng.multinomial(n_matches, np.full(n_matches, 1.0 / n_matches), size=B_RESAMPLES).astype(
        np.float64
    )


def bootstrap_metric(sums: FloatArray, counts: FloatArray, draws: FloatArray) -> CI:
    """CI for a delivery-weighted mean metric under match resampling."""
    num = draws @ sums
    den = draws @ counts
    samples = num / den
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return CI(point=float(sums.sum() / counts.sum()), lo=float(lo), hi=float(hi))


def bootstrap_paired_delta(
    sums_a: FloatArray, sums_b: FloatArray, counts: FloatArray, draws: FloatArray
) -> CI:
    """CI for metric(A) - metric(B) with the SAME match draws (paired)."""
    den = draws @ counts
    samples = (draws @ sums_a) / den - (draws @ sums_b) / den
    lo, hi = np.percentile(samples, [2.5, 97.5])
    point = float(sums_a.sum() / counts.sum() - sums_b.sum() / counts.sum())
    return CI(point=point, lo=float(lo), hi=float(hi))


def make_draws(n_matches: int, seed: int = BOOTSTRAP_SEED) -> FloatArray:
    return _count_matrix(n_matches, np.random.default_rng(seed))
