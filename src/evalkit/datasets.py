"""Task dataset assembly (SPEC_M2 §1, §2).

Emits per-(task, fmt, split) assembly frames: FEATURE_COLUMNS + fmt + y +
match_id. Test-split discipline is structural: `load_split` refuses "test"
unless `allow_test=True`, which only the P4 leaderboard runner passes.
"""

import polars as pl

from evalkit.features import build_features
from evalkit.labels import Disposition, build_labels
from evalkit.models.base import CLASS_INDEX
from evalkit.splits import load_deliveries, load_matches


class TestSplitTouchedError(Exception):
    """The test split is evaluated once, in the P4 leaderboard run — not here."""


def _slice(deliveries: pl.DataFrame, fmt: str, split: str) -> pl.DataFrame:
    return deliveries.filter((pl.col("fmt") == fmt) & (pl.col("temporal_split") == split))


def assemble_t1(deliveries: pl.DataFrame, fmt: str, split: str) -> pl.DataFrame:
    """T1: per-ball outcome. Super-over/no-result rows are excluded via the
    M1 excluded_from_tuples flag; every remaining delivery is a row."""
    rows = _slice(deliveries, fmt, split).filter(~pl.col("excluded_from_tuples"))
    y = rows["outcome_class"].replace_strict(CLASS_INDEX, return_dtype=pl.Int64)
    return build_features(rows).with_columns(y.alias("y"), rows["match_id"].alias("match_id"))


def assemble_t2(
    deliveries: pl.DataFrame, matches: pl.DataFrame, fmt: str, split: str
) -> pl.DataFrame:
    """T2: win probability. LABELED matches only; super-over deliveries out."""
    labels = build_labels(matches, deliveries)
    labeled = labels.filter(pl.col("disposition") == Disposition.LABELED.value)
    rows = (
        _slice(deliveries, fmt, split)
        .filter(pl.col("innings_idx") <= 2)  # super-over deliveries excluded
        .join(labeled.select("match_id", "y"), on="match_id", how="inner")
    )
    return build_features(rows).with_columns(
        rows["y"].cast(pl.Int64).alias("y"), rows["match_id"].alias("match_id")
    )


class DataBundle:
    """Loads the corpus once; hands out assembly frames per (task, fmt, split)."""

    def __init__(self) -> None:
        self.deliveries = load_deliveries()
        self.matches = load_matches()

    def load_split(
        self, task: str, fmt: str, split: str, *, allow_test: bool = False
    ) -> pl.DataFrame:
        if split == "test" and not allow_test:
            raise TestSplitTouchedError("test split is frozen until the P4 leaderboard run")
        if task == "t1":
            return assemble_t1(self.deliveries, fmt, split)
        if task == "t2":
            return assemble_t2(self.deliveries, self.matches, fmt, split)
        raise ValueError(f"unknown task {task!r}")
