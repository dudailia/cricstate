"""Temporal split interface: boundaries, integrity checks, metadata emit (SPEC_M2 §2).

The split itself is baked into the M1 tables (`temporal_split` column); this
module never re-derives it. It verifies integrity and reports the boundary
dates and per-split counts that go into every leaderboard's metadata.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

SPLITS = ("train", "val", "test")

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "v1"


class SplitIntegrityError(Exception):
    """A split invariant from SPEC_M2 §2 does not hold."""


@dataclass(frozen=True)
class SplitMetadata:
    date_ranges: dict[str, tuple[date, date]]  # split → (min, max) start_date
    match_counts: dict[str, int]
    delivery_counts: dict[str, int]
    fmt_match_counts: dict[str, dict[str, int]]  # fmt → split → n_matches


def load_matches(path: Path | None = None) -> pl.DataFrame:
    return pl.read_parquet(path or DATA_DIR / "matches.parquet")


def load_deliveries(path: Path | None = None) -> pl.DataFrame:
    return pl.read_parquet(path or DATA_DIR / "deliveries.parquet")


def check_integrity(matches: pl.DataFrame, deliveries: pl.DataFrame) -> None:
    """Raise SplitIntegrityError on any violation. Cheap; run before every eval."""
    unknown = set(matches["temporal_split"].unique().to_list()) - set(SPLITS)
    if unknown:
        raise SplitIntegrityError(f"unknown split labels: {sorted(unknown)}")

    multi = (
        deliveries.group_by("match_id")
        .agg(pl.col("temporal_split").n_unique().alias("n"))
        .filter(pl.col("n") > 1)
    )
    if multi.height:
        raise SplitIntegrityError(
            f"{multi.height} match_id(s) appear in more than one split, "
            f"e.g. {multi['match_id'][0]!r}"
        )

    joined = (
        deliveries.select("match_id", "temporal_split")
        .unique()
        .join(matches.select("match_id", "temporal_split"), on="match_id", how="inner")
    )
    disagree = joined.filter(pl.col("temporal_split") != pl.col("temporal_split_right"))
    if disagree.height:
        raise SplitIntegrityError(
            f"{disagree.height} match_id(s) have different splits in matches vs deliveries"
        )

    bounds = {s: _date_range(matches, s) for s in SPLITS}
    if not bounds["train"][1] < bounds["val"][0]:
        raise SplitIntegrityError(
            f"max(train)={bounds['train'][1]} not before min(val)={bounds['val'][0]}"
        )
    if not bounds["val"][1] < bounds["test"][0]:
        raise SplitIntegrityError(
            f"max(val)={bounds['val'][1]} not before min(test)={bounds['test'][0]}"
        )


def _date_range(matches: pl.DataFrame, split: str) -> tuple[date, date]:
    col = matches.filter(pl.col("temporal_split") == split)["start_date"]
    lo, hi = col.min(), col.max()
    if not isinstance(lo, date) or not isinstance(hi, date):
        raise SplitIntegrityError(f"split {split!r} is empty")
    return lo, hi


def split_metadata(matches: pl.DataFrame, deliveries: pl.DataFrame) -> SplitMetadata:
    """Boundary dates + per-split counts. Call after check_integrity."""
    date_ranges: dict[str, tuple[date, date]] = {}
    match_counts: dict[str, int] = {}
    delivery_counts: dict[str, int] = {}
    for s in SPLITS:
        m = matches.filter(pl.col("temporal_split") == s)
        lo, hi = m["start_date"].min(), m["start_date"].max()
        assert isinstance(lo, date) and isinstance(hi, date)  # non-empty per integrity
        date_ranges[s] = (lo, hi)
        match_counts[s] = m.height
        delivery_counts[s] = deliveries.filter(pl.col("temporal_split") == s).height
    fmt_counts: dict[str, dict[str, int]] = {}
    for row in (
        matches.group_by("fmt", "temporal_split").len().sort("fmt", "temporal_split").to_dicts()
    ):
        fmt_counts.setdefault(row["fmt"], {})[row["temporal_split"]] = row["len"]
    return SplitMetadata(
        date_ranges=date_ranges,
        match_counts=match_counts,
        delivery_counts=delivery_counts,
        fmt_match_counts=fmt_counts,
    )


def metadata_lines(meta: SplitMetadata) -> list[str]:
    """Markdown block for leaderboard metadata (SPEC_M2 §2: must be printed)."""
    lines = [
        "| split | matches | deliveries | start_date range |",
        "|---|---|---|---|",
    ]
    for s in SPLITS:
        lo, hi = meta.date_ranges[s]
        lines.append(f"| {s} | {meta.match_counts[s]} | {meta.delivery_counts[s]} | {lo} → {hi} |")
    lines += ["", "| fmt | train | val | test |", "|---|---|---|---|"]
    for fmt in sorted(meta.fmt_match_counts):
        c = meta.fmt_match_counts[fmt]
        lines.append(f"| {fmt} | {c.get('train', 0)} | {c.get('val', 0)} | {c.get('test', 0)} |")
    return lines
