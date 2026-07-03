"""T2 label resolution — FROZEN order per the P2 milestone prompt.

    (1) no-result/abandoned              -> EXCLUDE_NO_RESULT
    (2) outcome_winner set               -> y = 1[winner == first-batting team]
    (3) tie + exactly one of eliminator/bowl_out set
                                         -> y = 1[that team == first-batting team]
        (both set -> LabelResolutionError)
    (4) tie + neither set                -> EXCLUDE_TRUE_TIE (T2 only; T1 keeps match)
    (5) anything else                    -> LabelResolutionError

Labels come from parquet only (matches for outcomes, deliveries for the
first-batting team); evalkit never reads raw snapshot JSON.
"""

import hashlib
from enum import StrEnum

import polars as pl


class Disposition(StrEnum):
    LABELED = "LABELED"
    EXCLUDE_NO_RESULT = "EXCLUDE_NO_RESULT"
    EXCLUDE_TRUE_TIE = "EXCLUDE_TRUE_TIE"


class LabelResolutionError(Exception):
    """An outcome shape outside the frozen resolution order — stop, don't guess."""


def resolve_one(
    *,
    match_id: str,
    no_result: bool,
    outcome_result: str | None,
    outcome_winner: str | None,
    outcome_eliminator: str | None,
    outcome_bowl_out: str | None,
    first_batting_team: str | None,
) -> tuple[Disposition, int | None]:
    """Apply the frozen resolution order to one match. Total or raising."""
    if no_result:  # (1)
        return Disposition.EXCLUDE_NO_RESULT, None
    if outcome_winner is not None:  # (2)
        if first_batting_team is None:
            raise LabelResolutionError(f"{match_id}: winner set but no innings-1 rows")
        return Disposition.LABELED, int(outcome_winner == first_batting_team)
    if outcome_result == "tie":
        decided = [t for t in (outcome_eliminator, outcome_bowl_out) if t is not None]
        if len(decided) == 2:
            raise LabelResolutionError(f"{match_id}: tie with BOTH eliminator and bowl_out set")
        if len(decided) == 1:  # (3)
            if first_batting_team is None:
                raise LabelResolutionError(f"{match_id}: tie-breaker set but no innings-1 rows")
            return Disposition.LABELED, int(decided[0] == first_batting_team)
        return Disposition.EXCLUDE_TRUE_TIE, None  # (4)
    raise LabelResolutionError(  # (5)
        f"{match_id}: unmapped outcome shape "
        f"(result={outcome_result!r}, winner={outcome_winner!r}, "
        f"eliminator={outcome_eliminator!r}, bowl_out={outcome_bowl_out!r})"
    )


def first_batting_teams(deliveries: pl.DataFrame) -> pl.DataFrame:
    """match_id → first_batting_team, from innings-1 delivery rows (parquet only)."""
    return (
        deliveries.filter(pl.col("innings_idx") == 1)
        .group_by("match_id")
        .agg(pl.col("batting_team").first().alias("first_batting_team"))
    )


def build_labels(matches: pl.DataFrame, deliveries: pl.DataFrame) -> pl.DataFrame:
    """Per-match T2 labels: match_id, temporal_split, fmt, disposition, y.

    Deterministic: sorted by match_id; every in-scope match receives exactly
    one disposition; any unmapped shape raises LabelResolutionError.
    """
    joined = (
        matches.select(
            "match_id",
            "temporal_split",
            "fmt",
            "no_result",
            "outcome_result",
            "outcome_winner",
            "outcome_eliminator",
            "outcome_bowl_out",
        )
        .join(first_batting_teams(deliveries), on="match_id", how="left")
        .sort("match_id")
    )
    dispositions: list[str] = []
    ys: list[int | None] = []
    for row in joined.iter_rows(named=True):
        disp, y = resolve_one(
            match_id=row["match_id"],
            no_result=row["no_result"],
            outcome_result=row["outcome_result"],
            outcome_winner=row["outcome_winner"],
            outcome_eliminator=row["outcome_eliminator"],
            outcome_bowl_out=row["outcome_bowl_out"],
            first_batting_team=row["first_batting_team"],
        )
        dispositions.append(disp.value)
        ys.append(y)
    return joined.select("match_id", "temporal_split", "fmt").with_columns(
        pl.Series("disposition", dispositions, dtype=pl.Utf8),
        pl.Series("y", ys, dtype=pl.Int8),
    )


def labels_hash(labels: pl.DataFrame) -> str:
    """Determinism fingerprint over (match_id, disposition, y), sorted."""
    h = hashlib.sha256()
    for mid, disp, y in labels.sort("match_id").select("match_id", "disposition", "y").iter_rows():
        h.update(f"{mid}:{disp}:{y}\n".encode())
    return h.hexdigest()


def disposition_counts(labels: pl.DataFrame) -> pl.DataFrame:
    """Counts per (split, disposition) for the review-gate report."""
    return (
        labels.group_by("temporal_split", "disposition").len().sort("temporal_split", "disposition")
    )
