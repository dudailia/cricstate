from datetime import date

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from evalkit.labels import (
    Disposition,
    LabelResolutionError,
    build_labels,
    disposition_counts,
    labels_hash,
    resolve_one,
)


def resolve(
    *,
    no_result: bool = False,
    result: str | None = None,
    winner: str | None = None,
    eliminator: str | None = None,
    bowl_out: str | None = None,
    first: str | None = "A",
) -> tuple[Disposition, int | None]:
    return resolve_one(
        match_id="m",
        no_result=no_result,
        outcome_result=result,
        outcome_winner=winner,
        outcome_eliminator=eliminator,
        outcome_bowl_out=bowl_out,
        first_batting_team=first,
    )


# --- the seven mandated unit cases -------------------------------------------


def test_tie_plus_eliminator() -> None:
    assert resolve(result="tie", eliminator="A") == (Disposition.LABELED, 1)
    assert resolve(result="tie", eliminator="B") == (Disposition.LABELED, 0)


def test_tie_plus_bowl_out() -> None:
    assert resolve(result="tie", bowl_out="A") == (Disposition.LABELED, 1)
    assert resolve(result="tie", bowl_out="B") == (Disposition.LABELED, 0)


def test_true_tie_excluded_t2_only() -> None:
    assert resolve(result="tie") == (Disposition.EXCLUDE_TRUE_TIE, None)


def test_no_result_excluded() -> None:
    assert resolve(no_result=True, result="no result") == (
        Disposition.EXCLUDE_NO_RESULT,
        None,
    )


def test_win_by_runs() -> None:
    # first-batting team defends a total → winner is the first-batting side
    assert resolve(winner="A") == (Disposition.LABELED, 1)


def test_win_by_wickets() -> None:
    # chasing side wins → winner is not the first-batting side
    assert resolve(winner="B") == (Disposition.LABELED, 0)


def test_dl_win() -> None:
    assert resolve(winner="B", result=None) == (Disposition.LABELED, 0)


# --- raising branches ---------------------------------------------------------


def test_tie_with_both_tiebreakers_raises() -> None:
    with pytest.raises(LabelResolutionError, match="BOTH"):
        resolve(result="tie", eliminator="A", bowl_out="B")


def test_unmapped_shape_raises() -> None:
    with pytest.raises(LabelResolutionError, match="unmapped"):
        resolve(result="draw")


def test_winner_without_innings_rows_raises() -> None:
    with pytest.raises(LabelResolutionError, match="innings-1"):
        resolve(winner="A", first=None)


def test_no_result_beats_winner_in_resolution_order() -> None:
    # rule (1) fires before rule (2) by frozen order
    assert resolve(no_result=True, winner="A") == (Disposition.EXCLUDE_NO_RESULT, None)


# --- property test: total partition ------------------------------------------

team = st.sampled_from(["A", "B"])


@given(
    no_result=st.booleans(),
    result=st.sampled_from([None, "tie", "no result"]),
    winner=st.none() | team,
    eliminator=st.none() | team,
    bowl_out=st.none() | team,
)
def test_every_shape_gets_exactly_one_disposition(
    no_result: bool,
    result: str | None,
    winner: str | None,
    eliminator: str | None,
    bowl_out: str | None,
) -> None:
    try:
        disp, y = resolve(
            no_result=no_result,
            result=result,
            winner=winner,
            eliminator=eliminator,
            bowl_out=bowl_out,
        )
    except LabelResolutionError:
        # raising IS the specified behaviour for contradictory/unmapped shapes;
        # it must be one of the two frozen raise conditions
        assert (result == "tie" and eliminator and bowl_out and not no_result and not winner) or (
            result not in ("tie",) and winner is None and not no_result
        )
        return
    assert disp in set(Disposition)
    assert (y is not None) == (disp is Disposition.LABELED)
    assert y in (0, 1, None)


# --- frame-level build --------------------------------------------------------


def synthetic_tables() -> tuple[pl.DataFrame, pl.DataFrame]:
    matches = pl.DataFrame(
        {
            "match_id": ["m1", "m2", "m3", "m4"],
            "temporal_split": ["train", "train", "val", "test"],
            "fmt": ["t20", "t20", "odi", "t20"],
            "start_date": [date(2020, 1, 1)] * 4,
            "no_result": [False, True, False, False],
            "outcome_result": [None, "no result", "tie", "tie"],
            "outcome_winner": ["A", None, None, None],
            "outcome_eliminator": [None, None, None, "B"],
            "outcome_bowl_out": [None, None, None, None],
        }
    )
    deliveries = pl.DataFrame(
        {
            "match_id": ["m1", "m1", "m2", "m3", "m4"],
            "innings_idx": [1, 2, 1, 1, 1],
            "batting_team": ["A", "B", "X", "C", "A"],
        }
    )
    return matches, deliveries


def test_build_labels_and_counts() -> None:
    matches, deliveries = synthetic_tables()
    labels = build_labels(matches, deliveries)
    by_id = {r["match_id"]: r for r in labels.iter_rows(named=True)}
    assert by_id["m1"]["disposition"] == "LABELED" and by_id["m1"]["y"] == 1
    assert by_id["m2"]["disposition"] == "EXCLUDE_NO_RESULT"
    assert by_id["m3"]["disposition"] == "EXCLUDE_TRUE_TIE"
    assert by_id["m4"]["disposition"] == "LABELED" and by_id["m4"]["y"] == 0
    counts = disposition_counts(labels)
    assert (
        counts.filter((pl.col("temporal_split") == "train") & (pl.col("disposition") == "LABELED"))[
            "len"
        ].item()
        == 1
    )


def test_labels_hash_deterministic_and_order_insensitive() -> None:
    matches, deliveries = synthetic_tables()
    h1 = labels_hash(build_labels(matches, deliveries))
    h2 = labels_hash(build_labels(matches.reverse(), deliveries))
    assert h1 == h2
    assert len(h1) == 64
