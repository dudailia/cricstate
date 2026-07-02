from datetime import date

import polars as pl
import pytest

from evalkit.splits import SplitIntegrityError, check_integrity, metadata_lines, split_metadata


def mk_matches(rows: list[tuple[str, str, str, date]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "match_id": [r[0] for r in rows],
            "fmt": [r[1] for r in rows],
            "temporal_split": [r[2] for r in rows],
            "start_date": [r[3] for r in rows],
        }
    )


def mk_deliveries(matches: pl.DataFrame, per_match: int = 3) -> pl.DataFrame:
    return matches.select("match_id", "temporal_split").join(
        pl.DataFrame({"ball": list(range(per_match))}), how="cross"
    )


GOOD = mk_matches(
    [
        ("m1", "t20", "train", date(2020, 1, 1)),
        ("m2", "odi", "train", date(2021, 6, 1)),
        ("m3", "t20", "val", date(2022, 1, 1)),
        ("m4", "t20", "test", date(2023, 1, 1)),
    ]
)


def test_good_split_passes_and_reports() -> None:
    dels = mk_deliveries(GOOD)
    check_integrity(GOOD, dels)
    meta = split_metadata(GOOD, dels)
    assert meta.match_counts == {"train": 2, "val": 1, "test": 1}
    assert meta.delivery_counts == {"train": 6, "val": 3, "test": 3}
    assert meta.date_ranges["train"] == (date(2020, 1, 1), date(2021, 6, 1))
    assert meta.fmt_match_counts["t20"] == {"train": 1, "val": 1, "test": 1}
    assert meta.fmt_match_counts["odi"] == {"train": 1}
    text = "\n".join(metadata_lines(meta))
    assert "2020-01-01 → 2021-06-01" in text
    assert "| odi | 1 | 0 | 0 |" in text


def test_match_in_two_splits_fails() -> None:
    dels = pl.concat(
        [
            mk_deliveries(GOOD),
            pl.DataFrame({"match_id": ["m1"], "temporal_split": ["val"], "ball": [99]}),
        ]
    )
    with pytest.raises(SplitIntegrityError, match="more than one split"):
        check_integrity(GOOD, dels)


def test_train_val_date_overlap_fails() -> None:
    bad = mk_matches(
        [
            ("m1", "t20", "train", date(2022, 6, 1)),  # after val's min
            ("m3", "t20", "val", date(2022, 1, 1)),
            ("m4", "t20", "test", date(2023, 1, 1)),
        ]
    )
    with pytest.raises(SplitIntegrityError, match="max\\(train\\)"):
        check_integrity(bad, mk_deliveries(bad))


def test_val_test_date_overlap_fails() -> None:
    bad = mk_matches(
        [
            ("m1", "t20", "train", date(2020, 1, 1)),
            ("m3", "t20", "val", date(2023, 6, 1)),  # after test's min
            ("m4", "t20", "test", date(2023, 1, 1)),
        ]
    )
    with pytest.raises(SplitIntegrityError, match="max\\(val\\)"):
        check_integrity(bad, mk_deliveries(bad))


def test_empty_split_fails() -> None:
    bad = GOOD.filter(pl.col("temporal_split") != "test")
    with pytest.raises(SplitIntegrityError, match="empty"):
        check_integrity(bad, mk_deliveries(bad))


def test_matches_deliveries_split_disagreement_fails() -> None:
    dels = mk_deliveries(GOOD).with_columns(
        pl.when(pl.col("match_id") == "m4")
        .then(pl.lit("val"))
        .otherwise(pl.col("temporal_split"))
        .alias("temporal_split")
    )
    with pytest.raises(SplitIntegrityError):
        check_integrity(GOOD, dels)
