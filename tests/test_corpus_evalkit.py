"""Real-corpus checks for evalkit P1 (marked `corpus`; needs data/v1)."""

from pathlib import Path

import polars as pl
import pytest

from evalkit.features import build_features, denylist_violations
from evalkit.splits import check_integrity, load_deliveries, load_matches, split_metadata

pytestmark = pytest.mark.corpus

V1 = Path(__file__).resolve().parents[1] / "data" / "v1"


@pytest.fixture(scope="module")
def matches() -> pl.DataFrame:
    if not (V1 / "matches.parquet").exists():
        pytest.skip("corpus parquet not present")
    return load_matches()


@pytest.fixture(scope="module")
def deliveries() -> pl.DataFrame:
    return load_deliveries()


def test_real_split_integrity(matches: pl.DataFrame, deliveries: pl.DataFrame) -> None:
    check_integrity(matches, deliveries)


def test_real_split_metadata_shape(matches: pl.DataFrame, deliveries: pl.DataFrame) -> None:
    meta = split_metadata(matches, deliveries)
    assert sum(meta.match_counts.values()) == matches.height
    assert sum(meta.delivery_counts.values()) == deliveries.height
    assert set(meta.fmt_match_counts) == {"t20", "odi"}


def test_features_build_over_val_slice(deliveries: pl.DataFrame) -> None:
    val = deliveries.filter(pl.col("temporal_split") == "val")
    out = build_features(val)
    assert out.height == val.height
    assert denylist_violations(out.columns) == []
    # no nulls anywhere in the emitted matrix
    assert out.null_count().sum_horizontal().item() == 0
    # chase rows carry real chase features; innings-1 rows carry sentinels
    chase = out.filter(pl.col("is_chase") == 1.0)
    first = out.filter(pl.col("is_chase") == 0.0)
    assert chase.height > 0 and first.height > 0
    assert (first["target_runs"] == 0.0).all()
    assert (chase["target_runs"] > 0.0).all()
