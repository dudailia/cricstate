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


def test_actual_parquet_schemas_equal_goldens() -> None:
    from evalkit.freeze import load_golden, schema_as_dict

    for name in ("matches", "deliveries", "players"):
        actual = schema_as_dict(pl.read_parquet(V1 / f"{name}.parquet").schema)
        assert actual == load_golden(name), f"{name} schema drifted from golden"


def test_pinned_corpus_hash_recomputes_from_parquet(matches: pl.DataFrame) -> None:
    """The build's corpus hash is reconstructible from per-match stream hashes
    in parquet row order (= sorted-file order); drift = red build."""
    import hashlib

    from evalkit.freeze import PINNED_CORPUS_HASH

    h = hashlib.sha256()
    for mid, sh in matches.select("match_id", "stream_hash").iter_rows():
        h.update(f"{mid}:{sh}\n".encode())
    assert h.hexdigest() == PINNED_CORPUS_HASH


def test_pinned_labels_hash_recomputes(matches: pl.DataFrame, deliveries: pl.DataFrame) -> None:
    from evalkit.freeze import PINNED_LABELS_HASH
    from evalkit.labels import build_labels, labels_hash

    assert labels_hash(build_labels(matches, deliveries)) == PINNED_LABELS_HASH


def test_real_corpus_labels_total_partition(
    matches: pl.DataFrame, deliveries: pl.DataFrame
) -> None:
    """P2 property on real data: every in-scope match gets exactly one disposition."""
    from evalkit.labels import Disposition, build_labels, labels_hash

    labels = build_labels(matches, deliveries)
    assert labels.height == matches.height
    assert set(labels["disposition"].unique().to_list()) <= {d.value for d in Disposition}
    labeled = labels.filter(pl.col("disposition") == Disposition.LABELED.value)
    assert labeled["y"].is_in([0, 1]).all()
    assert labels_hash(labels) == labels_hash(build_labels(matches, deliveries))


@pytest.mark.parametrize("fmt", ["t20", "odi"])
def test_b1_t2_monotone_by_construction(
    matches: pl.DataFrame, deliveries: pl.DataFrame, fmt: str
) -> None:
    """Amendment #3: PAVA-smoothed B1 must satisfy the §5 property exactly."""
    from evalkit.datasets import assemble_t2
    from evalkit.models.b1_table import B1TableT2
    from evalkit.monotonicity import check_b1_t2_monotonicity

    model = B1TableT2(fmt)
    model.fit(
        assemble_t2(deliveries, matches, fmt, "train"),
        assemble_t2(deliveries, matches, fmt, "val"),
    )
    assert check_b1_t2_monotonicity(model) == []


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
