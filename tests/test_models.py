import numpy as np
import polars as pl
import pytest

from evalkit.canaries import ladder_inversion_canary
from evalkit.features import FEATURE_COLUMNS
from evalkit.models.b0_marginal import B0MarginalT1, B0MarginalT2
from evalkit.models.b1_table import (
    RATE_EDGES,
    B1TableT2,
    _Table,
    bucket_columns,
)
from evalkit.models.b2_logistic import B2Logistic
from evalkit.models.b3_gbm import B3Gbm
from evalkit.models.base import CLASSES, Predictor, clear_registry, get, register
from evalkit.policy import t2_leaderboard_calibration


def frame(n: int, y: list[int], **cols: list[float]) -> pl.DataFrame:
    """Assembly frame: FEATURE_COLUMNS + fmt + y + match_id, defaults zero."""
    base = {c: [0.0] * n for c in FEATURE_COLUMNS}
    base.update({k: [float(x) for x in v] for k, v in cols.items()})
    return pl.DataFrame(base).with_columns(
        pl.Series("fmt", ["t20"] * n),
        pl.Series("y", y, dtype=pl.Int64),
        pl.Series("match_id", [f"m{i % 3}" for i in range(n)]),
    )


def test_registry_round_trip_and_duplicate() -> None:
    clear_registry()
    model = B0MarginalT2()
    register("t2", "t20", model)
    assert get("t2", "t20", "B0_marginal") is model
    assert isinstance(model, Predictor)
    with pytest.raises(ValueError, match="duplicate"):
        register("t2", "t20", B0MarginalT2())
    clear_registry()


def test_b0_t1_laplace() -> None:
    train = frame(4, y=[0, 0, 1, 4])
    model = B0MarginalT1()
    model.fit(train, train)
    probs = model.predict_proba(train)
    k = len(CLASSES)
    assert probs.shape == (4, k)
    assert probs[0, 0] == pytest.approx((2 + 1) / (4 + k))
    assert probs[0, 5] == pytest.approx((0 + 1) / (4 + k))  # unseen class, Laplace
    assert probs.sum(axis=1) == pytest.approx(np.ones(4))


def test_b0_t2_is_match_level_not_delivery_level() -> None:
    # match m0 (2 rows, y=1), m1 (1 row, y=0), m2 (1 row, y=0):
    # delivery rate = 0.5 but match rate = 1/3
    df = frame(4, y=[1, 0, 0, 1])  # match_id cycles m0,m1,m2,m0
    model = B0MarginalT2()
    model.fit(df, df)
    assert model.base_rate == pytest.approx(1 / 3)


def test_bucket_edge_cases() -> None:
    df = frame(
        3,
        y=[0, 0, 0],
        legal_balls=[0.0, 36.0, 96.0],
        wickets=[0.0, 3.0, 7.0],
        crr=[0.0, 6.0, 13.0],
        rrr=[0.0, 0.0, 12.0],
        is_chase=[0.0, 0.0, 1.0],
    )
    b = bucket_columns(df, "t20")
    # amendment #2 edge case: legal_balls == 0 → crr 0.0 → LOWEST rate band
    assert b[0].tolist() == [0, 0, 0, 0]
    # crr exactly 6.0 → band (4,6] (index 1); wickets 3 → band 2-3 (index 1)
    assert b[1].tolist() == [0, 1, 1, 1]
    # chase row: rrr 12 → band (10,12] (index 4); wickets 7 → 6+ (index 3)
    assert b[2].tolist() == [1, 3, 3, 4]
    assert len(RATE_EDGES) == 5


def test_shrinkage_pulls_small_leaves_to_parent() -> None:
    buckets = np.array([[0, 0, 0, 0]] * 100 + [[0, 0, 0, 1]] * 2, dtype=np.int64)
    targets = np.array([[1.0]] * 100 + [[0.0]] * 2)
    strong = _Table(buckets, targets, tau=800.0)
    weak = _Table(buckets, targets, tau=50.0)
    # the 2-row leaf: strong shrinkage keeps it near the parent (~0.98),
    # weak shrinkage lets it drop further
    assert strong.estimates[(0, 0, 0, 1)][0] > weak.estimates[(0, 0, 0, 1)][0]
    # unseen leaf falls back to parent estimate
    unseen = strong.lookup(np.array([[0, 0, 0, 5]], dtype=np.int64))
    assert unseen[0, 0] == pytest.approx(strong.estimates[(0, 0, 0)][0])


def synthetic_t2(n: int = 3000, seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Chase-shaped synthetic: higher rrr → first side wins more often."""
    rng = np.random.default_rng(seed)
    rrr = rng.uniform(0, 14, n)
    wkts = rng.integers(0, 10, n)
    p = 1 / (1 + np.exp(-(0.4 * (rrr - 7) + 0.2 * (wkts - 4))))
    y = (rng.uniform(0, 1, n) < p).astype(int)
    df = frame(
        n,
        y=list(y),
        rrr=list(rrr),
        wickets=list(map(float, wkts)),
        legal_balls=list(rng.uniform(0, 120, n)),
        is_chase=[1.0] * n,
    )
    half = n // 2
    return df.head(half), df.tail(n - half)


def test_b1_t2_fit_predict_and_tau_choice() -> None:
    train, val = synthetic_t2()
    model = B1TableT2("t20")
    model.fit(train, val)
    assert model.tau in (50.0, 200.0, 800.0)
    p = model.predict_proba(val)
    assert p.shape == (val.height,)
    assert np.all((p > 0) & (p < 1))
    # signal recovered: high-rrr rows get higher p than low-rrr rows
    hi = p[val["rrr"].to_numpy() > 10].mean()
    lo = p[val["rrr"].to_numpy() < 4].mean()
    assert hi > lo + 0.1


def test_b2_and_b3_smoke_t2() -> None:
    train, val = synthetic_t2(1500)
    for model in (B2Logistic("t2"), B3Gbm("t2")):
        model.fit(train, val)
        p = model.predict_proba(val)
        assert p.shape == (val.height,)
        assert np.all((p >= 0) & (p <= 1))
    b2 = B2Logistic("t2")
    b2.fit(train, val)
    assert b2.c in (0.01, 0.1, 1.0, 10.0)


def test_ladder_inversion_canary_logic() -> None:
    ok = ladder_inversion_canary(0.60, 0.55, 0.53, where="x")
    assert ok.passed
    b2_bad = ladder_inversion_canary(0.60, 0.6011, 0.50, where="x")
    assert not b2_bad.passed
    b3_bad = ladder_inversion_canary(0.60, 0.55, 0.5551, where="x")
    assert not b3_bad.passed


def test_thin_cell_policy() -> None:
    assert t2_leaderboard_calibration(150) == "platt"
    assert t2_leaderboard_calibration(299) == "platt"
    assert t2_leaderboard_calibration(300) == "isotonic"
    assert t2_leaderboard_calibration(1477) == "isotonic"
