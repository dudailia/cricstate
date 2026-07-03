import numpy as np
import polars as pl
import pytest

from evalkit.bootstrap import (
    bootstrap_metric,
    bootstrap_paired_delta,
    make_draws,
    per_match_losses,
)


def test_per_match_losses_aggregates_in_order() -> None:
    ids = pl.Series("match_id", ["a", "a", "b", "a", "c"])
    losses = np.array([1.0, 2.0, 10.0, 3.0, 5.0])
    sums, counts = per_match_losses(losses, ids)
    assert sums.tolist() == [6.0, 10.0, 5.0]
    assert counts.tolist() == [3.0, 1.0, 1.0]


def test_bootstrap_metric_point_and_coverage() -> None:
    rng = np.random.default_rng(0)
    n_matches = 400
    counts = rng.integers(50, 200, n_matches).astype(float)
    means = rng.normal(0.6, 0.05, n_matches)
    sums = counts * means
    draws = make_draws(n_matches)
    ci = bootstrap_metric(sums, counts, draws)
    truth = sums.sum() / counts.sum()
    assert ci.point == pytest.approx(truth)
    assert ci.lo < truth < ci.hi
    assert (ci.hi - ci.lo) < 0.02  # tight at n=400 matches


def test_paired_delta_detects_real_gap_and_none() -> None:
    rng = np.random.default_rng(1)
    n = 300
    counts = rng.integers(80, 120, n).astype(float)
    base = rng.normal(0.7, 0.08, n)
    sums_b = counts * base
    sums_a = counts * (base - 0.02)  # A is uniformly better by 0.02
    draws = make_draws(n)
    delta = bootstrap_paired_delta(sums_a, sums_b, counts, draws)
    assert delta.point == pytest.approx(-0.02, abs=1e-9)
    assert delta.excludes_zero() and delta.hi < 0
    same = bootstrap_paired_delta(sums_b, sums_b, counts, draws)
    assert same.point == 0.0 and not same.excludes_zero()


def test_draws_are_deterministic() -> None:
    assert np.array_equal(make_draws(50), make_draws(50))
    assert make_draws(50).shape == (10_000, 50)
    # each resample draws exactly n matches
    assert np.all(make_draws(50).sum(axis=1) == 50)


def test_paired_beats_unpaired_precision() -> None:
    """The point of pairing: shared match draws cancel between-match variance."""
    rng = np.random.default_rng(2)
    n = 200
    counts = np.full(n, 100.0)
    base = rng.normal(0.7, 0.2, n)  # huge between-match spread
    sums_b = counts * base
    sums_a = counts * (base - 0.01)
    draws = make_draws(n)
    paired = bootstrap_paired_delta(sums_a, sums_b, counts, draws)
    a = bootstrap_metric(sums_a, counts, draws)
    b = bootstrap_metric(sums_b, counts, draws)
    paired_width = paired.hi - paired.lo
    naive_width = (a.hi - a.lo) + (b.hi - b.lo)
    assert paired_width < naive_width / 10
    assert paired.excludes_zero()
