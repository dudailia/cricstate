import numpy as np
import pytest

from evalkit.pava import monotone_smooth_2d, pava_1d


def test_pava_known_answer() -> None:
    # classic: [1, 3, 2] with equal weights → [1, 2.5, 2.5]
    out = pava_1d(np.array([1.0, 3.0, 2.0]), np.ones(3))
    assert out.tolist() == [1.0, 2.5, 2.5]


def test_pava_weights_matter() -> None:
    # heavy first element dominates the pooled block
    out = pava_1d(np.array([3.0, 1.0]), np.array([9.0, 1.0]))
    assert out[0] == out[1] == pytest.approx(2.8)


def test_pava_decreasing_direction() -> None:
    out = pava_1d(np.array([1.0, 3.0, 2.0]), np.ones(3), increasing=False)
    assert np.all(np.diff(out) <= 1e-12)
    assert out[0] == pytest.approx(2.0)  # pooled (1,3) → 2, then 2


def test_pava_preserves_already_monotone() -> None:
    v = np.array([0.1, 0.2, 0.2, 0.9])
    assert pava_1d(v, np.ones(4)).tolist() == v.tolist()


def test_pava_weighted_mean_is_preserved() -> None:
    rng = np.random.default_rng(3)
    v, w = rng.uniform(0, 1, 30), rng.uniform(0.5, 5, 30)
    out = pava_1d(v, w)
    assert np.average(out, weights=w) == pytest.approx(np.average(v, weights=w))


def test_monotone_smooth_2d_enforces_both_directions() -> None:
    rng = np.random.default_rng(7)
    grid = rng.uniform(0, 1, (4, 6))
    w = rng.uniform(1, 100, (4, 6))
    out = monotone_smooth_2d(grid, w, rate_increasing=True, wickets_increasing=False)
    assert np.all(np.diff(out, axis=1) >= -1e-12)  # rate non-decreasing
    assert np.all(np.diff(out, axis=0) <= 1e-12)  # wickets non-increasing
    up = monotone_smooth_2d(grid, w, rate_increasing=True, wickets_increasing=True)
    assert np.all(np.diff(up, axis=0) >= -1e-12)


def test_monotone_smooth_2d_identity_on_conforming_grid() -> None:
    grid = np.array([[0.5, 0.6], [0.3, 0.4]])  # rate ↑, wickets ↓ already
    out = monotone_smooth_2d(grid, np.ones((2, 2)))
    assert np.allclose(out, grid)
