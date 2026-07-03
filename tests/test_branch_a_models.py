"""Branch A phase A1: logit augmentation, M_flat-as-unshrunk, lambda tuning."""

import numpy as np
import polars as pl
import pytest
from experiments.branch_a.effects import fit_effects
from experiments.branch_a.models import (
    LAMBDA_FLAT,
    IdentityEffects,
    K,
    LambdaCurve,
    fit_identity_effects,
    tune_lambda,
)


def mk_frame(strikers: list[str | None], bowlers: list[str | None], y: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "striker_id": pl.Series(strikers, dtype=pl.Utf8),
            "bowler_id": pl.Series(bowlers, dtype=pl.Utf8),
            "y": pl.Series(y, dtype=pl.Int64),
        }
    )


def test_augment_is_logit_addition() -> None:
    rng = np.random.default_rng(0)
    train = mk_frame(["s"] * 600, ["b"] * 600, list(rng.integers(0, K, 600)))
    effects = fit_identity_effects(train, lam=50.0)
    base = rng.dirichlet(np.ones(K), size=3)
    df = mk_frame(["s", None, "unseen"], ["b", "b", None], [0, 0, 0])
    out = effects.augment(base, df)
    # manual recomputation for row 0
    logits = np.log(base[0]) + effects.striker.offsets["s"] + effects.bowler.offsets["b"]
    expect = np.exp(logits) / np.exp(logits).sum()
    assert np.allclose(out[0], expect)
    # null striker + seen bowler: only the bowler offset applies
    logits1 = np.log(base[1]) + effects.bowler.offsets["b"]
    assert np.allclose(out[1], np.exp(logits1) / np.exp(logits1).sum())
    # unseen striker + null bowler: unchanged probabilities
    assert np.allclose(out[2], base[2])
    assert np.allclose(out.sum(axis=1), 1.0)


def test_flat_lambda_reproduces_empirical_table() -> None:
    """M_flat (lambda->0) is the per-player empirical distribution — the
    unshrunk MLE of the same augmentation family."""
    y = [0] * 30 + [1] * 10
    train = mk_frame(["s"] * 40, ["b"] * 40, y)
    eff = fit_effects(train["striker_id"], train["y"].to_numpy(), lam=LAMBDA_FLAT, n_classes=K)
    implied = np.exp(eff.population_logp + eff.offsets["s"])
    implied /= implied.sum()
    assert implied[0] == pytest.approx(0.75, abs=1e-3)
    assert implied[1] == pytest.approx(0.25, abs=1e-3)


def synthetic_split(
    n: int = 12_000, seed: int = 3
) -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray]:
    """Two player types with opposite real effects + many sparse players.
    Base model = population marginal, so identity is the only signal."""
    rng = np.random.default_rng(seed)
    p_bar = np.full(K, 1.0 / K)
    hot = p_bar * np.exp(np.linspace(-0.5, 0.5, K))
    hot /= hot.sum()
    cold = p_bar * np.exp(np.linspace(0.5, -0.5, K))
    cold /= cold.sum()
    strikers: list[str | None] = []
    ys: list[int] = []
    for _ in range(n):
        r = rng.uniform()
        if r < 0.4:
            strikers.append("hot")
            ys.append(int(rng.choice(K, p=hot)))
        elif r < 0.8:
            strikers.append("cold")
            ys.append(int(rng.choice(K, p=cold)))
        else:
            strikers.append(f"sparse{rng.integers(0, 2000)}")
            ys.append(int(rng.choice(K, p=p_bar)))
    df = mk_frame(strikers, [None] * n, ys)
    half = n // 2
    base_val = np.tile(p_bar, (n - half, 1))
    return df.head(half), df.tail(n - half), base_val


def test_tune_lambda_prefers_shrinkage_and_reports_curve() -> None:
    train, val, base_val = synthetic_split()
    grid = (1.0, 10.0, 100.0, 1000.0, 10000.0)
    effects, curve = tune_lambda(train, val, base_val, grid=grid)
    assert curve.best_lam in grid
    assert len(curve.val_nll) == len(grid)
    # sparse players + real signal → interior optimum beats both extremes
    assert min(curve.val_nll) < curve.val_nll[0]  # better than ~unshrunk
    assert min(curve.val_nll) < curve.val_nll[-1]  # better than ~fully shrunk
    assert isinstance(effects, IdentityEffects)


def test_unimodal_detector() -> None:
    assert LambdaCurve((1, 2, 3), (0.5, 0.4, 0.6), 2).is_unimodal()
    assert LambdaCurve((1, 2, 3), (0.6, 0.5, 0.4), 3).is_unimodal()  # monotone edge
    assert not LambdaCurve((1, 2, 3, 4), (0.5, 0.6, 0.4, 0.7), 3).is_unimodal()


def test_tuning_never_sees_test_shaped_data() -> None:
    """tune_lambda's signature admits train and val only — the test split
    cannot reach it without an explicit new call site (A3's single touch)."""
    import inspect

    from experiments.branch_a import models

    params = list(inspect.signature(tune_lambda).parameters)
    assert params == ["train", "val", "base_val_probs", "grid"]
    assert "test" not in inspect.getsource(models).replace("test split", "")
