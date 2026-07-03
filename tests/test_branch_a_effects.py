"""Branch A phase A0: shrunk player effects (train-only, unseen -> mean)."""

import numpy as np
import polars as pl
import pytest
from experiments.branch_a.effects import PlayerEffects, fit_effects

K = 4  # small alphabet keeps synthetic tests readable; K=11 in the real run


def synthetic_train(
    n_heavy: int = 20_000, n_light: int = 8, seed: int = 7
) -> tuple[pl.Series, np.ndarray, np.ndarray, np.ndarray]:
    """A large neutral population (null ids, distribution p_bar) dominates the
    fold so the train marginal ≈ p_bar; 'heavy' has a known offset and lots of
    data; 'light' has the same true offset but almost no data."""
    rng = np.random.default_rng(seed)
    n_pop = 200_000
    p_bar = np.array([0.5, 0.25, 0.15, 0.10])
    true_offset = np.array([-0.4, 0.3, 0.2, 0.4])
    p_player = p_bar * np.exp(true_offset)
    p_player /= p_player.sum()
    ids = [None] * n_pop + ["heavy"] * n_heavy + ["light"] * n_light
    y = np.concatenate(
        [
            rng.choice(K, size=n_pop, p=p_bar),
            rng.choice(K, size=n_heavy, p=p_player),
            rng.choice(K, size=n_light, p=p_player),
        ]
    ).astype(np.int64)
    # raw offsets are gauge-dependent (softmax shift-invariant); tests compare
    # implied normalized distributions instead
    return pl.Series("pid", ids), y, p_bar, p_player


def implied_distribution(effects: PlayerEffects, pid: str) -> np.ndarray:
    logits = effects.population_logp + effects.offsets[pid]
    p = np.exp(logits)
    result: np.ndarray = p / p.sum()
    return result


def test_recovers_known_offsets_with_data() -> None:
    ids, y, _, p_player = synthetic_train()
    effects = fit_effects(ids, y, lam=50.0, n_classes=K)
    est = implied_distribution(effects, "heavy")
    assert np.max(np.abs(est - p_player)) < 0.01  # 20k balls → tight recovery


def test_sparse_player_is_shrunk_toward_population() -> None:
    ids, y, _, _ = synthetic_train()
    effects = fit_effects(ids, y, lam=200.0, n_classes=K)
    pop = np.exp(effects.population_logp)  # the model's own population marginal
    light = implied_distribution(effects, "light")
    heavy = implied_distribution(effects, "heavy")
    # light (8 balls, λ=200) hugs the population; heavy (20k balls) escapes it
    assert np.abs(light - pop).max() < 0.02
    assert np.abs(heavy - pop).max() > 0.05
    assert np.abs(light - pop).max() < np.abs(heavy - pop).max()


def test_unseen_and_null_ids_get_population_mean() -> None:
    ids, y, _, _ = synthetic_train()
    effects = fit_effects(ids, y, lam=100.0, n_classes=K)
    q = pl.Series("pid", ["never_seen", None, "heavy"])
    m = effects.offset_matrix(q)
    assert np.all(m[0] == 0.0)  # unseen → zero offset = population mean
    assert np.all(m[1] == 0.0)  # null (observation gap) → population mean
    assert np.any(m[2] != 0.0)
    assert effects.seen(q).tolist() == [False, False, True]


def test_lambda_extremes() -> None:
    ids, y, _, _ = synthetic_train()
    hard = fit_effects(ids, y, lam=1e9, n_classes=K)
    assert np.abs(hard.offsets["heavy"]).max() < 1e-4  # λ→∞ ⇒ offsets → 0
    soft = fit_effects(ids, y, lam=1e-6, n_classes=K)
    emp = np.bincount(y[np.array([p == "heavy" for p in ids])], minlength=K)
    emp_dist = emp / emp.sum()
    assert np.allclose(implied_distribution(soft, "heavy"), emp_dist, atol=1e-4)
    with pytest.raises(ValueError, match="positive"):
        fit_effects(ids, y, lam=0.0, n_classes=K)


def test_train_only_estimation_is_frozen() -> None:
    """The API sees one fold; applying to other data cannot change the fit."""
    ids, y, _, _ = synthetic_train()
    effects = fit_effects(ids, y, lam=100.0, n_classes=K)
    before = {k: v.copy() for k, v in effects.offsets.items()}
    # simulate a val/test pass: query a mix of seen/unseen ids repeatedly
    val_ids = pl.Series("pid", ["heavy", "light", "new1", None, "new2"] * 100)
    _ = effects.offset_matrix(val_ids)
    _ = effects.seen(val_ids)
    assert set(effects.offsets) == set(before)  # no new players appeared
    for k, v in before.items():
        assert np.array_equal(effects.offsets[k], v)  # nothing mutated
    with pytest.raises(AttributeError):
        effects.lam = 1.0  # type: ignore[misc]  # frozen dataclass


def test_deterministic_across_runs_and_row_order() -> None:
    ids, y, _, _ = synthetic_train()
    a = fit_effects(ids, y, lam=100.0, n_classes=K)
    perm = np.random.default_rng(0).permutation(len(y))
    b = fit_effects(pl.Series("pid", [ids[int(i)] for i in perm]), y[perm], lam=100.0, n_classes=K)
    assert set(a.offsets) == set(b.offsets)
    for pid in a.offsets:
        assert np.allclose(a.offsets[pid], b.offsets[pid])
