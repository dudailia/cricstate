import numpy as np

from evalkit import cache
from evalkit.models.b0_marginal import B0MarginalT2
from tests.test_models import frame


def test_cache_round_trip_and_staleness(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(cache, "ARTIFACTS", tmp_path)
    df = frame(4, y=[1, 0, 0, 1])
    model = B0MarginalT2()
    model.fit(df, df)
    cache.store("t2", "t20", model)

    loaded = cache.load("t2", "t20", "B0_marginal")
    assert loaded is not None
    assert np.array_equal(loaded.predict_proba(df), model.predict_proba(df))

    # absent → None
    assert cache.load("t2", "odi", "B0_marginal") is None

    # stale fingerprint (e.g., corpus regenerated under a new hash) → None
    fp = tmp_path / "t2" / "t20" / "B0_marginal" / "fingerprint.json"
    fp.write_text(fp.read_text().replace("c08e4eba", "deadbeef"))
    assert cache.load("t2", "t20", "B0_marginal") is None
