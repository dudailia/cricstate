from pathlib import Path

import numpy as np

from evalkit.metrics import ece, reliability_data
from evalkit.reports import plot_reliability


def test_reliability_png_is_written(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(np.int64)
    data = reliability_data(p, y)
    out = tmp_path / "plots" / "reliability.png"
    plot_reliability(data, out, title="synthetic", ece_value=ece(p, y))
    assert out.exists() and out.stat().st_size > 5_000  # a real PNG, not a stub
