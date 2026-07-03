"""B1 — Bucketed empirical table with hierarchical shrinkage (SPEC_M2 §5,
as amended #2: innings 1 carries a CRR band with the RRR edges).

Bucket hierarchy (leaf depth 4 in both innings):
    level 0  global
    level 1  innings (1 | 2)
    level 2  x over-phase   (t20: balls [0,36),[36,72),[72,96),[96,∞);
                             odi: [0,60),[60,180),[180,240),[240,∞))
    level 3  x wickets band (0-1, 2-3, 4-5, 6+)
    level 4  x rate band    (innings 1: CRR band; innings 2: RRR band;
                             edges (-inf,4],(4,6],(6,8],(8,10],(10,12],(12,inf))

legal_balls == 0 → CRR band is the lowest band (amendment #2: defined, not
inferred; the builder emits crr = 0.0 there, which lands in (-inf,4]).

Each level's estimate is shrunk toward its parent recursively:
    p̂_level = (n·p_level + τ·p̂_parent) / (n + τ),   τ ∈ {50, 200, 800} on val.
"""

import numpy as np
import polars as pl
from numpy.typing import NDArray

from evalkit.metrics import nll_binary, nll_multiclass
from evalkit.models.base import CLASSES, to_y

PHASE_EDGES = {"t20": (36, 72, 96), "odi": (60, 180, 240)}
WICKET_EDGES = (2, 4, 6)  # bands 0-1, 2-3, 4-5, 6+
RATE_EDGES = (4.0, 6.0, 8.0, 10.0, 12.0)  # 6 bands
TAU_GRID = (50.0, 200.0, 800.0)

BucketKey = tuple[int, ...]


def bucket_columns(df: pl.DataFrame, fmt: str) -> NDArray[np.int64]:
    """[n, 4] int matrix: innings, phase, wickets band, rate band."""
    balls = df["legal_balls"].to_numpy()
    innings = (df["is_chase"].to_numpy() >= 1.0).astype(np.int64)  # 0 = innings 1
    phase = np.searchsorted(np.array(PHASE_EDGES[fmt]), balls, side="right")
    wkts = np.searchsorted(np.array(WICKET_EDGES), df["wickets"].to_numpy(), side="right")
    rate = np.where(innings == 1, df["rrr"].to_numpy(), df["crr"].to_numpy())
    rate_band = np.searchsorted(np.array(RATE_EDGES), rate, side="left")
    return np.stack([innings, phase, wkts, rate_band], axis=1)


class _Table:
    """Shrunken estimates for every prefix level, built once per τ."""

    def __init__(self, buckets: NDArray[np.int64], targets: NDArray[np.float64], tau: float):
        # targets: [n, d] (d=1 for T2, d=K one-hot for T1)
        self.tau = tau
        self.estimates: dict[BucketKey, NDArray[np.float64]] = {}
        root = targets.mean(axis=0)
        self.estimates[()] = root
        for level in range(1, 5):
            keys = buckets[:, :level]
            uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
            sums = np.zeros((len(uniq), targets.shape[1]))
            np.add.at(sums, inv, targets)
            for i, key_arr in enumerate(uniq):
                key: BucketKey = tuple(int(v) for v in key_arr)
                parent = self.estimates[key[:-1]]
                n = counts[i]
                self.estimates[key] = (sums[i] + tau * parent) / (n + tau)

    def lookup(self, buckets: NDArray[np.int64]) -> NDArray[np.float64]:
        """Leaf estimate; unseen leaves fall back to the deepest known prefix."""
        out = np.empty((len(buckets), self.estimates[()].shape[0]))
        for i, row in enumerate(buckets):
            key: BucketKey = tuple(int(v) for v in row)
            while key not in self.estimates:
                key = key[:-1]
            out[i] = self.estimates[key]
        return out


class B1TableT2:
    name = "B1_table"
    version = "1.0"

    def __init__(self, fmt: str):
        self.fmt = fmt
        self.tau: float = TAU_GRID[0]
        self.table: _Table | None = None

    def fit(self, train: pl.DataFrame, val: pl.DataFrame) -> None:
        buckets = bucket_columns(train, self.fmt)
        targets = to_y(train).astype(np.float64).reshape(-1, 1)
        val_buckets = bucket_columns(val, self.fmt)
        val_y = to_y(val)
        best: tuple[float, float, _Table | None] = (np.inf, TAU_GRID[0], None)
        for tau in TAU_GRID:
            table = _Table(buckets, targets, tau)
            p = np.clip(table.lookup(val_buckets)[:, 0], 1e-6, 1 - 1e-6)
            score = nll_binary(p, val_y)
            if score < best[0]:
                best = (score, tau, table)
        _, self.tau, self.table = best

    def predict_proba(self, df: pl.DataFrame) -> NDArray[np.float64]:
        assert self.table is not None, "fit first"
        return np.clip(self.table.lookup(bucket_columns(df, self.fmt))[:, 0], 1e-6, 1 - 1e-6)


class B1TableT1:
    name = "B1_table"
    version = "1.0"

    def __init__(self, fmt: str):
        self.fmt = fmt
        self.tau: float = TAU_GRID[0]
        self.table: _Table | None = None

    def fit(self, train: pl.DataFrame, val: pl.DataFrame) -> None:
        buckets = bucket_columns(train, self.fmt)
        y = to_y(train)
        onehot = np.zeros((len(y), len(CLASSES)))
        onehot[np.arange(len(y)), y] = 1.0
        val_buckets = bucket_columns(val, self.fmt)
        val_y = to_y(val)
        best: tuple[float, float, _Table | None] = (np.inf, TAU_GRID[0], None)
        for tau in TAU_GRID:
            table = _Table(buckets, onehot, tau)
            probs = self._normalize(table.lookup(val_buckets))
            score = nll_multiclass(probs, val_y)
            if score < best[0]:
                best = (score, tau, table)
        _, self.tau, self.table = best

    @staticmethod
    def _normalize(p: NDArray[np.float64]) -> NDArray[np.float64]:
        p = np.clip(p, 1e-9, None)
        result: NDArray[np.float64] = p / p.sum(axis=1, keepdims=True)
        return result

    def predict_proba(self, df: pl.DataFrame) -> NDArray[np.float64]:
        assert self.table is not None, "fit first"
        return self._normalize(self.table.lookup(bucket_columns(df, self.fmt)))
