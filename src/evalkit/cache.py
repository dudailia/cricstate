"""Fitted-artifact cache (SPEC_M2 amendment #4).

artifacts/{task}/{fmt}/{model}/ holds a pickled fitted Predictor plus a
fingerprint (corpus hash, labels hash, model version, seed, feature list).
run-all loads from cache when the fingerprint matches; anything else refits.
Fits are deterministic, so a valid cache reproduces the exact predictions.
"""

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

from evalkit.features import FEATURE_COLUMNS
from evalkit.freeze import PINNED_CORPUS_HASH, PINNED_LABELS_HASH
from evalkit.models.base import SEED, Predictor

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"


@dataclass(frozen=True)
class Fingerprint:
    corpus_hash: str
    labels_hash: str
    model_name: str
    model_version: str
    seed: int
    n_features: int

    @staticmethod
    def current(model: Predictor) -> "Fingerprint":
        return Fingerprint(
            corpus_hash=PINNED_CORPUS_HASH,
            labels_hash=PINNED_LABELS_HASH,
            model_name=model.name,
            model_version=model.version,
            seed=SEED,
            n_features=len(FEATURE_COLUMNS),
        )


def _dir(task: str, fmt: str, name: str) -> Path:
    return ARTIFACTS / task / fmt / name


def store(task: str, fmt: str, model: Predictor) -> None:
    d = _dir(task, fmt, model.name)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "model.pkl", "wb") as fh:
        pickle.dump(model, fh)
    with open(d / "fingerprint.json", "w") as fh:
        json.dump(asdict(Fingerprint.current(model)), fh, indent=1)


def load(task: str, fmt: str, name: str) -> Predictor | None:
    """The cached model, or None if absent/stale (fingerprint mismatch)."""
    d = _dir(task, fmt, name)
    fp_path, pkl_path = d / "fingerprint.json", d / "model.pkl"
    if not (fp_path.exists() and pkl_path.exists()):
        return None
    with open(fp_path) as fh:
        stored = json.load(fh)
    with open(pkl_path, "rb") as fh:
        model: Predictor = pickle.load(fh)
    if stored != asdict(Fingerprint.current(model)):
        return None
    return model
