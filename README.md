# cricstate — Module 1: State Core

## Data attribution & license

Ball-by-ball data comes from [Cricsheet](https://cricsheet.org), maintained by
Stephen Rushe, and is licensed under the
[Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).
Derived tables built from that data (everything under `data/v1/`) inherit the
same CC BY-SA 4.0 terms: attribute Cricsheet and share adaptations alike.
The pinned snapshot is recorded in `data/MANIFEST`.

Canonical cricket game-state representation and deterministic reconstruction from
Cricsheet ball-by-ball data. Zero machine learning: a validated deterministic
automaton. See [`docs/SPEC_M1.md`](docs/SPEC_M1.md) for the authoritative spec.

## Layout

```
src/cricstate/        the package: schemas, parser, validator, δ, replay, build
tests/                pytest + hypothesis suites
tests/golden/         10 committed pathological Cricsheet matches (see spec §11)
data/raw/             pinned Cricsheet snapshot (not in git; see data/MANIFEST)
data/quarantine/      quarantine log from the corpus build (not in git)
data/v1/              matches/deliveries/players parquet (not in git)
docs/                 SPEC_M1.md, STATS.md
```

## Commands

```
uv sync                       # install (Python 3.12, locked)
uv run pytest -m "not corpus" # unit + property + golden tests (CI set)
uv run pytest                 # everything, incl. corpus tests (needs snapshot)
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run python -m cricstate.download   # fetch + hash the Cricsheet snapshot
uv run python -m cricstate.build      # full-corpus build → data/v1/ + docs/STATS.md
```

## Module 3: how a challenger plugs in (SPEC_M2 §8, §12.8)

Implement the `Predictor` protocol (`evalkit.models.base`):

```python
class MyModel:
    name = "M3_mymodel"
    version = "1.0"
    def fit(self, train: pl.DataFrame, val: pl.DataFrame) -> None: ...
    def predict_proba(self, df: pl.DataFrame) -> np.ndarray: ...  # [n,K] T1 / [n] T2
```

Frames are assembly frames (whitelisted `FEATURE_COLUMNS` + `fmt` + `y` +
`match_id`); use `to_x`/`to_y`, never read `match_id`. Register per cell with
`evalkit.models.base.register(task, fmt, model)`; artifacts cache under
`artifacts/{task}/{fmt}/{name}/`. Evaluation is `uv run evalkit run-all`
(cached) or `--cold` (full refit, < 60 min documented).

**The frozen decision rule (SPEC_M2 §6 — may not be renegotiated):** a
challenger beats the bar for a (task, fmt) cell iff, against the best
baseline on that cell: (1) ΔNLL < 0 with the 95% match-level paired-bootstrap
CI excluding 0 on **both** val and test; (2) relative NLL improvement ≥ 0.5%;
(3) post-calibration ECE not worse by more than 0.005; (4) evaluated on the
frozen test split, touched once. Close results are "did not beat the bar."
The recorded M3 bar: **t2/t20 = 0.508 nats**; t2/odi = 0.512 is directional
only (n = 143 test matches).

## Known macOS quirk

uv sets the macOS `UF_HIDDEN` flag on `.venv` contents, and CPython 3.12 skips
hidden `.pth` files — which silently breaks the editable install
(`ModuleNotFoundError: cricstate`). If that happens after a fresh `uv sync`:

```
chflags -R nohidden .venv
```
