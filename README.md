# cricstate — Module 1: State Core

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

## Known macOS quirk

uv sets the macOS `UF_HIDDEN` flag on `.venv` contents, and CPython 3.12 skips
hidden `.pth` files — which silently breaks the editable install
(`ModuleNotFoundError: cricstate`). If that happens after a fresh `uv sync`:

```
chflags -R nohidden .venv
```
