# Reproducing cricstate

Every result is deterministic and hash-pinned. This guide reproduces the corpus,
the leaderboard, the two gate experiments, and the paper figures, and states the
exact values you should see.

## 0. Environment

```bash
git clone https://github.com/dudailia/cricstate && cd cricstate
uv sync            # Python 3.12, exact versions from uv.lock
```

`uv` is the source of truth; `requirements.txt` mirrors it for pip users.
On macOS, if you hit `ModuleNotFoundError: cricstate` after `uv sync`, run
`chflags -R nohidden .venv` (uv marks `.venv` hidden and CPython 3.12 skips
hidden `.pth` files).

## 1. Fast path — verify without rebuilding the corpus

The frozen evidence set and the presentation layer regenerate in seconds and
need **no data download**:

```bash
uv run pytest -m "not corpus"                 # 190 unit/property tests
uv run python scripts/generate_figures.py     # report/figures + report/tables
```

The figures and tables are rebuilt from [`results/summary.json`](../results/summary.json)
alone — no models are run.

## 2. Full path — rebuild everything from the data snapshot

```bash
uv run python -m cricstate.download   # fetch + SHA-256-pin the Cricsheet snapshot
uv run python -m cricstate.build      # ~7 min: corpus v1.2 -> data/v1/*.parquet
uv run evalkit run-all --cold         # ~18 min cold: refit B0–B3, single test eval
```

Expected, byte-stable across runs (seed 1337 end-to-end):

| quantity | value |
|---|---|
| corpus hash | `c08e4eba45ff7a71a51c4490cfe159a2ca34a7e5382bbc902041d147a11a6781` |
| labels hash | `e11bf9c83276d10871b401263c351d7e4d93da85183b3c1d141fefbdad20324f` |
| deliveries | 4,748,382 rows · matches 16,754 |
| B3 t1/t20 test NLL | 1.61438 [1.60858, 1.61995] |
| B3 t2/t20 test NLL | 0.49036 [0.47547, 0.50519] |

Two consecutive `uv run evalkit run-all` invocations produce **byte-identical**
`docs/LEADERBOARD.md`. The pinned hashes are asserted by
`uv run pytest -m corpus` (which needs the built corpus present).

## 3. The gate experiments

```bash
# Branch A — player identity (single test touch; verdict AMBIGUOUS +0.31%)
uv run python experiments/branch_a/run.py     # -> docs/BRANCH_A_REPORT.md

# Branch C — conditions latent (partial: validation only, frozen at C1)
# The C1 validation curve lives in artifacts/branch_c/c1_kappa_curve.json;
# the arm was frozen before its test evaluation by design.
```

## 4. What "reproducible" means here

- **Fixed seeds** (1337 end-to-end; bootstrap seed 90210).
- **Match-level paired bootstrap**, 10,000 resamples — never ball-level.
- **Golden schema + hash-pin tests** fail the build if any table or hash drifts.
- **Test split evaluated once** per release; development never touches it.
- Presentation (`report/`) is a pure function of `results/summary.json`.
