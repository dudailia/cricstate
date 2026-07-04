# Contributing

cricstate is a **frozen research artifact**. The corpus (`v1.2`), the evaluation
harness, the leaderboard, and the reported conclusions are pinned by hash and
must not change — that immutability is what makes the numbers trustworthy. Most
contributions therefore fall into two buckets: **fixes to the presentation
layer**, and **new challenger models evaluated under the frozen rules**.

## Ground rules

- **Never modify frozen experimental code or numbers.** `src/cricstate`,
  `src/evalkit`, `experiments/`, their tests, and the committed
  `docs/LEADERBOARD.md` / `results/summary.json` are frozen. A change that alters
  the corpus hash (`c08e4eba…`), the labels hash (`e11bf9c8…`), or any reported
  value will fail CI by design (`tests/test_freeze.py`).
- **The test split is evaluated once.** Tune on validation only; the frozen §6
  decision rule (see below) may not be renegotiated after results exist.
- **Reuse the harness, don't reimplement it.** New models import
  `evalkit.{splits,metrics,bootstrap,calibrate}` and the frozen B3.

## Local checks (must pass before a PR)

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest -m "not corpus"
```

## Adding a Module-3 challenger

Implement the `Predictor` protocol (`evalkit.models.base`), fit on train, tune
on validation, and register per `(task, fmt)` cell:

```python
class MyModel:
    name = "M3_mymodel"
    version = "1.0"
    def fit(self, train: pl.DataFrame, val: pl.DataFrame) -> None: ...
    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:  # [n,K] T1 / [n] T2
        ...
```

Models see only whitelisted within-match state features — no player or venue
identity, no odds. **The frozen decision rule (SPEC_M2 §6):** a challenger beats
the bar for a `(task, fmt)` cell iff, against the best baseline: (1) ΔNLL < 0
with the 95% match-level paired-bootstrap CI excluding 0 on **both** val and
test; (2) relative NLL improvement ≥ 0.5%; (3) post-calibration ECE not worse by
more than 0.005; (4) evaluated on the frozen test split, touched once. Close
results are "did not beat the bar."

## Presentation-layer contributions

Docs, figures, and site changes are welcome. Figures/tables regenerate from the
frozen evidence set with `uv run python scripts/generate_figures.py` — never
hand-edit the numbers.

## Style

Ruff (lint + format, line length 100) and `mypy --strict` are enforced in CI.
Match the surrounding code; prefer small, reviewable diffs.
