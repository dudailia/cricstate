<!-- cricstate is a frozen research artifact. See CONTRIBUTING.md. -->

## What this changes

<!-- One or two sentences. -->

## Category

- [ ] Presentation only (docs, figures from `summary.json`, site) — **no numbers changed**
- [ ] New Module-3 challenger (evaluated under the frozen §6 rule)
- [ ] Tooling / CI / DX
- [ ] Other (explain)

## Freeze checklist

- [ ] Does **not** modify `src/cricstate`, `src/evalkit`, `experiments/`, or their tests
- [ ] Does **not** alter the corpus hash `c08e4eba…`, labels hash `e11bf9c8…`, or any reported number
- [ ] Figures/tables (if touched) were regenerated with `uv run python scripts/generate_figures.py`, not hand-edited

## Checks

```
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest -m "not corpus"
```

- [ ] All green locally
