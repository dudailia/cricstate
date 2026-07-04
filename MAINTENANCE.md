# Maintenance policy

cricstate is a **finished research artifact**. The science is frozen; the
repository is maintained as a public record and a benchmark.

## Frozen forever

These do not change, in any PR, for any reason:

- Reported numbers, verdicts, and conclusions (`results/summary.json`,
  `docs/LEADERBOARD.md`, `report/paper.md`, branch reports).
- The datasets and their hashes (`data/MANIFEST`, corpus v1.2, labels).
- The evaluation harness semantics (`src/evalkit/`), the frozen feature
  whitelist, the splits, the calibration protocol.
- The decision rule (SPEC_M2 §6). It was fixed before results existed and is
  not renegotiable after them — that ordering is the point of the project.
- The test split's touch count. It was evaluated once; there is no second
  touch.

## Welcome

- Typo and link fixes in prose; clarity improvements that don't alter claims.
- Presentation: figure styling (regenerated from the frozen
  `results/summary.json` only), site/docs polish, packaging.
- Tooling: CI maintenance, dependency *security* patches that keep the test
  suite green (functional upgrades are pointless here — versions are pinned
  so the artifact stays reproducible, which is why `uv.lock` is not routinely
  bumped).
- **Challenger models** under the Module 3 protocol: implement the
  `Predictor` protocol (`evalkit.models.base`), fit on train, tune on val
  only, and face the frozen rule. This is the only scientific extension path.

## Rejected on sight

- Anything that reruns or re-tunes the frozen experiments, edits reported
  numbers, or "improves" a result.
- New features for the frozen models, new feature columns for the whitelist,
  post-hoc calibration changes.
- Renegotiation of verdicts (e.g. arguing AMBIGUOUS up to a win).

## Sanity checks for any PR

```
uv run pytest -m "not corpus"    # 190 unit/property tests must pass
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

If a change touches `src/visualization/` or the site, regenerate figures
(`uv run python scripts/generate_figures.py`) and confirm `report/tables/`
is byte-identical — that diff being empty is the proof the numbers didn't
move.
