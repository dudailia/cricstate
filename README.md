# cricstate

**A calibrated, leakage-audited cricket win-probability benchmark** — built on
16,754 professionally-parsed matches (4.7M deliveries) with a frozen,
pre-committed decision rule that can *reject* a model.

Most sports-prediction results are unfalsifiable: tuned on the data they report,
scored without uncertainty, calibrated after the fact. cricstate is the
opposite, by construction:

- **Temporal splits, baked into the data** — train ends 2024-11-02, test starts
  2025-08-30; split integrity is a red-build test, not a convention.
- **Leakage canaries in CI** — a shuffled-target model must score exactly at
  the base rate; a poisoned outcome column must be structurally unreachable by
  the feature builder; the whole feature surface is a frozen whitelist.
- **Match-level paired bootstrap** (B = 10,000) — ball-level resampling is
  forbidden because within-match dependence makes it fake precision.
- **A decision rule frozen before results existed** (SPEC_M2 §6): a challenger
  beats the bar only with the 95% CI of ΔNLL excluding zero on *both* val and
  test, ≥ 0.5% relative improvement, and no calibration regression — on a test
  split evaluated **once**. Close results are "did not beat the bar."

## Headline result — T20 win probability (test split, evaluated once)

| model | test NLL [95% CI] | skill vs B0 |
|---|---|---|
| B0 — match base rate | 0.69275 [0.69195, 0.69356] | +0.000 |
| B1 — bucketed table, monotone by construction | 0.54078 [0.52615, 0.55549] | +0.219 |
| B2 — regularized logistic | 0.51189 [0.49689, 0.52679] | +0.261 |
| **B3 — gradient-boosted trees** | **0.49036 [0.47547, 0.50519]** | **+0.292** |

n = 1,489 test matches / 343,287 deliveries. All numbers post-calibration
(isotonic, fit on validation only). **B3 reaches 0.490 test NLL — a +0.29
skill score over the base rate** — and is the bar any future model must beat
under the frozen rule.

## The headline: we measured player identity — and declined to build the tower

The obvious next step for any cricket model is player modeling: batter form,
bowler matchups, a hierarchical tower of identity effects. Before building it,
we **measured** it, under a pre-registered gate experiment with frozen verdict
bands (`docs/BRANCH_A_REPORT.md`):

- Train-only empirical-Bayes player effects (striker + bowler) on top of the
  frozen B3 state model, λ tuned on val, single test touch, shuffled-identity
  canary, match-level bootstrap.
- **Result: player identity is worth +0.31% NLL — 0.007 bits per ball.**
  Real (ΔNLL −0.00504 [−0.00561, −0.00449], CI excludes zero) but economically
  negligible, and that's *with* honest dilution: 5% of balls have an unknown
  incoming batter, 14–19% of test deliveries involve players unseen in train.
- The unshrunk version (M_flat) is **0.158 nats worse than no identity at
  all** — raw per-player tables destroy a good state model.
- Frozen verdict: **AMBIGUOUS, at the band floor.** Per the pre-committed
  rule: the cheap increment (M_shrunk) ships on the leaderboard; **Branches
  B/C — the hierarchical modeling tower — were declined on the evidence.**

Most projects build the tower because it's interesting. The measurement said
no. That refusal — cheap, pre-committed, and documented — is what this
repository is for.

### Where naive models die: the endgame

Calibration by game phase (T20 win probability, test split, predicted vs
observed win rate for the team batting first):

| bucket | n | B0 p̄ / observed | B3 p̄ / observed |
|---|---|---|---|
| chase, overs 17–20 | 19,190 | 0.508 / **0.659** | 0.674 / 0.659 |
| last 30 balls | 72,781 | 0.508 / 0.564 | 0.571 / 0.564 |

The constant-rate model predicts 0.508 when the true rate is 0.659 — a
15-point miss exactly where matches are decided. The calibrated ladder closes
it to ~1 point. Every leaderboard ships these bucket tables; the failure mode
is measured, not assumed away.

### The honest negative result

On the ODI cell, **B3 did NOT beat B2**: ΔNLL test −0.006 with 95% CI
[−0.019, **+0.008**] — the interval includes zero at n = 136 test matches, so
the frozen rule says *did not beat the bar*, full stop. A thin cell producing
wide intervals and a refused close call is the methodology working, not a
caveat to bury: the same rule that certifies the T20 result rejects this one.

## Reproducibility

Everything is deterministic and fingerprinted:

```
corpus            16,754 matches / 4,748,382 deliveries (Cricsheet snapshot 2026-07-02, pinned by SHA256)
corpus hash       c08e4eba45ff7a71a51c4490cfe159a2ca34a7e5382bbc902041d147a11a6781
seed              1337 end-to-end (bootstrap seed 90210)
test split        evaluated once — this release
```

Two consecutive `uv run evalkit run-all` invocations produce **byte-identical**
`docs/LEADERBOARD.md`. Golden schema files and pinned corpus/label hashes are
red-build tests: the data contract cannot drift silently. Parsing is a
deterministic automaton with quarantine-not-crash semantics — 100% of the
22,211 snapshot files either parse clean (99.1% of in-scope T20/ODI) or land
in a quarantine log with a closed-enum reason code.

```
uv sync                          # Python 3.12, locked deps
uv run pytest -m "not corpus"    # unit + property tests (CI set)
uv run python -m cricstate.download   # fetch + hash the Cricsheet snapshot
uv run python -m cricstate.build      # rebuild the corpus (~7 min, hash-checked)
uv run evalkit run-all           # regenerate the leaderboard (32s cached / ~18 min cold)
```

## Plugging in a challenger (Module 3+)

Implement the `Predictor` protocol (`evalkit.models.base`), fit on train, tune
on val only, and register per (task, format) cell:

```python
class MyModel:
    name = "M3_mymodel"
    version = "1.0"
    def fit(self, train: pl.DataFrame, val: pl.DataFrame) -> None: ...
    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:  # [n,K] T1 / [n] T2
        ...
```

Models see only the whitelisted within-match state features — no player
identities, no venue identities, no odds. The bar to beat (T20 win
probability): **0.490 test NLL**, under the frozen §6 rule above. The rule may
not be renegotiated after results exist.

## Data attribution & license

Ball-by-ball data comes from [Cricsheet](https://cricsheet.org), maintained by
Stephen Rushe, licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Derived
tables built from that data inherit the same terms: attribute Cricsheet and
share adaptations alike. The pinned snapshot is recorded in `data/MANIFEST`.

## Repository map

```
src/cricstate/      Module 1 — deterministic state core: parser, validator,
                    quarantine, transition function δ, replay, corpus build
src/evalkit/        Module 2 — the measuring instrument: splits, features,
                    metrics, calibration, bootstrap, baselines B0–B3, canaries
tests/golden/       11 real pathological matches (super-over tie, D/L, penalty
                    runs, retired hurt, miscounted over, …) — exact round-trips
docs/               SPEC_M1, SPEC_M2 (+ gate-documented amendments),
                    LEADERBOARD.md, STATS.md, evidence packs, reliability plots
```

## Known macOS quirk

uv sets the macOS `UF_HIDDEN` flag on `.venv` contents, and CPython 3.12 skips
hidden `.pth` files — which silently breaks the editable install
(`ModuleNotFoundError: cricstate`). If that happens after a fresh `uv sync`:

```
chflags -R nohidden .venv
```
