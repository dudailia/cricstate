# SPEC_M1 — Module 1: State Core

**Version:** 1.0 · **Date:** 2026-07-02 · **Status:** Approved for implementation
**Scope:** Canonical cricket game-state representation + deterministic reconstruction from Cricsheet ball-by-ball data.

This module contains **zero machine learning**. It is a validated deterministic automaton. Its entire value is total correctness: every downstream number inherits its errors.

---

## 1. Responsibilities

The module owns exactly five things:

1. The canonical `MatchState` and `Delivery` schemas — the system-wide contract.
2. Ingestion and strict validation of raw Cricsheet JSON, with quarantine-not-crash semantics.
3. Entity resolution of player names to stable IDs.
4. The deterministic transition function δ that folds deliveries into state sequences.
5. Emission of versioned, columnar training tables and a streaming-compatible replay API.

**Non-responsibilities:** no prediction, no probabilities, no fitted parameters, no vision, no odds, no infrastructure beyond a single process.

## 2. Inputs

- Cricsheet JSON, format version **1.1.0** (pin it; quarantine other versions). Full archive ≈ 22,000 matches, ~141 MB zip.
- Each file: `meta` (data_version, created, revision), `info` (balls_per_over, venue, dates, event, gender, teams, outcome, toss, players, `registry.people` name→ID map), and nested `innings → overs → deliveries`.
- Snapshot is downloaded once, content-hashed (SHA256), and recorded with date in `data/MANIFEST`. Cricsheet revises past files; never re-download silently.

**v1 corpus scope:** men's and women's **T20** (internationals + franchise leagues) and **ODI**.
**Excluded from v1:** Tests (draws/declarations/follow-ons expand the terminal-outcome space), The Hundred (5-ball sets — schema supports via `balls_per_over`; outcome model shouldn't yet), super overs (parsed, flagged, excluded from training tuples).

## 3. Outputs

Three Parquet tables — `matches`, `deliveries`, `players` — plus a quarantine log, and two APIs:

```python
def apply(state: MatchState, event: Delivery) -> MatchState      # total function; raises QuarantineError
def replay(match_id: str) -> Iterator[tuple[MatchState, Delivery, MatchState]]
```

The `deliveries` table is the money artifact: **one row per ball, full pre-ball state as flat columns plus the outcome.** That table is Module 2's training set — `(state_before → outcome)` — so Module 2 starts as a `read_parquet` call.

## 4. Data Structures

```python
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Literal


class WicketKind(StrEnum):
    BOWLED = "bowled"
    CAUGHT = "caught"
    LBW = "lbw"
    RUN_OUT = "run out"
    STUMPED = "stumped"
    CAUGHT_BOWLED = "caught and bowled"
    HIT_WICKET = "hit wicket"
    RETIRED_OUT = "retired out"
    RETIRED_HURT = "retired hurt"          # NOT a dismissal: does not increment wickets
    OBSTRUCTING = "obstructing the field"
    TIMED_OUT = "timed out"
    HIT_TWICE = "hit the ball twice"


@dataclass(frozen=True, slots=True)
class Wicket:
    player_out: str                        # player_id
    kind: WicketKind
    fielders: tuple[str, ...] = ()         # player_ids


@dataclass(frozen=True, slots=True)
class BatterState:
    player_id: str
    runs: int = 0
    balls: int = 0                         # legal balls faced


@dataclass(frozen=True, slots=True)
class BowlerState:
    player_id: str
    legal_balls: int = 0
    runs_conceded: int = 0                 # wides + no-balls charge bowler; byes/legbyes do not
    wickets: int = 0


@dataclass(frozen=True, slots=True)
class Delivery:
    batter_id: str
    bowler_id: str
    non_striker_id: str
    runs_batter: int
    runs_extras: int
    runs_total: int
    wides: int = 0
    noballs: int = 0
    byes: int = 0
    legbyes: int = 0
    penalty: int = 0
    wickets: tuple[Wicket, ...] = ()
    source_confidence: float = 1.0         # vision seam: a future noisy producer emits
                                           # the SAME type with confidence < 1.0


@dataclass(frozen=True, slots=True)
class MatchState:
    schema_version: str                    # "1.0.0"
    match_id: str
    fmt: Literal["t20", "odi"]
    gender: str
    venue_id: str
    start_date: date
    competition: str | None
    innings_idx: int                       # 1-based; >2 ⇒ super over (excluded from tuples)
    batting_team: str
    bowling_team: str
    balls_per_over: int                    # from info — never hardcode 6
    max_balls: int                         # rain-revised value as given in data
    legal_balls: int
    runs: int
    wickets: int                           # dismissals only; retired-hurt ≠ wicket
    striker: BatterState | None            # None in the gap after a dismissal —
    non_striker: BatterState | None        #   incoming batter unknown until observed
    bowler: BowlerState
    target: int | None                     # chasing innings only; post-DLS revision as given
    dls_applied: bool
    in_powerplay: bool
    partnership_runs: int
    partnership_balls: int
    fow: tuple[tuple[int, int], ...]       # (runs_at_fall, legal_ball_index)
    last_ball_was_noball: bool             # free-hit derivation deferred; see §11
```

Immutability (`frozen=True`) is deliberate: `apply` returns a new state — trivially testable, hashable, safe to parallelize per match. The `striker: None` gap is a modeling decision, not an oversight: in live inference the incoming batter is genuinely unknown until the next ball; the schema represents that honestly rather than peeking ahead.

## 5. Algorithms

**Validation:** pin `meta.data_version == "1.1.0"`; JSON-Schema-validate strictly; any failure quarantines the whole match with a reason code from a closed enum: `E_SCHEMA`, `E_VERSION`, `E_REGISTRY_MISS`, `E_BALL_ACCOUNTING`, `E_UNKNOWN_WICKET_KIND`, `E_DEAD_STATE`, `E_FORMAT_OOS` (out-of-scope format), `E_OTHER`. Target ≥99.5% parse-or-quarantine; **zero uncaught exceptions, zero silent excepts.**

**Entity resolution:** per-file `registry.people` name→ID map, joined to Cricsheet's global people register. **No fuzzy matching in v1** — a name absent from the registry is a quarantine, not a guess.

**Transition function δ (core logic):**

```
apply(s, d):
  require innings_active(s) else E_DEAD_STATE
  legal  = (d.wides == 0 and d.noballs == 0)
  runs'  = s.runs + d.runs_total
  balls' = s.legal_balls + int(legal)
  striker.runs  += d.runs_batter
  striker.balls += int(legal)
  bowler ledger: wides + noballs charge the bowler; byes/legbyes do not
  for w in d.wickets:
      if w.kind != RETIRED_HURT: wickets' += 1; fow += ((runs', balls'),)
      vacate(w.player_out)                     # that batting slot → None
  crossings = d.runs_batter + d.byes + d.legbyes
              + max(d.wides - 1, 0)            # runs physically run on a wide
  if crossings is odd: swap striker / non_striker
  if legal and balls' % s.balls_per_over == 0: swap ends
  terminal if wickets' == 10 or balls' == max_balls or (target and runs' >= target)
```

Parity of *runs physically run* handles strike rotation for every case including boundaries (4 and 6 are even; an all-run four is even; a three or an overthrow five swaps). No boundary flag needed.

**Invariants (assert after every match replay):**

- Final runs equal the sum of delivery `runs_total`.
- `wickets ≤ 10`.
- Per-over legal-ball counts reconcile against the data — including `miscounted_overs` (umpire allowed a short/long over), which downgrades that check to a warning keyed to the metadata rather than a failure.
- Determinism: replaying any match twice yields byte-identical state streams; each match's stream is content-hashed for regression testing.

## 6. Mathematical Formulation

The module is a deterministic automaton (𝒮, ℰ, δ, s₀, ℱ): state space 𝒮 per the schema, event alphabet ℰ = the Delivery type, total transition δ: 𝒮 × ℰ → 𝒮 ∪ {⊥} with ⊥ = quarantine. A match is the fold s_T = δ(…δ(δ(s₀, e₁), e₂)…, e_T).

Derived features are **pure functions of state, computed at read time, never stored as state**:

- Required run rate: `RRR = 6 · (target − runs) / (max_balls − legal_balls)`
- Current run rate, balls remaining, wickets in hand, partnership figures.
- Phase indicators from `innings.powerplays` markers (e.g. from 0.1 to 5.6, type "mandatory").

## 7. Model Choices

None. Deterministic parsing beats any learned parser on the axes that matter here: auditability, reproducibility, zero silent failure. The one tempting place — name matching — is deliberately excluded (§5).

## 8. APIs Between Modules

The Parquet column contract **is** the inter-module API. It is semver'd: `schema_version` in every row; breaking change ⇒ major bump ⇒ downstream pins.

Forward-looking seam:

```python
class StateProducer(Protocol):
    def events(self, source_id: str) -> Iterator[Delivery]: ...
```

`CricsheetProducer` implements it at `source_confidence = 1.0`. A future vision producer implements the identical protocol with confidence < 1.0 and alternate hypotheses — nothing downstream changes. This protocol is the entire "video → state" integration surface: designed now, built only if ever justified.

## 9. Computational Complexity & Storage

- Time O(total deliveries); embarrassingly parallel per match. v1 corpus ≈ 4–6M delivery rows: **minutes on a laptop with Polars.**
- Storage: pinned raw snapshot (~141 MB zip, hashed, dated) + ~1–2 GB Parquet.
- Layout: `data/raw/snapshot_YYYY-MM-DD/`, `data/quarantine/`, `data/v1/{matches,deliveries,players}.parquet`.
- **No distributed anything.** No Kafka, no Docker, no k8s. Single process, single machine.

## 10. Training & Inference Interface (defined here, consumed by Module 2)

- Outcome alphabet collapsed to ~12 classes: {0, 1, 2, 3, 4, 6, wicket-grouped-by-type, wide, no-ball, bye/leg-bye}. Sub-typing preserved in columns.
- **Temporal split column** derived from `start_date`, baked into the table.
- Leakage rules enforced at emission: any venue/player aggregate joined later must be rolling-as-of-date; nothing downstream may condition on `outcome.*` fields.
- Live path: same δ fed by a `LiveFeedAdapter` implementing `StateProducer`; per-event update cost is microseconds.

## 11. Failure Modes (each gets a golden-file test)

1. Tie resolved by super over (`outcome.result = "tie"` + `eliminator`).
2. Rain-affected result under Duckworth-Lewis (`outcome.method = "D/L"`) — store revised targets as given; compute **no** par scores in v1 (official DLS is proprietary).
3. Uncontested toss.
4. Miscounted over (5-ball over allowed by umpire).
5. Penalty runs.
6. Retired hurt (and possible return).
7. Concussion replacements; 2005-era supersubs field.
8. Abandoned / no-result matches (parse, flag, exclude from tuples).
9. Stumping off a wide.
10. Wide with runs (crossing parity: `wides − 1`).

Known bias, documented not solved: **free hits are not encoded in Cricsheet** and the rule's scope changed over the years; v1 carries `last_ball_was_noball` and a written bias note instead of a guessed rule-epoch table.

Known data cap: **field placements are not in this data at all.** No module may claim to condition on field settings until a source exists.

## 12. Future Improvements (priority order)

1. Belief-state extension of `Delivery` (distribution over outcomes, for noisy producers).
2. Test-match support.
3. The Hundred via `balls_per_over` generalization.
4. External enrichment joins (bowler type, handedness) — ToS review before any scraping.
5. Vision producer against the §8 protocol — only after the vision-tax number justifies it.

## 13. Definition of Done (all required)

- ≥99.5% of the v1 corpus parses or quarantines with a coded reason.
- Invariant suite passes corpus-wide.
- All 10 golden pathological matches round-trip exactly.
- Determinism hashes stable across two full runs.
- Full build < 10 minutes on a laptop.

When green, Module 2 (per-ball outcome model) begins as a dataframe read — and not before.
