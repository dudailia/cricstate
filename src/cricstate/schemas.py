"""Canonical schemas — the system-wide contract (SPEC_M1 §4, verbatim)."""

from dataclasses import dataclass
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
    RETIRED_HURT = "retired hurt"  # NOT a dismissal: does not increment wickets
    OBSTRUCTING = "obstructing the field"
    TIMED_OUT = "timed out"
    HIT_TWICE = "hit the ball twice"


@dataclass(frozen=True, slots=True)
class Wicket:
    player_out: str  # player_id
    kind: WicketKind
    fielders: tuple[str, ...] = ()  # player_ids


@dataclass(frozen=True, slots=True)
class BatterState:
    player_id: str
    runs: int = 0
    balls: int = 0  # legal balls faced


@dataclass(frozen=True, slots=True)
class BowlerState:
    player_id: str
    legal_balls: int = 0
    runs_conceded: int = 0  # wides + no-balls charge bowler; byes/legbyes do not
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
    source_confidence: float = 1.0  # vision seam: a future noisy producer emits
    #                                 the SAME type with confidence < 1.0


@dataclass(frozen=True, slots=True)
class MatchState:
    schema_version: str  # "1.0.0"
    match_id: str
    fmt: Literal["t20", "odi"]
    gender: str
    venue_id: str
    start_date: date
    competition: str | None
    innings_idx: int  # 1-based; >2 ⇒ super over (excluded from tuples)
    batting_team: str
    bowling_team: str
    balls_per_over: int  # from info — never hardcode 6
    max_balls: int  # rain-revised value as given in data
    legal_balls: int
    runs: int
    wickets: int  # dismissals only; retired-hurt ≠ wicket
    striker: BatterState | None  # None in the gap after a dismissal —
    non_striker: BatterState | None  #   incoming batter unknown until observed
    bowler: BowlerState
    target: int | None  # chasing innings only; post-DLS revision as given
    dls_applied: bool
    in_powerplay: bool
    partnership_runs: int
    partnership_balls: int
    fow: tuple[tuple[int, int], ...]  # (runs_at_fall, legal_ball_index)
    last_ball_was_noball: bool  # free-hit derivation deferred; see SPEC §11
