"""Deterministic match replay: fold δ over every innings (SPEC §3, §5).

replay() yields (state_before, delivery, state_after) triples in match order.
On top of δ it owns the innings-level context δ cannot see:

- initial state per innings (target, max_balls incl. long miscounted overs,
  pre-innings penalty runs),
- `in_powerplay` on the pre-ball state, from the innings' powerplay markers,
- restoring returning bowlers'/batters' cumulative ledgers (MatchState carries
  only the current pair + bowler; the cache makes the emitted figures true),
- per-over legal-ball reconciliation: a short/long over that the metadata
  declares miscounted is a warning (see match_warnings); an undeclared one
  quarantines (E_BALL_ACCOUNTING).
"""

import hashlib
import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

from cricstate import SCHEMA_VERSION
from cricstate.parser import InningsData, ParsedMatch, parse_match
from cricstate.quarantine import QuarantineError, ReasonCode
from cricstate.schemas import BatterState, BowlerState, Delivery, MatchState
from cricstate.transition import apply

Triple = tuple[MatchState, Delivery, MatchState]


def _initial_state(
    pm: ParsedMatch, inn: InningsData, innings_idx: int, first_bowler_id: str
) -> MatchState:
    if inn.target_balls is not None:
        max_balls = inn.target_balls
    else:
        max_balls = pm.scheduled_overs * pm.balls_per_over
    # An umpire-allowed long over adds real legal balls to the allotment.
    max_balls += sum(max(0, allowed - pm.balls_per_over) for allowed in inn.miscounted.values())
    return MatchState(
        schema_version=SCHEMA_VERSION,
        match_id=pm.match_id,
        fmt=pm.fmt,
        gender=pm.gender,
        venue_id=pm.venue_id,
        start_date=pm.start_date,
        competition=pm.competition,
        innings_idx=innings_idx,
        batting_team=inn.batting_team,
        bowling_team=inn.bowling_team,
        balls_per_over=pm.balls_per_over,
        max_balls=max_balls,
        legal_balls=0,
        runs=inn.penalty_pre,
        wickets=0,
        striker=None,
        non_striker=None,
        bowler=BowlerState(player_id=first_bowler_id),
        target=inn.target_runs,
        dls_applied=pm.dls_applied,
        in_powerplay=False,
        partnership_runs=0,
        partnership_balls=0,
        fow=(),
        last_ball_was_noball=False,
    )


def _in_powerplay(inn: InningsData, over_no: int, ball_no: int) -> bool:
    ref = (over_no, ball_no)
    return any(start <= ref <= end for start, end in inn.powerplays)


def replay_parsed(pm: ParsedMatch) -> Iterator[Triple]:
    """Yield (s, d, s') for every delivery of every innings, in match order."""
    for i, inn in enumerate(pm.innings):
        first_delivery = next((ov.deliveries[0] for ov in inn.overs if ov.deliveries), None)
        if first_delivery is None:
            continue
        state = _initial_state(pm, inn, i + 1, first_delivery.bowler_id)
        bowler_ledgers: dict[str, BowlerState] = {}
        batter_ledgers: dict[str, BatterState] = {}
        last_over_idx = len(inn.overs) - 1
        for over_pos, over in enumerate(inn.overs):
            legal_in_over = 0
            for ball_pos, d in enumerate(over.deliveries):
                pre = replace(state, in_powerplay=_in_powerplay(inn, over.number, ball_pos + 1))
                # restore cumulative ledgers for returning players
                if pre.bowler.player_id != d.bowler_id and d.bowler_id in bowler_ledgers:
                    pre = replace(pre, bowler=bowler_ledgers[d.bowler_id])
                current = {b.player_id for b in (pre.striker, pre.non_striker) if b is not None}
                for pid in (d.batter_id, d.non_striker_id):
                    if pid not in current and pid in batter_ledgers:
                        if pre.striker is None and pid == d.batter_id:
                            pre = replace(pre, striker=batter_ledgers[pid])
                        elif pre.non_striker is None and pid == d.non_striker_id:
                            pre = replace(pre, non_striker=batter_ledgers[pid])
                post = apply(pre, d)
                yield pre, d, post
                state = post
                legal_in_over += int(d.wides == 0 and d.noballs == 0)
                bowler_ledgers[post.bowler.player_id] = post.bowler
                for b in (post.striker, post.non_striker):
                    if b is not None:
                        batter_ledgers[b.player_id] = b
            # per-over legal-ball reconciliation (SPEC §5 invariants):
            # declared miscounted overs are warnings, undeclared ones failures.
            if (
                over_pos != last_over_idx
                and legal_in_over != pm.balls_per_over
                and over.number not in inn.miscounted
            ):
                raise QuarantineError(
                    pm.match_id,
                    ReasonCode.E_BALL_ACCOUNTING,
                    f"innings {i + 1} over {over.number}: {legal_in_over} legal "
                    f"balls, expected {pm.balls_per_over}, not declared miscounted",
                )


def match_warnings(pm: ParsedMatch) -> list[str]:
    """Non-fatal reconciliation notes (declared miscounted overs)."""
    return [
        f"innings {i + 1} over {over_no}: umpire allowed {allowed} balls"
        for i, inn in enumerate(pm.innings)
        for over_no, allowed in sorted(inn.miscounted.items())
    ]


def stream_hash(pm: ParsedMatch) -> str:
    """Content hash of the full (s, d, s') stream — the determinism fingerprint.

    repr() of frozen dataclasses is deterministic given deterministic parse
    order, which JSON array order guarantees.
    """
    h = hashlib.sha256()
    for pre, d, post in replay_parsed(pm):
        h.update(repr(pre).encode())
        h.update(repr(d).encode())
        h.update(repr(post).encode())
    return h.hexdigest()


def load_match(path: Path) -> ParsedMatch:
    with open(path) as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise QuarantineError(path.stem, ReasonCode.E_SCHEMA, "top-level JSON not an object")
    return parse_match(raw, path.stem)


def replay(match_id: str) -> Iterator[Triple]:
    """SPEC §3 API: replay one match from the pinned snapshot by ID."""
    from cricstate.download import snapshot_dir

    pm = load_match(snapshot_dir() / "json" / f"{match_id}.json")
    yield from replay_parsed(pm)
