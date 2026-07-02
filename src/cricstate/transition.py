"""The deterministic transition function δ (SPEC §5).

`apply` is total: it returns the successor state or raises QuarantineError —
never anything else. It is a pure function of (state, event); the observation
gap (striker slots that are None after a dismissal) is filled from the event
itself, since the incoming batter is unknown until observed.

Two things δ deliberately does NOT know, because Delivery carries no over or
innings metadata (both are corrected/managed by replay()):
- real over boundaries in miscounted overs (δ predicts with balls_per_over),
- powerplay windows (`in_powerplay` is set by replay on the pre-ball state).
"""

from dataclasses import replace

from cricstate.quarantine import QuarantineError, ReasonCode
from cricstate.schemas import BatterState, BowlerState, Delivery, MatchState, WicketKind

# Dismissal kinds credited to the bowler in their ledger.
BOWLER_CREDITED = frozenset(
    {
        WicketKind.BOWLED,
        WicketKind.CAUGHT,
        WicketKind.LBW,
        WicketKind.STUMPED,
        WicketKind.CAUGHT_BOWLED,
        WicketKind.HIT_WICKET,
    }
)


def innings_active(s: MatchState) -> bool:
    if s.wickets >= 10:
        return False
    if s.legal_balls >= s.max_balls:
        return False
    # Under D/L the operative target varies with each interruption and only the
    # final revision is recorded, so reaching the stored target mid-innings is
    # not reliable evidence of a dead innings (observed corpus-wide).
    if s.target is not None and s.runs >= s.target and not s.dls_applied:
        return False
    return True


def _observe_batters(s: MatchState, d: Delivery) -> tuple[BatterState, BatterState]:
    """Reconcile state's batter slots with the players the event names.

    Fills None gaps with fresh ledgers, and follows the data when δ's
    predicted ends are swapped (real cricket crosses on some dismissals in
    ways parity alone cannot see). A named batter that matches neither slot
    while both are occupied is a ball-accounting defect.
    """
    known = {b.player_id: b for b in (s.striker, s.non_striker) if b is not None}
    if d.batter_id in known and d.non_striker_id in known:
        return known[d.batter_id], known[d.non_striker_id]
    n_empty = 2 - len(known)
    new_names = [p for p in (d.batter_id, d.non_striker_id) if p not in known]
    if len(new_names) > n_empty:
        raise QuarantineError(
            s.match_id,
            ReasonCode.E_BALL_ACCOUNTING,
            f"batters {new_names} appeared but only {n_empty} slot(s) vacant "
            f"(state pair: {sorted(known)})",
        )
    striker = known.get(d.batter_id, BatterState(player_id=d.batter_id))
    non_striker = known.get(d.non_striker_id, BatterState(player_id=d.non_striker_id))
    if d.batter_id == d.non_striker_id:
        raise QuarantineError(
            s.match_id, ReasonCode.E_SCHEMA, f"batter == non_striker: {d.batter_id!r}"
        )
    return striker, non_striker


def apply(state: MatchState, event: Delivery) -> MatchState:
    """δ: (state, event) -> state', raising QuarantineError as the bottom element."""
    if not innings_active(state):
        raise QuarantineError(
            state.match_id,
            ReasonCode.E_DEAD_STATE,
            f"delivery arrived after innings end (balls={state.legal_balls}/"
            f"{state.max_balls}, wkts={state.wickets}, runs={state.runs}, "
            f"target={state.target})",
        )
    observed_striker, observed_non_striker = _observe_batters(state, event)

    legal = event.wides == 0 and event.noballs == 0
    runs = state.runs + event.runs_total
    legal_balls = state.legal_balls + int(legal)

    striker: BatterState | None = replace(
        observed_striker,
        runs=observed_striker.runs + event.runs_batter,
        balls=observed_striker.balls + int(legal),
    )
    non_striker: BatterState | None = observed_non_striker

    bowler = state.bowler
    if bowler.player_id != event.bowler_id:
        # New (or returning) bowler: δ alone cannot recover a returning
        # bowler's earlier figures from MatchState; replay() restores them.
        bowler = BowlerState(player_id=event.bowler_id)
    bowler = replace(
        bowler,
        legal_balls=bowler.legal_balls + int(legal),
        runs_conceded=bowler.runs_conceded + event.runs_batter + event.wides + event.noballs,
        wickets=bowler.wickets + sum(1 for w in event.wickets if w.kind in BOWLER_CREDITED),
    )

    wickets = state.wickets
    fow = state.fow
    fell = False
    for w in event.wickets:
        if w.kind is not WicketKind.RETIRED_HURT:
            wickets += 1
            fow = (*fow, (runs, legal_balls))
            fell = True
        # vacate the departing batter's slot (retired hurt vacates too)
        if striker is not None and striker.player_id == w.player_out:
            striker = None
        elif non_striker is not None and non_striker.player_id == w.player_out:
            non_striker = None
        else:
            raise QuarantineError(
                state.match_id,
                ReasonCode.E_BALL_ACCOUNTING,
                f"wicket for {w.player_out!r} who is not at the crease",
            )

    crossings = event.runs_batter + event.byes + event.legbyes + max(event.wides - 1, 0)
    if crossings % 2 == 1:
        striker, non_striker = non_striker, striker
    if legal and legal_balls % state.balls_per_over == 0:
        striker, non_striker = non_striker, striker

    partnership_runs = 0 if fell else state.partnership_runs + event.runs_total
    partnership_balls = 0 if fell else state.partnership_balls + int(legal)

    return replace(
        state,
        legal_balls=legal_balls,
        runs=runs,
        wickets=wickets,
        striker=striker,
        non_striker=non_striker,
        bowler=bowler,
        partnership_runs=partnership_runs,
        partnership_balls=partnership_balls,
        fow=fow,
        last_ball_was_noball=event.noballs > 0,
    )
