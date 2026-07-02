"""SPEC §5 invariants as hypothesis property tests.

Synthetic innings are driven THROUGH δ: each generated step is turned into a
Delivery consistent with the current state's crease (new batters come from a
roster when a slot is vacant), then folded with apply(). Expected totals are
tracked independently of δ's arithmetic.
"""

from dataclasses import dataclass

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cricstate.quarantine import QuarantineError, ReasonCode
from cricstate.schemas import Delivery, MatchState, Wicket, WicketKind
from cricstate.transition import apply, innings_active
from tests.test_transition import fresh_state

DISMISSALS = [
    WicketKind.BOWLED,
    WicketKind.CAUGHT,
    WicketKind.LBW,
    WicketKind.RUN_OUT,
    WicketKind.STUMPED,
    WicketKind.RETIRED_HURT,
]


@dataclass(frozen=True)
class Step:
    kind: str  # legal | wide | noball | byes | legbyes
    runs_batter: int
    extra_runs: int  # runs beyond the mandatory penalty for wides/noballs
    wicket: WicketKind | None


steps = st.builds(
    Step,
    kind=st.sampled_from(["legal", "wide", "noball", "byes", "legbyes"]),
    runs_batter=st.integers(0, 6),
    extra_runs=st.integers(0, 4),
    wicket=st.none() | st.sampled_from(DISMISSALS),
)


def to_delivery(state: MatchState, step: Step, roster: list[str]) -> Delivery:
    striker_id = state.striker.player_id if state.striker else roster.pop()
    non_striker_id = state.non_striker.player_id if state.non_striker else roster.pop()
    runs_batter = wides = noballs = byes = legbyes = 0
    if step.kind == "legal":
        runs_batter = step.runs_batter
    elif step.kind == "wide":
        wides = 1 + step.extra_runs
    elif step.kind == "noball":
        noballs = 1
        runs_batter = step.runs_batter
    elif step.kind == "byes":
        byes = 1 + step.extra_runs
    else:
        legbyes = 1 + step.extra_runs
    wickets: tuple[Wicket, ...] = ()
    if step.wicket is not None:
        # stumped can't happen off a no-ball; run out can be either batter
        if not (step.wicket is WicketKind.STUMPED and noballs):
            out = non_striker_id if step.wicket is WicketKind.RUN_OUT else striker_id
            wickets = (Wicket(player_out=out, kind=step.wicket),)
    extras = wides + noballs + byes + legbyes
    return Delivery(
        batter_id=striker_id,
        bowler_id="bw1",
        non_striker_id=non_striker_id,
        runs_batter=runs_batter,
        runs_extras=extras,
        runs_total=runs_batter + extras,
        wides=wides,
        noballs=noballs,
        byes=byes,
        legbyes=legbyes,
        wickets=wickets,
    )


@settings(max_examples=300)
@given(st.lists(steps, min_size=1, max_size=140))
def test_invariants_hold_over_synthetic_innings(script: list[Step]) -> None:
    state = fresh_state()
    roster = [f"p{i}" for i in range(40, 0, -1)]
    expected_runs = 0
    expected_legal = 0
    expected_wickets = 0
    for step in script:
        if not innings_active(state):
            # pushing one more ball into a dead innings must quarantine
            with pytest.raises(QuarantineError) as exc:
                apply(state, to_delivery(state, step, roster))
            assert exc.value.record.reason is ReasonCode.E_DEAD_STATE
            break
        d = to_delivery(state, step, roster)
        prev = state
        state = apply(prev, d)

        # runs conservation, ball accounting, wicket bounds
        expected_runs += d.runs_total
        expected_legal += int(d.wides == 0 and d.noballs == 0)
        expected_wickets += sum(1 for w in d.wickets if w.kind is not WicketKind.RETIRED_HURT)
        assert state.runs == expected_runs
        assert state.legal_balls == expected_legal
        assert state.wickets == expected_wickets <= 10
        assert len(state.fow) == state.wickets

        # strike-rotation parity: crossings + over-boundary, XOR'd
        legal = d.wides == 0 and d.noballs == 0
        crossings = d.runs_batter + d.byes + d.legbyes + max(d.wides - 1, 0)
        swapped = (crossings % 2 == 1) ^ (legal and state.legal_balls % state.balls_per_over == 0)
        survivors = {b.player_id for b in (state.striker, state.non_striker) if b is not None}
        out = {w.player_out for w in d.wickets}
        assert survivors.isdisjoint(out)
        if not d.wickets:
            expect_striker = d.non_striker_id if swapped else d.batter_id
            assert state.striker is not None
            assert state.striker.player_id == expect_striker

        # batter ledger only moves for the striker, by the bat runs
        prev_ledger = {b.player_id: b for b in (prev.striker, prev.non_striker) if b is not None}
        for b in (state.striker, state.non_striker):
            if b is not None and b.player_id in prev_ledger:
                before = prev_ledger[b.player_id]
                if b.player_id == d.batter_id:
                    assert b.runs == before.runs + d.runs_batter
                    assert b.balls == before.balls + int(legal)
                else:
                    assert (b.runs, b.balls) == (before.runs, before.balls)


@settings(max_examples=200)
@given(st.lists(steps, min_size=1, max_size=80))
def test_determinism_same_fold_same_states(script: list[Step]) -> None:
    def run() -> list[MatchState]:
        state = fresh_state()
        roster = [f"p{i}" for i in range(40, 0, -1)]
        out = [state]
        for step in script:
            if not innings_active(state):
                break
            state = apply(state, to_delivery(state, step, roster))
            out.append(state)
        return out

    assert run() == run()
