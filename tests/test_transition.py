from datetime import date

import pytest

from cricstate import SCHEMA_VERSION
from cricstate.quarantine import QuarantineError, ReasonCode
from cricstate.schemas import (
    BowlerState,
    Delivery,
    MatchState,
    Wicket,
    WicketKind,
)
from cricstate.transition import apply, innings_active

BASE = MatchState(
    schema_version=SCHEMA_VERSION,
    match_id="synthetic",
    fmt="t20",
    gender="male",
    venue_id="v",
    start_date=date(2026, 1, 1),
    competition=None,
    innings_idx=1,
    batting_team="A",
    bowling_team="B",
    balls_per_over=6,
    max_balls=120,
    legal_balls=0,
    runs=0,
    wickets=0,
    striker=None,
    non_striker=None,
    bowler=BowlerState(player_id="bw1"),
    target=None,
    dls_applied=False,
    in_powerplay=True,
    partnership_runs=0,
    partnership_balls=0,
    fow=(),
    last_ball_was_noball=False,
)


def fresh_state(**over: object) -> MatchState:
    from dataclasses import replace

    return replace(BASE, **over)  # type: ignore[arg-type]


def ball(
    runs_batter: int = 0,
    wides: int = 0,
    noballs: int = 0,
    byes: int = 0,
    legbyes: int = 0,
    penalty: int = 0,
    wickets: tuple[Wicket, ...] = (),
    batter: str = "s1",
    non_striker: str = "n1",
    bowler: str = "bw1",
) -> Delivery:
    extras = wides + noballs + byes + legbyes + penalty
    return Delivery(
        batter_id=batter,
        bowler_id=bowler,
        non_striker_id=non_striker,
        runs_batter=runs_batter,
        runs_extras=extras,
        runs_total=runs_batter + extras,
        wides=wides,
        noballs=noballs,
        byes=byes,
        legbyes=legbyes,
        penalty=penalty,
        wickets=wickets,
    )


def test_plain_single_rotates_strike() -> None:
    s = apply(fresh_state(), ball(runs_batter=1))
    assert s.runs == 1 and s.legal_balls == 1
    assert s.striker is not None and s.striker.player_id == "n1"
    assert s.non_striker is not None and s.non_striker.player_id == "s1"
    assert s.non_striker.runs == 1 and s.non_striker.balls == 1


def test_boundary_keeps_strike() -> None:
    s = apply(fresh_state(), ball(runs_batter=4))
    assert s.striker is not None and s.striker.player_id == "s1"


def test_over_end_swaps_ends() -> None:
    s = apply(fresh_state(), ball())  # first ball observes s1/n1
    for _ in range(4):
        s = apply(s, ball(batter=_striker(s), non_striker=_non_striker(s)))
    assert _striker(s) == "s1"
    s = apply(s, ball(batter=_striker(s), non_striker=_non_striker(s)))
    assert s.legal_balls == 6
    assert _striker(s) == "n1"  # dot ball + over end: ends swap once


def test_wide_is_not_a_legal_ball_and_wide_with_runs_crosses() -> None:
    s = apply(fresh_state(), ball(wides=2))  # wide + one run physically run
    assert s.legal_balls == 0 and s.runs == 2
    assert _striker(s) == "n1"  # wides-1 = 1 crossing
    assert s.striker is not None and s.striker.balls == 0


def test_noball_sets_flag_and_charges_bowler() -> None:
    s = apply(fresh_state(), ball(runs_batter=2, noballs=1))
    assert s.last_ball_was_noball
    assert s.bowler.runs_conceded == 3  # 2 off the bat + the no-ball
    assert s.bowler.legal_balls == 0


def test_byes_and_legbyes_do_not_charge_bowler_but_cross() -> None:
    s = apply(fresh_state(), ball(byes=1))
    assert s.bowler.runs_conceded == 0
    assert s.runs == 1
    assert _striker(s) == "n1"


def test_penalty_runs_count_to_team_only_and_do_not_cross() -> None:
    s = apply(fresh_state(), ball(penalty=5))
    assert s.runs == 5
    assert s.bowler.runs_conceded == 0
    assert _striker(s) == "s1"


def test_wicket_increments_records_fow_and_vacates_slot() -> None:
    w = Wicket(player_out="s1", kind=WicketKind.BOWLED)
    s = apply(fresh_state(), ball(wickets=(w,)))
    assert s.wickets == 1
    assert s.fow == ((0, 1),)
    assert s.striker is None
    assert s.bowler.wickets == 1
    assert s.partnership_runs == 0 and s.partnership_balls == 0


def test_run_out_not_credited_to_bowler() -> None:
    w = Wicket(player_out="n1", kind=WicketKind.RUN_OUT, fielders=("f1",))
    s = apply(fresh_state(), ball(runs_batter=1, wickets=(w,)))
    assert s.wickets == 1
    assert s.bowler.wickets == 0


def test_retired_hurt_vacates_but_is_not_a_dismissal() -> None:
    w = Wicket(player_out="s1", kind=WicketKind.RETIRED_HURT)
    s = apply(fresh_state(), ball(wickets=(w,)))
    assert s.wickets == 0
    assert s.fow == ()
    assert s.striker is None  # vacated; incoming batter unknown until observed
    assert s.partnership_balls == 1  # partnership survives a retirement


def test_retired_not_out_has_retired_hurt_semantics() -> None:
    w = Wicket(player_out="s1", kind=WicketKind.RETIRED_NOT_OUT)
    s = apply(fresh_state(), ball(wickets=(w,)))
    assert s.wickets == 0  # not a dismissal
    assert s.fow == ()
    assert s.striker is None  # slot vacated; batter may return
    assert s.bowler.wickets == 0
    assert s.partnership_balls == 1  # partnership survives a retirement


def test_stumping_off_wide_dismisses_without_legal_ball() -> None:
    w = Wicket(player_out="s1", kind=WicketKind.STUMPED, fielders=("k1",))
    s = apply(fresh_state(), ball(wides=1, wickets=(w,)))
    assert s.legal_balls == 0
    assert s.wickets == 1
    assert s.bowler.wickets == 1


def test_new_batter_fills_vacant_slot_fresh() -> None:
    w = Wicket(player_out="s1", kind=WicketKind.BOWLED)
    s = apply(fresh_state(), ball(wickets=(w,)))
    s = apply(s, ball(batter="s2", non_striker="n1", runs_batter=4))
    assert _striker(s) == "s2"
    assert s.striker is not None and s.striker.runs == 4


def test_dead_state_quarantines() -> None:
    s = fresh_state(wickets=10)
    with pytest.raises(QuarantineError) as exc:
        apply(s, ball())
    assert exc.value.record.reason is ReasonCode.E_DEAD_STATE
    assert not innings_active(s)


def test_target_reached_is_terminal() -> None:
    s = fresh_state(target=150, runs=150)
    with pytest.raises(QuarantineError):
        apply(s, ball())


def test_dl_innings_may_continue_past_recorded_target() -> None:
    # only the final D/L revision is recorded; earlier operative targets were
    # higher, so play legitimately continues past the stored number
    s = fresh_state(target=150, runs=150, dls_applied=True)
    assert innings_active(s)
    assert apply(s, ball()).runs == 150


def test_unknown_batter_with_full_crease_quarantines() -> None:
    s = apply(fresh_state(), ball())  # crease now occupied by s1/n1
    with pytest.raises(QuarantineError) as exc:
        apply(s, ball(batter="intruder", non_striker="s1"))
    assert exc.value.record.reason is ReasonCode.E_BALL_ACCOUNTING


def _striker(s: MatchState) -> str:
    assert s.striker is not None
    return s.striker.player_id


def _non_striker(s: MatchState) -> str:
    assert s.non_striker is not None
    return s.non_striker.player_id
