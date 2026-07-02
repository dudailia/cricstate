from dataclasses import FrozenInstanceError

import pytest

from cricstate.schemas import BatterState, BowlerState, Delivery, Wicket, WicketKind


def test_delivery_is_frozen_and_hashable() -> None:
    d = Delivery(
        batter_id="b1",
        bowler_id="bw1",
        non_striker_id="ns1",
        runs_batter=4,
        runs_extras=0,
        runs_total=4,
    )
    with pytest.raises(FrozenInstanceError):
        d.runs_batter = 6  # type: ignore[misc]
    assert hash(d) == hash(
        Delivery(
            batter_id="b1",
            bowler_id="bw1",
            non_striker_id="ns1",
            runs_batter=4,
            runs_extras=0,
            runs_total=4,
        )
    )
    assert d.source_confidence == 1.0


def test_wicket_kind_covers_spec_enum_exactly() -> None:
    assert {k.value for k in WicketKind} == {
        "bowled",
        "caught",
        "lbw",
        "run out",
        "stumped",
        "caught and bowled",
        "hit wicket",
        "retired out",
        "retired hurt",
        "obstructing the field",
        "timed out",
        "hit the ball twice",
    }


def test_wicket_and_ledgers_are_frozen() -> None:
    w = Wicket(player_out="p1", kind=WicketKind.STUMPED, fielders=("k1",))
    with pytest.raises(FrozenInstanceError):
        w.kind = WicketKind.BOWLED  # type: ignore[misc]
    bat = BatterState(player_id="p1")
    bowl = BowlerState(player_id="p2")
    assert (bat.runs, bat.balls) == (0, 0)
    assert (bowl.legal_balls, bowl.runs_conceded, bowl.wickets) == (0, 0, 0)
