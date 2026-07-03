import copy
import json
from pathlib import Path
from typing import Any

import pytest

from cricstate.parser import ParsedMatch, parse_match
from cricstate.quarantine import QuarantineError, ReasonCode
from cricstate.schemas import WicketKind

GOLDEN = Path(__file__).parent / "golden"
GOLDEN_IDS = [p.stem for p in sorted(GOLDEN.glob("*.json"))]


def load(match_id: str) -> dict[str, Any]:
    with open(GOLDEN / f"{match_id}.json") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def parse(match_id: str) -> ParsedMatch:
    return parse_match(load(match_id), match_id)


@pytest.mark.parametrize("match_id", GOLDEN_IDS)
def test_all_goldens_parse(match_id: str) -> None:
    pm = parse(match_id)
    assert pm.match_id == match_id
    assert pm.fmt in ("t20", "odi")
    assert pm.balls_per_over == 6


def test_golden_tie_super_over() -> None:
    pm = parse("1187669")
    assert pm.outcome_result == "tie"
    assert pm.outcome_winner is None
    assert pm.outcome_eliminator == "England"  # M1.2: super-over winner
    assert pm.outcome_bowl_out is None
    assert any(inn.super_over for inn in pm.innings)
    assert len(pm.innings) > 2


def test_golden_dl_result() -> None:
    pm = parse("1499666")
    assert pm.dls_applied
    chasing = pm.innings[1]
    assert chasing.target_runs is not None
    assert chasing.target_balls is not None


def test_golden_penalty_runs() -> None:
    pm = parse("1298152")
    deliveries = [d for inn in pm.innings for ov in inn.overs for d in ov.deliveries]
    assert any(d.penalty > 0 for d in deliveries)


def test_golden_retired_hurt() -> None:
    pm = parse("804685")
    wickets = [
        w for inn in pm.innings for ov in inn.overs for d in ov.deliveries for w in d.wickets
    ]
    assert any(w.kind is WicketKind.RETIRED_HURT for w in wickets)


def test_golden_miscounted_over() -> None:
    pm = parse("65273")
    assert any(inn.miscounted == {2: 5} for inn in pm.innings)


def test_golden_stumped_off_wide() -> None:
    pm = parse("1197049")
    hits = [
        d
        for inn in pm.innings
        for ov in inn.overs
        for d in ov.deliveries
        if d.wides >= 1 and any(w.kind is WicketKind.STUMPED for w in d.wickets)
    ]
    assert hits


def test_golden_uncontested_toss() -> None:
    assert parse("1322004").toss_uncontested


def test_golden_no_result() -> None:
    pm = parse("1409478")
    assert pm.no_result
    assert pm.outcome_winner is None


def test_golden_concussion_replacement_parses() -> None:
    # replacements are match metadata, not state events; the file must parse.
    pm = parse("1334913")
    assert pm.innings


def test_golden_wide_with_runs() -> None:
    pm = parse("1534732")
    deliveries = [d for inn in pm.innings for ov in inn.overs for d in ov.deliveries]
    assert any(d.wides >= 2 for d in deliveries)


# --- quarantine negatives -----------------------------------------------------


def reason_of(raw: dict[str, Any]) -> ReasonCode:
    with pytest.raises(QuarantineError) as exc:
        parse_match(raw, "test")
    return exc.value.record.reason


def test_wrong_version_quarantines() -> None:
    raw = load("1534732")
    raw["meta"]["data_version"] = "1.1.0"
    assert reason_of(raw) is ReasonCode.E_VERSION


def test_out_of_scope_format_quarantines() -> None:
    raw = load("1534732")
    raw["info"]["match_type"] = "Test"
    assert reason_of(raw) is ReasonCode.E_FORMAT_OOS


def test_unknown_delivery_key_quarantines() -> None:
    raw = copy.deepcopy(load("1534732"))
    raw["innings"][0]["overs"][0]["deliveries"][0]["surprise"] = 1
    assert reason_of(raw) is ReasonCode.E_SCHEMA


def test_unknown_wicket_kind_quarantines() -> None:
    raw = copy.deepcopy(load("1197049"))
    for inn in raw["innings"]:
        for ov in inn["overs"]:
            for dl in ov["deliveries"]:
                for w in dl.get("wickets", []):
                    w["kind"] = "handled the ball"  # folded into obstructing; not in enum
    assert reason_of(raw) is ReasonCode.E_UNKNOWN_WICKET_KIND


def test_registry_miss_quarantines() -> None:
    raw = copy.deepcopy(load("1534732"))
    raw["innings"][0]["overs"][0]["deliveries"][0]["batter"] = "A Nonexistent Player"
    assert reason_of(raw) is ReasonCode.E_REGISTRY_MISS


def test_runs_arithmetic_mismatch_quarantines() -> None:
    raw = copy.deepcopy(load("1534732"))
    raw["innings"][0]["overs"][0]["deliveries"][0]["runs"]["total"] = 99
    assert reason_of(raw) is ReasonCode.E_BALL_ACCOUNTING
