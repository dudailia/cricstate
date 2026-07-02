"""Golden round-trips through the full parse → replay pipeline (SPEC §11, §13)."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from cricstate.parser import parse_match
from cricstate.quarantine import QuarantineError, ReasonCode
from cricstate.replay import load_match, match_warnings, replay_parsed, stream_hash
from cricstate.schemas import WicketKind

GOLDEN = Path(__file__).parent / "golden"
GOLDEN_IDS = [p.stem for p in sorted(GOLDEN.glob("*.json"))]


@pytest.mark.parametrize("match_id", GOLDEN_IDS)
def test_golden_replays_with_invariants(match_id: str) -> None:
    pm = load_match(GOLDEN / f"{match_id}.json")
    per_innings_sum: dict[int, int] = {}
    last_state = {}
    n = 0
    for pre, d, post in replay_parsed(pm):
        n += 1
        idx = post.innings_idx
        per_innings_sum[idx] = per_innings_sum.get(idx, 0) + d.runs_total
        last_state[idx] = post
        assert post.wickets <= 10
        assert post.legal_balls <= post.max_balls
        assert post.runs == pre.runs + d.runs_total
    assert n > 0 or pm.no_result
    # runs conservation: final state runs == Σ runs_total + pre-innings penalties
    for idx, final in last_state.items():
        pre_pen = pm.innings[idx - 1].penalty_pre
        assert final.runs == per_innings_sum[idx] + pre_pen


@pytest.mark.parametrize("match_id", GOLDEN_IDS)
def test_golden_determinism_hash_stable(match_id: str) -> None:
    a = stream_hash(load_match(GOLDEN / f"{match_id}.json"))
    b = stream_hash(load_match(GOLDEN / f"{match_id}.json"))
    assert a == b
    assert len(a) == 64


def test_super_over_innings_indexed_above_two() -> None:
    pm = load_match(GOLDEN / "1187669.json")
    super_idx = [i + 1 for i, inn in enumerate(pm.innings) if inn.super_over]
    assert super_idx and all(i > 2 for i in super_idx)


def test_miscounted_over_is_warning_not_failure() -> None:
    pm = load_match(GOLDEN / "65273.json")
    triples = list(replay_parsed(pm))  # must not raise despite the 5-ball over
    assert triples
    assert match_warnings(pm) == ["innings 1 over 2: umpire allowed 5 balls"]


def test_undeclared_short_over_quarantines() -> None:
    with open(GOLDEN / "1187669.json") as fh:
        raw: dict[str, Any] = json.load(fh)
    raw = copy.deepcopy(raw)
    # drop one LEGAL delivery from a middle over without declaring it miscounted
    deliveries = raw["innings"][0]["overs"][1]["deliveries"]
    for k, dl in enumerate(deliveries):
        ex = dl.get("extras", {})
        if not ex.get("wides") and not ex.get("noballs"):
            del deliveries[k]
            break
    pm = parse_match(raw, "mutated")
    with pytest.raises(QuarantineError) as exc:
        list(replay_parsed(pm))
    assert exc.value.record.reason is ReasonCode.E_BALL_ACCOUNTING


def test_retired_hurt_golden_does_not_count_as_wicket() -> None:
    pm = load_match(GOLDEN / "804685.json")
    for _, d, post in replay_parsed(pm):
        for w in d.wickets:
            if w.kind is WicketKind.RETIRED_HURT:
                assert (post.runs, post.legal_balls) not in post.fow[-1:] or not post.fow


def test_dl_golden_carries_revised_target() -> None:
    pm = load_match(GOLDEN / "1499666.json")
    chase = [t for t in replay_parsed(pm) if t[0].innings_idx == 2]
    assert chase
    pre = chase[0][0]
    assert pre.dls_applied
    assert pre.target == pm.innings[1].target_runs
    assert pre.max_balls == pm.innings[1].target_balls


def test_penalty_golden_conserves_runs() -> None:
    pm = load_match(GOLDEN / "1298152.json")
    triples = [t for t in replay_parsed(pm) if t[2].innings_idx == 1]
    total = sum(d.runs_total for _, d, _ in triples)
    assert triples[-1][2].runs == total + pm.innings[0].penalty_pre
    assert any(d.penalty > 0 for _, d, _ in triples)
