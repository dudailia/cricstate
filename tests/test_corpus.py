"""Tests that need the pinned snapshot on disk (marked `corpus`; skipped in CI)."""

from pathlib import Path

import pytest

from cricstate.replay import load_match, replay, replay_parsed, stream_hash

pytestmark = pytest.mark.corpus

GOLDEN = Path(__file__).parent / "golden"
GOLDEN_IDS = [p.stem for p in sorted(GOLDEN.glob("*.json"))]


def snapshot_json() -> Path:
    from cricstate.download import snapshot_dir

    d = snapshot_dir() / "json"
    if not d.is_dir():
        pytest.skip("pinned snapshot not present")
    return d


@pytest.mark.parametrize("match_id", GOLDEN_IDS)
def test_golden_byte_identical_to_snapshot(match_id: str) -> None:
    snap = snapshot_json() / f"{match_id}.json"
    assert snap.read_bytes() == (GOLDEN / f"{match_id}.json").read_bytes()


@pytest.mark.parametrize("match_id", GOLDEN_IDS)
def test_replay_api_matches_golden_stream(match_id: str) -> None:
    snap_hash = stream_hash(load_match(snapshot_json() / f"{match_id}.json"))
    golden_hash = stream_hash(load_match(GOLDEN / f"{match_id}.json"))
    assert snap_hash == golden_hash
    n_via_api = sum(1 for _ in replay(match_id))
    n_via_golden = sum(1 for _ in replay_parsed(load_match(GOLDEN / f"{match_id}.json")))
    assert n_via_api == n_via_golden
