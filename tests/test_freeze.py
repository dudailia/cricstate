"""Freeze enforcement tests (P3 pre-step). CI-runnable half + corpus half.

The CI half pins the DECLARED schemas (build.py constants) and the published
hashes in docs to the committed goldens — schema drift in code is red without
data. The corpus half (tests/test_corpus_evalkit.py) pins the ACTUAL parquet
schemas and recomputed hashes.
"""

from pathlib import Path

from cricstate.build import DELIVERY_SCHEMA, MATCH_SCHEMA, PLAYER_SCHEMA
from evalkit.freeze import (
    PINNED_CORPUS_HASH,
    PINNED_LABELS_HASH,
    load_golden,
    schema_as_dict,
)

DOCS = Path(__file__).resolve().parents[1] / "docs"


def test_declared_matches_schema_equals_golden() -> None:
    assert schema_as_dict(MATCH_SCHEMA) == load_golden("matches")


def test_declared_deliveries_schema_equals_golden() -> None:
    assert schema_as_dict(DELIVERY_SCHEMA) == load_golden("deliveries")


def test_declared_players_schema_equals_golden() -> None:
    assert schema_as_dict(PLAYER_SCHEMA) == load_golden("players")


def test_pinned_corpus_hash_matches_published_stats() -> None:
    assert PINNED_CORPUS_HASH in (DOCS / "STATS.md").read_text()


def test_pinned_labels_hash_matches_published_evidence() -> None:
    assert PINNED_LABELS_HASH in (DOCS / "EVIDENCE_M2_LABELS.md").read_text()
