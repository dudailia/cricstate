"""Freeze enforcement — mechanism, not diligence (P3 pre-step).

Golden schema files under tests/golden_schemas/ are the contract; changing a
parquet schema now requires an explicit golden update + version bump in the
same commit. Corpus and label hashes are pinned here; drift = red build.
"""

import json
from pathlib import Path

import polars as pl

GOLDEN_SCHEMA_DIR = Path(__file__).resolve().parents[2] / "tests" / "golden_schemas"

# Pinned at the P2 review gate (corpus tag v1.2). Drift = red build.
PINNED_CORPUS_HASH = "c08e4eba45ff7a71a51c4490cfe159a2ca34a7e5382bbc902041d147a11a6781"
PINNED_LABELS_HASH = "e11bf9c83276d10871b401263c351d7e4d93da85183b3c1d141fefbdad20324f"


def schema_as_dict(schema: pl.Schema | dict[str, pl.DataType]) -> dict[str, str]:
    return {name: str(dtype) for name, dtype in dict(schema).items()}


def load_golden(table: str) -> dict[str, str]:
    with open(GOLDEN_SCHEMA_DIR / f"{table}.json") as fh:
        golden: dict[str, str] = json.load(fh)
    return golden
