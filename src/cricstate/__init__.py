"""cricstate Module 1: deterministic cricket state core (see docs/SPEC_M1.md)."""

# 1.1.0 (M1.2): additive — matches.parquet gained outcome_eliminator and
# outcome_bowl_out; no existing column modified or removed.
SCHEMA_VERSION = "1.1.0"
# SPEC_M1 pins 1.1.0, but the snapshot archive ships 1.2.0 exclusively —
# the pin follows the snapshot (see data/MANIFEST for the full note).
PINNED_DATA_VERSION = "1.2.0"
