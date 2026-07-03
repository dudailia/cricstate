"""evalkit — Module 2: evaluation framework & baseline ladder (docs/SPEC_M2.md).

The measuring instrument and the bar. Consumes the M1 corpus at tag v1.1;
produces leaderboards every Module 3+ model must beat under the frozen
decision rule (SPEC_M2 §6).
"""

EVALKIT_VERSION = "0.2.0"
CORPUS_TAG = "v1.2"  # M1.2: matches.parquet gained outcome_eliminator/bowl_out
