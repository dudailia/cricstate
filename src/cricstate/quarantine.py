"""Quarantine-not-crash semantics: closed reason-code enum + the one exception type.

Any match that cannot be parsed/validated/replayed raises QuarantineError with a
code from this closed enum. Nothing in this package may swallow an exception
silently (enforced by ruff BLE/S110/S112).
"""

from dataclasses import dataclass
from enum import StrEnum


class ReasonCode(StrEnum):
    E_SCHEMA = "E_SCHEMA"
    E_VERSION = "E_VERSION"
    E_REGISTRY_MISS = "E_REGISTRY_MISS"
    E_BALL_ACCOUNTING = "E_BALL_ACCOUNTING"
    E_UNKNOWN_WICKET_KIND = "E_UNKNOWN_WICKET_KIND"
    E_DEAD_STATE = "E_DEAD_STATE"
    E_FORMAT_OOS = "E_FORMAT_OOS"  # out-of-scope format for the v1 corpus
    E_OTHER = "E_OTHER"


@dataclass(frozen=True, slots=True)
class QuarantineRecord:
    match_id: str
    reason: ReasonCode
    detail: str


class QuarantineError(Exception):
    """Total-function ⊥: the match is quarantined, never silently dropped."""

    def __init__(self, match_id: str, reason: ReasonCode, detail: str) -> None:
        super().__init__(f"{match_id}: {reason} — {detail}")
        self.record = QuarantineRecord(match_id=match_id, reason=reason, detail=detail)
