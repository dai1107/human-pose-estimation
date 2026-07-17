from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ViolationStatus = Literal["CLEAR", "CANDIDATE", "ACTIVE", "UNSURE"]


@dataclass(frozen=True)
class ViolationResult:
    code: str
    status: ViolationStatus
    confidence: float
    duration_ms: int
    started_ms: int | None

    @property
    def active(self) -> bool:
        return self.status == "ACTIVE"

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "status": self.status,
            "confidence": self.confidence,
            "duration_ms": self.duration_ms,
            "started_ms": self.started_ms,
            "active": self.active,
        }


class TemporalViolationTracker:
    """Require a condition to persist before exposing an active violation."""

    def __init__(self, code: str, min_hold_ms: int) -> None:
        self.code = str(code)
        self.min_hold_ms = max(0, int(min_hold_ms))
        self.reset()

    def reset(self) -> None:
        self.started_ms: int | None = None
        self.last_result = ViolationResult(
            self.code,
            "CLEAR",
            0.0,
            0,
            None,
        )

    def update(
        self,
        condition: bool | None,
        timestamp_ms: int | None,
        *,
        confidence: float,
    ) -> ViolationResult:
        resolved_confidence = max(0.0, min(1.0, float(confidence)))
        if condition is None or timestamp_ms is None:
            self.started_ms = None
            self.last_result = ViolationResult(
                self.code,
                "UNSURE",
                resolved_confidence,
                0,
                None,
            )
            return self.last_result
        if not condition:
            self.reset()
            return self.last_result
        if self.started_ms is None:
            self.started_ms = int(timestamp_ms)
        duration_ms = max(0, int(timestamp_ms) - self.started_ms)
        status: ViolationStatus = (
            "ACTIVE"
            if duration_ms >= self.min_hold_ms
            else "CANDIDATE"
        )
        self.last_result = ViolationResult(
            self.code,
            status,
            resolved_confidence,
            duration_ms,
            self.started_ms,
        )
        return self.last_result


__all__ = [
    "TemporalViolationTracker",
    "ViolationResult",
    "ViolationStatus",
]
