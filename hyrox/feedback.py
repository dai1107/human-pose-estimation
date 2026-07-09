from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


FeedbackLevel = Literal["info", "warn", "error"]


@dataclass(frozen=True)
class FeedbackMessage:
    level: FeedbackLevel
    code: str
    text: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.level not in {"info", "warn", "error"}:
            raise ValueError(f"unsupported feedback level: {self.level}")
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))

    def as_dict(self) -> dict[str, object]:
        return {
            "level": self.level,
            "code": self.code,
            "text": self.text,
            "confidence": self.confidence,
        }


__all__ = ["FeedbackMessage", "FeedbackLevel"]
