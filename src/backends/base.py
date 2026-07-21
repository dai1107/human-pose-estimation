from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Keypoint:
    name: str
    x: float
    y: float
    z: float = 0.0
    confidence: float = 0.0
    source_model: str = ""
    visibility: float | None = None
    presence: float | None = None


@dataclass(frozen=True)
class PoseResult:
    keypoints: list[Keypoint]
    connections: tuple[tuple[int, int], ...]
    model_name: str
    num_keypoints: int
    success: bool
    inference_time_ms: float
    bbox: tuple[float, float, float, float] | None = None
    timestamp_ms: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class PoseBackend(Protocol):
    model_name: str

    def detect(self, frame: Any, timestamp_ms: int | None = None) -> PoseResult:
        ...

    def close(self) -> None:
        ...
