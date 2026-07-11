from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Literal


PoseSource = Literal["mediapipe", "yolopose"]
BBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Keypoint:
    name: str
    x: float
    y: float
    z: float | None
    visibility: float
    confidence: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("keypoint name must not be empty")
        if not isfinite(self.x) or not isfinite(self.y):
            raise ValueError("keypoint x/y must be finite")
        if self.z is not None and not isfinite(self.z):
            raise ValueError("keypoint z must be finite or None")
        if not isfinite(self.visibility) or not 0.0 <= self.visibility <= 1.0:
            raise ValueError("visibility must be within [0, 1]")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")

    def is_valid(self, min_confidence: float = 0.0) -> bool:
        return (
            bool(self.name)
            and isfinite(self.x)
            and isfinite(self.y)
            and self.confidence >= min_confidence
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NormalizedPose:
    source: PoseSource
    frame_id: int
    timestamp_ms: int
    latency_ms: float
    image_width: int
    image_height: int
    keypoints: dict[str, Keypoint]
    bbox: BBox | None
    overall_confidence: float

    def __post_init__(self) -> None:
        if self.source not in {"mediapipe", "yolopose"}:
            raise ValueError(f"unsupported pose source: {self.source}")
        if self.frame_id < 0:
            raise ValueError("frame_id must be >= 0")
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be >= 0")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if not isfinite(self.latency_ms) or self.latency_ms < 0.0:
            raise ValueError("latency_ms must be finite and >= 0")
        if not isfinite(self.overall_confidence) or not 0.0 <= self.overall_confidence <= 1.0:
            raise ValueError("overall_confidence must be within [0, 1]")
        if any(name != point.name for name, point in self.keypoints.items()):
            raise ValueError("keypoint dictionary keys must match Keypoint.name")
        if self.bbox is not None:
            x1, y1, x2, y2 = self.bbox
            if not all(isfinite(value) for value in self.bbox) or x2 < x1 or y2 < y1:
                raise ValueError("bbox must contain finite ordered pixel coordinates")

    def get(self, name: str) -> Keypoint | None:
        return self.keypoints.get(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "frame_id": self.frame_id,
            "timestamp_ms": self.timestamp_ms,
            "latency_ms": self.latency_ms,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "keypoints": {name: point.to_dict() for name, point in self.keypoints.items()},
            "bbox": self.bbox,
            "overall_confidence": self.overall_confidence,
        }


__all__ = ["BBox", "Keypoint", "NormalizedPose", "PoseSource"]
