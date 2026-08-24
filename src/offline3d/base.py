from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


Offline3DStatus = Literal["COMPLETED", "UNAVAILABLE", "FAILED", "CANCELLED"]
ProgressCallback = Callable[[float, str], None]


def _numeric_data(value: Any) -> Any:
    """Normalize numpy-like native arrays without flattening WHAM structure."""

    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _numeric_data(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_numeric_data(item) for item in value]
    if isinstance(value, (int, float)):
        return float(value)
    return value


@dataclass(slots=True)
class Offline3DFrame:
    """One native offline-3D observation on the source video timeline."""

    timestamp_ms: float
    frame_index: int | None = None
    joints_3d: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    smpl_pose: list[Any] = field(default_factory=list)
    body_orientation: list[Any] = field(default_factory=list)
    body_translation: list[Any] = field(default_factory=list)
    camera_motion: list[Any] = field(default_factory=list)
    global_trajectory: list[Any] = field(default_factory=list)
    confidence: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ms": round(float(self.timestamp_ms), 6),
            "frame_index": self.frame_index,
            "joints_3d": {
                name: [float(value) for value in point]
                for name, point in self.joints_3d.items()
            },
            "smpl_pose": _numeric_data(self.smpl_pose),
            "body_orientation": _numeric_data(self.body_orientation),
            "body_translation": _numeric_data(self.body_translation),
            "camera_motion": _numeric_data(self.camera_motion),
            "global_trajectory": _numeric_data(self.global_trajectory),
            "confidence": (
                None if self.confidence is None else float(self.confidence)
            ),
            "extra": dict(self.extra),
        }


@dataclass(slots=True)
class Offline3DResult:
    """Backend-neutral result which preserves each backend's native data."""

    backend: str
    status: Offline3DStatus
    reference_source: str
    frames: list[Offline3DFrame] = field(default_factory=list)
    coordinate_system: str = "backend_native"
    angle_definition: str = "backend_native_not_comparable_to_legacy_3point"
    trajectory_confidence: float | None = None
    processing_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str = ""

    @classmethod
    def unavailable(cls, backend: str, reason: str) -> "Offline3DResult":
        return cls(
            backend=backend,
            status="UNAVAILABLE",
            reference_source=backend,
            warnings=[reason],
        )

    @classmethod
    def failed(cls, backend: str, error: str) -> "Offline3DResult":
        return cls(
            backend=backend,
            status="FAILED",
            reference_source=backend,
            error=error,
        )

    def as_dict(self, *, include_frames: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": "offline3d_result_v1",
            "backend": self.backend,
            "status": self.status,
            "reference_source": self.reference_source,
            "coordinate_system": self.coordinate_system,
            "angle_definition": self.angle_definition,
            "trajectory_confidence": self.trajectory_confidence,
            "processing_time_ms": round(float(self.processing_time_ms), 3),
            "frame_count": len(self.frames),
            "metadata": dict(self.metadata),
            "warnings": list(self.warnings),
            "error": self.error,
            "is_ground_truth": False,
        }
        if include_frames:
            payload["frames"] = [frame.as_dict() for frame in self.frames]
        return payload


class Offline3DBackend(ABC):
    """Common interface for slow, offline 3D video analysis."""

    name: str

    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether this installation has enough configuration to execute."""

    @abstractmethod
    def analyze(
        self,
        video_path: str | Path,
        *,
        output_dir: str | Path | None = None,
        progress: ProgressCallback | None = None,
    ) -> Offline3DResult:
        """Analyze a complete video without real-time/playback-clock constraints."""


def parse_joints(
    value: Mapping[str, Sequence[Any]] | Sequence[Sequence[Any]] | None,
    names: Sequence[str] | None = None,
) -> dict[str, tuple[float, float, float]]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        items = value.items()
    else:
        labels = list(names or ())
        items = (
            (labels[index] if index < len(labels) else f"joint_{index}", point)
            for index, point in enumerate(value)
        )
    joints: dict[str, tuple[float, float, float]] = {}
    for name, point in items:
        coords = list(point)
        if len(coords) < 3:
            continue
        joints[str(name)] = (float(coords[0]), float(coords[1]), float(coords[2]))
    return joints
