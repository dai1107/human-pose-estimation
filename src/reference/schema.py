from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.utils.time_utils import now_iso


CAMERA_VIEWS = {"side", "front", "front_left", "front_right", "unknown"}
MOVEMENT_SIDES = {"left", "right", "bilateral", "unknown"}


@dataclass(frozen=True)
class ClipRange:
    session_id: str
    start_ms: int | None = None
    end_ms: int | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    duration_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReferenceAction:
    reference_id: str
    name: str
    description: str = ""
    action_type: str = "generic_motion"
    camera_view: str = "unknown"
    movement_side: str = "unknown"
    created_at: str = field(default_factory=now_iso)
    source_session_ids: list[str] = field(default_factory=list)
    source_clip_ranges: list[dict[str, Any]] = field(default_factory=list)
    feature_set_name: str = "default_kinematics_v1"
    normalization_method: str = "body_relative"
    mirror_canonicalization_enabled: bool = False
    quality_summary: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    clip_count: int = 1

    def __post_init__(self) -> None:
        if self.camera_view not in CAMERA_VIEWS:
            object.__setattr__(self, "camera_view", "unknown")
        if self.movement_side not in MOVEMENT_SIDES:
            object.__setattr__(self, "movement_side", "unknown")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReferenceAction":
        return cls(
            reference_id=str(payload["reference_id"]),
            name=str(payload.get("name", payload["reference_id"])),
            description=str(payload.get("description", "")),
            action_type=str(payload.get("action_type", "generic_motion")),
            camera_view=str(payload.get("camera_view", "unknown")),
            movement_side=str(payload.get("movement_side", "unknown")),
            created_at=str(payload.get("created_at", now_iso())),
            source_session_ids=list(payload.get("source_session_ids", [])),
            source_clip_ranges=list(payload.get("source_clip_ranges", [])),
            feature_set_name=str(payload.get("feature_set_name", "default_kinematics_v1")),
            normalization_method=str(payload.get("normalization_method", "body_relative")),
            mirror_canonicalization_enabled=bool(payload.get("mirror_canonicalization_enabled", False)),
            quality_summary=dict(payload.get("quality_summary", {})),
            tags=list(payload.get("tags", [])),
            notes=str(payload.get("notes", "")),
            clip_count=int(payload.get("clip_count", 1)),
        )

