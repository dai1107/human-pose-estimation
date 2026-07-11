from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Iterable
from typing import Any

from hyrox.view_policy import normalize_camera_view, view_profile


@dataclass(frozen=True)
class CameraSource:
    camera_index: int
    camera_view: str
    mirror: bool = True
    name: str = ""

    def __post_init__(self) -> None:
        if self.camera_index < 0:
            raise ValueError("camera index must be >= 0")
        normalized = normalize_camera_view(self.camera_view)
        if view_profile(normalized) == "unknown":
            raise ValueError("multi-camera sources require an explicit front or side view")
        object.__setattr__(self, "camera_view", normalized)
        if not self.name:
            object.__setattr__(self, "name", f"camera_{self.camera_index}_{view_profile(normalized)}")


@dataclass(frozen=True)
class MultiCameraPlan:
    sources: tuple[CameraSource, ...]
    primary_camera_index: int
    synchronization_tolerance_ms: int = 50

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("at least one camera source is required")
        indices = [source.camera_index for source in self.sources]
        if len(indices) != len(set(indices)):
            raise ValueError("camera indices must be unique")
        if self.primary_camera_index not in indices:
            raise ValueError("primary camera must be included in sources")
        if self.synchronization_tolerance_ms < 0:
            raise ValueError("synchronization tolerance must be >= 0")

    @classmethod
    def from_sources(
        cls,
        sources: Iterable[CameraSource],
        *,
        primary_camera_index: int | None = None,
        synchronization_tolerance_ms: int = 50,
    ) -> MultiCameraPlan:
        resolved = tuple(sources)
        primary = resolved[0].camera_index if primary_camera_index is None and resolved else primary_camera_index
        if primary is None:
            raise ValueError("at least one camera source is required")
        return cls(resolved, primary, synchronization_tolerance_ms)

    @property
    def is_multiview(self) -> bool:
        return len(self.sources) > 1

    @property
    def view_profiles(self) -> frozenset[str]:
        return frozenset(view_profile(source.camera_view) for source in self.sources)

    @property
    def has_front_and_side(self) -> bool:
        return {"front", "side"} <= self.view_profiles


@dataclass(frozen=True)
class CapturedFrame:
    source: CameraSource
    timestamp_ms: int
    frame: Any


@dataclass(frozen=True)
class MultiCameraBundle:
    frames: tuple[CapturedFrame, ...]
    skew_ms: int
    synchronized: bool


class MultiCameraCapture:
    """Open and read one timestamped frame from every configured camera.

    Pose inference and decision fusion intentionally remain outside this class;
    each view must keep an independent backend and analyzer state.
    """

    def __init__(self, plan: MultiCameraPlan, *, width: int = 640, height: int = 480, fps: float = 30.0) -> None:
        self.plan = plan
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.fps = max(1.0, float(fps))
        self._captures: dict[int, Any] = {}

    @property
    def is_open(self) -> bool:
        return len(self._captures) == len(self.plan.sources)

    def open(self) -> None:
        if self.is_open:
            return
        import cv2

        self.close()
        try:
            for source in self.plan.sources:
                capture = cv2.VideoCapture(source.camera_index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(source.camera_index)
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                capture.set(cv2.CAP_PROP_FPS, self.fps)
                if not capture.isOpened():
                    capture.release()
                    raise RuntimeError(f"could not open camera {source.camera_index} ({source.camera_view})")
                self._captures[source.camera_index] = capture
        except Exception:
            self.close()
            raise

    def read(self) -> MultiCameraBundle:
        if not self.is_open:
            raise RuntimeError("multi-camera capture is not open")
        frames: list[CapturedFrame] = []
        for source in self.plan.sources:
            capture = self._captures[source.camera_index]
            ok, frame = capture.read()
            timestamp_ms = time.monotonic_ns() // 1_000_000
            if not ok or frame is None:
                raise RuntimeError(f"camera {source.camera_index} returned no frame")
            if source.mirror:
                import cv2

                frame = cv2.flip(frame, 1)
            frames.append(CapturedFrame(source, timestamp_ms, frame))
        timestamps = [item.timestamp_ms for item in frames]
        skew_ms = max(timestamps) - min(timestamps) if timestamps else 0
        return MultiCameraBundle(tuple(frames), skew_ms, skew_ms <= self.plan.synchronization_tolerance_ms)

    def close(self) -> None:
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()

    def __enter__(self) -> MultiCameraCapture:
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def parse_camera_source(spec: str) -> CameraSource:
    parts = [part.strip() for part in spec.split(":")]
    if len(parts) not in {2, 3}:
        raise ValueError("camera source must use INDEX:VIEW or INDEX:VIEW:MIRROR")
    try:
        index = int(parts[0])
    except ValueError as exc:
        raise ValueError(f"invalid camera index: {parts[0]}") from exc
    mirror = True
    if len(parts) == 3:
        value = parts[2].lower()
        if value not in {"mirror", "no-mirror"}:
            raise ValueError("camera mirror mode must be mirror or no-mirror")
        mirror = value == "mirror"
    return CameraSource(index, parts[1], mirror=mirror)


__all__ = [
    "CameraSource",
    "CapturedFrame",
    "MultiCameraBundle",
    "MultiCameraCapture",
    "MultiCameraPlan",
    "parse_camera_source",
]
