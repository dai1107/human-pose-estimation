from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.backends.base import Keypoint, PoseResult
from src.utils.keypoint_schema import MEDIAPIPE_33_NAMES, MEDIAPIPE_CONNECTIONS


POSE_CACHE_SCHEMA_VERSION = 1
_LANDMARK_COUNT = len(MEDIAPIPE_33_NAMES)
_NAME_TO_INDEX = {
    name: index for index, name in enumerate(MEDIAPIPE_33_NAMES)
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PoseCacheIdentity:
    video_sha256: str
    backend: str
    model_type: str
    model_file_sha256: str
    inference_width: int
    inference_height: int
    segmentation_enabled: bool
    pose_config_sha256: str
    source_width: int
    source_height: int
    source_fps: float
    source_frame_count: int

    @classmethod
    def create(
        cls,
        *,
        video_path: str | Path,
        backend: str,
        model_type: str,
        model_path: str | Path,
        inference_width: int,
        inference_height: int,
        segmentation_enabled: bool,
        pose_config: Mapping[str, Any],
        source_width: int,
        source_height: int,
        source_fps: float,
        source_frame_count: int,
    ) -> "PoseCacheIdentity":
        return cls(
            video_sha256=sha256_file(video_path),
            backend=str(backend),
            model_type=str(model_type),
            model_file_sha256=sha256_file(model_path),
            inference_width=int(inference_width),
            inference_height=int(inference_height),
            segmentation_enabled=bool(segmentation_enabled),
            pose_config_sha256=_canonical_hash(pose_config),
            source_width=int(source_width),
            source_height=int(source_height),
            source_fps=round(float(source_fps), 9),
            source_frame_count=int(source_frame_count),
        )

    def key_payload(self) -> dict[str, Any]:
        return {
            "video_sha256": self.video_sha256,
            "backend": self.backend,
            "model_type": self.model_type,
            "model_file_sha256": self.model_file_sha256,
            "inference_resolution": {
                "width": self.inference_width,
                "height": self.inference_height,
            },
            "segmentation_enabled": self.segmentation_enabled,
            "pose_config_sha256": self.pose_config_sha256,
            "source": {
                "width": self.source_width,
                "height": self.source_height,
                "fps": self.source_fps,
                "frame_count": self.source_frame_count,
            },
        }

    @property
    def cache_key(self) -> str:
        return _canonical_hash(self.key_payload())


@dataclass(frozen=True)
class PoseLandmarkCache:
    cache_dir: Path
    metadata: Mapping[str, Any]
    frame_index: np.ndarray
    timestamp_ms: np.ndarray
    image_landmarks: np.ndarray
    world_landmarks: np.ndarray
    visibility: np.ndarray
    presence: np.ndarray
    world_visibility: np.ndarray
    world_presence: np.ndarray
    pose_detected: np.ndarray
    segmentation_available: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.frame_index.shape[0])

    def pose_result(self, index: int) -> PoseResult:
        if index < 0 or index >= self.frame_count:
            raise IndexError(f"pose cache frame out of range: {index}")
        points = _array_to_keypoints(
            self.image_landmarks[index],
            self.visibility[index],
            self.presence[index],
            model_name=str(self.metadata.get("model_type", "mediapipe-cache")),
        )
        world_points = _array_to_keypoints(
            self.world_landmarks[index],
            self.world_visibility[index],
            self.world_presence[index],
            model_name=f"{self.metadata.get('model_type', 'mediapipe-cache')}-world",
        )
        success = bool(self.pose_detected[index])
        world_available = bool(
            success
            and any(
                isfinite(point.x)
                and isfinite(point.y)
                and isfinite(point.z)
                for point in world_points
            )
        )
        return PoseResult(
            keypoints=points if success else [],
            connections=MEDIAPIPE_CONNECTIONS,
            model_name=str(self.metadata.get("model_type", "mediapipe-cache")),
            num_keypoints=len(points) if success else 0,
            success=success,
            inference_time_ms=0.0,
            bbox=_bbox(points) if success else None,
            timestamp_ms=int(round(float(self.timestamp_ms[index]))),
            extra={
                "world_keypoints": world_points if world_available else [],
                "world_landmarks_available": world_available,
                "segmentation_available": bool(
                    self.segmentation_available[index]
                ),
                "pose_cache_hit": True,
                "performance": {
                    "resize_ms": 0.0,
                    "color_convert_ms": 0.0,
                    "pose_inference_ms": 0.0,
                },
            },
        )


class CachedPoseBackend:
    model_name = "mediapipe-pose-cache"
    support_tier = "product"

    def __init__(self, cache: PoseLandmarkCache) -> None:
        self.cache = cache
        self._next_index = 0

    def detect(
        self,
        _frame: Any,
        timestamp_ms: int | None = None,
    ) -> PoseResult:
        result = self.cache.pose_result(self._next_index)
        self._next_index += 1
        return (
            replace(result, timestamp_ms=int(timestamp_ms))
            if timestamp_ms is not None
            else result
        )

    def close(self) -> None:
        return None


class PoseCacheWriter:
    def __init__(self, identity: PoseCacheIdentity) -> None:
        self.identity = identity
        self._frame_index: list[int] = []
        self._timestamp_ms: list[float] = []
        self._image_landmarks: list[np.ndarray] = []
        self._world_landmarks: list[np.ndarray] = []
        self._visibility: list[np.ndarray] = []
        self._presence: list[np.ndarray] = []
        self._world_visibility: list[np.ndarray] = []
        self._world_presence: list[np.ndarray] = []
        self._pose_detected: list[bool] = []
        self._segmentation_available: list[bool] = []

    @property
    def frame_count(self) -> int:
        return len(self._frame_index)

    def append(
        self,
        *,
        frame_index: int,
        timestamp_ms: float,
        result: PoseResult,
    ) -> None:
        expected = self.frame_count
        if int(frame_index) != expected:
            raise RuntimeError(
                "写入关键点缓存的帧索引不连续："
                f"expected={expected}, actual={frame_index}"
            )
        image, visibility, presence = _keypoints_to_arrays(result.keypoints)
        raw_world = result.extra.get("world_keypoints")
        world_points = (
            raw_world if isinstance(raw_world, (list, tuple)) else ()
        )
        world, world_visibility, world_presence = _keypoints_to_arrays(
            world_points
        )
        self._frame_index.append(int(frame_index))
        self._timestamp_ms.append(float(timestamp_ms))
        self._image_landmarks.append(image)
        self._world_landmarks.append(world)
        self._visibility.append(visibility)
        self._presence.append(presence)
        self._world_visibility.append(world_visibility)
        self._world_presence.append(world_presence)
        self._pose_detected.append(bool(result.success and result.keypoints))
        self._segmentation_available.append(
            result.extra.get("segmentation_mask") is not None
            or bool(result.extra.get("segmentation_masks"))
        )

    def write(self, cache_root: str | Path) -> tuple[Path, Path]:
        if self.frame_count != self.identity.source_frame_count:
            raise RuntimeError(
                "关键点缓存帧数与源视频不一致："
                f"cache={self.frame_count}, source={self.identity.source_frame_count}"
            )
        cache_dir = Path(cache_root) / self.identity.video_sha256
        cache_dir.mkdir(parents=True, exist_ok=True)
        landmarks_path = cache_dir / "pose_landmarks.npz"
        metadata_path = cache_dir / "pose_metadata.json"
        token = uuid.uuid4().hex
        landmarks_temp = cache_dir / f".pose_landmarks.{token}.tmp"
        metadata_temp = cache_dir / f".pose_metadata.{token}.tmp"
        arrays = {
            "frame_index": np.asarray(self._frame_index, dtype=np.int64),
            "timestamp_ms": np.asarray(self._timestamp_ms, dtype=np.float64),
            "image_landmarks": np.stack(self._image_landmarks).astype(
                np.float32, copy=False
            ),
            "world_landmarks": np.stack(self._world_landmarks).astype(
                np.float32, copy=False
            ),
            "visibility": np.stack(self._visibility).astype(
                np.float32, copy=False
            ),
            "presence": np.stack(self._presence).astype(
                np.float32, copy=False
            ),
            "world_visibility": np.stack(self._world_visibility).astype(
                np.float32, copy=False
            ),
            "world_presence": np.stack(self._world_presence).astype(
                np.float32, copy=False
            ),
            "pose_detected": np.asarray(self._pose_detected, dtype=np.bool_),
            "segmentation_available": np.asarray(
                self._segmentation_available,
                dtype=np.bool_,
            ),
        }
        metadata = {
            "schema_version": POSE_CACHE_SCHEMA_VERSION,
            "cache_key": self.identity.cache_key,
            **self.identity.key_payload(),
            "frame_count": self.frame_count,
            "landmark_names": list(MEDIAPIPE_33_NAMES),
            "files": {
                "landmarks": landmarks_path.name,
                "metadata": metadata_path.name,
            },
        }
        try:
            with landmarks_temp.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            metadata_temp.write_text(
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(landmarks_temp, landmarks_path)
            os.replace(metadata_temp, metadata_path)
        finally:
            landmarks_temp.unlink(missing_ok=True)
            metadata_temp.unlink(missing_ok=True)
        return landmarks_path, metadata_path


def load_pose_cache(
    cache_root: str | Path,
    identity: PoseCacheIdentity,
) -> PoseLandmarkCache | None:
    cache_dir = Path(cache_root) / identity.video_sha256
    landmarks_path = cache_dir / "pose_landmarks.npz"
    metadata_path = cache_dir / "pose_metadata.json"
    if not landmarks_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != POSE_CACHE_SCHEMA_VERSION:
            return None
        if metadata.get("cache_key") != identity.cache_key:
            return None
        if metadata.get("landmark_names") != list(MEDIAPIPE_33_NAMES):
            return None
        with np.load(landmarks_path, allow_pickle=False) as archive:
            arrays = {
                name: np.asarray(archive[name])
                for name in (
                    "frame_index",
                    "timestamp_ms",
                    "image_landmarks",
                    "world_landmarks",
                    "visibility",
                    "presence",
                    "world_visibility",
                    "world_presence",
                    "pose_detected",
                    "segmentation_available",
                )
            }
        _validate_arrays(arrays, identity.source_frame_count)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    return PoseLandmarkCache(
        cache_dir=cache_dir,
        metadata=metadata,
        **arrays,
    )


def _keypoints_to_arrays(
    keypoints: Sequence[Keypoint],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.full((_LANDMARK_COUNT, 3), np.nan, dtype=np.float32)
    visibility = np.zeros(_LANDMARK_COUNT, dtype=np.float32)
    presence = np.zeros(_LANDMARK_COUNT, dtype=np.float32)
    for point in keypoints:
        index = _NAME_TO_INDEX.get(point.name)
        if index is None:
            continue
        coordinates[index] = (point.x, point.y, point.z)
        visibility[index] = (
            point.confidence
            if point.visibility is None
            else float(point.visibility)
        )
        presence[index] = (
            point.confidence
            if point.presence is None
            else float(point.presence)
        )
    return coordinates, visibility, presence


def _array_to_keypoints(
    coordinates: np.ndarray,
    visibility: np.ndarray,
    presence: np.ndarray,
    *,
    model_name: str,
) -> list[Keypoint]:
    points: list[Keypoint] = []
    for index, name in enumerate(MEDIAPIPE_33_NAMES):
        x, y, z = (float(value) for value in coordinates[index])
        visible = float(visibility[index])
        present = float(presence[index])
        points.append(
            Keypoint(
                name=name,
                x=x,
                y=y,
                z=z,
                confidence=min(visible, present),
                source_model=model_name,
                visibility=visible,
                presence=present,
            )
        )
    return points


def _bbox(
    keypoints: Sequence[Keypoint],
) -> tuple[float, float, float, float] | None:
    usable = [
        point
        for point in keypoints
        if point.confidence >= 0.2
        and isfinite(point.x)
        and isfinite(point.y)
    ]
    if not usable:
        return None
    xs = [point.x for point in usable]
    ys = [point.y for point in usable]
    return min(xs), min(ys), max(xs), max(ys)


def _validate_arrays(
    arrays: Mapping[str, np.ndarray],
    expected_frames: int,
) -> None:
    expected_shapes = {
        "frame_index": (expected_frames,),
        "timestamp_ms": (expected_frames,),
        "image_landmarks": (expected_frames, _LANDMARK_COUNT, 3),
        "world_landmarks": (expected_frames, _LANDMARK_COUNT, 3),
        "visibility": (expected_frames, _LANDMARK_COUNT),
        "presence": (expected_frames, _LANDMARK_COUNT),
        "world_visibility": (expected_frames, _LANDMARK_COUNT),
        "world_presence": (expected_frames, _LANDMARK_COUNT),
        "pose_detected": (expected_frames,),
        "segmentation_available": (expected_frames,),
    }
    for name, expected_shape in expected_shapes.items():
        if arrays[name].shape != expected_shape:
            raise ValueError(
                f"invalid pose cache shape for {name}: "
                f"expected={expected_shape}, actual={arrays[name].shape}"
            )
    if not np.array_equal(
        arrays["frame_index"],
        np.arange(expected_frames, dtype=np.int64),
    ):
        raise ValueError("pose cache frame indexes are not contiguous")


__all__ = [
    "CachedPoseBackend",
    "POSE_CACHE_SCHEMA_VERSION",
    "PoseCacheIdentity",
    "PoseCacheWriter",
    "PoseLandmarkCache",
    "load_pose_cache",
    "sha256_file",
]
