"""Versioned pose caches for the fixed web demonstration videos."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.backends.base import Keypoint, PoseResult
from src.paths import installation_root


SAMPLE_CACHE_SCHEMA_VERSION = 1
SAMPLE_CACHE_FORMAT = "web-sample-pose-v1"
SAMPLE_CACHE_DIR = installation_root() / "configs" / "sample_pose_cache"


def cache_path_for(action: str) -> Path:
    return SAMPLE_CACHE_DIR / f"{str(action).strip().lower()}.json.gz"


def expected_source_backend(action: str) -> str:
    return "yolo-guided-mediapipe" if action == "lunge" else "mediapipe"


def source_assets(action: str) -> tuple[Path, ...]:
    root = installation_root()
    assets = [
        root / "models" / "pose_landmarker_full.task",
        root / "models" / "hand_landmarker.task",
    ]
    if action == "lunge":
        assets.insert(0, root / "yolo11n-pose.pt")
    return tuple(assets)


@lru_cache(maxsize=64)
def _sha256_for_stat(path_text: str, size: int, modified_ns: int) -> str:
    del size, modified_ns
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "size": int(stat.st_size),
        "sha256": _sha256_for_stat(
            str(resolved),
            int(stat.st_size),
            int(stat.st_mtime_ns),
        ),
    }


def source_asset_fingerprints(action: str) -> dict[str, dict[str, Any]]:
    root = installation_root().resolve()
    return {
        path.resolve().relative_to(root).as_posix(): file_fingerprint(path)
        for path in source_assets(action)
    }


def serialize_pose_result(result: PoseResult) -> dict[str, Any]:
    def finite(value: float | None) -> float | None:
        if value is None:
            return None
        parsed = float(value)
        return round(parsed, 8) if math.isfinite(parsed) else None

    bbox = None
    if result.bbox is not None and all(math.isfinite(float(value)) for value in result.bbox):
        bbox = [round(float(value), 8) for value in result.bbox]
    return {
        "success": bool(result.success),
        "model_name": str(result.model_name),
        "source_inference_ms": round(float(result.inference_time_ms), 3),
        "bbox": bbox,
        "keypoints": [
            {
                "name": point.name,
                "x": finite(point.x),
                "y": finite(point.y),
                "z": finite(point.z),
                "confidence": finite(point.confidence),
                "source_model": point.source_model,
                "visibility": finite(point.visibility),
                "presence": finite(point.presence),
            }
            for point in result.keypoints
        ],
    }


def serialize_hand_detections(detections: Mapping[str, Any]) -> dict[str, Any]:
    def finite(value: object) -> float | None:
        parsed = float(value)
        return round(parsed, 8) if math.isfinite(parsed) else None

    return {
        str(side): {
            "score": finite(getattr(detection, "score", 0.0)),
            "landmarks": [
                {
                    "x": finite(getattr(point, "x", float("nan"))),
                    "y": finite(getattr(point, "y", float("nan"))),
                    "z": finite(getattr(point, "z", float("nan"))),
                    "visibility": finite(getattr(point, "visibility", 0.0)),
                    "presence": finite(getattr(point, "presence", 0.0)),
                }
                for point in getattr(detection, "landmarks", ())
            ],
        }
        for side, detection in detections.items()
    }


def _deserialize_hand_detections(raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    from src.biomechanics.types import LandmarkPoint
    from webui.hands import make_hand_detection

    detections: dict[str, Any] = {}
    for side, value in raw.items():
        if not isinstance(value, Mapping):
            continue
        raw_points = value.get("landmarks")
        if not isinstance(raw_points, list) or len(raw_points) < 21:
            continue
        points = [
            LandmarkPoint(
                x=float("nan") if point.get("x") is None else float(point["x"]),
                y=float("nan") if point.get("y") is None else float(point["y"]),
                z=float("nan") if point.get("z") is None else float(point["z"]),
                visibility=(
                    0.0
                    if point.get("visibility") is None
                    else float(point["visibility"])
                ),
                presence=(
                    0.0
                    if point.get("presence") is None
                    else float(point["presence"])
                ),
            )
            for point in raw_points[:21]
            if isinstance(point, Mapping)
        ]
        if len(points) == 21:
            detections[str(side)] = make_hand_detection(
                str(side),
                points,
                score=float(value.get("score") or 0.0),
            )
    return detections


def build_cache_payload(
    *,
    action: str,
    video_path: str | Path,
    source_backend: str,
    fps: float,
    width: int,
    height: int,
    connections: Sequence[Sequence[int]],
    frames: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SAMPLE_CACHE_SCHEMA_VERSION,
        "cache_format": SAMPLE_CACHE_FORMAT,
        "action": action,
        "source_backend": source_backend,
        "video_fingerprint": file_fingerprint(video_path),
        "source_assets": source_asset_fingerprints(action),
        "fps": round(float(fps), 6),
        "width": int(width),
        "height": int(height),
        "frame_count": len(frames),
        "connections": [[int(start), int(end)] for start, end in connections],
        "frames": list(frames),
    }


def write_cache_payload(payload: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(target, "wt", encoding="utf-8", compresslevel=9) as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
    return target


class SamplePoseCacheBackend:
    """Sequential PoseBackend adapter over trusted precomputed sample frames."""

    model_name = "sample-cache"
    support_tier = "product"

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.source_backend = str(payload["source_backend"])
        self._connections = tuple(
            (int(pair[0]), int(pair[1]))
            for pair in payload.get("connections", ())
        )
        self._frames = list(payload["frames"])
        self._index = 0

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def detect(self, frame: object, timestamp_ms: int | None = None) -> PoseResult:
        del frame
        if self._index >= len(self._frames):
            raise RuntimeError("预计算示例姿态帧已耗尽")
        cached = self._frames[self._index]
        self._index += 1

        def number(value: object, *, fallback: float = 0.0) -> float:
            return fallback if value is None else float(value)

        keypoints = [
            Keypoint(
                name=str(point["name"]),
                x=number(point.get("x"), fallback=float("nan")),
                y=number(point.get("y"), fallback=float("nan")),
                z=number(point.get("z"), fallback=float("nan")),
                confidence=number(point.get("confidence")),
                source_model=str(point.get("source_model", "")),
                visibility=(
                    None
                    if point.get("visibility") is None
                    else float(point["visibility"])
                ),
                presence=(
                    None
                    if point.get("presence") is None
                    else float(point["presence"])
                ),
            )
            for point in cached.get("keypoints", ())
        ]
        raw_bbox = cached.get("bbox")
        bbox = (
            tuple(float(value) for value in raw_bbox)
            if isinstance(raw_bbox, list) and len(raw_bbox) == 4
            else None
        )
        return PoseResult(
            keypoints=keypoints,
            connections=self._connections,
            model_name=str(cached.get("model_name", self.source_backend)),
            num_keypoints=len(keypoints),
            success=bool(cached.get("success", bool(keypoints))),
            inference_time_ms=0.0,
            bbox=bbox,
            timestamp_ms=timestamp_ms,
            extra={
                "sample_pose_cache": True,
                "cached_source_backend": self.source_backend,
                "cached_source_inference_ms": float(
                    cached.get("source_inference_ms", 0.0)
                ),
                "cached_hand_detections": _deserialize_hand_detections(
                    cached.get("hands")
                ),
            },
        )

    def close(self) -> None:
        return None


def load_sample_pose_backend(
    *,
    action: str,
    video_path: str | Path,
    total_frames: int,
    path: str | Path | None = None,
) -> SamplePoseCacheBackend | None:
    cache_path = Path(path) if path is not None else cache_path_for(action)
    if not cache_path.is_file():
        return None
    try:
        with gzip.open(cache_path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        valid = bool(
            payload.get("schema_version") == SAMPLE_CACHE_SCHEMA_VERSION
            and payload.get("cache_format") == SAMPLE_CACHE_FORMAT
            and payload.get("action") == action
            and payload.get("source_backend") == expected_source_backend(action)
            and payload.get("video_fingerprint") == file_fingerprint(video_path)
            and payload.get("source_assets") == source_asset_fingerprints(action)
            and int(payload.get("frame_count", -1)) == int(total_frames)
            and isinstance(payload.get("frames"), list)
            and len(payload["frames"]) == int(total_frames)
        )
        return SamplePoseCacheBackend(payload) if valid else None
    except (EOFError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


__all__ = [
    "SAMPLE_CACHE_FORMAT",
    "SAMPLE_CACHE_SCHEMA_VERSION",
    "SamplePoseCacheBackend",
    "build_cache_payload",
    "cache_path_for",
    "expected_source_backend",
    "file_fingerprint",
    "load_sample_pose_backend",
    "serialize_pose_result",
    "serialize_hand_detections",
    "source_asset_fingerprints",
    "write_cache_payload",
]
