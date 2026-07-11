from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from .keypoints import COMMON_KEYPOINTS, MEDIAPIPE_TO_COMMON, YOLO_COCO17_TO_COMMON
from .schema import BBox, Keypoint, NormalizedPose


def _get(value: object, name: str, default: object | None = None) -> object | None:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _safe_float(value: object, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if isfinite(number) else default


def _confidence(value: object, default: float = 0.0) -> float:
    number = _safe_float(value, default)
    return max(0.0, min(1.0, default if number is None else number))


def _metadata(
    result: object,
    timestamp_ms: int,
    frame_id: int,
    latency_ms: float | None,
) -> tuple[int, int, float]:
    resolved_timestamp = int(timestamp_ms)
    resolved_frame_id = int(frame_id)
    inferred_latency = _safe_float(_get(result, "inference_time_ms"), 0.0)
    resolved_latency = inferred_latency if latency_ms is None else _safe_float(latency_ms, inferred_latency)
    return resolved_timestamp, resolved_frame_id, max(0.0, float(resolved_latency or 0.0))


def _mean_confidence(keypoints: Mapping[str, Keypoint]) -> float:
    if not keypoints:
        return 0.0
    return sum(point.confidence for point in keypoints.values()) / len(keypoints)


def _bbox_from_keypoints(keypoints: Mapping[str, Keypoint]) -> BBox | None:
    valid = [point for point in keypoints.values() if isfinite(point.x) and isfinite(point.y)]
    if not valid:
        return None
    return (
        min(point.x for point in valid),
        min(point.y for point in valid),
        max(point.x for point in valid),
        max(point.y for point in valid),
    )


def _is_valid_xy(x: float | None, y: float | None, image_width: int, image_height: int) -> bool:
    return (
        x is not None
        and y is not None
        and isfinite(x)
        and isfinite(y)
        and -0.1 * image_width <= x <= 1.1 * image_width
        and -0.1 * image_height <= y <= 1.1 * image_height
    )


def _point_from_normalized_landmark(
    name: str,
    landmark: object,
    image_width: int,
    image_height: int,
) -> Keypoint | None:
    x = _safe_float(_get(landmark, "x"))
    y = _safe_float(_get(landmark, "y"))
    if x is None or y is None:
        return None
    x_pixels = x * image_width
    y_pixels = y * image_height
    if not _is_valid_xy(x_pixels, y_pixels, image_width, image_height):
        return None
    visibility = _confidence(_get(landmark, "visibility"), 1.0)
    presence_value = _get(landmark, "presence")
    confidence = visibility if presence_value is None else min(visibility, _confidence(presence_value, 1.0))
    return Keypoint(
        name=name,
        x=x_pixels,
        y=y_pixels,
        z=_safe_float(_get(landmark, "z")),
        visibility=visibility,
        confidence=confidence,
    )


def _backend_keypoints(result: object) -> dict[str, object]:
    points = _get(result, "keypoints")
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes, bytearray)):
        return {}
    return {str(_get(point, "name", "")): point for point in points if _get(point, "name")}


def _normalized_pose_from_backend_keypoints(
    result: object,
    *,
    source: str,
    image_width: int,
    image_height: int,
    timestamp_ms: int,
    frame_id: int,
    latency_ms: float | None,
) -> NormalizedPose | None:
    by_name = _backend_keypoints(result)
    if not by_name or not bool(_get(result, "success", True)):
        return None
    converted: dict[str, Keypoint] = {}
    for name in COMMON_KEYPOINTS:
        point = by_name.get(name)
        if point is None:
            continue
        x = _safe_float(_get(point, "x"))
        y = _safe_float(_get(point, "y"))
        if x is None or y is None:
            continue
        x_pixels = x * image_width
        y_pixels = y * image_height
        if not _is_valid_xy(x_pixels, y_pixels, image_width, image_height):
            continue
        confidence = _confidence(_get(point, "confidence"))
        converted[name] = Keypoint(
            name=name,
            x=x_pixels,
            y=y_pixels,
            z=_safe_float(_get(point, "z")) if source == "mediapipe" else None,
            visibility=confidence,
            confidence=confidence,
        )
    if not converted:
        return None
    bbox = _bbox_from_keypoints(converted)
    if source == "yolopose":
        extra_bbox = _get(_get(result, "extra", {}), "bbox_pixels")
        if isinstance(extra_bbox, Sequence):
            bbox = _bbox_to_pixels(extra_bbox, image_width, image_height, normalized=False) or bbox
        elif isinstance(_get(result, "bbox"), Sequence):
            bbox = _bbox_to_pixels(
                _get(result, "bbox"), image_width, image_height, normalized=True
            ) or bbox
    resolved_timestamp, resolved_frame_id, resolved_latency = _metadata(
        result, timestamp_ms, frame_id, latency_ms
    )
    return NormalizedPose(
        source="mediapipe" if source == "mediapipe" else "yolopose",
        frame_id=resolved_frame_id,
        timestamp_ms=resolved_timestamp,
        latency_ms=resolved_latency,
        image_width=image_width,
        image_height=image_height,
        keypoints=converted,
        bbox=bbox,
        overall_confidence=_mean_confidence(converted),
    )


def _mediapipe_landmarks(result: object) -> Sequence[object] | None:
    extra = _get(result, "extra", {})
    # Fusion results have already restored ROI-normalized keypoints to the full
    # frame. Their retained raw SDK object is still ROI-local and must not win.
    if _get(extra, "roi_bbox_pixels") is not None:
        return None
    raw_result = _get(extra, "raw_result")
    candidate = raw_result if raw_result is not None else result
    pose_landmarks = _get(candidate, "pose_landmarks")
    if pose_landmarks is None:
        return None
    landmarks = _get(pose_landmarks, "landmark")
    if isinstance(landmarks, Sequence):
        return landmarks
    if isinstance(pose_landmarks, Sequence) and pose_landmarks:
        first = pose_landmarks[0]
        landmarks = _get(first, "landmark")
        if isinstance(landmarks, Sequence):
            return landmarks
        if isinstance(first, Sequence):
            return first
    return None


def mediapipe_to_normalized_pose(
    result: Any,
    image_width: int,
    image_height: int,
    timestamp_ms: int,
    *,
    frame_id: int = 0,
    latency_ms: float | None = None,
) -> NormalizedPose | None:
    if result is None:
        return None
    landmarks = _mediapipe_landmarks(result)
    if landmarks is None:
        return _normalized_pose_from_backend_keypoints(
            result,
            source="mediapipe",
            image_width=image_width,
            image_height=image_height,
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
            latency_ms=latency_ms,
        )

    converted: dict[str, Keypoint] = {}
    for index, name in MEDIAPIPE_TO_COMMON.items():
        if index >= len(landmarks):
            continue
        point = _point_from_normalized_landmark(name, landmarks[index], image_width, image_height)
        if point is not None:
            converted[name] = point
    if not converted:
        return None
    resolved_timestamp, resolved_frame_id, resolved_latency = _metadata(
        result, timestamp_ms, frame_id, latency_ms
    )
    return NormalizedPose(
        source="mediapipe",
        frame_id=resolved_frame_id,
        timestamp_ms=resolved_timestamp,
        latency_ms=resolved_latency,
        image_width=image_width,
        image_height=image_height,
        keypoints=converted,
        bbox=_bbox_from_keypoints(converted),
        overall_confidence=_mean_confidence(converted),
    )


def _as_numpy(value: object | None) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None


def _batch_xy(array: np.ndarray | None) -> np.ndarray | None:
    if array is None:
        return None
    if array.ndim == 2 and array.shape[-1] >= 2:
        array = array[None, :, :]
    if array.ndim != 3 or array.shape[-1] < 2:
        return None
    return array[:, :, :2]


def _batch_scores(array: np.ndarray | None, people: int, points: int) -> np.ndarray:
    if array is None:
        return np.ones((people, points), dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    scores = np.zeros((people, points), dtype=float)
    if array.ndim == 2:
        rows = min(people, array.shape[0])
        columns = min(points, array.shape[1])
        scores[:rows, :columns] = array[:rows, :columns]
    return scores


def _batch_boxes(array: np.ndarray | None) -> np.ndarray | None:
    if array is None:
        return None
    if array.ndim == 1 and array.shape[0] >= 4:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] < 4:
        return None
    return array[:, :4]


def _xy_to_pixels(xy: np.ndarray, width: int, height: int, normalized: bool) -> np.ndarray:
    pixels = np.asarray(xy, dtype=float).copy()
    if normalized:
        pixels[:, 0] *= width
        pixels[:, 1] *= height
    return pixels


def _looks_normalized(array: np.ndarray) -> bool:
    finite = array[np.isfinite(array)]
    return bool(finite.size) and float(finite.min()) >= -0.1 and float(finite.max()) <= 1.1


def _bbox_to_pixels(bbox: Sequence[float], width: int, height: int, normalized: bool) -> BBox | None:
    if len(bbox) < 4:
        return None
    values = [_safe_float(value) for value in bbox[:4]]
    if any(value is None for value in values):
        return None
    x1, y1, x2, y2 = (float(value) for value in values if value is not None)
    if normalized:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


@dataclass(frozen=True)
class _YoloCandidate:
    xy_pixels: np.ndarray
    scores: np.ndarray
    model_bbox: BBox | None

    @property
    def mean_confidence(self) -> float:
        finite = self.scores[np.isfinite(self.scores)]
        return float(np.mean(finite)) if finite.size else 0.0

    @property
    def model_bbox_area(self) -> float:
        if self.model_bbox is None:
            return 0.0
        x1, y1, x2, y2 = self.model_bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iter_results(result: object) -> Iterable[object]:
    raw_result = _get(_get(result, "extra", {}), "raw_result")
    candidate = raw_result if raw_result is not None else result
    if isinstance(candidate, (list, tuple)):
        return candidate
    return (candidate,)


def _collect_yolo_candidates(result: object, width: int, height: int) -> list[_YoloCandidate]:
    candidates: list[_YoloCandidate] = []
    for item in _iter_results(result):
        keypoint_container = _get(item, "keypoints")
        if keypoint_container is None:
            continue
        xy_value = _get(keypoint_container, "xy")
        xy_normalized = False
        if xy_value is None:
            xy_value = _get(keypoint_container, "xyn")
            xy_normalized = xy_value is not None
        if xy_value is None and isinstance(keypoint_container, (Sequence, np.ndarray)):
            xy_value = keypoint_container
        xy_batch = _batch_xy(_as_numpy(xy_value))
        if xy_batch is None:
            continue
        if _get(keypoint_container, "xy") is None and _get(keypoint_container, "xyn") is None:
            xy_normalized = _looks_normalized(xy_batch)

        score_value = _get(keypoint_container, "conf")
        if score_value is None:
            score_value = _get(item, "scores")
        scores = _batch_scores(_as_numpy(score_value), xy_batch.shape[0], xy_batch.shape[1])

        box_container = _get(item, "boxes")
        box_value = _get(box_container, "xyxy") if box_container is not None else None
        box_normalized = False
        if box_value is None and box_container is not None:
            box_value = _get(box_container, "xyxyn")
            box_normalized = box_value is not None
        if box_value is None:
            box_value = _get(item, "bboxes")
        if box_value is None:
            box_value = _get(item, "bbox")
        boxes = _batch_boxes(_as_numpy(box_value))
        if boxes is not None and (box_container is None or _get(box_container, "xyxy") is None):
            box_normalized = box_normalized or _looks_normalized(boxes)

        for person_index in range(xy_batch.shape[0]):
            xy_pixels = _xy_to_pixels(xy_batch[person_index], width, height, xy_normalized)
            model_bbox = None
            if boxes is not None and person_index < len(boxes):
                model_bbox = _bbox_to_pixels(boxes[person_index], width, height, box_normalized)
            candidates.append(
                _YoloCandidate(
                    xy_pixels=xy_pixels,
                    scores=scores[person_index],
                    model_bbox=model_bbox,
                )
            )
    return candidates


def _select_yolo_candidate(candidates: Sequence[_YoloCandidate]) -> _YoloCandidate | None:
    if not candidates:
        return None
    with_bbox = [candidate for candidate in candidates if candidate.model_bbox is not None]
    if with_bbox:
        return max(with_bbox, key=lambda candidate: candidate.model_bbox_area)
    return max(candidates, key=lambda candidate: candidate.mean_confidence)


def yolopose_to_normalized_pose(
    result: Any,
    image_width: int,
    image_height: int,
    timestamp_ms: int,
    *,
    frame_id: int = 0,
    latency_ms: float | None = None,
) -> NormalizedPose | None:
    if result is None:
        return None
    candidates = _collect_yolo_candidates(result, image_width, image_height)
    selected = _select_yolo_candidate(candidates)
    if selected is None:
        return _normalized_pose_from_backend_keypoints(
            result,
            source="yolopose",
            image_width=image_width,
            image_height=image_height,
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
            latency_ms=latency_ms,
        )

    converted: dict[str, Keypoint] = {}
    for index, name in YOLO_COCO17_TO_COMMON.items():
        if index >= len(selected.xy_pixels):
            continue
        x = _safe_float(selected.xy_pixels[index, 0])
        y = _safe_float(selected.xy_pixels[index, 1])
        confidence = _confidence(selected.scores[index] if index < len(selected.scores) else 0.0)
        if confidence <= 0.0:
            continue
        if not _is_valid_xy(x, y, image_width, image_height):
            continue
        converted[name] = Keypoint(
            name=name,
            x=float(x),
            y=float(y),
            z=None,
            visibility=confidence,
            confidence=confidence,
        )
    if not converted:
        return None
    resolved_timestamp, resolved_frame_id, resolved_latency = _metadata(
        result, timestamp_ms, frame_id, latency_ms
    )
    return NormalizedPose(
        source="yolopose",
        frame_id=resolved_frame_id,
        timestamp_ms=resolved_timestamp,
        latency_ms=resolved_latency,
        image_width=image_width,
        image_height=image_height,
        keypoints=converted,
        bbox=selected.model_bbox or _bbox_from_keypoints(converted),
        overall_confidence=_mean_confidence(converted),
    )


def normalize_backend_pose_result(
    result: Any,
    image_width: int,
    image_height: int,
    timestamp_ms: int,
    *,
    frame_id: int,
    latency_ms: float | None = None,
) -> NormalizedPose | None:
    model_name = str(_get(result, "model_name", "")).lower()
    if "yolo" in model_name:
        return yolopose_to_normalized_pose(
            result,
            image_width,
            image_height,
            timestamp_ms,
            frame_id=frame_id,
            latency_ms=latency_ms,
        )
    if "mediapipe" in model_name:
        return mediapipe_to_normalized_pose(
            result,
            image_width,
            image_height,
            timestamp_ms,
            frame_id=frame_id,
            latency_ms=latency_ms,
        )
    raise ValueError(f"unsupported pose backend result: {model_name or 'unknown'}")


def format_normalized_pose_debug(pose: NormalizedPose | None) -> str:
    if pose is None:
        return "[pose] no person detected"
    bbox = "None" if pose.bbox is None else "(" + ", ".join(f"{value:.0f}" for value in pose.bbox) + ")"
    return (
        f"[pose] source={pose.source} frame={pose.frame_id} ts={pose.timestamp_ms}ms keypoints={len(pose.keypoints)} "
        f"confidence={pose.overall_confidence:.2f} latency={pose.latency_ms:.1f}ms bbox={bbox}"
    )


__all__ = [
    "format_normalized_pose_debug",
    "mediapipe_to_normalized_pose",
    "normalize_backend_pose_result",
    "yolopose_to_normalized_pose",
]
