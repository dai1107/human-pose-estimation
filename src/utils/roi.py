from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from src.backends.base import Keypoint, PoseResult


BBox = tuple[float, float, float, float]


def expand_bbox(bbox: BBox, scale: float) -> BBox:
    x1, y1, x2, y2 = bbox
    scale = max(1.0, float(scale))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    width = (x2 - x1) * scale
    height = (y2 - y1) * scale
    return cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0


def clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    x1 = min(max(0.0, x1), float(max(0, width - 1)))
    y1 = min(max(0.0, y1), float(max(0, height - 1)))
    x2 = min(max(0.0, x2), float(max(0, width - 1)))
    y2 = min(max(0.0, y2), float(max(0, height - 1)))
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, float(max(0, width - 1)), float(max(0, height - 1))
    return x1, y1, x2, y2


def smooth_bbox(prev_bbox: BBox, curr_bbox: BBox, alpha: float) -> BBox:
    alpha = max(0.0, min(1.0, float(alpha)))
    keep = 1.0 - alpha
    return tuple(prev * keep + curr * alpha for prev, curr in zip(prev_bbox, curr_bbox))  # type: ignore[return-value]


def crop_roi(frame: np.ndarray, bbox: BBox | None) -> tuple[np.ndarray, BBox | None]:
    if bbox is None:
        return frame, None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, width, height)
    left, top = int(round(x1)), int(round(y1))
    right, bottom = int(round(x2)), int(round(y2))
    if right <= left or bottom <= top:
        return frame, None
    return frame[top:bottom, left:right].copy(), (float(left), float(top), float(right), float(bottom))


def restore_keypoints_from_roi(
    keypoints: Sequence[Keypoint],
    bbox: BBox | None,
    roi_shape: tuple[int, ...],
    original_shape: tuple[int, ...],
) -> list[Keypoint]:
    if bbox is None:
        return list(keypoints)
    original_h, original_w = original_shape[:2]
    roi_h, roi_w = roi_shape[:2]
    if original_w <= 0 or original_h <= 0 or roi_w <= 0 or roi_h <= 0:
        return list(keypoints)
    x1, y1, x2, y2 = bbox
    bbox_w = max(1e-6, x2 - x1)
    bbox_h = max(1e-6, y2 - y1)
    restored: list[Keypoint] = []
    for point in keypoints:
        restored.append(
            replace(
                point,
                x=(x1 + point.x * bbox_w) / original_w,
                y=(y1 + point.y * bbox_h) / original_h,
            )
        )
    return restored


def restore_result_from_roi(
    result: PoseResult,
    bbox: BBox | None,
    roi_shape: tuple[int, ...],
    original_shape: tuple[int, ...],
) -> PoseResult:
    if bbox is None:
        return result
    restored = restore_keypoints_from_roi(result.keypoints, bbox, roi_shape, original_shape)
    normalized_bbox = normalize_bbox(bbox, original_shape[1], original_shape[0])
    return replace(result, keypoints=restored, bbox=normalized_bbox, extra={**result.extra, "roi_bbox_pixels": bbox})


def normalize_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    return x1 / width, y1 / height, x2 / width, y2 / height
