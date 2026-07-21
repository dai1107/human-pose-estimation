from __future__ import annotations

from math import hypot, isfinite, nan
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import numpy as np

from src.backends.base import Keypoint, PoseResult
from src.backends.mediapipe_backend import MediaPipeBackend
from src.backends.yolo_pose_backend import TargetSelect, YoloPoseBackend
from src.utils.keypoint_schema import (
    COCO_17_NAMES,
    MEDIAPIPE_33_NAMES,
    MEDIAPIPE_CONNECTIONS,
)


IDENTITY_MATCH_NAMES: tuple[str, ...] = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


class YoloGuidedMediaPipeBackend:
    """Use YOLO to select one athlete and MediaPipe to supplement that identity."""

    model_name = "yolo-guided-mediapipe"
    support_tier = "experimental"

    def __init__(
        self,
        yolo_model_path: str | Path = "yolo11n-pose.pt",
        mediapipe_model_path: str | Path = "models/pose_landmarker_full.task",
        *,
        target_select: TargetSelect = "tracking",
        device: str = "",
        num_poses: int = 5,
        min_match_points: int = 6,
        max_match_distance: float = 0.20,
        match_confidence: float = 0.20,
        temporal_foot_hold_ms: int = 1700,
        yolo_backend: Any | None = None,
        mediapipe_backend: Any | None = None,
    ) -> None:
        if min_match_points < 1:
            raise ValueError("min_match_points must be at least 1")
        if max_match_distance <= 0.0:
            raise ValueError("max_match_distance must be positive")
        self.min_match_points = int(min_match_points)
        self.max_match_distance = float(max_match_distance)
        self.match_confidence = float(match_confidence)
        self.temporal_foot_hold_ms = max(0, int(temporal_foot_hold_ms))
        self._foot_offsets: dict[
            str,
            tuple[float, float, float, int],
        ] = {}
        self._fallback_timestamp_ms = -1
        self.yolo_backend = yolo_backend or YoloPoseBackend(
            str(yolo_model_path),
            target_select=target_select,
            device=device,
        )
        self.mediapipe_backend = mediapipe_backend or MediaPipeBackend(
            mediapipe_model_path,
            output_segmentation_masks=False,
            num_poses=num_poses,
        )

    def detect(self, frame: np.ndarray, timestamp_ms: int | None = None) -> PoseResult:
        yolo_result = self.yolo_backend.detect(frame, timestamp_ms=timestamp_ms)
        if not yolo_result.success or not yolo_result.keypoints:
            return PoseResult(
                keypoints=[],
                connections=MEDIAPIPE_CONNECTIONS,
                model_name=self.model_name,
                num_keypoints=0,
                success=False,
                inference_time_ms=yolo_result.inference_time_ms,
                bbox=yolo_result.bbox,
                timestamp_ms=yolo_result.timestamp_ms,
                extra={
                    **dict(yolo_result.extra),
                    "identity_matched": False,
                    "identity_match_distance": None,
                    "identity_match_points": 0,
                    "mediapipe_candidate_count": 0,
                    "mediapipe_supplemental_available": False,
                    "mediapipe_temporal_foot_points": 0,
                    "mediapipe_temporal_foot_age_ms": None,
                },
            )

        mediapipe_error: str | None = None
        try:
            mediapipe_result = self.mediapipe_backend.detect(
                frame,
                timestamp_ms=timestamp_ms,
            )
        except Exception as exc:
            mediapipe_error = f"{type(exc).__name__}: {exc}"
            mediapipe_result = None

        candidates = self._pose_candidates(mediapipe_result)
        matched_candidate, match_distance, match_points = self._select_identity(
            yolo_result.keypoints,
            candidates,
        )
        identity_matched = (
            matched_candidate is not None
            and match_distance is not None
            and match_points >= self.min_match_points
            and match_distance <= self.max_match_distance
        )
        frame_timestamp_ms = self._frame_timestamp(
            timestamp_ms,
            yolo_result.timestamp_ms,
        )
        supplemental, temporal_foot_points, temporal_foot_age_ms = (
            self._supplemental_keypoints(
                yolo_result.keypoints,
                matched_candidate if identity_matched else None,
                frame_timestamp_ms,
            )
        )
        keypoints = self._fuse_keypoints(yolo_result.keypoints, supplemental)
        supplemental_available = bool(
            supplemental
            and any(
                point.name not in COCO_17_NAMES and self._usable(point)
                for point in supplemental
            )
        )
        inference_time_ms = yolo_result.inference_time_ms
        if mediapipe_result is not None:
            inference_time_ms += mediapipe_result.inference_time_ms

        extra = {
            **dict(yolo_result.extra),
            "identity_matched": identity_matched,
            "identity_match_distance": match_distance,
            "identity_match_points": match_points,
            "mediapipe_candidate_count": len(candidates),
            "mediapipe_supplemental_available": supplemental_available,
            "mediapipe_temporal_foot_points": temporal_foot_points,
            "mediapipe_temporal_foot_age_ms": temporal_foot_age_ms,
        }
        if mediapipe_error:
            extra["mediapipe_error"] = mediapipe_error

        return PoseResult(
            keypoints=keypoints,
            connections=MEDIAPIPE_CONNECTIONS,
            model_name=self.model_name,
            num_keypoints=len(keypoints),
            success=True,
            inference_time_ms=inference_time_ms,
            bbox=yolo_result.bbox,
            timestamp_ms=yolo_result.timestamp_ms,
            extra=extra,
        )

    def close(self) -> None:
        try:
            self.yolo_backend.close()
        finally:
            self.mediapipe_backend.close()

    def _pose_candidates(self, result: PoseResult | None) -> list[list[Keypoint]]:
        if result is None:
            return []
        candidates = result.extra.get("pose_candidates")
        if isinstance(candidates, (list, tuple)):
            resolved = [
                list(candidate)
                for candidate in candidates
                if isinstance(candidate, (list, tuple))
            ]
            if resolved:
                return resolved
        if result.success and result.keypoints:
            return [list(result.keypoints)]
        return []

    def _select_identity(
        self,
        yolo_keypoints: Sequence[Keypoint],
        candidates: Sequence[Sequence[Keypoint]],
    ) -> tuple[list[Keypoint] | None, float | None, int]:
        yolo_by_name = {point.name: point for point in yolo_keypoints}
        best_candidate: list[Keypoint] | None = None
        best_distance: float | None = None
        best_points = 0

        for raw_candidate in candidates:
            candidate = list(raw_candidate)
            candidate_by_name = {point.name: point for point in candidate}
            distances = [
                hypot(
                    yolo_by_name[name].x - candidate_by_name[name].x,
                    yolo_by_name[name].y - candidate_by_name[name].y,
                )
                for name in IDENTITY_MATCH_NAMES
                if name in yolo_by_name
                and name in candidate_by_name
                and self._usable(yolo_by_name[name])
                and self._usable(candidate_by_name[name])
            ]
            if len(distances) < self.min_match_points:
                continue
            distance = float(median(distances))
            if best_distance is None or distance < best_distance:
                best_candidate = candidate
                best_distance = distance
                best_points = len(distances)

        return best_candidate, best_distance, best_points

    def _supplemental_keypoints(
        self,
        yolo_keypoints: Sequence[Keypoint],
        matched_candidate: Sequence[Keypoint] | None,
        timestamp_ms: int,
    ) -> tuple[list[Keypoint] | None, int, int | None]:
        yolo_by_name = {point.name: point for point in yolo_keypoints}
        if matched_candidate is not None:
            matched_by_name = {point.name: point for point in matched_candidate}
            for side in ("left", "right"):
                ankle = yolo_by_name.get(f"{side}_ankle")
                if ankle is None or not self._usable(ankle):
                    continue
                for suffix in ("heel", "foot_index"):
                    name = f"{side}_{suffix}"
                    foot = matched_by_name.get(name)
                    if foot is None or not self._usable(foot):
                        continue
                    offset_x = foot.x - ankle.x
                    offset_y = foot.y - ankle.y
                    if hypot(offset_x, offset_y) > 0.25:
                        continue
                    self._foot_offsets[name] = (
                        offset_x,
                        offset_y,
                        foot.confidence,
                        timestamp_ms,
                    )
            return list(matched_candidate), 0, 0

        if self.temporal_foot_hold_ms <= 0:
            return None, 0, None
        inferred: list[Keypoint] = []
        ages: list[int] = []
        for side in ("left", "right"):
            ankle = yolo_by_name.get(f"{side}_ankle")
            if ankle is None or not self._usable(ankle):
                continue
            for suffix in ("heel", "foot_index"):
                name = f"{side}_{suffix}"
                stored = self._foot_offsets.get(name)
                if stored is None:
                    continue
                offset_x, offset_y, foot_confidence, observed_ms = stored
                age_ms = max(0, timestamp_ms - observed_ms)
                if age_ms > self.temporal_foot_hold_ms:
                    continue
                confidence = min(0.78, ankle.confidence, foot_confidence)
                confidence -= 0.18 * age_ms / max(1, self.temporal_foot_hold_ms)
                if confidence < self.match_confidence:
                    continue
                inferred.append(
                    Keypoint(
                        name=name,
                        x=max(0.0, min(1.0, ankle.x + offset_x)),
                        y=max(0.0, min(1.0, ankle.y + offset_y)),
                        z=ankle.z,
                        confidence=max(0.0, confidence),
                        source_model="mediapipe-from-yolo-ankle",
                    )
                )
                ages.append(age_ms)
        return inferred or None, len(inferred), max(ages) if ages else None

    def _frame_timestamp(
        self,
        requested: int | None,
        detected: int | None,
    ) -> int:
        if requested is not None:
            resolved = int(requested)
        elif detected is not None:
            resolved = int(detected)
        else:
            resolved = self._fallback_timestamp_ms + 33
        if resolved <= self._fallback_timestamp_ms:
            resolved = self._fallback_timestamp_ms + 1
        self._fallback_timestamp_ms = resolved
        return resolved

    def _fuse_keypoints(
        self,
        yolo_keypoints: Sequence[Keypoint],
        mediapipe_keypoints: Sequence[Keypoint] | None,
    ) -> list[Keypoint]:
        yolo_by_name = {point.name: point for point in yolo_keypoints}
        mediapipe_by_name = {
            point.name: point
            for point in (mediapipe_keypoints or ())
        }
        fused: list[Keypoint] = []
        for name in MEDIAPIPE_33_NAMES:
            if name in COCO_17_NAMES:
                fused.append(
                    yolo_by_name.get(name)
                    or self._missing_keypoint(name)
                )
            else:
                fused.append(
                    mediapipe_by_name.get(name)
                    or self._missing_keypoint(name)
                )
        return fused

    def _usable(self, point: Keypoint) -> bool:
        return (
            point.confidence >= self.match_confidence
            and isfinite(point.x)
            and isfinite(point.y)
        )

    def _missing_keypoint(self, name: str) -> Keypoint:
        return Keypoint(
            name=name,
            x=nan,
            y=nan,
            z=nan,
            confidence=0.0,
            source_model=self.model_name,
        )
