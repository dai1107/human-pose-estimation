from __future__ import annotations

import math

import numpy as np
import pytest

from src.backends.base import Keypoint, PoseResult
from src.backends.yolo_guided_mediapipe_backend import (
    IDENTITY_MATCH_NAMES,
    YoloGuidedMediaPipeBackend,
)
from src.utils.keypoint_schema import COCO_17_NAMES, MEDIAPIPE_33_NAMES


class FakeBackend:
    def __init__(self, result: PoseResult) -> None:
        self.result = result
        self.detect_calls = 0
        self.closed = False

    def detect(self, frame: np.ndarray, timestamp_ms: int | None = None) -> PoseResult:
        del frame, timestamp_ms
        self.detect_calls += 1
        return self.result

    def close(self) -> None:
        self.closed = True


def _keypoints(names: tuple[str, ...], x: float, y: float, source: str) -> list[Keypoint]:
    return [
        Keypoint(
            name=name,
            x=x,
            y=y,
            confidence=0.95,
            source_model=source,
        )
        for name in names
    ]


def _result(
    keypoints: list[Keypoint],
    *,
    model_name: str,
    success: bool = True,
    extra: dict[str, object] | None = None,
) -> PoseResult:
    return PoseResult(
        keypoints=keypoints,
        connections=(),
        model_name=model_name,
        num_keypoints=len(keypoints),
        success=success,
        inference_time_ms=4.0,
        bbox=(0.1, 0.1, 0.9, 0.9) if success else None,
        timestamp_ms=33,
        extra=extra or {},
    )


def _candidate(x: float, y: float, *, foot_x: float) -> list[Keypoint]:
    points = _keypoints(MEDIAPIPE_33_NAMES, x, y, "mediapipe")
    by_name = {point.name: point for point in points}
    for name in ("left_heel", "right_heel", "left_foot_index", "right_foot_index"):
        by_name[name] = Keypoint(
            name,
            foot_x,
            y + 0.2,
            confidence=0.9,
            source_model="mediapipe",
        )
    return [by_name[name] for name in MEDIAPIPE_33_NAMES]


def test_selects_matching_person_and_only_uses_mediapipe_for_supplemental_points() -> None:
    yolo_points = _keypoints(COCO_17_NAMES, 0.50, 0.50, "yolo-pose")
    background = _candidate(0.85, 0.15, foot_x=0.91)
    athlete = _candidate(0.52, 0.51, foot_x=0.42)
    yolo = FakeBackend(_result(yolo_points, model_name="yolo-pose"))
    mediapipe = FakeBackend(
        _result(
            background,
            model_name="mediapipe",
            extra={"pose_candidates": [background, athlete]},
        )
    )
    backend = YoloGuidedMediaPipeBackend(
        yolo_backend=yolo,
        mediapipe_backend=mediapipe,
    )

    result = backend.detect(np.zeros((32, 32, 3), dtype=np.uint8), timestamp_ms=33)
    points = {point.name: point for point in result.keypoints}

    assert result.success
    assert result.model_name == "yolo-guided-mediapipe"
    assert len(result.keypoints) == 33
    assert points["left_knee"].x == pytest.approx(0.50)
    assert points["left_knee"].source_model == "yolo-pose"
    assert points["left_heel"].x == pytest.approx(0.42)
    assert points["left_foot_index"].source_model == "mediapipe"
    assert result.extra["identity_matched"] is True
    assert result.extra["identity_match_points"] == len(IDENTITY_MATCH_NAMES)
    assert result.extra["mediapipe_candidate_count"] == 2
    assert result.extra["mediapipe_supplemental_available"] is True


def test_rejects_background_candidate_and_marks_supplemental_points_missing() -> None:
    yolo_points = _keypoints(COCO_17_NAMES, 0.10, 0.80, "yolo-pose")
    background = _candidate(0.80, 0.10, foot_x=0.91)
    yolo = FakeBackend(_result(yolo_points, model_name="yolo-pose"))
    mediapipe = FakeBackend(
        _result(
            background,
            model_name="mediapipe",
            extra={"pose_candidates": [background]},
        )
    )
    backend = YoloGuidedMediaPipeBackend(
        yolo_backend=yolo,
        mediapipe_backend=mediapipe,
    )

    result = backend.detect(np.zeros((32, 32, 3), dtype=np.uint8))
    points = {point.name: point for point in result.keypoints}

    assert result.success
    assert result.extra["identity_matched"] is False
    assert result.extra["identity_match_distance"] > 0.20
    assert result.extra["mediapipe_supplemental_available"] is False
    assert points["left_knee"].confidence > 0.0
    assert points["left_heel"].confidence == 0.0
    assert math.isnan(points["left_heel"].x)


def test_does_not_run_or_return_mediapipe_when_yolo_has_no_target() -> None:
    yolo = FakeBackend(
        _result([], model_name="yolo-pose", success=False)
    )
    background = _candidate(0.50, 0.50, foot_x=0.50)
    mediapipe = FakeBackend(
        _result(
            background,
            model_name="mediapipe",
            extra={"pose_candidates": [background]},
        )
    )
    backend = YoloGuidedMediaPipeBackend(
        yolo_backend=yolo,
        mediapipe_backend=mediapipe,
    )

    result = backend.detect(np.zeros((32, 32, 3), dtype=np.uint8))

    assert not result.success
    assert result.keypoints == []
    assert mediapipe.detect_calls == 0
    assert result.extra["identity_matched"] is False


def test_temporal_feet_follow_current_yolo_ankles_and_expire() -> None:
    yolo = FakeBackend(
        _result(
            _keypoints(COCO_17_NAMES, 0.50, 0.50, "yolo-pose"),
            model_name="yolo-pose",
        )
    )
    athlete = _candidate(0.51, 0.51, foot_x=0.42)
    mediapipe = FakeBackend(
        _result(
            athlete,
            model_name="mediapipe",
            extra={"pose_candidates": [athlete]},
        )
    )
    backend = YoloGuidedMediaPipeBackend(
        yolo_backend=yolo,
        mediapipe_backend=mediapipe,
    )
    backend.detect(np.zeros((32, 32, 3), dtype=np.uint8), timestamp_ms=33)

    yolo.result = _result(
        _keypoints(COCO_17_NAMES, 0.60, 0.50, "yolo-pose"),
        model_name="yolo-pose",
    )
    background = _candidate(0.10, 0.10, foot_x=0.05)
    mediapipe.result = _result(
        background,
        model_name="mediapipe",
        extra={"pose_candidates": [background]},
    )
    held = backend.detect(
        np.zeros((32, 32, 3), dtype=np.uint8),
        timestamp_ms=1033,
    )
    held_points = {point.name: point for point in held.keypoints}

    assert held.extra["identity_matched"] is False
    assert held.extra["mediapipe_temporal_foot_points"] == 4
    assert held.extra["mediapipe_temporal_foot_age_ms"] == 1000
    assert held_points["left_heel"].x == pytest.approx(0.52)
    assert held_points["left_heel"].confidence >= 0.60
    assert held_points["left_heel"].source_model == "mediapipe-from-yolo-ankle"

    expired = backend.detect(
        np.zeros((32, 32, 3), dtype=np.uint8),
        timestamp_ms=1800,
    )
    expired_points = {point.name: point for point in expired.keypoints}

    assert expired.extra["mediapipe_temporal_foot_points"] == 0
    assert expired_points["left_heel"].confidence == 0.0


def test_close_releases_both_backends() -> None:
    yolo = FakeBackend(_result([], model_name="yolo-pose", success=False))
    mediapipe = FakeBackend(_result([], model_name="mediapipe", success=False))
    backend = YoloGuidedMediaPipeBackend(
        yolo_backend=yolo,
        mediapipe_backend=mediapipe,
    )

    backend.close()

    assert yolo.closed
    assert mediapipe.closed
