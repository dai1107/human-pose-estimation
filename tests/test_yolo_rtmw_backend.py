from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.backends.base import Keypoint, PoseResult
from src.backends.yolo_rtmw_backend import YoloRtmwWholeBodyBackend
from src.utils.keypoint_schema import COCO_17_NAMES


class FakeSession:
    def __init__(self) -> None:
        self.run_calls = 0

    def get_inputs(self) -> list[Any]:
        return [SimpleNamespace(name="input")]

    def get_providers(self) -> list[str]:
        return ["injected"]

    def run(self, *_args: Any, **_kwargs: Any) -> list[np.ndarray]:
        self.run_calls += 1
        raise AssertionError("tests inject decoded RTMW coordinates")


class FakeYoloBackend:
    def __init__(self, result: PoseResult) -> None:
        self.result = result
        self.detect_calls = 0
        self.closed = False

    def detect(
        self,
        _frame: np.ndarray,
        timestamp_ms: int | None = None,
    ) -> PoseResult:
        self.detect_calls += 1
        return replace(self.result, timestamp_ms=timestamp_ms)

    def close(self) -> None:
        self.closed = True


def _yolo_result(
    *,
    x: float = 0.5,
    y: float = 0.5,
    success: bool = True,
) -> PoseResult:
    points = (
        [
            Keypoint(
                name,
                x,
                y,
                confidence=0.95,
                source_model="yolo-pose",
            )
            for name in COCO_17_NAMES
        ]
        if success
        else []
    )
    return PoseResult(
        keypoints=points,
        connections=(),
        model_name="yolo-pose",
        num_keypoints=len(points),
        success=success,
        inference_time_ms=5.0,
        bbox=(0.1, 0.1, 0.9, 0.9) if success else None,
        extra={"bbox_pixels": (10.0, 10.0, 90.0, 90.0)} if success else {},
    )


def _wholebody_output(
    *,
    body_x: float = 50.0,
    body_y: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.full((133, 2), (body_x, body_y), dtype=np.float32)
    scores = np.full(133, 0.95, dtype=np.float32)
    coordinates[17] = (20.0, 82.0)
    coordinates[18] = (40.0, 82.0)
    coordinates[19] = (25.0, 78.0)
    coordinates[20] = (60.0, 82.0)
    coordinates[21] = (80.0, 82.0)
    coordinates[22] = (75.0, 78.0)
    for start in (91, 112):
        for hand_index in range(21):
            coordinates[start + hand_index] = (
                48.0 + (hand_index % 5),
                48.0 + (hand_index // 5),
            )
    return coordinates, scores


def _backend(yolo_result: PoseResult) -> tuple[YoloRtmwWholeBodyBackend, FakeSession]:
    session = FakeSession()
    backend = YoloRtmwWholeBodyBackend(
        yolo_backend=FakeYoloBackend(yolo_result),
        session=session,
    )
    return backend, session


def test_maps_matched_wholebody_feet_and_both_21_point_hands() -> None:
    backend, _ = _backend(_yolo_result())
    coordinates, scores = _wholebody_output()
    coordinates[13] = (60.0, 50.0)
    backend._infer = lambda _frame, _bbox: (coordinates, scores)  # type: ignore[method-assign]

    result = backend.detect(np.zeros((100, 100, 3), dtype=np.uint8), timestamp_ms=7)

    points = {point.name: point for point in result.keypoints}
    assert result.success is True
    assert result.model_name == "yolo-rtmw-wholebody"
    assert result.extra["identity_matched"] is True
    assert result.extra["identity_match_points"] == 13
    assert len(result.keypoints) == 33
    assert points["left_heel"].x == 0.25
    assert points["left_foot_index"].x == pytest.approx(0.30)
    assert points["right_heel"].x == 0.75
    assert points["right_foot_index"].x == pytest.approx(0.70)
    assert points["left_heel"].source_model == "rtmw-wholebody"
    assert points["left_knee"].x == pytest.approx(0.50)
    assert points["left_knee"].source_model == "yolo-pose"
    assert len(result.extra["rtmw_hand_keypoints"]["left"]) == 21
    assert len(result.extra["rtmw_hand_keypoints"]["right"]) == 21
    assert result.extra["rtmw_rejected_hands"] == {}


def test_rejects_identity_mismatch_and_returns_only_yolo_core_points() -> None:
    backend, _ = _backend(_yolo_result(x=0.1, y=0.1))
    coordinates, scores = _wholebody_output(body_x=90.0, body_y=90.0)
    backend._infer = lambda _frame, _bbox: (coordinates, scores)  # type: ignore[method-assign]

    result = backend.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    points = {point.name: point for point in result.keypoints}
    assert result.success is True
    assert result.extra["identity_matched"] is False
    assert result.extra["rtmw_wholebody_available"] is False
    assert result.extra["identity_match_distance"] > 0.20
    assert points["left_knee"].source_model == "yolo-pose"
    assert points["left_heel"].confidence == 0.0
    assert result.extra["rtmw_hand_keypoints"] == {}


def test_does_not_run_rtmw_without_a_yolo_target() -> None:
    backend, session = _backend(_yolo_result(success=False))

    result = backend.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert result.success is False
    assert result.keypoints == []
    assert result.extra["rtmw_wholebody_available"] is False
    assert session.run_calls == 0


def test_hides_rtmw_fingers_when_the_hand_is_occluded_inside_the_torso() -> None:
    yolo_result = _yolo_result()
    layout = {
        "left_shoulder": (0.40, 0.30),
        "right_shoulder": (0.60, 0.30),
        "left_hip": (0.43, 0.65),
        "right_hip": (0.57, 0.65),
        "left_wrist": (0.50, 0.42),
        "right_wrist": (0.75, 0.42),
    }
    yolo_points = [
        replace(
            point,
            x=layout.get(point.name, (point.x, point.y))[0],
            y=layout.get(point.name, (point.x, point.y))[1],
        )
        for point in yolo_result.keypoints
    ]
    yolo_result = replace(yolo_result, keypoints=yolo_points)
    backend, _ = _backend(yolo_result)
    coordinates, scores = _wholebody_output()
    for index, point in enumerate(yolo_points):
        coordinates[index] = (point.x * 100.0, point.y * 100.0)
    for hand_index in range(21):
        coordinates[91 + hand_index] = (
            48.0 + hand_index % 5,
            40.0 + hand_index // 5,
        )
        coordinates[112 + hand_index] = (
            73.0 + hand_index % 5,
            40.0 + hand_index // 5,
        )
    backend._infer = lambda _frame, _bbox: (coordinates, scores)  # type: ignore[method-assign]

    result = backend.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    points = {point.name: point for point in result.keypoints}
    assert set(result.extra["rtmw_hand_keypoints"]) == {"right"}
    assert result.extra["rtmw_rejected_hands"]["left"] == "torso_occlusion"
    assert points["left_thumb"].confidence == 0.0
    assert points["left_index"].confidence == 0.0
    assert points["left_pinky"].confidence == 0.0
    assert points["right_index"].confidence > 0.0


def test_closes_the_yolo_target_tracker() -> None:
    backend, _ = _backend(_yolo_result())
    yolo = backend.yolo_backend

    backend.close()

    assert yolo.closed is True
