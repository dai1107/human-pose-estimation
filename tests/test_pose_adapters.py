from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.backends.base import Keypoint as BackendKeypoint
from src.backends.base import PoseResult
from src.pose.adapters import (
    format_normalized_pose_debug,
    mediapipe_to_normalized_pose,
    normalize_backend_pose_result,
    yolopose_to_normalized_pose,
)
from src.pose.keypoints import COMMON_KEYPOINTS
from src.pose.schema import Keypoint, NormalizedPose


def _mediapipe_landmarks() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            x=0.1 + index * 0.01,
            y=0.2 + index * 0.005,
            z=-0.01 * index,
            visibility=0.9,
            presence=0.8,
        )
        for index in range(33)
    ]


def test_mediapipe_adapter_converts_tasks_result_to_pixels() -> None:
    raw_result = SimpleNamespace(pose_landmarks=[_mediapipe_landmarks()])

    pose = mediapipe_to_normalized_pose(
        raw_result,
        image_width=640,
        image_height=480,
        timestamp_ms=123,
        frame_id=7,
        latency_ms=11.5,
    )

    assert pose is not None
    assert pose.source == "mediapipe"
    assert pose.frame_id == 7
    assert pose.timestamp_ms == 123
    assert pose.latency_ms == 11.5
    assert len(pose.keypoints) == 17
    assert tuple(pose.keypoints) == COMMON_KEYPOINTS
    assert pose.keypoints["left_shoulder"].x == pytest.approx((0.1 + 11 * 0.01) * 640)
    assert pose.keypoints["left_shoulder"].y == pytest.approx((0.2 + 11 * 0.005) * 480)
    assert pose.keypoints["left_shoulder"].confidence == pytest.approx(0.8)
    assert pose.bbox is not None
    assert 0.0 <= pose.overall_confidence <= 1.0


def test_mediapipe_adapter_supports_solutions_landmark_container() -> None:
    raw_result = SimpleNamespace(
        pose_landmarks=SimpleNamespace(landmark=_mediapipe_landmarks())
    )

    pose = mediapipe_to_normalized_pose(raw_result, 320, 240, 50)

    assert pose is not None
    assert len(pose.keypoints) == 17


def test_backend_pose_result_can_be_normalized_without_raw_sdk_object() -> None:
    result = PoseResult(
        keypoints=[
            BackendKeypoint(name, 0.5, 0.25, z=0.1, confidence=0.75, source_model="mediapipe")
            for name in COMMON_KEYPOINTS
        ],
        connections=(),
        model_name="mediapipe",
        num_keypoints=17,
        success=True,
        inference_time_ms=6.25,
        timestamp_ms=900,
    )

    pose = normalize_backend_pose_result(result, 800, 600, 900, frame_id=42)

    assert pose is not None
    assert pose.get("right_knee") is not None
    assert pose.get("right_knee").x == 400.0
    assert pose.get("right_knee").y == 150.0
    assert pose.latency_ms == 6.25


def test_mediapipe_fusion_result_uses_full_frame_restored_keypoints() -> None:
    result = PoseResult(
        keypoints=[
            BackendKeypoint(name, 0.5, 0.25, z=0.1, confidence=0.75, source_model="mediapipe")
            for name in COMMON_KEYPOINTS
        ],
        connections=(),
        model_name="mediapipe",
        num_keypoints=17,
        success=True,
        inference_time_ms=6.0,
        timestamp_ms=100,
        extra={
            "raw_result": SimpleNamespace(pose_landmarks=[_mediapipe_landmarks()]),
            "roi_bbox_pixels": (200.0, 100.0, 600.0, 500.0),
        },
    )

    pose = normalize_backend_pose_result(result, 800, 600, 100, frame_id=4)

    assert pose is not None
    assert pose.keypoints["nose"].x == 400.0
    assert pose.keypoints["nose"].y == 150.0


def test_yolo_backend_pose_result_preserves_model_bbox_as_pixels() -> None:
    result = PoseResult(
        keypoints=[
            BackendKeypoint(name, 0.5, 0.25, confidence=0.75, source_model="yolo-pose")
            for name in COMMON_KEYPOINTS
        ],
        connections=(),
        model_name="yolo-pose",
        num_keypoints=17,
        success=True,
        inference_time_ms=8.0,
        bbox=(0.1, 0.2, 0.9, 0.8),
        timestamp_ms=100,
    )

    pose = normalize_backend_pose_result(result, 800, 600, 100, frame_id=3)

    assert pose is not None
    assert pose.source == "yolopose"
    assert pose.bbox == pytest.approx((80.0, 120.0, 720.0, 480.0))


def _yolo_raw_result(*, include_boxes: bool = True) -> SimpleNamespace:
    xy = np.zeros((2, 17, 2), dtype=float)
    xy[0, :, 0] = np.arange(17) + 20.0
    xy[0, :, 1] = np.arange(17) + 40.0
    xy[1, :, 0] = np.arange(17) + 100.0
    xy[1, :, 1] = np.arange(17) + 120.0
    scores = np.vstack((np.full(17, 0.95), np.full(17, 0.65)))
    boxes = None
    if include_boxes:
        boxes = SimpleNamespace(
            xyxy=np.array([[10.0, 20.0, 80.0, 180.0], [50.0, 60.0, 500.0, 440.0]])
        )
    return SimpleNamespace(
        keypoints=SimpleNamespace(xy=xy, conf=scores),
        boxes=boxes,
    )


def test_yolopose_adapter_selects_largest_bbox_and_uses_scores() -> None:
    pose = yolopose_to_normalized_pose(
        [_yolo_raw_result()],
        image_width=640,
        image_height=480,
        timestamp_ms=333,
        frame_id=10,
        latency_ms=18.0,
    )

    assert pose is not None
    assert pose.source == "yolopose"
    assert len(pose.keypoints) == 17
    assert pose.keypoints["right_knee"].confidence == pytest.approx(0.65)
    assert pose.keypoints["nose"].x == 100.0
    assert pose.bbox == pytest.approx((50.0, 60.0, 500.0, 440.0))


def test_yolopose_adapter_uses_confidence_when_boxes_are_missing() -> None:
    pose = yolopose_to_normalized_pose(
        [_yolo_raw_result(include_boxes=False)],
        image_width=640,
        image_height=480,
        timestamp_ms=100,
    )

    assert pose is not None
    assert pose.keypoints["nose"].x == 20.0
    assert pose.overall_confidence == pytest.approx(0.95)
    assert pose.bbox is not None


def test_yolopose_adapter_converts_explicit_normalized_coordinates() -> None:
    xyn = np.full((1, 17, 2), (0.5, 0.25), dtype=float)
    result = SimpleNamespace(
        keypoints=SimpleNamespace(xyn=xyn, conf=np.full((1, 17), 0.7)),
        boxes=SimpleNamespace(xyxyn=np.array([[0.1, 0.2, 0.9, 0.8]])),
    )

    pose = yolopose_to_normalized_pose(result, 800, 600, 100)

    assert pose is not None
    assert pose.keypoints["left_ankle"].x == 400.0
    assert pose.keypoints["left_ankle"].y == 150.0
    assert pose.bbox == pytest.approx((80.0, 120.0, 720.0, 480.0))


def test_yolopose_adapter_supports_plain_array_mock_format() -> None:
    result = {
        "keypoints": [[100.0 + index, 50.0 + index] for index in range(17)],
        "scores": [0.8] * 17,
        "bbox": [80.0, 40.0, 300.0, 500.0],
    }

    pose = yolopose_to_normalized_pose(result, 640, 540, 200)

    assert pose is not None
    assert pose.keypoints["nose"].x == 100.0
    assert pose.keypoints["right_knee"].confidence == 0.8
    assert pose.bbox == pytest.approx((80.0, 40.0, 300.0, 500.0))


def test_adapters_return_none_without_a_person() -> None:
    assert mediapipe_to_normalized_pose(SimpleNamespace(pose_landmarks=[]), 640, 480, 0) is None
    assert yolopose_to_normalized_pose([], 640, 480, 0) is None
    zero_confidence = SimpleNamespace(
        keypoints=SimpleNamespace(
            xy=np.zeros((1, 17, 2), dtype=float),
            conf=np.zeros((1, 17), dtype=float),
        ),
        boxes=None,
    )
    assert yolopose_to_normalized_pose(zero_confidence, 640, 480, 0) is None


def test_invalid_points_are_skipped_without_crashing() -> None:
    landmarks = _mediapipe_landmarks()
    landmarks[11] = SimpleNamespace(x=9.0, y=9.0, visibility=1.0)

    pose = mediapipe_to_normalized_pose(SimpleNamespace(pose_landmarks=[landmarks]), 640, 480, 0)

    assert pose is not None
    assert "left_shoulder" not in pose.keypoints
    assert len(pose.keypoints) == 16


def test_normalized_pose_schema_serializes_metadata_and_keypoints() -> None:
    point = Keypoint("nose", 10.0, 20.0, None, 0.9, 0.8)
    pose = NormalizedPose(
        source="yolopose",
        frame_id=2,
        timestamp_ms=66,
        latency_ms=7.5,
        image_width=100,
        image_height=80,
        keypoints={"nose": point},
        bbox=(5.0, 6.0, 15.0, 25.0),
        overall_confidence=0.8,
    )

    payload = pose.to_dict()

    assert payload["frame_id"] == 2
    assert payload["latency_ms"] == 7.5
    assert payload["keypoints"]["nose"]["z"] is None
    assert "source=yolopose" in format_normalized_pose_debug(pose)


def test_normalized_pose_rejects_invalid_source() -> None:
    with pytest.raises(ValueError, match="unsupported pose source"):
        NormalizedPose(
            source="unknown",  # type: ignore[arg-type]
            frame_id=0,
            timestamp_ms=0,
            latency_ms=0.0,
            image_width=10,
            image_height=10,
            keypoints={},
            bbox=None,
            overall_confidence=0.0,
        )


def test_keypoint_schema_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        Keypoint("nose", 1.0, 2.0, None, 1.0, 1.5)
