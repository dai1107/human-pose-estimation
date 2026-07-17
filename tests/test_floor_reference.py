from __future__ import annotations

import pytest

from hyrox.actions import LungeAnalyzer
from hyrox.features import extract_basic_pose_features
from hyrox.floor_reference import (
    FloorLine,
    LocalFloorReference,
    normalized_height_to_floor,
    signed_distance_to_floor,
)
from hyrox.geometry import PosePoint
from hyrox.landmark_names import (
    LEFT_ANKLE,
    LEFT_FOOT_INDEX,
    LEFT_HEEL,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    RIGHT_ANKLE,
    RIGHT_FOOT_INDEX,
    RIGHT_HEEL,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
)


def _standing_features(floor_y: float = 0.90) -> dict[str, object]:
    return {
        "min_knee_angle": 178.0,
        "min_hip_angle": 176.0,
        "body_height_norm": 0.72,
        "skeleton_height_estimate_norm": 0.70,
        "lower_body_visible_score": 0.95,
        "hip_center_x": 0.50,
        "hip_center_y": 0.45,
        "knee_center_x": 0.50,
        "knee_center_y": 0.68,
        "left_heel_x": 0.42,
        "left_heel_y": floor_y,
        "left_heel_confidence": 0.95,
        "right_heel_x": 0.58,
        "right_heel_y": floor_y,
        "right_heel_confidence": 0.95,
        "left_foot_index_x": 0.45,
        "left_foot_index_y": floor_y - 0.002,
        "left_foot_index_confidence": 0.94,
        "right_foot_index_x": 0.55,
        "right_foot_index_y": floor_y - 0.002,
        "right_foot_index_confidence": 0.94,
    }


def _standing_landmarks() -> list[dict[str, float]]:
    landmarks = [
        {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0, "presence": 0.0}
        for _ in range(33)
    ]

    def put(index: int, x: float, y: float) -> None:
        landmarks[index] = {
            "x": x,
            "y": y,
            "z": 0.0,
            "visibility": 0.95,
            "presence": 1.0,
        }

    for index, x, y in (
        (LEFT_SHOULDER, 0.40, 0.20),
        (RIGHT_SHOULDER, 0.60, 0.20),
        (LEFT_HIP, 0.40, 0.50),
        (RIGHT_HIP, 0.60, 0.50),
        (LEFT_KNEE, 0.40, 0.75),
        (RIGHT_KNEE, 0.60, 0.75),
        (LEFT_ANKLE, 0.40, 0.95),
        (RIGHT_ANKLE, 0.60, 0.95),
        (LEFT_HEEL, 0.39, 0.98),
        (RIGHT_HEEL, 0.61, 0.98),
        (LEFT_FOOT_INDEX, 0.43, 0.98),
        (RIGHT_FOOT_INDEX, 0.57, 0.98),
    ):
        put(index, x, y)
    return landmarks


def test_signed_and_body_normalized_height_support_a_tilted_floor() -> None:
    line = FloorLine(PosePoint(0.0, 0.80), PosePoint(1.0, 1.00))
    point = PosePoint(0.50, 0.70)

    distance = signed_distance_to_floor(point, line)

    assert distance == pytest.approx(0.20 / (1.0**2 + 0.2**2) ** 0.5)
    assert normalized_height_to_floor(point, 0.50, line) == pytest.approx(distance / 0.50)
    assert signed_distance_to_floor(None, line) is None
    assert normalized_height_to_floor(point, None, line) is None


def test_auto_floor_uses_stable_standing_foot_median_and_calibrates_height() -> None:
    estimator = LocalFloorReference(min_samples=5)
    result = None
    for frame, floor_y in enumerate((0.900, 0.902, 0.899, 0.901, 0.900), start=1):
        result = estimator.update(
            _standing_features(floor_y),
            timestamp_ms=frame * 100,
            frame_index=frame,
        )

    assert result is not None
    assert result.status == "READY"
    assert result.source == "auto"
    assert result.line is not None
    assert result.line.y_at(0.5) == pytest.approx(0.900)
    assert result.body_height == pytest.approx(0.72)
    assert result.body_height_source == "standing_calibration"
    assert result.confidence >= 0.80


def test_manual_two_point_floor_overrides_auto_floor() -> None:
    estimator = LocalFloorReference()
    estimator.set_manual_line((0.1, 0.85), (0.9, 0.93))

    features = _standing_features(0.89)
    result = estimator.enrich_features(features, timestamp_ms=100, frame_index=1)

    assert result.status == "READY"
    assert result.source == "manual"
    assert features["floor_line_y1"] == pytest.approx(0.85)
    assert features["floor_line_y2"] == pytest.approx(0.93)
    assert features["hip_center_height_to_floor"] is not None


def test_body_height_priority_uses_current_box_then_skeleton_fallback() -> None:
    estimator = LocalFloorReference()
    estimator.set_manual_line((0.1, 0.88), (0.9, 0.90))
    current = _standing_features(0.89)
    current["body_box_height_norm"] = 0.84

    current_result = estimator.update(current, timestamp_ms=100, frame_index=1)

    assert current_result.body_height == pytest.approx(0.84)
    assert current_result.body_height_source == "current_body_box"

    fallback_estimator = LocalFloorReference()
    fallback_estimator.set_manual_line((0.1, 0.88), (0.9, 0.90))
    fallback = _standing_features(0.89)
    fallback["body_height_norm"] = None
    fallback["body_box_height_norm"] = None
    fallback_result = fallback_estimator.update(fallback, timestamp_ms=100, frame_index=1)

    assert fallback_result.body_height == pytest.approx(0.70)
    assert fallback_result.body_height_source == "skeleton_segments"


def test_long_missing_feet_and_camera_shift_return_unsure() -> None:
    estimator = LocalFloorReference(min_samples=5)
    for frame in range(1, 6):
        estimator.update(
            _standing_features(0.90),
            timestamp_ms=frame * 100,
            frame_index=frame,
        )

    missing = _standing_features(0.90)
    for name in ("left_heel", "right_heel", "left_foot_index", "right_foot_index"):
        missing[f"{name}_confidence"] = None
    missing_result = estimator.update(missing, timestamp_ms=1300, frame_index=6)

    assert missing_result.status == "UNSURE"
    assert missing_result.reason_code == "FOOT_LANDMARKS_MISSING"

    moved = LocalFloorReference(min_samples=5)
    for frame in range(1, 6):
        moved.update(_standing_features(0.90), timestamp_ms=frame * 100, frame_index=frame)
    moved_result = None
    for frame in range(6, 13):
        moved_result = moved.update(
            _standing_features(0.80),
            timestamp_ms=frame * 100,
            frame_index=frame,
        )

    assert moved_result is not None
    assert moved_result.status == "UNSURE"
    assert moved_result.reason_code in {"CAMERA_MOVED", "FLOOR_NOT_CALIBRATED"}


def test_analyzer_enriches_features_with_shared_floor_reference() -> None:
    analyzer = LungeAnalyzer.from_config({"stable_frames": 1})
    state = None
    features = None
    for frame in range(1, 6):
        features = extract_basic_pose_features(
            _standing_landmarks(),
            image_width=640,
            image_height=480,
        )
        state = analyzer.update(features, timestamp_ms=frame * 100)

    assert state is not None
    assert features is not None
    assert features["floor_reference_status"] == "READY"
    assert features["floor_y"] == pytest.approx(0.98)
    assert features["body_height_reference"] == pytest.approx(0.75)
    assert features["hip_center_height_to_floor"] == pytest.approx(0.48)
    assert state["debug"]["floor_reference"]["status"] == "READY"
