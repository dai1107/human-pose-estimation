from __future__ import annotations

from types import SimpleNamespace

from src.biomechanics.hand_landmarks import (
    HAND_LANDMARK_INDEX,
    SUPPLEMENTAL_FINGER_CONNECTIONS,
    SUPPLEMENTAL_FINGER_DISPLAY_INDICES,
    SUPPLEMENTAL_FINGER_JOINTS,
    coerce_hand_landmarks,
)
from src.realtime_pose import LandmarkSmoother, infer_hand_side


def test_hand_landmarks_accept_none_visibility_and_presence() -> None:
    raw_point = SimpleNamespace(x=0.4, y=0.3, z=None, visibility=None, presence=None)

    hand_points = coerce_hand_landmarks([raw_point], expected_count=1)

    assert hand_points[0].x == 0.4
    assert hand_points[0].y == 0.3
    assert hand_points[0].z == 0.0
    assert hand_points[0].visibility == 1.0
    assert hand_points[0].presence == 1.0


def test_landmark_smoother_accepts_hand_landmarker_none_fields() -> None:
    raw_point = SimpleNamespace(x=0.4, y=0.3, z=None, visibility=None, presence=None)

    smoothed = LandmarkSmoother(alpha=0.65).smooth([raw_point], timestamp_ms=1000)

    assert smoothed[0].x == 0.4
    assert smoothed[0].y == 0.3
    assert smoothed[0].z == 0.0
    assert smoothed[0].visibility == 1.0
    assert smoothed[0].presence == 1.0


def test_hand_side_inference_ignores_missing_x_values() -> None:
    landmarks = [
        SimpleNamespace(x=None, y=0.3, z=0.0, visibility=None, presence=None),
        SimpleNamespace(x=0.3, y=0.4, z=0.0, visibility=None, presence=None),
    ]

    assert infer_hand_side(landmarks, fallback_index=1) == "left"


def test_five_finger_display_includes_tips_and_excludes_wrist() -> None:
    names = {name for name, _ in SUPPLEMENTAL_FINGER_JOINTS}

    assert len(SUPPLEMENTAL_FINGER_JOINTS) == 20
    assert len(SUPPLEMENTAL_FINGER_DISPLAY_INDICES) == 20
    assert len(SUPPLEMENTAL_FINGER_CONNECTIONS) == 15
    assert "wrist" not in names
    assert "index_finger_tip" in names
    assert "middle_finger_tip" in names
    assert "ring_finger_tip" in names
    assert HAND_LANDMARK_INDEX["wrist"] not in SUPPLEMENTAL_FINGER_DISPLAY_INDICES
