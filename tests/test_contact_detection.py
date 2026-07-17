from __future__ import annotations

import numpy as np
import pytest

from hyrox.base import BaseActionAnalyzer
from hyrox.contact import (
    ChestContactDetector,
    ContactResult,
    KneeContactDetector,
)


def _floor_features() -> dict[str, object]:
    return {
        "floor_reference_status": "READY",
        "floor_line_x1": 0.0,
        "floor_line_y1": 0.90,
        "floor_line_x2": 1.0,
        "floor_line_y2": 0.90,
        "body_height_reference": 0.70,
    }


def _knee_features(surface_height_ratio: float, confidence: float = 0.95) -> dict[str, object]:
    features = _floor_features()
    # With ankle_y=0.65 and a vertical shank:
    # ratio = (0.90-knee_y - 0.10*(knee_y-0.65)) / 0.70.
    knee_y = (0.965 - 0.70 * surface_height_ratio) / 1.10
    for side, x in (("left", 0.42), ("right", 0.58)):
        features.update(
            {
                f"{side}_knee_x": x,
                f"{side}_knee_y": knee_y,
                f"{side}_knee_confidence": confidence,
                f"{side}_ankle_x": x,
                f"{side}_ankle_y": 0.65,
                f"{side}_ankle_confidence": confidence,
            }
        )
    return features


def _chest_features(
    surface_height_ratio: float,
    *,
    confidence: float = 0.95,
    segmentation: bool = False,
) -> dict[str, object]:
    features = _floor_features()
    # shoulder_mid=(0.425,y), hip_mid=(0.575,y), torso=0.15,
    # so the virtual surface offset is 0.03.
    y = 0.90 - (surface_height_ratio * 0.70 + 0.03)
    features.update(
        {
            "left_shoulder_x": 0.40,
            "left_shoulder_y": y,
            "left_shoulder_confidence": confidence,
            "right_shoulder_x": 0.45,
            "right_shoulder_y": y,
            "right_shoulder_confidence": confidence,
            "left_hip_x": 0.55,
            "left_hip_y": y,
            "left_hip_confidence": confidence,
            "right_hip_x": 0.60,
            "right_hip_y": y,
            "right_hip_confidence": confidence,
        }
    )
    if segmentation:
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[89, 35:58] = 1.0
        features["_segmentation_mask"] = mask
    return features


def _repeat(
    detector: KneeContactDetector | ChestContactDetector,
    features: dict[str, object],
    *,
    phase: str,
    count: int = 3,
    start_frame: int = 1,
) -> ContactResult:
    result = ContactResult("UNSURE", 0.0, None, 0, [])
    for offset in range(count):
        frame = start_frame + offset
        result = detector.update(
            features,
            phase=phase,
            frame_index=frame,
            timestamp_ms=frame * 100,
        )
    return result


def test_knee_close_but_above_exit_is_no_contact() -> None:
    detector = KneeContactDetector()

    result = _repeat(detector, _knee_features(0.050), phase="bottom")

    assert result.status == "NO_CONTACT"
    assert result.surface_height_ratio == pytest.approx(0.050)


def test_knee_contact_requires_sustained_low_slow_evidence() -> None:
    detector = KneeContactDetector()

    result = _repeat(detector, _knee_features(0.010), phase="bottom")

    assert result.status == "CONTACT"
    assert result.confidence >= 0.72
    assert result.hold_ms == 200
    assert result.evidence_frames == [1, 2, 3]


def test_chest_proxy_distinguishes_hover_from_contact() -> None:
    hovering = ChestContactDetector()
    contact = ChestContactDetector()

    hover_result = _repeat(hovering, _chest_features(0.100), phase="chest_down")
    contact_result = _repeat(contact, _chest_features(0.005), phase="chest_down")

    assert hover_result.status == "NO_CONTACT"
    assert contact_result.status == "CONTACT"
    assert contact_result.confidence <= 0.74


def test_segmentation_overlap_can_raise_chest_proxy_confidence() -> None:
    without_mask = ChestContactDetector()
    with_mask = ChestContactDetector()

    no_mask_result = _repeat(
        without_mask,
        _chest_features(0.005),
        phase="chest_down",
    )
    mask_result = _repeat(
        with_mask,
        _chest_features(0.005, segmentation=True),
        phase="chest_down",
    )

    assert no_mask_result.confidence == pytest.approx(0.74)
    assert mask_result.status == "CONTACT"
    assert mask_result.confidence > 0.74


def test_single_frame_floor_jump_does_not_confirm_contact() -> None:
    detector = KneeContactDetector()

    first = detector.update(
        _knee_features(0.060),
        phase="descent",
        frame_index=1,
        timestamp_ms=100,
    )
    jumped = detector.update(
        _knee_features(0.005),
        phase="bottom",
        frame_index=2,
        timestamp_ms=200,
    )
    recovered = detector.update(
        _knee_features(0.060),
        phase="ascent",
        frame_index=3,
        timestamp_ms=300,
    )

    assert first.status == "NO_CONTACT"
    assert jumped.status == "UNSURE"
    assert recovered.status == "NO_CONTACT"


def test_missing_or_occluded_landmarks_are_not_observable() -> None:
    detector = KneeContactDetector()
    missing = _knee_features(0.005)
    for side in ("left", "right"):
        missing[f"{side}_knee_x"] = None

    missing_result = detector.update(
        missing,
        phase="bottom",
        frame_index=1,
        timestamp_ms=100,
    )
    occluded_result = detector.update(
        _knee_features(0.005, confidence=0.20),
        phase="bottom",
        frame_index=2,
        timestamp_ms=200,
    )

    assert missing_result.status == "NOT_OBSERVABLE"
    assert occluded_result.status == "NOT_OBSERVABLE"


def test_enter_exit_hysteresis_retains_then_releases_contact() -> None:
    detector = KneeContactDetector()
    confirmed = _repeat(detector, _knee_features(0.010), phase="bottom")

    retained = detector.update(
        _knee_features(0.025),
        phase="bottom",
        frame_index=4,
        timestamp_ms=400,
    )
    released = detector.update(
        _knee_features(0.040),
        phase="bottom",
        frame_index=5,
        timestamp_ms=500,
    )

    assert confirmed.status == "CONTACT"
    assert retained.status == "CONTACT"
    assert released.status == "NO_CONTACT"


def test_local_minimum_compensates_for_skipped_exact_contact_frame() -> None:
    detector = KneeContactDetector(sensitivity="high")
    detector.update(
        _knee_features(0.050),
        phase="descent",
        frame_index=1,
        timestamp_ms=100,
    )
    detector.update(
        _knee_features(0.010),
        phase="bottom",
        frame_index=2,
        timestamp_ms=200,
    )

    result = detector.update(
        _knee_features(0.016),
        phase="bottom",
        frame_index=3,
        timestamp_ms=300,
    )

    assert result.status == "CONTACT"
    assert 2 in result.evidence_frames


def test_unready_floor_is_unsure_and_phase_gates_contact() -> None:
    no_floor = _knee_features(0.005)
    no_floor["floor_reference_status"] = "UNSURE"
    detector = KneeContactDetector()

    floor_result = detector.update(
        no_floor,
        phase="bottom",
        frame_index=1,
        timestamp_ms=100,
    )
    standing_result = _repeat(
        KneeContactDetector(),
        _knee_features(0.005),
        phase="stand",
    )

    assert floor_result.status == "UNSURE"
    assert standing_result.status == "UNSURE"


def test_analyzer_exposes_lightweight_contacts_without_buffering_mask() -> None:
    analyzer = BaseActionAnalyzer(action="test")
    features = _chest_features(0.005, segmentation=True)
    features["visible_score"] = 0.95

    state = analyzer.update(features, timestamp_ms=100)

    assert "contacts" in state["debug"]
    assert state["debug"]["contacts"]["chest_proxy"]["status"] in {
        "UNSURE",
        "CONTACT",
    }
    assert features["chest_contact_proxy_status"] in {"UNSURE", "CONTACT"}
    assert "_segmentation_mask" not in analyzer._candidate_frames[-1]
