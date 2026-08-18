from __future__ import annotations

from src.biomechanics.joint_metrics import JointMetric
from src.biomechanics.pose_reliability import (
    REASON_CAMERA_MOVING,
    REASON_OCCLUDED,
    build_pose_reliability,
    select_metric_candidates,
)
from src.realtime.validation import ValidationBudgetGate, realtime_layer_contract


def _metric(
    *,
    confidence: float = 0.9,
    reliable: bool = True,
    observable: bool = True,
    reasons: tuple[str, ...] = (),
) -> JointMetric:
    return JointMetric(
        raw_2d=121.0,
        smooth_2d=120.0,
        raw_3d=111.0,
        smooth_3d=110.0,
        selected_value=110.0 if reliable else (120.0 if observable else None),
        source="3D" if reliable else ("2D" if observable else "UNAVAILABLE"),
        confidence=confidence,
        observable=observable,
        three_d_reliable=reliable,
        quality_reasons=reasons,
        legacy_selected_source="2d_shadow",
    )


def _reliability(metric: JointMetric, *, camera_moving: bool = False):
    return build_pose_reliability(
        {"left_knee_angle": metric},
        body_coordinate_system={"confidence": 0.9, "reliable": True},
        ground_estimation={"status": "READY", "ground_confidence": 0.9},
        foot_contact_evidence={
            "left": {"foot_contact_confidence": 0.8},
            "right": {"foot_contact_confidence": 0.7},
        },
        camera_motion={
            "state": "camera_unstable" if camera_moving else "camera_static",
            "camera_motion_score": 0.8 if camera_moving else 0.0,
        },
        reliability={"global_position_reliability": 0.2 if camera_moving else 1.0},
    )


def test_pose_reliability_normalizes_reasons_and_keeps_camera_global() -> None:
    pose = _reliability(
        _metric(reasons=("low_visibility", "two_d_three_d_conflict")),
        camera_moving=True,
    )

    assert REASON_OCCLUDED in pose.joint_reasons["left_knee"]
    assert REASON_CAMERA_MOVING in pose.reasons
    assert REASON_CAMERA_MOVING not in pose.joint_reasons["left_knee"]
    assert pose.joint_confidence["left_knee"] > 0.0
    assert all(reason.isupper() for reason in pose.reasons)


def test_metric_candidates_fall_back_without_changing_formal_2d() -> None:
    reliable = _metric()
    pose = _reliability(reliable)
    candidate = select_metric_candidates(
        {"left_knee_angle": reliable}, {}, pose
    )["left_knee_angle"]
    assert (candidate["selected_source"], candidate["selected_value"]) == ("3D", 110.0)
    assert (candidate["formal_source"], candidate["formal_value"]) == ("2d_shadow", 120.0)
    assert candidate["formal_threshold_replacement_allowed"] is False

    fallback = _metric(reliable=False)
    candidate = select_metric_candidates(
        {"left_knee_angle": fallback}, {}, _reliability(fallback)
    )["left_knee_angle"]
    assert candidate["selected_source"] == "2D"

    unsure = _metric(reliable=False, observable=False)
    candidate = select_metric_candidates(
        {"left_knee_angle": unsure}, {}, _reliability(unsure)
    )["left_knee_angle"]
    assert candidate["selected_source"] == "UNSURE"


def test_validation_budget_and_layer_contract_never_change_render_clock() -> None:
    gate = ValidationBudgetGate(
        warning_pose_age_ms=80.0,
        validation_budget_ms=4.0,
        maximum_stride=4,
    )
    assert gate.observe(pose_age_ms=120.0, validation_ms=8.0) == 2
    assert gate.observe(pose_age_ms=120.0, validation_ms=8.0) == 4

    contract = realtime_layer_contract(
        validation={"available": True, "validation_ms": 3.0},
        validation_stride=gate.stride,
    )
    assert contract["renderer_waits_for_validation"] is False
    assert contract["analysis"]["consumed_by_formal_rules"] is True
    assert contract["validation"]["consumed_by_formal_rules"] is False
    assert contract["validation"]["cadence_stride"] == 4
    assert contract["formal_threshold_replacement_allowed"] is False
