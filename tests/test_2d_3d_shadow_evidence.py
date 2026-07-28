from __future__ import annotations

from hyrox.contact import ContactResult
from hyrox.validity import (
    BodyRuleResult,
    RepCandidate,
    aggregate_rep_decision,
    apply_three_d_assist,
)
from src.biomechanics.shadow_evidence_3d import (
    BodyRelative3DTracker,
    ShadowEvidence3DConfig,
)
from tests.test_lunge_validity import _analyzer as lunge_analyzer
from tools.run_2d_3d_shadow_evidence import (
    enforce_no_three_d_valid_promotion,
    leave_one_video_out_folds,
)


def _point(name: str, x: float, y: float, z: float) -> dict[str, object]:
    return {
        "name": name,
        "x": x,
        "y": y,
        "z": z,
        "confidence": 0.99,
        "visibility": 0.99,
        "presence": 0.99,
    }


def _body(offset_y: float = 0.0) -> list[dict[str, object]]:
    values = {
        "left_shoulder": (-0.2, -0.6, 0.0),
        "right_shoulder": (0.2, -0.6, 0.0),
        "left_hip": (-0.15, 0.0, 0.0),
        "right_hip": (0.15, 0.0, 0.0),
        "left_knee": (-0.16, 0.6, -0.2),
        "right_knee": (0.16, 0.6, 0.2),
        "left_ankle": (-0.16, 1.2, -0.3),
        "right_ankle": (0.16, 1.2, 0.3),
        "left_heel": (-0.18, 1.22, -0.3),
        "right_heel": (0.18, 1.22, 0.3),
        "left_foot_index": (-0.16, 1.20, -0.4),
        "right_foot_index": (0.16, 1.20, 0.4),
    }
    return [
        _point(name, x, y + offset_y, z)
        for name, (x, y, z) in values.items()
    ]


def _payload(*, timing_delta: float) -> dict[str, object]:
    body = {
        "reliable": True,
        "foot_motion": {
            "takeoff_time_difference_ms": timing_delta,
        },
        "thresholds": {
            "synchronous_event_ms": 120.0,
            "conflict_event_ms": 220.0,
        },
    }
    return {
        "enabled": True,
        "decision_mode": "assist",
        "experimental_fusion_enabled": True,
        "experimental_angle_fusion_enabled": False,
        "experimental_body_fusion_enabled": True,
        "assist_confidence_boost": 0.04,
        "assist_conflict_confidence_cap": 0.49,
        "measurements": {},
        "body_relative": body,
    }


def _angle_payload(*, conflict: bool) -> dict[str, object]:
    return {
        "enabled": True,
        "decision_mode": "assist",
        "experimental_fusion_enabled": True,
        "experimental_angle_fusion_enabled": True,
        "experimental_body_fusion_enabled": False,
        "experimental_temporal_thresholds": {
            "angle_conflict_min_frames": 3,
            "angle_conflict_min_ratio": 0.5,
            "angle_support_min_frames": 3,
            "angle_support_min_ratio": 0.5,
        },
        "measurements": {
            "left_knee_angle": {
                "three_d_reliable": not conflict,
                "quality_reasons": (
                    ["two_d_three_d_conflict"] if conflict else []
                ),
            },
            "right_knee_angle": {
                "three_d_reliable": not conflict,
                "quality_reasons": (
                    ["two_d_three_d_conflict"] if conflict else []
                ),
            },
        },
    }
def test_hip_compensated_height_ignores_whole_body_vertical_translation() -> None:
    first = BodyRelative3DTracker().update(
        _body(),
        _body(),
        timestamp_ms=0,
        camera_view="front",
    )
    second = BodyRelative3DTracker().update(
        _body(2.0),
        _body(2.0),
        timestamp_ms=0,
        camera_view="front",
    )

    first_height = first["hip_compensated_height_downward_positive"]
    second_height = second["hip_compensated_height_downward_positive"]
    assert first_height["left_knee"] == second_height["left_knee"]
    assert first["contact_inference_allowed"] is False
    assert first["leg_depth_order"]["trailing_side_hint"] == "right"


def test_body_relative_tracker_reports_foot_speed_dwell_and_torso_relation() -> None:
    tracker = BodyRelative3DTracker(
        ShadowEvidence3DConfig(stationary_dwell_frames=2)
    )
    tracker.update(_body(), _body(), timestamp_ms=0, camera_view="front")
    second = tracker.update(
        _body(),
        _body(),
        timestamp_ms=100,
        camera_view="front",
    )

    assert second["foot_motion"]["left_speed_body_per_s"] == 0.0
    assert second["foot_motion"]["left_stationary_dwell_frames"] == 1
    assert second["torso_spatial"]["prone_horizontal_score"] == 0.0


def test_foot_timing_conflict_only_downgrades_a_two_d_decision() -> None:
    rule = BodyRuleResult(
        "simultaneous_takeoff",
        "PASS",
        0.90,
        evidence_frames=(1,),
    )
    original = aggregate_rep_decision((rule,))
    candidate = RepCandidate(
        action="burpee_broad_jump",
        start_frame=1,
        end_frame=1,
        frames=({"three_d_kinematics": _payload(timing_delta=300.0)},),
    )

    assisted, assessment = apply_three_d_assist(
        original,
        candidate,
        required_rules=("simultaneous_takeoff",),
    )

    assert original.status == "VALID"
    assert assisted.status == "UNSURE"
    assert assessment.status == "CONFLICT"


def test_experimental_angle_conflict_requires_temporal_consensus() -> None:
    rule = BodyRuleResult("full_knee_extension", "PASS", 0.90)
    original = aggregate_rep_decision((rule,))
    one_frame = RepCandidate(
        action="lunge",
        start_frame=1,
        end_frame=1,
        frames=({"three_d_kinematics": _angle_payload(conflict=True)},),
    )
    persistent = RepCandidate(
        action="lunge",
        start_frame=1,
        end_frame=3,
        frames=tuple(
            {"three_d_kinematics": _angle_payload(conflict=True)}
            for _ in range(3)
        ),
    )

    transient, transient_assessment = apply_three_d_assist(
        original,
        one_frame,
        required_rules=("full_knee_extension",),
    )
    conflicted, conflict_assessment = apply_three_d_assist(
        original,
        persistent,
        required_rules=("full_knee_extension",),
    )

    assert transient.status == "VALID"
    assert transient_assessment.status == "FALLBACK_2D"
    assert conflicted.status == "UNSURE"
    assert conflict_assessment.status == "CONFLICT"


def test_angle_conflict_with_an_independent_quality_failure_is_unavailable() -> None:
    payload = _angle_payload(conflict=True)
    measurement = payload["measurements"]["left_knee_angle"]
    measurement["quality_reasons"] = [
        "two_d_three_d_conflict",
        "low_visibility",
    ]
    payload["measurements"]["right_knee_angle"] = dict(measurement)
    original = aggregate_rep_decision(
        (BodyRuleResult("full_knee_extension", "PASS", 0.90),)
    )
    candidate = RepCandidate(
        action="lunge",
        start_frame=1,
        end_frame=3,
        frames=tuple(
            {"three_d_kinematics": payload}
            for _ in range(3)
        ),
    )

    assisted, assessment = apply_three_d_assist(
        original,
        candidate,
        required_rules=("full_knee_extension",),
    )

    assert assisted.status == "VALID"
    assert assessment.status == "FALLBACK_2D"


def test_lunge_depth_assist_changes_side_but_contact_remains_two_d() -> None:
    analyzer = lunge_analyzer()
    analyzer._best_side_contacts = {
        "left": ContactResult("NO_CONTACT", 0.9, 0.30, 3, [1, 2, 3]),
        "right": ContactResult("CONTACT", 0.9, 0.05, 3, [1, 2, 3]),
    }
    features = {
        "three_d_kinematics": {
            "decision_mode": "assist",
            "experimental_fusion_enabled": True,
            "body_relative": {
                "reliable": True,
                "leg_depth_order": {
                    "trailing_side_hint": "right",
                    "confidence": 0.9,
                },
            },
        }
    }

    analyzer._select_trailing_leg(features, 0.5, phase="bottom")
    rule = analyzer._contact_rule()

    assert analyzer.current_contact_leg == "right"
    assert analyzer.trailing_leg_source == "three_d_depth_assist"
    assert rule.status == "PASS"
    assert rule.value == 0.05


def test_lovo_folds_have_no_same_video_overlap() -> None:
    folds = leave_one_video_out_folds(["video_2", "video_1", "video_3"])

    assert len(folds) == 3
    assert all(
        fold["held_out_video_id"] not in fold["training_video_ids"]
        for fold in folds
    )
    assert all(len(fold["training_video_ids"]) == 2 for fold in folds)


def test_shadow_guard_blocks_valid_promotion() -> None:
    baseline = {
        "matches": [
            {
                "candidate": {"status": "UNSURE"},
                "human_rep": {"validity": "VALID"},
            }
        ],
        "exact_count_match": True,
    }
    assisted = {
        "matches": [
            {
                "candidate": {"status": "VALID", "reason_codes": []},
                "human_rep": {"validity": "VALID"},
            }
        ],
        "exact_count_match": True,
    }

    guarded = enforce_no_three_d_valid_promotion(baseline, assisted)

    assert guarded["matches"][0]["candidate"]["status"] == "UNSURE"
    assert (
        guarded["shadow_safety_guard"]["blocked_valid_promotion_count"]
        == 1
    )
