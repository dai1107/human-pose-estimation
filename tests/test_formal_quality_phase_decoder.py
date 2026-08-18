from __future__ import annotations

from hyrox.base import BaseActionAnalyzer, PhaseSequenceTracker
from hyrox.features import extract_basic_pose_features
from hyrox.phase_decoder import PhaseEvidenceScorer, TemporalPhaseDecoder
from hyrox.view_policy import action_view_capability, action_view_capability_matrix
from src.pose.formal_quality import FormalLandmarkQualityGate


def _landmarks(offset: float = 0.0) -> dict[str, dict[str, float]]:
    values = {
        "left_shoulder": (0.40, 0.25), "right_shoulder": (0.60, 0.25),
        "left_elbow": (0.38, 0.40), "right_elbow": (0.62, 0.40),
        "left_wrist": (0.36, 0.55), "right_wrist": (0.64, 0.55),
        "left_hip": (0.43, 0.52), "right_hip": (0.57, 0.52),
        "left_knee": (0.43, 0.72), "right_knee": (0.57, 0.72),
        "left_ankle": (0.43, 0.92), "right_ankle": (0.57, 0.92),
        "left_heel": (0.41, 0.94), "right_heel": (0.55, 0.94),
        "left_foot_index": (0.47, 0.95), "right_foot_index": (0.61, 0.95),
    }
    return {
        name: {"x": x + offset, "y": y, "visibility": 0.95, "presence": 0.95}
        for name, (x, y) in values.items()
    }


def test_predicted_joint_makes_dependent_metric_missing_with_reason() -> None:
    points = _landmarks()
    quality = {
        name: {"observable": True, "origin": "observed", "reason_codes": []}
        for name in points
    }
    quality["left_knee"] = {
        "observable": False,
        "origin": "predicted",
        "reason_codes": ["OCCLUDED"],
    }
    features = extract_basic_pose_features(
        points, 640, 480, landmark_quality=quality
    )
    assert features["left_knee_angle"] is None
    diagnostic = features["metric_observability"]["left_knee_angle"]
    assert diagnostic["status"] == "UNOBSERVABLE"
    assert "PREDICTED_EVIDENCE_FORBIDDEN" in diagnostic["reason_codes"]
    assert features["right_knee_angle"] is not None


def test_held_pose_cannot_confirm_endpoint() -> None:
    points = _landmarks()
    gate = FormalLandmarkQualityGate()
    quality = gate.evaluate(
        points,
        timestamp_ms=100,
        metadata={"stabilized_hold": True},
    ).as_dict()
    features = extract_basic_pose_features(points, 640, 480, formal_quality=quality)
    analyzer = BaseActionAnalyzer("test")
    analyzer.stable_phase = "start"
    analyzer.raw_phase = "start"
    analyzer.begin_frame(features, 100)
    _, stable = analyzer._advance_confirmed_phase("endpoint", 1)
    assert stable == "start"
    assert analyzer.raw_phase == "low_visibility"
    assert "ENDPOINT_EVIDENCE_UNOBSERVABLE" in analyzer.last_phase_scores["reason_codes"]


def test_isolated_landmark_jump_is_rejected_but_body_translation_is_not() -> None:
    gate = FormalLandmarkQualityGate(max_isolated_landmark_jump=0.15)
    first = _landmarks()
    gate.evaluate(first, timestamp_ms=0)
    jumped = _landmarks()
    jumped["left_wrist"]["x"] += 0.35
    quality = gate.evaluate(jumped, timestamp_ms=33)
    assert quality.landmarks["left_wrist"]["observable"] is False
    assert "LANDMARK_JUMP" in quality.landmarks["left_wrist"]["reason_codes"]

    translated_gate = FormalLandmarkQualityGate(max_isolated_landmark_jump=0.15)
    translated_gate.evaluate(first, timestamp_ms=0)
    translated = translated_gate.evaluate(_landmarks(0.20), timestamp_ms=33)
    assert translated.landmarks["left_wrist"]["observable"] is True


def test_identity_discontinuity_resets_progress_but_preserves_counts() -> None:
    analyzer = BaseActionAnalyzer("test")
    analyzer.rep_sequence = PhaseSequenceTracker(("start", "end"))
    analyzer.rep_sequence.update("start")
    analyzer.pose_valid_rep_count = analyzer.rep_count = 3
    analyzer.candidate_count = 4
    analyzer._candidate_phases.add("start")
    analyzer.begin_frame(
        {
            "visible_score": 0.9,
            "identity_continuity": {
                "status": "DISCONTINUOUS",
                "reason_codes": ["IDENTITY_CENTER_JUMP"],
            },
        },
        200,
    )
    assert analyzer.pose_valid_rep_count == analyzer.rep_count == 3
    assert analyzer.candidate_count == 4
    assert analyzer.rep_sequence.progress == 0
    assert analyzer.identity_reset_count == 1
    assert analyzer.last_identity_reset_reason == ("IDENTITY_CENTER_JUMP",)


def test_phase_decoder_enforces_duration_hysteresis_and_legal_transition() -> None:
    decoder = TemporalPhaseDecoder()
    first = decoder.update(
        "bottom",
        current_stable_phase="start",
        minimum_duration_frames=2,
        legal_sequence=("start", "down", "bottom", "up", "start"),
    )
    second = decoder.update(
        "bottom",
        current_stable_phase=first.stable_phase,
        minimum_duration_frames=2,
        legal_sequence=("start", "down", "bottom", "up", "start"),
    )
    assert first.held_for_hysteresis is True
    assert second.transition_legal is False
    assert second.stable_phase == "start"
    decoder.update(
        "down",
        current_stable_phase="start",
        minimum_duration_frames=1,
        legal_sequence=("start", "down", "bottom", "up", "start"),
    )
    legal = decoder.update(
        "bottom",
        current_stable_phase="down",
        minimum_duration_frames=1,
        legal_sequence=("start", "down", "bottom", "up", "start"),
    )
    assert legal.transition_legal is True
    assert legal.stable_phase == "bottom"


def test_view_capability_conditions_scores_without_becoming_decision_gate() -> None:
    matrix = action_view_capability_matrix()
    assert set(matrix) >= {"lunge", "rowing", "wall_ball", "sled_push"}
    side = action_view_capability("rowing", "side")
    front = action_view_capability("rowing", "front")
    assert side.level == "recommended"
    assert front.level == "not_recommended"
    scorer = PhaseEvidenceScorer()
    features = {"visible_score": 0.9, "formal_evidence_quality": 0.9}
    side_score = scorer.score("drive", features, view_multiplier=side.score_multiplier)
    front_score = scorer.score("drive", features, view_multiplier=front.score_multiplier)
    assert side_score.confidence > front_score.confidence
    assert front.as_dict()["decision_gate"] is False


def test_body_centered_coordinates_are_translation_invariant() -> None:
    first = extract_basic_pose_features(_landmarks(0.0), 640, 480)
    shifted = extract_basic_pose_features(_landmarks(0.10), 640, 480)
    assert first["left_wrist_body_x"] == shifted["left_wrist_body_x"]
    assert first["hip_knee_depth_body"] == shifted["hip_knee_depth_body"]
