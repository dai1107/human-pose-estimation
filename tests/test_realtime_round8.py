from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from hyrox.base import BaseActionAnalyzer
from hyrox.validity import (
    BodyRuleResult,
    RepCandidate,
    aggregate_rep_decision,
    apply_three_d_assist,
)
from src.biomechanics.kinematics_3d import (
    ANGLE_DEFINITIONS_3D,
    ThreeDKinematicsTracker,
    summarize_three_d_records,
)
from src.configuration import ConfigValidationError
from src.product_pose import ThreeDKinematicsConfig, load_product_pose_config
from src.utils.metrics import RealtimeMetrics
from tests.test_lunge_validity import _analyzer as lunge_analyzer
from tests.test_lunge_validity import _pose as lunge_pose
from tests.test_realtime_round7 import _knee_result
from tests.test_rowing import _analyzer as rowing_analyzer
from tests.test_rowing import _features as rowing_features
from tests.test_wall_ball import _run_rule_candidate as run_wall_ball_candidate
from tests.test_wall_ball import _features as wall_ball_features
from tests.test_wall_ball import _valid_bottom as wall_ball_bottom
from tests.test_wall_ball import _valid_throw as wall_ball_throw


def _assist_payload(
    *,
    mode: str = "assist",
    enabled: bool = True,
    reliable: bool = True,
    conflict_angles: tuple[str, ...] = (),
    boost: float = 0.05,
    conflict_cap: float = 0.49,
) -> dict[str, object]:
    measurements: dict[str, dict[str, object]] = {}
    for angle_name in ANGLE_DEFINITIONS_3D:
        conflict = angle_name in conflict_angles
        measurements[angle_name] = {
            "angle_2d": 170.0,
            "angle_3d": 130.0 if conflict else 170.0,
            "selected_angle": 170.0,
            "selected_source": "2d_assist" if mode == "assist" else "2d_shadow",
            "confidence": 0.95,
            "three_d_reliable": reliable and not conflict,
            "difference_deg": 40.0 if conflict else 0.0,
            "quality_reasons": (
                ["two_d_three_d_conflict"]
                if conflict
                else [] if reliable else ["world_landmarks_missing"]
            ),
        }
    return {
        "enabled": enabled,
        "decision_mode": mode,
        "assist_status": "conflict" if conflict_angles else "supporting",
        "assist_confidence_boost": boost,
        "assist_conflict_confidence_cap": conflict_cap,
        "measurements": measurements,
    }


def _candidate(action: str, payload: Mapping[str, object]) -> RepCandidate:
    return RepCandidate(
        action=action,
        start_frame=1,
        end_frame=1,
        frames=({"visible_score": 0.95, "three_d_kinematics": dict(payload)},),
    )


def _rule(rule_id: str, status: str = "PASS", confidence: float = 0.70) -> BodyRuleResult:
    return BodyRuleResult(
        rule_id=rule_id,
        status=status,  # type: ignore[arg-type]
        confidence=confidence,
        evidence_frames=(1,),
    )


def test_product_config_promotes_assist_but_keeps_shadow_compatible(tmp_path: Path) -> None:
    config = load_product_pose_config(Path("configs/product_pose.yaml"))

    assert config.three_d_kinematics.decision_mode == "assist"
    assert config.three_d_kinematics.assist_confidence_boost == pytest.approx(0.05)
    assert config.three_d_kinematics.assist_conflict_confidence_cap == pytest.approx(0.49)

    invalid = tmp_path / "invalid_assist.yaml"
    invalid.write_text(
        "product_pose:\n"
        "  backend: mediapipe\n"
        "  allow_experimental_backends: false\n"
        "three_d_kinematics:\n"
        "  enabled: true\n"
        "  decision_mode: assist\n"
        "  assist_confidence_boost: 1.1\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError, match="between 0 and 1"):
        load_product_pose_config(invalid)


def test_assist_tracker_keeps_selected_angle_2d_and_reports_support() -> None:
    tracker = ThreeDKinematicsTracker(
        ThreeDKinematicsConfig(decision_mode="assist")
    )

    attached, result = tracker.attach(
        _knee_result(),
        capture_timestamp_ns=1_000_000,
    )
    measurement = result.measurements["left_knee_angle"]

    assert measurement.selected_angle == measurement.angle_2d
    assert measurement.selected_source == "2d_assist"
    assert result.assist_status == "supporting"
    assert attached.extra["three_d_kinematics"]["decision_mode"] == "assist"

    metrics = RealtimeMetrics(backend="mediapipe", smoothing="one-euro")
    snapshot = metrics.update(
        attached,
        {},
        frame_started=1.0,
        frame_finished=1.1,
    )
    assert snapshot.three_d_assist_support_frame_ratio == 1.0
    assert snapshot.three_d_assist_conflict_frame_ratio == 0.0


def test_reliable_3d_boosts_mapped_rule_confidence_without_changing_status() -> None:
    original = aggregate_rep_decision((_rule("full_knee_extension"),))

    assisted, assessment = apply_three_d_assist(
        original,
        _candidate("lunge", _assist_payload()),
        required_rules=("full_knee_extension",),
    )

    assert original.status == assisted.status == "VALID"
    assert assisted.rules[0].status == "PASS"
    assert assisted.confidence == pytest.approx(0.75)
    assert assessment.status == "SUPPORTING"
    assert assessment.boosted_rules == ("full_knee_extension",)


def test_assist_never_promotes_an_unsure_2d_rule_to_valid() -> None:
    original = aggregate_rep_decision((_rule("full_knee_extension", "UNSURE"),))

    assisted, assessment = apply_three_d_assist(
        original,
        _candidate("lunge", _assist_payload()),
        required_rules=("full_knee_extension",),
    )

    assert original.status == assisted.status == "UNSURE"
    assert assisted.rules[0].status == "UNSURE"
    assert assessment.status == "SUPPORTING"


def test_severe_2d_3d_conflict_downgrades_related_decision_to_unsure() -> None:
    original = aggregate_rep_decision((_rule("full_knee_extension", confidence=0.95),))
    payload = _assist_payload(conflict_angles=("left_knee_angle",))

    assisted, assessment = apply_three_d_assist(
        original,
        _candidate("lunge", payload),
        required_rules=("full_knee_extension",),
    )

    assert original.status == "VALID"
    assert assisted.status == "UNSURE"
    assert assisted.rules[0].status == "PASS"
    assert assisted.confidence == pytest.approx(0.49)
    assert assisted.reason_codes[0] == "THREE_D_ASSIST_CONFLICT"
    assert assessment.conflicted_rules == ("full_knee_extension",)


@pytest.mark.parametrize(
    ("action", "rule_id"),
    (
        ("lunge", "trailing_knee_contact"),
        ("wall_ball", "hip_below_knee"),
        ("burpee_broad_jump", "chest_ground_contact"),
    ),
)
def test_ground_contact_and_image_position_rules_remain_strictly_2d(
    action: str,
    rule_id: str,
) -> None:
    original = aggregate_rep_decision((_rule(rule_id, confidence=0.95),))

    assisted, assessment = apply_three_d_assist(
        original,
        _candidate(
            action,
            _assist_payload(conflict_angles=("left_knee_angle",)),
        ),
        required_rules=(rule_id,),
    )

    assert assisted == original
    assert assessment.status == "NOT_APPLICABLE"


@pytest.mark.parametrize(
    "payload",
    (
        _assist_payload(reliable=False),
        _assist_payload(enabled=False),
        _assist_payload(mode="shadow"),
    ),
)
def test_unavailable_disabled_or_shadow_3d_is_exact_2d_fallback(
    payload: Mapping[str, object],
) -> None:
    original = aggregate_rep_decision((_rule("full_knee_extension"),))

    assisted, assessment = apply_three_d_assist(
        original,
        _candidate("lunge", payload),
        required_rules=("full_knee_extension",),
    )

    assert assisted == original
    assert assessment.status in {"FALLBACK_2D", "DISABLED", "SHADOW"}


def test_base_analyzer_applies_assist_before_observability_gate() -> None:
    analyzer = BaseActionAnalyzer(action="rowing")
    analyzer.begin_frame(
        {
            "visible_score": 0.95,
            "three_d_kinematics": _assist_payload(),
        },
        timestamp_ms=100,
    )

    decision = analyzer.register_completed_sequence(confidence=0.70)

    assert decision.status == "VALID"
    assert decision.confidence == pytest.approx(0.75)
    assert analyzer.last_three_d_assist_assessment is not None
    assert analyzer.last_three_d_assist_assessment.status == "SUPPORTING"


def _run_rowing(payload: Mapping[str, object] | None) -> list[dict[str, object]]:
    analyzer = rowing_analyzer()
    sequence = (
        rowing_features(),
        rowing_features(left_knee_angle=125, right_knee_angle=127),
        rowing_features(
            left_knee_angle=155,
            right_knee_angle=157,
            left_elbow_angle=105,
            right_elbow_angle=107,
            torso_angle=-15,
        ),
    )
    states: list[dict[str, object]] = []
    for timestamp, features in zip((0, 150, 300), sequence):
        frame: dict[str, object] = dict(features)
        if payload is not None:
            frame["three_d_kinematics"] = dict(payload)
        states.append(analyzer.update(frame, timestamp))
    return states


def test_rowing_assist_preserves_phase_sequence_and_2d_count_thresholds() -> None:
    baseline = _run_rowing(None)
    assisted = _run_rowing(_assist_payload())

    assert [state["phase"] for state in assisted] == [
        state["phase"] for state in baseline
    ]
    assert assisted[-1]["rep_count"] == baseline[-1]["rep_count"] == 1
    assert assisted[-1]["last_rep_decision"]["status"] == "VALID"
    assert assisted[-1]["last_rep_decision"]["rules"][0]["status"] == "PASS"
    assert assisted[-1]["last_rep_decision"]["confidence"] >= baseline[-1][
        "last_rep_decision"
    ]["confidence"]


def test_rowing_conflict_keeps_2d_phase_but_downgrades_completed_candidate() -> None:
    baseline = _run_rowing(None)
    conflicted = _run_rowing(
        _assist_payload(conflict_angles=("left_knee_angle",))
    )

    assert [state["phase"] for state in conflicted] == [
        state["phase"] for state in baseline
    ]
    assert conflicted[-1]["rep_count"] == 0
    assert conflicted[-1]["unsure_count"] == 1
    assert conflicted[-1]["last_rep_decision"]["status"] == "UNSURE"
    assert conflicted[-1]["last_rep_decision"]["rules"][0]["status"] == "PASS"


def _run_lunge(payload: Mapping[str, object] | None) -> dict[str, object]:
    analyzer = lunge_analyzer()
    frames = (
        ("stand", 0),
        ("descent", 100),
        ("bottom", 200),
        ("bottom", 300),
        ("bottom", 400),
        ("ascent", 500),
        ("stand", 600),
        ("stand", 700),
    )
    state: dict[str, object] = {}
    for phase, timestamp in frames:
        features = lunge_pose(phase)
        if payload is not None:
            features["three_d_kinematics"] = dict(payload)
        state = analyzer.update(features, timestamp)
    return state


def test_lunge_and_wall_ball_support_preserves_required_2d_rule_outcomes() -> None:
    lunge_baseline = _run_lunge(None)
    lunge_assisted = _run_lunge(_assist_payload())
    wall_baseline = run_wall_ball_candidate(start=wall_ball_features())
    wall_assisted = run_wall_ball_candidate(
        start={**wall_ball_features(), "three_d_kinematics": _assist_payload()},
        bottom={**wall_ball_bottom(), "three_d_kinematics": _assist_payload()},
        throw={**wall_ball_throw(), "three_d_kinematics": _assist_payload()},
    )

    for baseline, assisted in (
        (lunge_baseline, lunge_assisted),
        (wall_baseline, wall_assisted),
    ):
        assert assisted["rep_count"] == baseline["rep_count"] == 1
        assert assisted["last_rep_decision"]["status"] == "VALID"
        assert [rule["status"] for rule in assisted["last_rep_decision"]["rules"]] == [
            rule["status"] for rule in baseline["last_rep_decision"]["rules"]
        ]


def test_assist_summary_reports_modes_support_fallback_and_conflicts() -> None:
    summary = summarize_three_d_records(
        [
            {"three_d_kinematics": {
                "decision_mode": "assist",
                "assist_status": "supporting",
                "three_d_available": True,
                "three_d_reliable": True,
                "three_d_reliable_ratio": 1.0,
                "three_d_conflict_ratio": 0.0,
            }},
            {"three_d_kinematics": {
                "decision_mode": "assist",
                "assist_status": "conflict",
                "three_d_available": True,
                "three_d_reliable": False,
                "three_d_reliable_ratio": 0.5,
                "three_d_conflict_ratio": 0.25,
                "quality_reasons": ["two_d_three_d_conflict"],
            }},
        ]
    )

    assert summary["decision_modes"] == {"assist": 2}
    assert summary["assist_statuses"] == {"conflict": 1, "supporting": 1}
    assert summary["mean_conflict_angle_ratio"] == pytest.approx(0.125)
