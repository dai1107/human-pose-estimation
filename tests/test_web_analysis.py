from __future__ import annotations

import pytest

from webui.analysis import RepVoiceFeedbackTracker, assess_action, enrich_report, official_rules_for, render_text_report, standards_for, visible_feedback


def test_action_assessment_uses_phase_specific_angle_ranges() -> None:
    good = assess_action(
        "lunge",
        "bottom",
        {"min_knee_angle": 98.0, "left_knee_angle": 98.0, "right_knee_angle": 101.0, "torso_angle": 12.0},
    )
    bad = assess_action(
        "lunge",
        "bottom",
        {"min_knee_angle": 142.0, "left_knee_angle": 142.0, "right_knee_angle": 145.0, "torso_angle": 12.0},
    )
    borderline = assess_action("lunge", "bottom", {"min_knee_angle": 130.0, "torso_angle": 12.0})

    assert good["status"] == "good"
    assert good["evaluable"] is True
    assert bad["status"] == "bad"
    assert bad["evaluable"] is True
    assert borderline["status"] == "unknown"
    assert borderline["evaluable"] is False
    assert {item["anchor"] for item in bad["angles"]} >= {"left_knee", "right_knee"}
    assert any(item["range_text"] == "75–125°" for item in standards_for("lunge"))
    assert official_rules_for("lunge")[0]["pose_observable"] is True


def test_effort_feedback_is_hidden_during_recovery_but_retained_during_pull() -> None:
    items = [{"level": "warn", "code": "ARMS_ONLY_PULL", "text": "配合髋腿发力"}]

    assert visible_feedback(items, "recover") == []
    assert visible_feedback(items, "pull") == items


def test_wall_ball_rising_phase_does_not_reuse_bottom_angle_standard() -> None:
    rising = assess_action(
        "wall_ball",
        "drive",
        {"min_knee_angle": 132.0, "left_knee_angle": 132.0, "right_knee_angle": 134.0},
    )
    throwing = assess_action(
        "wall_ball",
        "throw_extension",
        {
            "min_knee_angle": 155.0,
            "min_elbow_angle": 130.0,
            "left_knee_angle": 155.0,
            "right_knee_angle": 157.0,
            "left_elbow_angle": 130.0,
            "right_elbow_angle": 132.0,
        },
    )

    assert rising["status"] == "good"
    assert rising["evaluable"] is False
    assert rising["evaluation_mode"] == "not_applicable"
    assert rising["criteria"] == []
    assert throwing["status"] == "good"


@pytest.mark.parametrize(
    ("action", "standard_phase", "transition_phase", "features"),
    [
        ("lunge", "bottom", "descent", {"min_knee_angle": 145.0, "torso_angle": 50.0}),
        ("farmers_carry", "carrying", "rest", {"torso_angle": 50.0, "shoulder_tilt": 0.20}),
        ("rowing", "catch", "recovery", {"min_knee_angle": 140.0}),
        ("skierg", "pull_down", "return", {"min_knee_angle": 70.0, "torso_angle": 5.0}),
        ("burpee_broad_jump", "chest_down", "flight_or_move", {"torso_angle": 30.0}),
        ("sled_push", "drive", "reset", {"torso_angle": 5.0, "min_knee_angle": 70.0}),
        ("sled_pull", "pull", "recover", {"left_knee_angle": 70.0, "right_knee_angle": 75.0, "torso_angle": 60.0}),
    ],
)
def test_all_other_actions_only_evaluate_standard_bearing_phases(
    action: str,
    standard_phase: str,
    transition_phase: str,
    features: dict[str, float],
) -> None:
    feedback = [{"level": "warn", "code": "KEEP_FORM", "text": "保持动作稳定"}]

    standard = assess_action(action, standard_phase, features)
    transition = assess_action(action, transition_phase, features, feedback)

    assert standard["status"] == "bad"
    assert standard["evaluable"] is True
    assert standard["evaluation_mode"] == "standard"
    assert transition["status"] == "good"
    assert transition["evaluable"] is False
    assert transition["evaluation_mode"] == "not_applicable"
    assert transition["criteria"] == []
    assert transition["problem_codes"] == ["KEEP_FORM"]


@pytest.mark.parametrize("phase", ["reach", "recover"])
def test_sled_pull_forward_movement_is_green_but_not_evaluable(phase: str) -> None:
    result = assess_action(
        "sled_pull",
        phase,
        {"left_elbow_angle": 90.0, "right_elbow_angle": 90.0, "torso_angle": 50.0},
        [{"level": "warn", "code": "NO_CLEAR_PULL", "text": "继续完成拉动"}],
    )

    assert result["status"] == "good"
    assert result["evaluable"] is False
    assert result["criteria"] == []
    assert result["problem_codes"] == ["NO_CLEAR_PULL"]


def test_unrecognized_or_low_quality_phase_stays_neutral() -> None:
    result = assess_action(
        "sled_pull",
        "unknown",
        {"left_knee_angle": 70.0, "right_knee_angle": 70.0, "torso_angle": 60.0},
        [{"level": "warn", "code": "LOW_VISIBILITY", "text": "请保证全身入镜"}],
    )

    assert result["status"] == "unknown"
    assert result["evaluable"] is False
    assert result["evaluation_mode"] == "unavailable"
    assert result["problem_codes"] == ["LOW_VISIBILITY"]

    blurred = assess_action("lunge", "bottom", {"visible_score": 0.3, "min_knee_angle": 145.0})
    assert blurred["status"] == "unknown"
    assert blurred["evaluation_mode"] == "low_quality"


def test_report_is_generated_with_compliance_and_standard_ranges() -> None:
    report = enrich_report(
        {
            "summary": {"action": "sled_push", "action_label": "推雪橇", "reps": 2},
            "frames": [
                {"timestamp_unix_ms": 1000, "reps": 0, "assessment": {"status": "good", "evaluable": True}, "detected_issues": []},
                {
                    "timestamp_unix_ms": 1400,
                    "reps": 1,
                    "assessment": {"status": "bad", "evaluable": True},
                    "detected_issues": [{"level": "warn", "code": "TORSO_TOO_UPRIGHT", "text": "身体过直"}],
                },
                {
                    "timestamp_unix_ms": 1800,
                    "reps": 1,
                    "assessment": {"status": "good", "evaluable": False, "evaluation_mode": "not_applicable"},
                    "detected_issues": [{"level": "warn", "code": "WALKING", "text": "向前移动"}],
                },
                {"timestamp_unix_ms": 2200, "reps": 2, "assessment": {"status": "good", "evaluable": False}, "detected_issues": []},
            ],
        }
    )

    assert report["analysis"]["compliance_rate"] == 50.0
    assert report["analysis"]["evaluable_frames"] == 2
    assert report["analysis"]["overall_status"] == "建议重点调整"
    assert report["analysis"]["common_issues"][0]["code"] == "TORSO_TOO_UPRIGHT"
    assert any(item["range_text"] == "20–70°" for item in report["analysis"]["standards"])
    assert [item["title"] for item in report["analysis"]["rep_details"]] == ["第 1 次蹬步", "第 2 次蹬步"]
    text_report = render_text_report(report)
    assert "逐次动作表现" in text_report
    assert "第 1 次蹬步" in text_report
    assert "HYROX Singles Rulebook 26/27" in text_report


def test_each_completed_rep_summarizes_strengths_and_improvements() -> None:
    report = enrich_report(
        {
            "summary": {"action": "lunge", "action_label": "沙袋弓步", "reps": 1},
            "frames": [
                {
                    "timestamp_unix_ms": 1000,
                    "reps": 0,
                    "assessment": {
                        "status": "good",
                        "evaluable": True,
                        "criteria": [
                            {
                                "label": "后膝接近地面的角度参考",
                                "passed": True,
                                "clear_failure": False,
                                "value": 95,
                                "unit": "°",
                                "range_text": "75–125°",
                            }
                        ],
                    },
                    "detected_issues": [],
                },
                {
                    "timestamp_unix_ms": 1500,
                    "reps": 1,
                    "assessment": {"status": "bad", "evaluable": True, "criteria": []},
                    "detected_issues": [{"level": "warn", "code": "LEAN_TOO_MUCH", "text": "躯干前倾过多"}],
                },
            ],
        }
    )

    detail = report["analysis"]["rep_details"][0]
    assert detail["title"] == "第 1 次弓步"
    assert any("后膝接近地面" in item for item in detail["positives"])
    assert any("躯干前倾过多" in item for item in detail["improvements"])


def test_completed_rep_voice_feedback_reuses_report_improvements() -> None:
    tracker = RepVoiceFeedbackTracker()
    tracker.update(
        action="lunge",
        reps=0,
        timestamp_ms=1000,
        assessment={
            "criteria": [
                {
                    "label": "躯干稳定",
                    "clear_failure": True,
                    "passed": False,
                    "range_text": "0–30°",
                    "unit": "°",
                    "value": 48,
                }
            ]
        },
        detected_issues=[
            {"level": "warn", "code": "LEAN_TOO_MUCH", "text": "躯干前倾过多"},
            {"level": "warn", "code": "LOW_VISIBILITY", "text": "请保证全身入镜"},
        ],
    )
    event = tracker.update(
        action="lunge",
        reps=1,
        timestamp_ms=1600,
        assessment={"criteria": []},
        detected_issues=[{"level": "warn", "code": "LEAN_TOO_MUCH", "text": "躯干前倾过多"}],
    )

    assert event is not None
    assert event["id"] == "lunge:rep:1"
    assert "第 1 次" in event["speech"]
    assert any("躯干稳定" in item for item in event["improvements"])
    assert any("躯干前倾过多" in item for item in event["improvements"])
    assert "全身入镜" not in event["speech"]


def test_farmers_carry_speaks_only_after_issue_persists_and_honors_cooldown() -> None:
    tracker = RepVoiceFeedbackTracker()
    issue = [{"level": "warn", "code": "TORSO_LEAN", "text": "保持核心稳定"}]

    assert tracker.update(action="farmers_carry", reps=0, assessment={}, detected_issues=issue, timestamp_ms=1000) is None
    event = tracker.update(action="farmers_carry", reps=0, assessment={}, detected_issues=issue, timestamp_ms=2300)
    assert event is not None
    assert event["mode"] == "continuous"
    assert "保持核心稳定" in event["speech"]
    assert tracker.update(action="farmers_carry", reps=0, assessment={}, detected_issues=issue, timestamp_ms=5000) == event
    repeated = tracker.update(action="farmers_carry", reps=0, assessment={}, detected_issues=issue, timestamp_ms=10_400)
    assert repeated is not None
    assert repeated["id"] != event["id"]
