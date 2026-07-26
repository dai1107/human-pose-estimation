from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import shutil

import numpy as np
import pytest

from src.action_gating import (
    ACTION_FEATURE_NAMES,
    ActionFeatureWindow,
    AutoActionShadowRuntime,
    LogisticActionModel,
    SwitchProtectedActionGate,
    grouped_cross_validate_logistic,
)
from src.contracts import (
    Evidence,
    build_coordinate_output,
    build_latency_output,
    extend_action_state,
    load_contract_bundle,
)
from src.configuration import ConfigValidationError
from tools.replay_hyrox_video import DEBUG_CSV_COLUMNS, build_debug_row, build_parser
from tools.dataset.round10_shadow import build_round10_reports, evaluate_data_readiness


def test_round10_contract_bundle_is_versioned_safe_and_default_off() -> None:
    bundle = load_contract_bundle()

    assert bundle.versions == {
        "action_gating": "action_gating_v1",
        "scoring_correction": "scoring_correction_v1",
        "coordinate_spaces": "coordinate_spaces_v1",
        "realtime_latency": "realtime_latency_v1",
    }
    assert bundle.action_gating.enabled_by_default is False
    assert bundle.action_gating.runtime_mode == "shadow"
    assert bundle.action_gating.classes[-3:] == ("idle", "transition", "unknown")
    assert bundle.realtime_latency.queue_capacity == 1
    assert bundle.realtime_latency.analysis_uses_prediction is False
    assert bundle.realtime_latency.display_prediction_only is True


def test_round10_contracts_reject_non_boolean_safety_fields(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "configs" / "contracts"
    target = tmp_path / "contracts"
    shutil.copytree(source, target)
    path = target / "action_gating_v1.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "enabled_by_default: false",
            "enabled_by_default: disabled",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match="must be true or false"):
        load_contract_bundle(target)


def test_evidence_rejects_unknown_source_or_status() -> None:
    with pytest.raises(ValueError, match="source"):
        Evidence("x", "magic", "PASS", 1.0, "lunge", "stand")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="status"):
        Evidence("x", "rule", "VALID", 1.0, "lunge", "stand")


def test_additive_contract_output_preserves_legacy_state_and_traceability() -> None:
    bundle = load_contract_bundle()
    legacy = {
        "action": "lunge",
        "phase": "stand",
        "rep_count": 2,
        "feedback_messages": [
            {"code": "HIP_NOT_EXTENDED", "text": "结束时站直"},
            {"code": "EXTRA_STEP", "text": "两次动作之间不要补步"},
            {"code": "THIRD", "text": "不应显示"},
        ],
        "last_rep_decision": {
            "status": "NO_REP",
            "rules": [
                {
                    "rule_id": "full_hip_extension",
                    "status": "FAIL",
                    "confidence": 0.91,
                    "value": 150.0,
                    "reason_code": "HIP_NOT_EXTENDED",
                    "evidence_frames": [10, 11],
                }
            ],
        },
    }

    output = extend_action_state(
        legacy,
        bundle=bundle,
        action="lunge",
        action_source="manual",
    )

    assert output["rep_count"] == 2
    assert output["contract_versions"] == bundle.versions
    assert output["action_source"] == "manual"
    assert output["evidence"][0]["source"] == "rule"
    assert output["evidence"][0]["reason_codes"] == ("HIP_NOT_EXTENDED",)
    scoring = output["scoring_correction"]
    assert scoring["validity"] == "NO_REP"
    assert scoring["rubric_calibrated"] is False
    assert scoring["score_breakdown"]["overall"]["score"] is None
    assert len(scoring["corrections"]) == 2
    assert scoring["corrections"][0]["body_part"] == "hips"
    assert scoring["corrections"][0]["evidence_ids"] == [
        "lunge:full_hip_extension:0"
    ]


def test_coordinate_contract_names_spaces_and_falls_back_to_2d() -> None:
    bundle = load_contract_bundle()
    output = build_coordinate_output(
        bundle.coordinate_spaces,
        three_d_kinematics={
            "three_d_reliable_ratio": 0.0,
            "failure_reasons": ["z_unstable"],
        },
    )

    assert output["analysis_space"] == "image_normalized_2d"
    assert output["fallback_to_2d"] is True
    assert "metric_depth_3d" in output["coordinate_spaces"]
    assert output["metric_depth_landmarks"] is None
    assert output["three_d_failure_reasons"] == ["z_unstable"]


def test_latency_contract_suppresses_all_stale_outputs_without_analysis_prediction() -> None:
    contract = load_contract_bundle().realtime_latency
    fresh = build_latency_output(
        contract,
        capture_timestamp=10.0,
        pose_input_timestamp=10.01,
        pose_finished_timestamp=10.03,
        analysis_timestamp=10.04,
        render_timestamp=10.05,
        source_frame_id=7,
        prediction_horizon_ms=15.0,
    )
    stale = build_latency_output(
        contract,
        capture_timestamp=10.0,
        pose_input_timestamp=10.01,
        pose_finished_timestamp=10.03,
        analysis_timestamp=10.04,
        render_timestamp=10.20,
        source_frame_id=7,
    )

    assert fresh["stale"] is False
    assert fresh["analysis_uses_prediction"] is False
    assert stale["stale"] is True
    assert stale["suppress_pose"] is True
    assert stale["suppress_action"] is True
    assert stale["suppress_correction"] is True


def _probabilities(contract, action: str, confidence: float = 0.9) -> dict[str, float]:
    remainder = (1.0 - confidence) / (len(contract.classes) - 1)
    return {name: confidence if name == action else remainder for name in contract.classes}


def test_switch_guard_requires_duration_hysteresis_cooldown_and_rejects_stale() -> None:
    contract = replace(load_contract_bundle().action_gating, switch_cooldown_ms=200)
    guard = SwitchProtectedActionGate(contract, model_version="test", model_hash="abc")

    started = guard.update(_probabilities(contract, "lunge"), timestamp_ms=0)
    waiting = guard.update(_probabilities(contract, "lunge"), timestamp_ms=499)
    committed = guard.update(_probabilities(contract, "lunge"), timestamp_ms=500)

    assert started.predicted_action == "unknown"
    assert started.switch_reason == "candidate_started"
    assert started.action_state == "setup"
    assert waiting.switch_reason == "minimum_duration"
    assert committed.predicted_action == "lunge"
    assert committed.switch_committed is True

    low = guard.update(_probabilities(contract, "unknown"), timestamp_ms=700)
    exited = guard.update(_probabilities(contract, "unknown"), timestamp_ms=1000)
    assert low.predicted_action == "lunge"
    assert low.switch_reason == "exit_duration"
    assert exited.predicted_action == "unknown"
    assert exited.switch_reason == "exit_committed"

    stale = guard.update(_probabilities(contract, "wall_ball"), timestamp_ms=1500, stale=True)
    assert stale.predicted_action == "unknown"
    assert stale.stale is True
    assert stale.switch_reason == "stale_pose"


def test_manual_override_is_explicit_and_never_looks_like_auto() -> None:
    contract = load_contract_bundle().action_gating
    guard = SwitchProtectedActionGate(contract, model_version="test", model_hash="abc")

    prediction = guard.update(
        _probabilities(contract, "rowing"),
        timestamp_ms=10,
        manual_override="wall_ball",
    )

    assert prediction.predicted_action == "wall_ball"
    assert prediction.action_source == "manual"
    assert prediction.action_confidence == 1.0


def _synthetic_model(contract) -> LogisticActionModel:
    rng = np.random.default_rng(42)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, name in enumerate(contract.classes):
        for _ in range(3):
            row = rng.normal(0.0, 0.05, len(ACTION_FEATURE_NAMES))
            row[class_index % len(ACTION_FEATURE_NAMES)] += 2.0
            rows.append(row)
            labels.append(name)
    return LogisticActionModel.fit(
        np.asarray(rows),
        labels,
        classes=contract.classes,
        epochs=300,
        training_metadata={"data_provenance": "synthetic_contract_test_only"},
    )


def test_logistic_baseline_serializes_hash_and_shadow_does_not_switch_formal_analyzer(tmp_path: Path) -> None:
    contract = replace(load_contract_bundle().action_gating, minimum_window_frames=2)
    model = _synthetic_model(contract)
    path = model.save(tmp_path / "action_model.json")
    loaded = LogisticActionModel.load(path)
    runtime = AutoActionShadowRuntime(loaded, contract)
    features = {name: 0.1 for name in ("visible_score", "min_knee_angle", "body_center_x")}

    first = runtime.update(features, timestamp_ms=0)
    second = runtime.update(features, timestamp_ms=34)

    assert loaded.model_hash == model.model_hash
    assert first["predicted_action"] == "unknown"
    assert first["switch_reason"] == "window_warming"
    assert sum(second["action_probabilities"].values()) == pytest.approx(1.0)
    assert second["action_source"] == "auto_shadow"
    assert "formal_action" not in second


def test_logistic_baseline_grouped_cv_has_no_group_overlap() -> None:
    contract = load_contract_bundle().action_gating
    rng = np.random.default_rng(7)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    for group_index in range(3):
        for class_index, class_name in enumerate(contract.classes):
            row = rng.normal(0.0, 0.02, len(ACTION_FEATURE_NAMES))
            row[class_index] = 3.0
            rows.append(row)
            labels.append(class_name)
            groups.append(f"subject_{group_index}")

    result = grouped_cross_validate_logistic(
        np.asarray(rows),
        labels,
        groups,
        classes=contract.classes,
        maximum_folds=3,
        epochs=300,
    )

    assert result["group_exclusive"] is True
    assert result["group_count"] == 3
    assert result["fold_count"] == 3
    assert result["macro_f1"] >= 0.7
    assert all(fold["group_overlap"] == [] for fold in result["folds"])


def test_action_feature_window_is_causal_and_bounded() -> None:
    contract = replace(
        load_contract_bundle().action_gating,
        minimum_window_frames=1,
        maximum_window_frames=3,
        window_ms=100,
    )
    window = ActionFeatureWindow(contract)
    for timestamp, center in ((0, 0.0), (50, 0.2), (100, 0.4), (200, 1.0)):
        window.update({"body_center_x": center}, timestamp_ms=timestamp)

    assert window.frame_count == 2
    vector = window.vector()
    center_index = ACTION_FEATURE_NAMES.index("body_center_x_delta")
    assert vector[center_index] == pytest.approx(0.6)


def test_replay_shadow_is_explicit_default_off_and_auto_is_not_a_formal_action() -> None:
    parser = build_parser()
    defaults = parser.parse_args(["--video", "sample.mp4", "--hyrox-action", "lunge"])
    enabled = parser.parse_args(
        [
            "--video", "sample.mp4",
            "--hyrox-action", "lunge",
            "--auto-action-shadow",
            "--auto-action-model", "model.json",
            "--save-shadow-json", "shadow.json",
        ]
    )

    assert defaults.auto_action_shadow is False
    assert defaults.auto_action_model == ""
    assert enabled.auto_action_shadow is True
    assert enabled.auto_action_model == "model.json"
    with pytest.raises(SystemExit):
        parser.parse_args(["--video", "sample.mp4", "--hyrox-action", "auto"])


def test_debug_csv_adds_shadow_fields_without_removing_legacy_columns() -> None:
    gate = {
        "action_probabilities": {"lunge": 0.8, "unknown": 0.2},
        "predicted_action": "lunge",
        "action_confidence": 0.8,
        "action_state": "active",
        "supported_view": True,
        "equipment_context": "sandbag",
        "switch_candidate_since_ms": 100,
        "action_model_version": "v1",
        "action_model_hash": "abc",
        "action_source": "auto_shadow",
        "stale": False,
        "switch_committed": True,
    }
    row = build_debug_row(
        frame_index=1,
        timestamp_ms=33,
        has_pose=True,
        features={"visible_score": 0.9},
        state={"action": "lunge", "phase": "stand", "rep_count": 1},
        action_gate=gate,
    )

    assert {"frame_index", "action", "phase", "rep_count"} <= set(DEBUG_CSV_COLUMNS)
    assert row["predicted_action"] == "lunge"
    assert row["action_source"] == "auto_shadow"
    assert '"lunge": 0.8' in row["action_probabilities"]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_round10_reports_fail_closed_when_human_truth_is_pending(tmp_path: Path) -> None:
    root = tmp_path / "hyrox"
    record = {
        "record_id": "phone_lunge_001",
        "action": "lunge",
        "subject_id": "subject_pending",
        "usage_authorization": {"status": "pending_confirmation", "authorized_uses": []},
        "review_status": {"subject_identity": "pending"},
        "pose_cache": {"status": "complete", "target_track_id": "target_athlete_001"},
    }
    _write_json(root / "manifests" / "phone_records.json", {"records": [record]})
    _write_json(
        root / "manifests" / "data_roles_v1.json",
        {"assignments": [{"record_id": record["record_id"], "role": "unassigned_pending_review", "training_eligible": False}]},
    )
    _write_json(
        root / "annotations" / "action_segments_v1.json",
        {
            "records": [
                {
                    "record_id": record["record_id"],
                    "action_type": "lunge",
                    "video_action_review_status": "human_confirmation_pending",
                    "training_eligible": False,
                    "segments": [{"human_confirmed": False, "annotator_type": "ai"}],
                }
            ]
        },
    )
    _write_json(
        root / "annotations" / "object_scene_evidence_v1.json",
        {"records": [{"record_id": record["record_id"], "evidence": []}]},
    )
    _write_json(root / "reports" / "annotation_agreement_v1.json", {"eligible_reviewer_count": 0, "release_gate_passed": False})
    _write_json(
        root / "reports" / "round7_implementation_summary.json",
        {"roi_summary": {"precision_gate_passed": False, "latency_gate_passed": False}},
    )
    _write_json(
        root / "reports" / "round8_implementation_summary.json",
        {"temporal_summary": {"sensor_to_photon": {"status": "not_measured", "reason": "external measurement pending"}}},
    )
    _write_json(
        root / "reports" / "round9_active_review_queue_v1.json",
        {"records": [{"record_id": record["record_id"], "priority": 12}]},
    )

    readiness = evaluate_data_readiness(root, load_contract_bundle())
    paths = build_round10_reports(root)
    summary = json.loads(paths["implementation_summary"].read_text(encoding="utf-8"))
    ablation = json.loads(paths["ablation"].read_text(encoding="utf-8"))

    assert readiness["ready"] is False
    assert readiness["training_eligible_record_count"] == 0
    assert "usage_authorization_incomplete" in readiness["blockers"]
    assert summary["status"] == "engineering_complete_human_training_gate_pending"
    assert summary["training_performed"] is False
    assert summary["automatic_action_gating_default_enabled"] is False
    assert ablation["offline_contract_closure"]["engineering_closed_loop_count"] == 1
    assert ablation["offline_contract_closure"]["reviewed_truth_closed_loop_count"] == 0
    assert all(experiment["metrics"] is None for experiment in ablation["experiments"])
