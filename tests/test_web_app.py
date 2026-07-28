from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from webui.app import _backend_plan, create_app


class FakeEngine:
    def __init__(self) -> None:
        self.started_with: dict[str, Any] | None = None
        self.stopped = False
        self.settings: dict[str, Any] = {}

    def snapshot(self) -> dict[str, Any]:
        return {
            "running": self.started_with is not None and not self.stopped,
            "status": "running" if self.started_with is not None and not self.stopped else "idle",
            "status_text": "分析中" if self.started_with is not None and not self.stopped else "等待开始",
        }

    def start(self, config: dict[str, Any]) -> None:
        self.started_with = config
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True

    def update_settings(self, values: dict[str, Any]) -> dict[str, Any]:
        self.settings.update(values)
        return self.snapshot()

    def request_recording(self, enabled: bool) -> dict[str, Any]:
        self.settings["recording"] = enabled
        return self.snapshot()

    def save_screenshot(self) -> Any:
        raise RuntimeError("当前还没有可保存的画面")

    def wait_for_frame(self, version: int, timeout: float = 2.0) -> tuple[int, bytes | None]:
        return version, None


def csrf_headers(client: Any) -> dict[str, str]:
    response = client.get("/api/options")
    assert response.status_code == 200
    return {"X-CSRF-Token": response.json["csrf_token"]}


def test_web_home_and_options_are_available() -> None:
    client = create_app(FakeEngine()).test_client()

    page = client.get("/")
    options = client.get("/api/options")

    assert page.status_code == 200
    assert "HYROX 动作分析台" in page.get_data(as_text=True)
    assert 'id="videoRepCount"' in page.get_data(as_text=True)
    assert 'id="poseValidRepCount"' in page.get_data(as_text=True)
    assert "完整动作周期" in page.get_data(as_text=True)
    assert 'id="voiceToggle"' in page.get_data(as_text=True)
    assert 'id="fingerToggle" type="checkbox">' in page.get_data(as_text=True)
    assert options.status_code == 200
    assert {item["value"] for item in options.json["actions"]} >= {"lunge", "wall_ball", "rowing"}
    assert len(options.json["samples"]) == 8
    assert {item["action"] for item in options.json["samples"]} == {
        "lunge", "wall_ball", "farmers_carry", "rowing", "skierg", "burpee_broad_jump", "sled_push", "sled_pull"
    }
    assert options.json["standards"]["rowing"]
    assert options.json["official_rules"]["wall_ball"]
    assert options.json["realtime"]["target_fps"] == 30
    assert options.json["realtime"]["camera_fps"] == 60
    assert options.json["realtime"]["max_requests_in_flight"] == 1
    assert options.json["realtime"]["inference_long_edge"] == 640
    assert options.json["realtime"]["jpeg_quality"] == 0.65
    assert options.json["realtime"]["max_pose_age_ms"] == 150
    assert options.json["realtime"]["hide_pose_after_ms"] == 300
    assert options.json["realtime"]["rendering"] == {
        "angle_text_fps": 12.0,
        "metrics_fps": 5.0,
        "stats_fps": 3.0,
        "timing_sample_capacity": 240,
    }
    assert options.json["realtime"]["camera"] == {
        "preferred_width": 640,
        "preferred_height": 480,
        "preferred_fps": 60.0,
        "fallback_fps": 30.0,
        "diagnostic_sample_fps": 5.0,
        "low_light_luma": 55.0,
        "fps_warning_ratio": 0.8,
        "interval_anomaly_ratio": 1.8,
        "duplicate_warning_ratio": 0.2,
    }
    assert options.json["realtime"]["local_first"] == {
        "web_pipeline": "local_browser",
        "desktop_pipeline": "local_device",
        "server_pose_fallback": True,
        "neural_prediction_enabled": False,
    }
    assert options.json["realtime"]["browser_pose"]["enabled"] is True
    assert options.json["realtime"]["browser_pose"]["worker_url"] == "/static/workers/pose_worker.js"
    assert options.json["realtime"]["browser_pose"]["model_url"] == "/assets/models/pose_landmarker_full.task"
    assert options.json["realtime"]["browser_pose"]["model_urls"] == {
        "lite": "/assets/models/pose_landmarker_lite.task",
        "full": "/assets/models/pose_landmarker_full.task",
    }
    assert options.json["realtime"]["browser_pose"]["model_preference"] == "auto"
    assert options.json["realtime"]["browser_pose"]["analysis_model"] == "full"
    assert options.json["realtime"]["browser_pose"]["benchmark_duration_ms"] == 3000
    assert options.json["realtime"]["browser_pose"]["lite_auto_approved"] is False
    assert options.json["realtime"]["browser_pose"]["max_inference_ms"] == 100
    assert options.json["realtime"]["browser_pose"]["slow_frame_limit"] == 12
    assert options.json["realtime"]["browser_pose"]["analysis_smoothing"] == {
        "profile": "responsive",
        "prediction_enabled": False,
    }
    display = options.json["realtime"]["browser_pose"]["display_smoothing"]
    assert display["profile"] == "ultra_responsive"
    assert display["min_cutoff"] == pytest.approx(2.2)
    assert display["beta"] == pytest.approx(0.12)
    assert display["max_raw_weight"] == pytest.approx(0.45)
    assert display["prediction_enabled"] is True
    prediction = options.json["realtime"]["browser_pose"]["display_prediction"]
    assert prediction["enabled"] is True
    assert prediction["mode"] == "constant_velocity"
    assert prediction["max_horizon_ms"] == pytest.approx(45)
    assert prediction["maximum_body_scale_displacement"] == pytest.approx(0.06)
    assert prediction["minimum_visibility"] == pytest.approx(0.70)
    assert prediction["velocity_decay"] == pytest.approx(0.85)
    assert prediction["disable_after_gap_ms"] == pytest.approx(100)
    model = client.get(options.json["realtime"]["browser_pose"]["model_url"])
    assert model.status_code == 200
    assert model.mimetype == "application/octet-stream"
    lite_model = client.get(options.json["realtime"]["browser_pose"]["model_urls"]["lite"])
    assert lite_model.status_code == 200
    assert lite_model.mimetype == "application/octet-stream"


@pytest.mark.skipif(
    not (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "hyrox"
        / "manifests"
        / "phone_records.json"
    ).is_file(),
    reason="local HYROX review dataset is not available",
)
def test_human_review_workspace_loads_real_queue_and_quick_review_materials() -> None:
    client = create_app(FakeEngine()).test_client()

    page = client.get("/review")
    bootstrap = client.get("/api/review/bootstrap")

    assert page.status_code == 200
    assert "HYROX 人工复核台" in page.get_data(as_text=True)
    assert "当前：单人复核" in page.get_data(as_text=True)
    assert "阶段与错误区间" in page.get_data(as_text=True)
    assert "ONI 主体复核" in page.get_data(as_text=True)
    assert "视角先验复核" in page.get_data(as_text=True)
    assert "ONI 错误真值" in page.get_data(as_text=True)
    assert "自动动作识别暂缓" in page.get_data(as_text=True)
    assert "这是 AI 候选，不是正确答案" in page.get_data(as_text=True)
    assert "载入候选次数" in page.get_data(as_text=True)
    assert bootstrap.status_code == 200
    assert bootstrap.json["protocol_version"] == "human_review_v1.0"
    assert bootstrap.json["review_policy"] == "single_human_review_sufficient_for_current_stage"
    assert bootstrap.json["formal_action_selection"] == "manual_only"
    assert bootstrap.json["automatic_action_gating_default_enabled"] is False
    assert len(bootstrap.json["records"]) == 30
    assert sum(1 for item in bootstrap.json["records"] if item["core"]) == 15
    assert len(bootstrap.json["oni_records"]) == 64
    assert len(bootstrap.json["view_prior_records"]) == 64
    assert bootstrap.json["error_truth_records"]
    assert bootstrap.json["tasks"]["core_fine_annotation"]["record_count"] == 15
    assert bootstrap.json["tasks"]["oni_subject_review"]["record_count"] == 32
    assert len(bootstrap.json["dashboard"]["task_completion"]) == 7
    assert bootstrap.json["tasks"]["remaining_rgb_fine_annotation"]["record_count"] == 15
    disagreement_task = bootstrap.json["tasks"]["high_disagreement_clip_review"]
    assert disagreement_task["frame_count"] == 2439
    assert disagreement_task["clip_count"] > 0
    skierg_two = next(
        item
        for item in bootstrap.json["records"]
        if item["record_id"] == "phone_skierg_002"
    )
    assert skierg_two["subject_id_suggestion"] == "subject_group_skierg_02"
    assert skierg_two["dataset_role_suggestion"] == "validation"
    assert bootstrap.json["labels"]["phase_labels_zh"]["contact"] == "后膝接触"
    assert bootstrap.json["labels"]["error_labels_zh"]["NO_ERROR"] == "无错误"

    bound = client.post(
        "/api/review/session",
        headers={"X-CSRF-Token": bootstrap.json["csrf_token"]},
        json={"role": "a", "reviewer_id": "quick_reviewer", "independence_confirmed": True},
    )
    assert bound.status_code == 200
    record_id = next(item["record_id"] for item in bootstrap.json["records"] if item["core"])
    detail = client.get(f"/api/review/records/{record_id}?role=a&reviewer_id=quick_reviewer&quick=1")
    assert detail.status_code == 200
    assert detail.json["blind_complete"] is False
    assert detail.json["proposal"] is not None
    assert detail.json["proposal"]["core_annotations"]["reps"]
    assert detail.json["proposal"]["core_annotations"]["reps"][0]["errors"] is not None
    assert detail.json["record"]["source_filename"]
    assert detail.json["record"]["recording_intent"]
    assert detail.json["record"]["subject_id_suggestion"]
    assert detail.json["record"]["disagreement_clips"]
    noncore_detail = client.get(
        "/api/review/records/phone_rowing_001"
        "?role=a&reviewer_id=quick_reviewer&quick=1"
    )
    assert noncore_detail.status_code == 200
    noncore_proposal = noncore_detail.json["proposal"]["core_annotations"]
    assert noncore_proposal["proposal_semantics"] == "analysis_cycle"
    assert noncore_proposal["is_ground_truth"] is False
    assert len(noncore_proposal["reps"]) == 8
    assert noncore_proposal["reps"][0]["phases"]
    assert noncore_proposal["reps"][0]["events"]
    oni = bootstrap.json["oni_records"][0]
    oni_detail = client.get(f"/api/review/oni/{oni['record_id']}/{oni['modality']}")
    assert oni_detail.status_code == 200
    assert len(oni_detail.json["checkpoints"]) == 24
    assert client.get(oni_detail.json["record"]["preview_url"]).status_code == 200
    assert client.get("/api/review/agreement").status_code == 403


def test_human_review_client_explains_and_imports_ai_candidates() -> None:
    source = Path("webui/static/review.js").read_text(encoding="utf-8")

    assert 'FOOT_DESYNCHRONIZED: "双脚起跳或落地不同步"' in source
    assert 'NO_KNEE_CONTACT: "后膝未触地"' in source
    assert "function renderProposalRep" in source
    assert "function importProposal" in source
    assert "phase_error_intervals" in source
    assert "formal action" not in source
    assert 'toast("动作已人工切换；请核对阶段和事件选项")' in source
    assert "当前没有逐次 AI proposal" in source
    assert "已载入草稿，请逐项人工核对" in source
    assert "当前算法没有检出候选次数/分析周期" in source


def test_human_review_save_is_role_bound_audited_and_frame_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import webui.app as web_app

    dataset_root = tmp_path / "datasets" / "hyrox"
    (dataset_root / "manifests").mkdir(parents=True)
    (dataset_root / "annotations").mkdir()
    (dataset_root / "reports").mkdir()
    record = {
        "record_id": "phone_lunge_001",
        "source_file": "raw/phone_rgb/test.mp4",
        "source_filename": "隐藏意图.mp4",
        "action": "lunge",
        "subject_id": "subject_pending",
        "camera_view": "side",
        "recording_intent": "error",
        "recording_intent_raw": "隐藏",
        "expected_errors_unverified": ["NO_KNEE_CONTACT"],
        "target_athlete": {"track_id": "target_athlete_001"},
        "video": {
            "decoded_frame_count": 10,
            "fps": 30.0,
            "duration_seconds": 0.333333,
            "resolution": "720x1280",
        },
    }
    (dataset_root / "manifests" / "phone_records.json").write_text(
        json.dumps({"records": [record]}, ensure_ascii=False),
        encoding="utf-8",
    )
    for filename in (
        "action_segments_v1.json",
        "core_rep_phase_event_error_v1.json",
        "object_scene_evidence_v1.json",
        "scoring_correction_v1.json",
    ):
        (dataset_root / "annotations" / filename).write_text('{"records":[]}', encoding="utf-8")
    (dataset_root / "reports" / "round9_active_review_queue_v1.json").write_text(
        '{"records":[{"record_id":"phone_lunge_001","priority":1}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(web_app, "PROJECT_ROOT", tmp_path)
    client = web_app.create_app(FakeEngine()).test_client()
    bootstrap = client.get("/api/review/bootstrap")
    headers = {"X-CSRF-Token": bootstrap.json["csrf_token"]}
    assert client.post(
        "/api/review/session",
        headers=headers,
        json={"role": "a", "reviewer_id": "reviewer_test", "independence_confirmed": True},
    ).status_code == 200

    review = {
        "quick_review": {
            "status": "complete",
            "action": "lunge",
            "target_status": "correct",
            "video_usability": "usable",
            "usable_start_frame": 0,
            "usable_end_frame": 9,
            "segments": [{"label": "target_action", "start_frame": 0, "end_frame": 9}],
            "reps": [
                {
                    "rep_id": "rep_001",
                    "start_frame": 0,
                    "end_frame": 9,
                    "validity": "VALID",
                    "phase_gap_reason": "5–9 帧为尚未归类的恢复阶段",
                }
            ],
            "phase_error_intervals": [
                {
                    "rep_id": "rep_001",
                    "start_frame": 0,
                    "end_frame": 4,
                    "phase": "descent",
                    "error_code": "NO_KNEE_CONTACT",
                    "observability": "OBSERVABLE",
                }
            ],
            "events": [{"event_type": "bottom_reached", "frame_index": 5}],
            "notes": "可供后续处理",
        }
    }
    saved = client.put(
        "/api/review/records/phone_lunge_001",
        headers=headers,
        json={"role": "a", "reviewer_id": "reviewer_test", "review": review},
    )
    assert saved.status_code == 200
    output = json.loads(
        (dataset_root / "reviews" / "human_v1" / "reviewer_a" / "records" / "phone_lunge_001.json").read_text(
            encoding="utf-8"
        )
    )
    assert output["reviewer_role"] == "reviewer_a"
    assert output["review"]["quick_review"]["events"][0]["timestamp_ms"] == pytest.approx(166.667)
    assert output["audit_log"][0]["changes"]
    assert output["eligibility_overrides_written"] is False
    refreshed = client.get("/api/review/bootstrap")
    assert refreshed.json["records"][0]["reviewer_a"]["complete"] is True
    exported = client.get("/api/review/export?scope=a")
    assert exported.status_code == 200
    with zipfile.ZipFile(io.BytesIO(exported.data)) as archive:
        names = set(archive.namelist())
        assert "human_v1/reviewer_a/codex_handoff.json" in names
        assert "human_v1/reviewer_a/records/phone_lunge_001.json" in names
        assert not any(name.startswith("human_v1/agreement/") for name in names)

    incomplete = {
        "quick_review": {
            "status": "complete",
            "action": "lunge",
            "usable_start_frame": 0,
            "usable_end_frame": None,
            "segments": [{"label": "target_action", "start_frame": 0, "end_frame": None}],
            "reps": [],
            "phase_error_intervals": [],
            "events": [],
            "notes": "",
        }
    }
    partial_saved = client.put(
        "/api/review/records/phone_lunge_001",
        headers=headers,
        json={
            "role": "a",
            "reviewer_id": "reviewer_test",
            "base_revision": 1,
            "review": incomplete,
        },
    )
    assert partial_saved.status_code == 200
    assert partial_saved.json["revision"] == 2

    record_path = (
        dataset_root
        / "reviews"
        / "human_v1"
        / "reviewer_a"
        / "records"
        / "phone_lunge_001.json"
    )
    broken = json.loads(record_path.read_text(encoding="utf-8"))
    broken["revision"] = 3
    record_path.write_text(
        json.dumps(broken, ensure_ascii=False),
        encoding="utf-8",
    )
    repaired = client.put(
        "/api/review/records/phone_lunge_001",
        headers=headers,
        json={
            "role": "a",
            "reviewer_id": "reviewer_test",
            "base_revision": 3,
            "review": incomplete,
        },
    )
    assert repaired.status_code == 200
    repaired_payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert [item["revision"] for item in repaired_payload["audit_log"]] == [1, 2, 3, 4]

    review["quick_review"]["events"][0]["frame_index"] = 10
    out_of_range_saved = client.put(
        "/api/review/records/phone_lunge_001",
        headers=headers,
        json={"role": "a", "reviewer_id": "reviewer_test", "base_revision": 4, "review": review},
    )
    assert out_of_range_saved.status_code == 200
    assert out_of_range_saved.json["revision"] == 5


def test_single_reviewer_can_save_independent_oni_depth_subject_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import webui.app as web_app

    dataset_root = tmp_path / "datasets" / "hyrox"
    for directory in ("manifests", "annotations", "reports/round11_subject_previews/oni_test_001", "oni_tracks/oni_test_001"):
        (dataset_root / directory).mkdir(parents=True, exist_ok=True)
    (dataset_root / "manifests" / "phone_records.json").write_text('{"records":[]}', encoding="utf-8")
    (dataset_root / "manifests" / "oni_records.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "record_id": "oni_test_001",
                        "action": "lunge",
                        "camera_view": "side",
                        "recording_intent_code": "standard",
                        "subject_id": "subject_pending",
                        "expected_errors_unverified": ["NO_KNEE_CONTACT"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (dataset_root / "reports" / "oni_subject_audit_v1.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "record_id": "oni_test_001",
                        "modalities": {
                            "depth": {"sampled_checkpoint_count": 2},
                            "ir": {"sampled_checkpoint_count": 2},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    for modality in ("depth", "ir"):
        rows = [
            {
                "modality": modality,
                "source_frame_index": frame,
                "bbox_px": [1, 2, 3, 4],
                "confidence": 0.5,
            }
            for frame in (1, 21)
        ]
        (dataset_root / "oni_tracks" / "oni_test_001" / f"{modality}_target_proposals.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        (dataset_root / "reports" / "round11_subject_previews" / "oni_test_001" / f"{modality}_subject_proposals.jpg").write_bytes(
            b"\xff\xd8\xff\xd9"
        )
    for filename in (
        "action_segments_v1.json",
        "core_rep_phase_event_error_v1.json",
        "object_scene_evidence_v1.json",
        "scoring_correction_v1.json",
    ):
        (dataset_root / "annotations" / filename).write_text('{"records":[]}', encoding="utf-8")
    (dataset_root / "reports" / "round9_active_review_queue_v1.json").write_text('{"records":[]}', encoding="utf-8")

    monkeypatch.setattr(web_app, "PROJECT_ROOT", tmp_path)
    client = web_app.create_app(FakeEngine()).test_client()
    bootstrap = client.get("/api/review/bootstrap")
    headers = {"X-CSRF-Token": bootstrap.json["csrf_token"]}
    assert client.post(
        "/api/review/session",
        headers=headers,
        json={"role": "a", "reviewer_id": "oni_reviewer"},
    ).status_code == 200
    assert len(bootstrap.json["oni_records"]) == 2

    review = {
        "status": "complete",
        "overall_target_status": "correct",
        "same_subject_throughout": "yes",
        "observability": "OBSERVABLE",
        "checkpoints": [
            {
                "frame_index": frame,
                "target_status": "correct",
                "bbox_status": "correct",
                "surface_reliable": "yes",
                "notes": "",
            }
            for frame in (1, 21)
        ],
        "notes": "两个检查点均为同一目标运动者",
    }
    saved = client.put(
        "/api/review/oni/oni_test_001/depth",
        headers=headers,
        json={"reviewer_id": "oni_reviewer", "review": review},
    )
    assert saved.status_code == 200
    output_path = dataset_root / "reviews" / "human_v1" / "reviewer_a" / "oni_records" / "oni_test_001__depth.json"
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["review_policy"] == "single_human_review_sufficient_for_current_stage"
    assert output["review"]["same_subject_throughout"] == "yes"
    assert output["eligibility_overrides_written"] is False
    refreshed = client.get("/api/review/bootstrap")
    depth_item = next(item for item in refreshed.json["oni_records"] if item["modality"] == "depth")
    ir_item = next(item for item in refreshed.json["oni_records"] if item["modality"] == "ir")
    assert depth_item["complete"] is True
    assert ir_item["complete"] is False

    view_detail = client.get("/api/review/oni/oni_test_001/depth?mode=view_prior")
    assert view_detail.status_code == 200
    assert view_detail.json["review_mode"] == "view_prior"
    assert view_detail.json["saved_review"] is None
    view_review = {
        **review,
        "review_mode": "view_prior",
        "confirmed_view": "side",
        "action_usability": "usable",
        "usable_start_frame": 1,
        "usable_end_frame": 21,
        "full_body_visibility": "visible",
        "floor_visibility": "visible",
        "equipment_visibility": "partial",
        "identity_switch_intervals": [],
        "observability_items": [
            {
                "item_code": code,
                "status": "OBSERVABLE",
                "reason": "",
                "start_frame": 1,
                "end_frame": 21,
                "evidence_frames": [1],
                "notes": "",
            }
            for code in (
                "rear_knee_contact",
                "hip_extension",
                "leg_alternation",
                "extra_steps",
                "trunk_angle",
                "left_right_symmetry",
            )
        ],
    }
    view_saved = client.put(
        "/api/review/oni/oni_test_001/depth",
        headers=headers,
        json={"reviewer_id": "oni_reviewer", "base_revision": 0, "review": view_review},
    )
    assert view_saved.status_code == 200
    assert output_path.exists()
    view_output_path = (
        dataset_root
        / "reviews"
        / "human_v1"
        / "reviewer_a"
        / "view_prior_records"
        / "oni_test_001__depth.json"
    )
    assert view_output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8"))["revision"] == 1
    view_output = json.loads(view_output_path.read_text(encoding="utf-8"))
    assert view_output["artifact_type"] == "oni_modality_view_prior_human_review"
    assert view_output["eligibility"]["view_policy_calibration_eligible"] is True
    view_review["observability_items"][0].update(
        {"status": "UNOBSERVABLE", "reason": "", "asserted_error": True}
    )
    unrestricted_draft = client.put(
        "/api/review/oni/oni_test_001/depth",
        headers=headers,
        json={"reviewer_id": "oni_reviewer", "base_revision": 1, "review": view_review},
    )
    assert unrestricted_draft.status_code == 200
    assert unrestricted_draft.json["revision"] == 2
    refreshed = client.get("/api/review/bootstrap")
    view_depth = next(item for item in refreshed.json["view_prior_records"] if item["modality"] == "depth")
    assert view_depth["complete"] is True
    independent = client.get("/api/review/export/view_observability_review_v1")
    assert independent.status_code == 200
    exported_view = json.loads(independent.data)
    exported_depth = next(
        item
        for item in exported_view["records"]
        if item["record_id"] == "oni_test_001" and item["modality"] == "depth"
    )
    assert exported_depth["confirmed_view"] == "side"
    assert exported_depth["training_eligible"] is False

    error_detail = client.get("/api/review/oni/oni_test_001/depth?mode=error_truth")
    assert error_detail.status_code == 200
    error_review = {
        **review,
        "review_mode": "error_truth",
        "error_truth_items": [
            {
                "error_code": "NO_KNEE_CONTACT",
                "truth_status": "confirmed",
                "observability": "OBSERVABLE",
                "evidence_frames": [1],
                "notes": "后膝在当前模态中可确认未触地",
            }
        ],
    }
    error_saved = client.put(
        "/api/review/oni/oni_test_001/depth",
        headers=headers,
        json={"reviewer_id": "oni_reviewer", "base_revision": 0, "review": error_review},
    )
    assert error_saved.status_code == 200
    error_output_path = (
        dataset_root
        / "reviews"
        / "human_v1"
        / "reviewer_a"
        / "error_truth_records"
        / "oni_test_001__depth.json"
    )
    assert error_output_path.exists()
    error_export = client.get("/api/review/export/oni_error_truth_review_v1")
    assert error_export.status_code == 200
    exported_errors = json.loads(error_export.data)
    exported_error = next(
        item
        for item in exported_errors["records"]
        if item["record_id"] == "oni_test_001" and item["modality"] == "depth"
    )
    assert exported_error["error_truth_items"][0]["truth_status"] == "confirmed"
    assert exported_error["training_eligible"] is False


def test_browser_realtime_client_uses_video_frame_callback_and_single_in_flight_request() -> None:
    source = Path("webui/static/app.js").read_text(encoding="utf-8")

    assert "requestInFlight: false" in source
    assert "if (ui.requestInFlight || ui.socket.bufferedAmount" in source
    assert "ui.requestInFlight = true;" in source
    assert "finishFrameRequest(frameId)" in source
    assert "video.requestVideoFrameCallback(onVideoFrame)" in source
    assert "requestAnimationFrame(fallbackLoop)" in source
    assert "scheduleNextCapture" not in source
    assert source.index("renderPoseForVideoFrame(frameMeta, now)") < source.index("void captureLatestFrame(frameMeta)")
    for field in (
        "sessionId", "frameId", "presentedFrames", "mediaTime", "presentationTime",
        "expectedDisplayTime", "captureTime", "width", "height",
    ):
        assert field in source
    assert "lastRenderedPoseFrameId" in source
    assert "result.session_id === ui.activeRealtimeSessionId" in source
    assert "result.run_id === ui.activeRealtimeRunId" in source
    assert "lastDiscardedFrameId" in source
    assert "ui.requestTimeout = setTimeout" in source
    assert "inference_long_edge" in source
    assert 'ui.realtimeConfig.jpeg_quality ?? 0.65' in source
    assert "new TextEncoder().encode(JSON.stringify" in source
    assert "now - ui.lastResultAt" in source
    assert "hideAfter * 0.8" in source
    assert "now - captureMs" not in source
    assert 'mode === "camera" ? "未连接" : "本机处理"' in source
    assert 'ui.sourceMode === "camera" && ui.running && !ui.manualStop' in source
    assert '"sample-cache": "预计算示例结果"' in source
    assert "Math.round(angle.value)}° 3D" in source


def test_browser_pose_worker_uses_latest_frame_slot_and_landmark_protocol() -> None:
    source = Path("webui/static/app.js").read_text(encoding="utf-8")
    worker = Path("webui/static/workers/pose_worker.js").read_text(encoding="utf-8")
    display_filter = Path("webui/static/workers/display_pose_filter.mjs").read_text(encoding="utf-8")

    assert "poseWorkerBusy: false" in source
    assert "poseWorkerPending: null" in source
    assert "closePoseTransfer(ui.poseWorkerPending)" in source
    assert 'transferMode: "video-frame"' in source
    assert 'transferMode: "image-bitmap"' in source
    assert 'type: "pose_frame"' in source
    assert "rawImageLandmarks" in source
    assert "rawWorldLandmarks" in source
    assert "canvas.toDataURL" not in source
    assert 'type: "benchmark_complete"' in worker
    assert 'type: "switch_model"' in source
    assert "selectAutoModel" in worker
    assert 'runningMode: "VIDEO"' in worker
    assert "detectForVideo(input, timestampMs)" in worker
    assert "outputSegmentationMasks: false" in worker
    assert "DisplayPoseFilter" in worker
    assert "new OneEuroFilter" in display_filter
    assert "imageRawHistory" in display_filter
    assert "worldRawHistory" in display_filter
    assert "#rawWeight" in display_filter
    assert "EXTREMITY_LANDMARKS" in display_filter
    assert "CORE_LANDMARKS" in display_filter
    assert "FACE_LANDMARKS" in display_filter
    assert "rawImageLandmarks" in source
    assert "message.imageLandmarks" in source
    assert "display_filter: message.displayFilter" in source
    assert "image_landmarks: serializeLocalLandmarks(message.rawImageLandmarks)" in source
    assert "world_landmarks: serializeLocalLandmarks(message.rawWorldLandmarks)" in source
    assert "image_landmarks: serializeLocalLandmarks(message.imageLandmarks)" not in source
    assert "drawSkeleton(result, opacity, renderStart, prediction.landmarks)" in source
    assert "ui.latestResult.keypoints = prediction" not in source
    assert "prediction_horizon_ms" not in source
    assert "prediction_point_count" not in source
    assert "prediction_clamped_point_count" not in source
    assert 'local_first?.server_pose_fallback !== false' in source
    assert 'type: "camera_diagnostics"' in source
    assert 'message_type == "camera_diagnostics"' in Path("webui/app.py").read_text(
        encoding="utf-8"
    )


def test_browser_render_cache_keeps_finger_landmarks_with_pose_prediction() -> None:
    source = Path("webui/static/app.js").read_text(encoding="utf-8")

    assert "const supplementalFingerLandmarkNames" in source
    assert "const renderLandmarkNames = [...poseLandmarkNames, ...supplementalFingerLandmarkNames]" in source
    assert "new Uint8Array(renderLandmarkNames.length)" in source
    assert "new Map(renderLandmarkNames.map" in source
    assert "new Int16Array(renderLandmarkNames.length * 2)" in source
    assert "if (displayLandmarks) {" in source
    assert "index < poseLandmarkNames.length" in source
    assert "index < renderLandmarkNames.length" in source


def test_file_videos_analyze_every_frame_at_the_source_playback_rate() -> None:
    source = Path("webui/app.py").read_text(encoding="utf-8")

    assert 'mode="one-euro"' in source
    assert "sample_frame_step" not in source
    assert "capture.grab()" not in source
    assert 'config["source_mode"] != "camera"' in source
    assert "remaining = (1.0 / source_fps) - elapsed" in source
    assert "self._stop_event.wait(remaining)" in source


def test_camera_analysis_can_be_started_and_stopped_from_api() -> None:
    engine = FakeEngine()
    client = create_app(engine).test_client()
    headers = csrf_headers(client)

    response = client.post(
        "/api/start",
        headers=headers,
        json={
            "source_mode": "camera",
            "camera_index": 0,
            "action": "wall_ball",
            "camera_view": "front",
            "sensitivity": "medium",
            "backend": "auto",
            "landmark_profile": "full",
            "mirror": True,
        },
    )

    assert response.status_code == 200
    assert engine.started_with is not None
    assert engine.started_with["action"] == "wall_ball"
    assert engine.started_with["source_name"] == "服务器摄像头 0"
    assert client.post("/api/stop", headers=headers).status_code == 200
    assert engine.stopped is True


def test_camera_analysis_rejects_experimental_backend_in_product_api() -> None:
    engine = FakeEngine()
    client = create_app(engine).test_client()
    headers = csrf_headers(client)

    response = client.post(
        "/api/start",
        headers=headers,
        json={
            "source_mode": "camera",
            "camera_index": 0,
            "action": "lunge",
            "backend": "yolo-mediapipe",
        },
    )

    assert response.status_code == 400
    assert engine.started_with is None
    assert "无效的识别后端" in response.json["error"]


def test_start_rejects_unknown_action() -> None:
    client = create_app(FakeEngine()).test_client()
    headers = csrf_headers(client)

    response = client.post(
        "/api/start",
        headers=headers,
        json={"source_mode": "camera", "action": "unknown_action"},
    )

    assert response.status_code == 400
    assert "无效的动作" in response.json["error"]


def test_sample_action_and_video_are_linked_by_the_api() -> None:
    engine = FakeEngine()
    client = create_app(engine).test_client()
    options = client.get("/api/options")
    sample = next(item for item in options.json["samples"] if item["action"] == "rowing")
    headers = {"X-CSRF-Token": options.json["csrf_token"]}

    started = client.post(
        "/api/start",
        headers=headers,
        json={"source_mode": "sample", "video_id": sample["id"], "action": "rowing"},
    )
    mismatch = client.post(
        "/api/start",
        headers=headers,
        json={"source_mode": "sample", "video_id": sample["id"], "action": "lunge"},
    )

    assert started.status_code == 200
    assert engine.started_with is not None
    assert engine.started_with["action"] == "rowing"
    assert mismatch.status_code == 400
    assert "不一致" in mismatch.json["error"]


def test_backend_plan_limits_internal_tracking_to_trusted_lunge_sample() -> None:
    assert _backend_plan({"source_mode": "sample", "action": "lunge", "backend": "auto"}) == (
        "auto",
        "tracking",
    )
    assert _backend_plan({"source_mode": "camera", "action": "lunge", "backend": "auto"}) == (
        "auto",
        "tracking",
    )
    assert _backend_plan(
        {
            "source_mode": "sample",
            "action": "lunge",
            "backend": "rtmw-wholebody",
        }
    ) == ("rtmw-wholebody", "tracking")
    assert _backend_plan(
        {
            "source_mode": "sample",
            "action": "lunge",
            "backend": "mediapipe",
            "bundled_sample_tracking": True,
        }
    ) == ("yolo-mediapipe", "tracking")


def test_web_product_page_only_offers_mediapipe_pose() -> None:
    response = create_app(FakeEngine()).test_client().get("/")

    assert response.status_code == 200
    assert b'value="mediapipe"' in response.data
    assert b'<select id="poseModelSelect">' in response.data
    assert b'<option value="auto" selected>' in response.data
    assert b'value="yolo-mediapipe"' not in response.data
    assert b'value="yolo-pose"' not in response.data
    assert b'value="rtmw-wholebody"' not in response.data


def test_server_screenshot_is_disabled_for_privacy() -> None:
    client = create_app(FakeEngine()).test_client()
    headers = csrf_headers(client)

    response = client.post("/api/screenshot", headers=headers)

    assert response.status_code == 410
    assert response.json["code"] == "server_screenshot_disabled"


def test_server_recording_is_disabled_for_privacy() -> None:
    client = create_app(FakeEngine()).test_client()
    headers = csrf_headers(client)

    response = client.post("/api/record", headers=headers, json={"enabled": True})

    assert response.status_code == 410
    assert response.json["code"] == "recording_disabled"


def test_upload_checks_actual_media_content() -> None:
    client = create_app(FakeEngine()).test_client()
    headers = csrf_headers(client)

    response = client.post(
        "/api/upload",
        headers=headers,
        data={"video": (io.BytesIO(b"not a real video"), "pretend.mp4")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.json["code"] == "invalid_media"
    assert client.delete("/api/session", headers=headers).status_code == 200


def test_shared_access_link_sets_cookie_and_protects_api() -> None:
    client = create_app(FakeEngine(), access_token="unit-secret").test_client()

    assert client.get("/").status_code == 401
    assert client.get("/api/options").status_code == 401

    accepted = client.get("/?access_token=unit-secret")
    assert accepted.status_code == 302
    assert client.get("/").status_code == 200
    assert client.get("/api/options").status_code == 200


def test_health_check_does_not_require_shared_access_token() -> None:
    client = create_app(FakeEngine(), access_token="unit-secret").test_client()

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json == {"status": "ok"}


def test_mutating_api_rejects_missing_csrf_token() -> None:
    client = create_app(FakeEngine()).test_client()
    assert client.get("/api/options").status_code == 200

    response = client.post("/api/stop")

    assert response.status_code == 403
    assert response.json["code"] == "csrf_failed"


def test_security_headers_and_cookie_attributes() -> None:
    client = create_app(FakeEngine()).test_client()

    response = client.get("/", base_url="https://pose.example.test")

    assert response.headers["Permissions-Policy"] == "camera=(self), microphone=(), geolocation=()"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert "worker-src 'self' blob:" in response.headers["Content-Security-Policy"]
    assert "'wasm-unsafe-eval'" in response.headers["Content-Security-Policy"]
    cookie = response.headers.get("Set-Cookie", "")
    assert "HttpOnly" in cookie
    assert "Secure" in cookie
    assert "SameSite=Lax" in cookie


def test_anonymous_browser_sessions_are_isolated() -> None:
    engines: list[FakeEngine] = []

    def make_engine(_: str) -> FakeEngine:
        value = FakeEngine()
        engines.append(value)
        return value

    app = create_app(engine_factory=make_engine)
    client_a = app.test_client()
    client_b = app.test_client()
    headers_a = csrf_headers(client_a)
    headers_b = csrf_headers(client_b)
    payload = {
        "source_mode": "camera",
        "camera_index": 0,
        "action": "lunge",
        "camera_view": "side",
        "sensitivity": "medium",
        "backend": "auto",
        "landmark_profile": "full",
    }

    assert client_a.post("/api/start", headers=headers_a, json=payload).status_code == 200
    assert client_b.post("/api/start", headers=headers_b, json={**payload, "action": "rowing"}).status_code == 200
    assert len(engines) == 2
    assert engines[0].started_with["action"] == "lunge"
    assert engines[1].started_with["action"] == "rowing"

    assert client_a.post("/api/stop", headers=headers_a).status_code == 200
    assert engines[0].stopped is True
    assert engines[1].stopped is False
