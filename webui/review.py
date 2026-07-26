from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import re
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Blueprint, Response, g, jsonify, render_template, request, send_file


PROTOCOL_VERSION = "human_review_v1.0"
REVIEW_SCHEMA_VERSION = 1
VALID_ROLES = {"a": "reviewer_a", "b": "reviewer_b"}
CORE_ACTIONS = {"burpee_broad_jump", "lunge", "wall_ball"}
ONI_MODALITIES = ("depth", "ir")
TIMELINE_LABELS = (
    "idle",
    "setup",
    "target_action",
    "transition",
    "unknown_motion",
    "target_out_of_frame",
)
VALIDITY_VALUES = ("VALID", "NO_REP", "UNSURE")
OBSERVABILITY_VALUES = ("OBSERVABLE", "PARTIAL", "UNOBSERVABLE", "UNKNOWN")
ACTION_LABELS = {
    "burpee_broad_jump": "波比跳远",
    "lunge": "负重箭步蹲",
    "wall_ball": "投掷药球",
    "farmers_carry": "农夫行走",
    "rowing": "划船机",
    "skierg": "滑雪机",
    "sled_push": "推雪橇",
    "sled_pull": "拉雪橇",
    "unknown_ood": "未知 / OOD",
}
PHASES_BY_ACTION = {
    "burpee_broad_jump": ["hands_down", "chest_down", "takeoff", "flight", "landing", "stabilization"],
    "lunge": ["descent", "bottom", "contact", "ascent", "stand"],
    "wall_ball": ["descent", "bottom", "ascent", "stand", "release", "recovery"],
}
EVENTS_BY_ACTION = {
    "burpee_broad_jump": [
        "hands_down",
        "chest_contact_candidate",
        "left_takeoff",
        "right_takeoff",
        "takeoff_candidate",
        "left_landing",
        "right_landing",
        "landing_candidate",
        "stabilized",
    ],
    "lunge": ["rep_start", "bottom_reached", "rear_knee_contact_candidate", "full_extension"],
    "wall_ball": ["rep_start", "bottom_reached", "ball_release_candidate", "recovery"],
}
ERRORS_BY_ACTION = {
    "burpee_broad_jump": ["FOOT_DESYNCHRONIZED", "HANDS_FEET_TOO_FAR", "NO_CHEST_CONTACT", "EXTRA_STEP"],
    "lunge": ["NO_KNEE_CONTACT", "SAME_LEG_CONSECUTIVE", "HIP_NOT_EXTENDED", "EXTRA_STEP"],
    "wall_ball": ["NOT_DEEP_ENOUGH", "HEEL_RISE"],
}
PHASE_LABELS_ZH = {
    "hands_down": "手撑地",
    "chest_down": "胸部下降/触地",
    "takeoff": "起跳",
    "flight": "腾空",
    "landing": "落地",
    "stabilization": "落地稳定",
    "descent": "下降",
    "bottom": "最低点",
    "contact": "后膝接触",
    "ascent": "站起",
    "stand": "完全伸展",
    "release": "出球",
    "recovery": "接球/恢复",
}
ERROR_LABELS_ZH = {
    "NO_ERROR": "无错误",
    "FOOT_DESYNCHRONIZED": "双脚起跳或落地不同步",
    "HANDS_FEET_TOO_FAR": "手脚距离过远",
    "NO_CHEST_CONTACT": "胸部未触地",
    "EXTRA_STEP": "出现额外补步或碎步",
    "NO_KNEE_CONTACT": "后膝未触地",
    "SAME_LEG_CONSECUTIVE": "同一条腿连续迈步",
    "HIP_NOT_EXTENDED": "髋部未完全伸展",
    "NOT_DEEP_ENOUGH": "下蹲深度不足",
    "HEEL_RISE": "脚跟抬起",
    "OTHER": "其他错误（请备注）",
    "UNSURE": "无法确认",
}
ONI_LABELS_ZH = {
    "modalities": {"depth": "深度 Depth", "ir": "红外 IR"},
    "status": {"draft": "草稿", "complete": "已完成", "blocked": "暂不可用"},
    "target_status": {
        "unreviewed": "尚未复核",
        "correct": "候选框是目标运动者",
        "wrong": "候选框选错人",
        "missing": "目标运动者未被框出",
        "multiple_people_unsure": "多人重叠，无法确认",
    },
    "bbox_status": {
        "unreviewed": "尚未复核",
        "correct": "框位置合适",
        "too_large": "框过大",
        "too_small": "框过小",
        "wrong_target": "框到其他人",
        "missing": "没有候选框",
    },
    "same_subject": {"yes": "始终是同一目标", "no": "发生目标切换", "unsure": "无法确认"},
    "yes_no_unsure": {"yes": "是", "no": "否", "unsure": "无法确认"},
}
SPECIAL_FOCUS = {
    "phone_burpee_broad_jump_002": [
        "逐帧查看 195–220 帧",
        "分别裁决 205、210 帧的 TARGET_IDENTITY_SWITCH proposal",
        "若确认切换，记录准确起止帧；若拒绝，记录外观/姿态关联低为何不构成换人",
    ],
    "phone_sled_push_005": [
        "确认 0–294 帧是否确实没有可用目标轨迹",
        "检查 295–300、305、306–309、345 帧的重入、丢失与 stale propagation",
        "检查 550 帧前后两个 person candidate 是否为同一人，并裁决重初始化边界",
    ],
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return copy.deepcopy(default)
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _safe_reviewer_id(value: Any) -> str:
    text = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.\-\u4e00-\u9fff]{2,64}", text):
        raise ValueError("复核者 ID 需为 2–64 位中文、字母、数字、点、下划线或连字符")
    return text


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else key
            flattened.update(_flatten(value[key], child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            flattened.update(_flatten(item, f"{prefix}[{index}]"))
        if not value:
            flattened[prefix] = []
    else:
        flattened[prefix] = value
    return flattened


def _diff(old: Any, new: Any) -> list[dict[str, Any]]:
    before = _flatten(old or {})
    after = _flatten(new or {})
    changes: list[dict[str, Any]] = []
    for key in sorted(set(before) | set(after)):
        if before.get(key) != after.get(key):
            changes.append({"path": key, "old_value": before.get(key), "new_value": after.get(key)})
    return changes


def _walk_frame_values(value: Any, frame_count: int, fps: float, path: str = "") -> None:
    if isinstance(value, dict):
        start = value.get("start_frame")
        end = value.get("end_frame")
        if start not in (None, "") and end not in (None, "") and int(start) > int(end):
            raise ValueError(f"{path or '记录'}：start_frame 不能大于 end_frame")
        for key, child in list(value.items()):
            child_path = f"{path}.{key}" if path else key
            if key in {"frame_index", "start_frame", "end_frame", "left_frame_index", "right_frame_index"}:
                if child in (None, ""):
                    continue
                try:
                    frame = int(child)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{child_path} 必须是整数帧号") from exc
                if frame < 0 or frame >= frame_count:
                    raise ValueError(f"{child_path}={frame} 超出 0–{frame_count - 1}")
                value[key] = frame
                if key == "frame_index":
                    value["timestamp_ms"] = round(frame / fps * 1000.0, 3)
            else:
                _walk_frame_values(child, frame_count, fps, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_frame_values(child, frame_count, fps, f"{path}[{index}]")


def _validate_fine_annotations(review: dict[str, Any], action: str) -> None:
    quick = review.get("quick_review") or {}
    intervals = quick.get("phase_error_intervals") or []
    valid_phases = set(PHASES_BY_ACTION.get(action, []))
    valid_errors = set(ERRORS_BY_ACTION.get(action, [])) | {"NO_ERROR", "OTHER", "UNSURE"}
    rep_ids = {str(item.get("rep_id", "")).strip() for item in (quick.get("reps") or [])}
    if str(quick.get("status", "")) == "complete" and (
        not rep_ids or not intervals or not (quick.get("events") or [])
    ):
        raise ValueError("核心动作标记完成前必须填写逐次边界、阶段错误区间和关键事件")
    for index, item in enumerate(intervals):
        if not isinstance(item, dict):
            raise ValueError(f"阶段错误区间 {index + 1} 必须是对象")
        phase = str(item.get("phase", ""))
        error_code = str(item.get("error_code", ""))
        rep_id = str(item.get("rep_id", "")).strip()
        if phase not in valid_phases:
            raise ValueError(f"阶段错误区间 {index + 1} 的阶段不属于动作 {action}")
        if error_code not in valid_errors:
            raise ValueError(f"阶段错误区间 {index + 1} 的错误选项无效")
        if rep_id and rep_ids and rep_id not in rep_ids:
            raise ValueError(f"阶段错误区间 {index + 1} 引用了不存在的次数 {rep_id}")


def _record_path(review_root: Path, role: str, record_id: str) -> Path:
    return review_root / VALID_ROLES[role] / "records" / f"{record_id}.json"


def _oni_record_path(review_root: Path, record_id: str, modality: str) -> Path:
    return review_root / VALID_ROLES["a"] / "oni_records" / f"{record_id}__{modality}.json"


def _load_oni_checkpoints(dataset_root: Path, record_id: str, modality: str) -> list[dict[str, Any]]:
    path = dataset_root / "oni_tracks" / record_id / f"{modality}_target_proposals.jsonl"
    checkpoints: list[dict[str, Any]] = []
    if not path.exists():
        return checkpoints
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                checkpoints.append(json.loads(line))
    return checkpoints


def _oni_review_complete(payload: dict[str, Any] | None) -> bool:
    return str(((payload or {}).get("review") or {}).get("status", "")) == "complete"


def _load_role_records(review_root: Path, role: str) -> dict[str, dict[str, Any]]:
    directory = review_root / VALID_ROLES[role] / "records"
    if not directory.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        payload = _read_json(path, {})
        if isinstance(payload, dict) and payload.get("record_id"):
            records[str(payload["record_id"])] = payload
    return records


def _record_gate_summary(payload: dict[str, Any] | None, *, core: bool) -> dict[str, str]:
    gates = ((payload or {}).get("review") or {}).get("gates") or {}
    result = {f"g{index}": str((gates.get(f"g{index}") or {}).get("status", "pending")) for index in range(9)}
    if not core:
        for index in (4, 5, 6, 8):
            if result[f"g{index}"] == "pending":
                result[f"g{index}"] = "not_applicable"
    return result


def _review_complete(payload: dict[str, Any] | None, *, core: bool) -> bool:
    if not payload:
        return False
    quick = ((payload.get("review") or {}).get("quick_review") or {})
    quick_status = str(quick.get("status", ""))
    if quick_status == "complete":
        if not core:
            return True
        return (
            bool(quick.get("reps"))
            and bool(quick.get("phase_error_intervals"))
            and bool(quick.get("events"))
        )
    gates = _record_gate_summary(payload, core=core)
    applicable = [f"g{i}" for i in range(9) if core or i not in {4, 5, 6, 8}]
    return bool(payload.get("review_finished_at")) and all(gates[key] in {"pass", "not_applicable"} for key in applicable)


def _protocol_payload(project_root: Path) -> dict[str, Any]:
    summary = _read_json(project_root / "datasets/hyrox/reports/round9_implementation_summary.json", {})
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "artifact_type": "human_review_protocol_frozen",
        "protocol_version": PROTOCOL_VERSION,
        "frozen_at": _utc_now(),
        "source_commit": None,
        "frame_index_origin": 0,
        "interval_semantics": "closed",
        "timestamp_formula": "timestamp_ms = frame_index / fps * 1000",
        "timeline_labels": list(TIMELINE_LABELS),
        "validity_values": list(VALIDITY_VALUES),
        "observability_values": list(OBSERVABILITY_VALUES),
        "proposal_policy": "AI proposals are optional references loaded only by an explicit human action",
        "review_policy": "single_human_review_sufficient_for_current_stage",
        "independence_policy": "not_applicable_in_current_single_reviewer_stage",
        "automatic_action_gating_default_enabled": False,
        "formal_action_selection": "manual_only",
        "artifact_hashes": summary.get("artifact_hashes", {}),
        "release_gate_defaults": {
            "human_confirmed": False,
            "is_ground_truth": False,
            "training_eligible": False,
            "golden_eligible": False,
            "evaluation_eligible": False,
            "example_eligible": False,
        },
    }


def _build_agreement(review_root: Path, record_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    records_a = _load_role_records(review_root, "a")
    records_b = _load_role_records(review_root, "b")
    disagreements: list[dict[str, Any]] = []
    compared = 0
    equal = 0
    for record_id in sorted(set(records_a) & set(records_b)):
        compared += 1
        left = _flatten(records_a[record_id].get("review", {}))
        right = _flatten(records_b[record_id].get("review", {}))
        record_equal = True
        for field in sorted(set(left) | set(right)):
            if field.startswith("environment.") or field.startswith("summary.notes"):
                continue
            if left.get(field) == right.get(field):
                continue
            record_equal = False
            fingerprint = hashlib.sha256(f"{record_id}|{field}".encode("utf-8")).hexdigest()[:16]
            disagreements.append(
                {
                    "disagreement_id": f"dis_{fingerprint}",
                    "record_id": record_id,
                    "field": field,
                    "reviewer_a": left.get(field),
                    "reviewer_b": right.get(field),
                    "ai_proposal": None,
                    "status": "pending_adjudication",
                }
            )
        if record_equal:
            equal += 1
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "artifact_type": "raw_human_agreement",
        "protocol_version": PROTOCOL_VERSION,
        "generated_at": _utc_now(),
        "reviewer_a_record_count": len(records_a),
        "reviewer_b_record_count": len(records_b),
        "records_compared": compared,
        "records_with_complete_raw_match": equal,
        "raw_record_match_rate": round(equal / compared, 6) if compared else None,
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
        "note": "原始一致性在仲裁前计算；仲裁结果不会覆盖 A/B 原始文件。",
    }


def _aggregate_role(review_root: Path, role: str) -> dict[str, Any]:
    role_name = VALID_ROLES[role]
    records = _load_role_records(review_root, role)
    sections = {
        "governance.json": ("g0", "g1"),
        "target_tracks.json": ("g2",),
        "action_segments.json": ("g3",),
        "core_annotations.json": ("g4", "g5", "g6"),
        "object_scene.json": ("g7",),
        "scoring_correction.json": ("g8",),
    }
    artifacts: dict[str, Any] = {}
    for filename, gates in sections.items():
        artifacts[filename] = {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "artifact_type": filename.removesuffix(".json"),
            "protocol_version": PROTOCOL_VERSION,
            "reviewer_role": role_name,
            "generated_at": _utc_now(),
            "records": [
                {
                    "record_id": record_id,
                    "reviewer_id": payload.get("reviewer_id"),
                    "review_started_at": payload.get("review_started_at"),
                    "review_finished_at": payload.get("review_finished_at"),
                    "blind_review": payload.get("blind_review"),
                    "proposal_review": (payload.get("review") or {}).get("proposal_review", {}),
                    "gates": {gate: ((payload.get("review") or {}).get("gates") or {}).get(gate, {}) for gate in gates},
                }
                for record_id, payload in sorted(records.items())
            ],
        }
    role_dir = review_root / role_name
    for filename, artifact in artifacts.items():
        _atomic_json(role_dir / filename, artifact)
    return artifacts


def _handoff_payload(review_root: Path, record_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    records_a = _load_role_records(review_root, "a")
    complete_a = [
        record_id
        for record_id, payload in records_a.items()
        if record_id in record_index
        and _review_complete(payload, core=record_index[record_id]["action"] in CORE_ACTIONS)
    ]
    oni_paths = sorted((review_root / VALID_ROLES["a"] / "oni_records").glob("*.json"))
    oni_payloads = [_read_json(path, {}) for path in oni_paths]
    oni_complete = [item for item in oni_payloads if _oni_review_complete(item)]
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "artifact_type": "codex_handoff",
        "protocol_version": PROTOCOL_VERSION,
        "generated_at": _utc_now(),
        "review_root": "datasets/hyrox/reviews/human_v1",
        "source_proposals_must_remain_unchanged": True,
        "review_policy": "single_human_review_sufficient_for_current_stage",
        "reviewer_a": {"saved": len(records_a), "complete": len(complete_a), "complete_record_ids": sorted(complete_a)},
        "oni_modality_reviews": {"saved": len(oni_payloads), "complete": len(oni_complete), "expected": 64},
        "agreement": {"status": "not_applicable_single_reviewer_stage"},
        "release_gate_passed": len(complete_a) == len(record_index) and len(oni_complete) == 64,
        "formal_artifacts_updated": False,
        "required_codex_actions": [
            "validate every human review JSON against frame bounds, closed intervals and gate dependencies",
            "preserve the single-reviewer raw files and audit history",
            "treat double-review agreement and adjudication as not applicable in the current stage",
            "validate core rep, phase, event and error intervals before formal annotation updates",
            "validate Depth and IR subject conclusions independently",
            "update formal annotation artifacts only after all current-stage gates pass",
            "rerun schema, temporal containment, leakage and artifact-hash checks",
        ],
        "blocking_note": "当前阶段单人复核即可；subject ID 暂不作为阻断项。授权、标注完整性、结构与防泄漏检查仍须通过。",
    }


def create_review_blueprint(project_root: Path) -> Blueprint:
    blueprint = Blueprint("human_review", __name__)
    dataset_root = project_root / "datasets/hyrox"
    review_root = dataset_root / "reviews/human_v1"
    manifest_path = dataset_root / "manifests/phone_records.json"
    oni_manifest_path = dataset_root / "manifests/oni_records.json"
    write_lock = threading.RLock()
    track_cache: dict[str, list[dict[str, Any]]] = {}
    review_sessions: dict[str, dict[str, str]] = {}

    def source_data() -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        manifest = _read_json(manifest_path, {"records": []})
        record_index = {item["record_id"]: item for item in manifest.get("records", [])}
        action = _read_json(dataset_root / "annotations/action_segments_v1.json", {"records": []})
        core = _read_json(dataset_root / "annotations/core_rep_phase_event_error_v1.json", {"records": []})
        scene = _read_json(dataset_root / "annotations/object_scene_evidence_v1.json", {"records": []})
        scoring = _read_json(dataset_root / "annotations/scoring_correction_v1.json", {"records": []})
        queue = _read_json(dataset_root / "reports/round9_active_review_queue_v1.json", {"records": []})
        return record_index, action, core, scene, scoring, queue

    def oni_source_data() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        manifest = _read_json(oni_manifest_path, {"records": []})
        audit = _read_json(dataset_root / "reports/oni_subject_audit_v1.json", {"records": []})
        return (
            {item["record_id"]: item for item in manifest.get("records", [])},
            {item["record_id"]: item for item in audit.get("records", [])},
        )

    def json_error(message: str, status: int, code: str) -> tuple[Response, int]:
        return jsonify({"error": message, "code": code}), status

    def bound_session() -> dict[str, str] | None:
        session_id = str(getattr(g.pose_session, "session_id", ""))
        return review_sessions.get(session_id)

    def require_bound_session(*roles: str) -> tuple[dict[str, str] | None, tuple[Response, int] | None]:
        binding = bound_session()
        if binding is None:
            return None, json_error("请先在人工复核台确认角色与独立性声明", 403, "review_session_required")
        if roles and binding["role"] not in roles:
            return None, json_error("当前复核角色无权访问此内容", 403, "review_role_forbidden")
        return binding, None

    def saved_review(role: str, record_id: str) -> dict[str, Any] | None:
        return _read_json(_record_path(review_root, role, record_id), None)

    def check_reviewer(payload: dict[str, Any] | None, reviewer_id: str) -> None:
        if payload and payload.get("reviewer_id") not in {None, "", reviewer_id}:
            raise PermissionError("该角色已有另一复核者 ID 的记录；为保护独立结果，当前页面不能覆盖")

    @blueprint.get("/review")
    def review_page() -> str:
        return render_template("review.html")

    @blueprint.get("/api/review/bootstrap")
    def review_bootstrap() -> Response:
        record_index, _, _, _, _, queue = source_data()
        oni_index, oni_audit_index = oni_source_data()
        queue_index = {item["record_id"]: item for item in queue.get("records", [])}
        records_a = _load_role_records(review_root, "a")
        records_b = _load_role_records(review_root, "b")
        items: list[dict[str, Any]] = []
        for order, (record_id, record) in enumerate(record_index.items(), start=1):
            core = record.get("action") in CORE_ACTIONS
            queue_item = queue_index.get(record_id, {})
            items.append(
                {
                    "record_id": record_id,
                    "display_id": f"记录 {order:02d}",
                    "action_candidate": record.get("action"),
                    "action_label": ACTION_LABELS.get(str(record.get("action")), str(record.get("action"))),
                    "core": core,
                    "frame_count": record.get("video", {}).get("decoded_frame_count"),
                    "fps": record.get("video", {}).get("fps"),
                    "duration_seconds": record.get("video", {}).get("duration_seconds"),
                    "resolution": record.get("video", {}).get("resolution"),
                    "priority": queue_item.get("priority", 0),
                    "priority_reasons": queue_item.get("reasons", []),
                    "active_boundary_frames": queue_item.get("active_boundary_frames", []),
                    "special_focus": SPECIAL_FOCUS.get(record_id, []),
                    "reviewer_a": {
                        "saved": record_id in records_a,
                        "complete": _review_complete(records_a.get(record_id), core=core),
                        "gates": _record_gate_summary(records_a.get(record_id), core=core),
                    },
                    "reviewer_b": {
                        "saved": record_id in records_b,
                        "complete": _review_complete(records_b.get(record_id), core=core),
                        "gates": _record_gate_summary(records_b.get(record_id), core=core),
                    },
                }
            )
        items.sort(key=lambda item: (-int(item["priority"]), item["record_id"]))
        oni_items: list[dict[str, Any]] = []
        for order, (record_id, record) in enumerate(oni_index.items(), start=1):
            audit_record = oni_audit_index.get(record_id, {})
            for modality in ONI_MODALITIES:
                saved = _read_json(_oni_record_path(review_root, record_id, modality), None)
                modality_audit = (audit_record.get("modalities") or {}).get(modality, {})
                oni_items.append(
                    {
                        "task_id": f"{record_id}__{modality}",
                        "record_id": record_id,
                        "display_id": f"ONI {order:02d}",
                        "modality": modality,
                        "modality_label": ONI_LABELS_ZH["modalities"][modality],
                        "action_candidate": record.get("action"),
                        "action_label": ACTION_LABELS.get(str(record.get("action")), str(record.get("action"))),
                        "camera_view": record.get("camera_view"),
                        "recording_intent_code": record.get("recording_intent_code"),
                        "checkpoint_count": modality_audit.get("sampled_checkpoint_count", 0),
                        "saved": saved is not None,
                        "complete": _oni_review_complete(saved),
                    }
                )
        return jsonify(
            {
                "protocol_version": PROTOCOL_VERSION,
                "review_policy": "single_human_review_sufficient_for_current_stage",
                "automatic_action_gating_default_enabled": False,
                "formal_action_selection": "manual_only",
                "tasks": {
                    "core_fine_annotation": {
                        "label": "15 段核心动作逐次/事件精标",
                        "record_count": sum(1 for item in items if item["core"]),
                    },
                    "oni_subject_review": {
                        "label": "ONI Depth/IR 主体人工复核",
                        "record_count": len(oni_index),
                        "modality_task_count": len(oni_items),
                    },
                },
                "records": items,
                "oni_records": oni_items,
                "labels": {
                    "actions": ACTION_LABELS,
                    "timeline": TIMELINE_LABELS,
                    "validity": VALIDITY_VALUES,
                    "observability": OBSERVABILITY_VALUES,
                    "phases_by_action": PHASES_BY_ACTION,
                    "events_by_action": EVENTS_BY_ACTION,
                    "errors_by_action": ERRORS_BY_ACTION,
                    "phase_labels_zh": PHASE_LABELS_ZH,
                    "error_labels_zh": ERROR_LABELS_ZH,
                    "oni": ONI_LABELS_ZH,
                },
                "csrf_token": g.pose_session.csrf_token,
                "review_root": "datasets/hyrox/reviews/human_v1",
            }
        )

    @blueprint.post("/api/review/session")
    def bind_review_session() -> Response | tuple[Response, int]:
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return json_error("需要 JSON 请求体", 400, "invalid_json")
        role = str(body.get("role", ""))
        if role not in {"a", "b", "adjudicator"}:
            return json_error("复核角色无效", 400, "invalid_role")
        try:
            reviewer_id = _safe_reviewer_id(body.get("reviewer_id"))
        except ValueError as exc:
            return json_error(str(exc), 400, "invalid_reviewer")
        session_id = str(getattr(g.pose_session, "session_id", ""))
        with write_lock:
            existing = review_sessions.get(session_id)
            if existing and existing != {"role": role, "reviewer_id": reviewer_id}:
                return json_error("当前浏览器会话已绑定其他复核角色，请关闭后新建会话", 409, "review_session_conflict")
            review_sessions[session_id] = {"role": role, "reviewer_id": reviewer_id}
        return jsonify(
            {
                "ok": True,
                "role": role,
                "reviewer_id": reviewer_id,
                "review_policy": "single_human_review_sufficient_for_current_stage",
            }
        )

    @blueprint.get("/api/review/records/<record_id>")
    def review_record(record_id: str) -> Response | tuple[Response, int]:
        role = request.args.get("role", "")
        quick_mode = request.args.get("quick") == "1"
        binding, error = require_bound_session("a", "b")
        if error:
            return error
        try:
            reviewer_id = _safe_reviewer_id(request.args.get("reviewer_id"))
        except ValueError as exc:
            return json_error(str(exc), 400, "invalid_reviewer")
        if role not in VALID_ROLES:
            return json_error("role 必须为 a 或 b", 400, "invalid_role")
        if binding != {"role": role, "reviewer_id": reviewer_id}:
            return json_error("请求角色与当前独立复核会话不一致", 403, "review_session_mismatch")
        record_index, action, core, scene, scoring, _ = source_data()
        record = record_index.get(record_id)
        if record is None:
            return json_error("记录不存在", 404, "record_not_found")
        saved = saved_review(role, record_id)
        try:
            check_reviewer(saved, reviewer_id)
        except PermissionError as exc:
            return json_error(str(exc), 409, "reviewer_conflict")
        blind_complete = bool((saved or {}).get("blind_review", {}).get("completed_at"))
        reveal = quick_mode or (request.args.get("include_proposal") == "1" and blind_complete)
        proposal: dict[str, Any] | None = None
        if reveal:
            def find(container: dict[str, Any]) -> Any:
                return next((item for item in container.get("records", []) if item.get("record_id") == record_id), None)

            proposal = {
                "action_segments": find(action),
                "core_annotations": find(core),
                "object_scene": find(scene),
                "scoring_correction": find(scoring),
            }
        video = record.get("video", {})
        public_record = {
            "record_id": record_id,
            "action_candidate": record.get("action") if (blind_complete or quick_mode) else None,
            "action_label": ACTION_LABELS.get(str(record.get("action"))) if (blind_complete or quick_mode) else "首轮盲审后显示",
            "source_filename": record.get("source_filename") if (blind_complete or quick_mode) else None,
            "recording_intent": record.get("recording_intent") if (blind_complete or quick_mode) else None,
            "recording_intent_raw": record.get("recording_intent_raw") if (blind_complete or quick_mode) else None,
            "expected_errors_unverified": record.get("expected_errors_unverified") if (blind_complete or quick_mode) else None,
            "subject_id_current": record.get("subject_id"),
            "target_track_id": (record.get("target_athlete") or {}).get("track_id"),
            "camera_view_current": record.get("camera_view"),
            "other_people_present": record.get("other_people_present"),
            "sha256": record.get("sha256"),
            "integrity": record.get("integrity"),
            "video": video,
            "core": record.get("action") in CORE_ACTIONS,
            "special_focus": SPECIAL_FOCUS.get(record_id, []),
            "video_url": f"/api/review/records/{record_id}/video",
            "track_url": f"/api/review/records/{record_id}/track",
            "sheet_urls": {
                kind: f"/api/review/records/{record_id}/sheet/{kind}"
                for kind in ("round7", "round8", "round9")
                if (dataset_root / "reports" / f"{kind}_review_sheets" / f"{record_id}.jpg").exists()
            },
        }
        return jsonify(
            {
                "record": public_record,
                "saved_review": saved,
                "blind_complete": blind_complete,
                "proposal": proposal,
            }
        )

    @blueprint.get("/api/review/records/<record_id>/video")
    def review_video(record_id: str) -> Response | tuple[Response, int]:
        _, error = require_bound_session("a", "b", "adjudicator")
        if error:
            return error
        record_index, *_ = source_data()
        record = record_index.get(record_id)
        if record is None:
            return json_error("记录不存在", 404, "record_not_found")
        path = dataset_root / str(record["source_file"])
        if not path.exists():
            return json_error("原始视频不存在", 404, "video_not_found")
        return send_file(path, mimetype="video/mp4", conditional=True, max_age=0)

    @blueprint.get("/api/review/records/<record_id>/sheet/<kind>")
    def review_sheet(record_id: str, kind: str) -> Response | tuple[Response, int]:
        _, error = require_bound_session("a", "b", "adjudicator")
        if error:
            return error
        if kind not in {"round7", "round8", "round9"}:
            return json_error("审阅图类型无效", 404, "sheet_not_found")
        record_index, *_ = source_data()
        if record_id not in record_index:
            return json_error("记录不存在", 404, "record_not_found")
        path = dataset_root / "reports" / f"{kind}_review_sheets" / f"{record_id}.jpg"
        if not path.exists():
            return json_error("该记录没有此审阅图", 404, "sheet_not_found")
        return send_file(path, mimetype="image/jpeg", conditional=True, max_age=0)

    @blueprint.get("/api/review/records/<record_id>/track")
    def review_track(record_id: str) -> Response | tuple[Response, int]:
        _, error = require_bound_session("a", "b", "adjudicator")
        if error:
            return error
        record_index, *_ = source_data()
        if record_id not in record_index:
            return json_error("记录不存在", 404, "record_not_found")
        if record_id not in track_cache:
            path = dataset_root / "tracks" / record_id / "people.jsonl"
            compact: list[dict[str, Any]] = []
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        item = json.loads(line)
                        source_track = item.get("source_candidate_track_id")
                        candidate = next(
                            (candidate for candidate in item.get("candidates", []) if candidate.get("track_id") == source_track),
                            None,
                        )
                        compact.append(
                            {
                                "frame_index": item.get("frame_index"),
                                "bbox_xyxy": (candidate or {}).get("bbox_xyxy"),
                                "target_locked": item.get("target_locked"),
                                "target_status": item.get("target_status"),
                                "source_track_id": source_track,
                                "events": item.get("events", []),
                            }
                        )
            track_cache[record_id] = compact
        return jsonify({"record_id": record_id, "frames": track_cache[record_id]})

    @blueprint.put("/api/review/records/<record_id>")
    def save_review_record(record_id: str) -> Response | tuple[Response, int]:
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return json_error("需要 JSON 请求体", 400, "invalid_json")
        role = str(body.get("role", ""))
        if role not in VALID_ROLES:
            return json_error("role 必须为 a 或 b", 400, "invalid_role")
        try:
            reviewer_id = _safe_reviewer_id(body.get("reviewer_id"))
        except ValueError as exc:
            return json_error(str(exc), 400, "invalid_reviewer")
        binding, error = require_bound_session("a", "b")
        if error:
            return error
        if binding != {"role": role, "reviewer_id": reviewer_id}:
            return json_error("请求角色与当前独立复核会话不一致", 403, "review_session_mismatch")
        record_index, *_ = source_data()
        record = record_index.get(record_id)
        if record is None:
            return json_error("记录不存在", 404, "record_not_found")
        review = body.get("review")
        if not isinstance(review, dict):
            return json_error("review 必须为对象", 400, "invalid_review")
        frame_count = int(record.get("video", {}).get("decoded_frame_count", 0))
        fps = float(record.get("video", {}).get("fps", 0.0))
        if frame_count <= 0 or fps <= 0:
            return json_error("manifest 中的帧数或 FPS 无效", 409, "invalid_manifest")
        try:
            _walk_frame_values(review, frame_count, fps)
            if str((review.get("quick_review") or {}).get("action", record.get("action"))) in CORE_ACTIONS:
                _validate_fine_annotations(
                    review,
                    str((review.get("quick_review") or {}).get("action", record.get("action"))),
                )
        except ValueError as exc:
            return json_error(str(exc), 422, "frame_validation_failed")
        path = _record_path(review_root, role, record_id)
        with write_lock:
            old = _read_json(path, None)
            try:
                check_reviewer(old, reviewer_id)
            except PermissionError as exc:
                return json_error(str(exc), 409, "reviewer_conflict")
            old_revision = int((old or {}).get("revision", 0))
            base_revision = body.get("base_revision")
            if base_revision not in (None, old_revision):
                return json_error("记录已在其他页面更新，请刷新后重试", 409, "revision_conflict")
            blind = copy.deepcopy((old or {}).get("blind_review", {}))
            if body.get("freeze_blind_pass") and not blind.get("completed_at"):
                blind = {
                    "filename_intent_hidden": True,
                    "ai_proposal_hidden": True,
                    "completed_at": _utc_now(),
                    "snapshot": copy.deepcopy(review),
                }
            now = _utc_now()
            changes = _diff((old or {}).get("review", {}), review)
            audit_log = list((old or {}).get("audit_log", []))
            if changes:
                audit_log.append(
                    {
                        "revision": old_revision + 1,
                        "reviewer_id": reviewer_id,
                        "reviewed_at": now,
                        "reason": str(body.get("change_reason") or "draft autosave"),
                        "evidence_frames": body.get("evidence_frames") or [],
                        "changes": changes,
                    }
                )
            finished_at = (old or {}).get("review_finished_at")
            if body.get("finish_review"):
                finished_at = now
            payload = {
                "schema_version": REVIEW_SCHEMA_VERSION,
                "artifact_type": "human_review_record",
                "protocol_version": PROTOCOL_VERSION,
                "reviewer_role": VALID_ROLES[role],
                "reviewer_id": reviewer_id,
                "reviewer_type": "human",
                "record_id": record_id,
                "revision": old_revision + 1,
                "review_started_at": (old or {}).get("review_started_at") or now,
                "review_finished_at": finished_at,
                "blind_review": blind,
                "review": review,
                "audit_log": audit_log,
                "source_artifacts_unchanged": True,
                "eligibility_overrides_written": False,
                "saved_at": now,
            }
            _atomic_json(path, payload)
            protocol_path = review_root / "protocol_frozen.json"
            if not protocol_path.exists():
                _atomic_json(protocol_path, _protocol_payload(project_root))
        return jsonify(
            {
                "ok": True,
                "revision": payload["revision"],
                "saved_at": payload["saved_at"],
                "blind_complete": bool(blind.get("completed_at")),
                "review_finished_at": payload["review_finished_at"],
            }
        )

    @blueprint.get("/api/review/oni/<record_id>/<modality>")
    def review_oni_record(record_id: str, modality: str) -> Response | tuple[Response, int]:
        binding, error = require_bound_session("a")
        if error:
            return error
        if modality not in ONI_MODALITIES:
            return json_error("模态必须是 depth 或 ir", 404, "invalid_modality")
        oni_index, audit_index = oni_source_data()
        record = oni_index.get(record_id)
        if record is None:
            return json_error("ONI 记录不存在", 404, "record_not_found")
        checkpoints = _load_oni_checkpoints(dataset_root, record_id, modality)
        audit = audit_index.get(record_id, {})
        saved = _read_json(_oni_record_path(review_root, record_id, modality), None)
        return jsonify(
            {
                "record": {
                    "record_id": record_id,
                    "action_candidate": record.get("action"),
                    "action_label": ACTION_LABELS.get(str(record.get("action")), str(record.get("action"))),
                    "modality": modality,
                    "modality_label": ONI_LABELS_ZH["modalities"][modality],
                    "camera_view": record.get("camera_view"),
                    "recording_intent_code": record.get("recording_intent_code"),
                    "expected_errors_unverified": record.get("expected_errors_unverified", []),
                    "subject_id_current": record.get("subject_id"),
                    "preview_url": f"/api/review/oni/{record_id}/{modality}/preview",
                    "checkpoint_count": len(checkpoints),
                    "modality_audit": (audit.get("modalities") or {}).get(modality, {}),
                },
                "checkpoints": checkpoints,
                "saved_review": saved,
                "review_policy": "single_human_review_sufficient_for_current_stage",
            }
        )

    @blueprint.get("/api/review/oni/<record_id>/<modality>/preview")
    def review_oni_preview(record_id: str, modality: str) -> Response | tuple[Response, int]:
        _, error = require_bound_session("a")
        if error:
            return error
        if modality not in ONI_MODALITIES:
            return json_error("模态必须是 depth 或 ir", 404, "invalid_modality")
        oni_index, _ = oni_source_data()
        if record_id not in oni_index:
            return json_error("ONI 记录不存在", 404, "record_not_found")
        path = (
            dataset_root
            / "reports"
            / "round11_subject_previews"
            / record_id
            / f"{modality}_subject_proposals.jpg"
        )
        if not path.exists():
            return json_error("该模态的主体预览图不存在", 404, "preview_not_found")
        return send_file(path, mimetype="image/jpeg", conditional=True, max_age=0)

    @blueprint.put("/api/review/oni/<record_id>/<modality>")
    def save_oni_review(record_id: str, modality: str) -> Response | tuple[Response, int]:
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return json_error("需要 JSON 请求体", 400, "invalid_json")
        binding, error = require_bound_session("a")
        if error:
            return error
        if modality not in ONI_MODALITIES:
            return json_error("模态必须是 depth 或 ir", 404, "invalid_modality")
        try:
            reviewer_id = _safe_reviewer_id(body.get("reviewer_id"))
        except ValueError as exc:
            return json_error(str(exc), 400, "invalid_reviewer")
        if binding != {"role": "a", "reviewer_id": reviewer_id}:
            return json_error("请求复核者与当前单人复核会话不一致", 403, "review_session_mismatch")
        oni_index, _ = oni_source_data()
        if record_id not in oni_index:
            return json_error("ONI 记录不存在", 404, "record_not_found")
        review = body.get("review")
        if not isinstance(review, dict):
            return json_error("review 必须是对象", 400, "invalid_review")
        checkpoints = _load_oni_checkpoints(dataset_root, record_id, modality)
        allowed_frames = {int(item["source_frame_index"]) for item in checkpoints}
        rows = review.get("checkpoints")
        if not isinstance(rows, list):
            return json_error("checkpoints 必须是数组", 422, "invalid_checkpoints")
        valid_targets = set(ONI_LABELS_ZH["target_status"])
        valid_boxes = set(ONI_LABELS_ZH["bbox_status"])
        seen_frames: set[int] = set()
        try:
            for index, row in enumerate(rows):
                frame = int(row.get("frame_index"))
                if frame not in allowed_frames:
                    raise ValueError(f"检查点 {index + 1} 的帧号不在该模态的 24 个抽样检查点中")
                if frame in seen_frames:
                    raise ValueError(f"检查点帧号 {frame} 重复")
                seen_frames.add(frame)
                row["frame_index"] = frame
                if str(row.get("target_status", "")) not in valid_targets:
                    raise ValueError(f"检查点 {index + 1} 的主体结论无效")
                if str(row.get("bbox_status", "")) not in valid_boxes:
                    raise ValueError(f"检查点 {index + 1} 的框结论无效")
            if str(review.get("status", "")) == "complete":
                if seen_frames != allowed_frames:
                    raise ValueError("标记完成前必须复核该模态全部 24 个检查点")
                if any(
                    str(row.get("target_status")) == "unreviewed"
                    or str(row.get("bbox_status")) == "unreviewed"
                    for row in rows
                ):
                    raise ValueError("标记完成前必须逐项填写主体和候选框结论")
                if str(review.get("same_subject_throughout", "")) not in {"yes", "no", "unsure"}:
                    raise ValueError("标记完成前必须填写是否始终为同一主体")
        except (TypeError, ValueError) as exc:
            return json_error(str(exc), 422, "oni_review_validation_failed")
        path = _oni_record_path(review_root, record_id, modality)
        with write_lock:
            old = _read_json(path, None)
            try:
                check_reviewer(old, reviewer_id)
            except PermissionError as exc:
                return json_error(str(exc), 409, "reviewer_conflict")
            old_revision = int((old or {}).get("revision", 0))
            if body.get("base_revision") not in (None, old_revision):
                return json_error("记录已在其他页面更新，请刷新后重试", 409, "revision_conflict")
            now = _utc_now()
            changes = _diff((old or {}).get("review", {}), review)
            audit_log = list((old or {}).get("audit_log", []))
            if changes:
                audit_log.append(
                    {
                        "revision": old_revision + 1,
                        "reviewer_id": reviewer_id,
                        "reviewed_at": now,
                        "reason": str(body.get("change_reason") or "ONI subject review update"),
                        "evidence_frames": sorted(seen_frames),
                        "changes": changes,
                    }
                )
            payload = {
                "schema_version": REVIEW_SCHEMA_VERSION,
                "artifact_type": "oni_modality_subject_human_review",
                "protocol_version": PROTOCOL_VERSION,
                "review_policy": "single_human_review_sufficient_for_current_stage",
                "reviewer_role": VALID_ROLES["a"],
                "reviewer_id": reviewer_id,
                "reviewer_type": "human",
                "record_id": record_id,
                "modality": modality,
                "revision": old_revision + 1,
                "review_started_at": (old or {}).get("review_started_at") or now,
                "review_finished_at": now if review.get("status") == "complete" else None,
                "review": review,
                "audit_log": audit_log,
                "source_artifacts_unchanged": True,
                "eligibility_overrides_written": False,
                "saved_at": now,
            }
            _atomic_json(path, payload)
        return jsonify({"ok": True, "revision": payload["revision"], "saved_at": now})

    @blueprint.get("/api/review/agreement")
    def review_agreement() -> Response:
        _, error = require_bound_session("adjudicator")
        if error:
            return error
        record_index, *_ = source_data()
        agreement = _build_agreement(review_root, record_index)
        with write_lock:
            _atomic_json(review_root / "agreement/raw_agreement.json", agreement)
            _atomic_json(
                review_root / "agreement/disagreement_queue.json",
                {
                    "schema_version": REVIEW_SCHEMA_VERSION,
                    "artifact_type": "human_review_disagreement_queue",
                    "generated_at": agreement["generated_at"],
                    "disagreements": agreement["disagreements"],
                },
            )
        return jsonify(agreement)

    @blueprint.put("/api/review/adjudication")
    def save_adjudication() -> Response | tuple[Response, int]:
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return json_error("需要 JSON 请求体", 400, "invalid_json")
        try:
            adjudicator_id = _safe_reviewer_id(body.get("adjudicator_id"))
        except ValueError as exc:
            return json_error(str(exc), 400, "invalid_adjudicator")
        binding, error = require_bound_session("adjudicator")
        if error:
            return error
        if binding != {"role": "adjudicator", "reviewer_id": adjudicator_id}:
            return json_error("仲裁者 ID 与当前会话不一致", 403, "review_session_mismatch")
        decisions = body.get("decisions")
        if not isinstance(decisions, list):
            return json_error("decisions 必须为数组", 400, "invalid_decisions")
        for index, decision in enumerate(decisions):
            if not isinstance(decision, dict) or not decision.get("disagreement_id"):
                return json_error(f"decisions[{index}] 缺少 disagreement_id", 422, "invalid_decision")
            if not str(decision.get("reason", "")).strip():
                return json_error(f"decisions[{index}] 缺少仲裁理由", 422, "invalid_decision")
            decision["adjudicator_id"] = adjudicator_id
            decision["adjudicated_at"] = _utc_now()
        payload = {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "artifact_type": "human_review_adjudication_decisions",
            "protocol_version": PROTOCOL_VERSION,
            "adjudicator_id": adjudicator_id,
            "updated_at": _utc_now(),
            "decisions": decisions,
            "raw_reviews_unchanged": True,
        }
        with write_lock:
            _atomic_json(review_root / "adjudication/decisions.json", payload)
        return jsonify({"ok": True, "decision_count": len(decisions), "updated_at": payload["updated_at"]})

    @blueprint.get("/api/review/adjudication")
    def get_adjudication() -> Response:
        _, error = require_bound_session("adjudicator")
        if error:
            return error
        payload = _read_json(
            review_root / "adjudication/decisions.json",
            {
                "schema_version": REVIEW_SCHEMA_VERSION,
                "artifact_type": "human_review_adjudication_decisions",
                "protocol_version": PROTOCOL_VERSION,
                "decisions": [],
            },
        )
        return jsonify(payload)

    @blueprint.get("/api/review/export")
    def export_review() -> Response | tuple[Response, int]:
        scope = request.args.get("scope", "all")
        if scope not in {"a", "b", "all"}:
            return json_error("scope 必须为 a、b 或 all", 400, "invalid_scope")
        binding, error = require_bound_session("a", "b", "adjudicator")
        if error:
            return error
        allowed_scope = "all" if binding["role"] == "adjudicator" else binding["role"]
        if scope != allowed_scope:
            return json_error("当前角色不能导出其他复核者的原始结果", 403, "export_scope_forbidden")
        record_index, *_ = source_data()
        with write_lock:
            protocol_path = review_root / "protocol_frozen.json"
            if not protocol_path.exists():
                _atomic_json(protocol_path, _protocol_payload(project_root))
            if scope == "all":
                _aggregate_role(review_root, "a")
                _aggregate_role(review_root, "b")
            if scope == "all":
                agreement = _build_agreement(review_root, record_index)
                _atomic_json(review_root / "agreement/raw_agreement.json", agreement)
                _atomic_json(
                    review_root / "agreement/disagreement_queue.json",
                    {
                        "schema_version": REVIEW_SCHEMA_VERSION,
                        "artifact_type": "human_review_disagreement_queue",
                        "generated_at": agreement["generated_at"],
                        "disagreements": agreement["disagreements"],
                    },
                )
                handoff = _handoff_payload(review_root, record_index)
                _atomic_json(review_root / "codex_handoff.json", handoff)
                prompt = (
                    "请读取 datasets/hyrox/reviews/human_v1/codex_handoff.json、protocol_frozen.json、"
                    "reviewer_a/、reviewer_b/、agreement/ 与 adjudication/。先校验帧范围、闭区间、"
                    "G0–G9 依赖、UNOBSERVABLE 语义和 A/B 独立性；保留原 proposal 与 A/B 原始结果。"
                    "只有双人复核、全部分歧仲裁、授权/subject_id、防泄漏及结构门全部通过后，"
                    "才能更新正式 annotation JSON 和 eligibility。请先报告阻塞项与建议的下一步操作，再执行获准的工程更新。\n"
                )
                (review_root / "CODEX_NEXT_STEPS.md").write_text(prompt, encoding="utf-8")
            else:
                role_name = VALID_ROLES[scope]
                role_records = _load_role_records(review_root, scope)
                oni_payloads = (
                    [
                        _read_json(path, {})
                        for path in sorted(
                            (review_root / role_name / "oni_records").glob("*.json")
                        )
                    ]
                    if scope == "a"
                    else []
                )
                role_handoff = {
                    "schema_version": REVIEW_SCHEMA_VERSION,
                    "artifact_type": "codex_single_reviewer_handoff",
                    "protocol_version": PROTOCOL_VERSION,
                    "generated_at": _utc_now(),
                    "reviewer_role": role_name,
                    "review_policy": "single_human_review_sufficient_for_current_stage",
                    "saved_record_count": len(role_records),
                    "completed_record_count": sum(
                        1
                        for record_id, payload in role_records.items()
                        if record_id in record_index
                        and _review_complete(
                            payload,
                            core=record_index[record_id].get("action")
                            in CORE_ACTIONS,
                        )
                    ),
                    "release_gate_passed": (
                        scope == "a"
                        and len(role_records) == len(record_index)
                        and all(
                            _review_complete(
                                payload,
                                core=record_index[record_id].get("action")
                                in CORE_ACTIONS,
                            )
                            for record_id, payload in role_records.items()
                            if record_id in record_index
                        )
                        and len(oni_payloads) == 64
                        and all(_oni_review_complete(payload) for payload in oni_payloads)
                    ),
                    "oni_modality_review_count": len(oni_payloads),
                    "oni_modality_complete_count": sum(
                        _oni_review_complete(payload) for payload in oni_payloads
                    ),
                    "oni_modality_expected_count": 64 if scope == "a" else 0,
                    "records": [
                        {
                            "record_id": record_id,
                            "revision": payload.get("revision"),
                            "saved_at": payload.get("saved_at"),
                            "result": (payload.get("review") or {}).get("quick_review", {}),
                        }
                        for record_id, payload in sorted(role_records.items())
                    ],
                    "instructions": [
                        "validate frame bounds, closed intervals and audit history",
                        "use reps, phase_error_intervals and events for core fine annotations",
                        "process ONI Depth and IR subject reviews independently",
                        "treat the current stage as single-human review; do not require reviewer B",
                        "keep formal action selection manual and automatic recognition deferred",
                        "preserve original proposal files and original videos",
                    ],
                }
                _atomic_json(review_root / role_name / "codex_handoff.json", role_handoff)
                (review_root / role_name / "CODEX_NEXT_STEPS.md").write_text(
                    "请读取本目录 codex_handoff.json、records/ 下的手机结果和 oni_records/ 下的"
                    "Depth/IR 独立主体结果。当前阶段单人复核即可。先校验帧号、闭区间、rep、"
                    "phase_error_intervals、events 和 24 个 ONI 检查点；正式动作来源保持人工选择，"
                    "自动识别暂缓。保留原始视频和原 proposal；没有人工结论的字段不要补成真值。\n",
                    encoding="utf-8",
                )

            memory = io.BytesIO()
            with zipfile.ZipFile(memory, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(review_root.rglob("*")):
                    if not path.is_file():
                        continue
                    relative = path.relative_to(review_root)
                    if scope in {"a", "b"}:
                        role_name = VALID_ROLES[scope]
                        if relative != Path("protocol_frozen.json") and relative.parts[0] != role_name:
                            continue
                    archive.write(path, Path("human_v1") / relative)
            memory.seek(0)
        filename = f"hyrox_human_review_{scope}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        return send_file(memory, mimetype="application/zip", as_attachment=True, download_name=filename, max_age=0)

    return blueprint
