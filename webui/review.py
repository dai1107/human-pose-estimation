from __future__ import annotations

import copy
import gzip
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

from hyrox.features import extract_basic_pose_features
from hyrox.registry import create_action_analyzer


PROTOCOL_VERSION = "human_review_v1.0"
REVIEW_SCHEMA_VERSION = 1
VALID_ROLES = {"a": "reviewer_a", "b": "reviewer_b"}
CORE_ACTIONS = {"burpee_broad_jump", "lunge", "wall_ball"}
PHONE_ACTIONS = {
    "burpee_broad_jump",
    "lunge",
    "wall_ball",
    "farmers_carry",
    "rowing",
    "skierg",
    "sled_push",
    "sled_pull",
}
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
OBSERVABILITY_VALUES = ("OBSERVABLE", "PARTIAL", "UNOBSERVABLE", "UNSURE", "UNKNOWN")
VIEW_VALUES = ("front", "back", "side", "oblique_front", "oblique_back", "unsure")
VIEW_LABELS_ZH = {
    "front": "正面",
    "back": "背面",
    "side": "侧面",
    "oblique_front": "斜前方",
    "oblique_back": "斜后方",
    "unsure": "无法确认",
}
OBSERVABILITY_ITEMS_BY_ACTION = {
    "burpee_broad_jump": [
        ("foot_synchronization", "双脚同步"),
        ("hands_feet_distance", "手脚距离"),
        ("chest_contact", "胸部触地"),
        ("takeoff_landing", "起跳与落地"),
        ("extra_steps", "补步或碎步"),
        ("left_right_symmetry", "左右对称"),
    ],
    "lunge": [
        ("rear_knee_contact", "后膝触地"),
        ("hip_extension", "髋部伸展"),
        ("leg_alternation", "双腿交替"),
        ("extra_steps", "补步或碎步"),
        ("trunk_angle", "躯干角度"),
        ("left_right_symmetry", "左右对称"),
    ],
    "wall_ball": [
        ("squat_depth", "下蹲深度"),
        ("heel_rise", "脚跟抬起"),
        ("trunk_lean", "躯干倾斜"),
        ("ball_release", "球释放"),
        ("target_hit", "目标命中"),
        ("left_right_symmetry", "左右对称"),
    ],
}
VISIBILITY_VALUES = {"visible", "partial", "not_visible", "unsure"}
ONI_ACTION_USABILITY_VALUES = {"usable", "partially_usable", "unusable", "unsure"}
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
    "farmers_carry": ["ready", "carrying", "rest"],
    "rowing": ["catch", "drive", "finish", "recovery"],
    "skierg": ["top", "pull_down", "bottom", "return"],
    "sled_push": ["setup", "drive", "step", "reset"],
    "sled_pull": ["ready", "reach", "pull", "recover"],
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
    "farmers_carry": ["monitor_start", "carry_start", "carry_stop", "monitor_end"],
    "rowing": ["catch_reached", "drive_start", "finish_reached", "recovery_end"],
    "skierg": ["top_reached", "pull_start", "bottom_reached", "return_end"],
    "sled_push": ["drive_start", "step_contact", "drive_end"],
    "sled_pull": ["reach_reached", "pull_start", "pull_finish", "recovery_end"],
}
ERRORS_BY_ACTION = {
    "burpee_broad_jump": ["FOOT_DESYNCHRONIZED", "HANDS_FEET_TOO_FAR", "NO_CHEST_CONTACT", "EXTRA_STEP"],
    "lunge": ["NO_KNEE_CONTACT", "SAME_LEG_CONSECUTIVE", "HIP_NOT_EXTENDED", "EXTRA_STEP"],
    "wall_ball": ["NOT_DEEP_ENOUGH", "HEEL_RISE"],
    "farmers_carry": ["ARM_NOT_EXTENDED_VIOLATION", "ARM_NOT_BY_SIDE_VIOLATION", "LEAN_LEFT_RIGHT", "TORSO_LEAN"],
    "rowing": ["HANDLE_AROUND_KNEES", "TOO_MUCH_BACK_LEAN", "EARLY_ARM_PULL"],
    "skierg": ["ARMS_NOT_HIGH_ENOUGH", "NO_HIP_HINGE", "TOO_MUCH_SQUAT", "ASYMMETRIC_PULL", "RUSHED_RETURN"],
    "sled_push": ["TORSO_TOO_UPRIGHT", "TORSO_TOO_LOW", "SHORT_STEPS", "NO_LEG_DRIVE", "HIP_TOO_HIGH_OR_BACK_ROUND"],
    "sled_pull": ["SLED_PULL_KNEELING_VIOLATION", "SLED_PULL_SEATED_VIOLATION", "NOT_STANDING", "OVER_LEAN_BACK", "ARMS_ONLY_PULL", "NO_CLEAR_PULL", "ASYMMETRIC_PULL"],
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
    "ready": "准备",
    "carrying": "持续行走",
    "rest": "停止/休息",
    "catch": "划船起始",
    "drive": "蹬腿拉动",
    "finish": "划船结束",
    "top": "顶部",
    "pull_down": "下拉",
    "return": "回程",
    "setup": "准备姿态",
    "step": "蹬地步",
    "reset": "重置",
    "reach": "前伸",
    "pull": "拉动",
    "recover": "恢复",
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
    "ARM_NOT_EXTENDED_VIOLATION": "手臂未充分伸展",
    "ARM_NOT_BY_SIDE_VIOLATION": "手臂未保持在身体两侧",
    "LEAN_LEFT_RIGHT": "身体左右倾斜",
    "TORSO_LEAN": "躯干前后倾斜",
    "HANDLE_AROUND_KNEES": "桨把绕膝",
    "TOO_MUCH_BACK_LEAN": "结束阶段身体后仰过多",
    "EARLY_ARM_PULL": "手臂拉动过早",
    "ARMS_NOT_HIGH_ENOUGH": "回到顶部时手未充分上举",
    "NO_HIP_HINGE": "下拉时髋部折叠不足",
    "TOO_MUCH_SQUAT": "下拉时下蹲过多",
    "ASYMMETRIC_PULL": "左右拉动不对称",
    "RUSHED_RETURN": "回程过快",
    "TORSO_TOO_UPRIGHT": "推雪橇时身体过直",
    "TORSO_TOO_LOW": "推雪橇时身体压得过低",
    "SHORT_STEPS": "步幅过小",
    "NO_LEG_DRIVE": "腿部驱动不足",
    "HIP_TOO_HIGH_OR_BACK_ROUND": "髋部过高或背部不稳定",
    "SLED_PULL_KNEELING_VIOLATION": "拉雪橇时跪姿",
    "SLED_PULL_SEATED_VIOLATION": "拉雪橇时坐姿",
    "NOT_STANDING": "拉雪橇时未保持站立",
    "OVER_LEAN_BACK": "拉雪橇时后仰过多",
    "ARMS_ONLY_PULL": "只用手臂拉动",
    "NO_CLEAR_PULL": "没有清晰拉动",
    "OTHER": "其他错误（请备注）",
    "UNSURE": "无法确认",
}

EVENT_LABELS_ZH = {
    "hands_down": "双手撑地",
    "chest_contact_candidate": "胸部触地",
    "left_takeoff": "左脚起跳",
    "right_takeoff": "右脚起跳",
    "takeoff_candidate": "起跳",
    "left_landing": "左脚落地",
    "right_landing": "右脚落地",
    "landing_candidate": "落地",
    "stabilized": "落地稳定",
    "rep_start": "本次开始",
    "bottom_reached": "到达最低点",
    "rear_knee_contact_candidate": "后膝触地",
    "full_extension": "完全伸展",
    "ball_release_candidate": "出球",
    "recovery": "接球/恢复",
    "monitor_start": "监控区间开始",
    "carry_start": "行走开始",
    "carry_stop": "行走停止",
    "monitor_end": "监控区间结束",
    "catch_reached": "到达划船起始",
    "drive_start": "驱动开始",
    "finish_reached": "到达划船结束",
    "recovery_end": "恢复结束",
    "top_reached": "到达顶部",
    "pull_start": "拉动开始",
    "return_end": "回程结束",
    "step_contact": "蹬地/落脚",
    "drive_end": "驱动结束",
    "reach_reached": "到达前伸",
    "pull_finish": "拉动结束",
}

TEMPORARY_SUBJECT_SECOND_RECORDS = {
    "phone_skierg_002",
    "phone_sled_push_004",
    "phone_sled_push_005",
}

NONCORE_EVENT_FROM_PHASE = {
    "farmers_carry": {
        "carrying": ("carry_start", "start"),
        "rest": ("carry_stop", "start"),
    },
    "rowing": {
        "catch": ("catch_reached", "start"),
        "drive": ("drive_start", "start"),
        "finish": ("finish_reached", "start"),
        "recovery": ("recovery_end", "end"),
    },
    "skierg": {
        "top": ("top_reached", "start"),
        "pull_down": ("pull_start", "start"),
        "bottom": ("bottom_reached", "start"),
        "return": ("return_end", "end"),
    },
    "sled_push": {
        "drive": ("drive_start", "start"),
        "step": ("step_contact", "start"),
        "reset": ("drive_end", "start"),
    },
    "sled_pull": {
        "reach": ("reach_reached", "start"),
        "pull": ("pull_start", "start"),
        "recover": ("pull_finish", "start"),
    },
}

NONCORE_CYCLE_PHASE_ORDER = {
    "rowing": ["recovery", "catch", "drive", "finish"],
    "skierg": ["pull_down", "bottom", "return", "top"],
    "sled_push": ["drive", "step"],
    "sled_pull": ["pull", "recover", "reach"],
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
        "non_target_person": "非目标人物",
        "no_candidate": "无候选",
        "subject_switch": "主体切换",
        "unable": "无法判断",
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


def _temporary_subject_id(record_id: str, action: str) -> str:
    """Return the user-approved temporary grouping for the small phone set.

    These IDs are dataset groups, not a claim that identity was independently
    verified across actions.  SkiErg and sled push use two groups; every other
    action currently uses one.
    """

    suffix = "02" if record_id in TEMPORARY_SUBJECT_SECOND_RECORDS else "01"
    safe_action = re.sub(r"[^a-z0-9_]+", "_", action.lower()).strip("_") or "unknown"
    return f"subject_group_{safe_action}_{suffix}"


def _temporary_dataset_role(subject_id: str) -> str:
    return "validation" if subject_id.endswith("_02") else "development"


def _build_disagreement_clips(
    report: dict[str, Any],
    record_index: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Expand high-disagreement frames into stable 0.5–1.0 second clips."""

    output: dict[str, list[dict[str, Any]]] = {}
    for record_report in report.get("records", []):
        record_id = str(record_report.get("record_id", ""))
        record = record_index.get(record_id)
        if not record:
            continue
        fps = float((record.get("video") or {}).get("fps") or 0.0)
        frame_count = int((record.get("video") or {}).get("decoded_frame_count") or 0)
        if fps <= 0 or frame_count <= 0:
            continue
        source_items = [
            item
            for item in (record_report.get("top_review_frames") or [])
            if isinstance(item, dict) and item.get("review_required")
        ]
        by_frame = {
            int(item["frame_index"]): item
            for item in source_items
            if 0 <= int(item.get("frame_index", -1)) < frame_count
        }
        frames = sorted(by_frame)
        if not frames:
            output[record_id] = []
            continue
        half_window = max(1, round(fps * 0.375))
        minimum_length = max(1, round(fps * 0.5))
        maximum_length = max(1, round(fps * 1.0))
        clusters: list[dict[str, Any]] = []
        for anchor in frames:
            proposed_start = max(0, anchor - half_window)
            proposed_end = min(frame_count - 1, anchor + half_window)
            if (
                clusters
                and proposed_start <= int(clusters[-1]["end_frame"]) + 1
                and max(int(clusters[-1]["end_frame"]), proposed_end)
                - min(int(clusters[-1]["start_frame"]), proposed_start)
                + 1
                <= maximum_length
            ):
                cluster = clusters[-1]
                cluster["start_frame"] = min(int(cluster["start_frame"]), proposed_start)
                cluster["end_frame"] = max(int(cluster["end_frame"]), proposed_end)
                cluster["anchor_frames"].append(anchor)
            else:
                clusters.append(
                    {
                        "start_frame": proposed_start,
                        "end_frame": proposed_end,
                        "anchor_frames": [anchor],
                    }
                )
        clips: list[dict[str, Any]] = []
        for index, cluster in enumerate(clusters, start=1):
            anchors = list(cluster["anchor_frames"])
            priorities = [
                float(by_frame[frame].get("review_priority_score") or 0.0)
                for frame in anchors
            ]
            start = int(cluster["start_frame"])
            end = int(cluster["end_frame"])
            missing = minimum_length - (end - start + 1)
            if missing > 0:
                extend_left = min(start, missing // 2)
                start -= extend_left
                missing -= extend_left
                extend_right = min(frame_count - 1 - end, missing)
                end += extend_right
                missing -= extend_right
                if missing > 0:
                    start = max(0, start - missing)
            clips.append(
                {
                    "clip_id": f"{record_id}_disagreement_{index:03d}_{start}_{end}",
                    "start_frame": start,
                    "end_frame": end,
                    "duration_seconds": round((end - start + 1) / fps, 3),
                    "anchor_frames": anchors,
                    "anchor_frame_count": len(anchors),
                    "priority_score": max(priorities, default=0.0),
                    "review_status": "pending",
                    "evidence_type": "unspecified",
                    "observability": "UNKNOWN",
                    "notes": "",
                }
            )
        output[record_id] = clips
    return output


def _load_full_disagreement_report(
    dataset_root: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Replace capped summary samples with every review-required cache row."""

    records: list[dict[str, Any]] = []
    for record in summary.get("records", []):
        record_id = str(record.get("record_id", ""))
        path = (
            dataset_root
            / "pose_cache"
            / record_id
            / "backend_agreement.jsonl.gz"
        )
        review_rows: list[dict[str, Any]] = []
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if item.get("review_required"):
                        review_rows.append(item)
        else:
            review_rows = [
                item
                for item in (record.get("top_review_frames") or [])
                if isinstance(item, dict) and item.get("review_required")
            ]
        records.append({**record, "top_review_frames": review_rows})
    return {**summary, "records": records}


def _target_action_bounds(
    action_row: dict[str, Any] | None,
    frame_count: int,
) -> tuple[int, int]:
    segments = (action_row or {}).get("segments") or []
    target = next(
        (
            item
            for item in segments
            if str(item.get("timeline_label") or item.get("label"))
            == "target_action"
        ),
        None,
    )
    if target:
        return int(target["start_frame"]), int(target["end_frame"])
    return 0, max(0, frame_count - 1)


def _merge_phase_spans(
    spans: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for span in spans:
        if (
            merged
            and merged[-1]["phase"] == span["phase"]
            and int(span["start_frame"]) <= int(merged[-1]["end_frame"]) + 1
        ):
            merged[-1]["end_frame"] = max(
                int(merged[-1]["end_frame"]), int(span["end_frame"])
            )
        else:
            merged.append(dict(span))
    return merged


def _fallback_phase_spans(
    start: int,
    end: int,
    phases: list[str],
) -> list[dict[str, Any]]:
    """Create an explicitly low-confidence scaffold when no phase is detected."""

    if start > end or not phases:
        return []
    length = end - start + 1
    output: list[dict[str, Any]] = []
    for index, phase in enumerate(phases):
        phase_start = start + (length * index) // len(phases)
        phase_end = start + (length * (index + 1)) // len(phases) - 1
        if index == len(phases) - 1:
            phase_end = end
        if phase_start <= phase_end:
            output.append(
                {
                    "phase": phase,
                    "start_frame": phase_start,
                    "end_frame": phase_end,
                    "notes": "低置信度等分阶段脚手架，必须人工核对",
                }
            )
    return output


def _build_noncore_candidate_annotations(
    dataset_root: Path,
    record: dict[str, Any],
    action_row: dict[str, Any] | None,
    counting_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay the existing pose cache into editable, non-ground-truth proposals."""

    record_id = str(record["record_id"])
    action = str(record["action"])
    video = record.get("video") or {}
    frame_count = int(video.get("decoded_frame_count") or 0)
    width = int(video.get("width") or 1)
    height = int(video.get("height") or 1)
    active_start, active_end = _target_action_bounds(action_row, frame_count)
    pose_relative = (record.get("pose_cache") or {}).get("causal_analysis_pose")
    pose_path = dataset_root / str(
        pose_relative
        or f"pose_cache/{record_id}/causal_analysis_pose.jsonl.gz"
    )
    allowed_phases = set(PHASES_BY_ACTION.get(action, []))
    phase_spans: list[dict[str, Any]] = []
    count_events: list[dict[str, Any]] = []
    previous_phase: str | None = None
    phase_start: int | None = None
    previous_frame: int | None = None
    previous_count = 0
    final_state: dict[str, Any] = {}
    if pose_path.exists() and action in PHONE_ACTIONS - CORE_ACTIONS:
        analyzer = create_action_analyzer(
            action,
            camera_view=str(record.get("camera_view", "unknown")),
            live_mode=False,
        )
        with gzip.open(pose_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                frame = json.loads(line)
                frame_index = int(frame.get("frame_index", -1))
                if frame_index < active_start or frame_index > active_end:
                    continue
                eligible = bool(frame.get("formal_pose_eligible")) and bool(
                    frame.get("may_drive_rules_or_training")
                )
                features = (
                    extract_basic_pose_features(
                        frame.get("image_normalized_2d"),
                        width,
                        height,
                    )
                    if eligible
                    else None
                )
                final_state = analyzer.update(
                    features,
                    int(round(float(frame.get("source_timestamp_ms", 0.0)))),
                )
                debug = final_state.get("debug") or {}
                phase = str(
                    debug.get("stable_phase")
                    or final_state.get("phase")
                    or "unknown"
                )
                normalized_phase = phase if phase in allowed_phases else None
                if normalized_phase != previous_phase:
                    if (
                        previous_phase is not None
                        and phase_start is not None
                        and previous_frame is not None
                    ):
                        phase_spans.append(
                            {
                                "phase": previous_phase,
                                "start_frame": phase_start,
                                "end_frame": previous_frame,
                            }
                        )
                    previous_phase = normalized_phase
                    phase_start = frame_index if normalized_phase else None
                current_count = int(final_state.get("rep_count", 0) or 0)
                if current_count > previous_count:
                    count_events.append(
                        {
                            "count": current_count,
                            "frame_index": frame_index,
                            "terminal_phase": phase,
                        }
                    )
                previous_count = current_count
                previous_frame = frame_index
    if (
        previous_phase is not None
        and phase_start is not None
        and previous_frame is not None
    ):
        phase_spans.append(
            {
                "phase": previous_phase,
                "start_frame": phase_start,
                "end_frame": previous_frame,
            }
        )
    phase_spans = _merge_phase_spans(phase_spans)
    if counting_row and counting_row.get("count_events"):
        count_events = [
            {
                "count": int(item.get("count", index)),
                "frame_index": int(item["frame_index"]),
                "terminal_phase": str(item.get("terminal_phase", "")),
            }
            for index, item in enumerate(
                counting_row.get("count_events") or [], start=1
            )
            if active_start <= int(item.get("frame_index", -1)) <= active_end
        ]
    if not count_events and action != "farmers_carry":
        terminal_phase = {
            "rowing": "finish",
            "skierg": "top",
            "sled_push": "step",
            "sled_pull": "reach",
        }.get(action)
        if terminal_phase:
            count_events = [
                {
                    "count": index,
                    "frame_index": int(span["end_frame"]),
                    "terminal_phase": terminal_phase,
                }
                for index, span in enumerate(
                    (
                        item
                        for item in phase_spans
                        if item["phase"] == terminal_phase
                    ),
                    start=1,
                )
            ]

    if action == "farmers_carry":
        rep_bounds = [(active_start, active_end)]
        rep_ids = ["monitor_001"]
    else:
        rep_bounds = []
        rep_ids = []
        next_start = active_start
        for index, event in enumerate(count_events, start=1):
            end = int(event["frame_index"])
            if end < next_start:
                continue
            rep_bounds.append((next_start, end))
            rep_ids.append(f"cycle_{index:03d}")
            next_start = end + 1
        if not rep_bounds and active_start <= active_end:
            rep_bounds = [(active_start, active_end)]
            rep_ids = ["candidate_001"]

    reps: list[dict[str, Any]] = []
    phase_event_rules = NONCORE_EVENT_FROM_PHASE.get(action, {})
    for rep_id, (start, end) in zip(rep_ids, rep_bounds):
        clipped_phases = [
            {
                **span,
                "start_frame": max(start, int(span["start_frame"])),
                "end_frame": min(end, int(span["end_frame"])),
            }
            for span in phase_spans
            if int(span["end_frame"]) >= start
            and int(span["start_frame"]) <= end
        ]
        clipped_phases = [
            span
            for span in _merge_phase_spans(clipped_phases)
            if int(span["start_frame"]) <= int(span["end_frame"])
        ]
        if action != "farmers_carry":
            clipped_phases = _fallback_phase_spans(
                start,
                end,
                NONCORE_CYCLE_PHASE_ORDER.get(
                    action, PHASES_BY_ACTION.get(action, [])
                ),
            )
        elif not clipped_phases:
            clipped_phases = _fallback_phase_spans(
                start, end, PHASES_BY_ACTION.get(action, [])
            )
        events: list[dict[str, Any]] = []
        if action == "farmers_carry":
            events.append(
                {"event_type": "monitor_start", "frame_index": start}
            )
        for span in clipped_phases:
            event_rule = phase_event_rules.get(str(span["phase"]))
            if not event_rule:
                continue
            event_type, position = event_rule
            frame_index = int(
                span["start_frame"] if position == "start" else span["end_frame"]
            )
            if not any(
                item["event_type"] == event_type
                and int(item["frame_index"]) == frame_index
                for item in events
            ):
                events.append(
                    {"event_type": event_type, "frame_index": frame_index}
                )
        if action == "farmers_carry":
            events.append({"event_type": "monitor_end", "frame_index": end})
        if not events and EVENTS_BY_ACTION.get(action):
            events.append(
                {
                    "event_type": EVENTS_BY_ACTION[action][0],
                    "frame_index": start,
                }
            )
        phases = [
            {
                **span,
                "notes": span.get(
                    "notes", "姿态状态机候选阶段，待人工核对"
                ),
            }
            for span in clipped_phases
        ]
        reps.append(
            {
                "rep_id": rep_id,
                "start_frame": start,
                "end_frame": end,
                "validity": "UNSURE",
                "phases": phases,
                "events": events,
                "errors": [],
                "phase_gap_reason": (
                    "候选阶段可能存在低可见度或 unknown 空白，待人工核对"
                ),
                "notes": (
                    "姿态状态机生成的分析周期候选，非人工真值"
                    if action != "farmers_carry"
                    else "连续监控区间候选，不代表官方次数"
                ),
            }
        )
    return {
        "record_id": record_id,
        "action_type": action,
        "proposal_semantics": (
            "continuous_monitor"
            if action == "farmers_carry"
            else str(final_state.get("count_semantics") or "analysis_cycle")
        ),
        "official_rep_count_supported": bool(
            final_state.get("official_rep_count_supported", False)
        ),
        "source": "causal_pose_state_machine_replay",
        "is_ground_truth": False,
        "human_confirmation_required": True,
        "reps": reps,
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


def _audit_revision_continuous(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return True
    revision = int(payload.get("revision", 0))
    audit = payload.get("audit_log") or []
    return [int(item.get("revision", -1)) for item in audit] == list(range(1, revision + 1))


def _repair_audit_log(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Preserve known entries and fill missing historical revision markers."""

    revision = max(0, int(payload.get("revision", 0)))
    known: dict[int, dict[str, Any]] = {}
    for item in payload.get("audit_log") or []:
        if not isinstance(item, dict):
            continue
        number = int(item.get("revision", -1))
        if 1 <= number <= revision and number not in known:
            known[number] = copy.deepcopy(item)
    repaired: list[dict[str, Any]] = []
    for number in range(1, revision + 1):
        if number in known:
            repaired.append(known[number])
            continue
        repaired.append(
            {
                "revision": number,
                "reviewer_id": str(payload.get("reviewer_id") or "unknown"),
                "reviewed_at": str(payload.get("saved_at") or _utc_now()),
                "reason": (
                    "automatic audit chain repair: historical revision "
                    "existed without an audit entry"
                ),
                "evidence_frames": [],
                "changes": [],
                "repair_metadata": {
                    "content_changes_reconstructed": False,
                    "repair_is_not_a_human_annotation": True,
                },
            }
        )
    return repaired


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


def _enrich_frame_values(value: Any, fps: float) -> None:
    """Normalize usable frame numbers without rejecting a saveable draft."""

    if isinstance(value, dict):
        for key, child in list(value.items()):
            if key in {"frame_index", "start_frame", "end_frame", "left_frame_index", "right_frame_index"}:
                if child in (None, ""):
                    continue
                try:
                    frame = int(child)
                except (TypeError, ValueError):
                    continue
                value[key] = frame
                if key == "frame_index" and fps > 0:
                    value["timestamp_ms"] = round(frame / fps * 1000.0, 3)
            else:
                _enrich_frame_values(child, fps)
    elif isinstance(value, list):
        for child in value:
            _enrich_frame_values(child, fps)


def _validate_fine_annotations(review: dict[str, Any], action: str) -> None:
    quick = review.get("quick_review") or {}
    fine_complete = str(quick.get("fine_annotation_status", "")) == "complete" or (
        action in CORE_ACTIONS
        and not quick.get("fine_annotation_status")
        and str(quick.get("status", "")) == "complete"
    )
    intervals = quick.get("phase_error_intervals") or []
    valid_phases = set(PHASES_BY_ACTION.get(action, []))
    valid_errors = set(ERRORS_BY_ACTION.get(action, [])) | {"NO_ERROR", "OTHER", "UNSURE"}
    reps = quick.get("reps") or []
    rep_ids = {str(item.get("rep_id", "")).strip() for item in reps}
    if fine_complete and (
        not rep_ids or not intervals or not (quick.get("events") or [])
    ):
        raise ValueError("人工 4 精标完成前必须填写逐次边界、阶段错误区间和关键事件")
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
        rep = next((candidate for candidate in reps if str(candidate.get("rep_id", "")).strip() == rep_id), None)
        if rep and (
            int(item.get("start_frame", -1)) < int(rep.get("start_frame", -1))
            or int(item.get("end_frame", -1)) > int(rep.get("end_frame", -1))
        ):
            raise ValueError(f"阶段错误区间 {index + 1} 必须属于对应 rep")
        if str(item.get("observability", "")) == "UNOBSERVABLE" and error_code not in {"NO_ERROR", "UNSURE"}:
            raise ValueError(f"阶段错误区间 {index + 1} 不可观察时不能保存肯定错误")
    ordered_reps = sorted(reps, key=lambda item: int(item.get("start_frame", -1)))
    for index, rep in enumerate(ordered_reps):
        if not str(rep.get("rep_id", "")).strip():
            raise ValueError(f"第 {index + 1} 次缺少 rep_id")
        if index and int(rep.get("start_frame", -1)) <= int(ordered_reps[index - 1].get("end_frame", -1)):
            raise ValueError("rep 不允许重叠")
    for index, event in enumerate(quick.get("events") or []):
        rep_id = str(event.get("rep_id", "")).strip()
        if not rep_id and len(reps) == 1:
            rep_id = str(reps[0].get("rep_id", "")).strip()
            event["rep_id"] = rep_id
        rep = next((candidate for candidate in reps if str(candidate.get("rep_id", "")).strip() == rep_id), None)
        if rep is None:
            raise ValueError(f"关键事件 {index + 1} 必须关联有效 rep")
        frame = int(event.get("frame_index", -1))
        if frame < int(rep.get("start_frame", -1)) or frame > int(rep.get("end_frame", -1)):
            raise ValueError(f"关键事件 {index + 1} 必须落在对应 rep 内")
    if fine_complete:
        for rep in reps:
            rep_id = str(rep.get("rep_id", "")).strip()
            rep_intervals = sorted(
                (
                    item for item in intervals
                    if str(item.get("rep_id", "")).strip() == rep_id
                ),
                key=lambda item: int(item.get("start_frame", -1)),
            )
            has_gap = (
                not rep_intervals
                or int(rep_intervals[0].get("start_frame", -1)) > int(rep.get("start_frame", -1))
                or int(rep_intervals[-1].get("end_frame", -1)) < int(rep.get("end_frame", -1))
                or any(
                    int(current.get("start_frame", -1)) > int(previous.get("end_frame", -1)) + 1
                    for previous, current in zip(rep_intervals, rep_intervals[1:])
                )
            )
            if has_gap and not str(rep.get("phase_gap_reason", "")).strip():
                raise ValueError(f"{rep_id} 的阶段区间不连续，必须填写空白原因")
        no_rep = str(quick.get("overall_result", "")) == "NO_REP" or any(
            str(rep.get("validity", "")) == "NO_REP" for rep in reps
        )
        positive_errors = [
            item for item in intervals
            if str(item.get("error_code", "")) not in {"", "NO_ERROR", "UNSURE"}
        ]
        if no_rep and not positive_errors and not str(quick.get("no_rep_reason", "")).strip():
            raise ValueError("NO_REP 但没有肯定错误时必须填写“不构成一次的原因”")
    valid_clip_statuses = {
        "pending",
        "action_evidence_clear",
        "pose_backend_failure",
        "occluded_or_missing",
        "not_relevant",
    }
    valid_evidence_types = {
        "unspecified",
        "chest_contact",
        "rear_knee_contact",
        "takeoff_landing",
        "heel_toe",
        "wrist_or_hand",
        "phase_boundary",
        "other",
    }
    for index, clip in enumerate(quick.get("disagreement_clips") or []):
        if str(clip.get("review_status", "pending")) not in valid_clip_statuses:
            raise ValueError(f"高分歧片段 {index + 1} 的复核结论无效")
        if str(clip.get("evidence_type", "unspecified")) not in valid_evidence_types:
            raise ValueError(f"高分歧片段 {index + 1} 的关注证据无效")


def _oni_eligibility(review: dict[str, Any] | None) -> dict[str, bool]:
    review = review or {}
    complete = str(review.get("status", "")) == "complete"
    subject_confirmed = (
        str(review.get("overall_target_status", "")) == "correct"
        and str(review.get("same_subject_throughout", "")) == "yes"
    )
    view_complete = (
        str(review.get("confirmed_view", "")) in set(VIEW_VALUES) - {"unsure"}
        and str(review.get("action_usability", "")) in {"usable", "partially_usable"}
        and bool(review.get("observability_items"))
        and all(
            str(item.get("status", "")) in {"OBSERVABLE", "PARTIAL", "UNOBSERVABLE", "UNSURE"}
            for item in (review.get("observability_items") or [])
        )
    )
    return {
        "unreviewed": not complete,
        "view_policy_calibration_eligible": complete and subject_confirmed and view_complete,
        "rgb_rule_calibration_eligible": False,
        "training_eligible": False,
        "release_eligible": False,
    }


def _validate_oni_view_prior(
    review: dict[str, Any],
    *,
    action: str,
    allowed_frames: set[int],
) -> None:
    if not allowed_frames:
        raise ValueError("当前模态没有可供复核的检查点")
    minimum_frame, maximum_frame = min(allowed_frames), max(allowed_frames)
    confirmed_view = str(review.get("confirmed_view", "unsure"))
    if confirmed_view not in VIEW_VALUES:
        raise ValueError("人工确认视角无效")
    if str(review.get("action_usability", "unsure")) not in ONI_ACTION_USABILITY_VALUES:
        raise ValueError("动作可用性无效")
    for field in ("full_body_visibility", "floor_visibility", "equipment_visibility"):
        if str(review.get(field, "unsure")) not in VISIBILITY_VALUES:
            raise ValueError(f"{field} 的可见性无效")
    usable_start = review.get("usable_start_frame")
    usable_end = review.get("usable_end_frame")
    if usable_start not in (None, "") or usable_end not in (None, ""):
        if usable_start in (None, "") or usable_end in (None, ""):
            raise ValueError("可用动作区间必须同时填写起止帧")
        if int(usable_start) < minimum_frame or int(usable_end) > maximum_frame or int(usable_start) > int(usable_end):
            raise ValueError("可用动作区间超出当前模态帧范围")
    for index, interval in enumerate(review.get("identity_switch_intervals") or []):
        start = int(interval.get("start_frame", -1))
        end = int(interval.get("end_frame", -1))
        if start < minimum_frame or end > maximum_frame or start > end:
            raise ValueError(f"主体切换区间 {index + 1} 起止帧无效")
    expected_codes = {code for code, _ in OBSERVABILITY_ITEMS_BY_ACTION.get(action, [])}
    rows = review.get("observability_items") or []
    seen_codes: set[str] = set()
    for index, item in enumerate(rows):
        code = str(item.get("item_code", ""))
        if code not in expected_codes or code in seen_codes:
            raise ValueError(f"可观察性判断项 {index + 1} 无效或重复")
        seen_codes.add(code)
        status = str(item.get("status", ""))
        if status not in {"OBSERVABLE", "PARTIAL", "UNOBSERVABLE", "UNSURE"}:
            raise ValueError(f"可观察性判断项 {index + 1} 状态无效")
        if status == "UNOBSERVABLE" and bool(item.get("asserted_error")):
            raise ValueError(f"可观察性判断项 {index + 1} 不可观察时不能保存肯定错误")
        if status == "UNOBSERVABLE" and not str(item.get("reason", "")).strip():
            raise ValueError(f"可观察性判断项 {index + 1} 不可观察时必须填写原因")
        for frame in item.get("evidence_frames") or []:
            if int(frame) not in allowed_frames:
                raise ValueError(f"可观察性判断项 {index + 1} 的证据帧不属于当前模态检查点")


def _validate_oni_error_truth(
    review: dict[str, Any],
    *,
    expected_errors: set[str],
    allowed_frames: set[int],
) -> None:
    rows = review.get("error_truth_items") or []
    seen: set[str] = set()
    for index, item in enumerate(rows):
        error_code = str(item.get("error_code", ""))
        if error_code not in expected_errors or error_code in seen:
            raise ValueError(f"错误真值项 {index + 1} 无效或重复")
        seen.add(error_code)
        truth_status = str(item.get("truth_status", "unreviewed"))
        if truth_status not in {"unreviewed", "confirmed", "rejected", "unsure"}:
            raise ValueError(f"错误真值项 {index + 1} 结论无效")
        observability = str(item.get("observability", "UNSURE"))
        if observability not in {"OBSERVABLE", "PARTIAL", "UNOBSERVABLE", "UNSURE"}:
            raise ValueError(f"错误真值项 {index + 1} 可观察性无效")
        if observability == "UNOBSERVABLE" and truth_status == "confirmed":
            raise ValueError(f"错误真值项 {index + 1} 不可观察时不能确认错误发生")
        for frame in item.get("evidence_frames") or []:
            if int(frame) not in allowed_frames:
                raise ValueError(f"错误真值项 {index + 1} 的证据帧不属于当前模态检查点")


def _record_path(review_root: Path, role: str, record_id: str) -> Path:
    return review_root / VALID_ROLES[role] / "records" / f"{record_id}.json"


def _oni_record_path(review_root: Path, record_id: str, modality: str) -> Path:
    return review_root / VALID_ROLES["a"] / "oni_records" / f"{record_id}__{modality}.json"


def _oni_view_record_path(review_root: Path, record_id: str, modality: str) -> Path:
    return review_root / VALID_ROLES["a"] / "view_prior_records" / f"{record_id}__{modality}.json"


def _oni_error_record_path(review_root: Path, record_id: str, modality: str) -> Path:
    return review_root / VALID_ROLES["a"] / "error_truth_records" / f"{record_id}__{modality}.json"


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


def _candidate_loss_intervals(checkpoints: list[dict[str, Any]]) -> list[dict[str, int]]:
    intervals: list[dict[str, int]] = []
    start: int | None = None
    previous: int | None = None
    for item in checkpoints:
        frame = int(item["source_frame_index"])
        lost = not item.get("bbox_px") or str(item.get("target_lock_status", "")).lower() in {
            "missing",
            "lost",
            "no_candidate",
        }
        if lost:
            if start is None:
                start = frame
            previous = frame
        elif start is not None and previous is not None:
            intervals.append({"start_frame": start, "end_frame": previous})
            start = previous = None
    if start is not None and previous is not None:
        intervals.append({"start_frame": start, "end_frame": previous})
    return intervals


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
    if (
        quick_status == "blocked"
        and str(quick.get("video_usability", "")).lower() == "unusable"
    ):
        return True
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


def _fine_review_complete(payload: dict[str, Any] | None, *, core: bool) -> bool:
    if not payload:
        return False
    quick = ((payload.get("review") or {}).get("quick_review") or {})
    fine_status = str(quick.get("fine_annotation_status", ""))
    if not fine_status and core:
        fine_status = str(quick.get("status", ""))
    return (
        fine_status == "complete"
        and bool(quick.get("reps"))
        and bool(quick.get("phase_error_intervals"))
        and bool(quick.get("events"))
    )


def _disagreement_progress(
    payload: dict[str, Any] | None,
    clips: list[dict[str, Any]],
) -> dict[str, Any]:
    saved_rows = (
        (((payload or {}).get("review") or {}).get("quick_review") or {}).get(
            "disagreement_clips"
        )
        or []
    )
    saved_by_id = {
        str(item.get("clip_id")): item
        for item in saved_rows
        if isinstance(item, dict) and item.get("clip_id")
    }
    completed = sum(
        str((saved_by_id.get(str(clip["clip_id"])) or {}).get("review_status", "pending"))
        != "pending"
        for clip in clips
    )
    return {
        "saved": bool(saved_rows),
        "complete": bool(clips) and completed == len(clips),
        "completed_clip_count": completed,
        "clip_count": len(clips),
        "pending_clip_count": max(0, len(clips) - completed),
    }


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


def _build_review_exports(
    review_root: Path,
    oni_index: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    generated_at = _utc_now()
    view_records: list[dict[str, Any]] = []
    subject_records: list[dict[str, Any]] = []
    error_records: list[dict[str, Any]] = []
    summary_groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record_id, source in sorted(oni_index.items()):
        for modality in ONI_MODALITIES:
            subject_saved = _read_json(_oni_record_path(review_root, record_id, modality), None)
            view_saved = _read_json(_oni_view_record_path(review_root, record_id, modality), None)
            error_saved = _read_json(_oni_error_record_path(review_root, record_id, modality), None)
            subject_review = (subject_saved or {}).get("review") or {}
            view_review = (view_saved or {}).get("review") or {}
            error_review = (error_saved or {}).get("review") or {}
            subject_eligibility = (subject_saved or {}).get("eligibility") or _oni_eligibility(subject_review)
            view_eligibility = (view_saved or {}).get("eligibility") or _oni_eligibility(view_review)
            error_eligibility = (error_saved or {}).get("eligibility") or _oni_eligibility(error_review)
            subject_common = {
                "record_id": record_id,
                "action": source.get("action"),
                "modality": modality,
                "revision": (subject_saved or {}).get("revision", 0),
                "audit_revision_continuous": _audit_revision_continuous(subject_saved),
                "saved_at": (subject_saved or {}).get("saved_at"),
                "audit_log": (subject_saved or {}).get("audit_log", []),
                "unreviewed": subject_eligibility["unreviewed"],
                "training_eligible": subject_eligibility["training_eligible"],
                "release_eligible": subject_eligibility["release_eligible"],
                "view_policy_calibration_eligible": False,
            }
            view_common = {
                "record_id": record_id,
                "action": source.get("action"),
                "modality": modality,
                "revision": (view_saved or {}).get("revision", 0),
                "audit_revision_continuous": _audit_revision_continuous(view_saved),
                "saved_at": (view_saved or {}).get("saved_at"),
                "audit_log": (view_saved or {}).get("audit_log", []),
                "unreviewed": view_eligibility["unreviewed"],
                "training_eligible": view_eligibility["training_eligible"],
                "release_eligible": view_eligibility["release_eligible"],
                "view_policy_calibration_eligible": view_eligibility["view_policy_calibration_eligible"],
            }
            subject_records.append(
                {
                    **subject_common,
                    "overall_target_status": subject_review.get("overall_target_status", "unreviewed"),
                    "same_subject_throughout": subject_review.get("same_subject_throughout", "unsure"),
                    "identity_switch_intervals": subject_review.get("identity_switch_intervals", []),
                    "checkpoints": subject_review.get("checkpoints", []),
                }
            )
            observability_items = view_review.get("observability_items", [])
            view_records.append(
                {
                    **view_common,
                    "original_view": source.get("camera_view"),
                    "original_view_raw": source.get("camera_view_raw"),
                    "confirmed_view": view_review.get("confirmed_view", "unsure"),
                    "action_usability": view_review.get("action_usability", "unsure"),
                    "usable_start_frame": view_review.get("usable_start_frame"),
                    "usable_end_frame": view_review.get("usable_end_frame"),
                    "full_body_visibility": view_review.get("full_body_visibility", "unsure"),
                    "floor_visibility": view_review.get("floor_visibility", "unsure"),
                    "equipment_visibility": view_review.get("equipment_visibility", "unsure"),
                    "observability_items": observability_items,
                }
            )
            error_records.append(
                {
                    "record_id": record_id,
                    "action": source.get("action"),
                    "modality": modality,
                    "revision": (error_saved or {}).get("revision", 0),
                    "audit_revision_continuous": _audit_revision_continuous(error_saved),
                    "saved_at": (error_saved or {}).get("saved_at"),
                    "audit_log": (error_saved or {}).get("audit_log", []),
                    "unreviewed": error_eligibility["unreviewed"],
                    "training_eligible": False,
                    "release_eligible": False,
                    "view_policy_calibration_eligible": False,
                    "recording_intent_code": source.get("recording_intent_code"),
                    "expected_errors_unverified": source.get("expected_errors_unverified", []),
                    "error_truth_status": error_review.get("status", "unreviewed"),
                    "error_truth_items": error_review.get("error_truth_items", []),
                }
            )
            for item in observability_items:
                key = (
                    str(source.get("action")),
                    str(view_review.get("confirmed_view", "unsure")),
                    str(item.get("item_code")),
                )
                group = summary_groups.setdefault(
                    key,
                    {
                        "action": key[0],
                        "view": key[1],
                        "judgement_item": key[2],
                        "modality_review_count": 0,
                        "status_counts": {
                            "OBSERVABLE": 0,
                            "PARTIAL": 0,
                            "UNOBSERVABLE": 0,
                            "UNSURE": 0,
                        },
                    },
                )
                group["modality_review_count"] += 1
                status = str(item.get("status", "UNSURE"))
                if status in group["status_counts"]:
                    group["status_counts"][status] += 1
    base = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "generated_at": generated_at,
    }
    return {
        "view_observability_review_v1.json": {
            **base,
            "artifact_type": "view_observability_review_v1",
            "records": view_records,
        },
        "oni_subject_review_v1.json": {
            **base,
            "artifact_type": "oni_subject_review_v1",
            "records": subject_records,
        },
        "oni_error_truth_review_v1.json": {
            **base,
            "artifact_type": "oni_error_truth_review_v1",
            "records": error_records,
        },
        "view_observability_summary_v1.json": {
            **base,
            "artifact_type": "action_view_judgement_observability_summary_v1",
            "groups": list(summary_groups.values()),
        },
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
    disagreement_cache: dict[str, Any] = {"mtime_ns": None, "clips": {}}
    noncore_proposal_cache: dict[str, dict[str, Any]] = {}
    counting_report = _read_json(
        dataset_root / "reports/human_review_counting_regression_v1.json",
        {"records": []},
    )
    counting_index = {
        str(item["record_id"]): item
        for item in counting_report.get("records", [])
        if isinstance(item, dict) and item.get("record_id")
    }

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

    def disagreement_source_data(
        record_index: dict[str, dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        path = dataset_root / "reports/backend_agreement_v1.json"
        mtime_ns = path.stat().st_mtime_ns if path.exists() else None
        if disagreement_cache["mtime_ns"] != mtime_ns:
            report = _load_full_disagreement_report(
                dataset_root,
                _read_json(path, {"records": []}),
            )
            disagreement_cache["clips"] = _build_disagreement_clips(
                report, record_index
            )
            disagreement_cache["mtime_ns"] = mtime_ns
        return disagreement_cache["clips"]

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
        disagreement_clips = disagreement_source_data(record_index)
        items: list[dict[str, Any]] = []
        for order, (record_id, record) in enumerate(record_index.items(), start=1):
            core = record.get("action") in CORE_ACTIONS
            queue_item = queue_index.get(record_id, {})
            action_code = str(record.get("action"))
            subject_suggestion = _temporary_subject_id(record_id, action_code)
            record_clips = disagreement_clips.get(record_id, [])
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
                    "subject_id_suggestion": subject_suggestion,
                    "dataset_role_suggestion": _temporary_dataset_role(subject_suggestion),
                    "high_disagreement_frame_count": sum(
                        int(clip.get("anchor_frame_count", 0))
                        for clip in record_clips
                    ),
                    "disagreement_clip_count": len(record_clips),
                    "disagreement_review": _disagreement_progress(
                        records_a.get(record_id), record_clips
                    ),
                    "reviewer_a": {
                        "saved": record_id in records_a,
                        "complete": _review_complete(records_a.get(record_id), core=core),
                        "fine_complete": _fine_review_complete(
                            records_a.get(record_id), core=core
                        ),
                        "gates": _record_gate_summary(records_a.get(record_id), core=core),
                    },
                    "reviewer_b": {
                        "saved": record_id in records_b,
                        "complete": _review_complete(records_b.get(record_id), core=core),
                        "fine_complete": _fine_review_complete(
                            records_b.get(record_id), core=core
                        ),
                        "gates": _record_gate_summary(records_b.get(record_id), core=core),
                    },
                }
            )
        items.sort(key=lambda item: (-int(item["priority"]), item["record_id"]))
        oni_items: list[dict[str, Any]] = []
        view_prior_items: list[dict[str, Any]] = []
        error_truth_items: list[dict[str, Any]] = []
        for order, (record_id, record) in enumerate(oni_index.items(), start=1):
            audit_record = oni_audit_index.get(record_id, {})
            for modality in ONI_MODALITIES:
                saved = _read_json(_oni_record_path(review_root, record_id, modality), None)
                view_saved = _read_json(_oni_view_record_path(review_root, record_id, modality), None)
                error_saved = _read_json(_oni_error_record_path(review_root, record_id, modality), None)
                modality_audit = (audit_record.get("modalities") or {}).get(modality, {})
                saved_review_payload = (saved or {}).get("review") or {}
                view_review_payload = (view_saved or {}).get("review") or {}
                error_review_payload = (error_saved or {}).get("review") or {}
                base_item = {
                        "task_id": f"{record_id}__{modality}",
                        "record_id": record_id,
                        "display_id": f"ONI {order:02d}",
                        "modality": modality,
                        "modality_label": ONI_LABELS_ZH["modalities"][modality],
                        "action_candidate": record.get("action"),
                        "action_label": ACTION_LABELS.get(str(record.get("action")), str(record.get("action"))),
                        "camera_view": record.get("camera_view"),
                        "camera_view_label": VIEW_LABELS_ZH.get(str(record.get("camera_view")), str(record.get("camera_view"))),
                        "recording_intent_code": record.get("recording_intent_code"),
                        "checkpoint_count": modality_audit.get("sampled_checkpoint_count", 0),
                        "confidence_p50": modality_audit.get("confidence_p50"),
                        "low_confidence": float(modality_audit.get("confidence_p50") or 0.0) < 0.45,
                }
                oni_items.append(
                    {
                        **base_item,
                        "subject_switch": str(saved_review_payload.get("same_subject_throughout", "")) == "no",
                        "conflict": bool(saved_review_payload.get("conflict")) or not _audit_revision_continuous(saved),
                        "eligibility": _oni_eligibility(saved_review_payload),
                        "saved": saved is not None,
                        "complete": _oni_review_complete(saved),
                    }
                )
                view_prior_items.append(
                    {
                        **base_item,
                        "subject_switch": str(view_review_payload.get("same_subject_throughout", "")) == "no",
                        "conflict": bool(view_review_payload.get("conflict")) or not _audit_revision_continuous(view_saved),
                        "confirmed_view": view_review_payload.get("confirmed_view"),
                        "eligibility": _oni_eligibility(view_review_payload),
                        "saved": view_saved is not None,
                        "complete": _oni_review_complete(view_saved),
                    }
                )
                if record.get("expected_errors_unverified"):
                    error_truth_items.append(
                        {
                            **base_item,
                            "subject_switch": str(error_review_payload.get("same_subject_throughout", "")) == "no",
                            "conflict": bool(error_review_payload.get("conflict")) or not _audit_revision_continuous(error_saved),
                            "eligibility": _oni_eligibility(error_review_payload),
                            "saved": error_saved is not None,
                            "complete": _oni_review_complete(error_saved),
                        }
                    )
        dashboard_groups: dict[tuple[str, str, str], dict[str, Any]] = {}
        for item in view_prior_items:
            key = (
                str(item.get("action_candidate")),
                str(item.get("camera_view") or "unsure"),
                str(item.get("modality")),
            )
            group = dashboard_groups.setdefault(
                key,
                {
                    "action": key[0],
                    "action_label": item.get("action_label"),
                    "view": key[1],
                    "view_label": item.get("camera_view_label"),
                    "modality": key[2],
                    "total": 0,
                    "complete": 0,
                },
            )
            group["total"] += 1
            group["complete"] += int(bool(item.get("complete")))
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
                    "remaining_rgb_fine_annotation": {
                        "label": "人工 4：其余 15 段手机 RGB 精标",
                        "record_count": sum(1 for item in items if not item["core"]),
                    },
                    "high_disagreement_clip_review": {
                        "label": "人工 4：高分歧短片段复核",
                        "record_count": sum(
                            1 for item in items if item["disagreement_clip_count"]
                        ),
                        "frame_count": sum(
                            int(item["high_disagreement_frame_count"])
                            for item in items
                        ),
                        "clip_count": sum(
                            int(item["disagreement_clip_count"]) for item in items
                        ),
                    },
                    "oni_subject_review": {
                        "label": "ONI Depth/IR 主体人工复核",
                        "record_count": len(oni_index),
                        "modality_task_count": len(oni_items),
                    },
                    "oni_view_prior_review": {
                        "label": "ONI 视角先验复核",
                        "record_count": len(oni_index),
                        "modality_task_count": len(view_prior_items),
                    },
                    "oni_error_truth_review": {
                        "label": "ONI 错误真值复核",
                        "record_count": len({item["record_id"] for item in error_truth_items}),
                        "modality_task_count": len(error_truth_items),
                    },
                },
                "records": items,
                "oni_records": oni_items,
                "view_prior_records": view_prior_items,
                "error_truth_records": error_truth_items,
                "dashboard": {
                    "task_completion": [
                        {
                            "task": "phone_rgb_quick_review",
                            "label": "手机 RGB 快速复核",
                            "total": len(items),
                            "complete": sum(bool(item["reviewer_a"]["saved"]) for item in items),
                        },
                        {
                            "task": "phone_rgb_fine_review",
                            "label": "核心 15 条 RGB 精细复核",
                            "total": sum(bool(item["core"]) for item in items),
                            "complete": sum(
                                bool(item["reviewer_a"]["fine_complete"]) for item in items if item["core"]
                            ),
                        },
                        {
                            "task": "phone_rgb_remaining_fine_review",
                            "label": "其余 15 条 RGB 精细复核",
                            "total": sum(not bool(item["core"]) for item in items),
                            "complete": sum(
                                bool(item["reviewer_a"]["fine_complete"]) for item in items if not item["core"]
                            ),
                        },
                        {
                            "task": "phone_rgb_disagreement_review",
                            "label": "高分歧短片段复核",
                            "total": sum(
                                int(item["disagreement_review"]["clip_count"]) for item in items
                            ),
                            "complete": sum(
                                int(item["disagreement_review"]["completed_clip_count"]) for item in items
                            ),
                        },
                        {
                            "task": "oni_subject_review",
                            "label": "ONI 主体复核",
                            "total": len(oni_items),
                            "complete": sum(bool(item["complete"]) for item in oni_items),
                        },
                        {
                            "task": "oni_view_prior_review",
                            "label": "ONI 视角先验复核",
                            "total": len(view_prior_items),
                            "complete": sum(bool(item["complete"]) for item in view_prior_items),
                        },
                        {
                            "task": "oni_error_truth_review",
                            "label": "ONI 错误真值复核",
                            "total": len(error_truth_items),
                            "complete": sum(bool(item["complete"]) for item in error_truth_items),
                        },
                    ],
                    "action_view_modality": list(dashboard_groups.values()),
                },
                "labels": {
                    "actions": ACTION_LABELS,
                    "timeline": TIMELINE_LABELS,
                    "validity": VALIDITY_VALUES,
                    "observability": OBSERVABILITY_VALUES,
                    "views": VIEW_LABELS_ZH,
                    "observability_items_by_action": {
                        action: [{"item_code": code, "label": label} for code, label in items]
                        for action, items in OBSERVABILITY_ITEMS_BY_ACTION.items()
                    },
                    "phases_by_action": PHASES_BY_ACTION,
                    "events_by_action": EVENTS_BY_ACTION,
                    "errors_by_action": ERRORS_BY_ACTION,
                    "phase_labels_zh": PHASE_LABELS_ZH,
                    "error_labels_zh": ERROR_LABELS_ZH,
                    "event_labels_zh": EVENT_LABELS_ZH,
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

            action_proposal = find(action)
            fine_proposal = find(core)
            if fine_proposal is None and str(record.get("action")) in PHONE_ACTIONS - CORE_ACTIONS:
                if record_id not in noncore_proposal_cache:
                    noncore_proposal_cache[record_id] = (
                        _build_noncore_candidate_annotations(
                            dataset_root,
                            record,
                            action_proposal,
                            counting_index.get(record_id),
                        )
                    )
                fine_proposal = noncore_proposal_cache[record_id]
            proposal = {
                "action_segments": action_proposal,
                "core_annotations": fine_proposal,
                "object_scene": find(scene),
                "scoring_correction": find(scoring),
            }
        video = record.get("video", {})
        action_code = str(record.get("action"))
        subject_suggestion = _temporary_subject_id(record_id, action_code)
        record_disagreement_clips = disagreement_source_data(record_index).get(
            record_id, []
        )
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
            "subject_id_suggestion": subject_suggestion,
            "dataset_role_suggestion": _temporary_dataset_role(subject_suggestion),
            "subject_grouping_note": (
                "临时数据分组：滑雪机、推雪橇各分两组，其他动作各一组；"
                "不代表跨动作身份已独立核验。"
            ),
            "disagreement_clips": record_disagreement_clips,
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
        _enrich_frame_values(review, float(record.get("video", {}).get("fps", 0.0) or 0.0))
        path = _record_path(review_root, role, record_id)
        with write_lock:
            old = _read_json(path, None)
            try:
                check_reviewer(old, reviewer_id)
            except PermissionError as exc:
                return json_error(str(exc), 409, "reviewer_conflict")
            old_revision = int((old or {}).get("revision", 0))
            if old and not _audit_revision_continuous(old):
                old = copy.deepcopy(old)
                old["audit_log"] = _repair_audit_log(old)
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
        companion_modality = "ir" if modality == "depth" else "depth"
        companion_checkpoints = _load_oni_checkpoints(dataset_root, record_id, companion_modality)
        audit = audit_index.get(record_id, {})
        review_mode = request.args.get("mode", "subject")
        if review_mode not in {"subject", "view_prior", "error_truth"}:
            return json_error("复核模式必须是 subject、view_prior 或 error_truth", 400, "invalid_review_mode")
        saved_path = {
            "subject": _oni_record_path,
            "view_prior": _oni_view_record_path,
            "error_truth": _oni_error_record_path,
        }[review_mode](review_root, record_id, modality)
        saved = _read_json(saved_path, None)
        return jsonify(
            {
                "record": {
                    "record_id": record_id,
                    "action_candidate": record.get("action"),
                    "action_label": ACTION_LABELS.get(str(record.get("action")), str(record.get("action"))),
                    "modality": modality,
                    "modality_label": ONI_LABELS_ZH["modalities"][modality],
                    "camera_view": record.get("camera_view"),
                    "camera_view_label": VIEW_LABELS_ZH.get(str(record.get("camera_view")), str(record.get("camera_view"))),
                    "camera_view_raw": record.get("camera_view_raw"),
                    "recording_intent_code": record.get("recording_intent_code"),
                    "expected_errors_unverified": record.get("expected_errors_unverified", []),
                    "subject_id_current": record.get("subject_id"),
                    "preview_url": f"/api/review/oni/{record_id}/{modality}/preview",
                    "preview_urls": {
                        item: f"/api/review/oni/{record_id}/{item}/preview"
                        for item in ONI_MODALITIES
                    },
                    "checkpoint_count": len(checkpoints),
                    "modality_audit": (audit.get("modalities") or {}).get(modality, {}),
                    "candidate_loss_intervals": _candidate_loss_intervals(checkpoints),
                },
                "checkpoints": checkpoints,
                "companion_modality": companion_modality,
                "companion_checkpoints": companion_checkpoints,
                "saved_review": saved,
                "review_policy": "single_human_review_sufficient_for_current_stage",
                "review_mode": review_mode,
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
        rows = review.get("checkpoints")
        if not isinstance(rows, list):
            rows = []
        seen_frames: set[int] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                frame = int(row.get("frame_index"))
                seen_frames.add(frame)
            except (TypeError, ValueError):
                continue
        review_mode = str(review.get("review_mode", "subject"))
        if review_mode not in {"subject", "view_prior", "error_truth"}:
            return json_error("review_mode 必须是 subject、view_prior 或 error_truth", 422, "invalid_review_mode")
        path = {
            "subject": _oni_record_path,
            "view_prior": _oni_view_record_path,
            "error_truth": _oni_error_record_path,
        }[review_mode](review_root, record_id, modality)
        with write_lock:
            old = _read_json(path, None)
            try:
                check_reviewer(old, reviewer_id)
            except PermissionError as exc:
                return json_error(str(exc), 409, "reviewer_conflict")
            old_revision = int((old or {}).get("revision", 0))
            if old and not _audit_revision_continuous(old):
                old = copy.deepcopy(old)
                old["audit_log"] = _repair_audit_log(old)
            if body.get("base_revision") not in (None, old_revision):
                return json_error("记录已在其他页面更新，请刷新后重试", 409, "revision_conflict")
            now = _utc_now()
            changes = _diff((old or {}).get("review", {}), review)
            audit_log = list((old or {}).get("audit_log", []))
            evidence_frames = set(seen_frames)
            for item in review.get("observability_items") or []:
                if not isinstance(item, dict):
                    continue
                for frame in item.get("evidence_frames") or []:
                    try:
                        evidence_frames.add(int(frame))
                    except (TypeError, ValueError):
                        continue
            audit_log.append(
                {
                    "revision": old_revision + 1,
                    "reviewer_id": reviewer_id,
                    "reviewed_at": now,
                    "reason": str(body.get("change_reason") or "ONI subject/view prior review update"),
                    "evidence_frames": sorted(evidence_frames),
                    "changes": changes,
                }
            )
            eligibility = _oni_eligibility(review)
            payload = {
                "schema_version": REVIEW_SCHEMA_VERSION,
                "artifact_type": (
                    "oni_modality_view_prior_human_review"
                    if str(review.get("review_mode", "")) == "view_prior"
                    else (
                        "oni_modality_error_truth_human_review"
                        if str(review.get("review_mode", "")) == "error_truth"
                        else "oni_modality_subject_human_review"
                    )
                ),
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
                "eligibility": eligibility,
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
        oni_index, _ = oni_source_data()
        with write_lock:
            independent_exports = _build_review_exports(review_root, oni_index)
            export_directory = review_root / VALID_ROLES["a"] / "exports"
            for filename, payload in independent_exports.items():
                _atomic_json(export_directory / filename, payload)
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

    @blueprint.get("/api/review/export/<export_name>")
    def export_independent_review(export_name: str) -> Response | tuple[Response, int]:
        _, error = require_bound_session("a", "adjudicator")
        if error:
            return error
        filename = f"{export_name}.json" if not export_name.endswith(".json") else export_name
        oni_index, _ = oni_source_data()
        exports = _build_review_exports(review_root, oni_index)
        if filename not in exports:
            return json_error("独立导出类型不存在", 404, "export_not_found")
        with write_lock:
            path = review_root / VALID_ROLES["a"] / "exports" / filename
            _atomic_json(path, exports[filename])
        return send_file(path, mimetype="application/json", as_attachment=True, download_name=filename, max_age=0)

    return blueprint
