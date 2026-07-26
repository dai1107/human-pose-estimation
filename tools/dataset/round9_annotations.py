from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


CORE_ACTIONS = {"burpee_broad_jump", "lunge", "wall_ball"}
TIMELINE_LABELS = {
    "idle",
    "setup",
    "target_action",
    "transition",
    "unknown_motion",
    "target_out_of_frame",
}
PROPOSAL_LAYERS = (
    "filename_prior",
    "rule_proposal",
    "dtw_proposal",
    "teacher_pose_proposal",
    "object_detector_proposal",
    "human_annotation",
    "reviewed_ground_truth",
)

PHASES = {
    "lunge": ("stand", "descent", "bottom", "contact", "ascent", "stand"),
    "burpee_broad_jump": (
        "hands_down",
        "chest_down",
        "takeoff",
        "flight",
        "landing",
        "stabilization",
    ),
    "wall_ball": ("stand", "descent", "bottom", "ascent", "release", "recovery"),
}

EVENTS = {
    "lunge": ("rep_start", "bottom_reached", "rear_knee_contact_candidate", "full_extension"),
    "burpee_broad_jump": (
        "hands_down",
        "chest_contact_candidate",
        "takeoff_candidate",
        "landing_candidate",
    ),
    "wall_ball": ("rep_start", "bottom_reached", "ball_release_candidate", "recovery"),
}

ERROR_TO_CORRECTION = {
    "FOOT_DESYNCHRONIZED": (
        "burpee_takeoff_together_v1",
        "起跳时双脚同时离地",
        "takeoff",
        "feet",
    ),
    "HANDS_FEET_TOO_FAR": (
        "burpee_land_feet_near_hands_v1",
        "收腿时让双脚落到双手附近",
        "takeoff",
        "hands_and_feet",
    ),
    "NO_CHEST_CONTACT": (
        "burpee_chest_to_floor_v1",
        "俯卧阶段让胸部明确触地",
        "chest_down",
        "chest",
    ),
    "EXTRA_STEP": (
        "stabilize_without_extra_step_v1",
        "落地后先稳定，避免补步",
        "stabilization",
        "feet",
    ),
    "SAME_LEG_CONSECUTIVE": (
        "lunge_alternate_legs_v1",
        "下一次换腿迈步",
        "stand",
        "legs",
    ),
    "NO_KNEE_CONTACT": (
        "lunge_rear_knee_contact_v1",
        "下降至后膝轻触地面",
        "bottom",
        "rear_knee",
    ),
    "HIP_NOT_EXTENDED": (
        "lunge_finish_tall_v1",
        "站起结束时充分伸髋",
        "stand",
        "hip",
    ),
    "NOT_DEEP_ENOUGH": (
        "wall_ball_squat_below_parallel_v1",
        "下蹲时让髋部继续下降至膝线以下",
        "bottom",
        "hip_and_knee",
    ),
    "HEEL_RISE": (
        "wall_ball_keep_heels_grounded_v1",
        "下蹲阶段保持脚跟稳定着地",
        "descent",
        "feet",
    ),
}

ACTION_OBJECTS = {
    "burpee_broad_jump": ("floor_region", "lane_or_finish_line"),
    "lunge": ("lunge_load", "floor_region", "lane_or_finish_line"),
    "wall_ball": ("wall_ball", "wall_ball_target", "floor_region"),
    "rowing": ("erg_handle", "erg_display_roi"),
    "farmers_carry": ("farmers_carry_weight", "lane_or_finish_line"),
    "skierg": ("erg_handle", "erg_display_roi"),
    "sled_push": ("sled", "lane_or_finish_line"),
    "sled_pull": ("sled", "sled_rope", "lane_or_finish_line"),
}

RUBRICS = {
    "burpee_broad_jump": {
        "version": "burpee_broad_jump_round9_trial_v1",
        "weights": {"completion": 0.4, "control": 0.25, "symmetry": 0.2, "technique": 0.15},
    },
    "lunge": {
        "version": "lunge_round9_trial_v1",
        "weights": {"completion": 0.4, "control": 0.2, "symmetry": 0.2, "technique": 0.2},
    },
    "wall_ball": {
        "version": "wall_ball_round9_trial_v1",
        "weights": {"completion": 0.4, "control": 0.2, "symmetry": 0.1, "technique": 0.3},
    },
}


@dataclass(frozen=True)
class PoseSample:
    frame_index: int
    timestamp_ms: float
    target_locked: bool
    formal_pose_eligible: bool
    joints: dict[str, tuple[float, float, float]]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_pose(path: Path) -> list[PoseSample]:
    samples: list[PoseSample] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            joints = {
                item["name"]: (
                    float(item["x"]),
                    float(item["y"]),
                    float(item.get("visibility", item.get("confidence", 0.0))),
                )
                for item in row.get("image_normalized_2d", [])
            }
            samples.append(
                PoseSample(
                    frame_index=int(row["frame_index"]),
                    timestamp_ms=float(row["source_timestamp_ms"]),
                    target_locked=bool(row.get("target_locked")),
                    formal_pose_eligible=bool(row.get("formal_pose_eligible")),
                    joints=joints,
                )
            )
    return samples


def _mean_joint(sample: PoseSample, names: Sequence[str], axis: int) -> float | None:
    values = [
        sample.joints[name][axis]
        for name in names
        if name in sample.joints and sample.joints[name][2] >= 0.35
    ]
    return sum(values) / len(values) if values else None


def _fill(values: Sequence[float | None]) -> list[float]:
    result = list(values)
    last: float | None = None
    for index, value in enumerate(result):
        if value is not None and math.isfinite(value):
            last = value
        elif last is not None:
            result[index] = last
    last = None
    for index in range(len(result) - 1, -1, -1):
        value = result[index]
        if value is not None and math.isfinite(value):
            last = value
        elif last is not None:
            result[index] = last
    return [float(value) if value is not None else 0.0 for value in result]


def _smooth(values: Sequence[float], radius: int = 4) -> list[float]:
    if not values:
        return []
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    result = []
    for index in range(len(values)):
        lo = max(0, index - radius)
        hi = min(len(values), index + radius + 1)
        result.append((prefix[hi] - prefix[lo]) / (hi - lo))
    return result


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[position]


def _active_bounds(samples: Sequence[PoseSample]) -> tuple[int, int, list[dict[str, int]]]:
    valid = [sample.frame_index for sample in samples if sample.formal_pose_eligible]
    if not valid:
        return 0, max(0, len(samples) - 1), [{"start_frame": 0, "end_frame": max(0, len(samples) - 1)}]

    centers_x = _fill(
        [
            _mean_joint(sample, ("left_hip", "right_hip", "left_shoulder", "right_shoulder"), 0)
            for sample in samples
        ]
    )
    centers_y = _fill(
        [
            _mean_joint(sample, ("left_hip", "right_hip", "left_shoulder", "right_shoulder"), 1)
            for sample in samples
        ]
    )
    energy = [0.0]
    for index in range(1, len(samples)):
        dx = centers_x[index] - centers_x[index - 1]
        dy = centers_y[index] - centers_y[index - 1]
        energy.append(math.hypot(dx, dy))
    energy = _smooth(energy, 8)
    positive = [value for value in energy if value > 1e-6]
    threshold = max(0.0007, _quantile(positive, 0.35) if positive else 0.0007)
    moving = [index for index, value in enumerate(energy) if value >= threshold]
    start = max(valid[0], (moving[0] - 15) if moving else valid[0])
    end = min(valid[-1], (moving[-1] + 15) if moving else valid[-1])
    if end - start < max(20, len(samples) // 3):
        start, end = valid[0], valid[-1]

    gaps: list[dict[str, int]] = []
    gap_start: int | None = None
    for index in range(start, end + 1):
        eligible = samples[index].formal_pose_eligible
        if not eligible and gap_start is None:
            gap_start = index
        if eligible and gap_start is not None:
            if index - gap_start >= 3:
                gaps.append({"start_frame": gap_start, "end_frame": index - 1})
            gap_start = None
    if gap_start is not None:
        gaps.append({"start_frame": gap_start, "end_frame": end})
    return start, end, gaps


def _find_peaks(
    values: Sequence[float],
    start: int,
    end: int,
    *,
    min_distance: int,
    minimum_prominence: float,
) -> list[int]:
    smoothed = _smooth(values, 3)
    radius = max(5, min_distance // 3)
    candidates: list[tuple[float, int]] = []
    for index in range(max(start + radius, 1), min(end - radius, len(values) - 2) + 1):
        if smoothed[index] < smoothed[index - 1] or smoothed[index] < smoothed[index + 1]:
            continue
        local_floor = min(
            min(smoothed[index - radius : index]),
            min(smoothed[index + 1 : index + radius + 1]),
        )
        prominence = smoothed[index] - local_floor
        if prominence >= minimum_prominence:
            candidates.append((prominence, index))
    selected: list[int] = []
    for _, index in sorted(candidates, reverse=True):
        if all(abs(index - other) >= min_distance for other in selected):
            selected.append(index)
    return sorted(selected)


def _joint_angle(
    sample: PoseSample, first: str, vertex: str, third: str
) -> float | None:
    angles: list[float] = []
    for side in ("left", "right"):
        a = sample.joints.get(f"{side}_{first}")
        b = sample.joints.get(f"{side}_{vertex}")
        c = sample.joints.get(f"{side}_{third}")
        if not a or not b or not c or min(a[2], b[2], c[2]) < 0.35:
            continue
        first_vector = (a[0] - b[0], a[1] - b[1])
        third_vector = (c[0] - b[0], c[1] - b[1])
        denominator = math.hypot(*first_vector) * math.hypot(*third_vector)
        if denominator <= 1e-9:
            continue
        cosine = max(
            -1.0,
            min(
                1.0,
                (
                    first_vector[0] * third_vector[0]
                    + first_vector[1] * third_vector[1]
                )
                / denominator,
            ),
        )
        angles.append(math.degrees(math.acos(cosine)))
    return sum(angles) / len(angles) if angles else None


def _burpee_chest_anchors(
    samples: Sequence[PoseSample], start: int, end: int
) -> list[int]:
    candidates: list[int] = []
    for index in range(start, end + 1):
        sample = samples[index]
        shoulder = _mean_joint(sample, ("left_shoulder", "right_shoulder"), 1)
        hip = _mean_joint(sample, ("left_hip", "right_hip"), 1)
        wrist = _mean_joint(sample, ("left_wrist", "right_wrist"), 1)
        if shoulder is None or hip is None or wrist is None:
            continue
        body_values = [
            value
            for value in (
                shoulder,
                hip,
                _mean_joint(sample, ("left_knee", "right_knee"), 1),
                _mean_joint(sample, ("left_ankle", "right_ankle"), 1),
            )
            if value is not None
        ]
        elbow_angle = _joint_angle(sample, "shoulder", "elbow", "wrist")
        if (
            max(body_values) - min(body_values) <= 0.09
            and shoulder - wrist >= -0.10
            and elbow_angle is not None
            and elbow_angle <= 130.0
        ):
            candidates.append(index)

    groups: list[list[int]] = []
    for index in candidates:
        if not groups or index - groups[-1][-1] > 5:
            groups.append([index])
        else:
            groups[-1].append(index)
    anchors: list[int] = []
    for group in groups:
        if len(group) < 2:
            continue
        anchor = max(
            group,
            key=lambda index: (
                _mean_joint(
                    samples[index], ("left_shoulder", "right_shoulder"), 1
                )
                or 0.0
            ),
        )
        if not anchors or anchor - anchors[-1] >= 40:
            anchors.append(anchor)
    return anchors


def _rep_anchors(action: str, samples: Sequence[PoseSample], start: int, end: int) -> list[int]:
    if action == "burpee_broad_jump":
        chest_anchors = _burpee_chest_anchors(samples, start, end)
        if chest_anchors:
            return chest_anchors
        names, min_distance, prominence = ("left_shoulder", "right_shoulder"), 42, 0.055
    elif action == "lunge":
        names, min_distance, prominence = ("left_hip", "right_hip"), 24, 0.008
    else:
        names, min_distance, prominence = ("left_hip", "right_hip"), 28, 0.025
    signal = _fill([_mean_joint(sample, names, 1) for sample in samples])
    peaks = _find_peaks(
        signal,
        start,
        end,
        min_distance=min_distance,
        minimum_prominence=prominence,
    )
    if not peaks:
        peaks = [round((start + end) / 2)]
    return peaks


def _phase_segments(
    phases: Sequence[str], rep_start: int, anchor: int, rep_end: int
) -> list[dict[str, Any]]:
    length = max(1, rep_end - rep_start + 1)
    if length < len(phases):
        phases = phases[:length]
    points = [
        rep_start + round(length * index / len(phases))
        for index in range(len(phases) + 1)
    ]
    points[0], points[-1] = rep_start, rep_end + 1
    return [
        {
            "phase": phase,
            "start_frame": points[index],
            "end_frame": max(points[index], points[index + 1] - 1),
            "rep_anchor_frame": anchor,
            "annotation_source": "teacher_pose_proposal",
            "review_status": "human_review_pending",
        }
        for index, phase in enumerate(phases)
    ]


def _source_error_codes(record: dict[str, Any]) -> list[str]:
    codes = list(record.get("expected_errors_unverified", []))
    if record["record_id"] == "phone_lunge_004":
        codes.append("EXTRA_STEP")
    return codes


def _hash_payload(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _artifact(kind: str, now: str, **fields: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": kind,
        "generated_at": now,
        **fields,
    }


def build_round9_artifacts(dataset_root: Path) -> dict[Path, dict[str, Any]]:
    manifest = _read_json(dataset_root / "manifests" / "phone_records.json")
    object_candidates = _read_json(dataset_root / "reports" / "object_scene_visibility_v1.json")
    object_by_record = {row["record_id"]: row for row in object_candidates["records"]}
    now = _utc_now()

    action_rows: list[dict[str, Any]] = []
    core_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    scoring_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    total_reps = 0
    total_errors = 0

    for record in manifest["records"]:
        record_id = record["record_id"]
        frame_count = int(record["video"]["decoded_frame_count"])
        fps = float(record["video"]["fps"])
        pose_path = dataset_root / "pose_cache" / record_id / "causal_analysis_pose.jsonl.gz"
        samples = _load_pose(pose_path)
        if len(samples) != frame_count:
            raise ValueError(f"{record_id}: pose frame count {len(samples)} != {frame_count}")
        active_start, active_end, gaps = _active_bounds(samples)

        segments: list[dict[str, Any]] = []
        if active_start > 0:
            segments.append(
                {
                    "timeline_label": "setup",
                    "start_frame": 0,
                    "end_frame": active_start - 1,
                    "action_type": record["action"],
                    "action_confidence": 0.6,
                }
            )
        segments.append(
            {
                "timeline_label": "target_action",
                "start_frame": active_start,
                "end_frame": active_end,
                "action_type": record["action"],
                "action_confidence": 0.75,
            }
        )
        if active_end < frame_count - 1:
            segments.append(
                {
                    "timeline_label": "transition",
                    "start_frame": active_end + 1,
                    "end_frame": frame_count - 1,
                    "action_type": record["action"],
                    "action_confidence": 0.55,
                }
            )
        for segment in segments:
            segment.update(
                {
                    "annotator_id": "codex_assistant",
                    "annotator_type": "ai_assisted_visual_and_pose_review",
                    "review_status": "human_confirmation_pending",
                    "human_confirmed": False,
                }
            )
        action_rows.append(
            {
                "record_id": record_id,
                "source_filename": record["source_filename"],
                "action_type": record["action"],
                "recording_intent": record["recording_intent"],
                "recording_intent_raw": record["recording_intent_raw"],
                "recording_intent_verified": False,
                "target_track_id": record["target_athlete"]["track_id"],
                "target_identity_status": record["review_status"]["subject_identity"],
                "frame_count": frame_count,
                "fps": fps,
                "segments": segments,
                "target_out_of_frame_candidates": gaps,
                "video_action_review_status": "ai_first_pass_complete_human_confirmation_pending",
                "training_eligible": False,
            }
        )
        action_counts[record["action"]] += 1

        candidate = object_by_record.get(record_id, {})
        expected_objects = ACTION_OBJECTS.get(record["action"], ())
        evidence = []
        for object_class in expected_objects:
            candidate_count = int(candidate.get("candidate_counts", {}).get(object_class, 0))
            evidence.append(
                {
                    "object_class": object_class,
                    "object_track_id": None,
                    "start_frame": active_start,
                    "end_frame": active_end,
                    "object_visible": "unknown",
                    "athlete_object_contact": "UNOBSERVABLE",
                    "target_region": "UNOBSERVABLE",
                    "line_crossing": "UNOBSERVABLE",
                    "display_reading": None,
                    "observability": "UNOBSERVABLE",
                    "evidence_source": "object_detector_proposal",
                    "candidate_frame_count": candidate_count,
                    "review_status": "human_confirmation_pending",
                    "rule_truth_generated": False,
                }
            )
        object_rows.append(
            {
                "record_id": record_id,
                "action": record["action"],
                "target_track_id": record["target_athlete"]["track_id"],
                "evidence": evidence,
                "unknown_weight_distance_power_policy": "never_infer_from_appearance",
            }
        )

        disagreement_count = 0
        agreement_path = dataset_root / "pose_cache" / record_id / "backend_agreement.jsonl.gz"
        if agreement_path.exists():
            with gzip.open(agreement_path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("review_required"):
                        disagreement_count += 1
        queue_rows.append(
            {
                "record_id": record_id,
                "priority": disagreement_count,
                "reasons": (
                    ["multi_backend_disagreement"] if disagreement_count else ["boundary_review"]
                ),
                "active_boundary_frames": [active_start, active_end],
                "review_status": "human_review_pending",
            }
        )

        if record["action"] not in CORE_ACTIONS:
            continue

        anchors = _rep_anchors(record["action"], samples, active_start, active_end)
        boundaries = [active_start]
        boundaries.extend(round((left + right) / 2) for left, right in zip(anchors, anchors[1:]))
        boundaries.append(active_end + 1)
        reps: list[dict[str, Any]] = []
        source_errors = _source_error_codes(record)
        for index, anchor in enumerate(anchors):
            rep_start = boundaries[index]
            rep_end = boundaries[index + 1] - 1
            phases = _phase_segments(PHASES[record["action"]], rep_start, anchor, rep_end)
            phase_centers = [
                round((phase["start_frame"] + phase["end_frame"]) / 2) for phase in phases
            ]
            events = []
            for event_index, event_type in enumerate(EVENTS[record["action"]]):
                event_frame = phase_centers[min(event_index, len(phase_centers) - 1)]
                events.append(
                    {
                        "event_type": event_type,
                        "frame_index": event_frame,
                        "timestamp_ms": round(event_frame * 1000.0 / fps, 3),
                        "observability": "POSE_PROXY_ONLY",
                        "evidence_source": "teacher_pose_proposal",
                        "review_status": "independent_double_review_pending",
                        "is_ground_truth": False,
                    }
                )
            errors = []
            for error_code in source_errors:
                correction = ERROR_TO_CORRECTION.get(error_code)
                errors.append(
                    {
                        "error_code": error_code,
                        "start_frame": rep_start,
                        "end_frame": rep_end,
                        "severity": "UNSURE",
                        "affected_side": "unknown",
                        "confidence": 0.35,
                        "criterion_id": f"{record['action']}.{error_code.lower()}.round9_trial_v1",
                        "phase": correction[2] if correction else "unknown",
                        "measured_value": None,
                        "unit": None,
                        "pass_range": None,
                        "fail_range": None,
                        "observability": "UNREVIEWED",
                        "evidence_source": "filename_prior",
                        "annotator_id": None,
                        "review_status": "human_review_pending",
                        "is_ground_truth": False,
                    }
                )
            reps.append(
                {
                    "rep_id": f"{record_id}_rep_{index + 1:03d}",
                    "start_frame": rep_start,
                    "end_frame": rep_end,
                    "validity": "UNSURE",
                    "target_track_id": record["target_athlete"]["track_id"],
                    "phases": phases,
                    "events": events,
                    "errors": errors,
                    "review_status": "human_frame_review_pending",
                    "training_eligible": False,
                }
            )
            total_errors += len(errors)
        total_reps += len(reps)
        core_rows.append(
            {
                "record_id": record_id,
                "source_filename": record["source_filename"],
                "action": record["action"],
                "frame_count": frame_count,
                "target_track_id": record["target_athlete"]["track_id"],
                "recording_intent_prior": {
                    "intent": record["recording_intent"],
                    "raw": record["recording_intent_raw"],
                    "expected_errors_unverified": source_errors,
                    "must_not_be_used_as_ground_truth": True,
                },
                "reps": reps,
                "core_annotation_status": "structured_proposal_complete_human_frame_review_pending",
            }
        )

        rubric = RUBRICS[record["action"]]
        proposed_components = {
            "completion_score": None,
            "control_score": None,
            "symmetry_score": None,
            "technique_score": None,
            "overall_score": None,
        }
        corrections = []
        for error_code in source_errors:
            correction = ERROR_TO_CORRECTION.get(error_code)
            if not correction:
                continue
            corrections.append(
                {
                    "correction_id": correction[0],
                    "trigger_error_code": error_code,
                    "priority": 1,
                    "cue_short": correction[1],
                    "cue_detailed": correction[1],
                    "target_phase": correction[2],
                    "body_part": correction[3],
                    "expected_change": f"reduce_{error_code.lower()}",
                    "suppress_if": [
                        "low_confidence",
                        "target_lost",
                        "unsupported_view",
                        "required_equipment_unobservable",
                    ],
                    "review_status": "coach_review_pending",
                }
            )
        scoring_rows.append(
            {
                "record_id": record_id,
                "action": record["action"],
                "validity_gate": "UNSURE",
                **proposed_components,
                "score_confidence": 0.0,
                "unobservable_components": list(proposed_components),
                "scoring_rubric_version": rubric["version"],
                "weights": rubric["weights"],
                "correction_candidates": corrections,
                "cue_issued": None,
                "next_rep_error_change": None,
                "athlete_response": None,
                "review_status": "two_rule_knowledgeable_reviewers_pending",
            }
        )

    queue_rows.sort(key=lambda row: (-row["priority"], row["record_id"]))
    action_payload = _artifact(
        "action_segments_v1",
        now,
        status="structured_ai_assisted_first_pass_human_confirmation_pending",
        allowed_labels=sorted(TIMELINE_LABELS),
        record_count=len(action_rows),
        human_confirmed_record_count=0,
        records=action_rows,
    )
    core_payload = _artifact(
        "core_rep_phase_event_error_v1",
        now,
        status="proposals_complete_ground_truth_pending",
        core_actions=sorted(CORE_ACTIONS),
        record_count=len(core_rows),
        rep_proposal_count=total_reps,
        error_proposal_count=total_errors,
        proposal_is_ground_truth=False,
        records=core_rows,
    )
    object_payload = _artifact(
        "object_scene_evidence_v1",
        now,
        status="candidate_layer_complete_human_confirmation_pending",
        record_count=len(object_rows),
        unobservable_is_not_pass=True,
        records=object_rows,
    )
    scoring_payload = _artifact(
        "scoring_correction_v1",
        now,
        status="rubric_and_correction_schema_complete_dual_expert_scoring_pending",
        records=scoring_rows,
    )
    agreement_payload = _artifact(
        "annotation_agreement_v1",
        now,
        status="not_computable_independent_human_review_pending",
        eligible_reviewer_count=0,
        ai_assisted_reviewer_count=1,
        event_anchor_agreement=None,
        error_label_agreement=None,
        scoring_agreement=None,
        reason="No independent human reviewers have supplied annotations; AI output is not counted as human agreement.",
        release_gate_passed=False,
    )
    proposal_payload = _artifact(
        "proposal_acceptance_bias_v1",
        now,
        status="awaiting_human_decisions",
        proposal_layers=list(PROPOSAL_LAYERS),
        proposal_counts={
            "filename_prior": total_errors,
            "teacher_pose_proposal": total_reps,
            "object_detector_proposal": sum(
                len(record["evidence"]) for record in object_rows
            ),
        },
        accepted=None,
        modified=None,
        rejected=None,
        performance_evaluation_allowed=False,
        leakage_guard="A proposal shown to an annotator may not be reused as independent evidence.",
    )
    gap_payload = _artifact(
        "continuous_ood_gap_v1",
        now,
        status="data_not_ready",
        current_record_count=len(action_rows),
        current_action_counts=dict(sorted(action_counts.items())),
        observed_continuous_mixed_action_records=0,
        independently_reviewed_idle_records=0,
        independently_reviewed_transition_records=0,
        independently_reviewed_unknown_ood_records=0,
        blocking_gaps=[
            "continuous transitions among multiple HYROX actions",
            "non-HYROX exercise and arbitrary-motion OOD",
            "idle/setup clips with background people",
            "unknown equipment and unsupported camera views",
            "independent subjects, devices, distances, lighting and occlusion",
        ],
        prioritized_capture_plan=[
            {"priority": 1, "scenario": "8-action continuous circuit with idle and transitions"},
            {"priority": 2, "scenario": "non-HYROX exercises plus arbitrary unknown motion"},
            {"priority": 3, "scenario": "background athlete moves while target is idle"},
            {"priority": 4, "scenario": "SkiErg/Sled Push/Sled Pull errors and boundary cases"},
        ],
        auto_action_training_ready=False,
    )
    queue_payload = _artifact(
        "round9_active_review_queue_v1",
        now,
        status="ready_for_human_review",
        selection_policy="multi-backend disagreement first, then action boundaries",
        records=queue_rows,
    )

    outputs = {
        dataset_root / "annotations" / "action_segments_v1.json": action_payload,
        dataset_root
        / "annotations"
        / "core_rep_phase_event_error_v1.json": core_payload,
        dataset_root / "annotations" / "object_scene_evidence_v1.json": object_payload,
        dataset_root / "annotations" / "scoring_correction_v1.json": scoring_payload,
        dataset_root / "reports" / "annotation_agreement_v1.json": agreement_payload,
        dataset_root / "reports" / "proposal_acceptance_bias_v1.json": proposal_payload,
        dataset_root / "reports" / "continuous_ood_gap_v1.json": gap_payload,
        dataset_root / "reports" / "round9_active_review_queue_v1.json": queue_payload,
    }
    summary = _artifact(
        "round9_implementation_summary",
        now,
        status="engineering_complete_human_ground_truth_and_double_review_pending",
        required_artifacts_present=True,
        record_count=len(action_rows),
        core_record_count=len(core_rows),
        core_rep_proposal_count=total_reps,
        human_confirmed_record_count=0,
        independent_human_reviewer_count=0,
        training_eligible_record_count=0,
        release_gate_passed=False,
        release_blockers=[
            "30/30 video action intervals require human confirmation",
            "15/15 core records require frame-level human review",
            "key events, errors and score subset require two independent reviewers",
            "usage authorization and subject identity remain pending",
        ],
        artifact_hashes={
            path.name: _hash_payload(payload) for path, payload in outputs.items()
        },
        safeguards={
            "filename_prior_is_ground_truth": False,
            "proposal_is_ground_truth": False,
            "unobservable_equipment_treated_as_pass": False,
            "ai_reviewer_counted_as_human": False,
            "training_before_release_gate": False,
        },
    )
    outputs[dataset_root / "reports" / "round9_implementation_summary.json"] = summary
    return outputs


def validate_round9_artifacts(outputs: dict[Path, dict[str, Any]]) -> None:
    by_name = {path.name: payload for path, payload in outputs.items()}
    action = by_name["action_segments_v1.json"]
    core = by_name["core_rep_phase_event_error_v1.json"]
    objects = by_name["object_scene_evidence_v1.json"]
    scoring = by_name["scoring_correction_v1.json"]
    if action["record_count"] != 30 or len(action["records"]) != 30:
        raise ValueError("Round 9 requires exactly 30 phone records")
    if core["record_count"] != 15 or len(core["records"]) != 15:
        raise ValueError("Round 9 core pilot requires exactly 15 records")
    if len(objects["records"]) != 30 or len(scoring["records"]) != 15:
        raise ValueError("Object/scoring record coverage is incomplete")
    for record in action["records"]:
        expected = 0
        for segment in record["segments"]:
            if segment["timeline_label"] not in TIMELINE_LABELS:
                raise ValueError(f"Unknown timeline label: {segment['timeline_label']}")
            if segment["start_frame"] != expected:
                raise ValueError(f"{record['record_id']}: non-contiguous timeline")
            expected = segment["end_frame"] + 1
        if expected != record["frame_count"]:
            raise ValueError(f"{record['record_id']}: timeline does not cover every frame")
        if record["training_eligible"]:
            raise ValueError("Unreviewed Round 9 proposal became training eligible")
    for record in core["records"]:
        if not record["reps"]:
            raise ValueError(f"{record['record_id']}: no rep proposal")
        for rep in record["reps"]:
            if rep["validity"] != "UNSURE" or rep["training_eligible"]:
                raise ValueError("Unreviewed rep must remain UNSURE and training-ineligible")
            if rep["phases"][0]["start_frame"] != rep["start_frame"]:
                raise ValueError("Phase proposal does not start at the rep boundary")
            if rep["phases"][-1]["end_frame"] != rep["end_frame"]:
                raise ValueError("Phase proposal does not end at the rep boundary")
            for left, right in zip(rep["phases"], rep["phases"][1:]):
                if left["end_frame"] + 1 != right["start_frame"]:
                    raise ValueError("Phase proposal is not contiguous")
            if any(error["is_ground_truth"] for error in rep["errors"]):
                raise ValueError("Filename prior was promoted to ground truth")


def run_round9(dataset_root: Path) -> dict[str, Any]:
    outputs = build_round9_artifacts(dataset_root)
    validate_round9_artifacts(outputs)
    for path, payload in outputs.items():
        _write_json(path, payload)
    return outputs[dataset_root / "reports" / "round9_implementation_summary.json"]
