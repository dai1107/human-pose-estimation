"""Evaluate the formal HYROX guidance rules against reviewed phone-RGB labels."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from hyrox.features import extract_basic_pose_features
from hyrox.config import load_lunge_config, load_wall_ball_config
from hyrox.registry import create_action_analyzer
from src.backends.base import Keypoint, PoseResult
from src.biomechanics.kinematics_3d import ThreeDKinematicsTracker
from src.biomechanics.shadow_evidence_3d import ShadowEvidence3DConfig
from src.product_pose import ThreeDKinematicsConfig, ThreeDQualityConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_ACTIONS = {"lunge", "burpee_broad_jump", "wall_ball"}
ERROR_SIGNALS: dict[str, frozenset[str]] = {
    "NO_KNEE_CONTACT": frozenset({"TRAILING_KNEE_NO_CONTACT"}),
    "SAME_LEG_CONSECUTIVE": frozenset({"SAME_CONTACT_LEG_REPEATED"}),
    "EXTRA_STEP": frozenset({"EXTRA_STEP_OR_SHUFFLE", "EXTRA_STEPS"}),
    "HIP_NOT_EXTENDED": frozenset(
        {"FULL_HIP_EXTENSION_NOT_HELD", "STAND_EXTENSION"}
    ),
    "NOT_DEEP_ENOUGH": frozenset({"HIP_NOT_BELOW_KNEE", "SQUAT_NOT_DEEP"}),
    "HEEL_RISE": frozenset({"HEEL_RISE"}),
    "FOOT_DESYNCHRONIZED": frozenset(
        {
            "SIMULTANEOUS_TAKEOFF_ASYNCHRONOUS",
            "SIMULTANEOUS_LANDING_ASYNCHRONOUS",
            "TAKEOFF_STAGGER_PROXY_FAIL",
            "LANDING_STAGGER_PROXY_FAIL",
            "FEET_STAGGERED",
        }
    ),
    "NO_CHEST_CONTACT": frozenset(
        {"CHEST_NOT_LOW"}
    ),
    "HANDS_FEET_TOO_FAR": frozenset(
        {
            "LEGAL_HAND_PLACEMENT_PROXY_TOO_FAR",
            "HANDS_FEET_TOO_FAR",
        }
    ),
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feedback_codes(state: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for item in state.get("feedback_messages") or []:
        if isinstance(item, Mapping):
            code = item.get("code")
        else:
            code = getattr(item, "code", None)
        if code:
            result.add(str(code))
    return result


def _human_errors(record: Mapping[str, Any]) -> list[str]:
    return sorted(
        {
            str(item.get("error_code"))
            for item in record.get("phase_error_intervals") or []
            if isinstance(item, Mapping)
            and str(item.get("error_code", "")) not in {"", "NO_ERROR"}
        }
    )


def _match_statuses(
    human_reps: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Monotonically align runtime candidates to reviewed closed intervals."""

    ordered_human = sorted(
        human_reps,
        key=lambda item: (
            int(item.get("start_frame", 0)),
            int(item.get("end_frame", 0)),
        ),
    )
    ordered_candidates = sorted(
        candidates,
        key=lambda item: (
            int(item.get("alignment_frame", item["source_frame"])),
            int(item["source_frame"]),
        ),
    )
    matches: list[dict[str, Any]] = []
    human_index = 0
    candidate_index = 0
    while (
        human_index < len(ordered_human)
        and candidate_index < len(ordered_candidates)
    ):
        human = ordered_human[human_index]
        candidate = ordered_candidates[candidate_index]
        frame = int(candidate.get("alignment_frame", candidate["source_frame"]))
        start = int(human["start_frame"])
        end = int(human["end_frame"])
        tolerance = max(5, min(30, round((end - start + 1) * 0.25)))
        if frame < start - tolerance:
            matches.append(
                {
                    "candidate": candidate,
                    "human_rep": None,
                    "terminal_frame_error": None,
                    "status_match": False,
                }
            )
            candidate_index += 1
            continue
        if frame > end + tolerance:
            matches.append(
                {
                    "candidate": None,
                    "human_rep": human,
                    "terminal_frame_error": None,
                    "status_match": False,
                }
            )
            human_index += 1
            continue
        matches.append(
            {
                "candidate": candidate,
                "human_rep": human,
                "terminal_frame_error": frame - int(human["end_frame"]),
                "status_match": (
                    str(candidate.get("status")) == str(human.get("validity"))
                ),
            }
        )
        human_index += 1
        candidate_index += 1
    for candidate in ordered_candidates[candidate_index:]:
        matches.append(
            {
                "candidate": candidate,
                "human_rep": None,
                "terminal_frame_error": None,
                "status_match": False,
            }
        )
    for human in ordered_human[human_index:]:
        matches.append(
            {
                "candidate": None,
                "human_rep": human,
                "terminal_frame_error": None,
                "status_match": False,
            }
        )
    return matches


def _candidate_alignment_frame(candidate: Mapping[str, Any]) -> int:
    """Prefer the biomechanical terminal event over delayed rule settlement."""

    events = candidate.get("events")
    if isinstance(events, Mapping):
        landing_frames = events.get("landing_frames")
        if isinstance(landing_frames, Mapping):
            resolved = [
                int(value)
                for value in landing_frames.values()
                if value is not None
            ]
            if resolved:
                return max(resolved)
        for name in (
            "full_extension_confirmed_frame",
            "bottom_confirmed_frame",
        ):
            value = events.get(name)
            if value is not None:
                return int(value)
    return int(candidate["source_frame"])


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "maximum": None,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": ordered[len(ordered) // 2],
        "maximum": ordered[-1],
    }


def _cache_keypoints(points: object) -> list[Keypoint]:
    if not isinstance(points, list):
        return []
    result = []
    for point in points:
        if not isinstance(point, Mapping) or not point.get("name"):
            continue
        try:
            result.append(
                Keypoint(
                    name=str(point["name"]),
                    x=float(point["x"]),
                    y=float(point["y"]),
                    z=float(point.get("z", 0.0)),
                    confidence=float(point.get("confidence", 0.0)),
                    source_model="reviewed_pose_cache",
                    visibility=float(
                        point.get("visibility", point.get("confidence", 0.0))
                    ),
                    presence=float(
                        point.get("presence", point.get("confidence", 0.0))
                    ),
                )
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    return result


def _three_d_payload_from_cache(
    tracker: ThreeDKinematicsTracker,
    frame: Mapping[str, Any],
    *,
    camera_view: str,
    config: ShadowEvidence3DConfig,
) -> dict[str, Any]:
    image = _cache_keypoints(frame.get("image_normalized_2d"))
    world = _cache_keypoints(frame.get("mp_world_body_3d"))
    timestamp_ms = int(round(float(frame.get("source_timestamp_ms", 0.0))))
    pose = PoseResult(
        keypoints=image,
        connections=(),
        model_name="reviewed_pose_cache",
        num_keypoints=len(image),
        success=bool(image),
        inference_time_ms=0.0,
        timestamp_ms=timestamp_ms,
        extra={
            "world_keypoints": world,
            "camera_view": camera_view,
        },
    )
    payload = tracker.update(pose).as_dict()
    payload["experimental_fusion_enabled"] = True
    payload["experimental_angle_fusion_enabled"] = (
        config.angle_assist_enabled
    )
    payload["experimental_body_fusion_enabled"] = config.body_assist_enabled
    payload["experimental_temporal_thresholds"] = {
        name: getattr(config, name)
        for name in (
            "angle_conflict_min_frames",
            "angle_conflict_min_ratio",
            "angle_support_min_frames",
            "angle_support_min_ratio",
        )
    }
    payload["experiment_scope"] = "internal_2d_plus_3d_shadow_evidence"
    return payload


def _error_confusion(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for error_code, accepted_signals in sorted(ERROR_SIGNALS.items()):
        tp = fp = fn = tn = 0
        for record in records:
            expected = error_code in set(record["expected_errors"])
            predicted = bool(
                accepted_signals.intersection(
                    record["strong_runtime_signals"]
                )
            )
            if expected and predicted:
                tp += 1
            elif predicted:
                fp += 1
            elif expected:
                fn += 1
            else:
                tn += 1
        result[error_code] = {
            "true_positive_records": tp,
            "false_positive_records": fp,
            "false_negative_records": fn,
            "true_negative_records": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        }
    return result


def _terminal_event_metrics(
    records: list[dict[str, Any]],
) -> dict[str, float | int | None]:
    rows: list[tuple[int, float]] = []
    for record in records:
        fps = float(record.get("fps", 0.0) or 0.0)
        for match in record["matches"]:
            error = match.get("terminal_frame_error")
            if (
                error is None
                or match.get("candidate") is None
                or match.get("human_rep") is None
            ):
                continue
            rows.append((int(error), fps))
    if not rows:
        return {
            "matched_terminal_event_count": 0,
            "mean_absolute_error_frames": None,
            "median_absolute_error_frames": None,
            "p90_absolute_error_frames": None,
            "maximum_absolute_error_frames": None,
            "mean_signed_error_frames": None,
            "within_5_frames_rate": None,
            "within_200ms_rate": None,
        }
    absolute = sorted(abs(error) for error, _fps in rows)
    p90_index = min(len(absolute) - 1, max(0, int(len(absolute) * 0.9) - 1))
    return {
        "matched_terminal_event_count": len(rows),
        "mean_absolute_error_frames": sum(absolute) / len(absolute),
        "median_absolute_error_frames": absolute[len(absolute) // 2],
        "p90_absolute_error_frames": absolute[p90_index],
        "maximum_absolute_error_frames": absolute[-1],
        "mean_signed_error_frames": (
            sum(error for error, _fps in rows) / len(rows)
        ),
        "within_5_frames_rate": (
            sum(abs(error) <= 5 for error, _fps in rows) / len(rows)
        ),
        "within_200ms_rate": (
            sum(
                fps > 0 and abs(error) / fps <= 0.2
                for error, fps in rows
            )
            / len(rows)
        ),
    }


def _group_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    matched_rows = [
        match
        for record in records
        for match in record["matches"]
    ]
    matched_rows = [
        row
        for row in matched_rows
        if row["candidate"] is not None and row["human_rep"] is not None
    ]
    human_rep_count = sum(record["human_rep_count"] for record in records)
    predicted_candidate_count = sum(
        record["predicted_candidate_count"] for record in records
    )
    human_statuses = ("VALID", "NO_REP", "UNSURE")
    status_confusion = {
        expected: {
            predicted: sum(
                str(row["human_rep"].get("validity")) == expected
                and str(row["candidate"].get("status")) == predicted
                for row in matched_rows
            )
            for predicted in human_statuses
        }
        for expected in human_statuses
    }
    definite_count = sum(
        str(row["candidate"].get("status")) in {"VALID", "NO_REP"}
        for row in matched_rows
    )
    unsure_count = sum(
        str(row["candidate"].get("status")) == "UNSURE"
        for row in matched_rows
    )
    exact_count_record_count = sum(
        bool(record["exact_count_match"]) for record in records
    )
    return {
        "record_count": len(records),
        "human_rep_count": human_rep_count,
        "predicted_candidate_count": predicted_candidate_count,
        "matched_candidate_count": len(matched_rows),
        "candidate_recall": (
            len(matched_rows) / human_rep_count if human_rep_count else None
        ),
        "candidate_precision": (
            len(matched_rows) / predicted_candidate_count
            if predicted_candidate_count
            else None
        ),
        "count_mae": (
            sum(abs(record["candidate_count_error"]) for record in records)
            / len(records)
            if records
            else None
        ),
        "exact_count_record_count": exact_count_record_count,
        "exact_count_rate": (
            exact_count_record_count / len(records) if records else None
        ),
        "matched_rep_status_count": sum(
            int(record["matched_status_count"]) for record in records
        ),
        "matched_rep_status_accuracy": (
            sum(bool(row["status_match"]) for row in matched_rows)
            / len(matched_rows)
            if matched_rows
            else None
        ),
        "definite_decision_coverage": (
            definite_count / len(matched_rows) if matched_rows else None
        ),
        "unsure_rate": (
            unsure_count / len(matched_rows) if matched_rows else None
        ),
        "status_confusion": status_confusion,
        "aligned_rep_row_count": len(matched_rows),
        "rep_status_error_rate": (
            1.0
            - sum(bool(row["status_match"]) for row in matched_rows)
            / max(human_rep_count, predicted_candidate_count)
            if max(human_rep_count, predicted_candidate_count)
            else None
        ),
        "error_confusion_by_code": _error_confusion(records),
        "terminal_event_metrics": _terminal_event_metrics(records),
    }


def _evaluate_record(
    root: Path,
    manifest_record: dict[str, Any],
    review: dict[str, Any],
    *,
    profile: str,
    shadow_evidence_config: ShadowEvidence3DConfig | None = None,
    analyzer_config_overrides: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    action = str(review["action"])
    video = manifest_record.get("video") or {}
    pose_relative = (manifest_record.get("pose_cache") or {}).get(
        "causal_analysis_pose"
    )
    if not isinstance(pose_relative, str):
        raise ValueError(f"{review['record_id']}: causal pose cache is missing")
    pose_path = root / pose_relative
    analyzer_config: dict[str, Any] | None = None
    if profile == "baseline" and action == "lunge":
        analyzer_config = load_lunge_config()
        analyzer_config["rgb_stand_height_proxy_enabled"] = False
    elif profile == "baseline" and action == "wall_ball":
        analyzer_config = load_wall_ball_config()
        analyzer_config["heel_rise_relative_to_toe_body_ratio_min"] = 1.0
    if analyzer_config_overrides:
        if analyzer_config is None:
            if action == "lunge":
                analyzer_config = load_lunge_config()
            elif action == "wall_ball":
                analyzer_config = load_wall_ball_config()
            else:
                analyzer_config = {}
        analyzer_config.update(dict(analyzer_config_overrides))
    analyzer = create_action_analyzer(
        action,
        analyzer_config,
        camera_view=str(manifest_record.get("camera_view", "unknown")),
        live_mode=False,
    )
    three_d_tracker = (
        ThreeDKinematicsTracker(
            ThreeDKinematicsConfig(
                enabled=True,
                decision_mode="assist",
                assist_confidence_boost=shadow_evidence_config.confidence_boost,
                assist_conflict_confidence_cap=(
                    shadow_evidence_config.conflict_confidence_cap
                ),
            ),
            ThreeDQualityConfig(
                max_2d_3d_difference_deg=(
                    shadow_evidence_config.max_2d_3d_difference_deg
                ),
            ),
            shadow_evidence_config=shadow_evidence_config,
        )
        if shadow_evidence_config is not None
        else None
    )
    start = int(review["usable_start_frame"])
    end = int(review["usable_end_frame"])
    candidates: list[dict[str, Any]] = []
    signals: set[str] = set()
    strong_signals: set[str] = set()
    signal_frames: dict[str, list[int]] = {}
    previous_candidates = 0
    eligible_frames = 0
    analyzed_frames = 0
    maximum_heel_rise_streak = 0
    maximum_heel_to_toe_lift = 0.0
    final_state: dict[str, Any] = {}
    raw_phase_counts: Counter[str] = Counter()
    stable_phase_counts: Counter[str] = Counter()
    annotated_phase_by_frame: dict[int, str] = {}
    for interval in review.get("phase_error_intervals") or []:
        if not isinstance(interval, Mapping):
            continue
        for frame_index in range(
            int(interval["start_frame"]),
            int(interval["end_frame"]) + 1,
        ):
            annotated_phase_by_frame[frame_index] = str(interval["phase"])
    annotated_events_by_frame: dict[int, list[dict[str, Any]]] = {}
    for event in review.get("events") or []:
        if not isinstance(event, Mapping):
            continue
        annotated_events_by_frame.setdefault(
            int(event["frame_index"]),
            [],
        ).append(dict(event))
    annotated_event_runtime_states: list[dict[str, Any]] = []
    stable_phase_transitions: list[dict[str, Any]] = []
    previous_reported_stable_phase: str | None = None
    metric_names = (
        "visible_score",
        "min_knee_angle",
        "min_hip_angle",
        "hip_center_y",
        "hip_knee_depth",
        "left_knee_y",
        "right_knee_y",
        "left_ankle_y",
        "right_ankle_y",
        "body_height_reference",
        "body_height_norm",
        "torso_angle",
        "knee_center_height_to_floor_body_ratio",
        "left_heel_height_to_floor_body_ratio",
        "right_heel_height_to_floor_body_ratio",
        "left_foot_index_height_to_floor_body_ratio",
        "right_foot_index_height_to_floor_body_ratio",
    )
    annotated_metrics: dict[str, dict[str, list[float]]] = {}
    three_d_available_frames = 0
    body_relative_reliable_frames = 0
    three_d_conflict_frames = 0
    with gzip.open(pose_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            frame = json.loads(line)
            frame_index = int(frame.get("frame_index", -1))
            if frame_index < start or frame_index > end:
                continue
            formal = bool(frame.get("formal_pose_eligible")) and bool(
                frame.get("may_drive_rules_or_training")
            )
            features = None
            if formal:
                eligible_frames += 1
                features = extract_basic_pose_features(
                    frame.get("image_normalized_2d"),
                    image_width=int(video.get("width", 1) or 1),
                    image_height=int(video.get("height", 1) or 1),
                )
                if three_d_tracker is not None:
                    three_d_payload = _three_d_payload_from_cache(
                        three_d_tracker,
                        frame,
                        camera_view=str(
                            manifest_record.get("camera_view", "unknown")
                        ),
                        config=shadow_evidence_config,
                    )
                    features["three_d_kinematics"] = three_d_payload
                    three_d_available_frames += int(
                        bool(three_d_payload.get("three_d_available"))
                    )
                    body = three_d_payload.get("body_relative")
                    body_relative_reliable_frames += int(
                        isinstance(body, Mapping)
                        and bool(body.get("reliable"))
                    )
                    three_d_conflict_frames += int(
                        str(three_d_payload.get("assist_status", "")).lower()
                        == "conflict"
                    )
            elif three_d_tracker is not None:
                three_d_tracker.reset()
            final_state = analyzer.update(
                features,
                int(round(float(frame.get("source_timestamp_ms", 0.0)))),
            )
            analyzed_frames += 1
            maximum_heel_rise_streak = max(
                maximum_heel_rise_streak,
                int(getattr(analyzer, "_heel_rise_frames", 0)),
            )
            enriched_for_feet = getattr(analyzer, "_current_features", {})
            if isinstance(enriched_for_feet, Mapping):
                for side in ("left", "right"):
                    try:
                        heel = float(
                            enriched_for_feet[
                                f"{side}_heel_height_to_floor_body_ratio"
                            ]
                        )
                        toe = float(
                            enriched_for_feet[
                                f"{side}_foot_index_height_to_floor_body_ratio"
                            ]
                        )
                    except (KeyError, TypeError, ValueError, OverflowError):
                        continue
                    maximum_heel_to_toe_lift = max(
                        maximum_heel_to_toe_lift,
                        heel - toe,
                    )
            debug = final_state.get("debug") or {}
            if isinstance(debug, Mapping):
                raw_phase = str(debug.get("raw_phase", "unknown"))
                stable_phase = str(debug.get("stable_phase", "unknown"))
                raw_phase_counts[raw_phase] += 1
                stable_phase_counts[stable_phase] += 1
                if stable_phase != previous_reported_stable_phase:
                    stable_phase_transitions.append(
                        {
                            "frame_index": frame_index,
                            "raw_phase": raw_phase,
                            "stable_phase": stable_phase,
                        }
                    )
                    previous_reported_stable_phase = stable_phase
                for event in annotated_events_by_frame.get(frame_index, ()):
                    annotated_event_runtime_states.append(
                        {
                            "rep_id": event.get("rep_id"),
                            "event_type": event.get("event_type"),
                            "frame_index": frame_index,
                            "runtime_raw_phase": raw_phase,
                            "runtime_stable_phase": stable_phase,
                            "runtime_visible_score": debug.get(
                                "visible_score"
                            ),
                            "runtime_validation_state": (
                                debug.get("wall_ball_validation_state")
                                or debug.get("lunge_validation_state")
                                or debug.get("burpee_validation_state")
                            ),
                        }
                    )
            annotated_phase = annotated_phase_by_frame.get(frame_index)
            if annotated_phase and features is not None:
                phase_metrics = annotated_metrics.setdefault(
                    annotated_phase,
                    {name: [] for name in metric_names},
                )
                enriched = getattr(analyzer, "_current_features", features)
                for name in metric_names:
                    value = (
                        enriched.get(name)
                        if isinstance(enriched, Mapping)
                        else None
                    )
                    try:
                        resolved = float(value)
                    except (TypeError, ValueError, OverflowError):
                        continue
                    if resolved == resolved:
                        phase_metrics[name].append(resolved)
            current_signals = _feedback_codes(final_state)
            strong_signals.update(
                current_signals.intersection({"HEEL_RISE"})
            )
            decision = analyzer.last_rep_decision
            if analyzer.candidate_count > previous_candidates and decision is not None:
                decision_payload = decision.as_dict()
                candidate_payload = (
                    {}
                    if analyzer.last_rep_candidate is None
                    else analyzer.last_rep_candidate.as_dict()
                )
                current_signals.update(
                    str(code) for code in decision_payload.get("reason_codes") or []
                )
                current_signals.update(
                    str(rule["reason_code"])
                    for rule in decision_payload.get("rules") or []
                    if isinstance(rule, Mapping) and rule.get("reason_code")
                )
                if decision.status == "NO_REP":
                    strong_signals.update(
                        str(rule["reason_code"])
                        for rule in decision_payload.get("rules") or []
                        if isinstance(rule, Mapping)
                        and rule.get("status") == "FAIL"
                        and rule.get("reason_code")
                    )
                candidate = {
                    "source_frame": frame_index,
                    "candidate_start_frame": candidate_payload.get(
                        "start_frame"
                    ),
                    "candidate_end_frame": candidate_payload.get("end_frame"),
                    "events": candidate_payload.get("events", {}),
                    "status": decision.status,
                    "confidence": decision.confidence,
                    "reason_codes": list(decision.reason_codes),
                    "rules": [
                        rule.as_dict() for rule in decision.rules
                    ],
                }
                candidate["alignment_frame"] = _candidate_alignment_frame(
                    candidate
                )
                candidates.append(candidate)
            previous_candidates = analyzer.candidate_count
            for code in current_signals:
                signals.add(code)
                signal_frames.setdefault(code, []).append(frame_index)

    finalize_pending = getattr(analyzer, "finalize_pending_candidate", None)
    if profile == "optimized" and callable(finalize_pending):
        decision = finalize_pending()
        if decision is not None:
            decision_payload = decision.as_dict()
            candidate_payload = (
                {}
                if analyzer.last_rep_candidate is None
                else analyzer.last_rep_candidate.as_dict()
            )
            final_signals = {
                str(code)
                for code in decision_payload.get("reason_codes") or []
            }
            final_signals.update(
                str(rule["reason_code"])
                for rule in decision_payload.get("rules") or []
                if isinstance(rule, Mapping) and rule.get("reason_code")
            )
            candidate = {
                "source_frame": end,
                "candidate_start_frame": candidate_payload.get("start_frame"),
                "candidate_end_frame": candidate_payload.get("end_frame"),
                "events": candidate_payload.get("events", {}),
                "status": decision.status,
                "confidence": decision.confidence,
                "reason_codes": list(decision.reason_codes),
                "rules": [rule.as_dict() for rule in decision.rules],
                "validation_boundary": "stream_end",
            }
            candidate["alignment_frame"] = _candidate_alignment_frame(
                candidate
            )
            candidates.append(candidate)
            for code in final_signals:
                signals.add(code)
                signal_frames.setdefault(code, []).append(end)
            if decision.status == "NO_REP":
                strong_signals.update(
                    str(rule["reason_code"])
                    for rule in decision_payload.get("rules") or []
                    if isinstance(rule, Mapping)
                    and rule.get("status") == "FAIL"
                    and rule.get("reason_code")
                )

    human_reps = [
        dict(item)
        for item in review.get("reps") or []
        if isinstance(item, dict)
    ]
    matches = _match_statuses(human_reps, candidates)
    expected_errors = _human_errors(review)
    predicted_error_classes = sorted(
        error_code
        for error_code, accepted in ERROR_SIGNALS.items()
        if accepted.intersection(strong_signals)
    )
    error_results = []
    for error in expected_errors:
        accepted = ERROR_SIGNALS.get(error, frozenset())
        hits = sorted(accepted.intersection(strong_signals))
        error_results.append(
            {
                "error_code": error,
                "supported_by_runtime": bool(accepted),
                "detected": bool(hits),
                "matching_signals": hits,
            }
        )
    human_status_counts = Counter(str(item.get("validity")) for item in human_reps)
    predicted_status_counts = Counter(
        str(item.get("status")) for item in candidates
    )
    return {
        "record_id": str(review["record_id"]),
        "action": action,
        "camera_view": manifest_record.get("camera_view"),
        "subject_group": str(
            review.get("subject_group", "subject_group_unassigned")
        ),
        "subject_group_is_temporary": bool(
            review.get("subject_group_is_temporary", True)
        ),
        "dataset_role": str(review.get("dataset_role", "unassigned")),
        "fps": float(video.get("fps", 0.0) or 0.0),
        "human_overall_result": review.get("overall_result"),
        "human_rep_count": len(human_reps),
        "predicted_candidate_count": len(candidates),
        "candidate_count_error": len(candidates) - len(human_reps),
        "human_status_counts": dict(sorted(human_status_counts.items())),
        "predicted_status_counts": dict(sorted(predicted_status_counts.items())),
        "matched_status_count": sum(
            bool(item["status_match"]) for item in matches
        ),
        "exact_count_match": len(candidates) == len(human_reps),
        "exact_count_and_status_match": (
            len(candidates) == len(human_reps)
            and all(bool(item["status_match"]) for item in matches)
        ),
        "expected_errors": expected_errors,
        "predicted_error_classes": predicted_error_classes,
        "unexpected_predicted_error_classes": sorted(
            set(predicted_error_classes).difference(expected_errors)
        ),
        "error_results": error_results,
        "runtime_signals": sorted(signals),
        "strong_runtime_signals": sorted(strong_signals),
        "runtime_signal_frames": {
            code: frames for code, frames in sorted(signal_frames.items())
        },
        "raw_phase_counts": dict(sorted(raw_phase_counts.items())),
        "stable_phase_counts": dict(sorted(stable_phase_counts.items())),
        "stable_phase_transitions": stable_phase_transitions,
        "annotated_event_runtime_states": (
            annotated_event_runtime_states
        ),
        "annotated_phase_feature_summary": {
            phase: {
                name: _summary(values)
                for name, values in metrics.items()
            }
            for phase, metrics in sorted(annotated_metrics.items())
        },
        "matches": matches,
        "analyzed_frame_count": analyzed_frames,
        "formal_pose_eligible_frame_count": eligible_frames,
        "formal_pose_eligible_ratio": (
            eligible_frames / analyzed_frames if analyzed_frames else 0.0
        ),
        "three_d_shadow": {
            "enabled": shadow_evidence_config is not None,
            "world_available_frame_count": three_d_available_frames,
            "body_relative_reliable_frame_count": (
                body_relative_reliable_frames
            ),
            "angle_conflict_frame_count": three_d_conflict_frames,
            "world_available_ratio_of_eligible": (
                three_d_available_frames / eligible_frames
                if eligible_frames
                else 0.0
            ),
            "body_relative_reliable_ratio_of_eligible": (
                body_relative_reliable_frames / eligible_frames
                if eligible_frames
                else 0.0
            ),
            "config": (
                None
                if shadow_evidence_config is None
                else shadow_evidence_config.as_dict()
            ),
            "contact_inference_allowed": False,
            "validity_promotion_allowed": False,
        },
        "maximum_heel_rise_streak": maximum_heel_rise_streak,
        "maximum_heel_to_toe_lift": maximum_heel_to_toe_lift,
        "pose_cache": pose_relative,
        "pose_cache_sha256": _sha256(pose_path),
        "final_state_counts": {
            "candidate_count": int(final_state.get("candidate_count", 0) or 0),
            "valid_count": int(final_state.get("rep_count", 0) or 0),
            "no_rep_count": int(final_state.get("no_rep_count", 0) or 0),
            "unsure_count": int(final_state.get("unsure_count", 0) or 0),
        },
        "final_floor_reference": final_state.get("floor_reference"),
    }


def build_guidance_report(
    dataset_root: str | Path,
    *,
    profile: str = "optimized",
) -> Path:
    if profile not in {"baseline", "optimized"}:
        raise ValueError(f"unknown evaluation profile: {profile}")
    root = Path(dataset_root)
    manifest_payload = _load(root / "manifests" / "phone_records.json")
    fine_path = root / "reviews" / "human_rgb_fine_annotations_v1.json"
    fine_payload = _load(fine_path)
    if fine_payload.get("oni_records_included") is not False:
        raise ValueError("reviewed RGB evaluation must not include ONI records")
    manifest = {
        str(item.get("record_id")): item
        for item in manifest_payload.get("records") or []
        if isinstance(item, dict)
    }
    records = []
    for review in fine_payload.get("records") or []:
        if not isinstance(review, dict) or review.get("action") not in TARGET_ACTIONS:
            continue
        if not bool(review.get("internal_rgb_rule_calibration_eligible")):
            continue
        record_id = str(review.get("record_id"))
        manifest_record = manifest.get(record_id)
        if not isinstance(manifest_record, dict):
            raise ValueError(f"{record_id}: missing from phone manifest")
        records.append(
            _evaluate_record(
                root,
                manifest_record,
                review,
                profile=profile,
            )
        )

    expected_error_rows = [
        row
        for record in records
        for row in record["error_results"]
    ]
    supported_error_rows = [
        row for row in expected_error_rows if row["supported_by_runtime"]
    ]
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    role_groups: dict[str, list[dict[str, Any]]] = {}
    subject_groups: dict[str, list[dict[str, Any]]] = {}
    action_view_subject_groups: dict[
        str, dict[str, dict[str, list[dict[str, Any]]]]
    ] = {}
    for record in records:
        action_groups = grouped.setdefault(str(record["action"]), {})
        action_groups.setdefault(str(record["camera_view"]), []).append(record)
        role_groups.setdefault(str(record["dataset_role"]), []).append(record)
        subject_groups.setdefault(str(record["subject_group"]), []).append(
            record
        )
        view_groups = action_view_subject_groups.setdefault(
            str(record["action"]), {}
        )
        subject_view_groups = view_groups.setdefault(
            str(record["camera_view"]), {}
        )
        subject_view_groups.setdefault(
            str(record["subject_group"]), []
        ).append(record)
    aggregate_metrics = _group_metrics(records)
    data_roles_path = root / "manifests" / "data_roles_v1.json"
    data_roles = _load(data_roles_path)
    assignments = [
        item
        for item in data_roles.get("assignments", [])
        if isinstance(item, dict)
    ]
    assigned_roles_by_subject: dict[str, set[str]] = {}
    for assignment in assignments:
        subject = str(assignment.get("subject_group", ""))
        role = str(assignment.get("role", ""))
        if subject and role in {"development", "validation", "test"}:
            assigned_roles_by_subject.setdefault(subject, set()).add(role)
    subject_role_conflicts = {
        subject: sorted(roles)
        for subject, roles in assigned_roles_by_subject.items()
        if len(roles) > 1
    }
    training_ids = {
        str(item.get("record_id", ""))
        for item in assignments
        if bool(item.get("training_eligible"))
    }
    evaluation_ids = {
        str(item.get("record_id", ""))
        for item in assignments
        if bool(item.get("evaluation_eligible"))
    }
    training_evaluation_overlap = sorted(training_ids & evaluation_ids)
    core_role_counts = dict(
        sorted(Counter(str(item["dataset_role"]) for item in records).items())
    )
    leakage_checks = {
        "data_roles_manifest": str(data_roles_path.relative_to(root)),
        "data_roles_manifest_sha256": _sha256(data_roles_path),
        "training_evaluation_record_overlap": training_evaluation_overlap,
        "subject_role_conflicts": subject_role_conflicts,
        "no_training_evaluation_record_overlap": not training_evaluation_overlap,
        "no_subject_role_conflicts": not subject_role_conflicts,
        "core_evaluation_role_counts": core_role_counts,
        "frozen_test_record_count": core_role_counts.get("test", 0),
        "independent_core_validation_available": bool(
            core_role_counts.get("validation", 0)
        ),
        "independent_core_test_available": bool(
            core_role_counts.get("test", 0)
        ),
        "status": (
            "no_detected_role_leakage_but_no_independent_core_holdout"
            if not training_evaluation_overlap
            and not subject_role_conflicts
            and not core_role_counts.get("test", 0)
            else "failed"
            if training_evaluation_overlap or subject_role_conflicts
            else "passed"
        ),
    }
    payload = {
        "schema_version": 1,
        "artifact_type": "reviewed_phone_rgb_guidance_evaluation_v1",
        "evaluation_profile": profile,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_fine_annotations": str(fine_path.relative_to(root)),
        "source_fine_annotations_sha256": _sha256(fine_path),
        "source_type": "phone_rgb",
        "oni_used": False,
        "target_actions": sorted(TARGET_ACTIONS),
        "record_count": len(records),
        "human_rep_count": sum(item["human_rep_count"] for item in records),
        "predicted_candidate_count": sum(
            item["predicted_candidate_count"] for item in records
        ),
        "exact_count_record_count": sum(
            bool(item["exact_count_match"]) for item in records
        ),
        "exact_count_and_status_record_count": sum(
            bool(item["exact_count_and_status_match"]) for item in records
        ),
        "matched_rep_status_count": sum(
            int(item["matched_status_count"]) for item in records
        ),
        "expected_error_record_count": len(expected_error_rows),
        "supported_expected_error_record_count": len(supported_error_rows),
        "detected_supported_error_record_count": sum(
            bool(item["detected"]) for item in supported_error_rows
        ),
        "rep_status_error_rate": aggregate_metrics[
            "rep_status_error_rate"
        ],
        "candidate_recall": aggregate_metrics["candidate_recall"],
        "candidate_precision": aggregate_metrics["candidate_precision"],
        "count_mae": aggregate_metrics["count_mae"],
        "exact_count_rate": aggregate_metrics["exact_count_rate"],
        "matched_rep_status_accuracy": aggregate_metrics[
            "matched_rep_status_accuracy"
        ],
        "definite_decision_coverage": aggregate_metrics[
            "definite_decision_coverage"
        ],
        "unsure_rate": aggregate_metrics["unsure_rate"],
        "status_confusion": aggregate_metrics["status_confusion"],
        "error_confusion_by_code": aggregate_metrics[
            "error_confusion_by_code"
        ],
        "terminal_event_metrics": aggregate_metrics[
            "terminal_event_metrics"
        ],
        "metrics_by_action_and_view": {
            action: {
                view: _group_metrics(group_records)
                for view, group_records in sorted(view_groups.items())
            }
            for action, view_groups in sorted(grouped.items())
        },
        "metrics_by_dataset_role": {
            role: _group_metrics(group_records)
            for role, group_records in sorted(role_groups.items())
        },
        "metrics_by_subject_group": {
            subject: _group_metrics(group_records)
            for subject, group_records in sorted(subject_groups.items())
        },
        "metrics_by_action_view_subject": {
            action: {
                view: {
                    subject: _group_metrics(group_records)
                    for subject, group_records in sorted(subjects.items())
                }
                for view, subjects in sorted(views.items())
            }
            for action, views in sorted(action_view_subject_groups.items())
        },
        "leakage_checks": leakage_checks,
        "fine_rgb_human_review": fine_payload.get(
            "fine_rgb_human_review"
        ),
        "supervised_model_training_eligible_record_count": int(
            fine_payload.get(
                "supervised_model_training_eligible_record_count",
                0,
            )
            or 0
        ),
        "limitations": [
            "All 15 fine RGB records are user-confirmed as manually reviewed and may be used for internal calibration and supervised experiments.",
            "The same small reviewed set is used for calibration and regression, so this is not an independent test result.",
            "This report does not claim cross-subject or production generalization.",
            "Temporary subject groups are used only for leakage-safe internal organization and are not real identity claims.",
            "The current core actions have no frozen independent test subject; split metrics must not be presented as hold-out performance.",
            "ONI Depth/IR and unfinished ONI subject review are not read or used.",
        ],
        "records": records,
    }
    return _write(
        root
        / "reports"
        / f"reviewed_phone_rgb_guidance_evaluation_{profile}_v1.json",
        payload,
    )


def build_comparison_report(dataset_root: str | Path) -> Path:
    root = Path(dataset_root)
    baseline_path = build_guidance_report(root, profile="baseline")
    optimized_path = build_guidance_report(root, profile="optimized")
    baseline = _load(baseline_path)
    optimized = _load(optimized_path)
    metrics = (
        "human_rep_count",
        "predicted_candidate_count",
        "exact_count_record_count",
        "exact_count_and_status_record_count",
        "matched_rep_status_count",
        "detected_supported_error_record_count",
        "supported_expected_error_record_count",
        "candidate_recall",
        "candidate_precision",
        "count_mae",
        "exact_count_rate",
        "matched_rep_status_accuracy",
        "definite_decision_coverage",
        "unsure_rate",
    )
    comparison = {
        key: {
            "baseline": baseline[key],
            "optimized": optimized[key],
            "delta": optimized[key] - baseline[key],
        }
        for key in metrics
    }
    payload = {
        "schema_version": 1,
        "artifact_type": "reviewed_phone_rgb_guidance_optimization_comparison_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_type": "phone_rgb",
        "oni_used": False,
        "baseline_report": str(baseline_path.relative_to(root)),
        "baseline_report_sha256": _sha256(baseline_path),
        "optimized_report": str(optimized_path.relative_to(root)),
        "optimized_report_sha256": _sha256(optimized_path),
        "metrics": comparison,
        "record_deltas": [
            {
                "record_id": after["record_id"],
                "action": after["action"],
                "human_rep_count": after["human_rep_count"],
                "baseline_candidate_count": before[
                    "predicted_candidate_count"
                ],
                "optimized_candidate_count": after[
                    "predicted_candidate_count"
                ],
                "candidate_count_delta": (
                    after["predicted_candidate_count"]
                    - before["predicted_candidate_count"]
                ),
                "baseline_detected_errors": [
                    row["error_code"]
                    for row in before["error_results"]
                    if row["detected"]
                ],
                "optimized_detected_errors": [
                    row["error_code"]
                    for row in after["error_results"]
                    if row["detected"]
                ],
            }
            for before, after in zip(
                baseline["records"],
                optimized["records"],
            )
        ],
        "conclusion_scope": (
            "Internal single-reviewer phone-RGB regression only; "
            "no cross-subject or production-generalization claim."
        ),
    }
    return _write(
        root
        / "reports"
        / "reviewed_phone_rgb_guidance_optimization_comparison_v1.json",
        payload,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate formal HYROX guidance against reviewed phone RGB."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/hyrox"),
    )
    parser.add_argument(
        "--profile",
        choices=("baseline", "optimized", "both"),
        default="both",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.dataset_root
    if not root.is_absolute():
        root = args.project_root.resolve() / root
    if args.profile == "both":
        output = build_comparison_report(root)
        payload = _load(output)
        print(
            json.dumps(
                {
                    "report": str(output),
                    "metrics": payload["metrics"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    output = build_guidance_report(root, profile=args.profile)
    payload = _load(output)
    print(
        json.dumps(
            {
                "report": str(output),
                "record_count": payload["record_count"],
                "human_rep_count": payload["human_rep_count"],
                "predicted_candidate_count": payload[
                    "predicted_candidate_count"
                ],
                "exact_count_record_count": payload[
                    "exact_count_record_count"
                ],
                "exact_count_and_status_record_count": payload[
                    "exact_count_and_status_record_count"
                ],
                "detected_supported_error_record_count": payload[
                    "detected_supported_error_record_count"
                ],
                "supported_expected_error_record_count": payload[
                    "supported_expected_error_record_count"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
