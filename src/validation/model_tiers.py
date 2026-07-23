"""Full/Lite MediaPipe accuracy and performance comparison on golden videos."""

from __future__ import annotations

from dataclasses import asdict
from math import hypot, isfinite
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import cv2

from hyrox.features import extract_basic_pose_features
from hyrox.registry import create_action_analyzer
from src.backends.base import PoseResult
from src.backends.mediapipe_backend import MediaPipeBackend
from src.biomechanics.kinematics_3d import ThreeDKinematicsTracker
from src.output_schema import versioned_payload
from src.paths import resolve_asset
from src.product_pose import load_product_pose_config
from src.utils.smoothing import KeypointSmoother
from src.validation.golden_videos import GoldenCase, GoldenObservation, compare_observation


OUTCOME_FIELDS = (
    "candidate_count",
    "pose_valid_rep_count",
    "no_rep_count",
    "unsure_count",
    "cycle_count",
    "rep_count",
)


def _percentile(values: Sequence[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) * ratio) + 0.999999) - 1))
    return ordered[index]


def _point_map(result: PoseResult) -> dict[str, object]:
    return {point.name: point for point in result.keypoints}


def _image_deltas(full: PoseResult, lite: PoseResult) -> list[float]:
    full_points = _point_map(full)
    lite_points = _point_map(lite)
    deltas: list[float] = []
    for name in full_points.keys() & lite_points.keys():
        full_point = full_points[name]
        lite_point = lite_points[name]
        deltas.append(hypot(float(full_point.x) - float(lite_point.x), float(full_point.y) - float(lite_point.y)))
    return deltas


def _angle_deltas(full: Mapping[str, object], lite: Mapping[str, object]) -> list[float]:
    deltas: list[float] = []
    for name in full.keys() & lite.keys():
        if not (name.endswith("_angle") or name.endswith("_angle_deg")):
            continue
        try:
            full_value = float(full[name])
            lite_value = float(lite[name])
        except (TypeError, ValueError, OverflowError):
            continue
        if isfinite(full_value) and isfinite(lite_value):
            deltas.append(abs(full_value - lite_value))
    return deltas


def _debug_mapping(state: Mapping[str, object], name: str) -> Mapping[str, object]:
    debug = state.get("debug")
    if not isinstance(debug, Mapping):
        return {}
    value = debug.get(name)
    return value if isinstance(value, Mapping) else {}


def _contact_statuses(state: Mapping[str, object]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for name, value in _debug_mapping(state, "contacts").items():
        if isinstance(value, Mapping) and value.get("status") is not None:
            statuses[str(name)] = str(value["status"])
    return statuses


def _foot_statuses(state: Mapping[str, object]) -> dict[str, str]:
    foot = _debug_mapping(state, "foot_events")
    statuses: dict[str, str] = {}
    for side in ("left", "right"):
        value = foot.get(side)
        if isinstance(value, Mapping) and value.get("state") is not None:
            statuses[f"{side}.state"] = str(value["state"])
    for group, fields in (("sync", ("takeoff_status", "landing_status")), ("stagger", ("status",))):
        value = foot.get(group)
        if not isinstance(value, Mapping):
            continue
        for field in fields:
            if value.get(field) is not None:
                statuses[f"{group}.{field}"] = str(value[field])
    return statuses


def _step_event_count(state: Mapping[str, object]) -> int:
    value = _debug_mapping(state, "foot_events").get("step_event_count", 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def _status_matches(full: Mapping[str, str], lite: Mapping[str, str]) -> tuple[int, int]:
    names = full.keys() & lite.keys()
    return sum(full[name] == lite[name] for name in names), len(names)


def _observation(case: GoldenCase, state: Mapping[str, object], total: int, detected: int) -> GoldenObservation:
    integer = lambda name: int(state.get(name, 0) or 0)
    return GoldenObservation(
        case_id=case.case_id,
        video=case.video,
        action=case.action,
        total_frames=total,
        pose_detected_frames=detected,
        pose_detected_rate=detected / max(1, total),
        candidate_count=integer("candidate_count"),
        pose_valid_rep_count=integer("pose_valid_rep_count"),
        no_rep_count=integer("no_rep_count"),
        unsure_count=integer("unsure_count"),
        cycle_count=integer("cycle_count"),
        rep_count=integer("rep_count"),
        final_phase=str(state.get("phase", "unknown")),
    )


def compare_model_tiers_case(
    case: GoldenCase,
    *,
    full_backend: MediaPipeBackend,
    lite_backend: MediaPipeBackend,
) -> dict[str, Any]:
    video_path = resolve_asset(case.video)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open golden video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0
    config = load_product_pose_config()
    full_tracker = ThreeDKinematicsTracker(config.three_d_kinematics, config.three_d_quality)
    lite_tracker = ThreeDKinematicsTracker(config.three_d_kinematics, config.three_d_quality)
    full_smoother = KeypointSmoother(mode="one-euro", max_missing_frames=5)
    lite_smoother = KeypointSmoother(mode="one-euro", max_missing_frames=5)
    full_analyzer = create_action_analyzer(case.action, camera_view=case.camera_view, live_mode=False)
    lite_analyzer = create_action_analyzer(case.action, camera_view=case.camera_view, live_mode=False)
    full_offset = max(0, int(getattr(full_backend, "_last_timestamp_ms", -1)) + 1)
    lite_offset = max(0, int(getattr(lite_backend, "_last_timestamp_ms", -1)) + 1)
    total = full_detected = lite_detected = matched_frames = 0
    image_deltas: list[float] = []
    angle_deltas: list[float] = []
    full_inference: list[float] = []
    lite_inference: list[float] = []
    contact_matches = contact_compared = 0
    foot_matches = foot_compared = 0
    full_three_d = lite_three_d = 0
    three_d_status_matches = 0
    full_state: Mapping[str, object] = {}
    lite_state: Mapping[str, object] = {}
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            total += 1
            timestamp_ms = max(0, int(round(total * 1000.0 / fps)))
            capture_ns = timestamp_ms * 1_000_000
            full_raw = full_backend.detect(frame, timestamp_ms=full_offset + timestamp_ms)
            lite_raw = lite_backend.detect(frame, timestamp_ms=lite_offset + timestamp_ms)
            full_inference.append(float(full_raw.inference_time_ms))
            lite_inference.append(float(lite_raw.inference_time_ms))
            full_result = full_smoother.smooth_result(full_raw, capture_timestamp_ns=capture_ns)
            lite_result = lite_smoother.smooth_result(lite_raw, capture_timestamp_ns=capture_ns)
            full_result, full_3d = full_tracker.attach(full_result, capture_timestamp_ns=capture_ns, pose_age_ms=0)
            lite_result, lite_3d = lite_tracker.attach(lite_result, capture_timestamp_ns=capture_ns, pose_age_ms=0)
            full_has_pose = bool(full_result.success and full_result.keypoints)
            lite_has_pose = bool(lite_result.success and lite_result.keypoints)
            full_detected += int(full_has_pose)
            lite_detected += int(lite_has_pose)
            full_three_d += int(full_3d.three_d_available)
            lite_three_d += int(lite_3d.three_d_available)
            three_d_status_matches += int(full_3d.assist_status == lite_3d.assist_status)
            height, width = frame.shape[:2]
            full_features = extract_basic_pose_features(full_result.keypoints, width, height) if full_has_pose else None
            lite_features = extract_basic_pose_features(lite_result.keypoints, width, height) if lite_has_pose else None
            if full_features is not None:
                full_features["three_d_kinematics"] = full_3d.as_dict()
            if lite_features is not None:
                lite_features["three_d_kinematics"] = lite_3d.as_dict()
            full_state = full_analyzer.attach_view_context(full_analyzer.update(full_features, timestamp_ms=timestamp_ms))
            lite_state = lite_analyzer.attach_view_context(lite_analyzer.update(lite_features, timestamp_ms=timestamp_ms))
            if full_has_pose and lite_has_pose:
                matched_frames += 1
                image_deltas.extend(_image_deltas(full_raw, lite_raw))
                angle_deltas.extend(_angle_deltas(full_features or {}, lite_features or {}))
            matches, compared = _status_matches(_contact_statuses(full_state), _contact_statuses(lite_state))
            contact_matches += matches
            contact_compared += compared
            matches, compared = _status_matches(_foot_statuses(full_state), _foot_statuses(lite_state))
            foot_matches += matches
            foot_compared += compared
    finally:
        capture.release()
    full_observation = _observation(case, full_state, total, full_detected)
    lite_observation = _observation(case, lite_state, total, lite_detected)
    return {
        "id": case.case_id,
        "action": case.action,
        "video": case.video,
        "total_frames": total,
        "matched_pose_frames": matched_frames,
        "full": {
            "observation": asdict(full_observation),
            "golden_failures": compare_observation(case, full_observation),
            "inference_p50_ms": median(full_inference) if full_inference else 0.0,
            "inference_p95_ms": _percentile(full_inference, 0.95),
            "three_d_available_rate": full_three_d / max(1, total),
            "step_event_count": _step_event_count(full_state),
        },
        "lite": {
            "observation": asdict(lite_observation),
            "golden_failures": compare_observation(case, lite_observation),
            "inference_p50_ms": median(lite_inference) if lite_inference else 0.0,
            "inference_p95_ms": _percentile(lite_inference, 0.95),
            "three_d_available_rate": lite_three_d / max(1, total),
            "step_event_count": _step_event_count(lite_state),
        },
        "differences": {
            "image_landmark_mean": sum(image_deltas) / max(1, len(image_deltas)),
            "image_landmark_p95": _percentile(image_deltas, 0.95),
            "joint_angle_mean_deg": sum(angle_deltas) / max(1, len(angle_deltas)),
            "joint_angle_p95_deg": _percentile(angle_deltas, 0.95),
            "contact_status_match_rate": contact_matches / max(1, contact_compared),
            "foot_status_match_rate": foot_matches / max(1, foot_compared),
            "three_d_assist_status_match_rate": three_d_status_matches / max(1, total),
        },
    }


def model_tier_gate(record: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    full = record["full"]
    lite = record["lite"]
    differences = record["differences"]
    if lite.get("golden_failures"):
        failures.append("Lite 未通过原黄金区间")
    full_observation = full["observation"]
    lite_observation = lite["observation"]
    for name in ("candidate_count", "pose_valid_rep_count", "no_rep_count", "cycle_count", "rep_count"):
        if int(full_observation[name]) != int(lite_observation[name]):
            failures.append(f"{name} 与 Full 不一致")
    if abs(int(full_observation["unsure_count"]) - int(lite_observation["unsure_count"])) > 1:
        failures.append("unsure_count 与 Full 相差超过 1")
    if float(lite_observation["pose_detected_rate"]) + 0.05 < float(full_observation["pose_detected_rate"]):
        failures.append("姿态检出率比 Full 低超过 5%")
    if float(differences["image_landmark_mean"]) > 0.08:
        failures.append("image landmarks 平均差异超过 0.08")
    if float(differences["joint_angle_mean_deg"]) > 12.0:
        failures.append("关节角平均差异超过 12°")
    if abs(int(full["step_event_count"]) - int(lite["step_event_count"])) > 2:
        failures.append("脚部事件数相差超过 2")
    if float(lite["three_d_available_rate"]) + 0.10 < float(full["three_d_available_rate"]):
        failures.append("3D Assist 可用率比 Full 低超过 10%")
    return failures


def build_model_tier_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cases = []
    for record in records:
        failures = model_tier_gate(record)
        cases.append({**record, "status": "passed" if not failures else "failed", "gate_failures": failures})
    approved = bool(cases) and all(case["status"] == "passed" for case in cases)
    return versioned_payload(
        "pose_model_tier_comparison",
        {
            "status": "passed" if approved else "failed",
            "lite_auto_approved": approved,
            "case_count": len(cases),
            "passed_count": sum(case["status"] == "passed" for case in cases),
            "cases": cases,
        },
    )


__all__ = [
    "build_model_tier_report",
    "compare_model_tiers_case",
    "model_tier_gate",
]
