"""Round-eight causal/display pose derivation and joint stability metrics."""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.backends.base import Keypoint, PoseResult
from src.product_pose import load_product_pose_config
from src.utils.keypoint_schema import MEDIAPIPE_33_NAMES, MEDIAPIPE_CONNECTIONS
from src.utils.smoothing import FAST_JOINT_NAMES, STABLE_JOINT_NAMES, KeypointSmoother


JOINT_GROUPS: Mapping[str, tuple[str, ...]] = {
    "wrist": ("left_wrist", "right_wrist"),
    "ankle": ("left_ankle", "right_ankle", "left_heel", "right_heel"),
    "knee": ("left_knee", "right_knee"),
    "hip": ("left_hip", "right_hip"),
    "shoulder": ("left_shoulder", "right_shoulder"),
    "torso": ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
}
FACE_NAMES = frozenset(MEDIAPIPE_33_NAMES[:11])
FILTER_VERSION = "round8_joint_adaptive_causal_v1"
DISPLAY_FILTER_VERSION = "round8_constrained_constant_velocity_v1"
OFFLINE_ASSIST_VERSION = "round8_centered_five_frame_v1"


def keypoint_from_payload(payload: Mapping[str, Any], *, source_model: str) -> Keypoint:
    def number(name: str, default: float = float("nan")) -> float:
        value = payload.get(name)
        if value is None:
            return default
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return numeric

    return Keypoint(
        name=str(payload["name"]),
        x=number("x"),
        y=number("y"),
        z=number("z", 0.0),
        confidence=number("confidence", 0.0),
        source_model=source_model,
        visibility=number("visibility", 0.0),
        presence=number("presence", 0.0),
    )


def pose_result_from_raw(record: Mapping[str, Any]) -> PoseResult:
    unified = record.get("unified_33") or {}
    image = [
        keypoint_from_payload(point, source_model=str(record["backend"]))
        for point in unified.get("image_normalized_2d") or []
    ]
    world = [
        keypoint_from_payload(
            point, source_model=f"{record['backend']}-world"
        )
        for point in unified.get("mp_world_body_3d") or []
    ]
    bbox = record.get("bbox_normalized_xyxy")
    return PoseResult(
        keypoints=image,
        connections=MEDIAPIPE_CONNECTIONS,
        model_name=str(record["backend"]),
        num_keypoints=len(image),
        success=bool(record.get("pose_success")),
        inference_time_ms=float(record.get("inference_time_ms") or 0.0),
        bbox=tuple(float(value) for value in bbox) if bbox else None,
        timestamp_ms=int(round(float(record["source_timestamp_ms"]))),
        extra={"world_keypoints": world},
    )


def filter_factories() -> dict[str, callable]:
    config = load_product_pose_config()
    return {
        "raw": lambda: KeypointSmoother(mode="none"),
        "ema_0_60": lambda: KeypointSmoother(
            mode="ema", ema_alpha=0.60, occlusion_guard=False
        ),
        "one_euro_stable": lambda: KeypointSmoother.from_config(
            config.analysis_smoothing,
            profile="stable",
            occlusion_guard=False,
        ),
        "one_euro_balanced": lambda: KeypointSmoother.from_config(
            config.analysis_smoothing,
            profile="balanced",
            occlusion_guard=False,
        ),
        "one_euro_responsive_current_baseline": lambda: KeypointSmoother.from_config(
            config.analysis_smoothing,
            profile="responsive",
            occlusion_guard=False,
        ),
        "joint_adaptive_round8": lambda: KeypointSmoother(
            mode="one-euro",
            profile="responsive",
            one_euro_min_cutoff=1.9,
            one_euro_beta=0.11,
            one_euro_d_cutoff=1.0,
            fast_joint_min_cutoff_scale=1.65,
            fast_joint_beta_scale=1.80,
            stable_joint_min_cutoff_scale=0.55,
            stable_joint_beta_scale=0.40,
            world_min_cutoff_scale=0.90,
            world_beta_scale=0.85,
            occlusion_guard=False,
        ),
        "joint_adaptive_round8_v2": lambda: KeypointSmoother(
            mode="one-euro",
            profile="responsive",
            one_euro_min_cutoff=2.0,
            one_euro_beta=0.12,
            one_euro_d_cutoff=1.0,
            fast_joint_min_cutoff_scale=1.50,
            fast_joint_beta_scale=1.80,
            stable_joint_min_cutoff_scale=0.75,
            stable_joint_beta_scale=0.50,
            world_min_cutoff_scale=0.90,
            world_beta_scale=0.85,
            occlusion_guard=False,
        ),
    }


def apply_filter(
    raw_records: Sequence[Mapping[str, Any]], profile: str
) -> list[PoseResult]:
    factories = filter_factories()
    if profile not in factories:
        raise KeyError(profile)
    smoother = factories[profile]()
    output: list[PoseResult] = []
    for record in raw_records:
        pose = pose_result_from_raw(record)
        output.append(
            smoother.smooth_result(
                pose,
                capture_timestamp_ns=int(
                    round(float(record["source_timestamp_ms"]) * 1_000_000.0)
                ),
            )
        )
    return output


def _point_map(result: PoseResult) -> dict[str, Keypoint]:
    return {point.name: point for point in result.keypoints}


def _body_scale(result: PoseResult) -> float:
    points = _point_map(result)
    if all(
        name in points
        for name in ("left_hip", "right_hip", "left_shoulder", "right_shoulder")
    ):
        hip = np.mean(
            [
                [points["left_hip"].x, points["left_hip"].y],
                [points["right_hip"].x, points["right_hip"].y],
            ],
            axis=0,
        )
        shoulder = np.mean(
            [
                [points["left_shoulder"].x, points["left_shoulder"].y],
                [points["right_shoulder"].x, points["right_shoulder"].y],
            ],
            axis=0,
        )
        value = float(np.linalg.norm(shoulder - hip))
        if math.isfinite(value) and value > 1e-6:
            return value
    return 0.25


def _trajectory(
    results: Sequence[PoseResult], name: str
) -> tuple[np.ndarray, np.ndarray]:
    values = np.full((len(results), 2), np.nan, dtype=float)
    confidence = np.zeros(len(results), dtype=float)
    for index, result in enumerate(results):
        point = _point_map(result).get(name)
        if (
            point is not None
            and point.confidence >= 0.2
            and math.isfinite(point.x)
            and math.isfinite(point.y)
        ):
            values[index] = (point.x, point.y)
            confidence[index] = point.confidence
    return values, confidence


def _lag_frames(raw: np.ndarray, candidate: np.ndarray, max_lag: int = 5) -> int:
    raw_velocity = np.diff(raw, axis=0)
    candidate_velocity = np.diff(candidate, axis=0)
    best_lag = 0
    best_score = -2.0
    for lag in range(max_lag + 1):
        left = raw_velocity[: len(raw_velocity) - lag or None]
        right = candidate_velocity[lag:]
        usable = np.all(np.isfinite(left), axis=1) & np.all(
            np.isfinite(right), axis=1
        )
        if int(np.sum(usable)) < 10:
            continue
        left_flat = left[usable].reshape(-1)
        right_flat = right[usable].reshape(-1)
        if np.std(left_flat) <= 1e-10 or np.std(right_flat) <= 1e-10:
            continue
        score = float(np.corrcoef(left_flat, right_flat)[0, 1])
        if math.isfinite(score) and score > best_score:
            best_score = score
            best_lag = lag
    return best_lag


def _jitter(values: np.ndarray, scales: np.ndarray) -> float | None:
    if len(values) < 3:
        return None
    acceleration = np.diff(values, n=2, axis=0)
    usable = np.all(np.isfinite(acceleration), axis=1)
    if not np.any(usable):
        return None
    aligned_scales = scales[2:][usable]
    aligned_scales = np.where(aligned_scales > 1e-6, aligned_scales, 0.25)
    normalized = np.linalg.norm(acceleration[usable], axis=1) / aligned_scales
    return float(np.median(normalized))


def _bone_variation(results: Sequence[PoseResult], names: Sequence[str]) -> float | None:
    lengths: list[float] = []
    for result in results:
        points = _point_map(result)
        if len(names) == 2 and all(name in points for name in names):
            first, second = points[names[0]], points[names[1]]
            if min(first.confidence, second.confidence) >= 0.2:
                length = math.hypot(first.x - second.x, first.y - second.y)
                if math.isfinite(length):
                    lengths.append(length)
    if not lengths or float(np.mean(lengths)) <= 1e-9:
        return None
    return float(np.std(lengths) / np.mean(lengths))


def _swap_rate(results: Sequence[PoseResult], group: Sequence[str]) -> float:
    pairs = [
        (name, name.replace("left_", "right_"))
        for name in group
        if name.startswith("left_")
        and name.replace("left_", "right_") in group
    ]
    swaps = 0
    comparisons = 0
    previous: dict[str, np.ndarray] = {}
    for result in results:
        points = _point_map(result)
        current = {
            name: np.asarray([point.x, point.y])
            for name, point in points.items()
            if point.confidence >= 0.2
            and math.isfinite(point.x)
            and math.isfinite(point.y)
        }
        for left, right in pairs:
            if all(name in current and name in previous for name in (left, right)):
                keep = np.linalg.norm(current[left] - previous[left])
                keep += np.linalg.norm(current[right] - previous[right])
                swap = np.linalg.norm(current[left] - previous[right])
                swap += np.linalg.norm(current[right] - previous[left])
                swaps += int(swap < 0.75 * keep)
                comparisons += 1
        previous = current
    return swaps / max(1, comparisons)


def _anchor_endpoint_errors(
    raw_results: Sequence[PoseResult],
    candidate_results: Sequence[PoseResult],
    anchors: Sequence[Mapping[str, Any]],
    *,
    fps: float,
) -> list[float]:
    errors: list[float] = []
    for anchor in anchors:
        frame = int(anchor["frame_index"])
        group = str(anchor.get("joint_group") or "torso")
        names = JOINT_GROUPS.get(group, JOINT_GROUPS["torso"])
        start, end = max(1, frame - 5), min(len(raw_results) - 2, frame + 5)
        if end <= start:
            continue

        def peak(results: Sequence[PoseResult]) -> int | None:
            scores = []
            for index in range(start, end + 1):
                previous = _point_map(results[index - 1])
                current = _point_map(results[index])
                following = _point_map(results[index + 1])
                acceleration = []
                for name in names:
                    if all(name in mapping for mapping in (previous, current, following)):
                        a, b, c = previous[name], current[name], following[name]
                        if min(a.confidence, b.confidence, c.confidence) >= 0.2:
                            acceleration.append(
                                math.hypot(c.x - 2 * b.x + a.x, c.y - 2 * b.y + a.y)
                            )
                scores.append((float(np.mean(acceleration)) if acceleration else -1.0, index))
            return max(scores)[1] if scores and max(scores)[0] >= 0 else None

        raw_peak = peak(raw_results)
        candidate_peak = peak(candidate_results)
        if raw_peak is not None and candidate_peak is not None:
            errors.append(abs(candidate_peak - raw_peak) * 1000.0 / max(1e-6, fps))
    return errors


def evaluate_temporal_profile(
    raw_results: Sequence[PoseResult],
    candidate_results: Sequence[PoseResult],
    *,
    fps: float,
    anchors: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    scales = np.asarray([_body_scale(result) for result in raw_results])
    groups: dict[str, Any] = {}
    all_lags: list[float] = []
    all_jitter: list[float] = []
    all_missing: list[float] = []
    all_swap: list[float] = []
    all_bone_variation: list[float] = []
    for group, names in JOINT_GROUPS.items():
        lags: list[float] = []
        jitters: list[float] = []
        missing: list[float] = []
        for name in names:
            raw_values, _ = _trajectory(raw_results, name)
            candidate_values, confidence = _trajectory(candidate_results, name)
            lags.append(
                _lag_frames(raw_values, candidate_values)
                * 1000.0
                / max(1e-6, fps)
            )
            value = _jitter(candidate_values, scales)
            if value is not None:
                jitters.append(value)
            missing.append(float(np.mean(confidence < 0.2)))
        swap_rate = _swap_rate(candidate_results, names)
        groups[group] = {
            "joint_temporal_lag_ms": float(np.mean(lags)) if lags else None,
            "jitter_normalized": float(np.mean(jitters)) if jitters else None,
            "missing_rate": float(np.mean(missing)) if missing else None,
            "left_right_swap_rate": swap_rate,
            "bone_length_variation": _bone_variation(
                candidate_results,
                ("left_shoulder", "left_hip")
                if group in {"shoulder", "torso", "hip"}
                else ("left_hip", "left_knee")
                if group == "knee"
                else ("left_knee", "left_ankle")
                if group == "ankle"
                else ("left_shoulder", "left_wrist"),
            ),
        }
        all_lags.extend(lags)
        all_jitter.extend(jitters)
        all_missing.extend(missing)
        all_swap.append(swap_rate)
        if groups[group]["bone_length_variation"] is not None:
            all_bone_variation.append(
                float(groups[group]["bone_length_variation"])
            )
    endpoint_errors = _anchor_endpoint_errors(
        raw_results,
        candidate_results,
        anchors,
        fps=fps,
    )
    return {
        "groups": groups,
        "summary": {
            "joint_temporal_lag_ms": float(np.mean(all_lags)) if all_lags else None,
            "event_endpoint_error_ms": (
                float(np.mean(endpoint_errors)) if endpoint_errors else None
            ),
            "jitter_normalized": (
                float(np.mean(all_jitter)) if all_jitter else None
            ),
            "missing_rate": float(np.mean(all_missing)) if all_missing else None,
            "left_right_swap_rate": (
                float(np.mean(all_swap)) if all_swap else None
            ),
            "bone_length_variation": (
                float(np.mean(all_bone_variation))
                if all_bone_variation
                else None
            ),
            "pose_age_at_render_ms": None,
            "pose_age_at_render_reason": (
                "offline file replay has no compositor/render clock; software "
                "inference latency is reported separately"
            ),
        },
    }


def aggregate_profile_metrics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summaries = [record["summary"] for record in records]
    fields = (
        "joint_temporal_lag_ms",
        "event_endpoint_error_ms",
        "jitter_normalized",
        "missing_rate",
        "left_right_swap_rate",
        "bone_length_variation",
    )
    output: dict[str, Any] = {}
    for field in fields:
        values = [
            float(summary[field])
            for summary in summaries
            if summary.get(field) is not None
        ]
        output[field] = {
            "count": len(values),
            "mean": float(np.mean(values)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
        }
    return output


def select_causal_profile(
    profile_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    raw_jitter = float(
        profile_reports["raw"]["jitter_normalized"]["mean"] or 1.0
    )
    baseline_name = "one_euro_responsive_current_baseline"
    scored = []
    for name, report in profile_reports.items():
        lag = float(report["joint_temporal_lag_ms"]["mean"] or 0.0)
        jitter = float(report["jitter_normalized"]["mean"] or raw_jitter)
        missing = float(report["missing_rate"]["mean"] or 0.0)
        endpoint = float(report["event_endpoint_error_ms"]["mean"] or lag)
        score = (
            0.35 * lag / 50.0
            + 0.35 * jitter / max(1e-9, raw_jitter)
            + 0.20 * endpoint / 50.0
            + 0.10 * missing
        )
        scored.append(
            {
                "profile": name,
                "lag_jitter_endpoint_score": score,
                "missing_rate": missing,
            }
        )
    eligible = [
        item
        for item in scored
        if item["profile"] != "raw" and item["missing_rate"] <= 0.10
    ]
    selected = min(eligible, key=lambda item: item["lag_jitter_endpoint_score"])
    baseline = next(item for item in scored if item["profile"] == baseline_name)
    return {
        "selected_profile": selected["profile"],
        "selected_score": selected["lag_jitter_endpoint_score"],
        "current_baseline_profile": baseline_name,
        "current_baseline_score": baseline["lag_jitter_endpoint_score"],
        "improves_current_baseline": (
            selected["lag_jitter_endpoint_score"]
            < baseline["lag_jitter_endpoint_score"]
        ),
        "ranked_profiles": sorted(
            scored, key=lambda item: item["lag_jitter_endpoint_score"]
        ),
        "selection_constraints": {
            "missing_rate_max": 0.10,
            "future_frames_allowed": False,
            "prediction_allowed": False,
        },
    }


def _prediction_scale(name: str) -> float:
    if name in FACE_NAMES:
        return 0.0
    if name in STABLE_JOINT_NAMES:
        return 0.45
    return 1.0


def predict_display_pose(
    causal_results: Sequence[PoseResult],
    *,
    fps: float,
    horizon_ms: float,
) -> list[PoseResult]:
    output: list[PoseResult] = []
    previous_velocity: dict[str, np.ndarray] = {}
    dt = 1.0 / max(1e-6, fps)
    horizon_seconds = max(0.0, min(45.0, float(horizon_ms))) / 1000.0
    for index, current in enumerate(causal_results):
        if index == 0 or not current.success:
            output.append(current)
            continue
        previous = _point_map(causal_results[index - 1])
        points: list[Keypoint] = []
        body_scale = _body_scale(current)
        max_displacement = 0.06 * body_scale
        for point in current.keypoints:
            prior = previous.get(point.name)
            if (
                prior is None
                or min(point.confidence, prior.confidence) < 0.70
                or not all(
                    math.isfinite(value)
                    for value in (point.x, point.y, prior.x, prior.y)
                )
            ):
                points.append(point)
                continue
            velocity = np.asarray(
                [(point.x - prior.x) / dt, (point.y - prior.y) / dt]
            )
            scale = _prediction_scale(point.name)
            old_velocity = previous_velocity.get(point.name)
            if old_velocity is not None and float(np.dot(velocity, old_velocity)) < 0:
                scale *= 0.25
            displacement = velocity * horizon_seconds * 0.85 * scale
            length = float(np.linalg.norm(displacement))
            if length > max_displacement > 0:
                displacement *= max_displacement / length
            if point.name in {"left_ankle", "right_ankle", "left_heel", "right_heel"}:
                displacement[0] = 0.0
            points.append(
                replace(
                    point,
                    x=point.x + float(displacement[0]),
                    y=point.y + float(displacement[1]),
                )
            )
            previous_velocity[point.name] = velocity
        extra = dict(current.extra)
        extra.update(
            {
                "display_only": True,
                "prediction_horizon_ms": horizon_ms,
                "filter_version": DISPLAY_FILTER_VERSION,
            }
        )
        output.append(replace(current, keypoints=points, extra=extra))
    return output


def overshoot_after_reversal(
    raw_results: Sequence[PoseResult],
    predicted_results: Sequence[PoseResult],
) -> float:
    values: list[float] = []
    for name in FAST_JOINT_NAMES:
        raw, _ = _trajectory(raw_results, name)
        predicted, _ = _trajectory(predicted_results, name)
        velocity = np.diff(raw, axis=0)
        for index in range(1, len(velocity)):
            if not (
                np.all(np.isfinite(velocity[index - 1]))
                and np.all(np.isfinite(velocity[index]))
                and np.all(np.isfinite(predicted[index + 1]))
            ):
                continue
            if float(np.dot(velocity[index - 1], velocity[index])) < 0:
                local = raw[max(0, index - 2) : min(len(raw), index + 4)]
                if np.any(np.all(np.isfinite(local), axis=1)):
                    finite = local[np.all(np.isfinite(local), axis=1)]
                    lower, upper = np.min(finite, axis=0), np.max(finite, axis=0)
                    excess = np.maximum(lower - predicted[index + 1], 0.0)
                    excess += np.maximum(predicted[index + 1] - upper, 0.0)
                    values.append(float(np.linalg.norm(excess)))
    return float(np.mean(values)) if values else 0.0


def support_foot_horizontal_drift(
    causal_results: Sequence[PoseResult],
    predicted_results: Sequence[PoseResult],
) -> float:
    values: list[float] = []
    for causal, predicted in zip(causal_results, predicted_results):
        source = _point_map(causal)
        display = _point_map(predicted)
        for name in (
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
        ):
            if name in source and name in display:
                values.append(abs(display[name].x - source[name].x))
    return float(np.mean(values)) if values else 0.0


def select_display_horizon(
    raw_records_by_record: Mapping[str, Sequence[Mapping[str, Any]]],
    causal_by_record: Mapping[str, Sequence[PoseResult]],
    fps_by_record: Mapping[str, float],
) -> dict[str, Any]:
    rows = []
    for horizon in (0.0, 15.0, 30.0, 45.0):
        lag_values = []
        jitter_values = []
        overshoot_values = []
        missing_values = []
        bone_variation_values = []
        foot_drift_values = []
        for record_id, raw_records in raw_records_by_record.items():
            raw = [pose_result_from_raw(record) for record in raw_records]
            predicted = predict_display_pose(
                causal_by_record[record_id],
                fps=fps_by_record[record_id],
                horizon_ms=horizon,
            )
            metrics = evaluate_temporal_profile(
                raw,
                predicted,
                fps=fps_by_record[record_id],
            )["summary"]
            lag_values.append(float(metrics["joint_temporal_lag_ms"] or 0.0))
            jitter_values.append(float(metrics["jitter_normalized"] or 0.0))
            missing_values.append(float(metrics["missing_rate"] or 0.0))
            bone_variation_values.append(
                float(metrics["bone_length_variation"] or 0.0)
            )
            overshoot_values.append(overshoot_after_reversal(raw, predicted))
            foot_drift_values.append(
                support_foot_horizontal_drift(
                    causal_by_record[record_id], predicted
                )
            )
        row = {
            "prediction_horizon_ms": horizon,
            "joint_temporal_lag_ms": float(np.mean(lag_values)),
            "jitter_normalized": float(np.mean(jitter_values)),
            "overshoot_after_reversal": float(np.mean(overshoot_values)),
            "missing_rate": float(np.mean(missing_values)),
            "bone_length_variation": float(
                np.mean(bone_variation_values)
            ),
            "support_foot_horizontal_drift": float(
                np.mean(foot_drift_values)
            ),
        }
        rows.append(row)
    baseline = rows[0]
    max_overshoot = max(0.005, baseline["overshoot_after_reversal"] + 0.002)
    eligible = [
        row
        for row in rows
        if row["overshoot_after_reversal"] <= max_overshoot
        and row["missing_rate"] <= baseline["missing_rate"] + 1e-9
        and row["jitter_normalized"] <= baseline["jitter_normalized"] * 1.20
        and row["bone_length_variation"]
        <= baseline["bone_length_variation"] * 1.05 + 1e-9
        and row["support_foot_horizontal_drift"] <= 1e-9
    ]
    selected = min(
        eligible or [baseline],
        key=lambda row: (
            row["joint_temporal_lag_ms"],
            row["jitter_normalized"],
        ),
    )
    return {
        "selected": selected,
        "grid": rows,
        "lag_reduced": (
            selected["joint_temporal_lag_ms"]
            < baseline["joint_temporal_lag_ms"]
        ),
        "overshoot_gate_passed": (
            selected["overshoot_after_reversal"] <= max_overshoot
        ),
        "support_foot_horizontal_prediction_scale": 0.0,
        "maximum_body_scale_displacement": 0.06,
        "display_only": True,
    }


def centered_offline_assist(results: Sequence[PoseResult]) -> list[PoseResult]:
    output: list[PoseResult] = []
    for index, result in enumerate(results):
        start, end = max(0, index - 2), min(len(results), index + 3)
        window = [_point_map(item) for item in results[start:end]]
        points = []
        for point in result.keypoints:
            values = [
                mapping[point.name]
                for mapping in window
                if point.name in mapping and mapping[point.name].confidence >= 0.2
            ]
            if not values:
                points.append(point)
                continue
            points.append(
                replace(
                    point,
                    x=float(np.mean([value.x for value in values])),
                    y=float(np.mean([value.y for value in values])),
                    z=float(np.mean([value.z for value in values])),
                )
            )
        extra = dict(result.extra)
        extra.update(
            {
                "future_frames_used": True,
                "annotation_assist_only": True,
                "filter_version": OFFLINE_ASSIST_VERSION,
            }
        )
        output.append(replace(result, keypoints=points, extra=extra))
    return output


def derived_point_payload(
    point: Keypoint,
    *,
    source_frame_id: int,
    source_timestamp_ms: float,
    filter_version: str,
    prediction_horizon_ms: float,
) -> dict[str, Any]:
    def finite(value: float) -> float | None:
        return float(value) if math.isfinite(float(value)) else None

    return {
        "name": point.name,
        "x": finite(point.x),
        "y": finite(point.y),
        "z": finite(point.z),
        "confidence": float(point.confidence),
        "visibility": (
            finite(point.visibility) if point.visibility is not None else None
        ),
        "presence": (
            finite(point.presence) if point.presence is not None else None
        ),
        "source_frame_id": int(source_frame_id),
        "source_timestamp_ms": float(source_timestamp_ms),
        "generated_at_ns": time.time_ns(),
        "filter_version": filter_version,
        "prediction_horizon_ms": float(prediction_horizon_ms),
    }


__all__ = [
    "DISPLAY_FILTER_VERSION",
    "FILTER_VERSION",
    "JOINT_GROUPS",
    "OFFLINE_ASSIST_VERSION",
    "aggregate_profile_metrics",
    "apply_filter",
    "centered_offline_assist",
    "derived_point_payload",
    "evaluate_temporal_profile",
    "filter_factories",
    "overshoot_after_reversal",
    "pose_result_from_raw",
    "predict_display_pose",
    "select_causal_profile",
    "select_display_horizon",
    "support_foot_horizontal_drift",
]
