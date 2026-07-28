"""Internal LOVO experiment for phase, baseline, contact and confidence evidence."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hyrox.features import extract_basic_pose_features
from src.biomechanics.shadow_evidence_3d import BodyRelative3DTracker
from src.temporal_evidence import (
    PHASE_VECTOR_NAMES,
    RidgePhaseModel,
    candidate_metrics,
    decode_phase_candidates,
    estimate_standing_baseline,
    phase_feature_matrix,
    phase_metrics,
    rle_roi_foreground_ratio,
)
from tools.evaluate_reviewed_rgb_guidance import (
    PROJECT_ROOT,
    TARGET_ACTIONS,
    _evaluate_record,
    _group_metrics,
    _load,
    _write,
)
from tools.run_2d_3d_shadow_evidence import _refresh_record_status_fields


PHASE_ORDERS: dict[str, tuple[str, ...]] = {
    "lunge": ("stand", "descent", "bottom", "contact", "ascent", "stand"),
    "wall_ball": ("stand", "descent", "bottom", "ascent", "release", "recovery"),
    "burpee_broad_jump": (
        "hands_down",
        "chest_down",
        "takeoff",
        "flight",
        "landing",
        "stabilization",
    ),
    "rowing": ("catch", "drive", "finish", "recovery"),
    "skierg": ("top", "pull_down", "bottom", "return", "top"),
    "sled_push": ("drive", "step"),
    "sled_pull": ("reach", "pull", "recover", "reach"),
    "farmers_carry": ("carrying", "rest"),
}

CONTACT_ACTIONS = {
    "lunge": {
        "phase": "contact",
        "error": "NO_KNEE_CONTACT",
        "event": "rear_knee_contact_candidate",
    },
    "burpee_broad_jump": {
        "phase": "chest_down",
        "error": "NO_CHEST_CONTACT",
        "event": "chest_contact_candidate",
    },
}

CONTACT_FEATURE_NAMES = (
    "surface_height_ratio",
    "surface_velocity",
    "near_floor_dwell",
    "torso_horizontal_score_2d",
    "visible_score",
    "segmentation_floor_overlap",
    "segmentation_foreground_fraction",
    "three_d_vertical_relation",
    "three_d_prone_score",
)


def _record_labels(review: Mapping[str, Any]) -> list[str]:
    start = int(review["usable_start_frame"])
    end = int(review["usable_end_frame"])
    labels: list[str | None] = [None] * (end - start + 1)
    for interval in review.get("phase_error_intervals") or []:
        if not isinstance(interval, Mapping):
            continue
        left = max(start, int(interval["start_frame"]))
        right = min(end, int(interval["end_frame"]))
        for frame in range(left, right + 1):
            labels[frame - start] = str(interval["phase"])
    previous = None
    for index, value in enumerate(labels):
        if value is not None:
            previous = value
        elif previous is not None:
            labels[index] = previous
    following = None
    for index in range(len(labels) - 1, -1, -1):
        value = labels[index]
        if value is not None:
            following = value
        elif following is not None:
            labels[index] = following
    fallback = next((value for value in labels if value is not None), "unknown")
    return [str(value or fallback) for value in labels]


def _load_frames(
    root: Path,
    manifest: Mapping[str, Any],
    review: Mapping[str, Any],
) -> dict[str, Any]:
    video = manifest.get("video") or {}
    start = int(review["usable_start_frame"])
    end = int(review["usable_end_frame"])
    path = root / manifest["pose_cache"]["causal_analysis_pose"]
    frames: list[dict[str, Any]] = [
        {"visible_score": 0.0} for _ in range(end - start + 1)
    ]
    body_tracker = BodyRelative3DTracker()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            frame_index = int(row["frame_index"])
            if frame_index < start or frame_index > end:
                continue
            features = extract_basic_pose_features(
                row.get("image_normalized_2d"),
                image_width=int(video.get("width", 1) or 1),
                image_height=int(video.get("height", 1) or 1),
            )
            body = body_tracker.update(
                row.get("image_normalized_2d") or [],
                row.get("mp_world_body_3d") or [],
                timestamp_ms=row.get("source_timestamp_ms"),
                camera_view=str(manifest.get("camera_view", "unknown")),
            )
            height = body.get("hip_compensated_height_downward_positive")
            torso = body.get("torso_spatial")
            if isinstance(height, Mapping):
                knees = [
                    float(height[name])
                    for name in ("left_knee", "right_knee")
                    if height.get(name) is not None
                ]
                features["three_d_knee_below_hip"] = (
                    max(knees) if knees else None
                )
            if isinstance(torso, Mapping):
                features["three_d_prone_score"] = torso.get(
                    "prone_horizontal_score"
                )
            features["formal_pose_eligible"] = bool(
                row.get("formal_pose_eligible")
            )
            frames[frame_index - start] = features
    baseline = estimate_standing_baseline(frames)
    matrix = phase_feature_matrix(frames, baseline)
    return {
        "record_id": str(review["record_id"]),
        "action": str(review["action"]),
        "start_frame": start,
        "end_frame": end,
        "frames": frames,
        "matrix": matrix,
        "labels": _record_labels(review),
        "baseline": baseline,
        "review": review,
        "manifest": manifest,
    }


def _phase_lovo(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_action: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_action.setdefault(record["action"], []).append(record)
    folds = []
    linear_metrics = []
    hmm_metrics = []
    naive_candidate_metrics = []
    calibrated_candidate_metrics = []
    for action, action_records in sorted(by_action.items()):
        classes = tuple(
            dict.fromkeys(
                phase
                for record in action_records
                for phase in record["labels"]
            )
        )
        order = PHASE_ORDERS[action]
        for held_out in action_records:
            training = [
                record
                for record in action_records
                if record["record_id"] != held_out["record_id"]
            ]
            matrix = np.vstack([record["matrix"] for record in training])
            labels = [
                phase for record in training for phase in record["labels"]
            ]
            model = RidgePhaseModel.fit(
                matrix,
                labels,
                [record["labels"] for record in training],
                classes=classes,
            )
            linear = model.predict_linear(held_out["matrix"])
            hmm = model.predict_causal_hmm(held_out["matrix"])
            linear_metric = phase_metrics(held_out["labels"], linear)
            hmm_metric = phase_metrics(held_out["labels"], hmm)
            linear_metrics.append(linear_metric)
            hmm_metrics.append(hmm_metric)

            human_candidates = [
                (
                    int(rep["start_frame"]) - held_out["start_frame"],
                    int(rep["end_frame"]) - held_out["start_frame"],
                )
                for rep in held_out["review"].get("reps") or []
            ]
            naive_candidates, naive_audit = decode_phase_candidates(
                linear,
                order,
                minimum_run_frames=1,
                allow_single_phase_skip=False,
            )
            naive_metric = candidate_metrics(
                human_candidates,
                naive_candidates,
            )
            naive_candidate_metrics.append(naive_metric)

            parameter_rows = []
            for minimum_run in (1, 2, 3, 4):
                for maximum_phase_skips in (0, 1, 2):
                    training_metrics = []
                    for record in training:
                        predicted = model.predict_causal_hmm(record["matrix"])
                        candidates, _audit = decode_phase_candidates(
                            predicted,
                            order,
                            minimum_run_frames=minimum_run,
                            maximum_phase_skips=maximum_phase_skips,
                        )
                        expected = [
                            (
                                int(rep["start_frame"]) - record["start_frame"],
                                int(rep["end_frame"]) - record["start_frame"],
                            )
                            for rep in record["review"].get("reps") or []
                        ]
                        training_metrics.append(
                            candidate_metrics(expected, candidates)
                        )
                    aggregate = _aggregate_candidate_metrics(training_metrics)
                    parameter_rows.append(
                        {
                            "minimum_run_frames": minimum_run,
                            "maximum_phase_skips": maximum_phase_skips,
                            "training_metrics": aggregate,
                        }
                    )
            selected = max(
                parameter_rows,
                key=lambda row: (
                    row["training_metrics"]["candidate_recall"] or 0.0,
                    row["training_metrics"]["candidate_precision"] or 0.0,
                    -(
                        row["training_metrics"]["terminal_mae_frames"]
                        if row["training_metrics"]["terminal_mae_frames"]
                        is not None
                        else 1e9
                    ),
                    -int(row["maximum_phase_skips"]),
                    -int(row["minimum_run_frames"]),
                ),
            )
            calibrated_candidates, calibrated_audit = decode_phase_candidates(
                hmm,
                order,
                minimum_run_frames=int(selected["minimum_run_frames"]),
                maximum_phase_skips=int(selected["maximum_phase_skips"]),
            )
            calibrated_metric = candidate_metrics(
                human_candidates,
                calibrated_candidates,
            )
            calibrated_candidate_metrics.append(calibrated_metric)
            folds.append(
                {
                    "held_out_video_id": held_out["record_id"],
                    "training_video_ids": [
                        record["record_id"] for record in training
                    ],
                    "action": action,
                    "linear_phase_metrics": linear_metric,
                    "causal_hmm_phase_metrics": hmm_metric,
                    "selected_candidate_parameters": {
                        key: selected[key]
                        for key in (
                            "minimum_run_frames",
                            "maximum_phase_skips",
                        )
                    },
                    "naive_candidate_audit": naive_audit,
                    "calibrated_candidate_audit": calibrated_audit,
                    "naive_candidate_metrics": naive_metric,
                    "calibrated_candidate_metrics": calibrated_metric,
                    "training_test_overlap": sorted(
                        {held_out["record_id"]}.intersection(
                            record["record_id"] for record in training
                        )
                    ),
                }
            )
    return {
        "model_family": "ridge_linear_emissions_plus_causal_hmm",
        "feature_names": list(PHASE_VECTOR_NAMES),
        "fold_count": len(folds),
        "linear_phase_metrics": _aggregate_phase_metrics(linear_metrics),
        "causal_hmm_phase_metrics": _aggregate_phase_metrics(hmm_metrics),
        "naive_linear_candidate_metrics": _aggregate_candidate_metrics(
            naive_candidate_metrics
        ),
        "calibrated_hmm_candidate_metrics": _aggregate_candidate_metrics(
            calibrated_candidate_metrics
        ),
        "recommended_candidate_role": (
            "deduplication_and_settlement_sidecar_not_candidate_source"
        ),
        "all_training_test_overlaps_empty": all(
            not fold["training_test_overlap"] for fold in folds
        ),
        "folds": folds,
    }


def _standing_lovo(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        record
        for record in records
        if record["action"] in {"lunge", "wall_ball"}
    ]
    fixed_rows = []
    personalized_rows = []
    folds = []
    for held_out in eligible:
        training = [
            record
            for record in eligible
            if record["record_id"] != held_out["record_id"]
            and record["action"] == held_out["action"]
        ]
        candidates = []
        for margin in (4.0, 6.0, 8.0, 10.0, 12.0, 15.0):
            metrics = _standing_metrics(training, personalized_margin=margin)
            candidates.append((metrics["f1"] or 0.0, -margin, margin, metrics))
        _score, _preference, margin, training_metrics = max(candidates)
        fixed = _standing_metrics([held_out], personalized_margin=None)
        personalized = _standing_metrics(
            [held_out],
            personalized_margin=margin,
        )
        fixed_rows.append(fixed)
        personalized_rows.append(personalized)
        folds.append(
            {
                "held_out_video_id": held_out["record_id"],
                "training_video_ids": [
                    record["record_id"] for record in training
                ],
                "selected_extension_margin_deg": margin,
                "training_metrics": training_metrics,
                "held_out_fixed": fixed,
                "held_out_personalized": personalized,
                "automatic_baseline": held_out["baseline"].as_dict(),
            }
        )
    return {
        "quality_gate": {
            "minimum_knee_angle_deg": 140.0,
            "minimum_hip_angle_deg": 135.0,
            "minimum_sample_count": 5,
            "maximum_angle_mad_deg": 10.0,
            "maximum_hip_speed_norm_per_frame": 0.02,
            "fallback": "fixed_thresholds",
        },
        "fixed_threshold_metrics": _sum_binary_metrics(fixed_rows),
        "personalized_baseline_metrics": _sum_binary_metrics(
            personalized_rows
        ),
        "reliable_baseline_video_count": sum(
            int(record["baseline"].reliable) for record in eligible
        ),
        "fallback_video_count": sum(
            int(not record["baseline"].reliable) for record in eligible
        ),
        "folds": folds,
    }


def _standing_metrics(
    records: Sequence[dict[str, Any]],
    *,
    personalized_margin: float | None,
) -> dict[str, float | int | None]:
    tp = fp = fn = tn = 0
    for record in records:
        baseline = record["baseline"]
        use_personalized = (
            personalized_margin is not None and baseline.reliable
        )
        knee_threshold = (
            165.0
            if not use_personalized
            else max(145.0, baseline.knee_angle - personalized_margin)
        )
        hip_threshold = (
            165.0
            if not use_personalized
            else max(140.0, baseline.hip_angle - personalized_margin)
        )
        for frame, phase in zip(record["frames"], record["labels"]):
            knees = [
                float(frame[name])
                for name in ("left_knee_angle", "right_knee_angle")
                if frame.get(name) is not None
            ]
            hips = [
                float(frame[name])
                for name in ("left_hip_angle", "right_hip_angle")
                if frame.get(name) is not None
            ]
            if not knees or not hips:
                continue
            predicted = min(knees) >= knee_threshold and min(hips) >= hip_threshold
            expected = phase == "stand" or (
                record["action"] == "wall_ball"
                and phase in {"release", "recovery"}
            )
            if predicted and expected:
                tp += 1
            elif predicted:
                fp += 1
            elif expected:
                fn += 1
            else:
                tn += 1
    return _binary_metrics(tp, fp, fn, tn)


def _add_mask_contact_features(root: Path, record: dict[str, Any]) -> None:
    action = record["action"]
    if action not in CONTACT_ACTIONS:
        return
    video = record["manifest"].get("video") or {}
    width = int(video.get("width", 1) or 1)
    height = int(video.get("height", 1) or 1)
    foot_y = [
        float(frame[name])
        for frame in record["frames"]
        for name in (
            "left_heel_y",
            "right_heel_y",
            "left_foot_index_y",
            "right_foot_index_y",
        )
        if frame.get(name) is not None
    ]
    floor_y = float(np.percentile(foot_y, 95)) if foot_y else 0.95
    body_height = max(record["baseline"].body_height, 0.2)
    raw_path = (
        root
        / record["manifest"]["pose_cache"]["raw_pose_paths"][
            "mediapipe_full"
        ]
    )
    start = record["start_frame"]
    end = record["end_frame"]
    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            frame_index = int(row["frame_index"])
            if frame_index < start or frame_index > end:
                continue
            frame = record["frames"][frame_index - start]
            if action == "lunge":
                candidates = [
                    (
                        float(frame[f"{side}_knee_y"]),
                        float(frame[f"{side}_knee_x"]),
                        side,
                    )
                    for side in ("left", "right")
                    if frame.get(f"{side}_knee_y") is not None
                    and frame.get(f"{side}_knee_x") is not None
                ]
                if candidates:
                    joint_y, center_x, side = max(candidates)
                    ankle_y = float(
                        frame.get(f"{side}_ankle_y") or joint_y
                    )
                    radius = 0.10 * abs(ankle_y - joint_y)
                    surface_y = joint_y + radius
                    frame["contact_surface_height_ratio"] = (
                        floor_y - surface_y
                    ) / body_height
                    frame["contact_center_x"] = center_x
                frame["contact_three_d_relation"] = frame.get(
                    "three_d_knee_below_hip"
                )
                roi_half_width = int(0.10 * width)
            else:
                shoulder_y = [
                    float(frame[name])
                    for name in ("left_shoulder_y", "right_shoulder_y")
                    if frame.get(name) is not None
                ]
                hip_y = [
                    float(frame[name])
                    for name in ("left_hip_y", "right_hip_y")
                    if frame.get(name) is not None
                ]
                center_x_values = [
                    float(frame[name])
                    for name in (
                        "left_shoulder_x",
                        "right_shoulder_x",
                        "left_hip_x",
                        "right_hip_x",
                    )
                    if frame.get(name) is not None
                ]
                if shoulder_y and hip_y and center_x_values:
                    chest_y = 0.65 * float(np.mean(shoulder_y)) + 0.35 * float(
                        np.mean(hip_y)
                    )
                    torso_length = abs(
                        float(np.mean(hip_y)) - float(np.mean(shoulder_y))
                    )
                    surface_y = chest_y + 0.20 * torso_length
                    frame["contact_surface_height_ratio"] = (
                        floor_y - surface_y
                    ) / body_height
                    frame["contact_center_x"] = float(
                        np.mean(center_x_values)
                    )
                frame["contact_three_d_relation"] = frame.get(
                    "three_d_prone_score"
                )
                roi_half_width = int(0.18 * width)
            center_x = frame.get("contact_center_x")
            if center_x is not None:
                pixel_x = int(float(center_x) * width)
                band = int(0.035 * height)
                frame["segmentation_floor_overlap"] = (
                    rle_roi_foreground_ratio(
                        row.get("mask"),
                        x0=pixel_x - roi_half_width,
                        y0=int(floor_y * height) - band,
                        x1=pixel_x + roi_half_width,
                        y1=int(floor_y * height) + band,
                    )
                )
            mask = row.get("mask")
            if isinstance(mask, Mapping):
                frame["segmentation_foreground_fraction"] = mask.get(
                    "foreground_fraction"
                )
    previous_height = None
    dwell = 0
    for frame in record["frames"]:
        current = _number(frame.get("contact_surface_height_ratio"))
        velocity = (
            0.0
            if current is None or previous_height is None
            else current - previous_height
        )
        if current is not None:
            previous_height = current
        dwell = dwell + 1 if current is not None and current <= 0.08 else 0
        frame["contact_surface_velocity"] = velocity
        frame["contact_near_floor_dwell"] = min(dwell, 10) / 10.0


def _contact_matrix(record: Mapping[str, Any], subset: str) -> np.ndarray:
    rows = []
    for frame in record["frames"]:
        torso = abs(_number(frame.get("torso_angle")) or 90.0)
        values = [
            _number(frame.get("contact_surface_height_ratio")),
            _number(frame.get("contact_surface_velocity")),
            _number(frame.get("contact_near_floor_dwell")),
            max(0.0, 1.0 - torso / 90.0),
            _number(frame.get("visible_score")),
            _number(frame.get("segmentation_floor_overlap")),
            _number(frame.get("segmentation_foreground_fraction")),
            _number(frame.get("contact_three_d_relation")),
            _number(frame.get("three_d_prone_score")),
        ]
        if subset == "two_d":
            values[5:] = [0.0] * 4
        elif subset == "two_d_segmentation":
            values[7:] = [0.0] * 2
        rows.append([0.0 if value is None else float(value) for value in values])
    return np.asarray(rows, dtype=float)


def _contact_frame_labels(record: Mapping[str, Any]) -> list[str]:
    spec = CONTACT_ACTIONS[record["action"]]
    start = record["start_frame"]
    labels = ["no_contact"] * len(record["frames"])
    for interval in record["review"].get("phase_error_intervals") or []:
        if not isinstance(interval, Mapping) or interval.get("phase") != spec["phase"]:
            continue
        positive = str(interval.get("error_code")) != spec["error"]
        for frame in range(
            max(start, int(interval["start_frame"])),
            min(record["end_frame"], int(interval["end_frame"])) + 1,
        ):
            labels[frame - start] = "contact" if positive else "no_contact"
    return labels


def _contact_lovo(
    root: Path,
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    eligible = [
        record for record in records if record["action"] in CONTACT_ACTIONS
    ]
    for record in eligible:
        _add_mask_contact_features(root, record)
    results = {}
    for subset in ("two_d", "two_d_segmentation", "two_d_segmentation_three_d"):
        fold_rows = []
        for held_out in eligible:
            training = [
                record
                for record in eligible
                if record["action"] == held_out["action"]
                and record["record_id"] != held_out["record_id"]
            ]
            matrix = np.vstack(
                [_contact_matrix(record, subset) for record in training]
            )
            labels = [
                label
                for record in training
                for label in _contact_frame_labels(record)
            ]
            model = RidgePhaseModel.fit(
                matrix,
                labels,
                [_contact_frame_labels(record) for record in training],
                classes=("no_contact", "contact"),
                l2=4.0,
            )
            train_rep_scores = [
                row
                for record in training
                for row in _contact_rep_scores(
                    record,
                    model.predict_proba(_contact_matrix(record, subset))[:, 1],
                )
            ]
            threshold = _contact_threshold(train_rep_scores)
            held_scores = _contact_rep_scores(
                held_out,
                model.predict_proba(
                    _contact_matrix(held_out, subset)
                )[:, 1],
            )
            metrics = _contact_score_metrics(held_scores, threshold)
            fold_rows.append(
                {
                    "held_out_video_id": held_out["record_id"],
                    "training_video_ids": [
                        record["record_id"] for record in training
                    ],
                    "threshold": threshold,
                    "metrics": metrics,
                    "rep_scores": held_scores,
                }
            )
        results[subset] = {
            "metrics": _sum_contact_metrics(
                [row["metrics"] for row in fold_rows]
            ),
            "folds": fold_rows,
        }
    two_d_folds = {
        row["held_out_video_id"]: row for row in results["two_d"]["folds"]
    }
    fused_folds = {
        row["held_out_video_id"]: row
        for row in results["two_d_segmentation_three_d"]["folds"]
    }
    shadow_folds = []
    for held_out_id, base_fold in two_d_folds.items():
        fused_fold = fused_folds[held_out_id]
        fused_scores = {
            (row["record_id"], row["rep_id"]): row
            for row in fused_fold["rep_scores"]
        }
        shadow_scores = []
        for base_score in base_fold["rep_scores"]:
            row = dict(base_score)
            fused = fused_scores[(row["record_id"], row["rep_id"])]
            row["predicted_event_frame"] = fused["predicted_event_frame"]
            row["event_frame_error"] = fused["event_frame_error"]
            row["decision_score_source"] = "two_d"
            row["event_anchor_source"] = "two_d_segmentation_three_d"
            shadow_scores.append(row)
        shadow_folds.append(
            {
                "held_out_video_id": held_out_id,
                "training_video_ids": base_fold["training_video_ids"],
                "threshold": base_fold["threshold"],
                "metrics": _contact_score_metrics(
                    shadow_scores,
                    float(base_fold["threshold"]),
                ),
                "rep_scores": shadow_scores,
            }
        )
    results["shadow_safe_fusion"] = {
        "decision_policy": (
            "two_d_contact_state_with_segmentation_three_d_event_anchor"
        ),
        "may_promote_unsure_to_contact": False,
        "may_infer_physical_contact": False,
        "metrics": _sum_contact_metrics(
            [row["metrics"] for row in shadow_folds]
        ),
        "folds": shadow_folds,
    }
    return {
        "feature_names": list(CONTACT_FEATURE_NAMES),
        "contact_semantics": (
            "human_reviewed_contact_proxy_not_force_or_metric_contact_truth"
        ),
        "recommended_policy": "shadow_safe_fusion",
        "ablations": results,
    }


def _contact_rep_scores(
    record: Mapping[str, Any],
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    spec = CONTACT_ACTIONS[record["action"]]
    errors_by_rep: dict[str, set[str]] = {}
    for interval in record["review"].get("phase_error_intervals") or []:
        if isinstance(interval, Mapping):
            errors_by_rep.setdefault(str(interval.get("rep_id")), set()).add(
                str(interval.get("error_code"))
            )
    events = {
        str(event.get("rep_id")): int(event["frame_index"])
        for event in record["review"].get("events") or []
        if isinstance(event, Mapping) and event.get("event_type") == spec["event"]
    }
    rows = []
    for rep in record["review"].get("reps") or []:
        rep_id = str(rep["rep_id"])
        left = max(0, int(rep["start_frame"]) - record["start_frame"])
        right = min(
            len(probabilities) - 1,
            int(rep["end_frame"]) - record["start_frame"],
        )
        if right < left:
            continue
        relative = int(np.argmax(probabilities[left : right + 1]))
        predicted_frame = record["start_frame"] + left + relative
        expected_contact = spec["error"] not in errors_by_rep.get(rep_id, set())
        human_event = events.get(rep_id)
        rows.append(
            {
                "record_id": record["record_id"],
                "rep_id": rep_id,
                "expected_contact": expected_contact,
                "score": float(probabilities[left + relative]),
                "predicted_event_frame": predicted_frame,
                "human_event_frame": human_event,
                "event_frame_error": (
                    None
                    if human_event is None
                    else predicted_frame - human_event
                ),
            }
        )
    return rows


def _contact_threshold(rows: Sequence[Mapping[str, Any]]) -> float:
    candidates = []
    for threshold in np.linspace(0.45, 0.90, 10):
        metrics = _contact_score_metrics(rows, float(threshold))
        fpr = metrics["false_positive"] / max(
            1,
            metrics["false_positive"] + metrics["true_negative"],
        )
        candidates.append(
            (
                fpr <= 0.10,
                metrics["recall"] or 0.0,
                metrics["precision"] or 0.0,
                threshold,
            )
        )
    feasible = [row for row in candidates if row[0]]
    return float(max(feasible or candidates)[3])


def _contact_score_metrics(
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    tp = fp = fn = tn = unsure = 0
    event_errors = []
    for row in rows:
        score = float(row["score"])
        predicted = (
            True
            if score >= threshold
            else False if score <= 1.0 - threshold else None
        )
        expected = bool(row["expected_contact"])
        if predicted is None:
            unsure += 1
        elif predicted and expected:
            tp += 1
        elif predicted:
            fp += 1
        elif expected:
            fn += 1
        else:
            tn += 1
        error = row.get("event_frame_error")
        if error is not None and expected:
            event_errors.append(abs(int(error)))
    metrics = _binary_metrics(tp, fp, fn, tn)
    metrics.update(
        {
            "unsure": unsure,
            "rep_count": len(rows),
            "unsure_rate": unsure / len(rows) if rows else None,
            "event_frame_mae": (
                sum(event_errors) / len(event_errors)
                if event_errors
                else None
            ),
            "event_frame_count": len(event_errors),
        }
    )
    return metrics


def _confidence_calibration(
    root: Path,
    manifests: Mapping[str, dict[str, Any]],
    reviews: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    core = [
        review for review in reviews if review["action"] in TARGET_ACTIONS
    ]
    baseline = {
        review["record_id"]: _evaluate_record(
            root,
            manifests[review["record_id"]],
            review,
            profile="optimized",
        )
        for review in core
    }
    calibrated = []
    fold_rows = []
    for held_out in core:
        held_id = held_out["record_id"]
        training = [
            baseline[review["record_id"]]
            for review in core
            if review["record_id"] != held_id
        ]
        thresholds = _fit_rule_confidence_thresholds(training)
        result = _apply_rule_confidence_thresholds(
            baseline[held_id],
            thresholds,
        )
        calibrated.append(result)
        fold_rows.append(
            {
                "held_out_video_id": held_id,
                "training_video_ids": [
                    review["record_id"]
                    for review in core
                    if review["record_id"] != held_id
                ],
                "thresholds_by_rule": thresholds,
                "promoted_unsure_to_no_rep": result[
                    "confidence_calibration"
                ]["promoted_unsure_to_no_rep"],
            }
        )
    baseline_records = [baseline[review["record_id"]] for review in core]
    baseline_metrics = _group_metrics(baseline_records)
    raw_calibrated_metrics = _group_metrics(calibrated)
    baseline_false_no_rep = _unsafe_no_rep_count(baseline_metrics)
    calibrated_false_no_rep = _unsafe_no_rep_count(raw_calibrated_metrics)
    safety_passed = calibrated_false_no_rep <= baseline_false_no_rep
    safe_records = calibrated if safety_passed else baseline_records
    return {
        "promotion_scope": "UNSURE_to_NO_REP_only_never_VALID",
        "minimum_training_precision": 1.0,
        "minimum_training_positive_reps": 5,
        "minimum_training_positive_videos": 3,
        "baseline_metrics": baseline_metrics,
        "raw_calibrated_metrics": raw_calibrated_metrics,
        "calibrated_metrics": _group_metrics(safe_records),
        "safety_gate": {
            "passed": safety_passed,
            "baseline_valid_or_unsure_to_no_rep": baseline_false_no_rep,
            "raw_calibrated_valid_or_unsure_to_no_rep": (
                calibrated_false_no_rep
            ),
            "selected_policy": (
                "lovo_calibration"
                if safety_passed
                else "baseline_no_promotion_fallback"
            ),
        },
        "folds": fold_rows,
        "records": safe_records,
    }


def _fit_rule_confidence_thresholds(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    by_rule: dict[str, list[tuple[float, bool, str]]] = {}
    for record in records:
        record_id = str(
            record.get("record_id")
            or record.get("video_id")
            or "unknown_record"
        )
        for match in record.get("matches") or []:
            if not isinstance(match, Mapping):
                continue
            candidate = match.get("candidate")
            human = match.get("human_rep")
            if not isinstance(candidate, Mapping) or not isinstance(human, Mapping):
                continue
            expected_no_rep = str(human.get("validity")) == "NO_REP"
            for rule in candidate.get("rules") or []:
                if not isinstance(rule, Mapping) or rule.get("status") != "FAIL":
                    continue
                by_rule.setdefault(str(rule.get("rule_id")), []).append(
                    (
                        float(rule.get("confidence", 0.0)),
                        expected_no_rep,
                        record_id,
                    )
                )
    thresholds = {}
    for rule_id, rows in by_rule.items():
        for threshold in sorted(
            {confidence for confidence, _label, _record_id in rows}
        ):
            selected = [
                (label, record_id)
                for confidence, label, record_id in rows
                if confidence >= threshold
            ]
            positives = sum(label for label, _record_id in selected)
            false_positives = len(selected) - positives
            positive_videos = {
                record_id for label, record_id in selected if label
            }
            if (
                positives >= 5
                and len(positive_videos) >= 3
                and false_positives == 0
            ):
                thresholds[rule_id] = threshold
                break
    return thresholds


def _unsafe_no_rep_count(metrics: Mapping[str, Any]) -> int:
    confusion = metrics.get("status_confusion") or {}
    return sum(
        int((confusion.get(expected) or {}).get("NO_REP", 0))
        for expected in ("VALID", "UNSURE")
    )


def _apply_rule_confidence_thresholds(
    record: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    result = copy.deepcopy(dict(record))
    promoted = 0
    for match in result.get("matches") or []:
        candidate = match.get("candidate") if isinstance(match, dict) else None
        if not isinstance(candidate, dict) or candidate.get("status") != "UNSURE":
            continue
        qualifying = [
            rule
            for rule in candidate.get("rules") or []
            if isinstance(rule, Mapping)
            and rule.get("status") == "FAIL"
            and str(rule.get("rule_id")) in thresholds
            and float(rule.get("confidence", 0.0))
            >= thresholds[str(rule.get("rule_id"))]
        ]
        if not qualifying:
            continue
        candidate["status"] = "NO_REP"
        candidate["reason_codes"] = list(
            dict.fromkeys(
                (
                    "TRAINING_CALIBRATED_CONFIDENT_FAIL",
                    *(candidate.get("reason_codes") or []),
                )
            )
        )
        promoted += 1
    _refresh_record_status_fields(result)
    result["confidence_calibration"] = {
        "promoted_unsure_to_no_rep": promoted,
        "valid_promotion_allowed": False,
    }
    return result


def _aggregate_phase_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    frames = sum(int(row["frame_count"]) for row in rows)
    human = sum(int(row["human_boundary_count"]) for row in rows)
    predicted = sum(int(row["predicted_boundary_count"]) for row in rows)
    matched = sum(int(row["matched_boundary_count"]) for row in rows)
    return {
        "record_count": len(rows),
        "frame_count": frames,
        "frame_accuracy": (
            sum(
                float(row["frame_accuracy"] or 0.0) * int(row["frame_count"])
                for row in rows
            )
            / frames
            if frames
            else None
        ),
        "human_boundary_count": human,
        "predicted_boundary_count": predicted,
        "matched_boundary_count": matched,
        "boundary_recall": matched / human if human else None,
        "boundary_precision": matched / predicted if predicted else None,
        "boundary_mae_frames": _weighted_mean(
            rows,
            "boundary_mae_frames",
            "matched_boundary_count",
        ),
    }


def _aggregate_candidate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    human = sum(int(row["human_candidate_count"]) for row in rows)
    predicted = sum(int(row["predicted_candidate_count"]) for row in rows)
    matched = sum(int(row["matched_candidate_count"]) for row in rows)
    return {
        "record_count": len(rows),
        "human_candidate_count": human,
        "predicted_candidate_count": predicted,
        "matched_candidate_count": matched,
        "candidate_recall": matched / human if human else None,
        "candidate_precision": matched / predicted if predicted else None,
        "missed_candidate_count": human - matched,
        "false_candidate_count": predicted - matched,
        "terminal_mae_frames": _weighted_mean(
            rows,
            "terminal_mae_frames",
            "matched_candidate_count",
        ),
    }


def _binary_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "accuracy": (tp + tn) / (tp + fp + fn + tn)
        if tp + fp + fn + tn
        else None,
    }


def _sum_binary_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return _binary_metrics(
        sum(int(row["true_positive"]) for row in rows),
        sum(int(row["false_positive"]) for row in rows),
        sum(int(row["false_negative"]) for row in rows),
        sum(int(row["true_negative"]) for row in rows),
    )


def _sum_contact_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = _sum_binary_metrics(rows)
    rep_count = sum(int(row["rep_count"]) for row in rows)
    unsure = sum(int(row["unsure"]) for row in rows)
    result.update(
        {
            "rep_count": rep_count,
            "unsure": unsure,
            "unsure_rate": unsure / rep_count if rep_count else None,
            "event_frame_count": sum(int(row["event_frame_count"]) for row in rows),
            "event_frame_mae": _weighted_mean(
                rows,
                "event_frame_mae",
                "event_frame_count",
            ),
        }
    )
    return result


def _weighted_mean(
    rows: Sequence[Mapping[str, Any]],
    value_name: str,
    weight_name: str,
) -> float | None:
    weighted = [
        (float(row[value_name]), int(row[weight_name]))
        for row in rows
        if row.get(value_name) is not None and int(row.get(weight_name, 0)) > 0
    ]
    total = sum(weight for _value, weight in weighted)
    return (
        sum(value * weight for value, weight in weighted) / total
        if total
        else None
    )


def _number(value: object) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return resolved if np.isfinite(resolved) else None


def _markdown(payload: Mapping[str, Any]) -> str:
    phase = payload["phase_boundary_experiment"]
    standing = payload["standing_baseline_experiment"]
    confidence = payload["confidence_calibration_experiment"]
    lines = [
        "# 时序、站立基线、接触代理与置信度内部实验",
        "",
        "所有监督选择均按视频逐一留出；结果仍是内部实验。",
        "",
        "## 阶段边界",
        "",
        "| 指标 | 线性逐帧 | 线性 + 因果 HMM |",
        "|---|---:|---:|",
        f"| frame accuracy | {_fmt(phase['linear_phase_metrics']['frame_accuracy'])} | {_fmt(phase['causal_hmm_phase_metrics']['frame_accuracy'])} |",
        f"| boundary recall | {_fmt(phase['linear_phase_metrics']['boundary_recall'])} | {_fmt(phase['causal_hmm_phase_metrics']['boundary_recall'])} |",
        f"| boundary MAE frames | {_fmt(phase['linear_phase_metrics']['boundary_mae_frames'])} | {_fmt(phase['causal_hmm_phase_metrics']['boundary_mae_frames'])} |",
        "",
        "## 候选边界",
        "",
        "| 指标 | 朴素线性阶段 | 校准 HMM 阶段 |",
        "|---|---:|---:|",
        f"| candidate recall | {_fmt(phase['naive_linear_candidate_metrics']['candidate_recall'])} | {_fmt(phase['calibrated_hmm_candidate_metrics']['candidate_recall'])} |",
        f"| candidate precision | {_fmt(phase['naive_linear_candidate_metrics']['candidate_precision'])} | {_fmt(phase['calibrated_hmm_candidate_metrics']['candidate_precision'])} |",
        f"| terminal MAE frames | {_fmt(phase['naive_linear_candidate_metrics']['terminal_mae_frames'])} | {_fmt(phase['calibrated_hmm_candidate_metrics']['terminal_mae_frames'])} |",
        "",
        "## 每视频站立基线",
        "",
        f"- 固定阈值 F1：{_fmt(standing['fixed_threshold_metrics']['f1'])}",
        f"- 个体基线 F1：{_fmt(standing['personalized_baseline_metrics']['f1'])}",
        "",
        "## 接触代理消融",
        "",
        "| 特征 | Precision | Recall | UNSURE | 事件 MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in payload["contact_proxy_experiment"]["ablations"].items():
        metrics = row["metrics"]
        lines.append(
            f"| {name} | {_fmt(metrics['precision'])} | {_fmt(metrics['recall'])} | "
            f"{_fmt(metrics['unsure_rate'])} | {_fmt(metrics['event_frame_mae'])} |"
        )
    lines.extend(
        [
            "",
            "## 规则置信度校准",
            "",
            f"- 基线 UNSURE：{_fmt(confidence['baseline_metrics']['unsure_rate'])}",
            f"- 校准后 UNSURE：{_fmt(confidence['calibrated_metrics']['unsure_rate'])}",
            f"- 基线状态准确率：{_fmt(confidence['baseline_metrics']['matched_rep_status_accuracy'])}",
            f"- 校准后状态准确率：{_fmt(confidence['calibrated_metrics']['matched_rep_status_accuracy'])}",
            "",
            "置信度校准只允许将有训练折零误报支持的明确 FAIL 从 UNSURE 恢复为 NO_REP，"
            "绝不升格 VALID。",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    return f"{float(value):.4f}" if isinstance(value, float) else str(value)


def _markdown_v2(payload: Mapping[str, Any]) -> str:
    phase = payload["phase_boundary_experiment"]
    standing = payload["standing_baseline_experiment"]
    contact = payload["contact_proxy_experiment"]
    confidence = payload["confidence_calibration_experiment"]
    lines = [
        "# 时序、站立基线、接触代理与置信度内部实验 v2",
        "",
        "所有监督选择均按视频逐一留出；结果仍是小样本、单人工复核内部实验。",
        "",
        "## 阶段边界",
        "",
        "| 指标 | 线性逐帧 | 线性 + 因果 HMM |",
        "|---|---:|---:|",
        f"| frame accuracy | {_fmt(phase['linear_phase_metrics']['frame_accuracy'])} | {_fmt(phase['causal_hmm_phase_metrics']['frame_accuracy'])} |",
        f"| boundary recall | {_fmt(phase['linear_phase_metrics']['boundary_recall'])} | {_fmt(phase['causal_hmm_phase_metrics']['boundary_recall'])} |",
        f"| boundary precision | {_fmt(phase['linear_phase_metrics']['boundary_precision'])} | {_fmt(phase['causal_hmm_phase_metrics']['boundary_precision'])} |",
        f"| boundary MAE frames | {_fmt(phase['linear_phase_metrics']['boundary_mae_frames'])} | {_fmt(phase['causal_hmm_phase_metrics']['boundary_mae_frames'])} |",
        "",
        "## 候选边界",
        "",
        "| 指标 | 朴素线性阶段 | 校准 HMM 阶段 |",
        "|---|---:|---:|",
        f"| candidate recall | {_fmt(phase['naive_linear_candidate_metrics']['candidate_recall'])} | {_fmt(phase['calibrated_hmm_candidate_metrics']['candidate_recall'])} |",
        f"| candidate precision | {_fmt(phase['naive_linear_candidate_metrics']['candidate_precision'])} | {_fmt(phase['calibrated_hmm_candidate_metrics']['candidate_precision'])} |",
        f"| terminal MAE frames | {_fmt(phase['naive_linear_candidate_metrics']['terminal_mae_frames'])} | {_fmt(phase['calibrated_hmm_candidate_metrics']['terminal_mae_frames'])} |",
        "",
        "HMM 结果只建议作为候选去抖、合并和结算侧车，不替代现有规则候选产生器。",
        "",
        "## 每视频站立基线",
        "",
        f"- 固定阈值 F1：{_fmt(standing['fixed_threshold_metrics']['f1'])}",
        f"- 质量门控个体基线 F1：{_fmt(standing['personalized_baseline_metrics']['f1'])}",
        f"- 可靠基线视频：{standing['reliable_baseline_video_count']}",
        f"- 回退固定阈值视频：{standing['fallback_video_count']}",
        "",
        "不合理的膝/髋角、角度或身高不稳定、双侧严重不一致会使个体基线失效并回退固定阈值。",
        "",
        "## 接触代理消融",
        "",
        "| 特征/策略 | Precision | Recall | UNSURE | 事件 MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in contact["ablations"].items():
        metrics = row["metrics"]
        lines.append(
            f"| {name} | {_fmt(metrics['precision'])} | "
            f"{_fmt(metrics['recall'])} | {_fmt(metrics['unsure_rate'])} | "
            f"{_fmt(metrics['event_frame_mae'])} |"
        )
    gate = confidence["safety_gate"]
    lines.extend(
        [
            "",
            "`shadow_safe_fusion` 保留二维接触状态，只允许分割和 3D 修正影子事件锚点；它不代表真实物理接触。",
            "",
            "## 规则置信度校准",
            "",
            f"- 基线 UNSURE：{_fmt(confidence['baseline_metrics']['unsure_rate'])}",
            f"- 原始校准 UNSURE：{_fmt(confidence['raw_calibrated_metrics']['unsure_rate'])}",
            f"- 安全门后 UNSURE：{_fmt(confidence['calibrated_metrics']['unsure_rate'])}",
            f"- 基线状态准确率：{_fmt(confidence['baseline_metrics']['matched_rep_status_accuracy'])}",
            f"- 安全门后状态准确率：{_fmt(confidence['calibrated_metrics']['matched_rep_status_accuracy'])}",
            f"- 误报硬门通过：{gate['passed']}",
            f"- 最终策略：{gate['selected_policy']}",
            "",
            "校准只允许 `UNSURE → NO_REP`，绝不升格 `VALID`；每条规则至少需要 5 个正例、3 段训练视频且训练零误报。若留出汇总的 `VALID/UNSURE → NO_REP` 增加，整项策略回退到基线。",
            "",
        ]
    )
    return "\n".join(lines)


def run_experiment(dataset_root: str | Path) -> tuple[Path, Path]:
    root = Path(dataset_root)
    manifest_payload = _load(root / "manifests" / "phone_records.json")
    fine_payload = _load(
        root / "reviews" / "human_rgb_fine_annotations_v1.json"
    )
    manifests = {
        str(item["record_id"]): dict(item)
        for item in manifest_payload.get("records") or []
        if isinstance(item, Mapping)
    }
    reviews = [
        dict(item)
        for item in fine_payload.get("records") or []
        if isinstance(item, Mapping)
        and bool(item.get("internal_rgb_rule_calibration_eligible"))
        and str(item.get("action")) in PHASE_ORDERS
    ]
    records = [
        _load_frames(root, manifests[review["record_id"]], review)
        for review in reviews
    ]
    payload = {
        "schema_version": 2,
        "artifact_type": "internal_temporal_evidence_lovo_v2",
        "evaluation_scope": "internal_single_reviewer_leave_one_video_out",
        "record_count": len(records),
        "rep_count": sum(len(review.get("reps") or []) for review in reviews),
        "phase_boundary_experiment": _phase_lovo(records),
        "standing_baseline_experiment": _standing_lovo(records),
        "contact_proxy_experiment": _contact_lovo(root, records),
        "confidence_calibration_experiment": _confidence_calibration(
            root,
            manifests,
            reviews,
        ),
        "safety_contract": {
            "models_may_create_valid_status": False,
            "contact_is_proxy_not_physical_force_truth": True,
            "contact_shadow_may_promote_unsure": False,
            "unreliable_standing_baseline_falls_back": True,
            "confidence_calibration_requires_false_no_rep_gate": True,
            "same_video_supervised_tuning_and_evaluation_overlap": False,
            "production_default_changed": False,
        },
        "limitations": [
            "Small single-reviewer internal RGB experiment.",
            "Video-level LOVO reduces direct leakage but is not subject-independent validation.",
            "Segmentation and MediaPipe world landmarks are model outputs, not contact or metric-depth truth.",
            "Contact-negative recordings are sparse; class-coverage gates and UNSURE remain necessary.",
            "Only two of eight Lunge/Wall Ball videos passed the automatic standing-baseline quality gate.",
            "The confidence calibration failed its aggregate false-NO_REP gate and therefore falls back to baseline.",
            "No temporal model is promoted to the product default by this experiment.",
        ],
    }
    json_path = _write(
        root / "reports" / "internal_temporal_evidence_lovo_v2.json",
        payload,
    )
    md_path = root / "reports" / "internal_temporal_evidence_lovo_v2.md"
    md_path.write_text(_markdown_v2(payload), encoding="utf-8")
    return json_path, md_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run internal temporal evidence LOVO experiments."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/hyrox"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.dataset_root
    if not root.is_absolute():
        root = args.project_root.resolve() / root
    json_path, md_path = run_experiment(root)
    payload = _load(json_path)
    print(
        json.dumps(
            {
                "json_report": str(json_path),
                "markdown_report": str(md_path),
                "phase": {
                    "linear": payload["phase_boundary_experiment"][
                        "linear_phase_metrics"
                    ],
                    "hmm": payload["phase_boundary_experiment"][
                        "causal_hmm_phase_metrics"
                    ],
                },
                "candidate": {
                    "naive": payload["phase_boundary_experiment"][
                        "naive_linear_candidate_metrics"
                    ],
                    "calibrated": payload["phase_boundary_experiment"][
                        "calibrated_hmm_candidate_metrics"
                    ],
                },
                "standing": payload["standing_baseline_experiment"],
                "contact": {
                    name: row["metrics"]
                    for name, row in payload["contact_proxy_experiment"][
                        "ablations"
                    ].items()
                },
                "confidence": {
                    "baseline": payload["confidence_calibration_experiment"][
                        "baseline_metrics"
                    ],
                    "raw_calibrated": payload[
                        "confidence_calibration_experiment"
                    ]["raw_calibrated_metrics"],
                    "calibrated": payload[
                        "confidence_calibration_experiment"
                    ]["calibrated_metrics"],
                    "safety_gate": payload[
                        "confidence_calibration_experiment"
                    ]["safety_gate"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
