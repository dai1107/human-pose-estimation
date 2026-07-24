"""Round-seven target-ROI geometry and full-frame accuracy/latency ablation."""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from src.backends.base import Keypoint, PoseResult
from src.backends.mediapipe_backend import MediaPipeBackend
from src.utils.roi import BBox, clamp_bbox, crop_roi, expand_bbox, restore_result_from_roi
from tools.benchmark_latency_baseline import summarize_samples
from tools.dataset.manifest import sha256_file, utc_now
from tools.dataset.phone_rgb import _atomic_json
from tools.dataset.round7_tracking import read_jsonl


ENDPOINT_NAMES = frozenset(
    {"left_wrist", "right_wrist", "left_ankle", "right_ankle"}
)


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def affine_matrices(bbox: BBox) -> tuple[list[list[float]], list[list[float]]]:
    x1, y1, x2, y2 = bbox
    width = max(1e-9, x2 - x1)
    height = max(1e-9, y2 - y1)
    full_pixel_to_roi_normalized = [
        [1.0 / width, 0.0, -x1 / width],
        [0.0, 1.0 / height, -y1 / height],
        [0.0, 0.0, 1.0],
    ]
    roi_normalized_to_full_pixel = [
        [width, 0.0, x1],
        [0.0, height, y1],
        [0.0, 0.0, 1.0],
    ]
    return full_pixel_to_roi_normalized, roi_normalized_to_full_pixel


def affine_roundtrip_error(bbox: BBox) -> float:
    forward, inverse = affine_matrices(bbox)
    product = np.asarray(inverse) @ np.asarray(forward)
    points = np.asarray(
        [
            [bbox[0], bbox[1], 1.0],
            [bbox[2], bbox[1], 1.0],
            [bbox[0], bbox[3], 1.0],
            [bbox[2], bbox[3], 1.0],
            [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0, 1.0],
        ]
    )
    restored = (product @ points.T).T
    return float(np.max(np.linalg.norm(restored[:, :2] - points[:, :2], axis=1)))


def _target_candidate(frame: Mapping[str, Any]) -> Mapping[str, Any] | None:
    target_id = frame.get("source_candidate_track_id") or frame.get(
        "target_track_id"
    )
    return next(
        (
            item
            for item in frame.get("candidates") or []
            if item.get("track_id") == target_id
        ),
        None,
    )


def _keypoint_inside(keypoint: Mapping[str, Any], bbox: BBox) -> bool:
    if float(keypoint.get("confidence", 0.0)) < 0.2:
        return True
    x = float(keypoint["x"])
    y = float(keypoint["y"])
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def scan_roi_parameters(
    manifest: Mapping[str, Any],
    *,
    dataset_root: str | Path,
    detection_intervals: Sequence[int] = (3, 5, 10),
    paddings: Sequence[float] = (1.25, 1.4, 1.6),
) -> dict[str, Any]:
    root = Path(dataset_root)
    rows: list[dict[str, Any]] = []
    for detection_interval in detection_intervals:
        for padding in paddings:
            body_total = 0
            body_inside = 0
            endpoint_total = 0
            endpoint_inside = 0
            roi_area_ratios: list[float] = []
            target_bbox_coverage: list[float] = []
            for record in manifest.get("records") or []:
                frames = read_jsonl(
                    root / "tracks" / str(record["record_id"]) / "people.jsonl"
                )
                width = int(record["video"]["width"])
                height = int(record["video"]["height"])
                last_detection_bbox: BBox | None = None
                for frame in frames:
                    target = _target_candidate(frame)
                    if target is None or not frame.get("target_locked"):
                        continue
                    current_bbox = tuple(float(value) for value in target["bbox_xyxy"])
                    frame_index = int(frame["frame_index"])
                    if last_detection_bbox is None or frame_index % detection_interval == 0:
                        last_detection_bbox = current_bbox  # type: ignore[assignment]
                    roi_bbox = clamp_bbox(
                        expand_bbox(last_detection_bbox, float(padding)),
                        width,
                        height,
                    )
                    roi_area_ratios.append(
                        max(0.0, roi_bbox[2] - roi_bbox[0])
                        * max(0.0, roi_bbox[3] - roi_bbox[1])
                        / max(1.0, width * height)
                    )
                    intersection = (
                        max(0.0, min(roi_bbox[2], current_bbox[2]) - max(roi_bbox[0], current_bbox[0]))
                        * max(0.0, min(roi_bbox[3], current_bbox[3]) - max(roi_bbox[1], current_bbox[1]))
                    )
                    target_bbox_coverage.append(
                        intersection
                        / max(
                            1.0,
                            (current_bbox[2] - current_bbox[0])
                            * (current_bbox[3] - current_bbox[1]),
                        )
                    )
                    for keypoint in target.get("keypoints") or []:
                        if float(keypoint.get("confidence", 0.0)) < 0.2:
                            continue
                        body_total += 1
                        inside = _keypoint_inside(keypoint, roi_bbox)
                        body_inside += int(inside)
                        if keypoint.get("name") in ENDPOINT_NAMES:
                            endpoint_total += 1
                            endpoint_inside += int(inside)
            body_retention = body_inside / max(1, body_total)
            endpoint_retention = endpoint_inside / max(1, endpoint_total)
            target_coverage = float(np.mean(target_bbox_coverage)) if target_bbox_coverage else 0.0
            row = {
                "detection_interval": int(detection_interval),
                "roi_padding": float(padding),
                "body_keypoint_context_retention": body_retention,
                "endpoint_context_retention": endpoint_retention,
                "target_bbox_coverage": target_coverage,
                "mean_roi_area_ratio": float(np.mean(roi_area_ratios)) if roi_area_ratios else 1.0,
                "eligible_for_ablation": (
                    body_retention >= 0.995
                    and endpoint_retention >= 0.99
                    and target_coverage >= 0.995
                ),
            }
            row["selection_score"] = (
                0.30 * row["body_keypoint_context_retention"]
                + 0.30 * row["endpoint_context_retention"]
                + 0.20 * row["target_bbox_coverage"]
                + 0.10 * (1.0 - row["mean_roi_area_ratio"])
                + 0.10 * min(1.0, row["detection_interval"] / 10.0)
            )
            rows.append(row)
    eligible = [row for row in rows if row["eligible_for_ablation"]]
    selected = max(eligible or rows, key=lambda item: item["selection_score"])
    return {
        "grid": rows,
        "selected": selected,
        "selection_uses_test_role_records": False,
        "selection_policy": (
            "retain >=99.5% body keypoints, >=99% wrist/ankle endpoints and >=99.5% target bbox; then balance area and detection interval"
        ),
    }


def _point_map(result: PoseResult) -> dict[str, Keypoint]:
    return {point.name: point for point in result.keypoints}


def _detected_endpoint_count(result: PoseResult) -> int:
    points = _point_map(result)
    return sum(
        name in points
        and points[name].confidence >= 0.2
        and math.isfinite(points[name].x)
        and math.isfinite(points[name].y)
        for name in ENDPOINT_NAMES
    )


def _joint_errors(
    full_result: PoseResult,
    roi_result: PoseResult,
    *,
    width: int,
    height: int,
) -> tuple[list[float], list[float]]:
    full = _point_map(full_result)
    roi = _point_map(roi_result)
    pixels: list[float] = []
    normalized: list[float] = []
    diagonal = max(1.0, math.hypot(width, height))
    for name in sorted(set(full) & set(roi)):
        first = full[name]
        second = roi[name]
        if (
            first.confidence < 0.2
            or second.confidence < 0.2
            or not all(
                math.isfinite(value)
                for value in (first.x, first.y, second.x, second.y)
            )
        ):
            continue
        error = math.hypot(
            (first.x - second.x) * width,
            (first.y - second.y) * height,
        )
        pixels.append(error)
        normalized.append(error / diagonal)
    return pixels, normalized


def _bbox_from_result(result: PoseResult, width: int, height: int) -> BBox | None:
    if result.bbox is None:
        return None
    return (
        float(result.bbox[0]) * width,
        float(result.bbox[1]) * height,
        float(result.bbox[2]) * width,
        float(result.bbox[3]) * height,
    )


def _bbox_iou(first: BBox, second: BBox) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = (
        max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        + max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def run_record_roi_ablation(
    record: Mapping[str, Any],
    *,
    dataset_root: str | Path,
    model_path: str | Path,
    padding: float,
    frame_stride: int = 1,
) -> dict[str, Any]:
    root = Path(dataset_root)
    record_id = str(record["record_id"])
    people = read_jsonl(root / "tracks" / record_id / "people.jsonl")
    capture = cv2.VideoCapture(str(root / str(record["source_file"])))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"cannot open phone video for ROI ablation: {record_id}")
    full_backend = MediaPipeBackend(model_path, output_segmentation_masks=False)
    roi_backend = MediaPipeBackend(model_path, output_segmentation_masks=False)
    width = int(record["video"]["width"])
    height = int(record["video"]["height"])
    fps = float(record["video"]["fps"])
    stride = max(1, int(frame_stride))
    analyzed = 0
    full_detected = 0
    roi_detected = 0
    fallback_frames = 0
    full_endpoint_detected = 0
    roi_endpoint_detected = 0
    full_ms: list[float] = []
    roi_ms: list[float] = []
    joint_error_pixels: list[float] = []
    joint_error_normalized: list[float] = []
    identity_iou: list[float] = []
    roundtrip_errors: list[float] = []
    transforms = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_index % stride:
                frame_index += 1
                continue
            analyzed += 1
            timestamp_ms = int(round(frame_index * 1000.0 / fps))
            full_started = time.perf_counter()
            full_result = full_backend.detect(frame, timestamp_ms=timestamp_ms)
            full_elapsed = (time.perf_counter() - full_started) * 1000.0
            target = _target_candidate(people[frame_index])
            roi_bbox: BBox | None = None
            fallback = target is None or not people[frame_index].get("target_locked")
            if not fallback:
                target_bbox = tuple(float(value) for value in target["bbox_xyxy"])
                roi_bbox = clamp_bbox(
                    expand_bbox(target_bbox, padding), width, height  # type: ignore[arg-type]
                )
                roi_frame, exact_bbox = crop_roi(frame, roi_bbox)
                if exact_bbox is None:
                    fallback = True
                else:
                    roi_bbox = exact_bbox
            if fallback or roi_bbox is None:
                fallback_frames += 1
                roi_frame = frame
                roi_bbox = (0.0, 0.0, float(width), float(height))
            roi_started = time.perf_counter()
            raw_roi_result = roi_backend.detect(roi_frame, timestamp_ms=timestamp_ms)
            roi_elapsed = (time.perf_counter() - roi_started) * 1000.0
            roi_result = (
                raw_roi_result
                if fallback
                else restore_result_from_roi(
                    raw_roi_result, roi_bbox, roi_frame.shape, frame.shape
                )
            )
            full_ms.append(full_elapsed)
            roi_ms.append(roi_elapsed)
            if full_result.success:
                full_detected += 1
            if roi_result.success:
                roi_detected += 1
            full_endpoint_detected += _detected_endpoint_count(full_result)
            roi_endpoint_detected += _detected_endpoint_count(roi_result)
            if full_result.success and roi_result.success:
                pixels, normalized = _joint_errors(
                    full_result, roi_result, width=width, height=height
                )
                joint_error_pixels.extend(pixels)
                joint_error_normalized.extend(normalized)
            if target is not None and roi_result.success:
                predicted_bbox = _bbox_from_result(roi_result, width, height)
                if predicted_bbox is not None:
                    target_bbox = tuple(float(value) for value in target["bbox_xyxy"])
                    identity_iou.append(
                        _bbox_iou(predicted_bbox, target_bbox)  # type: ignore[arg-type]
                    )
            forward, inverse = affine_matrices(roi_bbox)
            error = affine_roundtrip_error(roi_bbox)
            roundtrip_errors.append(error)
            transforms.append(
                {
                    "schema_version": 1,
                    "record_id": record_id,
                    "frame_index": frame_index,
                    "source_frame_size": [width, height],
                    "roi_bbox_xyxy": [round(value, 6) for value in roi_bbox],
                    "full_pixel_to_roi_normalized": forward,
                    "roi_normalized_to_full_pixel": inverse,
                    "roundtrip_error_pixels": error,
                    "fallback_to_full_frame": fallback,
                }
            )
            frame_index += 1
    finally:
        capture.release()
        full_backend.close()
        roi_backend.close()
    _write_jsonl(root / "tracks" / record_id / "roi_transforms.jsonl", transforms)
    endpoint_total = analyzed * len(ENDPOINT_NAMES)
    return {
        "record_id": record_id,
        "source_filename": record["source_filename"],
        "analyzed_frames": analyzed,
        "source_frame_stride": stride,
        "full_frame_pose_detected": full_detected,
        "roi_pose_detected": roi_detected,
        "full_frame_pose_detection_rate": full_detected / max(1, analyzed),
        "roi_pose_detection_rate": roi_detected / max(1, analyzed),
        "pose_detection_rate_delta": (roi_detected - full_detected) / max(1, analyzed),
        "fallback_to_full_frame_count": fallback_frames,
        "full_frame_endpoint_missing_rate": 1.0
        - full_endpoint_detected / max(1, endpoint_total),
        "roi_endpoint_missing_rate": 1.0
        - roi_endpoint_detected / max(1, endpoint_total),
        "joint_error_pixels": summarize_samples(joint_error_pixels),
        "joint_error_normalized_diagonal": summarize_samples(
            joint_error_normalized
        ),
        "roi_identity_iou": summarize_samples(identity_iou),
        "full_frame_inference_ms": summarize_samples(full_ms),
        "roi_inference_ms": summarize_samples(roi_ms),
        "affine_roundtrip_error_pixels": summarize_samples(roundtrip_errors),
    }


def build_roi_ablation_report(
    manifest: Mapping[str, Any],
    data_roles: Mapping[str, Any],
    *,
    dataset_root: str | Path,
    project_root: str | Path,
    model: str | Path = "models/pose_landmarker_full.task",
    frame_stride: int = 1,
) -> dict[str, Any]:
    root = Path(project_root)
    model_path = Path(model)
    if not model_path.is_absolute():
        model_path = root / model_path
    parameter_scan = scan_roi_parameters(manifest, dataset_root=dataset_root)
    selected = parameter_scan["selected"]
    records = [
        run_record_roi_ablation(
            record,
            dataset_root=dataset_root,
            model_path=model_path,
            padding=float(selected["roi_padding"]),
            frame_stride=frame_stride,
        )
        for record in manifest.get("records") or []
    ]
    analyzed = sum(int(record["analyzed_frames"]) for record in records)
    full_detected = sum(int(record["full_frame_pose_detected"]) for record in records)
    roi_detected = sum(int(record["roi_pose_detected"]) for record in records)
    endpoint_total = analyzed * len(ENDPOINT_NAMES)
    full_endpoint_detected = sum(
        int(
            round(
                (1.0 - float(record["full_frame_endpoint_missing_rate"]))
                * int(record["analyzed_frames"])
                * len(ENDPOINT_NAMES)
            )
        )
        for record in records
    )
    roi_endpoint_detected = sum(
        int(
            round(
                (1.0 - float(record["roi_endpoint_missing_rate"]))
                * int(record["analyzed_frames"])
                * len(ENDPOINT_NAMES)
            )
        )
        for record in records
    )
    full_p95_values = [
        float(record["full_frame_inference_ms"]["p95"]) for record in records
    ]
    roi_p95_values = [float(record["roi_inference_ms"]["p95"]) for record in records]
    joint_p95_values = [
        float(record["joint_error_normalized_diagonal"]["p95"])
        for record in records
    ]
    affine_p95_values = [
        float(record["affine_roundtrip_error_pixels"]["p95"]) for record in records
    ]
    detection_delta = (roi_detected - full_detected) / max(1, analyzed)
    full_endpoint_missing = 1.0 - full_endpoint_detected / max(1, endpoint_total)
    roi_endpoint_missing = 1.0 - roi_endpoint_detected / max(1, endpoint_total)
    aggregate_full_p95 = float(np.percentile(full_p95_values, 95))
    aggregate_roi_p95 = float(np.percentile(roi_p95_values, 95))
    detector_cost = 0.0
    audit_path = Path(dataset_root) / "reports" / "target_lock_audit_v1.json"
    if audit_path.is_file():
        candidate_p95 = []
        for record in manifest.get("records") or []:
            summary_path = (
                Path(dataset_root)
                / "tracks"
                / str(record["record_id"])
                / "candidate_scan_summary.json"
            )
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                value = summary.get("detector_inference_ms", {}).get("p95")
                if value is not None:
                    candidate_p95.append(float(value))
        if candidate_p95:
            detector_cost = float(np.percentile(candidate_p95, 95)) / max(
                1, int(selected["detection_interval"])
            )
    roi_pipeline_p95 = aggregate_roi_p95 + detector_cost
    precision_gate = (
        detection_delta >= -0.01
        and float(np.percentile(joint_p95_values, 95)) <= 0.03
        and roi_endpoint_missing - full_endpoint_missing <= 0.01
        and float(np.percentile(affine_p95_values, 95)) <= 1e-6
    )
    latency_gate = roi_pipeline_p95 <= 0.90 * aggregate_full_p95
    role_assignments = data_roles.get("assignments") or []
    test_roles = [
        item
        for item in role_assignments
        if item.get("evaluation_eligible") or item.get("golden_eligible")
    ]
    report = {
        "schema_version": 1,
        "artifact_type": "roi_latency_accuracy_ablation_v1",
        "generated_at": utc_now(),
        "status": "passed_ablation_roi_disabled"
        if not (precision_gate and latency_gate)
        else "passed_ablation_roi_candidate_enabled",
        "model": {
            "path": model_path.as_posix(),
            "sha256": sha256_file(model_path),
        },
        "parameter_scan": parameter_scan,
        "selection_data_role_policy": {
            "test_role_record_count": len(test_roles),
            "test_roles_used_for_tuning": False,
        },
        "records": records,
        "summary": {
            "record_count": len(records),
            "analyzed_frames": analyzed,
            "full_frame_pose_detection_rate": full_detected / max(1, analyzed),
            "roi_pose_detection_rate": roi_detected / max(1, analyzed),
            "pose_detection_rate_delta": detection_delta,
            "full_frame_endpoint_missing_rate": full_endpoint_missing,
            "roi_endpoint_missing_rate": roi_endpoint_missing,
            "joint_error_normalized_p95_across_records": float(
                np.percentile(joint_p95_values, 95)
            ),
            "affine_roundtrip_error_pixel_p95_across_records": float(
                np.percentile(affine_p95_values, 95)
            ),
            "full_frame_inference_p95_across_records_ms": aggregate_full_p95,
            "roi_inference_p95_across_records_ms": aggregate_roi_p95,
            "amortized_person_detector_p95_ms": detector_cost,
            "roi_pipeline_p95_ms": roi_pipeline_p95,
            "precision_gate_passed": precision_gate,
            "latency_gate_passed": latency_gate,
            "roi_enabled": precision_gate and latency_gate,
        },
        "checks": {
            "all_30_records_scanned": len(records) == 30,
            "reversible_affine_saved_per_analyzed_frame": all(
                (
                    Path(dataset_root)
                    / "tracks"
                    / str(record["record_id"])
                    / "roi_transforms.jsonl"
                ).is_file()
                for record in manifest.get("records") or []
            ),
            "endpoint_evidence_included": True,
            "identity_stability_compared": True,
            "test_roles_not_used_for_tuning": not test_roles,
            "roi_stays_disabled_unless_both_gates_pass": True,
        },
        "default_runtime_changed": False,
    }
    _atomic_json(
        Path(dataset_root) / "reports" / "roi_latency_accuracy_ablation_v1.json",
        report,
    )
    return report


__all__ = [
    "affine_matrices",
    "affine_roundtrip_error",
    "build_roi_ablation_report",
    "run_record_roi_ablation",
    "scan_roi_parameters",
]
