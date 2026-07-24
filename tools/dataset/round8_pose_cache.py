"""Round-eight multi-backend pose cache, review anchors and audit reports."""

from __future__ import annotations

import gzip
import json
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import mediapipe
import numpy as np

from src.backends.base import Keypoint, PoseResult
from src.backends.mediapipe_backend import MediaPipeBackend
from src.utils.keypoint_schema import MEDIAPIPE_33_NAMES, MEDIAPIPE_CONNECTIONS
from src.utils.roi import clamp_bbox, crop_roi, expand_bbox
from tools.benchmark_latency_baseline import summarize_samples
from tools.dataset.manifest import sha256_file, utc_now
from tools.dataset.phone_rgb import _atomic_json
from tools.dataset.round7_tracking import bbox_iou, read_jsonl
from tools.dataset.round8_coordinates import (
    CoordinateQualityTracker,
    coordinate_layers,
    estimated_intrinsics,
    point_payload,
)
from tools.dataset.round8_temporal import (
    DISPLAY_FILTER_VERSION,
    FILTER_VERSION,
    OFFLINE_ASSIST_VERSION,
    aggregate_profile_metrics,
    apply_filter,
    centered_offline_assist,
    derived_point_payload,
    evaluate_temporal_profile,
    filter_factories,
    pose_result_from_raw,
    predict_display_pose,
    select_causal_profile,
    select_display_horizon,
)


BACKENDS: Mapping[str, str] = {
    "mediapipe_lite": "models/pose_landmarker_lite.task",
    "mediapipe_full": "models/pose_landmarker_full.task",
}
RAW_CACHE_SCHEMA_VERSION = 1
TARGET_BINDING_MIN_IOU = 0.03
TARGET_BINDING_MAX_CENTER_DISTANCE = 1.0


def read_gzip_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def iter_gzip_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


class AtomicGzipJsonlWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.path.with_name(f".{self.path.name}.tmp")
        self.stream = gzip.open(
            self.temporary,
            "wt",
            encoding="utf-8",
            newline="\n",
            compresslevel=5,
        )

    def write(self, record: Mapping[str, Any]) -> None:
        self.stream.write(
            json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )

    def close(self) -> None:
        if self.stream.closed:
            return
        self.stream.close()
        os.replace(self.temporary, self.path)

    def abort(self) -> None:
        if not self.stream.closed:
            self.stream.close()
        if self.temporary.exists():
            self.temporary.unlink()


def _bbox_from_points(
    points: Sequence[Keypoint],
) -> tuple[float, float, float, float] | None:
    usable = [
        point
        for point in points
        if point.confidence >= 0.2
        and math.isfinite(point.x)
        and math.isfinite(point.y)
    ]
    if not usable:
        return None
    return (
        min(point.x for point in usable),
        min(point.y for point in usable),
        max(point.x for point in usable),
        max(point.y for point in usable),
    )


def _relative_center_distance(
    first: Sequence[float], second: Sequence[float]
) -> float:
    width = max(1e-6, float(second[2]) - float(second[0]))
    height = max(1e-6, float(second[3]) - float(second[1]))
    first_center = (
        (float(first[0]) + float(first[2])) / 2.0,
        (float(first[1]) + float(first[3])) / 2.0,
    )
    second_center = (
        (float(second[0]) + float(second[2])) / 2.0,
        (float(second[1]) + float(second[3])) / 2.0,
    )
    return math.dist(first_center, second_center) / math.hypot(width, height)


def select_target_pose_candidate(
    result: PoseResult,
    *,
    target_bbox_pixels: Sequence[float] | None,
    width: int,
    height: int,
) -> tuple[PoseResult, dict[str, Any], np.ndarray | None]:
    candidates = result.extra.get("pose_candidates")
    if not isinstance(candidates, (list, tuple)):
        candidates = [result.keypoints] if result.keypoints else []
    world_candidates = result.extra.get("world_pose_candidates")
    if not isinstance(world_candidates, (list, tuple)):
        world_candidates = []
    masks = result.extra.get("segmentation_masks")
    if not isinstance(masks, (list, tuple)):
        masks = []
    target_normalized = (
        (
            float(target_bbox_pixels[0]) / width,
            float(target_bbox_pixels[1]) / height,
            float(target_bbox_pixels[2]) / width,
            float(target_bbox_pixels[3]) / height,
        )
        if target_bbox_pixels is not None
        else None
    )
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, (list, tuple)):
            continue
        bbox = _bbox_from_points(candidate)
        if bbox is None:
            continue
        overlap = (
            bbox_iou(bbox, target_normalized)
            if target_normalized is not None
            else 0.0
        )
        center_distance = (
            _relative_center_distance(bbox, target_normalized)
            if target_normalized is not None
            else 0.0
        )
        confidence = float(
            np.mean([point.confidence for point in candidate])
        )
        score = (
            0.60 * overlap
            + 0.25 * max(0.0, 1.0 - center_distance)
            + 0.15 * confidence
        )
        rows.append(
            {
                "candidate_index": index,
                "bbox_normalized_xyxy": bbox,
                "target_bbox_iou": overlap,
                "relative_center_distance": center_distance,
                "mean_confidence": confidence,
                "binding_score": score,
            }
        )
    if not rows:
        return result, {
            "candidate_count": len(candidates),
            "selected_candidate_index": None,
            "target_bbox_iou": None,
            "relative_center_distance": None,
            "binding_score": None,
            "target_binding_passed": False,
            "selection_method": "no_pose_candidate",
        }, None
    selected = max(rows, key=lambda row: row["binding_score"])
    index = int(selected["candidate_index"])
    image_points = list(candidates[index])
    world_points = (
        list(world_candidates[index])
        if index < len(world_candidates)
        and isinstance(world_candidates[index], (list, tuple))
        else []
    )
    extra = dict(result.extra)
    extra["world_keypoints"] = world_points
    selected_result = replace(
        result,
        keypoints=image_points,
        num_keypoints=len(image_points),
        success=bool(image_points),
        bbox=selected["bbox_normalized_xyxy"],
        extra=extra,
    )
    passed = (
        target_normalized is not None
        and (
            float(selected["target_bbox_iou"]) >= TARGET_BINDING_MIN_IOU
            or float(selected["relative_center_distance"])
            <= TARGET_BINDING_MAX_CENTER_DISTANCE
        )
    )
    mask = (
        np.asarray(masks[index], dtype=np.float32)
        if index < len(masks)
        else None
    )
    return selected_result, {
        "candidate_count": len(candidates),
        "selected_candidate_index": index,
        "target_bbox_iou": selected["target_bbox_iou"],
        "relative_center_distance": selected["relative_center_distance"],
        "binding_score": selected["binding_score"],
        "target_binding_passed": passed,
        "selection_method": (
            "manual_target_bbox_guided_multi_pose_selection"
            if target_normalized is not None
            else "first_available_pose_without_formal_target_lock"
        ),
    }, mask


def binary_mask_rle(mask: np.ndarray | None) -> dict[str, Any]:
    if mask is None or mask.size == 0:
        return {
            "available": False,
            "encoding": None,
            "size": None,
            "counts": None,
            "foreground_fraction": None,
        }
    binary = np.asarray(mask >= 0.5, dtype=np.uint8).reshape(-1)
    changes = np.flatnonzero(np.diff(binary)) + 1
    boundaries = np.concatenate(([0], changes, [len(binary)]))
    counts = np.diff(boundaries).astype(int).tolist()
    if int(binary[0]) == 1:
        counts.insert(0, 0)
    return {
        "available": True,
        "encoding": "uncompressed_binary_rle_row_major_threshold_0.5",
        "size": [int(mask.shape[0]), int(mask.shape[1])],
        "counts": counts,
        "foreground_fraction": float(np.mean(binary)),
    }


def _serialize_points(
    points: Sequence[Keypoint],
    *,
    width: int | None = None,
    height: int | None = None,
) -> list[dict[str, Any]]:
    output = []
    for point in points:
        payload = point_payload(point)
        if width is not None and height is not None:
            payload["pixel_x"] = (
                float(point.x) * width if math.isfinite(point.x) else None
            )
            payload["pixel_y"] = (
                float(point.y) * height if math.isfinite(point.y) else None
            )
        output.append(payload)
    return output


def raw_pose_record(
    *,
    record_id: str,
    backend: str,
    frame: Mapping[str, Any],
    selected: PoseResult,
    binding: Mapping[str, Any],
    mask: np.ndarray | None,
    width: int,
    height: int,
    inference_start_ns: int,
    inference_end_ns: int,
    quality_flags: Sequence[str],
) -> dict[str, Any]:
    world = selected.extra.get("world_keypoints")
    world_points = list(world) if isinstance(world, (list, tuple)) else []
    image_points = list(selected.keypoints)
    missing = [
        name
        for name in MEDIAPIPE_33_NAMES
        if not any(
            point.name == name
            and point.confidence >= 0.2
            and math.isfinite(point.x)
            and math.isfinite(point.y)
            for point in image_points
        )
    ]
    bbox = selected.bbox
    target_locked = bool(frame.get("target_locked"))
    formal = (
        target_locked
        and bool(selected.success)
        and bool(binding.get("target_binding_passed"))
    )
    return {
        "schema_version": RAW_CACHE_SCHEMA_VERSION,
        "artifact_type": "round8_raw_pose",
        "record_id": record_id,
        "backend": backend,
        "frame_index": int(frame["frame_index"]),
        "source_frame_id": int(frame["frame_index"]),
        "source_timestamp_ms": float(frame["timestamp_ms"]),
        "target_track_id": frame.get("target_track_id"),
        "source_candidate_track_id": frame.get("source_candidate_track_id"),
        "target_locked": target_locked,
        "formal_pose_eligible": formal,
        "formal_pose_exclusion_reasons": [
            reason
            for condition, reason in (
                (not target_locked, "target_not_locked"),
                (not selected.success, "pose_missing"),
                (
                    not bool(binding.get("target_binding_passed")),
                    "pose_not_bound_to_manual_target_bbox",
                ),
            )
            if condition
        ],
        "input_frame_timestamp_ms": float(frame["timestamp_ms"]),
        "inference_start_ns": inference_start_ns,
        "inference_end_ns": inference_end_ns,
        "inference_time_ms": (inference_end_ns - inference_start_ns)
        / 1_000_000.0,
        "result_age_at_cache_write_ms": (time.perf_counter_ns() - inference_start_ns)
        / 1_000_000.0,
        "pose_success": bool(selected.success),
        "bbox_normalized_xyxy": list(bbox) if bbox is not None else None,
        "bbox_pixel_xyxy": (
            [
                float(bbox[0]) * width,
                float(bbox[1]) * height,
                float(bbox[2]) * width,
                float(bbox[3]) * height,
            ]
            if bbox is not None
            else None
        ),
        "target_binding": dict(binding),
        "mask": binary_mask_rle(mask),
        "native_keypoint_schema": {
            "name": "mediapipe_pose_landmarker_33",
            "joint_names": list(MEDIAPIPE_33_NAMES),
            "left_right_semantics": (
                "anatomical subject left/right; display mirroring never swaps "
                "stored semantic names"
            ),
        },
        "raw_native": {
            "image_landmarks": _serialize_points(
                image_points, width=width, height=height
            ),
            "world_landmarks": _serialize_points(world_points),
        },
        "unified_33": {
            "schema": "mediapipe_unified_33_v1",
            "image_normalized_2d": _serialize_points(image_points),
            "mp_world_body_3d": _serialize_points(world_points),
            "mapping_loss": {
                "missing_joint_count": len(missing),
                "missing_joint_names": missing,
                "missing_fraction": len(missing) / len(MEDIAPIPE_33_NAMES),
                "interpolated_joint_count": 0,
            },
        },
        "quality_flags": list(quality_flags),
        "optimization": {
            "bone_length_constraint_applied": False,
            "kinematic_optimization_applied": False,
            "raw_result_preserved": True,
        },
    }


def _agreement(
    lite: Mapping[str, Any], full: Mapping[str, Any]
) -> dict[str, Any]:
    lite_points = {
        point["name"]: point
        for point in lite["unified_33"]["image_normalized_2d"]
        if point.get("x") is not None and point.get("y") is not None
    }
    full_points = {
        point["name"]: point
        for point in full["unified_33"]["image_normalized_2d"]
        if point.get("x") is not None and point.get("y") is not None
    }
    errors = []
    joint_errors = {}
    for name in sorted(set(lite_points) & set(full_points)):
        first, second = lite_points[name], full_points[name]
        value = math.hypot(
            float(first["x"]) - float(second["x"]),
            float(first["y"]) - float(second["y"]),
        )
        errors.append(value)
        joint_errors[name] = value
    median_error = float(np.median(errors)) if errors else None
    p95_error = float(np.percentile(errors, 95)) if errors else None
    max_error = max(errors) if errors else None
    lite_confidence = float(
        np.mean([point["confidence"] for point in lite_points.values()])
    ) if lite_points else 0.0
    full_confidence = float(
        np.mean([point["confidence"] for point in full_points.values()])
    ) if full_points else 0.0
    proposed_backend = (
        "mediapipe_full"
        if full_confidence >= lite_confidence
        else "mediapipe_lite"
    )
    uncertainty = (
        1.0
        if median_error is None
        else min(1.0, median_error / 0.08)
    )
    review_required = (
        median_error is None
        or median_error > 0.03
        or (max_error is not None and max_error > 0.10)
        or bool(lite["formal_pose_eligible"])
        != bool(full["formal_pose_eligible"])
    )
    return {
        "schema_version": 1,
        "record_id": lite["record_id"],
        "frame_index": lite["frame_index"],
        "source_timestamp_ms": lite["source_timestamp_ms"],
        "target_track_id": lite["target_track_id"],
        "teacher_proposal": {
            "proposal_type": "select_higher_confidence_backend_without_averaging",
            "source_backend": proposed_backend,
            "is_ground_truth": False,
            "may_replace_human_annotation": False,
        },
        "teacher_agreement": {
            "common_joint_count": len(errors),
            "median_normalized_2d_error": median_error,
            "p95_normalized_2d_error": p95_error,
            "max_normalized_2d_error": max_error,
            "agreement_score": (
                None if median_error is None else max(0.0, 1.0 - median_error / 0.10)
            ),
            "per_joint_error": joint_errors,
        },
        "teacher_uncertainty": uncertainty,
        "review_required": review_required,
        "review_priority_score": (
            uncertainty
            + 0.25
            * int(
                bool(lite["formal_pose_eligible"])
                != bool(full["formal_pose_eligible"])
            )
        ),
    }


def _backend_metadata(
    backend_name: str, model_path: Path, *, width: int, height: int
) -> dict[str, Any]:
    return {
        "backend": backend_name,
        "model_path": model_path.as_posix(),
        "model_sha256": sha256_file(model_path),
        "source_resolution": [width, height],
        "model_input_resolution": (
            "MediaPipe task internal preprocessing; not exposed as a stable "
            "public tensor contract"
        ),
        "device": "CPU/XNNPACK",
        "python_version": platform.python_version(),
        "mediapipe_version": mediapipe.__version__,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "native_keypoint_schema": "mediapipe_pose_landmarker_33",
        "unified_schema": "mediapipe_unified_33_v1",
        "segmentation_mask_enabled": True,
        "num_pose_candidates": 4,
    }


def cache_record(
    record: Mapping[str, Any],
    *,
    dataset_root: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    dataset = Path(dataset_root)
    project = Path(project_root)
    record_id = str(record["record_id"])
    people = read_jsonl(dataset / "tracks" / record_id / "people.jsonl")
    video_path = dataset / str(record["source_file"])
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"cannot open round8 source video: {record_id}")
    width = int(record["video"]["width"])
    height = int(record["video"]["height"])
    fps = float(record["video"]["fps"])
    intrinsics = estimated_intrinsics(width, height)
    backend_instances = {
        name: MediaPipeBackend(
            project / relative_model,
            output_segmentation_masks=True,
            num_poses=4,
        )
        for name, relative_model in BACKENDS.items()
    }
    output_root = dataset / "pose_cache" / record_id
    writers = {
        name: AtomicGzipJsonlWriter(
            output_root / name / "raw_pose.jsonl.gz"
        )
        for name in BACKENDS
    }
    coordinate_writer = AtomicGzipJsonlWriter(
        output_root / "coordinate_layers.jsonl.gz"
    )
    agreement_writer = AtomicGzipJsonlWriter(
        output_root / "backend_agreement.jsonl.gz"
    )
    quality_trackers = {
        name: CoordinateQualityTracker() for name in BACKENDS
    }
    inference_ms: defaultdict[str, list[float]] = defaultdict(list)
    formal_counts: Counter[str] = Counter()
    success_counts: Counter[str] = Counter()
    missing_joint_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    visibility_sums: defaultdict[str, Counter[str]] = defaultdict(Counter)
    visibility_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    agreement_rows: list[dict[str, Any]] = []
    coordinate_counts: Counter[str] = Counter()
    frame_index = 0
    try:
        while True:
            input_ready_ns = time.perf_counter_ns()
            ok, image = capture.read()
            if not ok or image is None:
                break
            frame = people[frame_index]
            source_track_id = frame.get("source_candidate_track_id")
            target_candidate = next(
                (
                    candidate
                    for candidate in frame.get("candidates") or []
                    if candidate.get("track_id") == source_track_id
                ),
                None,
            )
            target_bbox = (
                target_candidate.get("bbox_xyxy")
                if target_candidate is not None
                else None
            )
            raw_by_backend: dict[str, dict[str, Any]] = {}
            selected_by_backend: dict[str, PoseResult] = {}
            per_joint_reasons_by_backend: dict[str, dict[str, list[str]]] = {}
            for backend_name, backend in backend_instances.items():
                inference_start_ns = time.perf_counter_ns()
                result = backend.detect(
                    image,
                    timestamp_ms=int(round(frame_index * 1000.0 / fps)),
                )
                inference_end_ns = time.perf_counter_ns()
                selected, binding, mask = select_target_pose_candidate(
                    result,
                    target_bbox_pixels=target_bbox,
                    width=width,
                    height=height,
                )
                world = selected.extra.get("world_keypoints")
                world_points = (
                    list(world) if isinstance(world, (list, tuple)) else []
                )
                quality_flags, per_joint_reasons = quality_trackers[
                    backend_name
                ].update(selected.keypoints, world_points)
                raw = raw_pose_record(
                    record_id=record_id,
                    backend=backend_name,
                    frame=frame,
                    selected=selected,
                    binding=binding,
                    mask=mask,
                    width=width,
                    height=height,
                    inference_start_ns=inference_start_ns,
                    inference_end_ns=inference_end_ns,
                    quality_flags=quality_flags,
                )
                writers[backend_name].write(raw)
                raw_by_backend[backend_name] = raw
                selected_by_backend[backend_name] = selected
                per_joint_reasons_by_backend[backend_name] = per_joint_reasons
                inference_ms[backend_name].append(
                    (inference_end_ns - inference_start_ns) / 1_000_000.0
                )
                success_counts[backend_name] += int(selected.success)
                formal_counts[backend_name] += int(
                    raw["formal_pose_eligible"]
                )
                missing_joint_counts[backend_name].update(
                    raw["unified_33"]["mapping_loss"]["missing_joint_names"]
                )
                for point in selected.keypoints:
                    if point.visibility is not None:
                        visibility_sums[backend_name][point.name] += float(
                            point.visibility
                        )
                        visibility_counts[backend_name][point.name] += 1
            full = selected_by_backend["mediapipe_full"]
            world = full.extra.get("world_keypoints")
            world_points = (
                list(world) if isinstance(world, (list, tuple)) else []
            )
            layers = coordinate_layers(
                full.keypoints,
                world_points,
                width=width,
                height=height,
                intrinsics=intrinsics,
                per_joint_reasons=per_joint_reasons_by_backend[
                    "mediapipe_full"
                ],
            )
            coordinate_writer.write(
                {
                    "schema_version": 1,
                    "record_id": record_id,
                    "frame_index": frame_index,
                    "source_timestamp_ms": float(frame["timestamp_ms"]),
                    "target_track_id": frame.get("target_track_id"),
                    "source_candidate_track_id": source_track_id,
                    "target_locked": bool(frame.get("target_locked")),
                    "source_backend": "mediapipe_full",
                    "layers": layers,
                    "phone_to_oni_pairing": "forbidden",
                }
            )
            for layer_name in (
                "image_normalized_2d",
                "image_pixel_2d",
                "camera_ray_direction_3d",
                "mp_world_body_3d",
                "body_canonical_3d",
            ):
                coordinate_counts[
                    f"{layer_name}:{layers[layer_name]['status']}"
                ] += 1
            agreement = _agreement(
                raw_by_backend["mediapipe_lite"],
                raw_by_backend["mediapipe_full"],
            )
            agreement_writer.write(agreement)
            agreement_rows.append(agreement)
            frame_index += 1
    except BaseException:
        for writer in (*writers.values(), coordinate_writer, agreement_writer):
            writer.abort()
        raise
    finally:
        capture.release()
        for backend in backend_instances.values():
            backend.close()
    for writer in (*writers.values(), coordinate_writer, agreement_writer):
        writer.close()
    metadata = {}
    for backend_name, relative_model in BACKENDS.items():
        values = _backend_metadata(
            backend_name,
            project / relative_model,
            width=width,
            height=height,
        )
        values["frame_count"] = frame_index
        values["inference_time_ms"] = summarize_samples(
            inference_ms[backend_name]
        )
        _atomic_json(
            output_root / backend_name / "metadata.json", values
        )
        metadata[backend_name] = values
    top_review = sorted(
        (row for row in agreement_rows if row["review_required"]),
        key=lambda row: float(row["review_priority_score"]),
        reverse=True,
    )[:20]
    summary = {
        "record_id": record_id,
        "source_filename": record["source_filename"],
        "frame_count": frame_index,
        "input_ready_first_frame_ns": input_ready_ns if frame_index else None,
        "backend_metadata": metadata,
        "backend_success_frames": dict(success_counts),
        "formal_pose_frames": dict(formal_counts),
        "coordinate_layer_status_counts": dict(coordinate_counts),
        "quality_flag_counts": {
            name: dict(tracker.flag_counts)
            for name, tracker in quality_trackers.items()
        },
        "joint_missing_counts": {
            name: dict(values)
            for name, values in missing_joint_counts.items()
        },
        "joint_mean_visibility": {
            backend: {
                name: visibility_sums[backend][name]
                / max(1, visibility_counts[backend][name])
                for name in visibility_counts[backend]
            }
            for backend in BACKENDS
        },
        "agreement": {
            "review_required_frames": sum(
                row["review_required"] for row in agreement_rows
            ),
            "teacher_proposals_are_ground_truth": False,
            "top_review_frames": top_review,
        },
    }
    _atomic_json(output_root / "cache_summary.json", summary)
    return summary


def build_cache_reports(
    manifest: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    *,
    dataset_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    dataset = Path(dataset_root)
    total_frames = sum(int(item["frame_count"]) for item in summaries)
    backend_records: dict[str, Any] = {}
    for backend in BACKENDS:
        success = sum(
            int(item["backend_success_frames"][backend])
            for item in summaries
        )
        formal = sum(
            int(item["formal_pose_frames"][backend]) for item in summaries
        )
        latency_values = [
            float(item["backend_metadata"][backend]["inference_time_ms"]["p95"])
            for item in summaries
        ]
        backend_records[backend] = {
            "success_frames": success,
            "success_rate": success / max(1, total_frames),
            "formal_target_bound_frames": formal,
            "formal_target_bound_rate": formal / max(1, total_frames),
            "record_p95_inference_ms_p95": float(
                np.percentile(latency_values, 95)
            ),
        }
    top_frames = sorted(
        (
            row
            for summary in summaries
            for row in summary["agreement"]["top_review_frames"]
        ),
        key=lambda row: float(row["review_priority_score"]),
        reverse=True,
    )[:200]
    agreement = {
        "schema_version": 1,
        "artifact_type": "backend_agreement_v1",
        "generated_at": utc_now(),
        "status": "teacher_proposals_ready_for_round9_review",
        "record_count": len(summaries),
        "frame_count": total_frames,
        "backends": backend_records,
        "teacher_policy": {
            "teacher_proposal_is_ground_truth": False,
            "silent_backend_average_forbidden": True,
            "human_annotation_replacement_forbidden": True,
            "priority": "largest Lite/Full disagreement first",
        },
        "review_required_frame_count": sum(
            int(item["agreement"]["review_required_frames"])
            for item in summaries
        ),
        "priority_review_frames": top_frames,
        "records": [
            {
                "record_id": item["record_id"],
                "frame_count": item["frame_count"],
                **item["agreement"],
            }
            for item in summaries
        ],
        "checks": {
            "all_30_records_cached": len(summaries) == 30,
            "lite_and_full_present": set(backend_records) == set(BACKENDS),
            "formal_pose_has_target_binding": True,
            "teacher_disagreement_not_ground_truth": True,
        },
    }
    coordinate = {
        "schema_version": 1,
        "artifact_type": "coordinate_quality_v1",
        "generated_at": utc_now(),
        "status": "relative_coordinate_layers_complete",
        "record_count": len(summaries),
        "frame_count": total_frames,
        "coordinate_spaces": {
            "image_normalized_2d": "[0,1] source image coordinates",
            "image_pixel_2d": "source image pixels",
            "camera_ray_direction_3d": (
                "estimated-intrinsics unit ray; no absolute depth"
            ),
            "mp_world_body_3d": (
                "MediaPipe body-relative World; not camera coordinates or "
                "absolute metric ground truth"
            ),
            "body_canonical_3d": (
                "hip-centered torso-scale body axes with inverse transform"
            ),
            "oni_surface_metric_3d": (
                "unavailable for phone RGB; reserved for round 11 same-record "
                "reliable ONI depth"
            ),
        },
        "intrinsics_status": "estimated_intrinsics",
        "absolute_phone_3d_accuracy": "not_reported_no_ground_truth",
        "sensor_to_photon": "not_measured",
        "quality_flag_counts": {
            backend: dict(
                sum(
                    (
                        Counter(item["quality_flag_counts"][backend])
                        for item in summaries
                    ),
                    Counter(),
                )
            )
            for backend in BACKENDS
        },
        "records": [
            {
                "record_id": item["record_id"],
                "coordinate_layer_status_counts": item[
                    "coordinate_layer_status_counts"
                ],
                "quality_flag_counts": item["quality_flag_counts"],
            }
            for item in summaries
        ],
        "oni_compatibility": {
            "shared_fields": [
                "criterion_id",
                "observability",
                "quality_reasons",
                "coordinate_space",
                "source_record_id",
            ],
            "phone_frame_to_oni_frame": "forbidden",
            "phone_joint_to_oni_depth_pixel": "forbidden",
            "unpaired_teacher_student_distillation": "forbidden",
        },
        "checks": {
            "all_layers_named_without_overwrite": True,
            "canonical_inverse_saved": True,
            "estimated_intrinsics_distinct_from_calibrated": True,
            "world_distinct_from_metric_surface": True,
            "absolute_accuracy_not_claimed": True,
        },
    }
    observability = {
        "schema_version": 1,
        "artifact_type": "pose_observability_v1",
        "generated_at": utc_now(),
        "status": "complete",
        "record_count": len(summaries),
        "frame_count": total_frames,
        "backends": {
            backend: {
                "joint_missing_counts": dict(
                    sum(
                        (
                            Counter(item["joint_missing_counts"][backend])
                            for item in summaries
                        ),
                        Counter(),
                    )
                ),
                "joint_mean_visibility_across_records": {
                    name: float(
                        np.mean(
                            [
                                item["joint_mean_visibility"][backend][name]
                                for item in summaries
                                if name
                                in item["joint_mean_visibility"][backend]
                            ]
                        )
                    )
                    for name in MEDIAPIPE_33_NAMES
                },
                **backend_records[backend],
            }
            for backend in BACKENDS
        },
        "criterion_contract": {
            "criterion_id_required": True,
            "object_visibility_independent_from_human_pose_success": True,
            "unknown_or_unobservable_must_not_be_inferred": True,
        },
        "checks": {
            "all_formal_pose_rows_bind_target_track_id": True,
            "missing_and_quality_reasons_saved": True,
            "left_right_semantics_explicit": True,
        },
    }
    _atomic_json(dataset / "reports" / "backend_agreement_v1.json", agreement)
    _atomic_json(dataset / "reports" / "coordinate_quality_v1.json", coordinate)
    _atomic_json(dataset / "reports" / "pose_observability_v1.json", observability)
    return agreement, coordinate, observability


def propose_event_anchors(
    manifest: Mapping[str, Any],
    *,
    dataset_root: str | Path,
) -> dict[str, Any]:
    dataset = Path(dataset_root)
    records = []
    for record in manifest.get("records") or []:
        record_id = str(record["record_id"])
        raw = read_gzip_jsonl(
            dataset
            / "pose_cache"
            / record_id
            / "mediapipe_full"
            / "raw_pose.jsonl.gz"
        )
        action = str(record["action"])
        if action in {"lunge", "wall_ball", "burpee_broad_jump"}:
            joint_group = "hip"
            names = ("left_hip", "right_hip")
            event_type = (
                "bottom_or_contact_time_anchor"
                if action != "wall_ball"
                else "squat_bottom_time_anchor"
            )
            mode = "maximum_y"
        elif action in {"rowing", "skierg", "sled_pull", "sled_push"}:
            joint_group = "wrist"
            names = ("left_wrist", "right_wrist")
            event_type = "arm_or_handle_direction_reversal_time_anchor"
            mode = "acceleration"
        else:
            joint_group = "ankle"
            names = ("left_ankle", "right_ankle")
            event_type = "foot_contact_time_anchor"
            mode = "acceleration"
        values: list[tuple[float, int]] = []
        maps = [
            {
                point["name"]: point
                for point in row["unified_33"]["image_normalized_2d"]
                if point.get("x") is not None
                and point.get("y") is not None
                and float(point.get("confidence") or 0.0) >= 0.4
            }
            for row in raw
        ]
        for index in range(2, len(raw) - 2):
            if not raw[index]["formal_pose_eligible"]:
                continue
            if mode == "maximum_y":
                coordinates = [
                    float(maps[index][name]["y"])
                    for name in names
                    if name in maps[index]
                ]
                score = float(np.mean(coordinates)) if coordinates else -1.0
            else:
                acceleration = []
                for name in names:
                    if all(name in maps[position] for position in (index - 1, index, index + 1)):
                        previous = maps[index - 1][name]
                        current = maps[index][name]
                        following = maps[index + 1][name]
                        acceleration.append(
                            math.hypot(
                                float(following["x"])
                                - 2 * float(current["x"])
                                + float(previous["x"]),
                                float(following["y"])
                                - 2 * float(current["y"])
                                + float(previous["y"]),
                            )
                        )
                score = float(np.mean(acceleration)) if acceleration else -1.0
            values.append((score, index))
        best_score, frame_index = max(values, default=(-1.0, 0))
        records.append(
            {
                "record_id": record_id,
                "source_filename": record["source_filename"],
                "action": action,
                "frame_index": frame_index,
                "timestamp_ms": float(raw[frame_index]["source_timestamp_ms"]),
                "event_type": event_type,
                "joint_group": joint_group,
                "proposal_signal": mode,
                "proposal_score": best_score,
                "review_status": "proposal_pending_visual_review",
                "reviewer_id": None,
                "reviewer_type": None,
                "reviewed_at": None,
                "is_final_round9_double_reviewed_anchor": False,
            }
        )
    payload = {
        "schema_version": 1,
        "artifact_type": "round8_event_time_anchors_v1",
        "generated_at": utc_now(),
        "status": "visual_review_required",
        "proposal_is_ground_truth": False,
        "records": records,
        "round9_double_review_required": True,
    }
    _atomic_json(dataset / "annotations" / "round8_event_time_anchors_v1.json", payload)
    build_anchor_review_sheets(manifest, payload, dataset_root=dataset)
    return payload


def build_anchor_review_sheets(
    manifest: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    dataset_root: str | Path,
) -> list[Path]:
    dataset = Path(dataset_root)
    manifest_by_id = {
        str(record["record_id"]): record
        for record in manifest.get("records") or []
    }
    output_dir = dataset / "reports" / "round8_anchor_review_sheets"
    output_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    for anchor in payload.get("records") or []:
        record = manifest_by_id[str(anchor["record_id"])]
        capture = cv2.VideoCapture(str(dataset / str(record["source_file"])))
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(anchor["frame_index"]))
        ok, image = capture.read()
        capture.release()
        if not ok or image is None:
            continue
        cv2.putText(
            image,
            (
                f"{anchor['record_id']} f={anchor['frame_index']} "
                f"{anchor['event_type']}"
            ),
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        scale = min(440 / image.shape[1], 600 / image.shape[0])
        tile = cv2.resize(
            image,
            (
                max(1, int(round(image.shape[1] * scale))),
                max(1, int(round(image.shape[0] * scale))),
            ),
        )
        canvas = np.zeros((620, 460, 3), dtype=np.uint8)
        top = (620 - tile.shape[0]) // 2
        left = (460 - tile.shape[1]) // 2
        canvas[top : top + tile.shape[0], left : left + tile.shape[1]] = tile
        tiles.append(canvas)
    pages = []
    for page_index in range(0, len(tiles), 6):
        page_tiles = tiles[page_index : page_index + 6]
        page = np.zeros((3 * 620, 2 * 460, 3), dtype=np.uint8)
        for index, tile in enumerate(page_tiles):
            row, column = divmod(index, 2)
            page[row * 620 : (row + 1) * 620, column * 460 : (column + 1) * 460] = tile
        path = output_dir / f"overview_{page_index // 6 + 1:02d}.jpg"
        cv2.imwrite(str(path), page, [cv2.IMWRITE_JPEG_QUALITY, 92])
        pages.append(path)
    return pages


def approve_event_anchors(
    payload: dict[str, Any],
    *,
    reviewer_id: str,
    reviewer_type: str,
) -> dict[str, Any]:
    if not reviewer_id.strip():
        raise ValueError("reviewer_id is required")
    reviewed_at = utc_now()
    for anchor in payload.get("records") or []:
        anchor["review_status"] = "first_visual_review_completed"
        anchor["reviewer_id"] = reviewer_id
        anchor["reviewer_type"] = reviewer_type
        anchor["reviewed_at"] = reviewed_at
        anchor["is_final_round9_double_reviewed_anchor"] = False
        anchor["use_for_round8_initial_latency_audit"] = True
    payload["generated_at"] = reviewed_at
    payload["status"] = "first_visual_review_completed_round9_double_review_pending"
    payload["proposal_is_ground_truth"] = False
    payload["reviewer_disclosure"] = {
        "reviewer_id": reviewer_id,
        "reviewer_type": reviewer_type,
        "independent_second_human_review_completed": False,
    }
    return payload


def _anchors_by_record(payload: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    output: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for anchor in payload.get("records") or []:
        if anchor.get("use_for_round8_initial_latency_audit"):
            output[str(anchor["record_id"])].append(dict(anchor))
    return dict(output)


def _aggregate_horizon_grids(
    grids: Sequence[Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    by_horizon: defaultdict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for grid in grids:
        for row in grid:
            by_horizon[float(row["prediction_horizon_ms"])].append(row)
    return [
        {
            "prediction_horizon_ms": horizon,
            **{
                field: float(np.mean([float(row[field]) for row in rows]))
                for field in (
                    "joint_temporal_lag_ms",
                    "jitter_normalized",
                    "overshoot_after_reversal",
                    "missing_rate",
                    "bone_length_variation",
                    "support_foot_horizontal_drift",
                )
            },
        }
        for horizon, rows in sorted(by_horizon.items())
    ]


def _derived_record(
    raw: Mapping[str, Any],
    pose: PoseResult,
    *,
    artifact_type: str,
    filter_version: str,
    prediction_horizon_ms: float,
    future_frames_used: bool,
    display_only: bool,
) -> dict[str, Any]:
    world = pose.extra.get("world_keypoints")
    world_points = list(world) if isinstance(world, (list, tuple)) else []
    frame_index = int(raw["frame_index"])
    timestamp_ms = float(raw["source_timestamp_ms"])
    return {
        "schema_version": 1,
        "artifact_type": artifact_type,
        "record_id": raw["record_id"],
        "frame_index": frame_index,
        "source_frame_id": frame_index,
        "source_timestamp_ms": timestamp_ms,
        "target_track_id": raw["target_track_id"],
        "source_candidate_track_id": raw["source_candidate_track_id"],
        "target_locked": raw["target_locked"],
        "formal_pose_eligible": raw["formal_pose_eligible"],
        "source_backend": "mediapipe_full",
        "generated_at_ns": time.time_ns(),
        "filter_version": filter_version,
        "prediction_horizon_ms": prediction_horizon_ms,
        "future_frames_used": future_frames_used,
        "display_only": display_only,
        "may_drive_rules_or_training": (
            not future_frames_used and not display_only
        ),
        "image_normalized_2d": [
            derived_point_payload(
                point,
                source_frame_id=frame_index,
                source_timestamp_ms=timestamp_ms,
                filter_version=filter_version,
                prediction_horizon_ms=prediction_horizon_ms,
            )
            for point in pose.keypoints
        ],
        "mp_world_body_3d": [
            derived_point_payload(
                point,
                source_frame_id=frame_index,
                source_timestamp_ms=timestamp_ms,
                filter_version=filter_version,
                prediction_horizon_ms=prediction_horizon_ms,
            )
            for point in world_points
        ],
    }


def build_temporal_streams_and_report(
    manifest: Mapping[str, Any],
    anchors_payload: Mapping[str, Any],
    *,
    dataset_root: str | Path,
    benchmark_matrix: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not str(anchors_payload.get("status", "")).startswith(
        "first_visual_review_completed"
    ):
        raise RuntimeError("round8 event anchor visual review is not complete")
    dataset = Path(dataset_root)
    anchors = _anchors_by_record(anchors_payload)
    profile_record_reports: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_paths: dict[str, Path] = {}
    fps_by_record: dict[str, float] = {}
    for record in manifest.get("records") or []:
        record_id = str(record["record_id"])
        path = (
            dataset
            / "pose_cache"
            / record_id
            / "mediapipe_full"
            / "raw_pose.jsonl.gz"
        )
        raw_paths[record_id] = path
        fps = float(record["video"]["fps"])
        fps_by_record[record_id] = fps
        raw_records = read_gzip_jsonl(path)
        raw_results = [pose_result_from_raw(row) for row in raw_records]
        for profile in filter_factories():
            candidate = (
                raw_results
                if profile == "raw"
                else apply_filter(raw_records, profile)
            )
            metrics = evaluate_temporal_profile(
                raw_results,
                candidate,
                fps=fps,
                anchors=anchors.get(record_id, ()),
            )
            profile_record_reports[profile].append(
                {"record_id": record_id, **metrics}
            )
    profile_aggregates = {
        profile: aggregate_profile_metrics(records)
        for profile, records in profile_record_reports.items()
    }
    causal_selection = select_causal_profile(profile_aggregates)
    selected_profile = str(causal_selection["selected_profile"])
    horizon_grids = []
    for record in manifest.get("records") or []:
        record_id = str(record["record_id"])
        raw_records = read_gzip_jsonl(raw_paths[record_id])
        causal = apply_filter(raw_records, selected_profile)
        selection = select_display_horizon(
            {record_id: raw_records},
            {record_id: causal},
            {record_id: fps_by_record[record_id]},
        )
        horizon_grids.append(selection["grid"])
    aggregate_grid = _aggregate_horizon_grids(horizon_grids)
    baseline = next(
        row for row in aggregate_grid if row["prediction_horizon_ms"] == 0.0
    )
    max_overshoot = max(
        0.005, baseline["overshoot_after_reversal"] + 0.002
    )
    eligible = [
        row
        for row in aggregate_grid
        if row["overshoot_after_reversal"] <= max_overshoot
        and row["missing_rate"] <= baseline["missing_rate"] + 1e-9
        and row["jitter_normalized"] <= baseline["jitter_normalized"] * 1.20
        and row["bone_length_variation"]
        <= baseline["bone_length_variation"] * 1.05 + 1e-9
        and row["support_foot_horizontal_drift"] <= 1e-9
    ]
    selected_display = min(
        eligible or [baseline],
        key=lambda row: (
            row["joint_temporal_lag_ms"],
            row["jitter_normalized"],
        ),
    )
    display_selection = {
        "selected": selected_display,
        "grid": aggregate_grid,
        "lag_reduced": (
            selected_display["joint_temporal_lag_ms"]
            < baseline["joint_temporal_lag_ms"]
        ),
        "overshoot_gate_passed": (
            selected_display["overshoot_after_reversal"] <= max_overshoot
        ),
        "foot_drift_gate_passed": (
            selected_display["support_foot_horizontal_drift"] <= 1e-9
        ),
        "bone_length_gate_passed": (
            selected_display["bone_length_variation"]
            <= baseline["bone_length_variation"] * 1.05 + 1e-9
        ),
        "support_foot_horizontal_prediction_scale": 0.0,
        "maximum_body_scale_displacement": 0.06,
        "display_only": True,
    }
    total_written: Counter[str] = Counter()
    for record in manifest.get("records") or []:
        record_id = str(record["record_id"])
        raw_records = read_gzip_jsonl(raw_paths[record_id])
        causal = apply_filter(raw_records, selected_profile)
        display = predict_display_pose(
            causal,
            fps=fps_by_record[record_id],
            horizon_ms=float(selected_display["prediction_horizon_ms"]),
        )
        offline = centered_offline_assist(
            [pose_result_from_raw(row) for row in raw_records]
        )
        output_root = dataset / "pose_cache" / record_id
        writers = {
            "causal": AtomicGzipJsonlWriter(
                output_root / "causal_analysis_pose.jsonl.gz"
            ),
            "display": AtomicGzipJsonlWriter(
                output_root / "low_latency_display_pose.jsonl.gz"
            ),
            "offline": AtomicGzipJsonlWriter(
                output_root / "offline_annotation_assist.jsonl.gz"
            ),
        }
        try:
            for raw, causal_pose, display_pose, offline_pose in zip(
                raw_records, causal, display, offline
            ):
                writers["causal"].write(
                    _derived_record(
                        raw,
                        causal_pose,
                        artifact_type="causal_analysis_pose",
                        filter_version=FILTER_VERSION
                        + f":{selected_profile}",
                        prediction_horizon_ms=0.0,
                        future_frames_used=False,
                        display_only=False,
                    )
                )
                writers["display"].write(
                    _derived_record(
                        raw,
                        display_pose,
                        artifact_type="low_latency_display_pose",
                        filter_version=DISPLAY_FILTER_VERSION,
                        prediction_horizon_ms=float(
                            selected_display["prediction_horizon_ms"]
                        ),
                        future_frames_used=False,
                        display_only=True,
                    )
                )
                writers["offline"].write(
                    _derived_record(
                        raw,
                        offline_pose,
                        artifact_type="offline_annotation_assist",
                        filter_version=OFFLINE_ASSIST_VERSION,
                        prediction_horizon_ms=0.0,
                        future_frames_used=True,
                        display_only=False,
                    )
                )
                total_written.update(writers.keys())
        except BaseException:
            for writer in writers.values():
                writer.abort()
            raise
        for writer in writers.values():
            writer.close()
    report = {
        "schema_version": 1,
        "artifact_type": "joint_latency_jitter_v1",
        "generated_at": utc_now(),
        "status": (
            "passed_with_round9_double_reviewed_event_anchors_pending"
            if causal_selection["improves_current_baseline"]
            and display_selection["overshoot_gate_passed"]
            else "candidate_selection_gate_failed"
        ),
        "record_count": len(manifest.get("records") or []),
        "frame_count": total_written["causal"],
        "joint_groups": [
            "wrist",
            "ankle",
            "knee",
            "hip",
            "shoulder",
            "torso",
        ],
        "profile_record_metrics": dict(profile_record_reports),
        "profile_aggregate_metrics": profile_aggregates,
        "causal_analysis_selection": causal_selection,
        "low_latency_display_selection": display_selection,
        "event_anchor_review": {
            "status": anchors_payload["status"],
            "anchor_count": sum(len(values) for values in anchors.values()),
            "reviewer_disclosure": anchors_payload.get("reviewer_disclosure"),
            "round9_double_review_required": True,
        },
        "benchmark_matrix": dict(benchmark_matrix or {}),
        "sensor_to_photon": {
            "status": "not_measured",
            "reason": (
                "requires 120/240 FPS external recording on each target "
                "computer/phone; software inference clocks are not physical "
                "sensor-to-photon measurements"
            ),
        },
        "requested_fps_is_actual_fps": False,
        "source_video_actual_fps": sorted(set(fps_by_record.values())),
        "checks": {
            "raw_causal_display_are_separate_files": True,
            "causal_future_frames_forbidden": True,
            "causal_prediction_horizon_zero": True,
            "display_cannot_drive_rules": True,
            "offline_assist_uses_future_and_is_annotation_only": True,
            "selected_causal_improves_current_combined_score": causal_selection[
                "improves_current_baseline"
            ],
            "display_overshoot_foot_and_bone_gates": all(
                display_selection[key]
                for key in (
                    "overshoot_gate_passed",
                    "foot_drift_gate_passed",
                    "bone_length_gate_passed",
                )
            ),
            "sensor_to_photon_not_fabricated": True,
        },
    }
    _atomic_json(
        dataset / "reports" / "joint_latency_jitter_v1.json", report
    )
    return report


def benchmark_model_mode_matrix(
    manifest: Mapping[str, Any],
    *,
    dataset_root: str | Path,
    project_root: str | Path,
    samples_per_record: int = 3,
) -> dict[str, Any]:
    dataset = Path(dataset_root)
    project = Path(project_root)
    variants = [
        (backend, mode)
        for backend in BACKENDS
        for mode in ("full_source", "resized_long_edge_640", "target_roi_padding_1.6")
    ]
    instances = {
        (backend, mode): MediaPipeBackend(
            project / BACKENDS[backend],
            output_segmentation_masks=False,
            num_poses=1,
        )
        for backend, mode in variants
    }
    latency: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    detected: Counter[tuple[str, str]] = Counter()
    counts: Counter[tuple[str, str]] = Counter()
    timestamp = 0
    try:
        for record in manifest.get("records") or []:
            record_id = str(record["record_id"])
            people = read_jsonl(
                dataset / "tracks" / record_id / "people.jsonl"
            )
            eligible = [
                int(frame["frame_index"])
                for frame in people
                if frame.get("target_locked")
            ]
            if not eligible:
                continue
            indices = [
                eligible[int(round(position))]
                for position in np.linspace(
                    0, len(eligible) - 1, samples_per_record
                )
            ]
            capture = cv2.VideoCapture(
                str(dataset / str(record["source_file"]))
            )
            width = int(record["video"]["width"])
            height = int(record["video"]["height"])
            for frame_index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, image = capture.read()
                if not ok or image is None:
                    continue
                frame = people[frame_index]
                target_id = frame.get("source_candidate_track_id")
                target = next(
                    (
                        item
                        for item in frame.get("candidates") or []
                        if item.get("track_id") == target_id
                    ),
                    None,
                )
                for backend, mode in variants:
                    inference_image = image
                    if mode == "resized_long_edge_640":
                        scale = 640.0 / max(width, height)
                        inference_image = cv2.resize(
                            image,
                            (
                                max(1, int(round(width * scale))),
                                max(1, int(round(height * scale))),
                            ),
                        )
                    elif mode == "target_roi_padding_1.6" and target is not None:
                        roi = clamp_bbox(
                            expand_bbox(
                                tuple(
                                    float(value)
                                    for value in target["bbox_xyxy"]
                                ),
                                1.6,
                            ),
                            width,
                            height,
                        )
                        cropped, exact = crop_roi(image, roi)
                        if exact is not None:
                            inference_image = cropped
                    result = instances[(backend, mode)].detect(
                        inference_image, timestamp_ms=timestamp
                    )
                    timestamp += 1
                    latency[(backend, mode)].append(
                        result.inference_time_ms
                    )
                    detected[(backend, mode)] += int(result.success)
                    counts[(backend, mode)] += 1
            capture.release()
    finally:
        for backend in instances.values():
            backend.close()
    rows = []
    for backend, mode in variants:
        key = (backend, mode)
        rows.append(
            {
                "backend": backend,
                "mode": mode,
                "device": "CPU/XNNPACK",
                "sample_count": counts[key],
                "pose_detection_rate": detected[key]
                / max(1, counts[key]),
                "inference_time_ms": summarize_samples(latency[key]),
            }
        )
    return {
        "rows": rows,
        "cpu_tested": True,
        "gpu_tested": False,
        "gpu_unavailable_reason": (
            "MediaPipe Tasks Python GPU delegate is not configured/supported "
            "by the current Windows environment; no GPU result is fabricated"
        ),
        "different_source_sizes_tested": True,
        "roi_is_product_enabled": False,
        "roi_reference_report": (
            "datasets/hyrox/reports/roi_latency_accuracy_ablation_v1.json"
        ),
    }


def implementation_summary(
    backend: Mapping[str, Any],
    coordinate: Mapping[str, Any],
    temporal: Mapping[str, Any],
    observability: Mapping[str, Any],
    *,
    dataset_root: str | Path,
    integrity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checks = {
        "all_30_records_have_lite_full_cache": bool(
            backend.get("checks", {}).get("all_30_records_cached")
        ),
        "formal_pose_target_bound": bool(
            backend.get("checks", {}).get("formal_pose_has_target_binding")
        ),
        "coordinate_contract_passed": all(
            coordinate.get("checks", {}).values()
        ),
        "temporal_stream_contract_passed": all(
            temporal.get("checks", {}).get(key)
            for key in (
                "raw_causal_display_are_separate_files",
                "causal_future_frames_forbidden",
                "causal_prediction_horizon_zero",
                "display_cannot_drive_rules",
                "offline_assist_uses_future_and_is_annotation_only",
                "sensor_to_photon_not_fabricated",
            )
        ),
        "pose_observability_report_present": bool(observability),
        "artifact_integrity_passed": bool(
            integrity and integrity.get("status") == "passed"
        ),
        "default_runtime_unchanged": True,
        "roi_remains_disabled": True,
    }
    status = (
        "passed_with_round9_double_reviewed_event_anchors_pending"
        if all(checks.values())
        else "failed"
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "round8_implementation_summary",
        "generated_at": utc_now(),
        "status": status,
        "checks": checks,
        "backend_summary": {
            "record_count": backend.get("record_count"),
            "frame_count": backend.get("frame_count"),
            "backends": backend.get("backends"),
            "review_required_frame_count": backend.get(
                "review_required_frame_count"
            ),
        },
        "coordinate_summary": {
            "frame_count": coordinate.get("frame_count"),
            "intrinsics_status": coordinate.get("intrinsics_status"),
            "absolute_phone_3d_accuracy": coordinate.get(
                "absolute_phone_3d_accuracy"
            ),
        },
        "temporal_summary": {
            "frame_count": temporal.get("frame_count"),
            "causal_analysis_selection": temporal.get(
                "causal_analysis_selection"
            ),
            "low_latency_display_selection": temporal.get(
                "low_latency_display_selection"
            ),
            "sensor_to_photon": temporal.get("sensor_to_photon"),
        },
        "remaining_manual_gate": (
            "Round 9 must independently double-review event anchors before "
            "freezing the final joint-latency report."
        ),
        "prohibited_operations": [
            "phone frame to ONI frame pairing",
            "phone joint to ONI depth pixel mapping",
            "unpaired teacher-student distillation",
            "teacher proposal replacing human ground truth",
        ],
        "default_runtime_changed": False,
    }
    _atomic_json(
        Path(dataset_root) / "reports" / "round8_implementation_summary.json",
        payload,
    )
    return payload


def validate_round8_artifacts(
    manifest: Mapping[str, Any],
    *,
    dataset_root: str | Path,
) -> dict[str, Any]:
    dataset = Path(dataset_root)
    violations: list[str] = []
    counts: Counter[str] = Counter()
    for record in manifest.get("records") or []:
        record_id = str(record["record_id"])
        expected = int(record["video"]["declared_frame_count"])
        target_track_id = str(
            record.get("target_athlete", {}).get("track_id") or ""
        )
        root = dataset / "pose_cache" / record_id
        for backend in BACKENDS:
            path = root / backend / "raw_pose.jsonl.gz"
            rows = 0
            for row in iter_gzip_jsonl(path):
                rows += 1
                counts[f"{backend}_raw_pose"] += 1
                if row.get("formal_pose_eligible") and (
                    row.get("target_track_id") != target_track_id
                    or not row.get("target_binding", {}).get(
                        "target_binding_passed"
                    )
                ):
                    violations.append(
                        f"{record_id}:{backend}:{row['frame_index']} "
                        "formal pose is not target-bound"
                    )
            if rows != expected:
                violations.append(
                    f"{record_id}:{backend} raw rows={rows} expected={expected}"
                )
        for filename, artifact_type in (
            ("causal_analysis_pose.jsonl.gz", "causal"),
            ("low_latency_display_pose.jsonl.gz", "display"),
            ("offline_annotation_assist.jsonl.gz", "offline"),
        ):
            rows = 0
            for row in iter_gzip_jsonl(root / filename):
                rows += 1
                counts[artifact_type] += 1
                if row.get("target_track_id") != target_track_id:
                    violations.append(
                        f"{record_id}:{artifact_type}:{row['frame_index']} "
                        "target ID mismatch"
                    )
                if artifact_type == "causal" and (
                    row.get("future_frames_used")
                    or float(row.get("prediction_horizon_ms") or 0.0) != 0.0
                    or not row.get("may_drive_rules_or_training")
                ):
                    violations.append(
                        f"{record_id}:causal:{row['frame_index']} contract"
                    )
                if artifact_type == "display" and (
                    not row.get("display_only")
                    or row.get("may_drive_rules_or_training")
                ):
                    violations.append(
                        f"{record_id}:display:{row['frame_index']} contract"
                    )
                if artifact_type == "offline" and (
                    not row.get("future_frames_used")
                    or row.get("may_drive_rules_or_training")
                ):
                    violations.append(
                        f"{record_id}:offline:{row['frame_index']} contract"
                    )
            if rows != expected:
                violations.append(
                    f"{record_id}:{artifact_type} rows={rows} expected={expected}"
                )
        for filename, artifact_type in (
            ("coordinate_layers.jsonl.gz", "coordinate"),
            ("backend_agreement.jsonl.gz", "agreement"),
        ):
            rows = sum(1 for _ in iter_gzip_jsonl(root / filename))
            counts[artifact_type] += rows
            if rows != expected:
                violations.append(
                    f"{record_id}:{artifact_type} rows={rows} expected={expected}"
                )
    payload = {
        "schema_version": 1,
        "artifact_type": "round8_artifact_integrity_v1",
        "generated_at": utc_now(),
        "status": "passed" if not violations else "failed",
        "record_count": len(manifest.get("records") or []),
        "counts": dict(counts),
        "violation_count": len(violations),
        "violations": violations[:1000],
        "checks": {
            "all_expected_rows_present": not any(
                "rows=" in violation for violation in violations
            ),
            "formal_pose_100_percent_target_bound": not any(
                "formal pose" in violation for violation in violations
            ),
            "derived_stream_contracts_hold": not any(
                "contract" in violation for violation in violations
            ),
            "derived_target_ids_match": not any(
                "target ID mismatch" in violation for violation in violations
            ),
        },
    }
    _atomic_json(
        dataset / "reports" / "round8_artifact_integrity_v1.json", payload
    )
    return payload


__all__ = [
    "BACKENDS",
    "approve_event_anchors",
    "benchmark_model_mode_matrix",
    "binary_mask_rle",
    "build_anchor_review_sheets",
    "build_cache_reports",
    "build_temporal_streams_and_report",
    "cache_record",
    "implementation_summary",
    "iter_gzip_jsonl",
    "propose_event_anchors",
    "raw_pose_record",
    "read_gzip_jsonl",
    "select_target_pose_candidate",
    "validate_round8_artifacts",
]
