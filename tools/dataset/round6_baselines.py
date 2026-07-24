"""Phone RGB pose, coordinate-quality and realtime latency baselines."""

from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from hyrox.features import extract_basic_pose_features
from hyrox.registry import create_action_analyzer
from src.backends.base import Keypoint
from src.backends.mediapipe_backend import MediaPipeBackend
from src.utils.keypoint_schema import MEDIAPIPE_33_NAMES, MEDIAPIPE_CONNECTIONS
from src.utils.smoothing import KeypointSmoother
from tools.benchmark_latency_baseline import run_baseline, summarize_samples

from .manifest import sha256_file, utc_now
from .phone_rgb import _atomic_json


FULL_BODY_LANDMARKS = frozenset(
    {
        "left_shoulder",
        "right_shoulder",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    }
)


def _finite_point(point: Keypoint, threshold: float = 0.0) -> bool:
    return (
        point.confidence >= threshold
        and math.isfinite(point.x)
        and math.isfinite(point.y)
        and math.isfinite(point.z)
    )


def _point_map(points: Sequence[Keypoint]) -> dict[str, Keypoint]:
    return {point.name: point for point in points}


def _distance(first: Keypoint, second: Keypoint, *, dimensions: int = 2) -> float:
    values = [(first.x - second.x) ** 2, (first.y - second.y) ** 2]
    if dimensions == 3:
        values.append((first.z - second.z) ** 2)
    return float(math.sqrt(sum(values)))


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    return str(value)


class CoordinateAccumulator:
    def __init__(self) -> None:
        self.analyzed_frames = 0
        self.pose_frames = 0
        self.world_frames = 0
        self.joint_observed: Counter[str] = Counter()
        self.joint_visible: Counter[str] = Counter()
        self.joint_world: Counter[str] = Counter()
        self.joint_jumps: Counter[str] = Counter()
        self.joint_conflicts: Counter[str] = Counter()
        self.visibility_samples: defaultdict[str, list[float]] = defaultdict(list)
        self.bone_2d: defaultdict[str, list[float]] = defaultdict(list)
        self.bone_world: defaultdict[str, list[float]] = defaultdict(list)
        self.left_right_order_flips: Counter[str] = Counter()
        self.left_right_order_comparisons: Counter[str] = Counter()
        self._previous_image: dict[str, dict[str, Keypoint]] = {}
        self._previous_world: dict[str, dict[str, Keypoint]] = {}
        self._previous_order: dict[str, dict[str, int]] = {}

    def observe(
        self,
        record_id: str,
        image_points: Sequence[Keypoint],
        world_points: Sequence[Keypoint],
    ) -> None:
        self.analyzed_frames += 1
        image = _point_map(image_points)
        world = _point_map(world_points)
        if image_points:
            self.pose_frames += 1
        if world_points:
            self.world_frames += 1
        previous_image = self._previous_image.get(record_id, {})
        previous_world = self._previous_world.get(record_id, {})

        for name in MEDIAPIPE_33_NAMES:
            point = image.get(name)
            world_point = world.get(name)
            image_ok = point is not None and _finite_point(point)
            world_ok = world_point is not None and _finite_point(world_point)
            if image_ok:
                self.joint_observed[name] += 1
                self.visibility_samples[name].append(float(point.visibility))
                if point.confidence >= 0.5:
                    self.joint_visible[name] += 1
            if world_ok:
                self.joint_world[name] += 1
            previous = previous_image.get(name)
            previous_w = previous_world.get(name)
            image_jump = bool(image_ok and previous is not None and _distance(point, previous) > 0.15)
            world_jump = bool(
                world_ok and previous_w is not None and _distance(world_point, previous_w, dimensions=3) > 0.25
            )
            if image_jump:
                self.joint_jumps[name] += 1
            if image_ok and world_ok and previous is not None and previous_w is not None and image_jump != world_jump:
                self.joint_conflicts[name] += 1

        for first_index, second_index in MEDIAPIPE_CONNECTIONS:
            first_name = MEDIAPIPE_33_NAMES[first_index]
            second_name = MEDIAPIPE_33_NAMES[second_index]
            key = f"{first_name}__{second_name}"
            first = image.get(first_name)
            second = image.get(second_name)
            if first is not None and second is not None and _finite_point(first, 0.2) and _finite_point(second, 0.2):
                self.bone_2d[key].append(_distance(first, second))
            first_w = world.get(first_name)
            second_w = world.get(second_name)
            if first_w is not None and second_w is not None and _finite_point(first_w, 0.2) and _finite_point(second_w, 0.2):
                self.bone_world[key].append(_distance(first_w, second_w, dimensions=3))

        previous_order = self._previous_order.setdefault(record_id, {})
        for left_name in (name for name in MEDIAPIPE_33_NAMES if name.startswith("left_")):
            right_name = "right_" + left_name.removeprefix("left_")
            left = image.get(left_name)
            right = image.get(right_name)
            if left is None or right is None or not _finite_point(left, 0.2) or not _finite_point(right, 0.2):
                continue
            pair = left_name.removeprefix("left_")
            delta = left.x - right.x
            if abs(delta) < 0.01:
                continue
            sign = 1 if delta > 0 else -1
            if pair in previous_order:
                self.left_right_order_comparisons[pair] += 1
                if previous_order[pair] != sign:
                    self.left_right_order_flips[pair] += 1
            previous_order[pair] = sign

        if image_points:
            self._previous_image[record_id] = image
        if world_points:
            self._previous_world[record_id] = world

    @staticmethod
    def _variation(values: Sequence[float]) -> dict[str, float | int | None]:
        samples = np.asarray(values, dtype=float)
        samples = samples[np.isfinite(samples)]
        if not samples.size:
            return {"samples": 0, "mean": None, "std": None, "coefficient_of_variation": None}
        mean = float(np.mean(samples))
        return {
            "samples": int(samples.size),
            "mean": mean,
            "std": float(np.std(samples)),
            "coefficient_of_variation": float(np.std(samples) / mean) if mean > 1e-9 else None,
        }

    def report(self, *, model_path: Path, analyzed_source_frames: int) -> dict[str, Any]:
        joint_metrics = {}
        denominator = max(1, self.analyzed_frames)
        for name in MEDIAPIPE_33_NAMES:
            visibility = summarize_samples(self.visibility_samples[name])
            joint_metrics[name] = {
                "missing_rate": 1.0 - (self.joint_observed[name] / denominator),
                "visibility_at_least_0_5_rate": self.joint_visible[name] / denominator,
                "visibility": visibility,
                "world_available_rate": self.joint_world[name] / denominator,
                "normalized_2d_jump_rate": self.joint_jumps[name] / max(1, self.joint_observed[name]),
                "image_world_temporal_conflict_rate": self.joint_conflicts[name] / max(1, self.joint_observed[name]),
            }
        pairs = {
            pair: {
                "ordering_comparisons": self.left_right_order_comparisons[pair],
                "ordering_flip_count": self.left_right_order_flips[pair],
                "ordering_flip_rate": self.left_right_order_flips[pair] / max(1, self.left_right_order_comparisons[pair]),
            }
            for pair in sorted(set(self.left_right_order_comparisons) | set(self.left_right_order_flips))
        }
        return {
            "schema_version": 1,
            "artifact_type": "coordinate_baseline_v1",
            "generated_at": utc_now(),
            "model": {"path": model_path.as_posix(), "sha256": sha256_file(model_path)},
            "analyzed_source_frames": analyzed_source_frames,
            "coordinate_spaces": {
                "image_normalized_2d": {
                    "available_frames": self.pose_frames,
                    "availability_rate": self.pose_frames / denominator,
                    "units": "normalized image width/height",
                },
                "pixel_2d": {
                    "available_frames": self.pose_frames,
                    "availability_rate": self.pose_frames / denominator,
                    "derivation": "image_normalized_2d multiplied by decoded frame dimensions",
                },
                "mediapipe_world": {
                    "available_frames": self.world_frames,
                    "availability_rate": self.world_frames / denominator,
                    "contract": "body-centred relative 3D only; not camera coordinates or metric venue coordinates",
                },
            },
            "joint_metrics": joint_metrics,
            "bone_segment_length_variation": {
                key: {"image_normalized_2d": self._variation(self.bone_2d[key]), "mediapipe_world": self._variation(self.bone_world[key])}
                for key in sorted(set(self.bone_2d) | set(self.bone_world))
            },
            "left_right_ordering_audit": {
                "method": "temporal flips of left-vs-right image x ordering; review signal, not identity-switch ground truth",
                "pairs": pairs,
            },
            "jump_definition": {"image_normalized_2d": ">0.15 per analyzed consecutive frame", "mediapipe_world": ">0.25 per analyzed consecutive frame"},
            "image_world_conflict_definition": "exactly one coordinate space exceeds its jump threshold on a jointly available temporal pair",
        }


def _full_body_visible(points: Sequence[Keypoint]) -> bool:
    mapping = _point_map(points)
    return all(
        name in mapping
        and _finite_point(mapping[name], 0.5)
        and -0.02 <= mapping[name].x <= 1.02
        and -0.02 <= mapping[name].y <= 1.02
        for name in FULL_BODY_LANDMARKS
    )


def _finish_low_pose_interval(
    output: list[dict[str, Any]], start: int | None, end: int, fps: float, *, minimum_frames: int
) -> None:
    if start is None or end < start or end - start + 1 < minimum_frames:
        return
    output.append(
        {
            "proposal_type": "idle_or_setup_or_exit_or_transition_or_occlusion",
            "start_frame": start,
            "end_frame": end,
            "start_ms": round(start * 1000.0 / fps, 3),
            "end_ms": round((end + 1) * 1000.0 / fps, 3),
            "proposal_basis": "contiguous raw pose detector miss",
            "human_confirmation_required": True,
            "may_be_background_negative": False,
        }
    )


def _state_summary(state: Mapping[str, object] | None) -> dict[str, Any]:
    if not state:
        return {}
    fields = (
        "action",
        "phase",
        "rep_count",
        "cycle_count",
        "candidate_count",
        "pose_valid_rep_count",
        "no_rep_count",
        "unsure_count",
        "last_rep_decision",
        "last_rep_observability",
    )
    return {field: _json_safe(state.get(field)) for field in fields if field in state}


def run_phone_pose_baseline(
    manifest: Mapping[str, Any],
    dataset_root: str | Path,
    *,
    model: str | Path = "models/pose_landmarker_full.task",
    frame_stride: int = 1,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    root = Path(dataset_root)
    model_path = Path(model)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    model_path = model_path.resolve()
    stride = max(1, int(frame_stride))
    coordinate = CoordinateAccumulator()
    per_record: list[dict[str, Any]] = []
    interval_candidates: dict[str, list[dict[str, Any]]] = {}
    total_source_frames = 0
    total_analyzed_frames = 0
    total_pose_frames = 0
    total_full_body_frames = 0
    all_inference_ms: list[float] = []
    all_pipeline_ms: list[float] = []

    for record in manifest.get("records") or []:
        record_id = str(record["record_id"])
        video_path = root / str(record["source_file"])
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"cannot open phone RGB backup for baseline: {video_path}")
        fps = float(record.get("video", {}).get("fps") or capture.get(cv2.CAP_PROP_FPS) or 30.0)
        analyzer = create_action_analyzer(
            str(record["action"]), camera_view=str(record["camera_view"]), live_mode=False
        )
        smoother = KeypointSmoother(mode="one-euro", max_missing_frames=5, occlusion_guard=True)
        backend = MediaPipeBackend(model_path, output_segmentation_masks=False)
        source_frames = 0
        analyzed = 0
        poses = 0
        full_body = 0
        record_inference: list[float] = []
        record_pipeline: list[float] = []
        low_pose_start: int | None = None
        candidates: list[dict[str, Any]] = []
        final_state: Mapping[str, object] | None = None
        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frame_index = source_frames
                source_frames += 1
                if frame_index % stride:
                    continue
                analyzed += 1
                timestamp_ms = max(0, int(round(frame_index * 1000.0 / fps)))
                pipeline_started = time.perf_counter()
                raw_result = backend.detect(frame, timestamp_ms=timestamp_ms)
                raw_has_pose = bool(raw_result.success and raw_result.keypoints)
                world_points = raw_result.extra.get("world_keypoints") or []
                coordinate.observe(record_id, raw_result.keypoints, world_points)
                result = smoother.smooth_result(raw_result)
                has_pose = bool(result.success and result.keypoints)
                features = None
                if has_pose:
                    features = extract_basic_pose_features(
                        result.keypoints,
                        image_width=frame.shape[1],
                        image_height=frame.shape[0],
                        segmentation_mask=None,
                    )
                final_state = analyzer.attach_view_context(
                    analyzer.update(features if has_pose else None, timestamp_ms=timestamp_ms)
                )
                pipeline_ms = (time.perf_counter() - pipeline_started) * 1000.0
                record_inference.append(float(raw_result.inference_time_ms))
                record_pipeline.append(pipeline_ms)
                if raw_has_pose:
                    poses += 1
                    if _full_body_visible(raw_result.keypoints):
                        full_body += 1
                    _finish_low_pose_interval(
                        candidates,
                        low_pose_start,
                        frame_index - stride,
                        fps,
                        minimum_frames=max(1, int(round(0.5 * fps / stride))),
                    )
                    low_pose_start = None
                elif low_pose_start is None:
                    low_pose_start = frame_index
        finally:
            capture.release()
            backend.close()
        _finish_low_pose_interval(
            candidates,
            low_pose_start,
            max(0, source_frames - 1),
            fps,
            minimum_frames=max(1, int(round(0.5 * fps / stride))),
        )
        interval_candidates[record_id] = candidates
        total_source_frames += source_frames
        total_analyzed_frames += analyzed
        total_pose_frames += poses
        total_full_body_frames += full_body
        all_inference_ms.extend(record_inference)
        all_pipeline_ms.extend(record_pipeline)
        per_record.append(
            {
                "record_id": record_id,
                "source_filename": record["source_filename"],
                "action": record["action"],
                "camera_view": record["camera_view"],
                "source_frame_count": source_frames,
                "analyzed_frame_count": analyzed,
                "frame_stride": stride,
                "raw_pose_detected_frames": poses,
                "raw_pose_detection_rate": poses / max(1, analyzed),
                "full_body_visible_frames": full_body,
                "full_body_visible_rate": full_body / max(1, analyzed),
                "target_athlete_locked": False,
                "subject_bound_metric_status": "provisional_until_round7_target_lock",
                "pose_inference_ms": summarize_samples(record_inference),
                "pose_and_rule_pipeline_ms": summarize_samples(record_pipeline),
                "current_rule_output": _state_summary(final_state),
                "accuracy": None,
                "accuracy_reason": "recording intent in filename is unverified and is not ground truth",
                "interval_review_candidates": candidates,
            }
        )

    phone_report = {
        "schema_version": 1,
        "artifact_type": "phone_rgb_data_baseline_v1",
        "generated_at": utc_now(),
        "pipeline": "decoded phone RGB -> MediaPipe -> existing smoothing -> existing action-specific rules",
        "model": {"path": model_path.as_posix(), "sha256": sha256_file(model_path)},
        "smoothing": {"mode": "one-euro", "max_missing_frames": 5, "occlusion_guard": True},
        "frame_stride": stride,
        "summary": {
            "record_count": len(per_record),
            "source_frame_count": total_source_frames,
            "analyzed_frame_count": total_analyzed_frames,
            "raw_pose_detected_frames": total_pose_frames,
            "raw_pose_detection_rate": total_pose_frames / max(1, total_analyzed_frames),
            "full_body_visible_frames": total_full_body_frames,
            "full_body_visible_rate": total_full_body_frames / max(1, total_analyzed_frames),
            "pose_inference_ms": summarize_samples(all_inference_ms),
            "pose_and_rule_pipeline_ms": summarize_samples(all_pipeline_ms),
            "target_locked_record_count": 0,
            "accuracy_computed": False,
        },
        "records": per_record,
        "interpretation_limits": [
            "filename recording intent is not used as ground truth",
            "no accuracy, precision or recall is computed",
            "pose/full-body rates are provisional because target athlete tracks are not yet locked",
            "background people must not be converted to negatives from this report",
        ],
        "baseline_isolation": {
            "legacy_compatibility_baseline": "reports/baseline/golden_results.json",
            "phone_rgb_data_baseline": "this artifact",
            "realtime_latency_baseline": "reports/baseline/realtime_latency_baseline_v1.json",
        },
    }
    coordinate_report = coordinate.report(
        model_path=model_path, analyzed_source_frames=total_analyzed_frames
    )
    return phone_report, coordinate_report, interval_candidates


def build_realtime_latency_baseline(
    manifest: Mapping[str, Any],
    dataset_root: str | Path,
    *,
    project_root: str | Path,
    max_frames: int = 300,
) -> dict[str, Any]:
    root = Path(project_root)
    records = list(manifest.get("records") or [])
    selected = next((record for record in records if record.get("example_candidate")), records[0])
    input_video = Path(dataset_root) / str(selected["source_file"])
    tiers = {}
    for tier in ("lite", "full"):
        model = root / "models" / f"pose_landmarker_{tier}.task"
        report = run_baseline(input_video, model=model, max_frames=max_frames, warmup_frames=5)
        report["model_tier"] = tier
        report["model_sha256"] = sha256_file(model)
        tiers[tier] = report
    prior_path = root / "reports" / "baseline" / "latency_report.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.is_file() else None
    camera = prior.get("camera_benchmark") if isinstance(prior, Mapping) else None
    return {
        "schema_version": 1,
        "artifact_type": "realtime_latency_baseline_v1",
        "generated_at": utc_now(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "mediapipe": importlib.metadata.version("mediapipe"),
        },
        "deterministic_phone_video": {
            "record_id": selected["record_id"],
            "source_filename": selected["source_filename"],
            "model_tiers": tiers,
        },
        "camera_chain_snapshot": {
            "source": "reports/baseline/latency_report.json" if camera is not None else None,
            "source_sha256": sha256_file(prior_path) if prior_path.is_file() else None,
            "report": camera,
            "status": "frozen_existing_measurement" if camera is not None else "measurement_pending",
        },
        "current_scheduler": {
            "mode": "synchronous_sequential",
            "queue_capacity": None,
            "queue_drop_count": None,
            "queue_drop_status": "not_instrumented_in_current_sync_pipeline",
        },
        "frame_age_and_display": {
            "pose_age_ms": None,
            "display_latency_ms": None,
            "sensor_to_photon_ms": None,
            "status": "external_or_timestamp_instrumentation_required",
            "measurement_plan": "use capture/pose/analysis/render timestamps plus 120/240 FPS external display recording",
        },
        "default_runtime_changed": False,
    }


def write_round6_baselines(
    manifest: Mapping[str, Any],
    dataset_root: str | Path,
    *,
    project_root: str | Path,
    model: str | Path = "models/pose_landmarker_full.task",
    frame_stride: int = 1,
    latency_max_frames: int = 300,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    root = Path(project_root)
    phone, coordinate, intervals = run_phone_pose_baseline(
        manifest, dataset_root, model=root / model, frame_stride=frame_stride
    )
    latency = build_realtime_latency_baseline(
        manifest, dataset_root, project_root=root, max_frames=latency_max_frames
    )
    output = root / "reports" / "baseline"
    _atomic_json(output / "phone_rgb_data_baseline_v1.json", phone)
    _atomic_json(output / "coordinate_baseline_v1.json", coordinate)
    _atomic_json(output / "realtime_latency_baseline_v1.json", latency)
    return phone, coordinate, latency, intervals


__all__ = [
    "CoordinateAccumulator",
    "build_realtime_latency_baseline",
    "run_phone_pose_baseline",
    "write_round6_baselines",
]
