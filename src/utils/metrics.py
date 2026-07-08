from __future__ import annotations

import csv
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np

from src.backends.base import PoseResult


@dataclass(frozen=True)
class RealtimeMetricsSnapshot:
    realtime_fps: float
    avg_fps: float
    inference_time_ms: float
    avg_inference_time_ms: float
    p95_inference_time_ms: float
    end_to_end_latency_ms: float
    num_keypoints: int
    success_rate: float
    avg_keypoint_confidence: float
    missing_rate_shoulder: float
    missing_rate_hip: float
    missing_rate_knee: float
    missing_rate_ankle: float
    person_lost_count: int
    keypoint_jitter: float
    angle_jitter: float
    yolo_detection_time_ms: float = 0.0
    avg_yolo_detection_time_ms: float = 0.0
    roi_success_rate: float = 0.0
    bbox_reuse_count: int = 0
    bbox_lost_count: int = 0
    fallback_to_full_frame_count: int = 0
    source_model_distribution: str = ""
    stabilized_hold_count: int = 0
    occlusion_guard_count: int = 0


class RealtimeMetrics:
    def __init__(
        self,
        backend: str,
        smoothing: str,
        input_name: str = "camera",
        person_detector: str = "none",
        fusion: str = "none",
        detector_every_n: int = 0,
        backend_device: str = "",
        detector_device: str = "",
    ) -> None:
        self.backend = backend
        self.smoothing = smoothing
        self.input_name = input_name
        self.person_detector = person_detector
        self.fusion = fusion
        self.detector_every_n = int(detector_every_n)
        self.backend_device = backend_device or "auto"
        self.detector_device = detector_device or "auto"
        self._backend_history = [self.backend]
        self._backend_device_history = [self.backend_device]
        self.started = time.perf_counter()
        self.frame_count = 0
        self.success_count = 0
        self.person_lost_count = 0
        self.roi_enabled_count = 0
        self.roi_success_count = 0
        self.bbox_reuse_count = 0
        self.bbox_lost_count = 0
        self.fallback_to_full_frame_count = 0
        self.stabilized_hold_count = 0
        self.occlusion_guard_count = 0
        self._source_model_counts: Counter[str] = Counter()
        self._last_num_keypoints = 0
        self._last_frame_time: float | None = None
        self._fps_values: list[float] = []
        self._inference_times: list[float] = []
        self._yolo_detection_times: list[float] = []
        self._latencies: list[float] = []
        self._keypoint_confidences: list[float] = []
        self._missing_counts = {"shoulder": 0, "hip": 0, "knee": 0, "ankle": 0}
        self._missing_totals = {"shoulder": 0, "hip": 0, "knee": 0, "ankle": 0}
        self._previous_points: dict[str, tuple[float, float]] = {}
        self._keypoint_jitters: list[float] = []
        self._previous_angles: dict[str, float] = {}
        self._angle_jitters: list[float] = []

    def set_backend(self, backend: str, backend_device: str = "") -> None:
        device = backend_device or "auto"
        if backend != self._backend_history[-1]:
            self._backend_history.append(backend)
            self._backend_device_history.append(device)
        self.backend = "->".join(self._backend_history)
        self.backend_device = "->".join(self._backend_device_history)

    def update(
        self,
        result: PoseResult,
        angles: dict[str, float | None],
        frame_started: float,
        frame_finished: float | None = None,
        roi_enabled: bool = False,
        roi_success: bool = False,
        yolo_detection_time_ms: float = 0.0,
        bbox_reused: bool = False,
        bbox_lost: bool = False,
        fallback_to_full_frame: bool = False,
        source_model_distribution: dict[str, int] | None = None,
    ) -> RealtimeMetricsSnapshot:
        frame_finished = frame_finished or time.perf_counter()
        self.frame_count += 1
        if result.success:
            self.success_count += 1
        else:
            self.person_lost_count += 1

        if self._last_frame_time is not None:
            dt = frame_finished - self._last_frame_time
            if dt > 0:
                self._fps_values.append(1.0 / dt)
        self._last_frame_time = frame_finished

        self._inference_times.append(float(result.inference_time_ms))
        if yolo_detection_time_ms > 0:
            self._yolo_detection_times.append(float(yolo_detection_time_ms))
        if roi_enabled:
            self.roi_enabled_count += 1
        if roi_success:
            self.roi_success_count += 1
        if bbox_reused:
            self.bbox_reuse_count += 1
        if bbox_lost:
            self.bbox_lost_count += 1
        if fallback_to_full_frame:
            self.fallback_to_full_frame_count += 1
        if result.extra.get("stabilized_hold"):
            self.stabilized_hold_count += 1
        guarded_keypoints = result.extra.get("occlusion_guarded_keypoints") or ()
        self.occlusion_guard_count += len(guarded_keypoints)
        self._record_source_models(result, source_model_distribution)
        self._latencies.append((frame_finished - frame_started) * 1000.0)
        self._last_num_keypoints = result.num_keypoints
        self._record_keypoint_confidence_and_missing(result)
        self._record_keypoint_jitter(result)
        self._record_angle_jitter(angles)
        return self.snapshot()

    def snapshot(self) -> RealtimeMetricsSnapshot:
        return RealtimeMetricsSnapshot(
            realtime_fps=self._fps_values[-1] if self._fps_values else 0.0,
            avg_fps=mean(self._fps_values) if self._fps_values else 0.0,
            inference_time_ms=self._inference_times[-1] if self._inference_times else 0.0,
            avg_inference_time_ms=mean(self._inference_times) if self._inference_times else 0.0,
            p95_inference_time_ms=float(np.percentile(self._inference_times, 95)) if self._inference_times else 0.0,
            end_to_end_latency_ms=self._latencies[-1] if self._latencies else 0.0,
            num_keypoints=self._last_num_keypoints,
            success_rate=self.success_count / self.frame_count if self.frame_count else 0.0,
            avg_keypoint_confidence=mean(self._keypoint_confidences) if self._keypoint_confidences else 0.0,
            missing_rate_shoulder=self._missing_rate("shoulder"),
            missing_rate_hip=self._missing_rate("hip"),
            missing_rate_knee=self._missing_rate("knee"),
            missing_rate_ankle=self._missing_rate("ankle"),
            person_lost_count=self.person_lost_count,
            keypoint_jitter=mean(self._keypoint_jitters) if self._keypoint_jitters else 0.0,
            angle_jitter=mean(self._angle_jitters) if self._angle_jitters else 0.0,
            yolo_detection_time_ms=self._yolo_detection_times[-1] if self._yolo_detection_times else 0.0,
            avg_yolo_detection_time_ms=mean(self._yolo_detection_times) if self._yolo_detection_times else 0.0,
            roi_success_rate=self.roi_success_count / self.roi_enabled_count if self.roi_enabled_count else 0.0,
            bbox_reuse_count=self.bbox_reuse_count,
            bbox_lost_count=self.bbox_lost_count,
            fallback_to_full_frame_count=self.fallback_to_full_frame_count,
            source_model_distribution=self._source_model_distribution_text(),
            stabilized_hold_count=self.stabilized_hold_count,
            occlusion_guard_count=self.occlusion_guard_count,
        )

    def write_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "input": self.input_name,
            "backend": self.backend,
            "fusion": self.fusion,
            "person_detector": self.person_detector,
            "detector_every_n": self.detector_every_n,
            "backend_device": self.backend_device,
            "detector_device": self.detector_device,
            "smoothing": self.smoothing,
            "total_frames": self.frame_count,
            "success_frames": self.success_count,
            "success_rate": self.snapshot().success_rate,
            "num_keypoints": self.snapshot().num_keypoints,
            "avg_fps": self.snapshot().avg_fps,
            "avg_inference_time_ms": self.snapshot().avg_inference_time_ms,
            "p95_inference_time_ms": self.snapshot().p95_inference_time_ms,
            "avg_end_to_end_latency_ms": mean(self._latencies) if self._latencies else 0.0,
            "avg_keypoint_confidence": self.snapshot().avg_keypoint_confidence,
            "missing_rate_shoulder": self.snapshot().missing_rate_shoulder,
            "missing_rate_hip": self.snapshot().missing_rate_hip,
            "missing_rate_knee": self.snapshot().missing_rate_knee,
            "missing_rate_ankle": self.snapshot().missing_rate_ankle,
            "person_lost_count": self.person_lost_count,
            "keypoint_jitter": self.snapshot().keypoint_jitter,
            "angle_jitter": self.snapshot().angle_jitter,
            "roi_enabled": self.roi_enabled_count > 0,
            "roi_success_rate": self.snapshot().roi_success_rate,
            "yolo_detection_time_ms": self.snapshot().yolo_detection_time_ms,
            "avg_yolo_detection_time_ms": self.snapshot().avg_yolo_detection_time_ms,
            "bbox_reuse_count": self.bbox_reuse_count,
            "bbox_lost_count": self.bbox_lost_count,
            "fallback_to_full_frame_count": self.fallback_to_full_frame_count,
            "source_model_distribution": self.snapshot().source_model_distribution,
            "stabilized_hold_count": self.stabilized_hold_count,
            "occlusion_guard_count": self.occlusion_guard_count,
        }
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def summary_lines(self) -> list[str]:
        snapshot = self.snapshot()
        return [
            f"frames={self.frame_count}",
            f"success_rate={snapshot.success_rate:.3f}",
            f"avg_fps={snapshot.avg_fps:.1f}",
            f"avg_inference_ms={snapshot.avg_inference_time_ms:.1f}",
            f"num_keypoints={snapshot.num_keypoints}",
            f"avg_keypoint_confidence={snapshot.avg_keypoint_confidence:.3f}",
            f"missing_rate_shoulder={snapshot.missing_rate_shoulder:.3f}",
            f"missing_rate_hip={snapshot.missing_rate_hip:.3f}",
            f"missing_rate_knee={snapshot.missing_rate_knee:.3f}",
            f"missing_rate_ankle={snapshot.missing_rate_ankle:.3f}",
            f"avg_yolo_detection_ms={snapshot.avg_yolo_detection_time_ms:.1f}",
            f"roi_success_rate={snapshot.roi_success_rate:.3f}",
            f"fallback_to_full_frame_count={snapshot.fallback_to_full_frame_count}",
            f"source_model_distribution={snapshot.source_model_distribution or 'none'}",
            f"stabilized_hold_count={snapshot.stabilized_hold_count}",
            f"occlusion_guard_count={snapshot.occlusion_guard_count}",
            f"p95_inference_ms={snapshot.p95_inference_time_ms:.1f}",
            f"person_lost_count={snapshot.person_lost_count}",
            f"keypoint_jitter={snapshot.keypoint_jitter:.5f}",
            f"angle_jitter={snapshot.angle_jitter:.3f}",
        ]

    def _record_source_models(self, result: PoseResult, distribution: dict[str, int] | None) -> None:
        if distribution:
            self._source_model_counts.update(distribution)
            return
        if not result.success:
            return
        for point in result.keypoints:
            self._source_model_counts[point.source_model or result.model_name or "unknown"] += 1

    def _source_model_distribution_text(self) -> str:
        if not self._source_model_counts:
            return ""
        return ";".join(f"{name}:{count}" for name, count in sorted(self._source_model_counts.items()))

    def _record_keypoint_confidence_and_missing(self, result: PoseResult) -> None:
        by_name = {point.name: point for point in result.keypoints} if result.success else {}
        if result.success:
            confidences = [float(point.confidence) for point in result.keypoints if np.isfinite(point.confidence)]
            self._keypoint_confidences.extend(confidences)

        for part in self._missing_counts:
            names = (f"left_{part}", f"right_{part}")
            for name in names:
                point = by_name.get(name)
                self._missing_totals[part] += 1
                if point is None or point.confidence < 0.2 or not np.isfinite(point.x) or not np.isfinite(point.y):
                    self._missing_counts[part] += 1

    def _missing_rate(self, part: str) -> float:
        total = self._missing_totals.get(part, 0)
        if total <= 0:
            return 0.0
        return self._missing_counts.get(part, 0) / total

    def _record_keypoint_jitter(self, result: PoseResult) -> None:
        if not result.success:
            self._previous_points.clear()
            return
        distances: list[float] = []
        next_points: dict[str, tuple[float, float]] = {}
        for point in result.keypoints:
            if point.confidence < 0.2:
                continue
            key = f"{point.source_model or result.model_name}:{point.name}"
            next_points[key] = (point.x, point.y)
            previous = self._previous_points.get(key)
            if previous is not None:
                dx = point.x - previous[0]
                dy = point.y - previous[1]
                distances.append((dx * dx + dy * dy) ** 0.5)
        if distances:
            self._keypoint_jitters.append(mean(distances))
        self._previous_points = next_points

    def _record_angle_jitter(self, angles: dict[str, float | None]) -> None:
        deltas: list[float] = []
        next_angles: dict[str, float] = {}
        for name, value in angles.items():
            if value is None:
                continue
            next_angles[name] = float(value)
            previous = self._previous_angles.get(name)
            if previous is not None:
                deltas.append(abs(float(value) - previous))
        if deltas:
            self._angle_jitters.append(mean(deltas))
        self._previous_angles = next_angles
