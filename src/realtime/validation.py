"""Latest-only validation worker and explicit realtime data-layer contract."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

from src.biomechanics.biomech_metrics import build_biomechanical_representation
from src.biomechanics.local_ground_frame import build_local_ground_frame


@dataclass(frozen=True, slots=True)
class ValidationTask:
    source_timestamp_ms: float
    world_points: tuple[object, ...]
    quality_points: tuple[object, ...]
    body_coordinate_system: Mapping[str, Any]
    ground_estimation: Mapping[str, Any]
    foot_contact_evidence: Mapping[str, Any]
    measurements: Mapping[str, object]
    local_ground_minimum_confidence: float
    biomech_minimum_confidence: float


class ValidationBudgetGate:
    """Degrade only validation cadence; never modifies render or pose clocks."""

    def __init__(
        self,
        *,
        warning_pose_age_ms: float = 80.0,
        validation_budget_ms: float = 4.0,
        maximum_stride: int = 4,
    ) -> None:
        self.warning_pose_age_ms = max(1.0, float(warning_pose_age_ms))
        self.validation_budget_ms = max(0.1, float(validation_budget_ms))
        self.maximum_stride = max(1, int(maximum_stride))
        self.stride = 1
        self._headroom = 0

    def observe(self, *, pose_age_ms: float, validation_ms: float) -> int:
        overloaded = (
            float(pose_age_ms) > self.warning_pose_age_ms
            or float(validation_ms) > self.validation_budget_ms
        )
        underloaded = (
            float(pose_age_ms) < self.warning_pose_age_ms * 0.60
            and float(validation_ms) < self.validation_budget_ms * 0.60
        )
        if overloaded:
            self.stride = min(self.maximum_stride, self.stride * 2)
            self._headroom = 0
        elif underloaded:
            self._headroom += 1
            if self._headroom >= 2:
                self.stride = max(1, self.stride // 2)
                self._headroom = 0
        else:
            self._headroom = 0
        return self.stride


class LatestValidationWorker:
    """Compute validation-only features outside display/analysis critical paths."""

    def __init__(self) -> None:
        self._jobs: queue.Queue[ValidationTask | None] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._latest = _empty_validation("warming_up")
        self._closed = False
        self.submitted_count = 0
        self.overwritten_count = 0
        self.completed_count = 0
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="pose-validation",
        )
        self._thread.start()

    def submit(self, task: ValidationTask) -> None:
        if self._closed:
            return
        try:
            self._jobs.put_nowait(task)
        except queue.Full:
            try:
                self._jobs.get_nowait()
                self.overwritten_count += 1
            except queue.Empty:
                pass
            try:
                self._jobs.put_nowait(task)
            except queue.Full:
                return
        self.submitted_count += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest)

    def reset(self) -> None:
        while True:
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            self._latest = _empty_validation("reset")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                pass
            self._jobs.put_nowait(None)
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while True:
            task = self._jobs.get()
            if task is None:
                return
            started = time.perf_counter()
            try:
                local_ground = build_local_ground_frame(
                    task.world_points,
                    body_coordinate_system=task.body_coordinate_system,
                    ground_estimation=task.ground_estimation,
                    foot_contact_evidence=task.foot_contact_evidence,
                    minimum_confidence=task.local_ground_minimum_confidence,
                )
                segments, metrics = build_biomechanical_representation(
                    task.world_points,
                    quality_points=task.quality_points,
                    body_coordinate_system=task.body_coordinate_system,
                    local_ground_frame=local_ground,
                    measurements=task.measurements,
                    minimum_quality=task.biomech_minimum_confidence,
                )
                result = {
                    "schema_version": 1,
                    "available": True,
                    "source_timestamp_ms": task.source_timestamp_ms,
                    "local_ground_frame": local_ground,
                    "segment_coordinates": segments,
                    "biomech_metrics": metrics,
                    "runs_off_render_thread": True,
                    "formal_threshold_replacement_allowed": False,
                }
            except Exception as exc:
                result = {
                    **_empty_validation("validation_failed"),
                    "source_timestamp_ms": task.source_timestamp_ms,
                    "error_type": type(exc).__name__,
                }
            result["validation_ms"] = (time.perf_counter() - started) * 1000.0
            with self._lock:
                self._latest = result
            self.completed_count += 1


def build_validation_task(
    result: object,
    kinematics: object,
    *,
    local_ground_minimum_confidence: float,
    biomech_minimum_confidence: float,
) -> ValidationTask:
    extra = getattr(result, "extra", {})
    world = extra.get("world_keypoints", ()) if isinstance(extra, Mapping) else ()
    return ValidationTask(
        source_timestamp_ms=float(getattr(result, "timestamp_ms", 0.0) or 0.0),
        world_points=tuple(world) if isinstance(world, Sequence) else (),
        quality_points=tuple(getattr(result, "keypoints", ())),
        body_coordinate_system=dict(getattr(kinematics, "body_coordinate_system", {})),
        ground_estimation=dict(getattr(kinematics, "ground_estimation", {})),
        foot_contact_evidence=dict(getattr(kinematics, "foot_contact_evidence", {})),
        measurements=dict(getattr(kinematics, "measurements", {})),
        local_ground_minimum_confidence=float(local_ground_minimum_confidence),
        biomech_minimum_confidence=float(biomech_minimum_confidence),
    )


def realtime_layer_contract(
    *,
    validation: Mapping[str, Any],
    validation_stride: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "display": {
            "source": "display_one_euro",
            "priority": "minimum_latency",
            "consumed_by_formal_rules": False,
        },
        "analysis": {
            "source": "analysis_one_euro",
            "priority": "stability",
            "prediction_allowed": False,
            "consumed_by_formal_rules": True,
        },
        "validation": {
            "source": "latest_validation_worker",
            "priority": "additional_3d_and_biomechanics",
            "source_timestamp_ms": validation.get("source_timestamp_ms"),
            "available": bool(validation.get("available")),
            "cadence_stride": max(1, int(validation_stride)),
            "validation_ms": _finite(validation.get("validation_ms"), 0.0),
            "consumed_by_formal_rules": False,
        },
        "renderer_waits_for_validation": False,
        "latest_frame_semantics_preserved": True,
        "formal_threshold_replacement_allowed": False,
    }


def _empty_validation(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "available": False,
        "reason": reason,
        "source_timestamp_ms": None,
        "validation_ms": 0.0,
        "local_ground_frame": {},
        "segment_coordinates": {},
        "biomech_metrics": {},
        "runs_off_render_thread": True,
        "formal_threshold_replacement_allowed": False,
    }


def _finite(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if isfinite(number) else default


__all__ = [
    "LatestValidationWorker",
    "ValidationBudgetGate",
    "ValidationTask",
    "build_validation_task",
    "realtime_layer_contract",
]
