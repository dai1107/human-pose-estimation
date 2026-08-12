"""Confidence-only temporal ground and foot-contact evidence."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from statistics import median
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class GroundEstimatorConfig:
    history_size: int = 31
    minimum_samples: int = 3
    minimum_contact_confidence: float = 0.55
    maximum_image_deviation: float = 0.025
    maximum_world_vertical_deviation_m: float = 0.08


class GroundEstimator:
    """Estimate a stable image-space ground from confidence-qualified feet.

    MediaPipe world landmarks are hip-relative, so the estimator deliberately
    treats its world-Y relation as auxiliary evidence, not a calibrated global
    ground plane.
    """

    def __init__(self, config: GroundEstimatorConfig | None = None) -> None:
        self.config = config or GroundEstimatorConfig()
        self.reset()

    def reset(self) -> None:
        capacity = max(3, int(self.config.history_size))
        self._image_ground_samples: deque[float] = deque(maxlen=capacity)
        self._world_support_samples: dict[str, deque[float]] = {
            "left": deque(maxlen=capacity),
            "right": deque(maxlen=capacity),
        }

    def update(
        self,
        image_points: Sequence[object],
        world_points: Sequence[object],
        foot_contact_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        image = _point_map(image_points)
        world = _point_map(world_points)
        observations: dict[str, dict[str, float | None]] = {}
        accepted_confidences: list[float] = []
        for side in ("left", "right"):
            names = (f"{side}_ankle", f"{side}_heel", f"{side}_foot_index")
            image_values = [_xy(image.get(name)) for name in names]
            world_values = [_xyz(world.get(name)) for name in names]
            image_valid = [point for point in image_values if point is not None]
            world_valid = [point for point in world_values if point is not None]
            contact = foot_contact_evidence.get(side)
            contact_confidence = (
                _number(contact.get("foot_contact_confidence"))
                if isinstance(contact, Mapping)
                else None
            )
            image_floor_y = (
                max(float(point[1]) for point in image_valid)
                if len(image_valid) == len(names)
                else None
            )
            world_support_y = (
                max(float(point[1]) for point in world_valid)
                if len(world_valid) == len(names)
                else None
            )
            accepted = (
                contact_confidence is not None
                and contact_confidence >= self.config.minimum_contact_confidence
                and image_floor_y is not None
            )
            if accepted:
                self._image_ground_samples.append(image_floor_y)
                accepted_confidences.append(contact_confidence)
                if world_support_y is not None:
                    self._world_support_samples[side].append(world_support_y)
            observations[side] = {
                "contact_confidence": contact_confidence,
                "image_floor_candidate_y": image_floor_y,
                "world_support_candidate_y": world_support_y,
            }

        ground_y = (
            median(self._image_ground_samples)
            if self._image_ground_samples
            else None
        )
        sample_count = len(self._image_ground_samples)
        dispersion = (
            median(abs(value - ground_y) for value in self._image_ground_samples)
            if ground_y is not None
            else None
        )
        count_score = min(
            1.0,
            sample_count / max(1, int(self.config.minimum_samples)),
        )
        stability_score = (
            0.0
            if dispersion is None
            else max(
                0.0,
                1.0
                - dispersion
                / max(self.config.maximum_image_deviation, 1e-8),
            )
        )
        contact_score = (
            float(np.mean(accepted_confidences))
            if accepted_confidences
            else (
                1.0
                if sample_count >= self.config.minimum_samples
                else 0.0
            )
        )
        ground_confidence = max(
            0.0,
            min(1.0, count_score * stability_score * contact_score),
        )
        ready = (
            ground_y is not None
            and sample_count >= self.config.minimum_samples
            and ground_confidence >= 0.50
        )

        fused: dict[str, Any] = {}
        for side, observation in observations.items():
            image_candidate = observation["image_floor_candidate_y"]
            world_candidate = observation["world_support_candidate_y"]
            contact_confidence = observation["contact_confidence"]
            two_d_proxy = (
                None
                if ground_y is None or image_candidate is None
                else max(
                    0.0,
                    1.0
                    - abs(image_candidate - ground_y)
                    / max(self.config.maximum_image_deviation, 1e-8),
                )
            )
            world_history = self._world_support_samples[side]
            world_baseline = median(world_history) if world_history else None
            world_relation = (
                None
                if world_baseline is None or world_candidate is None
                else max(
                    0.0,
                    1.0
                    - abs(world_candidate - world_baseline)
                    / max(
                        self.config.maximum_world_vertical_deviation_m,
                        1e-8,
                    ),
                )
            )
            components = [
                (0.45, contact_confidence),
                (0.35, two_d_proxy),
                (0.20, world_relation),
            ]
            available = [
                (weight, value)
                for weight, value in components
                if value is not None
            ]
            combined = (
                0.0
                if not ready or not available
                else ground_confidence
                * sum(weight * float(value) for weight, value in available)
                / sum(weight for weight, _ in available)
            )
            status = (
                "UNSURE"
                if not ready
                else "CONTACT_EVIDENCE"
                if combined >= 0.60
                else "NO_CONTACT_EVIDENCE"
                if combined <= 0.25
                else "UNSURE"
            )
            fused[side] = {
                "status": status,
                "confidence": max(0.0, min(1.0, combined)),
                "two_d_contact_proxy": two_d_proxy,
                "foot_contact_confidence_3d": contact_confidence,
                "world_vertical_relation": world_relation,
                "world_support_baseline_y": world_baseline,
            }

        return {
            "schema_version": 1,
            "status": "READY" if ready else "UNSURE",
            "ground_y_image_normalized": ground_y,
            "ground_confidence": ground_confidence,
            "sample_count": sample_count,
            "sample_median_absolute_deviation": dispersion,
            "contact_evidence": fused,
            "coordinate_semantics": (
                "image_ground_with_hip_relative_world_vertical_auxiliary"
            ),
            "evidence_only": True,
            "formal_floor_replacement_allowed": False,
            "formal_contact_replacement_allowed": False,
        }


def _point_map(points: Sequence[object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for point in points:
        name = point.get("name") if isinstance(point, Mapping) else getattr(point, "name", None)
        if name:
            result[str(name)] = point
    return result


def _value(point: object, name: str) -> object:
    return point.get(name) if isinstance(point, Mapping) else getattr(point, name, None)


def _xy(point: object | None) -> np.ndarray | None:
    return _array(point, ("x", "y"))


def _xyz(point: object | None) -> np.ndarray | None:
    return _array(point, ("x", "y", "z"))


def _array(point: object | None, axes: tuple[str, ...]) -> np.ndarray | None:
    if point is None:
        return None
    try:
        values = np.asarray([float(_value(point, axis)) for axis in axes], dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None
    return values if np.all(np.isfinite(values)) else None


def _number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if isfinite(result) else None


__all__ = ["GroundEstimator", "GroundEstimatorConfig"]
