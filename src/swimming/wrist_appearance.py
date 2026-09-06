"""Generic online wrist ROI appearance evidence for marked swimming clips."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import cv2
import numpy as np

from src.swimming.wrist_tracking import SIDES, Side, SwimWristTrackerConfig


@dataclass(frozen=True, slots=True)
class WristAppearanceDescriptor:
    vector: np.ndarray
    saturation_mean: float
    radius_px: int
    sample_count: int


@dataclass(frozen=True, slots=True)
class AppearanceUpdate:
    side: Side
    updated: bool
    reason: str
    update_count: int


def extract_wrist_appearance(
    frame: np.ndarray,
    *,
    wrist: tuple[float, float],
    elbow: tuple[float, float],
    roi_forearm_ratio: float = 0.25,
) -> WristAppearanceDescriptor | None:
    """Extract HSV/Lab histograms from a forearm-scaled circular wrist ROI."""

    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] < 3:
        return None
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return None
    wrist_px = np.asarray([wrist[0] * width, wrist[1] * height], dtype=np.float64)
    elbow_px = np.asarray([elbow[0] * width, elbow[1] * height], dtype=np.float64)
    if not np.isfinite(wrist_px).all() or not np.isfinite(elbow_px).all():
        return None
    forearm = float(np.linalg.norm(wrist_px - elbow_px))
    if forearm < 6.0:
        return None
    radius = int(round(np.clip(roi_forearm_ratio * forearm, 4.0, 36.0)))
    cx, cy = int(round(wrist_px[0])), int(round(wrist_px[1]))
    x1, x2 = max(0, cx - radius), min(width, cx + radius + 1)
    y1, y2 = max(0, cy - radius), min(height, cy + radius + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    yy, xx = np.ogrid[y1:y2, x1:x2]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    pixels = frame[y1:y2, x1:x2][mask]
    if pixels.shape[0] < 20:
        return None
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    histograms = [
        _histogram(hsv[:, 0], bins=16, maximum=180.0),
        _histogram(hsv[:, 1], bins=8, maximum=256.0),
        _histogram(hsv[:, 2], bins=8, maximum=256.0),
        _histogram(lab[:, 0], bins=8, maximum=256.0),
        _histogram(lab[:, 1], bins=8, maximum=256.0),
        _histogram(lab[:, 2], bins=8, maximum=256.0),
    ]
    vector = np.concatenate(histograms).astype(np.float64)
    vector /= max(float(np.sum(vector)), 1e-12)
    return WristAppearanceDescriptor(
        vector=vector,
        saturation_mean=float(np.mean(hsv[:, 1])) / 255.0,
        radius_px=radius,
        sample_count=int(pixels.shape[0]),
    )


class WristAppearanceModel:
    """Confidence-gated EMA appearance prototype for one anatomical track."""

    def __init__(self, side: Side, config: SwimWristTrackerConfig) -> None:
        self.side = side
        self.config = config
        self.prototype: np.ndarray | None = None
        self.saturation_mean: float | None = None
        self.update_count = 0

    @property
    def initialized(self) -> bool:
        return self.prototype is not None

    def cost(self, descriptor: WristAppearanceDescriptor | None) -> float | None:
        if descriptor is None or self.prototype is None:
            return None
        similarity = float(
            np.dot(self.prototype, descriptor.vector)
            / max(
                float(np.linalg.norm(self.prototype))
                * float(np.linalg.norm(descriptor.vector)),
                1e-12,
            )
        )
        saturation_delta = abs(
            float(self.saturation_mean or 0.0) - descriptor.saturation_mean
        )
        return float(np.clip(0.82 * (1.0 - similarity) + 0.18 * saturation_delta, 0.0, 2.0))

    def update(
        self,
        descriptor: WristAppearanceDescriptor | None,
        *,
        identity_confidence: float,
        visibility: float,
    ) -> AppearanceUpdate:
        if descriptor is None:
            return AppearanceUpdate(self.side, False, "appearance_missing", self.update_count)
        if identity_confidence < self.config.appearance_minimum_identity_confidence:
            return AppearanceUpdate(
                self.side, False, "identity_confidence_low", self.update_count
            )
        if visibility < self.config.appearance_minimum_visibility:
            return AppearanceUpdate(self.side, False, "wrist_visibility_low", self.update_count)
        alpha = max(0.0, min(1.0, self.config.appearance_ema_alpha))
        if self.prototype is None:
            self.prototype = descriptor.vector.astype(np.float64, copy=True)
            self.saturation_mean = descriptor.saturation_mean
            reason = "appearance_initialized"
        else:
            self.prototype = (
                (1.0 - alpha) * self.prototype + alpha * descriptor.vector
            )
            self.prototype /= max(float(np.sum(self.prototype)), 1e-12)
            self.saturation_mean = (
                (1.0 - alpha) * float(self.saturation_mean)
                + alpha * descriptor.saturation_mean
            )
            reason = "appearance_ema_updated"
        self.update_count += 1
        return AppearanceUpdate(self.side, True, reason, self.update_count)


class WristAppearanceBank:
    """Own left/right prototypes and produce candidate-to-track costs."""

    def __init__(self, config: SwimWristTrackerConfig | None = None) -> None:
        self.config = config or SwimWristTrackerConfig()
        self.models = {
            side: WristAppearanceModel(side, self.config) for side in SIDES
        }

    def costs(
        self,
        descriptors: dict[Side, WristAppearanceDescriptor | None],
    ) -> dict[tuple[Side, Side], float]:
        output: dict[tuple[Side, Side], float] = {}
        for candidate_side, descriptor in descriptors.items():
            for track_side, model in self.models.items():
                value = model.cost(descriptor)
                if value is not None and isfinite(value):
                    output[(candidate_side, track_side)] = value
        return output

    def update(
        self,
        side: Side,
        descriptor: WristAppearanceDescriptor | None,
        *,
        identity_confidence: float,
        visibility: float,
    ) -> AppearanceUpdate:
        return self.models[side].update(
            descriptor,
            identity_confidence=identity_confidence,
            visibility=visibility,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            side: {
                "initialized": model.initialized,
                "update_count": model.update_count,
                "saturation_mean": model.saturation_mean,
            }
            for side, model in self.models.items()
        }


def _histogram(values: np.ndarray, *, bins: int, maximum: float) -> np.ndarray:
    histogram, _ = np.histogram(values, bins=bins, range=(0.0, maximum))
    resolved = histogram.astype(np.float64)
    return resolved / max(float(np.sum(resolved)), 1.0)


__all__ = [
    "AppearanceUpdate",
    "WristAppearanceBank",
    "WristAppearanceDescriptor",
    "WristAppearanceModel",
    "extract_wrist_appearance",
]
