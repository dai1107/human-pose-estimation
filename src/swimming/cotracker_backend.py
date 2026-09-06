"""Optional offline CoTracker adapter for swimming wrist experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.swimming.wrist_tracking import SwimWristTrackerConfig


@dataclass(frozen=True, slots=True)
class CoTrackerAvailability:
    available: bool
    reason: str
    device: str | None
    source: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "device": self.device,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class CoTrackerWindowResult:
    tracks_normalized: np.ndarray
    visibility: np.ndarray
    query_count: int
    frame_count: int
    available: bool
    reason: str


class CoTrackerOfflineBackend:
    """Load official CoTracker only when explicitly available or authorized."""

    def __init__(
        self,
        config: SwimWristTrackerConfig | None = None,
        *,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        model: object | None = None,
        allow_torch_hub_download: bool = False,
    ) -> None:
        self.config = config or SwimWristTrackerConfig()
        self.model = model
        self._torch: Any | None = None
        self.device = device
        self.availability = CoTrackerAvailability(
            False, "not_initialized", None, None
        )
        try:
            import torch
        except Exception as exc:
            self.availability = CoTrackerAvailability(
                False,
                f"torch_unavailable:{type(exc).__name__}",
                None,
                None,
            )
            return
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model is not None:
            self.model = self._prepare_model(model)
            self.availability = CoTrackerAvailability(
                True, "available", self.device, "injected_model"
            )
            return
        checkpoint_path = Path(checkpoint) if checkpoint is not None else None
        try:
            from cotracker.predictor import CoTrackerOfflinePredictor

            if checkpoint_path is None or not checkpoint_path.is_file():
                self.availability = CoTrackerAvailability(
                    False,
                    "cotracker_checkpoint_missing",
                    self.device,
                    "installed_package",
                )
                return
            loaded = CoTrackerOfflinePredictor(checkpoint=str(checkpoint_path))
            self.model = self._prepare_model(loaded)
            self.availability = CoTrackerAvailability(
                True,
                "available",
                self.device,
                str(checkpoint_path.resolve()),
            )
            return
        except ImportError:
            if not allow_torch_hub_download:
                self.availability = CoTrackerAvailability(
                    False,
                    "cotracker_package_and_weights_unavailable",
                    self.device,
                    None,
                )
                return
        except Exception as exc:
            self.availability = CoTrackerAvailability(
                False,
                f"cotracker_load_failed:{type(exc).__name__}:{exc}",
                self.device,
                str(checkpoint_path) if checkpoint_path else None,
            )
            return
        try:
            loaded = torch.hub.load(
                "facebookresearch/co-tracker",
                "cotracker3_offline",
                trust_repo=True,
            )
            self.model = self._prepare_model(loaded)
            self.availability = CoTrackerAvailability(
                True,
                "available",
                self.device,
                "torch_hub:facebookresearch/co-tracker:cotracker3_offline",
            )
        except Exception as exc:
            self.availability = CoTrackerAvailability(
                False,
                f"torch_hub_load_failed:{type(exc).__name__}:{exc}",
                self.device,
                "torch_hub:facebookresearch/co-tracker:cotracker3_offline",
            )

    def track_window(
        self,
        frames_bgr: list[np.ndarray],
        queries: np.ndarray,
    ) -> CoTrackerWindowResult:
        """Track ``[time, x, y]`` pixel queries over one bounded frame window."""

        if frames_bgr:
            first_shape = frames_bgr[0].shape[:2]
            if any(frame.shape[:2] != first_shape for frame in frames_bgr):
                raise ValueError("all CoTracker window frames must have the same shape")
        if not self.availability.available or self.model is None or self._torch is None:
            return CoTrackerWindowResult(
                np.empty((0, 0, 2), dtype=np.float32),
                np.empty((0, 0), dtype=np.float32),
                int(len(queries)),
                int(len(frames_bgr)),
                False,
                self.availability.reason,
            )
        if not frames_bgr or queries.size == 0:
            return CoTrackerWindowResult(
                np.empty((len(frames_bgr), 0, 2), dtype=np.float32),
                np.empty((len(frames_bgr), 0), dtype=np.float32),
                0,
                len(frames_bgr),
                False,
                "frames_or_queries_missing",
            )
        height, width = first_shape
        maximum_width = max(64, int(self.config.cotracker_maximum_width))
        scale = min(1.0, maximum_width / max(1.0, float(width)))
        target_width = max(1, int(round(width * scale)))
        target_height = max(1, int(round(height * scale)))
        rgb_frames = [
            cv2.cvtColor(
                cv2.resize(frame, (target_width, target_height)),
                cv2.COLOR_BGR2RGB,
            )
            for frame in frames_bgr
        ]
        video = np.stack(rgb_frames).astype(np.float32)
        query_values = np.asarray(queries, dtype=np.float32).copy()
        if query_values.ndim != 2 or query_values.shape[1] != 3:
            raise ValueError("CoTracker queries must have shape [N, 3]")
        query_values[:, 1:] *= scale
        torch = self._torch
        video_tensor = (
            torch.from_numpy(video)
            .permute(0, 3, 1, 2)
            .unsqueeze(0)
            .to(self.device)
        )
        query_tensor = torch.from_numpy(query_values).unsqueeze(0).to(self.device)
        try:
            with torch.inference_mode():
                prediction = self.model(video_tensor, queries=query_tensor)
            tracks, visibility = prediction[:2]
            track_values = tracks.detach().float().cpu().numpy()[0]
            visibility_values = visibility.detach().float().cpu().numpy()[0]
        except Exception as exc:
            return CoTrackerWindowResult(
                np.empty((0, 0, 2), dtype=np.float32),
                np.empty((0, 0), dtype=np.float32),
                len(query_values),
                len(frames_bgr),
                False,
                f"cotracker_inference_failed:{type(exc).__name__}:{exc}",
            )
        track_values[..., 0] /= max(float(target_width), 1.0)
        track_values[..., 1] /= max(float(target_height), 1.0)
        return CoTrackerWindowResult(
            tracks_normalized=track_values,
            visibility=visibility_values,
            query_count=len(query_values),
            frame_count=len(frames_bgr),
            available=True,
            reason="accepted",
        )

    def _prepare_model(self, model: object) -> object:
        if hasattr(model, "to"):
            model = model.to(self.device)
        if hasattr(model, "eval"):
            model.eval()
        return model


__all__ = [
    "CoTrackerAvailability",
    "CoTrackerOfflineBackend",
    "CoTrackerWindowResult",
]
