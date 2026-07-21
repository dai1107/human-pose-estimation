"""Argument and import compatibility for the retired realtime loop."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from src.biomechanics.landmarks import LANDMARK_NAMES, coerce_landmark
from src.runtime_hand import infer_hand_side

BODY_JOINTS = frozenset(range(11, 33))
LANDMARK_PROFILES = {
    "full": frozenset(range(len(LANDMARK_NAMES))),
    "no-face": BODY_JOINTS,
    "upper-body": frozenset({11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}),
    "lower-body": frozenset({23, 24, 25, 26, 27, 28, 29, 30, 31, 32}),
}
HAND_OCCLUSION_RADIUS = 0.065
HAND_OCCLUSION_JUMP_THRESHOLD = 0.035


@dataclass(frozen=True)
class DrawLandmark:
    x: float
    y: float
    z: float
    visibility: float
    presence: float


def _is_near_hand_occlusion(
    landmark: DrawLandmark,
    occlusion_points: Sequence[DrawLandmark],
) -> bool:
    target = np.array([landmark.x, landmark.y], dtype=float)
    if not np.all(np.isfinite(target)):
        return False
    for hand_point in occlusion_points:
        if hand_point.visibility < 0.05 or hand_point.presence < 0.05:
            continue
        candidate = np.array([hand_point.x, hand_point.y], dtype=float)
        if np.all(np.isfinite(candidate)) and float(np.linalg.norm(target - candidate)) <= HAND_OCCLUSION_RADIUS:
            return True
    return False


class LandmarkSmoother:
    """Compatibility implementation retained for downstream imports only."""

    def __init__(self, alpha: float) -> None:
        self.alpha = max(0.0, min(1.0, alpha))
        self._previous: list[DrawLandmark] | None = None
        self._previous_timestamp_ms: int | None = None

    def reset(self) -> None:
        self._previous = None
        self._previous_timestamp_ms = None

    def smooth(
        self,
        landmarks: Sequence[object],
        timestamp_ms: int | None = None,
        occlusion_points: Sequence[DrawLandmark] | None = None,
        occlusion_guard_indices: frozenset[int] | None = None,
        self_occlusion_indices: frozenset[int] | None = None,
    ) -> list[DrawLandmark]:
        current = [
            DrawLandmark(
                x=point.x,
                y=point.y,
                z=point.z,
                visibility=point.visibility,
                presence=point.presence,
            )
            for point in (coerce_landmark(raw) for raw in landmarks)
        ]
        all_occlusion_points = list(occlusion_points or ())
        if self_occlusion_indices:
            all_occlusion_points.extend(current[index] for index in sorted(self_occlusion_indices) if index < len(current))
        if self.alpha <= 0.0 or self._previous is None or len(self._previous) != len(current):
            self._previous = current
            self._previous_timestamp_ms = timestamp_ms
            return current

        dt = 1.0 / 30.0
        if timestamp_ms is not None and self._previous_timestamp_ms is not None:
            delta_ms = timestamp_ms - self._previous_timestamp_ms
            if 0 < delta_ms <= 1000:
                dt = delta_ms / 1000.0
        smoothed: list[DrawLandmark] = []
        for index, (old, new) in enumerate(zip(self._previous, current)):
            confidence = max(0.0, min(1.0, (new.visibility + new.presence) / 2.0))
            displacement = float(np.linalg.norm(np.array([new.x - old.x, new.y - old.y, new.z - old.z], dtype=float)))
            speed = displacement / max(dt, 1e-3)
            dynamic_alpha = min(0.95, self.alpha + min(0.3, speed * 0.08))
            if displacement < 0.006:
                dynamic_alpha *= 0.65
            if confidence < 0.55:
                dynamic_alpha *= max(0.15, confidence)
            if displacement > 0.30 and confidence < 0.55:
                dynamic_alpha = 0.08
            if (
                occlusion_guard_indices is not None
                and index in occlusion_guard_indices
                and displacement > HAND_OCCLUSION_JUMP_THRESHOLD
                and _is_near_hand_occlusion(new, all_occlusion_points)
            ):
                dynamic_alpha = min(dynamic_alpha, 0.04)
            dynamic_alpha = max(0.03, min(0.95, dynamic_alpha))
            keep = 1.0 - dynamic_alpha
            smoothed.append(
                DrawLandmark(
                    x=old.x * keep + new.x * dynamic_alpha,
                    y=old.y * keep + new.y * dynamic_alpha,
                    z=old.z * keep + new.z * dynamic_alpha,
                    visibility=new.visibility,
                    presence=new.presence,
                )
            )
        self._previous = smoothed
        self._previous_timestamp_ms = timestamp_ms
        return smoothed


def build_legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deprecated MediaPipe realtime compatibility CLI.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=60.0)
    parser.add_argument("--camera-fourcc", default="MJPG")
    parser.add_argument("--model", default="models/pose_landmarker_full.task")
    parser.add_argument("--landmark-profile", default="no-face", choices=tuple(LANDMARK_PROFILES))
    parser.add_argument("--include-landmarks", default="")
    parser.add_argument("--exclude-landmarks", default="")
    parser.add_argument("--show-hands", action="store_true")
    parser.add_argument("--hand-model", default="models/hand_landmarker.task")
    parser.add_argument("--hand-detect-width", type=int, default=416)
    parser.add_argument("--max-hand-detect-fps", type=float, default=18.0)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--save-dir", default="outputs")
    parser.add_argument("--metrics-overlay", action="store_true")
    parser.add_argument("--session-autostart", action="store_true")
    parser.add_argument("--camera-view", default="unknown", choices=("side", "front", "front_left", "front_right", "unknown"))
    parser.add_argument("--detect-width", type=int, default=480)
    parser.add_argument("--max-detect-fps", type=float, default=30.0)
    parser.add_argument("--max-pending-ms", type=int, default=180)
    parser.add_argument("--max-result-lag-ms", type=int, default=280)
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument("--plot-on-save", dest="plot_on_save", action="store_true")
    plot_group.add_argument("--no-plot-on-save", dest="plot_on_save", action="store_false")
    parser.add_argument("--smoothing", nargs="?", type=float, const=0.65, default=0.65)
    mirror_group = parser.add_mutually_exclusive_group()
    mirror_group.add_argument("--mirror", dest="mirror", action="store_true")
    mirror_group.add_argument("--no-mirror", dest="mirror", action="store_false")
    parser.set_defaults(mirror=True, plot_on_save=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_legacy_parser().parse_args(argv)


def translate_legacy_args(args: argparse.Namespace) -> list[str]:
    """Translate supported legacy flags to the single maintained runtime."""
    translated = [
        "--backend", "mediapipe",
        "--camera", str(args.camera),
        "--width", str(args.width),
        "--height", str(args.height),
        "--camera-fps", str(args.camera_fps),
        "--camera-fourcc", str(args.camera_fourcc),
        "--model", str(args.model),
        "--landmark-profile", str(args.landmark_profile),
        "--hand-model", str(args.hand_model),
        "--hand-detect-width", str(args.hand_detect_width),
        "--max-hand-detect-fps", str(args.max_hand_detect_fps),
        "--max-hands", str(args.max_hands),
        "--save-dir", str(args.save_dir),
        "--camera-view", str(args.camera_view),
    ]
    translated.append("--mirror" if args.mirror else "--no-mirror")
    translated.extend(["--smoothing", "ema", "--ema-alpha", str(args.smoothing)] if args.smoothing > 0 else ["--smoothing", "none"])
    for enabled, flag in (
        (args.show_hands, "--show-hands"),
        (args.metrics_overlay, "--metrics-overlay"),
        (args.session_autostart, "--session-autostart"),
    ):
        if enabled:
            translated.append(flag)
    if args.record:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        translated.extend(["--record", str(Path(args.save_dir) / "recordings" / f"legacy_{stamp}.mp4")])
    return translated
