"""Extract and render one selected anatomical wrist trajectory.

The source video is rotated clockwise for inference because the bundled pose
models are substantially more reliable when the prone swimmer is presented
upright.  Coordinates are transformed back to the source image for the audit
video, while MediaPipe world landmarks are expressed relative to the left
shoulder for the reference-style side and overhead plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backends.mediapipe_backend import MediaPipeBackend
from src.utils.draw_utils import put_text
from src.utils.smoothing import OneEuroValueFilter


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass(frozen=True)
class WristbandCalibration:
    """Jointly learned thresholds for the same dark wristband in all clips."""

    score_floor: float
    margin_floor: float
    sample_count: int
    source_videos: tuple[str, ...]


class LeftArmIdentityGate:
    """Conservatively reject wrist points that break the selected-arm identity."""

    def __init__(self, *, min_confidence: float) -> None:
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.upper_arm_history: deque[float] = deque(maxlen=90)
        self.forearm_history: deque[float] = deque(maxlen=90)
        self.previous_relative: np.ndarray | None = None
        self.previous_frame: int | None = None

    def evaluate(
        self,
        image_points: Sequence[object],
        world_points: Sequence[object],
        *,
        frame_index: int,
        side: str = "left",
    ) -> tuple[bool, str]:
        selected = "right" if str(side).lower() == "right" else "left"
        opposite = "left" if selected == "right" else "right"
        arm_names = tuple(
            f"{arm_side}_{joint}"
            for arm_side in (selected, opposite)
            for joint in ("shoulder", "elbow", "wrist")
        )
        image = {name: _point_by_name(image_points, name) for name in arm_names}
        world = {name: _point_by_name(world_points, name) for name in arm_names}
        shoulder_name = f"{selected}_shoulder"
        elbow_name = f"{selected}_elbow"
        wrist_name = f"{selected}_wrist"
        required = (shoulder_name, elbow_name, wrist_name)
        if not all(_finite_point(world[name]) and _finite_point(image[name]) for name in required):
            return False, "left_chain_missing"
        confidence = min(float(getattr(world[name], "confidence", 0.0)) for name in required)
        if confidence < self.min_confidence:
            return False, "left_chain_low_confidence"

        ls = _xyz(world[shoulder_name])
        le = _xyz(world[elbow_name])
        lw = _xyz(world[wrist_name])
        upper = float(np.linalg.norm(le - ls))
        forearm = float(np.linalg.norm(lw - le))
        reach = float(np.linalg.norm(lw - ls))
        if not 0.12 <= upper <= 0.55:
            return False, "upper_arm_length_outlier"
        if not 0.11 <= forearm <= 0.52:
            return False, "forearm_length_outlier"
        if not 0.08 <= reach <= 0.88:
            return False, "shoulder_wrist_length_outlier"
        ratio = forearm / max(upper, 1e-6)
        if not 0.42 <= ratio <= 1.85:
            return False, "arm_segment_ratio_outlier"

        if len(self.upper_arm_history) >= 12:
            upper_median = float(np.median(tuple(self.upper_arm_history)))
            forearm_median = float(np.median(tuple(self.forearm_history)))
            if not 0.58 <= upper / max(upper_median, 1e-6) <= 1.72:
                return False, "upper_arm_history_jump"
            if not 0.55 <= forearm / max(forearm_median, 1e-6) <= 1.80:
                return False, "forearm_history_jump"

        opposite_shoulder_name = f"{opposite}_shoulder"
        opposite_elbow_name = f"{opposite}_elbow"
        opposite_wrist_name = f"{opposite}_wrist"
        bilateral = (
            opposite_shoulder_name,
            opposite_elbow_name,
            opposite_wrist_name,
        )
        if all(_finite_point(world[name]) for name in bilateral):
            rs = _xyz(world[opposite_shoulder_name])
            re = _xyz(world[opposite_elbow_name])
            rw = _xyz(world[opposite_wrist_name])
            canonical_elbows = float(np.linalg.norm(le - ls) + np.linalg.norm(re - rs))
            swapped_elbows = float(np.linalg.norm(re - ls) + np.linalg.norm(le - rs))
            if canonical_elbows > swapped_elbows * 1.18 and canonical_elbows - swapped_elbows > 0.07:
                return False, "left_right_elbow_identity_swap"
            canonical_wrists = float(np.linalg.norm(lw - le) + np.linalg.norm(rw - re))
            swapped_wrists = float(np.linalg.norm(rw - le) + np.linalg.norm(lw - re))
            if canonical_wrists > swapped_wrists * 1.16 and canonical_wrists - swapped_wrists > 0.07:
                return False, "left_right_wrist_identity_swap"

        image_wrist = image[wrist_name]
        image_elbow = image[elbow_name]
        forearm_image = _image_distance(image_wrist, image_elbow)
        if forearm_image > 0.34:
            return False, "image_forearm_too_long"

        left_hip = _point_by_name(image_points, "left_hip")
        right_hip = _point_by_name(image_points, "right_hip")
        other_shoulder = image.get(opposite_shoulder_name)
        if all(
            _finite_point(point)
            for point in (image[shoulder_name], other_shoulder, left_hip, right_hip)
        ):
            shoulder_axis = 0.5 * (
                float(getattr(image[shoulder_name], "y"))
                + float(getattr(other_shoulder, "y"))
            )
            hip_axis = 0.5 * (
                float(getattr(left_hip, "y")) + float(getattr(right_hip, "y"))
            )
            torso_length = abs(hip_axis - shoulder_axis)
            # The swimmer is rotated head-up for inference. A wrist may finish
            # beside the hip, but it cannot continue down the longitudinal body
            # axis into the thigh/calf/foot region.
            if torso_length >= 0.04 and float(getattr(image_wrist, "y")) > (
                hip_axis + 0.28 * torso_length + 0.018
            ):
                return False, "wrist_below_hip_region"

        lower_body_names = (
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        )
        for lower_name in lower_body_names:
            lower_point = _point_by_name(image_points, lower_name)
            if not _finite_point(lower_point):
                continue
            lower_distance = _image_distance(image_wrist, lower_point)
            if lower_distance < max(0.045, 0.52 * forearm_image):
                return False, "wrist_collapsed_to_foot"

        relative = lw - ls
        if self.previous_relative is not None and self.previous_frame is not None:
            gap = frame_index - self.previous_frame
            if 0 < gap <= 12:
                displacement = float(np.linalg.norm(relative - self.previous_relative))
                maximum = 0.18 + 0.075 * gap
                if displacement > maximum:
                    return False, "left_wrist_temporal_jump"

        self.upper_arm_history.append(upper)
        self.forearm_history.append(forearm)
        self.previous_relative = relative
        self.previous_frame = int(frame_index)
        return True, "accepted"


class RtmwLeftIdentityGate:
    """Validate RTMW's selected arm before using it as a handedness oracle."""

    def __init__(self) -> None:
        self.previous_relative: np.ndarray | None = None
        self.previous_frame: int | None = None

    def evaluate(
        self,
        points: Sequence[object],
        *,
        frame_index: int,
        side: str = "left",
    ) -> tuple[bool, str]:
        selected = "right" if str(side).lower() == "right" else "left"
        opposite = "left" if selected == "right" else "right"
        by_name = {
            name: _point_by_name(points, name)
            for name in tuple(
                f"{arm_side}_{joint}"
                for arm_side in (selected, opposite)
                for joint in ("shoulder", "elbow", "wrist")
            ) + ("left_hip", "right_hip")
        }
        shoulder_name = f"{selected}_shoulder"
        elbow_name = f"{selected}_elbow"
        wrist_name = f"{selected}_wrist"
        other_shoulder_name = f"{opposite}_shoulder"
        other_elbow_name = f"{opposite}_elbow"
        other_wrist_name = f"{opposite}_wrist"
        required = (shoulder_name, elbow_name, wrist_name)
        if not all(_finite_point(by_name[name]) for name in required):
            return False, "rtmw_left_chain_missing"
        confidence = min(float(getattr(by_name[name], "confidence", 0.0)) for name in required)
        if confidence < 0.30:
            return False, "rtmw_left_chain_low_confidence"
        shoulder = by_name[shoulder_name]
        elbow = by_name[elbow_name]
        wrist = by_name[wrist_name]
        upper = _image_distance(shoulder, elbow)
        forearm = _image_distance(elbow, wrist)
        if not 0.012 <= upper <= 0.30 or not 0.012 <= forearm <= 0.30:
            return False, "rtmw_arm_length_outlier"
        if all(_finite_point(by_name[name]) for name in (other_elbow_name, other_wrist_name)):
            canonical = forearm + _image_distance(by_name[other_elbow_name], by_name[other_wrist_name])
            swapped = _image_distance(elbow, by_name[other_wrist_name]) + _image_distance(
                by_name[other_elbow_name], wrist
            )
            if canonical > swapped * 1.15 and canonical - swapped > 0.045:
                return False, "rtmw_wrist_identity_swap"
        if all(_finite_point(by_name[name]) for name in ("left_hip", "right_hip", other_shoulder_name)):
            shoulder_axis = 0.5 * (
                float(getattr(shoulder, "y"))
                + float(getattr(by_name[other_shoulder_name], "y"))
            )
            hip_axis = 0.5 * (
                float(getattr(by_name["left_hip"], "y"))
                + float(getattr(by_name["right_hip"], "y"))
            )
            torso = abs(hip_axis - shoulder_axis)
            if torso >= 0.04 and float(getattr(wrist, "y")) > hip_axis + 0.30 * torso + 0.02:
                return False, "rtmw_wrist_below_hip"
        relative = np.asarray(
            [
                float(getattr(wrist, "x")) - float(getattr(shoulder, "x")),
                float(getattr(wrist, "y")) - float(getattr(shoulder, "y")),
            ],
            dtype=np.float64,
        )
        if self.previous_relative is not None and self.previous_frame is not None:
            gap = frame_index - self.previous_frame
            if 0 < gap <= 10 and float(np.linalg.norm(relative - self.previous_relative)) > 0.13 + 0.045 * gap:
                return False, "rtmw_left_wrist_temporal_jump"
        self.previous_relative = relative
        self.previous_frame = int(frame_index)
        return True, "accepted"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze and render one selected swimmer wrist trajectory."
    )
    parser.add_argument("input_video", type=Path)
    parser.add_argument(
        "--side",
        choices=("left", "right"),
        required=True,
        help="Anatomical wrist to track; the opposite wrist is excluded.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/swimming_left_hand"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/pose_landmarker_full.task"),
    )
    parser.add_argument(
        "--rtmw-model",
        type=Path,
        default=Path("models/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx"),
    )
    parser.add_argument(
        "--no-rtmw",
        action="store_true",
        help="Disable RTMW handedness cross-checking (lower precision).",
    )
    parser.add_argument(
        "--trail-seconds",
        type=float,
        default=2.6,
        help="Length of the moving trajectory trail in the rendered videos.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.08,
        help="Lowest wrist confidence retained before interpolation.",
    )
    parser.add_argument(
        "--joint-calibration-video",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional wristband-marked clip used jointly with the input to learn "
            "one shared dark-band appearance threshold. May be repeated."
        ),
    )
    return parser


def _open_video(path: Path) -> tuple[cv2.VideoCapture, VideoInfo]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open input video: {path}")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if width <= 0 or height <= 0 or fps <= 0.0 or frame_count <= 0:
        capture.release()
        raise RuntimeError(
            f"invalid video metadata: {width}x{height}, fps={fps}, frames={frame_count}"
        )
    return capture, VideoInfo(width, height, fps, frame_count)


def _point_by_name(points: Sequence[object], name: str) -> object | None:
    return next((point for point in points if getattr(point, "name", "") == name), None)


def _finite_point(point: object | None) -> bool:
    return bool(
        point is not None
        and math.isfinite(float(getattr(point, "x", math.nan)))
        and math.isfinite(float(getattr(point, "y", math.nan)))
        and math.isfinite(float(getattr(point, "z", math.nan)))
    )


def _xyz(point: object) -> np.ndarray:
    return np.asarray(
        [
            float(getattr(point, "x")),
            float(getattr(point, "y")),
            float(getattr(point, "z")),
        ],
        dtype=np.float64,
    )


def _image_distance(first: object, second: object) -> float:
    return math.hypot(
        float(getattr(first, "x")) - float(getattr(second, "x")),
        float(getattr(first, "y")) - float(getattr(second, "y")),
    )


def _wristband_observation(
    frame: np.ndarray,
    wrist: object | None,
    elbow: object | None,
) -> tuple[float, tuple[float, float] | None]:
    """Score and localize a dark wristband around one pose wrist candidate.

    The score is illumination-relative: a narrow dark patch must contrast with
    both the adjacent forearm and its local annulus.  This works above and
    below the waterline without relying on a fixed RGB color.
    """

    if not (_finite_point(wrist) and _finite_point(elbow)):
        return math.nan, None
    height, width = frame.shape[:2]
    wrist_px = np.asarray(
        [float(getattr(wrist, "x")) * width, float(getattr(wrist, "y")) * height],
        dtype=np.float64,
    )
    elbow_px = np.asarray(
        [float(getattr(elbow, "x")) * width, float(getattr(elbow, "y")) * height],
        dtype=np.float64,
    )
    forearm = elbow_px - wrist_px
    forearm_length = float(np.linalg.norm(forearm))
    if not 10.0 <= forearm_length <= 0.42 * math.hypot(width, height):
        return math.nan, None
    direction = forearm / max(forearm_length, 1e-6)
    perpendicular = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    value = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
    radius = int(round(max(4.0, min(18.0, 0.105 * forearm_length))))

    def pixels_in_disc(center: np.ndarray, disc_radius: int) -> np.ndarray:
        cx, cy = (int(round(float(center[0]))), int(round(float(center[1]))))
        x1, x2 = max(0, cx - disc_radius), min(width, cx + disc_radius + 1)
        y1, y2 = max(0, cy - disc_radius), min(height, cy + disc_radius + 1)
        if x2 <= x1 or y2 <= y1:
            return np.asarray([], dtype=np.float64)
        yy, xx = np.ogrid[y1:y2, x1:x2]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= disc_radius**2
        return value[y1:y2, x1:x2][mask].astype(np.float64)

    best_score = -math.inf
    best_center: np.ndarray | None = None
    # Search a compact region because pose wrists often land on the hand while
    # the physical band sits a few pixels proximally on the forearm.
    for along in (-0.04, 0.03, 0.10, 0.17):
        for across in (-0.055, 0.0, 0.055):
            center = wrist_px + along * forearm_length * direction + across * forearm_length * perpendicular
            core = pixels_in_disc(center, radius)
            proximal = pixels_in_disc(
                center + max(2.7 * radius, 0.25 * forearm_length) * direction,
                radius,
            )
            local = pixels_in_disc(center, max(radius + 3, int(round(2.1 * radius))))
            if core.size < 20 or proximal.size < 12 or local.size < 40:
                continue
            core_p30 = float(np.percentile(core, 30.0))
            core_median = float(np.median(core))
            context = max(float(np.median(proximal)), float(np.percentile(local, 68.0)))
            contrast = max(0.0, context - core_p30) / 255.0
            darkness = max(0.0, 1.0 - core_median / 255.0)
            dark_cutoff = min(105.0, context - 20.0)
            dark_fraction = float(np.mean(core <= dark_cutoff)) if dark_cutoff > 0.0 else 0.0
            score = 0.52 * contrast + 0.30 * darkness + 0.18 * dark_fraction
            if score > best_score:
                best_score = score
                best_center = center
    if best_center is None or not math.isfinite(best_score):
        return math.nan, None
    normalized = (
        max(0.0, min(1.0, float(best_center[0]) / width)),
        max(0.0, min(1.0, float(best_center[1]) / height)),
    )
    return float(best_score), normalized


def calibrate_wristband(
    video_paths: Sequence[Path],
    model_path: Path,
    *,
    sample_stride: int = 12,
) -> WristbandCalibration:
    """Pool both marked clips to learn one conservative wristband threshold."""

    top_scores: list[float] = []
    margins: list[float] = []
    sampled = 0
    backend = MediaPipeBackend(
        model_path,
        min_pose_detection_confidence=0.20,
        min_pose_presence_confidence=0.20,
        min_tracking_confidence=0.20,
    )
    try:
        for path in video_paths:
            capture, info = _open_video(path)
            try:
                frame_index = 0
                while frame_index < info.frame_count:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = capture.read()
                    if not ok:
                        break
                    rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    result = backend.detect(
                        rotated, int(round(frame_index * 1000.0 / info.fps))
                    )
                    scores: list[float] = []
                    for candidate_side in ("left", "right"):
                        score, _center = _wristband_observation(
                            rotated,
                            _point_by_name(result.keypoints, f"{candidate_side}_wrist"),
                            _point_by_name(result.keypoints, f"{candidate_side}_elbow"),
                        )
                        if math.isfinite(score):
                            scores.append(score)
                    if scores:
                        ordered = sorted(scores, reverse=True)
                        top_scores.append(ordered[0])
                        if len(ordered) >= 2:
                            margins.append(ordered[0] - ordered[1])
                    sampled += 1
                    frame_index += max(1, int(sample_stride))
            finally:
                capture.release()
    finally:
        backend.close()
    if len(top_scores) < 20:
        raise RuntimeError("too few wrist candidates to calibrate the shared wristband")
    score_floor = max(0.30, float(np.percentile(top_scores, 52.0)))
    positive_margins = np.asarray([value for value in margins if value > 0.0], dtype=np.float64)
    if positive_margins.size:
        margin_floor = max(0.035, 0.62 * float(np.percentile(positive_margins, 52.0)))
    else:
        margin_floor = 0.045
    return WristbandCalibration(
        score_floor=score_floor,
        margin_floor=margin_floor,
        sample_count=len(top_scores),
        source_videos=tuple(str(path.resolve()) for path in video_paths),
    )


def _rotated_to_source(x: float, y: float) -> tuple[float, float]:
    """Map normalized coordinates from a clockwise-rotated frame to source."""

    return float(y), float(1.0 - x)


def _nan_series(frame_count: int) -> np.ndarray:
    return np.full(frame_count, np.nan, dtype=np.float64)


def _candidate_pose_measurements(
    image_points: Sequence[object],
    world_points: Sequence[object],
    *,
    side: str,
    min_confidence: float,
) -> tuple[bool, tuple[float, float], tuple[float, float, float], float]:
    """Return a conservative, history-free arm candidate measurement."""

    wrist = _point_by_name(image_points, f"{side}_wrist")
    elbow = _point_by_name(image_points, f"{side}_elbow")
    world_wrist = _point_by_name(world_points, f"{side}_wrist")
    world_elbow = _point_by_name(world_points, f"{side}_elbow")
    world_shoulder = _point_by_name(world_points, f"{side}_shoulder")
    if not all(_finite_point(point) for point in (wrist, elbow)):
        return False, (math.nan, math.nan), (math.nan, math.nan, math.nan), 0.0
    source = _rotated_to_source(
        float(getattr(wrist, "x")), float(getattr(wrist, "y"))
    )
    if not all(
        _finite_point(point)
        for point in (world_wrist, world_elbow, world_shoulder)
    ):
        return False, source, (math.nan, math.nan, math.nan), 0.0
    confidence = min(
        float(getattr(world_wrist, "confidence", 0.0)),
        float(getattr(world_elbow, "confidence", 0.0)),
        float(getattr(world_shoulder, "confidence", 0.0)),
    )
    shoulder_xyz = _xyz(world_shoulder)
    elbow_xyz = _xyz(world_elbow)
    wrist_xyz = _xyz(world_wrist)
    upper = float(np.linalg.norm(elbow_xyz - shoulder_xyz))
    forearm = float(np.linalg.norm(wrist_xyz - elbow_xyz))
    reach = float(np.linalg.norm(wrist_xyz - shoulder_xyz))
    ratio = forearm / max(upper, 1e-6)
    valid = bool(
        confidence >= min_confidence
        and 0.12 <= upper <= 0.55
        and 0.11 <= forearm <= 0.52
        and 0.08 <= reach <= 0.88
        and 0.42 <= ratio <= 1.85
        and 0.008 <= _image_distance(wrist, elbow) <= 0.34
    )
    dx, dy, dz = wrist_xyz - shoulder_xyz
    relative_cm = (-dy * 100.0, -dx * 100.0, -dz * 100.0)
    return valid, source, relative_cm, confidence


def _apply_global_wristband_resolution(
    arrays: dict[str, np.ndarray],
    calibration: WristbandCalibration,
) -> dict[str, int]:
    """Resolve the physical banded wrist over the complete clip with Viterbi DP."""

    marker = arrays["candidate_marker_score"]
    source_x = arrays["candidate_source_x"]
    source_y = arrays["candidate_source_y"]
    valid = arrays["candidate_valid"]
    confidence = arrays["candidate_confidence"]
    frame_count = marker.shape[0]
    emission = np.full((frame_count, 2), 8.0, dtype=np.float64)
    anchors = np.zeros((frame_count, 2), dtype=np.bool_)
    scale = max(0.07, calibration.score_floor * 0.22)
    for index in range(frame_count):
        finite_scores = [
            float(marker[index, state])
            for state in range(2)
            if math.isfinite(float(marker[index, state]))
        ]
        for state in range(2):
            score = float(marker[index, state])
            if not math.isfinite(score):
                continue
            other = float(marker[index, 1 - state])
            margin = score - other if math.isfinite(other) else score
            anchors[index, state] = bool(
                score >= calibration.score_floor
                and margin >= calibration.margin_floor
            )
            marker_z = (score - calibration.score_floor) / scale
            emission[index, state] = (
                (0.0 if bool(valid[index, state]) else 4.8)
                - 1.45 * max(-1.5, min(2.8, marker_z))
                - 0.55 * max(0.0, min(1.0, float(confidence[index, state])))
                - (2.8 if bool(anchors[index, state]) else 0.0)
            )

    costs = np.full((frame_count, 2), math.inf, dtype=np.float64)
    back = np.zeros((frame_count, 2), dtype=np.int8)
    costs[0] = emission[0]
    for index in range(1, frame_count):
        for state in range(2):
            current_finite = math.isfinite(float(source_x[index, state])) and math.isfinite(
                float(source_y[index, state])
            )
            best_cost = math.inf
            best_previous = 0
            for previous_state in range(2):
                previous_finite = math.isfinite(
                    float(source_x[index - 1, previous_state])
                ) and math.isfinite(float(source_y[index - 1, previous_state]))
                if current_finite and previous_finite:
                    distance = math.hypot(
                        float(source_x[index, state] - source_x[index - 1, previous_state]),
                        float(source_y[index, state] - source_y[index - 1, previous_state]),
                    )
                    transition = min(9.0, 1.15 * (distance / 0.055) ** 2)
                else:
                    transition = 1.7
                # A label change is cheap when it preserves the physical path;
                # MediaPipe is explicitly allowed to swap anatomical names.
                if state != previous_state:
                    transition += 0.08
                candidate_cost = costs[index - 1, previous_state] + transition
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_previous = previous_state
            costs[index, state] = best_cost + emission[index, state]
            back[index, state] = best_previous

    states = np.zeros(frame_count, dtype=np.int8)
    states[-1] = int(np.argmin(costs[-1]))
    for index in range(frame_count - 1, 0, -1):
        states[index - 1] = back[index, states[index]]

    selected_anchor = anchors[np.arange(frame_count), states]
    anchor_indices = np.flatnonzero(selected_anchor)
    near_anchor = np.zeros(frame_count, dtype=np.bool_)
    if anchor_indices.size:
        last = -100_000
        distance_to_anchor = np.full(frame_count, 100_000, dtype=np.int32)
        for index in range(frame_count):
            if bool(selected_anchor[index]):
                last = index
            distance_to_anchor[index] = index - last
        following = 100_000
        for index in range(frame_count - 1, -1, -1):
            if bool(selected_anchor[index]):
                following = index
            distance_to_anchor[index] = np.minimum(
                distance_to_anchor[index], following - index
            )
        near_anchor = distance_to_anchor <= 54

    chosen = np.arange(frame_count), states
    accepted = valid[chosen] & near_anchor
    arrays["resolved_pose_side_right"] = states == 1
    arrays["wristband_anchor"] = selected_anchor
    arrays["wristband_score"] = marker[chosen]
    arrays["left_arm_identity_accepted"] = accepted
    arrays["world_available"] = accepted
    arrays["confidence"] = np.where(accepted, confidence[chosen], 0.0)
    for target, candidate_key in (
        ("source_x", "candidate_source_x"),
        ("source_y", "candidate_source_y"),
        ("forward_cm_raw", "candidate_forward_cm"),
        ("depth_cm_raw", "candidate_depth_cm"),
        ("lateral_cm_raw", "candidate_lateral_cm"),
    ):
        values = arrays[candidate_key][chosen].astype(np.float64, copy=True)
        values[~accepted] = np.nan
        arrays[target] = values
    # On strong anchors, use the localized band center for audit-video pixels.
    for target, candidate_key in (
        ("source_x", "candidate_band_source_x"),
        ("source_y", "candidate_band_source_y"),
    ):
        band_values = arrays[candidate_key][chosen]
        use_band = accepted & selected_anchor & np.isfinite(band_values)
        arrays[target][use_band] = band_values[use_band]
    switches = int(np.count_nonzero(states[1:] != states[:-1]))
    return {
        "global_path_label_switches": switches,
        "global_path_anchor_frames": int(selected_anchor.sum()),
        "global_path_accepted_frames": int(accepted.sum()),
    }


def extract_trajectory(
    input_video: Path,
    model_path: Path,
    *,
    min_confidence: float,
    rtmw_model_path: Path | None = None,
    side: str = "left",
    wristband_calibration: WristbandCalibration | None = None,
) -> tuple[VideoInfo, dict[str, np.ndarray], dict[str, object]]:
    selected_side = "right" if str(side).lower() == "right" else "left"
    capture, info = _open_video(input_video)
    candidate_shape = (info.frame_count, 2)
    arrays = {
        "time_s": np.arange(info.frame_count, dtype=np.float64) / info.fps,
        "source_x": _nan_series(info.frame_count),
        "source_y": _nan_series(info.frame_count),
        "forward_cm_raw": _nan_series(info.frame_count),
        "depth_cm_raw": _nan_series(info.frame_count),
        "lateral_cm_raw": _nan_series(info.frame_count),
        "confidence": np.zeros(info.frame_count, dtype=np.float64),
        "pose_detected": np.zeros(info.frame_count, dtype=np.bool_),
        "world_available": np.zeros(info.frame_count, dtype=np.bool_),
        "left_arm_identity_accepted": np.zeros(info.frame_count, dtype=np.bool_),
        "rtmw_identity_accepted": np.zeros(info.frame_count, dtype=np.bool_),
        "resolved_pose_side_right": np.zeros(info.frame_count, dtype=np.bool_),
        "wristband_score": _nan_series(info.frame_count),
        "wristband_anchor": np.zeros(info.frame_count, dtype=np.bool_),
        "candidate_source_x": np.full(candidate_shape, np.nan, dtype=np.float64),
        "candidate_source_y": np.full(candidate_shape, np.nan, dtype=np.float64),
        "candidate_band_source_x": np.full(candidate_shape, np.nan, dtype=np.float64),
        "candidate_band_source_y": np.full(candidate_shape, np.nan, dtype=np.float64),
        "candidate_forward_cm": np.full(candidate_shape, np.nan, dtype=np.float64),
        "candidate_depth_cm": np.full(candidate_shape, np.nan, dtype=np.float64),
        "candidate_lateral_cm": np.full(candidate_shape, np.nan, dtype=np.float64),
        "candidate_confidence": np.zeros(candidate_shape, dtype=np.float64),
        "candidate_valid": np.zeros(candidate_shape, dtype=np.bool_),
        "candidate_marker_score": np.full(candidate_shape, np.nan, dtype=np.float64),
    }
    calibration = wristband_calibration or WristbandCalibration(
        score_floor=0.36,
        margin_floor=0.045,
        sample_count=0,
        source_videos=(),
    )
    backend = MediaPipeBackend(
        model_path,
        min_pose_detection_confidence=0.20,
        min_pose_presence_confidence=0.20,
        min_tracking_confidence=0.20,
    )
    identity_gate = LeftArmIdentityGate(min_confidence=min_confidence)
    rtmw_backend = None
    rtmw_error = ""
    if rtmw_model_path is not None:
        try:
            from src.backends.yolo_rtmw_backend import YoloRtmwWholeBodyBackend

            rtmw_backend = YoloRtmwWholeBodyBackend(
                rtmw_model_path=rtmw_model_path,
                yolo_device="cpu",
                rtmw_device="cuda",
                min_match_points=3,
                max_match_distance=0.50,
            )
        except Exception as exc:
            rtmw_error = f"{type(exc).__name__}: {exc}"
            print(f"RTMW disabled: {rtmw_error}", flush=True)
    rejection_reasons: Counter[str] = Counter()
    rtmw_reasons: Counter[str] = Counter()
    last_resolved_side = selected_side
    last_anchor_frame = -10_000
    previous_source: np.ndarray | None = None
    previous_source_frame: int | None = None
    processed = 0
    try:
        while processed < info.frame_count:
            ok, frame = capture.read()
            if not ok:
                break
            rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            timestamp_ms = int(round(processed * 1000.0 / info.fps))
            result = backend.detect(rotated, timestamp_ms)
            arrays["pose_detected"][processed] = bool(result.success)
            world_points = result.extra.get("world_keypoints") or ()
            resolved_side: str | None = None
            marker_center: tuple[float, float] | None = None
            marker_current = False
            mp_observations: dict[str, tuple[float, tuple[float, float] | None]] = {}
            for candidate_side in ("left", "right"):
                mp_observations[candidate_side] = _wristband_observation(
                    rotated,
                    _point_by_name(result.keypoints, f"{candidate_side}_wrist"),
                    _point_by_name(result.keypoints, f"{candidate_side}_elbow"),
                )
                state = 1 if candidate_side == "right" else 0
                valid_candidate, source_candidate, world_candidate, candidate_confidence = (
                    _candidate_pose_measurements(
                        result.keypoints,
                        world_points,
                        side=candidate_side,
                        min_confidence=min_confidence,
                    )
                )
                arrays["candidate_valid"][processed, state] = valid_candidate
                arrays["candidate_source_x"][processed, state] = source_candidate[0]
                arrays["candidate_source_y"][processed, state] = source_candidate[1]
                arrays["candidate_forward_cm"][processed, state] = world_candidate[0]
                arrays["candidate_depth_cm"][processed, state] = world_candidate[1]
                arrays["candidate_lateral_cm"][processed, state] = world_candidate[2]
                arrays["candidate_confidence"][processed, state] = candidate_confidence
                observation_score, observation_center = mp_observations[candidate_side]
                arrays["candidate_marker_score"][processed, state] = observation_score
                if observation_center is not None:
                    band_x, band_y = _rotated_to_source(*observation_center)
                    arrays["candidate_band_source_x"][processed, state] = band_x
                    arrays["candidate_band_source_y"][processed, state] = band_y
            finite_mp_scores = {
                candidate_side: observation[0]
                for candidate_side, observation in mp_observations.items()
                if math.isfinite(observation[0])
            }
            if finite_mp_scores:
                ordered_mp = sorted(
                    finite_mp_scores, key=finite_mp_scores.get, reverse=True
                )
                mp_winner = ordered_mp[0]
                mp_margin = (
                    finite_mp_scores[mp_winner] - finite_mp_scores[ordered_mp[1]]
                    if len(ordered_mp) > 1
                    else finite_mp_scores[mp_winner]
                )
                if (
                    finite_mp_scores[mp_winner] >= calibration.score_floor
                    and mp_margin >= calibration.margin_floor
                ):
                    resolved_side = mp_winner
                    marker_center = mp_observations[mp_winner][1]
                    marker_current = True
                    arrays["wristband_score"][processed] = finite_mp_scores[mp_winner]
            if rtmw_backend is not None:
                rtmw_result = rtmw_backend.detect(rotated, timestamp_ms)
                rtmw_observations = {
                    candidate_side: _wristband_observation(
                        rotated,
                        _point_by_name(rtmw_result.keypoints, f"{candidate_side}_wrist"),
                        _point_by_name(rtmw_result.keypoints, f"{candidate_side}_elbow"),
                    )
                    for candidate_side in ("left", "right")
                }
                finite_rtmw_scores = {
                    candidate_side: observation[0]
                    for candidate_side, observation in rtmw_observations.items()
                    if math.isfinite(observation[0])
                }
                rtmw_anchor = False
                if finite_rtmw_scores:
                    ordered_rtmw = sorted(
                        finite_rtmw_scores, key=finite_rtmw_scores.get, reverse=True
                    )
                    rtmw_winner = ordered_rtmw[0]
                    rtmw_margin = (
                        finite_rtmw_scores[rtmw_winner]
                        - finite_rtmw_scores[ordered_rtmw[1]]
                        if len(ordered_rtmw) > 1
                        else finite_rtmw_scores[rtmw_winner]
                    )
                    rtmw_anchor = (
                        finite_rtmw_scores[rtmw_winner] >= calibration.score_floor
                        and rtmw_margin >= calibration.margin_floor
                    )
                if rtmw_anchor:
                    oracle_wrist = _point_by_name(
                        rtmw_result.keypoints, f"{rtmw_winner}_wrist"
                    )
                    candidates = {
                        candidate_side: _point_by_name(
                            result.keypoints, f"{candidate_side}_wrist"
                        )
                        for candidate_side in ("left", "right")
                    }
                    distances = {
                        candidate_side: _image_distance(oracle_wrist, point)
                        for candidate_side, point in candidates.items()
                        if _finite_point(point) and _finite_point(oracle_wrist)
                    }
                    if distances:
                        closest_side = min(distances, key=distances.get)
                        if distances[closest_side] <= 0.16:
                            rtmw_score = finite_rtmw_scores[rtmw_winner]
                            closest_state = 1 if closest_side == "right" else 0
                            old_candidate_score = float(
                                arrays["candidate_marker_score"][processed, closest_state]
                            )
                            arrays["candidate_marker_score"][processed, closest_state] = max(
                                old_candidate_score if math.isfinite(old_candidate_score) else -math.inf,
                                rtmw_score + 0.045,
                            )
                            rtmw_center = rtmw_observations[rtmw_winner][1]
                            if rtmw_center is not None:
                                band_x, band_y = _rotated_to_source(*rtmw_center)
                                arrays["candidate_band_source_x"][processed, closest_state] = band_x
                                arrays["candidate_band_source_y"][processed, closest_state] = band_y
                            current_score = float(arrays["wristband_score"][processed])
                            if (
                                resolved_side is None
                                or resolved_side == closest_side
                                or not math.isfinite(current_score)
                                or rtmw_score > current_score + 0.04
                            ):
                                resolved_side = closest_side
                                marker_center = rtmw_observations[rtmw_winner][1]
                                arrays["wristband_score"][processed] = rtmw_score
                            marker_current = True
                            arrays["rtmw_identity_accepted"][processed] = True
                            rtmw_reasons["wristband_anchor_matched"] += 1
                        else:
                            rtmw_reasons["rtmw_mediapipe_wrist_disagreement"] += 1
                else:
                    rtmw_reasons["wristband_not_distinct"] += 1

            # During splashes or underwater blur, choose the candidate that is
            # spatially continuous with the most recent accepted band point.
            if resolved_side is None and previous_source is not None:
                temporal_distances: dict[str, float] = {}
                for candidate_side in ("left", "right"):
                    point = _point_by_name(
                        result.keypoints, f"{candidate_side}_wrist"
                    )
                    if not _finite_point(point):
                        continue
                    sx, sy = _rotated_to_source(
                        float(getattr(point, "x")), float(getattr(point, "y"))
                    )
                    score = mp_observations[candidate_side][0]
                    marker_bonus = 0.035 * max(0.0, score) if math.isfinite(score) else 0.0
                    temporal_distances[candidate_side] = float(
                        np.linalg.norm(np.asarray([sx, sy]) - previous_source)
                    ) - marker_bonus
                if temporal_distances:
                    temporal_side = min(temporal_distances, key=temporal_distances.get)
                    gap = processed - int(
                        previous_source_frame
                        if previous_source_frame is not None
                        else processed
                    )
                    if temporal_distances[temporal_side] <= 0.11 + 0.018 * max(1, gap):
                        resolved_side = temporal_side
            if resolved_side is None and processed - last_anchor_frame <= 30:
                resolved_side = last_resolved_side
            if resolved_side is None:
                resolved_side = selected_side
            if marker_current:
                last_anchor_frame = processed
                arrays["wristband_anchor"][processed] = True
            last_resolved_side = resolved_side
            arrays["resolved_pose_side_right"][processed] = resolved_side == "right"
            if processed - last_anchor_frame > 30:
                accepted = False
                reason = "wristband_identity_unavailable"
            else:
                accepted, reason = identity_gate.evaluate(
                    result.keypoints,
                    world_points,
                    frame_index=processed,
                    side=resolved_side,
                )
            rejection_reasons[reason] += 1
            arrays["left_arm_identity_accepted"][processed] = accepted
            image_wrist = _point_by_name(result.keypoints, f"{resolved_side}_wrist")
            world_wrist = _point_by_name(world_points, f"{resolved_side}_wrist")
            world_shoulder = _point_by_name(world_points, f"{resolved_side}_shoulder")
            if accepted and _finite_point(image_wrist):
                if marker_current and marker_center is not None:
                    source_x, source_y = _rotated_to_source(*marker_center)
                else:
                    source_x, source_y = _rotated_to_source(
                        float(getattr(image_wrist, "x")),
                        float(getattr(image_wrist, "y")),
                    )
                arrays["source_x"][processed] = source_x
                arrays["source_y"][processed] = source_y
                previous_source = np.asarray([source_x, source_y], dtype=np.float64)
                previous_source_frame = processed
            if accepted and _finite_point(world_wrist) and _finite_point(world_shoulder):
                confidence = min(
                    float(getattr(world_wrist, "confidence", 0.0)),
                    float(getattr(world_shoulder, "confidence", 0.0)),
                )
                arrays["confidence"][processed] = confidence
                # Inference is performed on the clockwise-rotated image:
                # - rotated -Y is forward (toward the swimmer's head),
                # - rotated -X is source-image depth (downward),
                # - world Z is the single-view lateral estimate.
                dx = float(getattr(world_wrist, "x")) - float(
                    getattr(world_shoulder, "x")
                )
                dy = float(getattr(world_wrist, "y")) - float(
                    getattr(world_shoulder, "y")
                )
                dz = float(getattr(world_wrist, "z")) - float(
                    getattr(world_shoulder, "z")
                )
                arrays["forward_cm_raw"][processed] = -dy * 100.0
                arrays["depth_cm_raw"][processed] = -dx * 100.0
                arrays["lateral_cm_raw"][processed] = -dz * 100.0
                arrays["world_available"][processed] = True

            processed += 1
            if processed % 150 == 0 or processed == info.frame_count:
                print(f"pose extraction: {processed}/{info.frame_count}", flush=True)
    finally:
        capture.release()
        backend.close()
        if rtmw_backend is not None:
            rtmw_backend.close()

    if processed != info.frame_count:
        for key, values in tuple(arrays.items()):
            arrays[key] = values[:processed]
        info = VideoInfo(info.width, info.height, info.fps, processed)

    global_resolution_stats = _apply_global_wristband_resolution(
        arrays, calibration
    )

    stats: dict[str, object] = {
        "input_frame_count": info.frame_count,
        "pose_detected_frames": int(arrays["pose_detected"].sum()),
        "world_valid_frames": int(arrays["world_available"].sum()),
        "identity_gate_accepted_frames": int(arrays["left_arm_identity_accepted"].sum()),
        "identity_gate_rejected_frames": int(info.frame_count - arrays["left_arm_identity_accepted"].sum()),
        "identity_gate_rejection_reasons": dict(sorted(rejection_reasons.items())),
        "rtmw_enabled": rtmw_backend is not None,
        "rtmw_initialization_error": rtmw_error,
        "rtmw_identity_accepted_frames": int(arrays["rtmw_identity_accepted"].sum()),
        "rtmw_identity_reasons": dict(sorted(rtmw_reasons.items())),
        "mediapipe_right_side_selected_frames": int(arrays["resolved_pose_side_right"].sum()),
        "selected_anatomical_side": selected_side,
        "wristband_anchor_frames": int(arrays["wristband_anchor"].sum()),
        "wristband_joint_calibration": {
            "score_floor": calibration.score_floor,
            "margin_floor": calibration.margin_floor,
            "sample_count": calibration.sample_count,
            "source_videos": list(calibration.source_videos),
        },
        **global_resolution_stats,
    }
    return info, arrays, stats


def _remove_jump_outliers(values: np.ndarray, *, max_jump: float) -> np.ndarray:
    cleaned = values.astype(np.float64, copy=True)
    finite = np.flatnonzero(np.isfinite(cleaned))
    if finite.size < 3:
        return cleaned
    previous = int(finite[0])
    for index in finite[1:]:
        index = int(index)
        span = max(1, index - previous)
        if abs(cleaned[index] - cleaned[previous]) > max_jump * span:
            cleaned[index] = np.nan
        else:
            previous = index
    return cleaned


def _interpolate(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    if not finite.any():
        raise RuntimeError("no usable left-wrist trajectory points were detected")
    indices = np.arange(values.size, dtype=np.float64)
    filled = np.interp(indices, indices[finite], values[finite]).astype(np.float64)
    return filled, ~finite


def _median_filter(values: np.ndarray, radius: int = 2) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.asarray(
        [np.median(padded[index : index + 2 * radius + 1]) for index in range(values.size)],
        dtype=np.float64,
    )


def _one_euro(values: np.ndarray, fps: float) -> np.ndarray:
    filter_ = OneEuroValueFilter(
        min_cutoff=1.15,
        beta=0.055,
        d_cutoff=1.0,
        max_gap_ms_before_reset=500.0,
    )
    return np.asarray(
        [filter_.apply(float(value), index * 1000.0 / fps) for index, value in enumerate(values)],
        dtype=np.float64,
    )


def postprocess_trajectory(
    arrays: dict[str, np.ndarray],
    *,
    fps: float,
) -> dict[str, np.ndarray]:
    processed = dict(arrays)
    estimated_masks: list[np.ndarray] = []
    for source_key, target_key, max_jump in (
        ("forward_cm_raw", "forward_cm", 28.0),
        ("depth_cm_raw", "depth_cm", 24.0),
        ("lateral_cm_raw", "lateral_cm", 24.0),
        ("source_x", "source_x_smoothed", 0.24),
        ("source_y", "source_y_smoothed", 0.24),
    ):
        cleaned = _remove_jump_outliers(processed[source_key], max_jump=max_jump)
        filled, estimated = _interpolate(cleaned)
        filtered = _one_euro(_median_filter(filled), fps)
        processed[target_key] = filtered
        estimated_masks.append(estimated)
    processed["estimated"] = np.logical_or.reduce(estimated_masks)
    # Derivatives use the source video's native, fixed frame interval.  They
    # deliberately come from the chronological per-frame trajectory rather
    # than from a phase-normalized representative stroke.
    edge_order = 2 if processed["time_s"].size >= 3 else 1
    for position_key, speed_key, acceleration_key in (
        ("forward_cm", "forward_velocity_cm_s", "forward_acceleration_cm_s2"),
        ("depth_cm", "depth_velocity_cm_s", "depth_acceleration_cm_s2"),
        ("lateral_cm", "lateral_velocity_cm_s", "lateral_acceleration_cm_s2"),
    ):
        speed = np.gradient(processed[position_key], 1.0 / fps, edge_order=edge_order)
        processed[speed_key] = speed
        processed[acceleration_key] = np.gradient(
            speed, 1.0 / fps, edge_order=edge_order
        )
    processed["speed_cm_s"] = np.sqrt(
        processed["forward_velocity_cm_s"] ** 2
        + processed["depth_velocity_cm_s"] ** 2
        + processed["lateral_velocity_cm_s"] ** 2
    )
    processed["acceleration_cm_s2"] = np.sqrt(
        processed["forward_acceleration_cm_s2"] ** 2
        + processed["depth_acceleration_cm_s2"] ** 2
        + processed["lateral_acceleration_cm_s2"] ** 2
    )
    return processed


def _robust_limit(*series: np.ndarray, minimum: float, maximum: float) -> float:
    concatenated = np.concatenate([np.abs(values[np.isfinite(values)]) for values in series])
    if concatenated.size == 0:
        return minimum
    value = float(np.percentile(concatenated, 98.0)) * 1.15
    rounded = math.ceil(value / 10.0) * 10.0
    return max(minimum, min(maximum, rounded))


def _map_plot(
    forward: float,
    vertical: float,
    rect: tuple[int, int, int, int],
    forward_limit: float,
    vertical_limit: float,
) -> tuple[int, int]:
    x1, y1, x2, y2 = rect
    px = x1 + (forward + forward_limit) / (2.0 * forward_limit) * (x2 - x1)
    py = y1 + (vertical + vertical_limit) / (2.0 * vertical_limit) * (y2 - y1)
    return int(round(px)), int(round(py))


def _draw_panel(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    *,
    title: str,
    vertical_label: str,
    forward_limit: float,
    vertical_limit: float,
) -> None:
    x1, y1, x2, y2 = rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), (23, 31, 43), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (65, 79, 96), 1, cv2.LINE_AA)
    for fraction in (0.25, 0.5, 0.75):
        x = int(round(x1 + fraction * (x2 - x1)))
        y = int(round(y1 + fraction * (y2 - y1)))
        cv2.line(frame, (x, y1), (x, y2), (38, 49, 64), 1, cv2.LINE_AA)
        cv2.line(frame, (x1, y), (x2, y), (38, 49, 64), 1, cv2.LINE_AA)
    zero = _map_plot(0.0, 0.0, rect, forward_limit, vertical_limit)
    cv2.line(frame, (zero[0], y1), (zero[0], y2), (93, 108, 126), 1, cv2.LINE_AA)
    cv2.line(frame, (x1, zero[1]), (x2, zero[1]), (93, 108, 126), 1, cv2.LINE_AA)
    cv2.putText(frame, title, (x1, y1 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (235, 242, 248), 2, cv2.LINE_AA)
    cv2.putText(frame, "Forward (cm)", (x2 - 150, zero[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (178, 191, 207), 1, cv2.LINE_AA)
    cv2.putText(frame, vertical_label, (zero[0] + 8, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (178, 191, 207), 1, cv2.LINE_AA)
    for value in (-forward_limit, 0.0, forward_limit):
        x, _ = _map_plot(value, 0.0, rect, forward_limit, vertical_limit)
        cv2.putText(frame, f"{value:.0f}", (x - 13, zero[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (138, 153, 171), 1, cv2.LINE_AA)
    for value in (-vertical_limit, vertical_limit):
        _, y = _map_plot(0.0, value, rect, forward_limit, vertical_limit)
        cv2.putText(frame, f"{value:.0f}", (zero[0] + 7, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (138, 153, 171), 1, cv2.LINE_AA)


def _draw_trail(
    frame: np.ndarray,
    forward: np.ndarray,
    vertical: np.ndarray,
    confidence: np.ndarray,
    estimated: np.ndarray,
    *,
    current: int,
    trail_frames: int,
    rect: tuple[int, int, int, int],
    forward_limit: float,
    vertical_limit: float,
) -> None:
    start = max(0, current - trail_frames + 1)
    indices = range(start, current + 1, 2)
    for index in indices:
        point = _map_plot(
            float(forward[index]),
            float(vertical[index]),
            rect,
            forward_limit,
            vertical_limit,
        )
        age = (index - start + 1) / max(1, current - start + 1)
        reliability = max(0.20, min(1.0, float(confidence[index]) / 0.55))
        strength = age * reliability * (0.55 if bool(estimated[index]) else 1.0)
        color = (
            int(90 + 80 * strength),
            int(130 + 105 * strength),
            int(135 + 105 * strength),
        )
        cv2.circle(frame, point, 3, color, -1, cv2.LINE_AA)
    current_point = _map_plot(
        float(forward[current]),
        float(vertical[current]),
        rect,
        forward_limit,
        vertical_limit,
    )
    overlay = frame.copy()
    cv2.circle(overlay, current_point, 18, (255, 225, 85), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0.0, frame)
    cv2.circle(frame, current_point, 8, (80, 210, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, current_point, 4, (245, 250, 255), -1, cv2.LINE_AA)


def _draw_representative_cycle(
    frame: np.ndarray,
    forward: np.ndarray,
    vertical: np.ndarray,
    *,
    phase: float,
    rect: tuple[int, int, int, int],
    forward_limit: float,
    vertical_limit: float,
) -> None:
    for index in range(0, forward.size, 2):
        point = _map_plot(
            float(forward[index]),
            float(vertical[index]),
            rect,
            forward_limit,
            vertical_limit,
        )
        cv2.circle(frame, point, 3, (235, 218, 72), -1, cv2.LINE_AA)
    current_index = int(round(max(0.0, min(1.0, phase)) * (forward.size - 1)))
    current_point = _map_plot(
        float(forward[current_index]),
        float(vertical[current_index]),
        rect,
        forward_limit,
        vertical_limit,
    )
    overlay = frame.copy()
    cv2.circle(overlay, current_point, 19, (255, 225, 85), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0.0, frame)
    cv2.circle(frame, current_point, 9, (70, 200, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, current_point, 4, (248, 252, 255), -1, cv2.LINE_AA)


def _create_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create output video: {path}")
    return writer


def render_reference_video(
    output_path: Path,
    arrays: dict[str, np.ndarray],
    info: VideoInfo,
    *,
    trail_seconds: float,
    side: str,
) -> dict[str, object]:
    canvas_size = (720, 1280)
    side_rect = (58, 190, 662, 610)
    overhead_rect = (58, 760, 662, 1180)
    selected_side = "right" if str(side).lower() == "right" else "left"
    side_cn = "右" if selected_side == "right" else "左"
    trail_frames = max(2, int(round(trail_seconds * info.fps)))
    forward_limit = _robust_limit(arrays["forward_cm"], minimum=60.0, maximum=120.0)
    depth_limit = _robust_limit(arrays["depth_cm"], minimum=50.0, maximum=100.0)
    lateral_limit = _robust_limit(arrays["lateral_cm"], minimum=40.0, maximum=90.0)
    writer = _create_writer(output_path, info.fps, canvas_size)
    try:
        for index in range(info.frame_count):
            frame = np.full((canvas_size[1], canvas_size[0], 3), (14, 19, 27), dtype=np.uint8)
            put_text(frame, f"游泳{side_cn}手腕划动轨迹", (40, 58), (238, 246, 252))
            status = "模型直测" if not bool(arrays["estimated"][index]) else "遮挡插值"
            put_text(
                frame,
                f"时间 {arrays['time_s'][index]:05.2f}s  速度 {arrays['speed_cm_s'][index]:.1f} cm/s  {status}",
                (40, 104),
                (95, 225, 245) if status == "模型直测" else (80, 175, 255),
            )
            _draw_panel(
                frame,
                side_rect,
                title="侧视图  Side On",
                vertical_label="Depth (cm)",
                forward_limit=forward_limit,
                vertical_limit=depth_limit,
            )
            _draw_trail(
                frame,
                arrays["forward_cm"],
                arrays["depth_cm"],
                arrays["confidence"],
                arrays["estimated"],
                current=index,
                trail_frames=trail_frames,
                rect=side_rect,
                forward_limit=forward_limit,
                vertical_limit=depth_limit,
            )
            _draw_panel(
                frame,
                overhead_rect,
                title="俯视图  Overhead (AI estimated)",
                vertical_label="Lateral (cm)",
                forward_limit=forward_limit,
                vertical_limit=lateral_limit,
            )
            _draw_trail(
                frame,
                arrays["forward_cm"],
                arrays["lateral_cm"],
                arrays["confidence"],
                arrays["estimated"],
                current=index,
                trail_frames=trail_frames,
                rect=overhead_rect,
                forward_limit=forward_limit,
                vertical_limit=lateral_limit,
            )
            cv2.putText(
                frame,
                f"Native {info.fps:g} fps timing | relative to {selected_side} shoulder | monocular 3D",
                (72, 1235),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (142, 156, 173),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
            if (index + 1) % 300 == 0 or index + 1 == info.frame_count:
                print(f"reference render: {index + 1}/{info.frame_count}", flush=True)
    finally:
        writer.release()
    return {
        "forward_axis_limit_cm": forward_limit,
        "depth_axis_limit_cm": depth_limit,
        "lateral_axis_limit_cm": lateral_limit,
        "trail_seconds": float(trail_seconds),
        "motion_timing": "native_source_frames_no_phase_normalization",
    }


def render_audit_video(
    input_video: Path,
    output_path: Path,
    arrays: dict[str, np.ndarray],
    info: VideoInfo,
    *,
    trail_seconds: float,
    side: str,
) -> None:
    capture, capture_info = _open_video(input_video)
    if capture_info.frame_count < info.frame_count:
        capture.release()
        raise RuntimeError("source video became shorter during rendering")
    writer = _create_writer(output_path, info.fps, (info.width, info.height))
    trail_frames = max(2, int(round(trail_seconds * info.fps)))
    selected_side = "right" if str(side).lower() == "right" else "left"
    side_cn = "右" if selected_side == "right" else "左"
    try:
        for index in range(info.frame_count):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"cannot read source frame {index}")
            start = max(0, index - trail_frames + 1)
            for point_index in range(start, index + 1, 2):
                x = int(round(float(arrays["source_x_smoothed"][point_index]) * info.width))
                y = int(round(float(arrays["source_y_smoothed"][point_index]) * info.height))
                x = max(0, min(info.width - 1, x))
                y = max(0, min(info.height - 1, y))
                age = (point_index - start + 1) / max(1, index - start + 1)
                color = (int(75 + 85 * age), int(125 + 110 * age), int(135 + 110 * age))
                cv2.circle(frame, (x, y), 3, color, -1, cv2.LINE_AA)
            direct = bool(arrays["left_arm_identity_accepted"][index]) and not bool(
                arrays["estimated"][index]
            )
            if direct:
                x = int(round(float(arrays["source_x_smoothed"][index]) * info.width))
                y = int(round(float(arrays["source_y_smoothed"][index]) * info.height))
                point = (max(0, min(info.width - 1, x)), max(0, min(info.height - 1, y)))
                cv2.circle(frame, point, 13, (70, 205, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, point, 5, (245, 250, 255), -1, cv2.LINE_AA)
                label = f"{side_cn}手腕（高置信度直测）"
                label_color = (75, 225, 250)
            else:
                x = int(round(float(arrays["source_x_smoothed"][index]) * info.width))
                y = int(round(float(arrays["source_y_smoothed"][index]) * info.height))
                point = (max(0, min(info.width - 1, x)), max(0, min(info.height - 1, y)))
                cv2.circle(frame, point, 10, (80, 175, 255), 2, cv2.LINE_AA)
                label = f"{side_cn}手腕（遮挡插值）"
                label_color = (80, 175, 255)
            put_text(frame, label, (24, 42), label_color)
            put_text(
                frame,
                f"t={arrays['time_s'][index]:.2f}s  confidence={arrays['confidence'][index]:.2f}",
                (24, 76),
                (238, 243, 248),
            )
            writer.write(frame)
            if (index + 1) % 300 == 0 or index + 1 == info.frame_count:
                print(f"audit render: {index + 1}/{info.frame_count}", flush=True)
    finally:
        capture.release()
        writer.release()


def _find_cycles(values: np.ndarray, fps: float) -> list[int]:
    min_distance = max(1, int(round(0.9 * fps)))
    radius = max(4, int(round(0.25 * fps)))
    candidates: list[tuple[float, int]] = []
    for index in range(radius, values.size - radius):
        value = float(values[index])
        if value < values[index - 1] or value < values[index + 1]:
            continue
        local_floor = min(
            float(np.min(values[index - radius : index])),
            float(np.min(values[index + 1 : index + radius + 1])),
        )
        prominence = value - local_floor
        if prominence >= 12.0:
            candidates.append((prominence, index))
    selected: list[int] = []
    for _prominence, index in sorted(candidates, reverse=True):
        if all(abs(index - other) >= min_distance for other in selected):
            selected.append(index)
    return sorted(selected)


def _circular_smooth(values: np.ndarray, radius: int = 4) -> np.ndarray:
    if values.size <= 2 * radius:
        return values.copy()
    padded = np.pad(values, (radius, radius), mode="wrap")
    kernel = np.ones(2 * radius + 1, dtype=np.float64) / (2 * radius + 1)
    return np.convolve(padded, kernel, mode="valid")


def build_representative_cycle(
    arrays: dict[str, np.ndarray],
    fps: float,
    *,
    sample_count: int = 121,
) -> dict[str, np.ndarray | int]:
    peaks = _find_cycles(arrays["forward_cm"], fps)
    minimum = int(round(0.9 * fps))
    maximum = int(round(2.4 * fps))
    valid_intervals = [
        (start, end)
        for start, end in zip(peaks, peaks[1:])
        if minimum <= end - start <= maximum
    ]
    if len(valid_intervals) < 3:
        raise RuntimeError(
            "fewer than three stable left-arm cycles were found; cannot build representative trajectory"
        )
    target_phase = np.linspace(0.0, 1.0, sample_count)
    templates: dict[str, list[np.ndarray]] = {
        "forward_cm": [],
        "depth_cm": [],
        "lateral_cm": [],
    }
    for start, end in valid_intervals:
        source_phase = np.linspace(0.0, 1.0, end - start + 1)
        for key in templates:
            templates[key].append(
                np.interp(target_phase, source_phase, arrays[key][start : end + 1])
            )
    representative: dict[str, np.ndarray | int] = {}
    for key, cycles in templates.items():
        median_cycle = np.median(np.stack(cycles), axis=0)
        endpoint = 0.5 * (median_cycle[0] + median_cycle[-1])
        median_cycle[0] = endpoint
        median_cycle[-1] = endpoint
        representative[key] = _circular_smooth(median_cycle)

    interval_lengths = np.asarray([end - start for start, end in valid_intervals], dtype=np.float64)
    typical_length = max(1.0, float(np.median(interval_lengths)))
    phase = np.zeros(arrays["time_s"].size, dtype=np.float64)
    peak_array = np.asarray(peaks, dtype=np.int64)
    for index in range(phase.size):
        position = int(np.searchsorted(peak_array, index, side="right")) - 1
        if 0 <= position < peak_array.size - 1:
            start = int(peak_array[position])
            end = int(peak_array[position + 1])
            phase[index] = (index - start) / max(1, end - start)
        elif position < 0:
            phase[index] = ((index - int(peak_array[0])) / typical_length) % 1.0
        else:
            phase[index] = ((index - int(peak_array[-1])) / typical_length) % 1.0
    representative["phase"] = phase
    representative["cycle_count"] = len(valid_intervals)
    return representative


def write_csv(
    path: Path,
    arrays: dict[str, np.ndarray],
    info: VideoInfo,
    *,
    side: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_side = "right" if str(side).lower() == "right" else "left"
    x_field = f"{selected_side}_wrist_x_norm"
    y_field = f"{selected_side}_wrist_y_norm"
    x_raw_field = f"{selected_side}_wrist_x_norm_raw"
    y_raw_field = f"{selected_side}_wrist_y_norm_raw"
    fields = (
        "frame_index",
        "time_s",
        x_field,
        y_field,
        x_raw_field,
        y_raw_field,
        "forward_cm_raw",
        "depth_cm_raw",
        "lateral_cm_raw",
        "forward_cm",
        "depth_cm",
        "lateral_cm",
        "forward_velocity_cm_s",
        "depth_velocity_cm_s",
        "lateral_velocity_cm_s",
        "speed_cm_s",
        "forward_acceleration_cm_s2",
        "depth_acceleration_cm_s2",
        "lateral_acceleration_cm_s2",
        "acceleration_cm_s2",
        "confidence",
        "pose_detected",
        "world_available",
        "left_arm_identity_accepted",
        "rtmw_identity_accepted",
        "resolved_pose_side_right",
        "wristband_score",
        "wristband_anchor",
        "estimated",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(info.frame_count):
            writer.writerow(
                {
                    "frame_index": index,
                    "time_s": f"{arrays['time_s'][index]:.6f}",
                    x_field: f"{arrays['source_x_smoothed'][index]:.8f}",
                    y_field: f"{arrays['source_y_smoothed'][index]:.8f}",
                    x_raw_field: _csv_number(arrays["source_x"][index]),
                    y_raw_field: _csv_number(arrays["source_y"][index]),
                    "forward_cm_raw": _csv_number(arrays["forward_cm_raw"][index]),
                    "depth_cm_raw": _csv_number(arrays["depth_cm_raw"][index]),
                    "lateral_cm_raw": _csv_number(arrays["lateral_cm_raw"][index]),
                    "forward_cm": f"{arrays['forward_cm'][index]:.5f}",
                    "depth_cm": f"{arrays['depth_cm'][index]:.5f}",
                    "lateral_cm": f"{arrays['lateral_cm'][index]:.5f}",
                    "forward_velocity_cm_s": f"{arrays['forward_velocity_cm_s'][index]:.5f}",
                    "depth_velocity_cm_s": f"{arrays['depth_velocity_cm_s'][index]:.5f}",
                    "lateral_velocity_cm_s": f"{arrays['lateral_velocity_cm_s'][index]:.5f}",
                    "speed_cm_s": f"{arrays['speed_cm_s'][index]:.5f}",
                    "forward_acceleration_cm_s2": f"{arrays['forward_acceleration_cm_s2'][index]:.5f}",
                    "depth_acceleration_cm_s2": f"{arrays['depth_acceleration_cm_s2'][index]:.5f}",
                    "lateral_acceleration_cm_s2": f"{arrays['lateral_acceleration_cm_s2'][index]:.5f}",
                    "acceleration_cm_s2": f"{arrays['acceleration_cm_s2'][index]:.5f}",
                    "confidence": f"{arrays['confidence'][index]:.5f}",
                    "pose_detected": int(bool(arrays["pose_detected"][index])),
                    "world_available": int(bool(arrays["world_available"][index])),
                    "left_arm_identity_accepted": int(
                        bool(arrays["left_arm_identity_accepted"][index])
                    ),
                    "rtmw_identity_accepted": int(
                        bool(arrays["rtmw_identity_accepted"][index])
                    ),
                    "resolved_pose_side_right": int(
                        bool(arrays["resolved_pose_side_right"][index])
                    ),
                    "wristband_score": _csv_number(arrays["wristband_score"][index]),
                    "wristband_anchor": int(bool(arrays["wristband_anchor"][index])),
                    "estimated": int(bool(arrays["estimated"][index])),
                }
            )


def _csv_number(value: float) -> str:
    return "" if not math.isfinite(float(value)) else f"{float(value):.5f}"


def _range(values: np.ndarray) -> dict[str, float]:
    return {
        "p05": round(float(np.percentile(values, 5.0)), 3),
        "p50": round(float(np.percentile(values, 50.0)), 3),
        "p95": round(float(np.percentile(values, 95.0)), 3),
        "span_p05_p95": round(
            float(np.percentile(values, 95.0) - np.percentile(values, 5.0)),
            3,
        ),
    }


def write_summary(
    path: Path,
    *,
    input_video: Path,
    model_path: Path,
    info: VideoInfo,
    arrays: dict[str, np.ndarray],
    extraction_stats: dict[str, object],
    render_stats: dict[str, object],
    outputs: dict[str, Path],
    side: str,
) -> dict[str, object]:
    selected_side = "right" if str(side).lower() == "right" else "left"
    cycles = _find_cycles(arrays["forward_cm"], info.fps)
    direct = ~arrays["estimated"]
    direct_confidence = arrays["confidence"][arrays["left_arm_identity_accepted"]]
    if direct_confidence.size == 0:
        direct_confidence = np.asarray([0.0], dtype=np.float64)
    summary: dict[str, object] = {
        "input_video": str(input_video.resolve()),
        "model": str(model_path.resolve()),
        "selected_anatomical_side": selected_side,
        "method": "joint_dark_wristband_anchor_plus_rtmw_mediapipe_and_temporal_arm_identity_gates",
        "timebase": (
            "one output frame per source frame at native fps; chronological trajectory; "
            "no representative-cycle phase normalization"
        ),
        "coordinate_note": (
            "Side-on is inferred from the source image plane. Overhead lateral depth is a monocular "
            "MediaPipe world-landmark estimate and is less reliable than the side-on path."
        ),
        "video": {
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "frame_count": info.frame_count,
            "duration_s": round(info.frame_count / info.fps, 6),
        },
        "quality": {
            **extraction_stats,
            "pose_detection_rate": round(float(arrays["pose_detected"].mean()), 6),
            "world_direct_rate": round(float(arrays["world_available"].mean()), 6),
            "fully_direct_rate_after_outlier_checks": round(float(direct.mean()), 6),
            "estimated_frame_count": int(arrays["estimated"].sum()),
            "median_selected_wrist_confidence": round(float(np.median(direct_confidence)), 6),
            "p10_selected_wrist_confidence": round(float(np.percentile(direct_confidence, 10.0)), 6),
        },
        "trajectory_cm_relative_to_selected_shoulder": {
            "forward": _range(arrays["forward_cm"]),
            "depth": _range(arrays["depth_cm"]),
            "lateral_monocular_estimate": _range(arrays["lateral_cm"]),
        },
        "motion": {
            "speed_cm_s": _range(arrays["speed_cm_s"]),
            "acceleration_cm_s2": _range(arrays["acceleration_cm_s2"]),
            "nonzero_speed_standard_deviation_cm_s": round(
                float(np.std(arrays["speed_cm_s"])), 6
            ),
        },
        "estimated_selected_arm_cycles": len(cycles),
        "cycle_peak_frames": cycles,
        "render": render_stats,
        "outputs": {name: str(output.resolve()) for name, output in outputs.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_video = args.input_video.resolve()
    model_path = args.model.resolve()
    rtmw_model_path = None if args.no_rtmw else args.rtmw_model.resolve()
    output_dir = args.output_dir.resolve()
    if not input_video.is_file():
        raise FileNotFoundError(input_video)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if rtmw_model_path is not None and not rtmw_model_path.is_file():
        raise FileNotFoundError(rtmw_model_path)
    calibration_paths = [input_video]
    for calibration_video in args.joint_calibration_video:
        resolved = calibration_video.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        if resolved not in calibration_paths:
            calibration_paths.append(resolved)
    wristband_calibration = calibrate_wristband(calibration_paths, model_path)
    print(
        "joint wristband calibration: "
        f"videos={len(calibration_paths)} samples={wristband_calibration.sample_count} "
        f"score_floor={wristband_calibration.score_floor:.4f} "
        f"margin_floor={wristband_calibration.margin_floor:.4f}",
        flush=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_video.stem
    side_cn = "右手" if args.side == "right" else "左手"
    outputs = {
        "reference_video": output_dir / f"{stem}_{side_cn}腕轨迹解析_原速.mp4",
        "audit_video": output_dir / f"{stem}_{side_cn}腕轨迹核验_原速.mp4",
        "trajectory_csv": output_dir / f"{stem}_{side_cn}腕轨迹_原速.csv",
        "summary_json": output_dir / f"{stem}_{side_cn}腕轨迹摘要_原速.json",
    }
    info, raw_arrays, extraction_stats = extract_trajectory(
        input_video,
        model_path,
        min_confidence=max(0.0, min(1.0, float(args.min_confidence))),
        rtmw_model_path=rtmw_model_path,
        side=args.side,
        wristband_calibration=wristband_calibration,
    )
    arrays = postprocess_trajectory(raw_arrays, fps=info.fps)
    write_csv(outputs["trajectory_csv"], arrays, info, side=args.side)
    render_stats = render_reference_video(
        outputs["reference_video"],
        arrays,
        info,
        trail_seconds=max(0.2, float(args.trail_seconds)),
        side=args.side,
    )
    render_audit_video(
        input_video,
        outputs["audit_video"],
        arrays,
        info,
        trail_seconds=max(0.2, float(args.trail_seconds)),
        side=args.side,
    )
    summary = write_summary(
        outputs["summary_json"],
        input_video=input_video,
        model_path=model_path,
        info=info,
        arrays=arrays,
        extraction_stats=extraction_stats,
        render_stats=render_stats,
        outputs=outputs,
        side=args.side,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
