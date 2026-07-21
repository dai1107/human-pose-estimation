"""Pose-frame conversion and session lifecycle helpers."""

from __future__ import annotations

import argparse
from math import nan
from pathlib import Path

from src.backends.base import PoseResult
from src.biomechanics.landmarks import LANDMARK_NAMES
from src.biomechanics.normalization import normalize_landmarks
from src.biomechanics.types import LandmarkPoint, PoseFrame
from src.runtime_hand import HandDetection

POSE_NAME_TO_INDEX = {name: index for index, name in enumerate(LANDMARK_NAMES)}


def to_landmark_points(result: PoseResult) -> list[LandmarkPoint]:
    return keypoints_to_landmark_points(result.keypoints)


def keypoints_to_landmark_points(keypoints: object) -> list[LandmarkPoint]:
    points = [LandmarkPoint(nan, nan, nan, 0.0, 0.0) for _ in LANDMARK_NAMES]
    if not isinstance(keypoints, (list, tuple)):
        return points
    for point in keypoints:
        index = POSE_NAME_TO_INDEX.get(point.name)
        if index is None:
            continue
        confidence = max(0.0, min(1.0, float(point.confidence)))
        visibility = point.visibility if point.visibility is not None else confidence
        presence = point.presence if point.presence is not None else confidence
        points[index] = LandmarkPoint(
            x=float(point.x),
            y=float(point.y),
            z=float(point.z),
            visibility=max(0.0, min(1.0, float(visibility))),
            presence=max(0.0, min(1.0, float(presence))),
        )
    return points


def build_pose_frame_from_result(
    result: PoseResult,
    *,
    frame_index: int,
    mirror: bool,
    frame_shape: tuple[int, int, int],
    fps: float,
    hand_detections: dict[str, HandDetection] | None = None,
) -> PoseFrame:
    image_landmarks = to_landmark_points(result)
    world_landmarks = keypoints_to_landmark_points(result.extra.get("world_keypoints"))
    normalization = normalize_landmarks(image_landmarks if result.success else None)
    detections = hand_detections or {}
    image_hand_landmarks = {side: list(detection.landmarks) for side, detection in detections.items()}
    world_hand_landmarks = {side: list(detection.world_landmarks) for side, detection in detections.items()}
    height, width = frame_shape[:2]
    return PoseFrame(
        frame_index=frame_index,
        timestamp_ms=int(result.timestamp_ms or 0),
        pose_detected=bool(result.success),
        image_landmarks=image_landmarks,
        world_landmarks=world_landmarks,
        smoothed_landmarks=image_landmarks,
        normalized_landmarks=normalization.landmarks,
        hands_detected=bool(image_hand_landmarks),
        hand_landmarks=image_hand_landmarks,
        hand_world_landmarks=world_hand_landmarks,
        smoothed_hand_landmarks={side: list(points) for side, points in image_hand_landmarks.items()},
        normalization_success=normalization.success,
        normalization_message=normalization.message,
        mirror=mirror,
        camera_width=width,
        camera_height=height,
        fps=float(fps),
        three_d_kinematics=dict(result.extra.get("three_d_kinematics") or {}),
    )


def current_model_label(args: argparse.Namespace, backend_name: str) -> str:
    if backend_name == "yolo-pose":
        return Path(args.yolo_pose_model).name
    return Path(args.model).name
