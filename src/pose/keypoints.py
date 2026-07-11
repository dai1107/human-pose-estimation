from __future__ import annotations


COMMON_KEYPOINTS: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

MEDIAPIPE_TO_COMMON: dict[int, str] = {
    0: "nose",
    2: "left_eye",
    5: "right_eye",
    7: "left_ear",
    8: "right_ear",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}

# Ultralytics pose models used by this project expose the standard COCO 17
# order through result.keypoints.xy/result.keypoints.conf.
YOLO_COCO17_TO_COMMON: dict[int, str] = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

if tuple(MEDIAPIPE_TO_COMMON.values()) != COMMON_KEYPOINTS:
    raise RuntimeError("MediaPipe common keypoint mapping is out of sync")
if tuple(YOLO_COCO17_TO_COMMON.values()) != COMMON_KEYPOINTS:
    raise RuntimeError("YOLO COCO17 common keypoint mapping is out of sync")


__all__ = ["COMMON_KEYPOINTS", "MEDIAPIPE_TO_COMMON", "YOLO_COCO17_TO_COMMON"]
