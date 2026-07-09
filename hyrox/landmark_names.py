MEDIAPIPE_POSE_33_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

LANDMARK_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(MEDIAPIPE_POSE_33_NAMES)
}

NOSE = LANDMARK_INDEX["nose"]
LEFT_SHOULDER = LANDMARK_INDEX["left_shoulder"]
RIGHT_SHOULDER = LANDMARK_INDEX["right_shoulder"]
LEFT_ELBOW = LANDMARK_INDEX["left_elbow"]
RIGHT_ELBOW = LANDMARK_INDEX["right_elbow"]
LEFT_WRIST = LANDMARK_INDEX["left_wrist"]
RIGHT_WRIST = LANDMARK_INDEX["right_wrist"]
LEFT_HIP = LANDMARK_INDEX["left_hip"]
RIGHT_HIP = LANDMARK_INDEX["right_hip"]
LEFT_KNEE = LANDMARK_INDEX["left_knee"]
RIGHT_KNEE = LANDMARK_INDEX["right_knee"]
LEFT_ANKLE = LANDMARK_INDEX["left_ankle"]
RIGHT_ANKLE = LANDMARK_INDEX["right_ankle"]
LEFT_HEEL = LANDMARK_INDEX["left_heel"]
RIGHT_HEEL = LANDMARK_INDEX["right_heel"]
LEFT_FOOT_INDEX = LANDMARK_INDEX["left_foot_index"]
RIGHT_FOOT_INDEX = LANDMARK_INDEX["right_foot_index"]

HYROX_CORE_LANDMARKS: tuple[str, ...] = (
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
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

__all__ = [
    "MEDIAPIPE_POSE_33_NAMES",
    "LANDMARK_INDEX",
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
    "HYROX_CORE_LANDMARKS",
]
