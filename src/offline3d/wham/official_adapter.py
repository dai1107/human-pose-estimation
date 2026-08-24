from __future__ import annotations

"""Run the official WHAM demo and export the stable JSON contract used here.

This module is intentionally executed inside WHAM's isolated Python
environment.  The product process never imports torch, mmcv or WHAM itself.
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SMPL_JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine",
    "left_knee",
    "right_knee",
    "spine1",
    "left_ankle",
    "right_ankle",
    "spine2",
    "left_foot_index",
    "right_foot_index",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)


def _array(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _video_metadata(path: Path) -> tuple[float, int, int, int]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"WHAM adapter cannot open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if not math.isfinite(fps) or fps <= 0:
        raise RuntimeError("WHAM adapter received a video with invalid FPS")
    return fps, frame_count, width, height


def _normalize_vertices(value: Any) -> Any:
    import numpy as np

    vertices = np.asarray(_array(value), dtype=np.float64)
    while vertices.ndim > 3 and vertices.shape[0] == 1:
        vertices = vertices[0]
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"unexpected WHAM vertices shape: {vertices.shape}")
    return vertices


def _normalize_frames(value: Any, *, width: int | None = None) -> Any:
    import numpy as np

    array = np.asarray(_array(value))
    while array.ndim > (width or 1) and array.shape[0] == 1:
        array = array[0]
    return array


def _track_length(track: Mapping[str, Any]) -> int:
    ids = _normalize_frames(track.get("frame_ids", track.get("frame_id", [])))
    return int(ids.size)


def _select_track(results: Mapping[Any, Any]) -> tuple[Any, Mapping[str, Any]]:
    candidates = [
        (identifier, track)
        for identifier, track in results.items()
        if isinstance(track, Mapping) and _track_length(track) > 0
    ]
    if not candidates:
        raise RuntimeError("WHAM did not return a usable person track")
    return max(candidates, key=lambda item: _track_length(item[1]))


def _tracking_confidences(
    tracking: Mapping[Any, Any], track_id: Any, frame_count: int
) -> list[float]:
    import numpy as np

    track = tracking.get(track_id)
    if track is None:
        track = tracking.get(str(track_id))
    if not isinstance(track, Mapping):
        return [0.0] * frame_count
    keypoints = np.asarray(_array(track.get("keypoints", [])), dtype=np.float64)
    if keypoints.ndim != 3 or keypoints.shape[-1] < 3:
        return [0.0] * frame_count
    values = np.nanmean(keypoints[..., 2], axis=1)
    return [
        float(max(0.0, min(1.0, value))) if math.isfinite(float(value)) else 0.0
        for value in values[:frame_count]
    ]


def _regress_joints(wham_root: Path, vertices: Any) -> Any:
    import numpy as np

    regressor_path = wham_root / "dataset" / "body_models" / "J_regressor_wham.npy"
    if not regressor_path.is_file():
        raise FileNotFoundError(f"missing WHAM joint regressor: {regressor_path}")
    regressor = np.load(regressor_path)
    if regressor.ndim != 2 or regressor.shape[1] != vertices.shape[1]:
        raise ValueError(
            "WHAM joint regressor and vertex topology are incompatible: "
            f"regressor={regressor.shape}, vertices={vertices.shape}"
        )
    return np.einsum("jv,fvc->fjc", regressor, vertices)


def _joint_names(count: int) -> list[str]:
    return [
        SMPL_JOINT_NAMES[index]
        if index < len(SMPL_JOINT_NAMES)
        else f"wham_joint_{index}"
        for index in range(count)
    ]


def _project_joints(joints: Any, width: int, height: int) -> list[dict[str, float | str]]:
    focal = math.sqrt(float(width * width + height * height))
    names = _joint_names(int(joints.shape[0]))
    projected: list[dict[str, float | str]] = []
    for name, point in zip(names, joints):
        x, y, z = (float(value) for value in point[:3])
        if not all(math.isfinite(value) for value in (x, y, z)) or z <= 1e-6:
            continue
        projected.append(
            {
                "name": name,
                "x": (focal * x / z + width / 2.0) / width,
                "y": (focal * y / z + height / 2.0) / height,
            }
        )
    return projected


def _find_native_output(native_root: Path) -> Path:
    candidates = list(native_root.rglob("wham_output.pkl"))
    if not candidates:
        raise RuntimeError(f"official WHAM did not create wham_output.pkl under {native_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def run_official_wham(
    *,
    wham_root: Path,
    video_path: Path,
    output_dir: Path,
    estimate_local_only: bool,
    run_smplify: bool,
) -> Path:
    native_root = output_dir / "native"
    native_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(wham_root / "demo.py"),
        "--video",
        str(video_path),
        "--output_pth",
        str(native_root),
        "--save_pkl",
    ]
    if estimate_local_only:
        command.append("--estimate_local_only")
    if run_smplify:
        command.append("--run_smplify")
    completed = subprocess.run(command, cwd=wham_root, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"official WHAM demo exited with {completed.returncode}")
    return _find_native_output(native_root)


def convert_native_output(
    *,
    wham_root: Path,
    native_output: Path,
    video_path: Path,
) -> dict[str, Any]:
    import joblib
    import numpy as np

    fps, source_frame_count, width, height = _video_metadata(video_path)
    raw_results = joblib.load(native_output)
    if not isinstance(raw_results, Mapping):
        raise ValueError("WHAM native output root must be a mapping")
    track_id, track = _select_track(raw_results)
    frame_ids = np.asarray(
        _array(track.get("frame_ids", track.get("frame_id", []))), dtype=np.int64
    ).reshape(-1)
    vertices = _normalize_vertices(track.get("verts", track.get("verts_cam")))
    usable = min(len(frame_ids), len(vertices))
    if usable <= 0:
        raise RuntimeError("selected WHAM track contains no frames")
    frame_ids = frame_ids[:usable]
    vertices = vertices[:usable]
    joints = _regress_joints(wham_root, vertices)
    pose_camera = _normalize_frames(track.get("pose", []))
    pose_world = _normalize_frames(track.get("pose_world", []))
    translations = _normalize_frames(track.get("trans", []))
    trajectories = _normalize_frames(track.get("trans_world", []))

    tracking_path = native_output.parent / "tracking_results.pth"
    tracking = joblib.load(tracking_path) if tracking_path.is_file() else {}
    confidences = _tracking_confidences(
        tracking if isinstance(tracking, Mapping) else {}, track_id, usable
    )
    slam_path = native_output.parent / "slam_results.pth"
    slam = np.asarray(joblib.load(slam_path)) if slam_path.is_file() else np.empty((0, 7))
    names = _joint_names(int(joints.shape[1]))

    frames: list[dict[str, Any]] = []
    for position, frame_id in enumerate(frame_ids):
        joint_values = joints[position]
        camera_pose = pose_camera[position] if len(pose_camera) > position else []
        world_pose = pose_world[position] if len(pose_world) > position else camera_pose
        translation = translations[position] if len(translations) > position else []
        trajectory = trajectories[position] if len(trajectories) > position else []
        camera_motion = slam[int(frame_id)] if 0 <= int(frame_id) < len(slam) else []
        frames.append(
            {
                "frame_index": int(frame_id),
                "timestamp_ms": float(frame_id) * 1000.0 / fps,
                "joints_3d": {
                    name: [float(value) for value in point]
                    for name, point in zip(names, joint_values)
                },
                "smpl_pose": _array(world_pose),
                "body_orientation": _array(world_pose[:3]),
                "body_translation": _array(translation),
                "global_trajectory": _array(trajectory),
                "camera_motion": _array(camera_motion),
                "confidence": confidences[position] if position < len(confidences) else 0.0,
                "projected_joints_2d": _project_joints(joint_values, width, height),
            }
        )

    return {
        "schema_version": "wham-official-adapter-v1",
        "reference_source": "Official WHAM reconstructed 3D",
        "source_fps": fps,
        "joint_names": names,
        "coordinate_system": "wham_camera_joints_with_world_root_trajectory",
        "angle_definition": "SMPL-regressed three-point joints",
        "trajectory_confidence": (
            float(np.mean(confidences)) if confidences else None
        ),
        "frames": frames,
        "metadata": {
            "official_repository": "https://github.com/yohanshin/WHAM",
            "selected_track_id": str(track_id),
            "selected_track_frames": usable,
            "detected_track_count": len(raw_results),
            "source_frame_count": source_frame_count,
            "source_width": width,
            "source_height": height,
            "focal_length_px": math.sqrt(float(width * width + height * height)),
            "native_output": str(native_output),
            "formal_hyrox_rule_replacement_allowed": False,
        },
        "warnings": [
            "WHAM is reconstructed reference, not ground truth",
            "Longest person track was selected for single-athlete HYROX analysis",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Official WHAM to pose-estimation adapter")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wham-root", required=True)
    parser.add_argument("--native-output")
    parser.add_argument("--estimate-local-only", action="store_true")
    parser.add_argument("--run-smplify", action="store_true")
    args = parser.parse_args(argv)

    wham_root = Path(args.wham_root).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    if not (wham_root / "demo.py").is_file():
        raise FileNotFoundError(f"official WHAM checkout is incomplete: {wham_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    native_output = (
        Path(args.native_output).expanduser().resolve()
        if args.native_output
        else run_official_wham(
            wham_root=wham_root,
            video_path=video_path,
            output_dir=output_dir,
            estimate_local_only=bool(args.estimate_local_only),
            run_smplify=bool(args.run_smplify),
        )
    )
    payload = convert_native_output(
        wham_root=wham_root,
        native_output=native_output,
        video_path=video_path,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
