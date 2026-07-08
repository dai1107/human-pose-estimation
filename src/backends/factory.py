from __future__ import annotations

from argparse import Namespace

from src.backends.base import PoseBackend
from src.utils.device import resolve_torch_device


def create_backend(args: Namespace, backend_name: str | None = None) -> PoseBackend:
    name = backend_name or args.backend
    if name == "mediapipe":
        from src.backends.mediapipe_backend import MediaPipeBackend

        return MediaPipeBackend(args.model)
    if name == "yolo-pose":
        from src.backends.yolo_pose_backend import YoloPoseBackend

        return YoloPoseBackend(
            args.yolo_pose_model,
            target_select=args.target_select,
            device=resolve_torch_device(args.yolo_device),
        )
    raise ValueError(f"unknown backend: {name}")
