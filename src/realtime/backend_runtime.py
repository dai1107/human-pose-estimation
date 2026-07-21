"""Backend creation, validation, and hot-switch policy."""

from __future__ import annotations

import argparse

from src.backends.base import PoseBackend
from src.backends.catalog import is_experimental_backend
from src.backends.factory import create_backend
from src.product_pose import RealtimeSmoothingConfig
from src.runtime_logging import AppError, ExitCode
from src.utils.device import resolve_torch_device
from src.utils.smoothing import KeypointSmoother

RUNTIME_BACKENDS = ("mediapipe", "yolo-pose")


def validate_runtime_args(args: argparse.Namespace, resolved_backend: str) -> None:
    experimental_requested = (
        is_experimental_backend(resolved_backend)
        or args.fusion != "none"
        or args.person_detector != "none"
    )
    if experimental_requested and not args.experimental_backends and not args.input_video:
        raise AppError(
            "CFG003",
            "实时产品模式只支持 MediaPipe；实验后端需显式添加 --experimental-backends",
            exit_code=ExitCode.CONFIG_ERROR,
        )
    if args.fusion == "yolo-roi-mediapipe" and resolved_backend != "mediapipe":
        raise AppError("CFG003", "--fusion yolo-roi-mediapipe 只支持 --backend mediapipe", exit_code=ExitCode.CONFIG_ERROR)
    if args.fusion == "yolo-roi-mediapipe" and args.person_detector != "yolo":
        raise AppError("CFG003", "--fusion yolo-roi-mediapipe 需要 --person-detector yolo", exit_code=ExitCode.CONFIG_ERROR)
    if resolved_backend != "mediapipe" and args.person_detector != "none":
        raise AppError("CFG003", "--person-detector yolo 仅用于 MediaPipe ROI；当前后端应使用 none", exit_code=ExitCode.CONFIG_ERROR)


def backend_device_for(args: argparse.Namespace, backend_name: str) -> str:
    return resolve_torch_device(args.yolo_device) if backend_name == "yolo-pose" else "cpu"


def create_runtime_backend(args: argparse.Namespace, backend_name: str) -> tuple[PoseBackend, str]:
    return create_backend(args, backend_name=backend_name), backend_device_for(args, backend_name)


def create_runtime_smoother(
    args: argparse.Namespace,
    config: RealtimeSmoothingConfig | None = None,
) -> KeypointSmoother:
    config = RealtimeSmoothingConfig() if config is None else config
    return KeypointSmoother.from_config(
        config,
        mode=args.smoothing,
        profile=args.smoothing_profile,
        ema_alpha=args.ema_alpha,
        one_euro_min_cutoff=args.one_euro_min_cutoff,
        one_euro_beta=args.one_euro_beta,
        one_euro_d_cutoff=args.one_euro_d_cutoff,
        max_missing_frames=args.pose_hold_frames,
        occlusion_guard=args.occlusion_guard,
    )


def next_runtime_backend(current_backend: str) -> str:
    if current_backend not in RUNTIME_BACKENDS:
        raise ValueError(f"backend switching only supports: {', '.join(RUNTIME_BACKENDS)}")
    return "yolo-pose" if current_backend == "mediapipe" else "mediapipe"


def runtime_backend_switch_allowed(args: argparse.Namespace) -> tuple[bool, str]:
    if args.fusion != "none":
        return False, "fusion must be none"
    if args.person_detector != "none":
        return False, "person_detector must be none"
    if not args.experimental_backends:
        return False, "experimental_backends must be enabled"
    return True, ""
