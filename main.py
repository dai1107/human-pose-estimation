"""Public desktop entry point.

The implementation lives in :mod:`src.realtime` so this module stays a stable
script and packaging target.
"""

from __future__ import annotations

from src.realtime.app import (
    build_pose_frame_from_result,
    main,
    make_output_path,
    next_runtime_backend,
    open_capture,
    parse_args,
    read_capture_frame,
    runtime_backend_switch_allowed,
)
from src.realtime.hyrox_analysis import runtime_hyrox_config_path

__all__ = [
    "build_pose_frame_from_result",
    "main",
    "make_output_path",
    "next_runtime_backend",
    "open_capture",
    "parse_args",
    "read_capture_frame",
    "runtime_backend_switch_allowed",
    "runtime_hyrox_config_path",
]


if __name__ == "__main__":
    raise SystemExit(main())
