"""Command-line contract for the consolidated desktop runtime."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from hyrox.action_names import HYROX_ACTION_NAMES
from hyrox.view_policy import CAMERA_VIEWS
from src.runtime_hand import DEFAULT_HAND_DETECT_WIDTH, DEFAULT_MAX_HAND_DETECT_FPS
from src.version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MediaPipe product runtime for realtime pose and HYROX analysis."
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--log-dir", default="outputs/logs", help="Directory for rolling application logs. Default: outputs/logs.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs and tracebacks.")
    parser.add_argument(
        "--backend",
        default="mediapipe",
        choices=("auto", "mediapipe", "yolo-pose"),
        help=(
            "Pose backend. Product support: mediapipe (default); auto is a "
            "compatibility alias for mediapipe; yolo-pose is experimental."
        ),
    )
    parser.add_argument(
        "--action-type",
        default="auto",
        help="Compatibility metadata for offline experiments; product auto always resolves to MediaPipe.",
    )
    parser.add_argument(
        "--experimental-backends",
        action="store_true",
        help="Enable explicit experimental YOLO/fusion use and the desktop backend hotkey.",
    )
    parser.add_argument("--model", default="models/pose_landmarker_full.task", help="MediaPipe pose model path.")
    parser.add_argument("--yolo-pose-model", default="yolo11n-pose.pt", help="Experimental YOLO Pose model path. Default: yolo11n-pose.pt.")
    parser.add_argument("--yolo-device", default="auto", help="YOLO Pose device, e.g. auto, 0, cuda:0, cpu. Default: auto.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Default: 0.")
    parser.add_argument("--input-video", default="", help="Read frames from a video instead of opening a camera.")
    parser.add_argument("--width", type=int, default=640, help="Requested camera width. Default: 640.")
    parser.add_argument("--height", type=int, default=480, help="Requested camera height. Default: 480.")
    parser.add_argument("--camera-fps", type=float, default=60.0, help="Requested camera FPS. Default: 60.")
    parser.add_argument("--camera-fourcc", default="MJPG", help="Requested camera FourCC. Empty string leaves it unchanged.")
    parser.add_argument("--landmark-profile", default="full", choices=("full", "no-face", "upper-body", "lower-body"), help="Landmark display profile. Default: full.")
    parser.add_argument("--show-hands", action="store_true", help="Show supplemental five-finger hand landmarks at startup.")
    parser.add_argument("--hand-model", default="models/hand_landmarker.task", help="MediaPipe hand model path.")
    parser.add_argument("--hand-detect-width", type=int, default=DEFAULT_HAND_DETECT_WIDTH, help=f"Resize hand detector input to this width. Default: {DEFAULT_HAND_DETECT_WIDTH}.")
    parser.add_argument("--max-hand-detect-fps", type=float, default=DEFAULT_MAX_HAND_DETECT_FPS, help=f"Maximum hand detector submissions per second. Default: {DEFAULT_MAX_HAND_DETECT_FPS:g}.")
    parser.add_argument("--max-hands", type=int, default=2, help="Maximum number of hands to detect. Default: 2.")
    parser.add_argument("--record", default="", help="Save annotated video to this path.")
    parser.add_argument("--record-raw", default="", help="Save raw input frames to this path.")
    parser.add_argument("--save-metrics", default="", help="Append final metrics to this CSV path.")
    parser.add_argument("--save-dir", default="outputs", help="Directory for sessions, screenshots, and recordings. Default: outputs.")
    parser.add_argument("--headless", action="store_true", help="Do not open an OpenCV window; useful for input-video batch evaluation.")
    parser.add_argument("--normalized-pose-debug", action="store_true", help="Print the unified NormalizedPose summary every 30 frames. Default: off.")
    parser.add_argument("--hyrox-debug", action="store_true", help="Show HYROX debug pose features overlay. Default: off.")
    parser.add_argument("--hyrox-action", default="none", choices=("none", *HYROX_ACTION_NAMES), help="Enable HYROX action analysis. Default: none.")
    parser.add_argument("--hyrox-sensitivity", default="medium", choices=("low", "medium", "high"), help="HYROX action sensitivity. Default: medium.")
    parser.add_argument("--hyrox-config", default="", help="HYROX analyzer config path. Empty selects the action-specific default.")
    parser.add_argument("--metrics-overlay", action="store_true", help="Show kinematic metrics panel at startup.")
    parser.add_argument("--session-autostart", action="store_true", help="Start a kinematic data session automatically.")
    parser.add_argument("--camera-view", default="unknown", choices=CAMERA_VIEWS, help="Camera view used by view-sensitive evaluation. Default: unknown.")
    parser.add_argument("--person-detector", default="none", choices=("none", "yolo"), help="Experimental optional person detector. Default: none.")
    parser.add_argument("--detector-model", default="yolo11n.pt", help="YOLO person detector model. Default: yolo11n.pt.")
    parser.add_argument("--detector-device", default="auto", help="YOLO person detector device. Default: auto.")
    parser.add_argument("--detector-every-n", type=int, default=5, help="Run YOLO every N frames. Default: 5.")
    parser.add_argument("--bbox-expand", type=float, default=1.25, help="Expand detector bbox by this scale. Default: 1.25.")
    parser.add_argument("--bbox-smoothing", type=float, default=0.6, help="BBox smoothing alpha. Default: 0.6.")
    parser.add_argument("--target-select", default="tracking", choices=("tracking", "confidence", "area"), help="Athlete target selection policy.")
    parser.add_argument("--fusion", default="none", choices=("none", "yolo-roi-mediapipe"), help="Experimental fusion strategy. Default: none.")
    parser.add_argument("--smoothing", default="one-euro", choices=("none", "ema", "one-euro"), help="Keypoint smoothing mode.")
    parser.add_argument("--ema-alpha", type=float, default=0.6, help="EMA alpha. Default: 0.6.")
    parser.add_argument(
        "--smoothing-profile",
        choices=("stable", "balanced", "responsive"),
        default=None,
        help="One Euro profile. Default: product realtime_smoothing.profile.",
    )
    parser.add_argument("--one-euro-min-cutoff", type=float, default=None, help="Override the selected One Euro min cutoff.")
    parser.add_argument("--one-euro-beta", type=float, default=None, help="Override the selected One Euro beta.")
    parser.add_argument("--one-euro-d-cutoff", type=float, default=None, help="Override the selected One Euro derivative cutoff.")
    parser.add_argument("--pose-hold-frames", type=int, default=5, help="Hold the last valid pose for short tracking drops. Default: 5.")
    occlusion_group = parser.add_mutually_exclusive_group()
    occlusion_group.add_argument("--occlusion-guard", dest="occlusion_guard", action="store_true", help="Suppress body-joint jumps near hands. Default.")
    occlusion_group.add_argument("--no-occlusion-guard", dest="occlusion_guard", action="store_false", help="Disable hand/body occlusion guard.")
    mirror_group = parser.add_mutually_exclusive_group()
    mirror_group.add_argument("--mirror", dest="mirror", action="store_true", help="Mirror display for camera input. Default.")
    mirror_group.add_argument("--no-mirror", dest="mirror", action="store_false", help="Disable mirror display.")
    parser.set_defaults(mirror=True, occlusion_guard=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)
