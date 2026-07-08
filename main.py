from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import cv2

from src.backends.base import PoseBackend
from src.backends.factory import create_backend
from src.detectors.yolo_person_detector import YoloPersonDetector
from src.fusion.yolo_roi_mediapipe import FusionFrameStats, YoloRoiMediaPipeFusion
from src.realtime.feedback_engine import FeedbackEngine
from src.utils.angle_utils import body_angles
from src.utils.backend_policy import resolve_backend_choice
from src.utils.draw_utils import draw_bbox, draw_pose_result, draw_realtime_overlay
from src.utils.device import resolve_torch_device
from src.utils.metrics import RealtimeMetrics
from src.utils.smoothing import KeypointSmoother


RUNTIME_BACKENDS = ("mediapipe", "yolo-pose")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime-first human keypoint detection baseline.")
    parser.add_argument("--backend", default="auto", choices=("auto", "mediapipe", "yolo-pose"), help="Pose backend. Default: auto.")
    parser.add_argument("--action-type", default="auto", help="Action type for --backend auto, e.g. squat, rowing, ski_erg, or HYROX video stem.")
    parser.add_argument("--model", default="models/pose_landmarker_full.task", help="MediaPipe pose model path.")
    parser.add_argument("--yolo-pose-model", default="yolo11n-pose.pt", help="YOLO Pose model path. Default: yolo11n-pose.pt.")
    parser.add_argument("--yolo-device", default="auto", help="YOLO Pose device, e.g. auto, 0, cuda:0, cpu. Default: auto.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Default: 0.")
    parser.add_argument("--input-video", default="", help="Read frames from a video instead of opening a camera.")
    parser.add_argument("--width", type=int, default=640, help="Requested camera width. Default: 640.")
    parser.add_argument("--height", type=int, default=480, help="Requested camera height. Default: 480.")
    parser.add_argument("--camera-fps", type=float, default=60.0, help="Requested camera FPS. Default: 60.")
    parser.add_argument("--camera-fourcc", default="MJPG", help="Requested camera FourCC. Empty string leaves it unchanged.")
    parser.add_argument("--record", default="", help="Save annotated video to this path.")
    parser.add_argument("--record-raw", default="", help="Save raw input frames to this path.")
    parser.add_argument("--save-metrics", default="", help="Append final metrics to this CSV path.")
    parser.add_argument("--headless", action="store_true", help="Do not open an OpenCV window; useful for input-video batch evaluation.")
    parser.add_argument("--person-detector", default="none", choices=("none", "yolo"), help="Optional person detector. Default: none.")
    parser.add_argument("--detector-model", default="yolo11n.pt", help="YOLO person detector model. Default: yolo11n.pt.")
    parser.add_argument("--detector-device", default="auto", help="YOLO person detector device, e.g. auto, 0, cuda:0, cpu. Default: auto.")
    parser.add_argument("--detector-every-n", type=int, default=5, help="Run YOLO every N frames. Default: 5.")
    parser.add_argument("--bbox-expand", type=float, default=1.25, help="Expand detector bbox by this scale. Default: 1.25.")
    parser.add_argument("--bbox-smoothing", type=float, default=0.6, help="BBox smoothing alpha. Default: 0.6.")
    parser.add_argument("--target-select", default="confidence", choices=("confidence", "area"), help="Select person by confidence or area.")
    parser.add_argument("--fusion", default="none", choices=("none", "yolo-roi-mediapipe"), help="Fusion strategy. Default: none.")
    parser.add_argument("--smoothing", default="one-euro", choices=("none", "ema", "one-euro"), help="Keypoint smoothing mode.")
    parser.add_argument("--ema-alpha", type=float, default=0.6, help="EMA alpha. Default: 0.6.")
    parser.add_argument("--one-euro-min-cutoff", type=float, default=1.0, help="One Euro min cutoff. Default: 1.0.")
    parser.add_argument("--one-euro-beta", type=float, default=0.01, help="One Euro beta. Default: 0.01.")
    parser.add_argument("--one-euro-d-cutoff", type=float, default=1.0, help="One Euro derivative cutoff. Default: 1.0.")
    parser.add_argument("--pose-hold-frames", type=int, default=5, help="Hold the last valid pose for short tracking drops. Use 0 to disable. Default: 5.")
    occlusion_group = parser.add_mutually_exclusive_group()
    occlusion_group.add_argument("--occlusion-guard", dest="occlusion_guard", action="store_true", help="Suppress body-joint jumps near hands. Default.")
    occlusion_group.add_argument("--no-occlusion-guard", dest="occlusion_guard", action="store_false", help="Disable hand/body occlusion guard.")
    mirror_group = parser.add_mutually_exclusive_group()
    mirror_group.add_argument("--mirror", dest="mirror", action="store_true", help="Mirror display for camera input. Default.")
    mirror_group.add_argument("--no-mirror", dest="mirror", action="store_false", help="Disable mirror display.")
    parser.set_defaults(mirror=True)
    parser.set_defaults(occlusion_guard=True)
    return parser.parse_args(argv)


def open_capture(args: argparse.Namespace) -> tuple[cv2.VideoCapture, str, float]:
    if args.input_video:
        capture = cv2.VideoCapture(args.input_video)
        if not capture.isOpened():
            raise RuntimeError(f"could not open input video: {args.input_video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        return capture, "video", fps if fps > 0 else 30.0

    if sys.platform.startswith("win"):
        capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(args.camera)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if args.camera_fourcc.strip():
        fourcc = args.camera_fourcc.strip().upper()
        if len(fourcc) != 4:
            raise RuntimeError("--camera-fourcc must be exactly 4 characters, or empty")
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.camera_fps > 0:
        capture.set(cv2.CAP_PROP_FPS, args.camera_fps)
    if not capture.isOpened():
        raise RuntimeError(f"camera {args.camera} could not be opened")
    fps = capture.get(cv2.CAP_PROP_FPS)
    print(
        f"Camera {args.camera} opened "
        f"(requested FPS: {args.camera_fps:g}, reported FPS: {fps if fps > 0 else 0:.1f}, FourCC: {args.camera_fourcc or 'unchanged'})"
    )
    return capture, "camera", fps if fps > 0 else 30.0


def create_writer(path: str, fps: float, frame_shape: tuple[int, int, int]) -> cv2.VideoWriter:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame_shape[:2]
    suffix = output.suffix.lower()
    fourcc_name = "mp4v" if suffix == ".mp4" else "XVID"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*fourcc_name), max(1.0, min(60.0, fps)), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not create video writer: {output}")
    return writer


def timestamp_for_frame(input_mode: str, started_ns: int, frame_index: int, fps: float) -> int:
    if input_mode == "video":
        return int(round(frame_index * 1000.0 / max(fps, 1.0)))
    return int((time.monotonic_ns() - started_ns) / 1_000_000)


def validate_runtime_args(args: argparse.Namespace, resolved_backend: str) -> None:
    if args.fusion == "yolo-roi-mediapipe" and resolved_backend != "mediapipe":
        raise RuntimeError("--fusion yolo-roi-mediapipe only supports --backend mediapipe")
    if args.fusion == "yolo-roi-mediapipe" and args.person_detector != "yolo":
        raise RuntimeError("--fusion yolo-roi-mediapipe requires --person-detector yolo")
    if resolved_backend != "mediapipe" and args.person_detector != "none":
        raise RuntimeError("--person-detector yolo is only for MediaPipe ROI; use --person-detector none with this backend")


def backend_device_for(args: argparse.Namespace, backend_name: str) -> str:
    return resolve_torch_device(args.yolo_device) if backend_name == "yolo-pose" else "cpu"


def create_runtime_backend(args: argparse.Namespace, backend_name: str) -> tuple[PoseBackend, str]:
    return create_backend(args, backend_name=backend_name), backend_device_for(args, backend_name)


def create_runtime_smoother(args: argparse.Namespace) -> KeypointSmoother:
    return KeypointSmoother(
        mode=args.smoothing,
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
    return True, ""


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    backend = None
    fusion_runner = None
    capture = None
    record_writer = None
    raw_writer = None
    resolved_backend = resolve_backend_choice(args.backend, action_type=args.action_type, input_video=args.input_video)
    runtime_backend_device = backend_device_for(args, resolved_backend)
    runtime_detector_device = resolve_torch_device(args.detector_device) if args.person_detector == "yolo" else "none"
    metrics = RealtimeMetrics(
        backend=resolved_backend,
        smoothing=args.smoothing,
        input_name=args.input_video or "camera",
        person_detector=args.person_detector,
        fusion=args.fusion,
        detector_every_n=args.detector_every_n,
        backend_device=runtime_backend_device,
        detector_device=runtime_detector_device,
    )
    try:
        validate_runtime_args(args, resolved_backend)
        print(f"Resolved backend: {resolved_backend} (requested: {args.backend}, action_type: {args.action_type})")
        backend, runtime_backend_device = create_runtime_backend(args, resolved_backend)
        yolo_detector = None
        if args.fusion == "yolo-roi-mediapipe":
            yolo_detector = YoloPersonDetector(
                model_path=args.detector_model,
                every_n=args.detector_every_n,
                bbox_expand=args.bbox_expand,
                bbox_smoothing=args.bbox_smoothing,
                target_select=args.target_select,
                device=runtime_detector_device,
            )
            fusion_runner = YoloRoiMediaPipeFusion(backend, yolo_detector)
        capture, input_mode, source_fps = open_capture(args)
        smoother = create_runtime_smoother(args)
        feedback_engine = FeedbackEngine()
        started_ns = time.monotonic_ns()
        frame_index = 0
        window_name = "Realtime Keypoint Baseline"
        status_message = ""
        status_until = 0.0
        if not args.headless:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            ok, raw_frame = capture.read()
            if not ok or raw_frame is None:
                break
            frame_started = time.perf_counter()
            frame_index += 1
            display_frame = cv2.flip(raw_frame, 1) if input_mode == "camera" and args.mirror else raw_frame.copy()

            if args.record_raw:
                if raw_writer is None:
                    raw_writer = create_writer(args.record_raw, source_fps, raw_frame.shape)
                raw_writer.write(raw_frame)

            timestamp_ms = timestamp_for_frame(input_mode, started_ns, frame_index, source_fps)
            if fusion_runner is not None:
                result, fusion_stats = fusion_runner.detect(display_frame, timestamp_ms=timestamp_ms)
            else:
                result = backend.detect(display_frame, timestamp_ms=timestamp_ms)
                fusion_stats = FusionFrameStats()
            result = smoother.smooth_result(result)
            angles = body_angles(result)
            feedback = feedback_engine.update(result, angles)
            frame_finished = time.perf_counter()
            snapshot = metrics.update(
                result,
                angles,
                frame_started=frame_started,
                frame_finished=frame_finished,
                roi_enabled=fusion_stats.roi_enabled,
                roi_success=fusion_stats.roi_success,
                yolo_detection_time_ms=fusion_stats.yolo_detection_time_ms,
                bbox_reused=fusion_stats.bbox_reused,
                bbox_lost=fusion_stats.bbox_lost,
                fallback_to_full_frame=fusion_stats.fallback_to_full_frame,
                source_model_distribution=getattr(fusion_stats, "source_model_distribution", None),
            )

            annotated = display_frame.copy()
            draw_pose_result(annotated, result)
            draw_bbox(annotated, result.bbox)
            overlay_status = status_message if time.perf_counter() < status_until else ""
            draw_realtime_overlay(
                annotated,
                backend=resolved_backend,
                fusion=args.fusion,
                person_detector=args.person_detector,
                detector_every_n=args.detector_every_n,
                smoothing=args.smoothing,
                input_mode=input_mode,
                result=result,
                metrics=snapshot,
                feedback=feedback,
                recording=bool(args.record),
                raw_recording=bool(args.record_raw),
                angles=angles,
                status_message=overlay_status,
            )

            if args.record:
                if record_writer is None:
                    record_writer = create_writer(args.record, source_fps, annotated.shape)
                record_writer.write(annotated)

            if not args.headless:
                cv2.imshow(window_name, annotated)
                delay = 1 if input_mode == "camera" else max(1, int(round(1000.0 / max(source_fps, 1.0))))
                key = cv2.waitKey(delay) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("b"), ord("B")):
                    allowed, reason = runtime_backend_switch_allowed(args)
                    if not allowed:
                        status_message = f"switch disabled: {reason}"
                        status_until = time.perf_counter() + 2.5
                        print(f"Backend switch ignored: {reason}")
                        continue

                    target_backend = next_runtime_backend(resolved_backend)
                    print(f"Switching backend: {resolved_backend} -> {target_backend}")
                    try:
                        new_backend, new_backend_device = create_runtime_backend(args, target_backend)
                    except Exception as exc:
                        status_message = f"switch failed: {target_backend}"
                        status_until = time.perf_counter() + 2.5
                        print(f"Backend switch failed: {exc}", file=sys.stderr)
                        continue

                    if backend is not None:
                        try:
                            backend.close()
                        except Exception as exc:
                            print(f"WARN: failed to close old backend: {exc}", file=sys.stderr)
                    backend = new_backend
                    resolved_backend = target_backend
                    runtime_backend_device = new_backend_device
                    metrics.set_backend(resolved_backend, runtime_backend_device)
                    smoother = create_runtime_smoother(args)
                    status_message = f"backend switched to {resolved_backend}"
                    status_until = time.perf_counter() + 2.5
                    print(f"Backend switched: {resolved_backend} (device: {runtime_backend_device})")

        if args.save_metrics:
            metrics.write_csv(args.save_metrics)
        print("Metrics summary:")
        for line in metrics.summary_lines():
            print(f"  {line}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if record_writer is not None:
            record_writer.release()
        if raw_writer is not None:
            raw_writer.release()
        if fusion_runner is not None and hasattr(fusion_runner, "close"):
            fusion_runner.close()
        if backend is not None:
            backend.close()
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
