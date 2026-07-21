"""Long-running video-loop validation with process and latency telemetry."""

from __future__ import annotations

import ctypes
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Mapping

import cv2
import numpy as np

from hyrox.features import extract_basic_pose_features
from hyrox.registry import create_action_analyzer
from src.backends.mediapipe_backend import MediaPipeBackend
from src.output_schema import versioned_payload
from src.paths import resolve_asset
from src.utils.smoothing import KeypointSmoother


@dataclass(frozen=True)
class EnduranceThresholds:
    min_fps: float = 1.0
    max_p95_latency_ms: float = 1000.0
    max_memory_growth_mb: float = 512.0
    max_read_failure_rate: float = 0.01


@dataclass(frozen=True)
class EnduranceObservation:
    target_duration_seconds: float
    elapsed_seconds: float
    total_frames: int
    pose_detected_frames: int
    pose_detected_rate: float
    average_fps: float
    average_latency_ms: float
    p95_latency_ms: float
    memory_start_mb: float
    memory_end_mb: float
    memory_peak_mb: float
    memory_growth_mb: float
    source_reopen_count: int
    read_failure_count: int
    read_failure_rate: float
    completed: bool
    output_integrity: bool
    final_phase: str


def process_rss_bytes() -> int:
    """Return current RSS without adding a runtime dependency such as psutil."""
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess  # type: ignore[attr-defined]
        get_current_process.restype = ctypes.c_void_p
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo  # type: ignore[attr-defined]
        get_process_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        get_process_memory_info.restype = ctypes.c_int
        process = get_current_process()
        ok = get_process_memory_info(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        if ok:
            return int(counters.WorkingSetSize)
    if sys.platform.startswith("linux"):
        try:
            pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
            return pages * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            pass
    try:
        import resource

        usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return usage if sys.platform == "darwin" else usage * 1024
    except (ImportError, OSError, ValueError):
        return 0


def run_video_endurance(
    *,
    video: str | Path,
    model: str | Path,
    duration_seconds: float,
    action: str = "none",
    camera_view: str = "unknown",
    memory_sample_interval_seconds: float = 1.0,
) -> EnduranceObservation:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0")
    video_path = resolve_asset(video)
    if not video_path.exists():
        raise FileNotFoundError(f"endurance video not found: {video_path}")
    backend = MediaPipeBackend(resolve_asset(model), output_segmentation_masks=False)
    smoother = KeypointSmoother(mode="one-euro", max_missing_frames=5)
    analyzer = None if action == "none" else create_action_analyzer(action, camera_view=camera_view, live_mode=False)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        backend.close()
        raise RuntimeError(f"could not open endurance video: {video_path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    source_fps = source_fps if source_fps > 0 else 30.0
    started = time.perf_counter()
    memory_samples = [process_rss_bytes()]
    last_memory_sample = started
    latencies: list[float] = []
    total_frames = 0
    detected_frames = 0
    source_reopen_count = 0
    read_failure_count = 0
    frame_in_loop = 0
    final_state: Mapping[str, object] = {}
    runtime_finished = started
    try:
        while time.perf_counter() - started < duration_seconds:
            ok, frame = capture.read()
            if not ok or frame is None:
                if frame_in_loop == 0:
                    read_failure_count += 1
                    break
                capture.release()
                capture = cv2.VideoCapture(str(video_path))
                if not capture.isOpened():
                    read_failure_count += 1
                    break
                source_reopen_count += 1
                frame_in_loop = 0
                smoother.reset()
                continue
            frame_started = time.perf_counter()
            total_frames += 1
            frame_in_loop += 1
            timestamp_ms = int(round(total_frames * 1000.0 / source_fps))
            result = smoother.smooth_result(backend.detect(frame, timestamp_ms=timestamp_ms))
            has_pose = bool(result.success and result.keypoints)
            detected_frames += int(has_pose)
            if analyzer is not None:
                features = None
                if has_pose:
                    height, width = frame.shape[:2]
                    features = extract_basic_pose_features(
                        result.keypoints,
                        image_width=width,
                        image_height=height,
                        segmentation_mask=None,
                    )
                final_state = analyzer.attach_view_context(
                    analyzer.update(features if has_pose else None, timestamp_ms=timestamp_ms)
                )
            latencies.append((time.perf_counter() - frame_started) * 1000.0)
            now = time.perf_counter()
            if now - last_memory_sample >= max(0.05, memory_sample_interval_seconds):
                memory_samples.append(process_rss_bytes())
                last_memory_sample = now
    finally:
        runtime_finished = time.perf_counter()
        memory_samples.append(process_rss_bytes())
        capture.release()
        backend.close()
    elapsed = runtime_finished - started
    start_mb = memory_samples[0] / (1024 * 1024)
    end_mb = memory_samples[-1] / (1024 * 1024)
    peak_mb = max(memory_samples) / (1024 * 1024)
    completed = elapsed >= duration_seconds and total_frames > 0
    output_integrity = completed and total_frames == len(latencies)
    return EnduranceObservation(
        target_duration_seconds=float(duration_seconds),
        elapsed_seconds=elapsed,
        total_frames=total_frames,
        pose_detected_frames=detected_frames,
        pose_detected_rate=detected_frames / total_frames if total_frames else 0.0,
        average_fps=total_frames / elapsed if elapsed > 0 else 0.0,
        average_latency_ms=mean(latencies) if latencies else 0.0,
        p95_latency_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
        memory_start_mb=start_mb,
        memory_end_mb=end_mb,
        memory_peak_mb=peak_mb,
        memory_growth_mb=end_mb - start_mb,
        source_reopen_count=source_reopen_count,
        read_failure_count=read_failure_count,
        read_failure_rate=read_failure_count / max(1, total_frames + read_failure_count),
        completed=completed,
        output_integrity=output_integrity,
        final_phase=str(final_state.get("phase", "not_enabled")),
    )


def evaluate_thresholds(
    observation: EnduranceObservation,
    thresholds: EnduranceThresholds,
) -> list[str]:
    failures: list[str] = []
    if not observation.completed:
        failures.append("target duration was not completed")
    if not observation.output_integrity:
        failures.append("frame/latency output integrity check failed")
    if observation.average_fps < thresholds.min_fps:
        failures.append(f"average_fps={observation.average_fps:.3f} < {thresholds.min_fps:.3f}")
    if observation.p95_latency_ms > thresholds.max_p95_latency_ms:
        failures.append(f"p95_latency_ms={observation.p95_latency_ms:.3f} > {thresholds.max_p95_latency_ms:.3f}")
    if observation.memory_growth_mb > thresholds.max_memory_growth_mb:
        failures.append(f"memory_growth_mb={observation.memory_growth_mb:.3f} > {thresholds.max_memory_growth_mb:.3f}")
    if observation.read_failure_rate > thresholds.max_read_failure_rate:
        failures.append(f"read_failure_rate={observation.read_failure_rate:.6f} > {thresholds.max_read_failure_rate:.6f}")
    return failures


def build_endurance_report(
    observation: EnduranceObservation,
    thresholds: EnduranceThresholds,
) -> dict[str, object]:
    failures = evaluate_thresholds(observation, thresholds)
    return versioned_payload(
        "pose_endurance_report",
        {
            "status": "passed" if not failures else "failed",
            "failures": failures,
            "thresholds": asdict(thresholds),
            "observation": asdict(observation),
        },
    )
