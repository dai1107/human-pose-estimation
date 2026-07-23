"""Camera backend benchmarking and device-local backend selection cache."""

from __future__ import annotations

import json
import math
import platform
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CACHE_SCHEMA_VERSION = 1
DEFAULT_BACKEND_CACHE = Path("outputs/camera_backend_cache.json")
CAMERA_BACKENDS = ("default", "dshow", "msmf")
CAMERA_BENCHMARK_CONFIGS = (
    (640, 480, 30.0, "MJPG"),
    (640, 480, 60.0, "MJPG"),
    (1280, 720, 30.0, "MJPG"),
    (640, 480, 30.0, "YUY2"),
    (640, 480, 60.0, "YUY2"),
    (1280, 720, 30.0, "YUY2"),
)


def supported_backend_names(system_platform: str | None = None) -> tuple[str, ...]:
    return CAMERA_BACKENDS if (system_platform or sys.platform).startswith("win") else ("default",)


def backend_api_code(name: str) -> int | None:
    normalized = str(name).strip().lower()
    if normalized == "default":
        return None
    if normalized == "dshow":
        return cv2.CAP_DSHOW
    if normalized == "msmf":
        return cv2.CAP_MSMF
    raise ValueError(f"unknown camera backend: {name}")


def camera_cache_key(
    *,
    camera_index: int,
    width: int,
    height: int,
    fps: float,
    fourcc: str,
    system_platform: str | None = None,
) -> str:
    platform_name = (system_platform or sys.platform).lower()
    normalized_fps = f"{float(fps):g}"
    return (
        f"{platform_name}|camera={int(camera_index)}|"
        f"{int(width)}x{int(height)}@{normalized_fps}|{str(fourcc).strip().upper() or '-'}"
    )


def load_backend_cache(path: str | Path = DEFAULT_BACKEND_CACHE) -> dict[str, Any]:
    cache_path = Path(path)
    try:
        value = json.loads(cache_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {"schema_version": CACHE_SCHEMA_VERSION, "selections": {}}
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != CACHE_SCHEMA_VERSION
        or not isinstance(value.get("selections"), dict)
    ):
        return {"schema_version": CACHE_SCHEMA_VERSION, "selections": {}}
    return value


def select_cached_backend(
    path: str | Path,
    *,
    camera_index: int,
    width: int,
    height: int,
    fps: float,
    fourcc: str,
    system_platform: str | None = None,
) -> str | None:
    key = camera_cache_key(
        camera_index=camera_index,
        width=width,
        height=height,
        fps=fps,
        fourcc=fourcc,
        system_platform=system_platform,
    )
    selection = load_backend_cache(path).get("selections", {}).get(key)
    if not isinstance(selection, Mapping):
        return None
    backend = str(selection.get("backend", "")).lower()
    return backend if backend in supported_backend_names(system_platform) else None


def save_backend_selections(
    results: Sequence[Mapping[str, Any]],
    path: str | Path = DEFAULT_BACKEND_CACHE,
    *,
    system_platform: str | None = None,
) -> dict[str, Any]:
    cache = load_backend_cache(path)
    selections = dict(cache.get("selections", {}))
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        if not result.get("opened") or int(result.get("successful_reads", 0)) <= 0:
            continue
        key = camera_cache_key(
            camera_index=int(result["camera_index"]),
            width=int(result["requested_width"]),
            height=int(result["requested_height"]),
            fps=float(result["requested_fps"]),
            fourcc=str(result["requested_fourcc"]),
            system_platform=system_platform,
        )
        grouped.setdefault(key, []).append(result)
    for key, candidates in grouped.items():
        best = min(candidates, key=_benchmark_score)
        selections[key] = {
            "backend": str(best["backend"]),
            "measured_at": str(best.get("measured_at", "")),
            "actual_fps": best.get("actual_fps"),
            "read_p95_ms": best.get("read_p95_ms"),
            "duplicate_frame_ratio": best.get("duplicate_frame_ratio"),
            "frame_interval_anomaly_ratio": best.get("frame_interval_anomaly_ratio"),
        }
    updated = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "machine": {
            "platform": (system_platform or sys.platform).lower(),
            "platform_release": platform.release(),
        },
        "selections": selections,
    }
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(cache_path)
    return updated


def benchmark_camera_backend(
    *,
    camera_index: int,
    backend: str,
    width: int,
    height: int,
    fps: float,
    fourcc: str,
    duration_seconds: float = 3.0,
    warmup_seconds: float = 0.5,
    sensor_to_photon_ms: float | None = None,
    capture_factory: Callable[..., Any] = cv2.VideoCapture,
    clock: Callable[[], float] = time.perf_counter,
    process_clock: Callable[[], float] = time.process_time,
) -> dict[str, Any]:
    """Measure capture behavior; sensor-to-photon is never estimated from reads."""

    started_at = datetime.now(timezone.utc).isoformat()
    result = _empty_result(
        camera_index=camera_index,
        backend=backend,
        width=width,
        height=height,
        fps=fps,
        fourcc=fourcc,
        measured_at=started_at,
        sensor_to_photon_ms=sensor_to_photon_ms,
    )
    capture = None
    try:
        api_code = backend_api_code(backend)
        capture = (
            capture_factory(camera_index)
            if api_code is None
            else capture_factory(camera_index, api_code)
        )
        if not capture.isOpened():
            result["error"] = "open_failed"
            return result
        result["opened"] = True
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        result.update(_reported_capture_settings(capture, fallback_backend=backend))

        warmup_deadline = clock() + max(0.0, float(warmup_seconds))
        while clock() < warmup_deadline:
            capture.read()

        wall_start = clock()
        cpu_start = process_clock()
        deadline = wall_start + max(0.05, float(duration_seconds))
        read_durations: list[float] = []
        frame_intervals: list[float] = []
        brightness_samples: list[float] = []
        previous_signature: np.ndarray | None = None
        previous_frame_at: float | None = None
        duplicates = 0
        attempts = 0
        successes = 0
        while clock() < deadline:
            read_start = clock()
            ok, frame = capture.read()
            read_end = clock()
            attempts += 1
            read_durations.append(max(0.0, (read_end - read_start) * 1000.0))
            if not ok or frame is None:
                continue
            successes += 1
            if previous_frame_at is not None:
                frame_intervals.append(max(0.0, (read_end - previous_frame_at) * 1000.0))
            previous_frame_at = read_end
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            signature = cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA)
            brightness_samples.append(float(np.mean(signature)))
            if previous_signature is not None:
                difference = float(np.mean(cv2.absdiff(signature, previous_signature)))
                if difference <= 1.0:
                    duplicates += 1
            previous_signature = signature
        wall_end = clock()
        cpu_end = process_clock()
        elapsed = max(1e-9, wall_end - wall_start)
        interval_p50 = _percentile(frame_intervals, 50)
        actual_fps = successes / elapsed
        expected_frames = max(0.0, fps * elapsed)
        result.update(
            {
                "attempted_reads": attempts,
                "successful_reads": successes,
                "open_success_rate": 1.0,
                "read_success_rate": successes / attempts if attempts else 0.0,
                "actual_fps": actual_fps,
                "read_p50_ms": _percentile(read_durations, 50),
                "read_p95_ms": _percentile(read_durations, 95),
                "frame_interval_p50_ms": interval_p50,
                "frame_interval_p95_ms": _percentile(frame_intervals, 95),
                "estimated_drop_ratio": (
                    max(0.0, min(1.0, (expected_frames - successes) / expected_frames))
                    if expected_frames
                    else 0.0
                ),
                "duplicate_frame_ratio": (
                    duplicates / (successes - 1) if successes > 1 else 0.0
                ),
                "mean_brightness": (
                    statistics.fmean(brightness_samples) if brightness_samples else None
                ),
                "frame_interval_anomaly_ratio": _interval_anomaly_ratio(frame_intervals),
                "cpu_ratio": max(0.0, (cpu_end - cpu_start) / elapsed),
            }
        )
        if successes == 0:
            result["error"] = "no_frames"
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:500]
        return result
    finally:
        if capture is not None:
            capture.release()


def benchmark_camera_matrix(
    *,
    camera_index: int = 0,
    backends: Iterable[str] | None = None,
    configs: Iterable[tuple[int, int, float, str]] = CAMERA_BENCHMARK_CONFIGS,
    duration_seconds: float = 3.0,
    warmup_seconds: float = 0.5,
) -> list[dict[str, Any]]:
    selected_backends = tuple(backends or supported_backend_names())
    return [
        benchmark_camera_backend(
            camera_index=camera_index,
            backend=backend,
            width=width,
            height=height,
            fps=fps,
            fourcc=fourcc,
            duration_seconds=duration_seconds,
            warmup_seconds=warmup_seconds,
        )
        for width, height, fps, fourcc in configs
        for backend in selected_backends
    ]


def _empty_result(
    *,
    camera_index: int,
    backend: str,
    width: int,
    height: int,
    fps: float,
    fourcc: str,
    measured_at: str,
    sensor_to_photon_ms: float | None,
) -> dict[str, Any]:
    return {
        "measured_at": measured_at,
        "camera_index": int(camera_index),
        "backend": backend,
        "opened": False,
        "requested_width": int(width),
        "requested_height": int(height),
        "requested_fps": float(fps),
        "requested_fourcc": fourcc,
        "actual_width": None,
        "actual_height": None,
        "reported_fps": None,
        "reported_fourcc": "",
        "reported_backend": "",
        "attempted_reads": 0,
        "successful_reads": 0,
        "open_success_rate": 0.0,
        "read_success_rate": 0.0,
        "actual_fps": 0.0,
        "read_p50_ms": None,
        "read_p95_ms": None,
        "frame_interval_p50_ms": None,
        "frame_interval_p95_ms": None,
        "estimated_drop_ratio": None,
        "duplicate_frame_ratio": None,
        "mean_brightness": None,
        "frame_interval_anomaly_ratio": None,
        "cpu_ratio": None,
        "sensor_to_photon_ms": sensor_to_photon_ms,
        "sensor_to_photon_source": "manual" if sensor_to_photon_ms is not None else None,
        "pose_detection_rate": None,
        "error": "",
    }


def _reported_capture_settings(capture: Any, *, fallback_backend: str) -> dict[str, Any]:
    try:
        backend_name = str(capture.getBackendName())
    except (AttributeError, cv2.error):
        backend_name = fallback_backend
    return {
        "actual_width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "actual_height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "reported_fps": _finite_or_none(capture.get(cv2.CAP_PROP_FPS)),
        "reported_fourcc": decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC)),
        "reported_backend": backend_name,
    }


def decode_fourcc(value: float) -> str:
    try:
        packed = int(round(float(value)))
    except (TypeError, ValueError, OverflowError):
        return ""
    return "".join(chr((packed >> (8 * index)) & 0xFF) for index in range(4)).rstrip("\x00")


def _finite_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _interval_anomaly_ratio(intervals: Sequence[float]) -> float:
    if len(intervals) < 2:
        return 0.0
    median = statistics.median(intervals)
    if median <= 0:
        return 0.0
    return sum(interval > median * 1.8 for interval in intervals) / len(intervals)


def _benchmark_score(result: Mapping[str, Any]) -> tuple[float, ...]:
    requested_fps = max(1.0, float(result.get("requested_fps", 1.0)))
    actual_fps = max(0.0, float(result.get("actual_fps", 0.0)))
    fps_deficit = max(0.0, requested_fps - actual_fps) / requested_fps
    return (
        fps_deficit,
        float(result.get("estimated_drop_ratio") or 0.0),
        float(result.get("duplicate_frame_ratio") or 0.0),
        float(result.get("frame_interval_anomaly_ratio") or 0.0),
        float(result.get("read_p95_ms") or 0.0),
        float(result.get("cpu_ratio") or 0.0),
    )


__all__ = [
    "CAMERA_BACKENDS",
    "CAMERA_BENCHMARK_CONFIGS",
    "DEFAULT_BACKEND_CACHE",
    "backend_api_code",
    "benchmark_camera_backend",
    "benchmark_camera_matrix",
    "camera_cache_key",
    "decode_fourcc",
    "load_backend_cache",
    "save_backend_selections",
    "select_cached_backend",
    "supported_backend_names",
]
