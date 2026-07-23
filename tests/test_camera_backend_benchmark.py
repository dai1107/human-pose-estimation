from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.camera.backend_benchmark import (
    benchmark_camera_backend,
    camera_cache_key,
    load_backend_cache,
    save_backend_selections,
    select_cached_backend,
    supported_backend_names,
)


class AdvancingClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class FakeCapture:
    def __init__(self, clock: AdvancingClock, *, opened: bool = True) -> None:
        self.clock = clock
        self.opened = opened
        self.released = False
        self.frame_index = 0
        self.properties = {
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*"MJPG")),
        }

    def isOpened(self) -> bool:
        return self.opened

    def set(self, name: int, value: float) -> bool:
        self.properties[name] = float(value)
        return True

    def get(self, name: int) -> float:
        return self.properties.get(name, 0.0)

    def getBackendName(self) -> str:
        return "FAKE"

    def read(self) -> tuple[bool, np.ndarray]:
        self.clock.value += 1 / 30
        self.frame_index += 1
        return True, np.full((48, 64, 3), self.frame_index % 255, dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def test_supported_camera_backends_are_platform_scoped() -> None:
    assert supported_backend_names("win32") == ("default", "dshow", "msmf")
    assert supported_backend_names("linux") == ("default",)


def test_camera_backend_benchmark_reports_measured_metrics_without_fake_photon_data() -> None:
    clock = AdvancingClock()
    capture = FakeCapture(clock)

    result = benchmark_camera_backend(
        camera_index=0,
        backend="default",
        width=640,
        height=480,
        fps=30,
        fourcc="MJPG",
        duration_seconds=0.2,
        warmup_seconds=0,
        capture_factory=lambda *_args: capture,
        clock=clock,
        process_clock=lambda: 0.0,
    )

    assert result["opened"] is True
    assert result["successful_reads"] > 0
    assert result["actual_fps"] > 0
    assert result["read_p50_ms"] is not None
    assert result["read_p95_ms"] is not None
    assert result["mean_brightness"] is not None
    assert result["sensor_to_photon_ms"] is None
    assert result["sensor_to_photon_source"] is None
    assert result["pose_detection_rate"] is None
    assert capture.released is True


def test_backend_cache_selects_best_result_for_exact_device_configuration(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "camera-cache.json"
    base: dict[str, Any] = {
        "measured_at": "2026-07-23T00:00:00+00:00",
        "camera_index": 1,
        "opened": True,
        "requested_width": 640,
        "requested_height": 480,
        "requested_fps": 60.0,
        "requested_fourcc": "MJPG",
        "successful_reads": 100,
        "estimated_drop_ratio": 0.1,
        "duplicate_frame_ratio": 0.02,
        "frame_interval_anomaly_ratio": 0.03,
        "read_p95_ms": 20.0,
        "cpu_ratio": 0.1,
    }
    default = {**base, "backend": "default", "actual_fps": 42.0}
    msmf = {**base, "backend": "msmf", "actual_fps": 58.0, "read_p95_ms": 17.0}

    saved = save_backend_selections(
        [default, msmf],
        cache_path,
        system_platform="win32",
    )

    key = camera_cache_key(
        camera_index=1,
        width=640,
        height=480,
        fps=60,
        fourcc="MJPG",
        system_platform="win32",
    )
    assert saved["selections"][key]["backend"] == "msmf"
    assert load_backend_cache(cache_path)["selections"][key]["backend"] == "msmf"
    assert select_cached_backend(
        cache_path,
        camera_index=1,
        width=640,
        height=480,
        fps=60,
        fourcc="MJPG",
        system_platform="win32",
    ) == "msmf"
    assert select_cached_backend(
        cache_path,
        camera_index=1,
        width=1280,
        height=720,
        fps=30,
        fourcc="MJPG",
        system_platform="win32",
    ) is None
