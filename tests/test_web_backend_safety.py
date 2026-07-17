from __future__ import annotations

from typing import Any

import webui.app as web_app
import webui.realtime as web_realtime


class FakeMediaPipeBackend:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


def test_sample_video_backend_disables_native_segmentation_masks(monkeypatch: Any) -> None:
    FakeMediaPipeBackend.calls.clear()
    monkeypatch.setattr(web_app, "MediaPipeBackend", FakeMediaPipeBackend)

    _, resolved = web_app.PoseStreamEngine()._create_backend(
        "auto",
        "farmers_carry",
        "",
    )

    assert resolved == "mediapipe"
    assert FakeMediaPipeBackend.calls[0][1]["output_segmentation_masks"] is False


def test_realtime_backend_disables_native_segmentation_masks(monkeypatch: Any) -> None:
    FakeMediaPipeBackend.calls.clear()
    monkeypatch.setattr(web_realtime, "MediaPipeBackend", FakeMediaPipeBackend)

    _, resolved = web_realtime.default_backend_factory("auto", "farmers_carry")

    assert resolved == "mediapipe"
    assert FakeMediaPipeBackend.calls[0][1]["output_segmentation_masks"] is False
