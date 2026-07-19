from __future__ import annotations

from typing import Any

import webui.app as web_app
import webui.realtime as web_realtime


class FakeMediaPipeBackend:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


class FakeYoloGuidedMediaPipeBackend:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


class FakeYoloRtmwBackend:
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


def test_lunge_sample_uses_yolo_guided_mediapipe(monkeypatch: Any) -> None:
    FakeYoloGuidedMediaPipeBackend.calls.clear()
    monkeypatch.setattr(
        web_app,
        "YoloGuidedMediaPipeBackend",
        FakeYoloGuidedMediaPipeBackend,
    )

    _, resolved = web_app.PoseStreamEngine()._create_backend(
        "yolo-pose",
        "lunge",
        "",
        target_select="tracking",
    )

    assert resolved == "yolo-guided-mediapipe"
    assert FakeYoloGuidedMediaPipeBackend.calls[0][1]["target_select"] == "tracking"


def test_lunge_mediapipe_uses_yolo_identity_lock(monkeypatch: Any) -> None:
    FakeYoloGuidedMediaPipeBackend.calls.clear()
    monkeypatch.setattr(
        web_app,
        "YoloGuidedMediaPipeBackend",
        FakeYoloGuidedMediaPipeBackend,
    )

    _, resolved = web_app.PoseStreamEngine()._create_backend(
        "mediapipe",
        "lunge",
        "",
        target_select="tracking",
    )

    assert resolved == "yolo-guided-mediapipe"
    assert FakeYoloGuidedMediaPipeBackend.calls[0][1]["target_select"] == "tracking"


def test_realtime_lunge_yolo_uses_yolo_guided_mediapipe(monkeypatch: Any) -> None:
    FakeYoloGuidedMediaPipeBackend.calls.clear()
    monkeypatch.setattr(
        web_realtime,
        "YoloGuidedMediaPipeBackend",
        FakeYoloGuidedMediaPipeBackend,
    )

    _, resolved = web_realtime.default_backend_factory("yolo-pose", "lunge")

    assert resolved == "yolo-guided-mediapipe"
    assert FakeYoloGuidedMediaPipeBackend.calls[0][1]["target_select"] == "tracking"


def test_realtime_lunge_mediapipe_uses_yolo_identity_lock(
    monkeypatch: Any,
) -> None:
    FakeYoloGuidedMediaPipeBackend.calls.clear()
    monkeypatch.setattr(
        web_realtime,
        "YoloGuidedMediaPipeBackend",
        FakeYoloGuidedMediaPipeBackend,
    )

    _, resolved = web_realtime.default_backend_factory("mediapipe", "lunge")

    assert resolved == "yolo-guided-mediapipe"
    assert FakeYoloGuidedMediaPipeBackend.calls[0][1]["target_select"] == "tracking"


def test_sample_can_select_rtmw_wholebody(monkeypatch: Any) -> None:
    FakeYoloRtmwBackend.calls.clear()
    monkeypatch.setattr(web_app, "YoloRtmwWholeBodyBackend", FakeYoloRtmwBackend)

    _, resolved = web_app.PoseStreamEngine()._create_backend(
        "rtmw-wholebody",
        "lunge",
        "",
        target_select="tracking",
    )

    assert resolved == "yolo-rtmw-wholebody"
    assert FakeYoloRtmwBackend.calls[0][1]["target_select"] == "tracking"


def test_realtime_can_select_rtmw_wholebody(monkeypatch: Any) -> None:
    FakeYoloRtmwBackend.calls.clear()
    monkeypatch.setattr(
        web_realtime,
        "YoloRtmwWholeBodyBackend",
        FakeYoloRtmwBackend,
    )

    _, resolved = web_realtime.default_backend_factory("rtmw-wholebody", "wall_ball")

    assert resolved == "yolo-rtmw-wholebody"
    assert FakeYoloRtmwBackend.calls[0][1]["target_select"] == "tracking"


def test_rtmw_initialization_failure_has_a_visible_safe_fallback(
    monkeypatch: Any,
) -> None:
    class BrokenRtmw:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("missing provider")

    FakeYoloGuidedMediaPipeBackend.calls.clear()
    monkeypatch.setattr(web_realtime, "YoloRtmwWholeBodyBackend", BrokenRtmw)
    monkeypatch.setattr(
        web_realtime,
        "YoloGuidedMediaPipeBackend",
        FakeYoloGuidedMediaPipeBackend,
    )

    _, resolved = web_realtime.default_backend_factory("rtmw-wholebody", "lunge")

    assert resolved == "yolo-guided-mediapipe-fallback"
    assert FakeYoloGuidedMediaPipeBackend.calls
