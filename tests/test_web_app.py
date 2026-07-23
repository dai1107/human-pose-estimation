from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

from webui.app import _backend_plan, create_app


class FakeEngine:
    def __init__(self) -> None:
        self.started_with: dict[str, Any] | None = None
        self.stopped = False
        self.settings: dict[str, Any] = {}

    def snapshot(self) -> dict[str, Any]:
        return {
            "running": self.started_with is not None and not self.stopped,
            "status": "running" if self.started_with is not None and not self.stopped else "idle",
            "status_text": "分析中" if self.started_with is not None and not self.stopped else "等待开始",
        }

    def start(self, config: dict[str, Any]) -> None:
        self.started_with = config
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True

    def update_settings(self, values: dict[str, Any]) -> dict[str, Any]:
        self.settings.update(values)
        return self.snapshot()

    def request_recording(self, enabled: bool) -> dict[str, Any]:
        self.settings["recording"] = enabled
        return self.snapshot()

    def save_screenshot(self) -> Any:
        raise RuntimeError("当前还没有可保存的画面")

    def wait_for_frame(self, version: int, timeout: float = 2.0) -> tuple[int, bytes | None]:
        return version, None


def csrf_headers(client: Any) -> dict[str, str]:
    response = client.get("/api/options")
    assert response.status_code == 200
    return {"X-CSRF-Token": response.json["csrf_token"]}


def test_web_home_and_options_are_available() -> None:
    client = create_app(FakeEngine()).test_client()

    page = client.get("/")
    options = client.get("/api/options")

    assert page.status_code == 200
    assert "HYROX 动作分析台" in page.get_data(as_text=True)
    assert 'id="videoRepCount"' in page.get_data(as_text=True)
    assert 'id="poseValidRepCount"' in page.get_data(as_text=True)
    assert "完整动作周期" in page.get_data(as_text=True)
    assert 'id="voiceToggle"' in page.get_data(as_text=True)
    assert 'id="fingerToggle" type="checkbox">' in page.get_data(as_text=True)
    assert options.status_code == 200
    assert {item["value"] for item in options.json["actions"]} >= {"lunge", "wall_ball", "rowing"}
    assert len(options.json["samples"]) == 8
    assert {item["action"] for item in options.json["samples"]} == {
        "lunge", "wall_ball", "farmers_carry", "rowing", "skierg", "burpee_broad_jump", "sled_push", "sled_pull"
    }
    assert options.json["standards"]["rowing"]
    assert options.json["official_rules"]["wall_ball"]
    assert options.json["realtime"]["target_fps"] == 30
    assert options.json["realtime"]["camera_fps"] == 60
    assert options.json["realtime"]["max_requests_in_flight"] == 1
    assert options.json["realtime"]["inference_long_edge"] == 640
    assert options.json["realtime"]["jpeg_quality"] == 0.65
    assert options.json["realtime"]["max_pose_age_ms"] == 150
    assert options.json["realtime"]["hide_pose_after_ms"] == 300
    assert options.json["realtime"]["rendering"] == {
        "angle_text_fps": 12.0,
        "metrics_fps": 5.0,
        "stats_fps": 3.0,
        "timing_sample_capacity": 240,
    }
    assert options.json["realtime"]["camera"] == {
        "preferred_width": 640,
        "preferred_height": 480,
        "preferred_fps": 60.0,
        "fallback_fps": 30.0,
        "diagnostic_sample_fps": 5.0,
        "low_light_luma": 55.0,
        "fps_warning_ratio": 0.8,
        "interval_anomaly_ratio": 1.8,
        "duplicate_warning_ratio": 0.2,
    }
    assert options.json["realtime"]["local_first"] == {
        "web_pipeline": "local_browser",
        "desktop_pipeline": "local_device",
        "server_pose_fallback": True,
        "neural_prediction_enabled": False,
    }
    assert options.json["realtime"]["browser_pose"]["enabled"] is True
    assert options.json["realtime"]["browser_pose"]["worker_url"] == "/static/workers/pose_worker.js"
    assert options.json["realtime"]["browser_pose"]["model_url"] == "/assets/models/pose_landmarker_full.task"
    assert options.json["realtime"]["browser_pose"]["model_urls"] == {
        "lite": "/assets/models/pose_landmarker_lite.task",
        "full": "/assets/models/pose_landmarker_full.task",
    }
    assert options.json["realtime"]["browser_pose"]["model_preference"] == "auto"
    assert options.json["realtime"]["browser_pose"]["analysis_model"] == "full"
    assert options.json["realtime"]["browser_pose"]["benchmark_duration_ms"] == 3000
    assert options.json["realtime"]["browser_pose"]["lite_auto_approved"] is False
    assert options.json["realtime"]["browser_pose"]["max_inference_ms"] == 100
    assert options.json["realtime"]["browser_pose"]["slow_frame_limit"] == 12
    assert options.json["realtime"]["browser_pose"]["analysis_smoothing"] == {
        "profile": "responsive",
        "prediction_enabled": False,
    }
    display = options.json["realtime"]["browser_pose"]["display_smoothing"]
    assert display["profile"] == "ultra_responsive"
    assert display["min_cutoff"] == pytest.approx(2.2)
    assert display["beta"] == pytest.approx(0.12)
    assert display["max_raw_weight"] == pytest.approx(0.45)
    assert display["prediction_enabled"] is True
    prediction = options.json["realtime"]["browser_pose"]["display_prediction"]
    assert prediction["enabled"] is True
    assert prediction["mode"] == "constant_velocity"
    assert prediction["max_horizon_ms"] == pytest.approx(45)
    assert prediction["maximum_body_scale_displacement"] == pytest.approx(0.06)
    assert prediction["minimum_visibility"] == pytest.approx(0.70)
    assert prediction["velocity_decay"] == pytest.approx(0.85)
    assert prediction["disable_after_gap_ms"] == pytest.approx(100)
    model = client.get(options.json["realtime"]["browser_pose"]["model_url"])
    assert model.status_code == 200
    assert model.mimetype == "application/octet-stream"
    lite_model = client.get(options.json["realtime"]["browser_pose"]["model_urls"]["lite"])
    assert lite_model.status_code == 200
    assert lite_model.mimetype == "application/octet-stream"


def test_browser_realtime_client_uses_video_frame_callback_and_single_in_flight_request() -> None:
    source = Path("webui/static/app.js").read_text(encoding="utf-8")

    assert "requestInFlight: false" in source
    assert "if (ui.requestInFlight || ui.socket.bufferedAmount" in source
    assert "ui.requestInFlight = true;" in source
    assert "finishFrameRequest(frameId)" in source
    assert "video.requestVideoFrameCallback(onVideoFrame)" in source
    assert "requestAnimationFrame(fallbackLoop)" in source
    assert "scheduleNextCapture" not in source
    assert source.index("renderPoseForVideoFrame(frameMeta, now)") < source.index("void captureLatestFrame(frameMeta)")
    for field in (
        "sessionId", "frameId", "presentedFrames", "mediaTime", "presentationTime",
        "expectedDisplayTime", "captureTime", "width", "height",
    ):
        assert field in source
    assert "lastRenderedPoseFrameId" in source
    assert "result.session_id === ui.activeRealtimeSessionId" in source
    assert "result.run_id === ui.activeRealtimeRunId" in source
    assert "lastDiscardedFrameId" in source
    assert "ui.requestTimeout = setTimeout" in source
    assert "inference_long_edge" in source
    assert 'ui.realtimeConfig.jpeg_quality ?? 0.65' in source
    assert "new TextEncoder().encode(JSON.stringify" in source
    assert "now - ui.lastResultAt" in source
    assert "hideAfter * 0.8" in source
    assert "now - captureMs" not in source
    assert 'mode === "camera" ? "未连接" : "本机处理"' in source
    assert 'ui.sourceMode === "camera" && ui.running && !ui.manualStop' in source
    assert '"sample-cache": "预计算示例结果"' in source
    assert "Math.round(angle.value)}° 3D" in source


def test_browser_pose_worker_uses_latest_frame_slot_and_landmark_protocol() -> None:
    source = Path("webui/static/app.js").read_text(encoding="utf-8")
    worker = Path("webui/static/workers/pose_worker.js").read_text(encoding="utf-8")
    display_filter = Path("webui/static/workers/display_pose_filter.mjs").read_text(encoding="utf-8")

    assert "poseWorkerBusy: false" in source
    assert "poseWorkerPending: null" in source
    assert "closePoseTransfer(ui.poseWorkerPending)" in source
    assert 'transferMode: "video-frame"' in source
    assert 'transferMode: "image-bitmap"' in source
    assert 'type: "pose_frame"' in source
    assert "rawImageLandmarks" in source
    assert "rawWorldLandmarks" in source
    assert "canvas.toDataURL" not in source
    assert 'type: "benchmark_complete"' in worker
    assert 'type: "switch_model"' in source
    assert "selectAutoModel" in worker
    assert 'runningMode: "VIDEO"' in worker
    assert "detectForVideo(input, timestampMs)" in worker
    assert "outputSegmentationMasks: false" in worker
    assert "DisplayPoseFilter" in worker
    assert "new OneEuroFilter" in display_filter
    assert "imageRawHistory" in display_filter
    assert "worldRawHistory" in display_filter
    assert "#rawWeight" in display_filter
    assert "EXTREMITY_LANDMARKS" in display_filter
    assert "CORE_LANDMARKS" in display_filter
    assert "FACE_LANDMARKS" in display_filter
    assert "rawImageLandmarks" in source
    assert "message.imageLandmarks" in source
    assert "display_filter: message.displayFilter" in source
    assert "image_landmarks: serializeLocalLandmarks(message.rawImageLandmarks)" in source
    assert "world_landmarks: serializeLocalLandmarks(message.rawWorldLandmarks)" in source
    assert "image_landmarks: serializeLocalLandmarks(message.imageLandmarks)" not in source
    assert "drawSkeleton(result, opacity, renderStart, prediction.landmarks)" in source
    assert "ui.latestResult.keypoints = prediction" not in source
    assert "prediction_horizon_ms" not in source
    assert "prediction_point_count" not in source
    assert "prediction_clamped_point_count" not in source
    assert 'local_first?.server_pose_fallback !== false' in source
    assert 'type: "camera_diagnostics"' in source
    assert 'message_type == "camera_diagnostics"' in Path("webui/app.py").read_text(
        encoding="utf-8"
    )


def test_file_videos_analyze_every_frame_at_the_source_playback_rate() -> None:
    source = Path("webui/app.py").read_text(encoding="utf-8")

    assert 'mode="one-euro"' in source
    assert "sample_frame_step" not in source
    assert "capture.grab()" not in source
    assert 'config["source_mode"] != "camera"' in source
    assert "remaining = (1.0 / source_fps) - elapsed" in source
    assert "self._stop_event.wait(remaining)" in source


def test_camera_analysis_can_be_started_and_stopped_from_api() -> None:
    engine = FakeEngine()
    client = create_app(engine).test_client()
    headers = csrf_headers(client)

    response = client.post(
        "/api/start",
        headers=headers,
        json={
            "source_mode": "camera",
            "camera_index": 0,
            "action": "wall_ball",
            "camera_view": "front",
            "sensitivity": "medium",
            "backend": "auto",
            "landmark_profile": "full",
            "mirror": True,
        },
    )

    assert response.status_code == 200
    assert engine.started_with is not None
    assert engine.started_with["action"] == "wall_ball"
    assert engine.started_with["source_name"] == "服务器摄像头 0"
    assert client.post("/api/stop", headers=headers).status_code == 200
    assert engine.stopped is True


def test_camera_analysis_rejects_experimental_backend_in_product_api() -> None:
    engine = FakeEngine()
    client = create_app(engine).test_client()
    headers = csrf_headers(client)

    response = client.post(
        "/api/start",
        headers=headers,
        json={
            "source_mode": "camera",
            "camera_index": 0,
            "action": "lunge",
            "backend": "yolo-mediapipe",
        },
    )

    assert response.status_code == 400
    assert engine.started_with is None
    assert "无效的识别后端" in response.json["error"]


def test_start_rejects_unknown_action() -> None:
    client = create_app(FakeEngine()).test_client()
    headers = csrf_headers(client)

    response = client.post(
        "/api/start",
        headers=headers,
        json={"source_mode": "camera", "action": "unknown_action"},
    )

    assert response.status_code == 400
    assert "无效的动作" in response.json["error"]


def test_sample_action_and_video_are_linked_by_the_api() -> None:
    engine = FakeEngine()
    client = create_app(engine).test_client()
    options = client.get("/api/options")
    sample = next(item for item in options.json["samples"] if item["action"] == "rowing")
    headers = {"X-CSRF-Token": options.json["csrf_token"]}

    started = client.post(
        "/api/start",
        headers=headers,
        json={"source_mode": "sample", "video_id": sample["id"], "action": "rowing"},
    )
    mismatch = client.post(
        "/api/start",
        headers=headers,
        json={"source_mode": "sample", "video_id": sample["id"], "action": "lunge"},
    )

    assert started.status_code == 200
    assert engine.started_with is not None
    assert engine.started_with["action"] == "rowing"
    assert mismatch.status_code == 400
    assert "不一致" in mismatch.json["error"]


def test_backend_plan_limits_internal_tracking_to_trusted_lunge_sample() -> None:
    assert _backend_plan({"source_mode": "sample", "action": "lunge", "backend": "auto"}) == (
        "auto",
        "tracking",
    )
    assert _backend_plan({"source_mode": "camera", "action": "lunge", "backend": "auto"}) == (
        "auto",
        "tracking",
    )
    assert _backend_plan(
        {
            "source_mode": "sample",
            "action": "lunge",
            "backend": "rtmw-wholebody",
        }
    ) == ("rtmw-wholebody", "tracking")
    assert _backend_plan(
        {
            "source_mode": "sample",
            "action": "lunge",
            "backend": "mediapipe",
            "bundled_sample_tracking": True,
        }
    ) == ("yolo-mediapipe", "tracking")


def test_web_product_page_only_offers_mediapipe_pose() -> None:
    response = create_app(FakeEngine()).test_client().get("/")

    assert response.status_code == 200
    assert b'value="mediapipe"' in response.data
    assert b'<select id="poseModelSelect">' in response.data
    assert b'<option value="auto" selected>' in response.data
    assert b'value="yolo-mediapipe"' not in response.data
    assert b'value="yolo-pose"' not in response.data
    assert b'value="rtmw-wholebody"' not in response.data


def test_server_screenshot_is_disabled_for_privacy() -> None:
    client = create_app(FakeEngine()).test_client()
    headers = csrf_headers(client)

    response = client.post("/api/screenshot", headers=headers)

    assert response.status_code == 410
    assert response.json["code"] == "server_screenshot_disabled"


def test_server_recording_is_disabled_for_privacy() -> None:
    client = create_app(FakeEngine()).test_client()
    headers = csrf_headers(client)

    response = client.post("/api/record", headers=headers, json={"enabled": True})

    assert response.status_code == 410
    assert response.json["code"] == "recording_disabled"


def test_upload_checks_actual_media_content() -> None:
    client = create_app(FakeEngine()).test_client()
    headers = csrf_headers(client)

    response = client.post(
        "/api/upload",
        headers=headers,
        data={"video": (io.BytesIO(b"not a real video"), "pretend.mp4")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.json["code"] == "invalid_media"
    assert client.delete("/api/session", headers=headers).status_code == 200


def test_shared_access_link_sets_cookie_and_protects_api() -> None:
    client = create_app(FakeEngine(), access_token="unit-secret").test_client()

    assert client.get("/").status_code == 401
    assert client.get("/api/options").status_code == 401

    accepted = client.get("/?access_token=unit-secret")
    assert accepted.status_code == 302
    assert client.get("/").status_code == 200
    assert client.get("/api/options").status_code == 200


def test_health_check_does_not_require_shared_access_token() -> None:
    client = create_app(FakeEngine(), access_token="unit-secret").test_client()

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json == {"status": "ok"}


def test_mutating_api_rejects_missing_csrf_token() -> None:
    client = create_app(FakeEngine()).test_client()
    assert client.get("/api/options").status_code == 200

    response = client.post("/api/stop")

    assert response.status_code == 403
    assert response.json["code"] == "csrf_failed"


def test_security_headers_and_cookie_attributes() -> None:
    client = create_app(FakeEngine()).test_client()

    response = client.get("/", base_url="https://pose.example.test")

    assert response.headers["Permissions-Policy"] == "camera=(self), microphone=(), geolocation=()"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert "worker-src 'self' blob:" in response.headers["Content-Security-Policy"]
    assert "'wasm-unsafe-eval'" in response.headers["Content-Security-Policy"]
    cookie = response.headers.get("Set-Cookie", "")
    assert "HttpOnly" in cookie
    assert "Secure" in cookie
    assert "SameSite=Lax" in cookie


def test_anonymous_browser_sessions_are_isolated() -> None:
    engines: list[FakeEngine] = []

    def make_engine(_: str) -> FakeEngine:
        value = FakeEngine()
        engines.append(value)
        return value

    app = create_app(engine_factory=make_engine)
    client_a = app.test_client()
    client_b = app.test_client()
    headers_a = csrf_headers(client_a)
    headers_b = csrf_headers(client_b)
    payload = {
        "source_mode": "camera",
        "camera_index": 0,
        "action": "lunge",
        "camera_view": "side",
        "sensitivity": "medium",
        "backend": "auto",
        "landmark_profile": "full",
    }

    assert client_a.post("/api/start", headers=headers_a, json=payload).status_code == 200
    assert client_b.post("/api/start", headers=headers_b, json={**payload, "action": "rowing"}).status_code == 200
    assert len(engines) == 2
    assert engines[0].started_with["action"] == "lunge"
    assert engines[1].started_with["action"] == "rowing"

    assert client_a.post("/api/stop", headers=headers_a).status_code == 200
    assert engines[0].stopped is True
    assert engines[1].stopped is False
