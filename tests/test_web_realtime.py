from __future__ import annotations

import struct
import json
import threading
import time

import cv2
import numpy as np
import pytest
from simple_websocket import Client
from werkzeug.serving import make_server

from src.backends.base import Keypoint, PoseResult
from webui.app import create_app
from webui.realtime import (
    FramePacket,
    FrameRequest,
    LatestFrameQueue,
    RealtimePoseSession,
    RealtimeProtocolError,
    unpack_frame,
    unpack_frame_request,
    validate_manual_floor_points,
    validate_settings,
)
from webui.realtime import _profile_names


class FakePoseBackend:
    def __init__(self) -> None:
        self.closed = False

    def detect(self, frame: np.ndarray, timestamp_ms: int | None = None) -> PoseResult:
        points = [
            Keypoint("left_shoulder", 0.4, 0.3, confidence=0.95),
            Keypoint("right_shoulder", 0.6, 0.3, confidence=0.95),
            Keypoint("left_hip", 0.45, 0.58, confidence=0.9),
            Keypoint("right_hip", 0.55, 0.58, confidence=0.9),
        ]
        return PoseResult(
            keypoints=points,
            connections=((0, 1), (0, 2), (1, 3), (2, 3)),
            model_name="fake",
            num_keypoints=len(points),
            success=True,
            inference_time_ms=7.5,
            timestamp_ms=timestamp_ms,
        )

    def close(self) -> None:
        self.closed = True


def make_packet(sequence: int, width: int = 320, height: int = 240) -> bytes:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 1] = 80
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return struct.pack(">I", sequence) + encoded.tobytes()


def make_v2_packet(
    frame_id: int,
    *,
    session_id: str,
    run_id: str,
    action: str = "none",
    backend: str = "mediapipe",
    client_capture_ms: float = 1234.5,
    width: int = 320,
    height: int = 240,
    timing: dict[str, float] | None = None,
    frame_meta: dict[str, object] | None = None,
) -> bytes:
    jpeg = make_packet(frame_id, width=width, height=height)[4:]
    metadata = json.dumps(
        {
            "session_id": session_id,
            "run_id": run_id,
            "frame_id": frame_id,
            "client_capture_ms": client_capture_ms,
            "action": action,
            "backend": backend,
            "timing": timing or {},
            "frame_meta": frame_meta or {},
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return b"PSV2" + struct.pack(">I", len(metadata)) + metadata + jpeg


def wait_for_result(session: RealtimePoseSession, timeout: float = 2.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = session.next_result(timeout=0.05)
        if result is not None:
            return result
    raise AssertionError("realtime result was not published")


def make_browser_pose_payload(
    *, session_id: str, run_id: str, frame_id: int = 1
) -> dict[str, object]:
    landmarks = [
        {"x": 0.25 + index * 0.01, "y": 0.3 + index * 0.005, "z": 0.01, "visibility": 0.95, "presence": 0.9}
        for index in range(33)
    ]
    world = [
        {"x": index * 0.01, "y": index * 0.005, "z": index * 0.002, "visibility": 0.95, "presence": 0.9}
        for index in range(33)
    ]
    return {
        "type": "pose_frame",
        "session_id": session_id,
        "run_id": run_id,
        "frame_id": frame_id,
        "action": "none",
        "backend": "mediapipe",
        "capture_timestamp_ms": 1000.0 + frame_id,
        "presentation_timestamp_ms": 1001.0 + frame_id,
        "image_landmarks": landmarks,
        "world_landmarks": world,
        "pose_inference_ms": 8.25,
        "pose_model": "full",
        "pose_model_benchmark": {
            "selectedModel": "full",
            "reason": "Full P95 不超过 20 ms",
            "mainThreadLongTaskCount": 0,
            "stats": {
                "full": {"samples": 45, "inferenceP50Ms": 14.0, "inferenceP95Ms": 18.0, "poseFps": 30.0, "detectionRate": 1.0},
                "lite": {"samples": 45, "inferenceP50Ms": 9.0, "inferenceP95Ms": 12.0, "poseFps": 30.0, "detectionRate": 1.0},
            },
        },
        "display_filter": {
            "profile": "ultra_responsive",
            "predictionEnabled": True,
            "rawBlendEnabled": True,
            "blendedPointCount": 12,
            "meanRawWeight": 0.18,
            "maxRawWeight": 0.42,
        },
        "source": "browser_mediapipe",
        "frame_meta": {
            "sessionId": session_id,
            "frameId": frame_id,
            "presentedFrames": frame_id,
            "mediaTime": frame_id / 30,
            "presentationTime": 1001.0 + frame_id,
            "expectedDisplayTime": 1008.0 + frame_id,
            "captureTime": 1000.0 + frame_id,
            "processingDuration": 0.001,
            "width": 640,
            "height": 480,
            "callbackSource": "requestVideoFrameCallback",
        },
        "timing": {"camera_frame_presented_ms": 1001.0 + frame_id},
    }


def make_camera_diagnostics_payload() -> dict[str, object]:
    return {
        "type": "camera_diagnostics",
        "diagnostics": {
            "settings": {
                "width": 640,
                "height": 480,
                "frameRate": 59.94,
                "deviceId": "camera-device-token",
                "resizeMode": "none",
                "facingMode": "user",
            },
            "requestedFps": 60,
            "actualPresentedFps": 58.7,
            "frameIntervalP50Ms": 16.9,
            "frameIntervalP95Ms": 22.4,
            "frameIntervalAnomalyRatio": 0.03,
            "brightnessMean": 102.5,
            "duplicateFrameRatio": 0.02,
            "sampleCount": 40,
            "warnings": [],
        },
    }


def test_latest_frame_queue_discards_stale_frame() -> None:
    frames = LatestFrameQueue()
    frames.put_latest(FramePacket(1, b"one", 1.0))
    frames.put_latest(FramePacket(2, b"two", 2.0))

    result = frames.get(timeout=0.01)

    assert result.sequence == 2
    assert frames.dropped == 1


def test_finger_nodes_can_be_hidden_independently() -> None:
    names = {"left_wrist", "left_index", "left_thumb", "right_pinky", "left_knee"}

    assert validate_settings({"show_fingers": False})["show_fingers"] is False
    assert _profile_names("full", names, show_fingers=False) == {"left_wrist", "left_knee"}


def test_validate_settings_rejects_rtmw_in_product_mode() -> None:
    with pytest.raises(RealtimeProtocolError, match="无效的识别模型"):
        validate_settings({"backend": "rtmw-wholebody"})


def test_validate_settings_accepts_rtmw_in_explicit_experimental_mode() -> None:
    assert (
        validate_settings(
            {"backend": "rtmw-wholebody"},
            allow_experimental_backends=True,
        )["backend"]
        == "rtmw-wholebody"
    )


def test_validate_settings_accepts_explicit_yolo_mediapipe() -> None:
    assert (
        validate_settings(
            {"backend": "yolo-mediapipe"},
            allow_experimental_backends=True,
        )["backend"]
        == "yolo-mediapipe"
    )


def test_validate_manual_floor_points_accepts_two_normalized_points() -> None:
    points = validate_manual_floor_points([[0.1, 0.82], [0.9, 0.91]])

    assert points == [[0.1, 0.82], [0.9, 0.91]]
    assert validate_settings({"manual_floor_points": points})["manual_floor_points"] == points


@pytest.mark.parametrize(
    "points",
    (
        [[0.1, 0.8]],
        [[-0.1, 0.8], [0.9, 0.9]],
        [[0.50, 0.2], [0.52, 0.9]],
        [[0.1, 0.1], [0.2, 0.9]],
    ),
)
def test_validate_manual_floor_points_rejects_invalid_lines(points: list[list[float]]) -> None:
    with pytest.raises(RealtimeProtocolError):
        validate_manual_floor_points(points)


def test_frame_protocol_validates_header_size_and_media_signature() -> None:
    with pytest.raises(RealtimeProtocolError, match="不完整"):
        unpack_frame(b"\x00\x00")
    with pytest.raises(RealtimeProtocolError, match="JPEG"):
        unpack_frame(struct.pack(">I", 1) + b"not-an-image")

    sequence, jpeg = unpack_frame(make_packet(42))

    assert sequence == 42
    assert jpeg.startswith(b"\xff\xd8\xff")


def test_frame_protocol_v2_restores_request_identity_and_capture_time() -> None:
    request = unpack_frame_request(
        make_v2_packet(
            42,
            session_id="session-42",
            run_id="run-7",
            action="lunge",
            client_capture_ms=9876.25,
            frame_meta={
                "sessionId": "session-42",
                "frameId": 42,
                "presentedFrames": 50,
                "mediaTime": 1.25,
                "presentationTime": 9870.0,
                "expectedDisplayTime": 9876.0,
                "captureTime": 9860.0,
                "processingDuration": 0.002,
                "width": 640,
                "height": 480,
                "callbackSource": "requestVideoFrameCallback",
            },
        )
    )

    assert isinstance(request, FrameRequest)
    assert request.frame_id == 42
    assert request.session_id == "session-42"
    assert request.run_id == "run-7"
    assert request.action == "lunge"
    assert request.backend == "mediapipe"
    assert request.client_capture_ms == pytest.approx(9876.25)
    assert request.frame_meta["frameId"] == 42
    assert request.frame_meta["presentedFrames"] == 50
    assert request.jpeg.startswith(b"\xff\xd8\xff")


def test_frame_protocol_v2_requires_complete_context_identity() -> None:
    jpeg = make_packet(1)[4:]
    metadata = json.dumps(
        {
            "frame_id": 1,
            "client_capture_ms": 10.0,
            "run_id": "run",
            "action": "none",
            "backend": "mediapipe",
        }
    ).encode("utf-8")
    packet = b"PSV2" + struct.pack(">I", len(metadata)) + metadata + jpeg

    with pytest.raises(RealtimeProtocolError, match="session_id"):
        unpack_frame_request(packet)


def test_frame_protocol_rejects_mismatched_video_frame_identity() -> None:
    packet = make_v2_packet(
        3,
        session_id="identity",
        run_id="run",
        frame_meta={
            "sessionId": "identity",
            "frameId": 2,
            "callbackSource": "requestVideoFrameCallback",
        },
    )
    with pytest.raises(RealtimeProtocolError, match="frameId"):
        unpack_frame_request(packet)


def test_realtime_session_returns_pose_json_and_downloadable_reports() -> None:
    backend = FakePoseBackend()
    session = RealtimePoseSession(
        "unit-session",
        backend_factory=lambda _requested, _action: (backend, "fake"),
        max_receive_fps=1000,
    )
    session.mark_connected()
    state = session.start({"action": "none", "backend": "auto", "landmark_profile": "full"})

    assert session.submit(make_packet(7)) is True
    result = wait_for_result(session)

    assert result["type"] == "result"
    assert result["session_id"] == "unit-session"
    assert result["run_id"] == state["run_id"]
    assert result["frame_id"] == 7
    assert result["sequence"] == 7
    assert result["pose_detected"] is True
    assert result["voice_feedback"] is None
    assert result["metrics"]["width"] == 320
    assert result["metrics"]["inference_ms"] == 7.5
    assert result["server_inference_ms"] == 7.5
    assert result["pose_age_ms"] >= 0
    assert {point["name"] for point in result["keypoints"]} == {
        "left_shoulder", "right_shoulder", "left_hip", "right_hip"
    }
    assert session.report()["summary"]["processed_frames"] == 1
    assert "keypoints_json" in session.report_csv()

    session.stop()
    assert backend.closed is True


def test_browser_pose_frame_runs_rules_without_server_pose_inference() -> None:
    backend_factory_calls: list[tuple[str, str]] = []

    def forbidden_backend_factory(backend: str, action: str):
        backend_factory_calls.append((backend, action))
        raise AssertionError("server pose backend must not run for browser landmarks")

    session = RealtimePoseSession(
        "browser-local-session",
        backend_factory=forbidden_backend_factory,
        max_receive_fps=1000,
    )
    session.mark_connected()
    state = session.start({"action": "none", "backend": "mediapipe"})

    assert session.submit_pose_frame(make_browser_pose_payload(
        session_id="browser-local-session",
        run_id=state["run_id"],
    )) is True
    result = wait_for_result(session)

    assert backend_factory_calls == []
    assert result["source"] == "browser_mediapipe"
    assert result["metrics"]["backend"] == "browser-mediapipe-full"
    assert result["metrics"]["inference_ms"] == pytest.approx(8.2, abs=0.1)
    assert result["frame_meta"]["frameId"] == 1
    assert result["pose_detected"] is True
    assert len(result["keypoints"]) == 27  # six finger proxy points hidden by default
    assert result["three_d_kinematics"]["world_landmark_count"] == 33
    assert result["pose_model_benchmark"]["selected_model"] == "full"
    assert result["display_filter"]["profile"] == "ultra_responsive"
    assert result["display_filter"]["prediction_enabled"] is True
    session.stop()


def test_browser_pose_frame_rejects_stale_identity() -> None:
    session = RealtimePoseSession(
        "browser-identity-session",
        backend_factory=lambda *_: (FakePoseBackend(), "fake"),
        max_receive_fps=1000,
    )
    session.mark_connected()
    state = session.start({"action": "none", "backend": "mediapipe"})
    payload = make_browser_pose_payload(session_id="wrong-session", run_id=state["run_id"])
    with pytest.raises(RealtimeProtocolError, match="会话"):
        session.submit_pose_frame(payload)
    session.stop()


def test_browser_pose_frame_rejects_unknown_model_tier() -> None:
    session = RealtimePoseSession(
        "browser-model-tier",
        backend_factory=lambda *_: (FakePoseBackend(), "fake"),
        max_receive_fps=1000,
    )
    session.mark_connected()
    started = session.start({"action": "none", "backend": "mediapipe"})
    payload = make_browser_pose_payload(
        session_id=session.session_id,
        run_id=str(started["run_id"]),
    )
    payload["pose_model"] = "heavy"

    with pytest.raises(RealtimeProtocolError, match="模型档位"):
        session.submit_pose_frame(payload)

    session.stop()


def test_browser_pose_frame_rejects_display_prediction_protocol_fields() -> None:
    session = RealtimePoseSession(
        "prediction-isolation",
        backend_factory=lambda *_: (FakePoseBackend(), "fake"),
        max_receive_fps=1000,
    )
    started = session.start({"action": "none", "backend": "mediapipe"})
    payload = make_browser_pose_payload(
        session_id=session.session_id,
        run_id=str(started["run_id"]),
    )
    payload["predicted_landmarks"] = payload["image_landmarks"]

    with pytest.raises(RealtimeProtocolError, match="显示预测"):
        session.submit_pose_frame(payload)

    assert session.report()["summary"]["processed_frames"] == 0
    session.stop()


def test_web_latency_timeline_round_trips_into_downloadable_report() -> None:
    session = RealtimePoseSession(
        "latency-session",
        backend_factory=lambda _requested, _action: (FakePoseBackend(), "fake"),
        max_receive_fps=1000,
    )
    session.mark_connected()
    state = session.start({"action": "none", "backend": "mediapipe"})
    assert session.submit(make_v2_packet(
        1,
        session_id="latency-session",
        run_id=state["run_id"],
        timing={
            "camera_frame_presented_ms": 100.0,
            "frame_copy_start_ms": 101.0,
            "frame_copy_end_ms": 102.0,
            "encode_start_ms": 102.0,
            "encode_end_ms": 104.0,
            "socket_send_ms": 105.0,
            "expected_display_time_ms": 133.0,
        },
        frame_meta={
            "sessionId": "latency-session",
            "frameId": 1,
            "presentedFrames": 8,
            "mediaTime": 0.25,
            "presentationTime": 100.0,
            "expectedDisplayTime": 108.0,
            "captureTime": 98.0,
            "processingDuration": 0.001,
            "width": 640,
            "height": 480,
            "callbackSource": "requestVideoFrameCallback",
        },
    ))
    result = wait_for_result(session)
    assert result["frame_meta"]["frameId"] == 1
    timing = dict(result["latency_timing"])
    assert "server_receive_ms" in timing
    assert "inference_start_ms" in timing
    assert "inference_end_ms" in timing
    timing.update({
        "socket_result_send_ms": timing["inference_end_ms"] + 1.0,
        "client_result_receive_ms": 125.0,
        "pose_render_start_ms": 127.0,
        "pose_render_end_ms": 129.0,
        "expected_display_time_ms": 133.0,
        "video_frame_presented_at_render_ms": 120.0,
    })
    assert session.record_latency_audit({"frame_id": 1, "timing": timing}) is True
    report = session.report()
    assert report["summary"]["latency_audit"]["sample_count"] == 1
    assert report["frames"][0]["latency"]["pose_video_age_difference_ms"] == 20.0
    assert "pose_video_age_difference_ms" in session.report_csv()
    with pytest.raises(RealtimeProtocolError, match="未知字段"):
        session.record_latency_audit(
            {
                "frame_id": 1,
                "timing": {
                    **timing,
                    "predicted_landmarks": 33,
                },
            }
        )
    session.stop()


def test_camera_diagnostics_are_bounded_and_included_in_report_summary() -> None:
    session = RealtimePoseSession("camera-diagnostics-session")
    session.start({"action": "none"})

    assert session.record_camera_diagnostics(make_camera_diagnostics_payload()) is True
    diagnostics = session.report()["summary"]["camera_diagnostics"]
    assert diagnostics["settings"] == {
        "width": 640,
        "height": 480,
        "frameRate": 59.94,
        "deviceId": "camera-device-token",
        "resizeMode": "none",
        "facingMode": "user",
    }
    assert diagnostics["actualPresentedFps"] == pytest.approx(58.7)
    assert diagnostics["sampleCount"] == 40
    assert diagnostics["warnings"] == []

    invalid_payloads = []
    invalid_ratio = make_camera_diagnostics_payload()
    invalid_ratio["diagnostics"]["duplicateFrameRatio"] = 1.01  # type: ignore[index]
    invalid_payloads.append(invalid_ratio)
    unknown_warning = make_camera_diagnostics_payload()
    unknown_warning["diagnostics"]["warnings"] = ["run_neural_predictor"]  # type: ignore[index]
    invalid_payloads.append(unknown_warning)
    long_device_id = make_camera_diagnostics_payload()
    long_device_id["diagnostics"]["settings"]["deviceId"] = "x" * 257  # type: ignore[index]
    invalid_payloads.append(long_device_id)
    nested_unknown = make_camera_diagnostics_payload()
    nested_unknown["diagnostics"]["settings"]["capabilities"] = {}  # type: ignore[index]
    invalid_payloads.append(nested_unknown)

    for payload in invalid_payloads:
        with pytest.raises(RealtimeProtocolError, match="摄像头"):
            session.record_camera_diagnostics(payload)

    session.clear_results()
    assert session.report()["summary"]["camera_diagnostics"] == {}
    session.stop()


def test_realtime_session_rejects_stale_v2_session_run_action_backend_and_frame() -> None:
    session = RealtimePoseSession(
        "identity-session",
        backend_factory=lambda _requested, _action: (FakePoseBackend(), "fake"),
        max_receive_fps=1000,
    )
    state = session.start({"action": "none", "backend": "mediapipe"})
    run_id = str(state["run_id"])

    with pytest.raises(RealtimeProtocolError, match="其他会话"):
        session.submit(make_v2_packet(1, session_id="other", run_id=run_id))
    with pytest.raises(RealtimeProtocolError, match="已停止"):
        session.submit(make_v2_packet(1, session_id="identity-session", run_id="old"))
    with pytest.raises(RealtimeProtocolError, match="旧动作"):
        session.submit(
            make_v2_packet(1, session_id="identity-session", run_id=run_id, action="lunge")
        )
    with pytest.raises(RealtimeProtocolError, match="旧后端"):
        session.submit(
            make_v2_packet(
                1,
                session_id="identity-session",
                run_id=run_id,
                backend="auto",
            )
        )

    assert session.submit(
        make_v2_packet(1, session_id="identity-session", run_id=run_id)
    )
    with pytest.raises(RealtimeProtocolError, match="严格递增"):
        session.submit(make_v2_packet(1, session_id="identity-session", run_id=run_id))
    session.stop()


def test_settings_change_rejects_inference_result_from_old_action_context() -> None:
    started = threading.Event()
    release = threading.Event()

    class SlowBackend(FakePoseBackend):
        def detect(self, frame: np.ndarray, timestamp_ms: int | None = None) -> PoseResult:
            started.set()
            assert release.wait(2.0)
            return super().detect(frame, timestamp_ms)

    session = RealtimePoseSession(
        "settings-session",
        backend_factory=lambda _requested, _action: (SlowBackend(), "fake"),
        max_receive_fps=1000,
    )
    session.start({"action": "none"})
    assert session.submit(make_packet(1))
    assert started.wait(1.0)

    session.update_settings({"action": "lunge"})
    release.set()
    result = wait_for_result(session)

    assert result["type"] == "frame_dropped"
    assert result["reason"] == "stale_context"
    assert result["frame_id"] == 1
    assert session.snapshot()["action"] == "lunge"
    assert session.report()["summary"]["processed_frames"] == 0
    session.stop()


def test_v2_client_capture_intervals_drive_server_pose_timestamps() -> None:
    timestamps: list[int] = []

    class TimestampBackend(FakePoseBackend):
        def detect(self, frame: np.ndarray, timestamp_ms: int | None = None) -> PoseResult:
            assert timestamp_ms is not None
            timestamps.append(timestamp_ms)
            return super().detect(frame, timestamp_ms)

    session = RealtimePoseSession(
        "clock-session",
        backend_factory=lambda _requested, _action: (TimestampBackend(), "fake"),
        max_receive_fps=1000,
    )
    session.mark_connected()
    state = session.start({"action": "none", "backend": "mediapipe"})
    run_id = str(state["run_id"])

    assert session.submit(
        make_v2_packet(
            1,
            session_id="clock-session",
            run_id=run_id,
            client_capture_ms=1000.0,
        )
    )
    wait_for_result(session)
    time.sleep(0.02)
    assert session.submit(
        make_v2_packet(
            2,
            session_id="clock-session",
            run_id=run_id,
            client_capture_ms=1033.0,
        )
    )
    wait_for_result(session)

    assert timestamps[1] - timestamps[0] == 33
    session.stop()


def test_disconnect_invalidates_an_inflight_result_from_old_connection() -> None:
    started = threading.Event()
    release = threading.Event()

    class SlowBackend(FakePoseBackend):
        def detect(self, frame: np.ndarray, timestamp_ms: int | None = None) -> PoseResult:
            started.set()
            assert release.wait(2.0)
            return super().detect(frame, timestamp_ms)

    session = RealtimePoseSession(
        "disconnect-session",
        backend_factory=lambda _requested, _action: (SlowBackend(), "fake"),
        max_receive_fps=1000,
    )
    session.mark_connected()
    session.start({"action": "none"})
    assert session.submit(make_packet(1))
    assert started.wait(1.0)

    session.mark_disconnected()
    session.mark_connected()
    release.set()
    result = wait_for_result(session)

    assert result["type"] == "frame_dropped"
    assert result["reason"] == "stale_context"
    assert session.report()["summary"]["processed_frames"] == 0
    session.stop()


def test_realtime_session_rejects_oversized_dimensions_without_crashing_worker() -> None:
    session = RealtimePoseSession(
        "dimension-session",
        backend_factory=lambda _requested, _action: (FakePoseBackend(), "fake"),
        max_receive_fps=1000,
    )
    session.mark_connected()
    session.start({"action": "none"})

    assert session.submit(make_packet(1, width=1281, height=10)) is True
    result = wait_for_result(session)

    assert result["type"] == "error"
    assert result["code"] == "invalid_dimensions"
    session.stop()


def test_realtime_session_limits_receive_rate() -> None:
    session = RealtimePoseSession(
        "rate-session",
        backend_factory=lambda _requested, _action: (FakePoseBackend(), "fake"),
        max_receive_fps=1,
    )
    session.start({"action": "none"})

    assert session.submit(make_packet(1)) is True
    assert session.submit(make_packet(2)) is False
    session.stop()


class IdleEngine:
    def snapshot(self) -> dict[str, object]:
        return {"running": False, "status": "idle", "status_text": "等待开始"}

    def stop(self) -> None:
        return None


def test_websocket_handshake_start_frame_result_and_stop() -> None:
    def realtime_factory(session_id: str, gate: threading.BoundedSemaphore) -> RealtimePoseSession:
        return RealtimePoseSession(
            session_id,
            backend_factory=lambda _requested, _action: (FakePoseBackend(), "fake"),
            inference_gate=gate,
            max_receive_fps=1000,
        )

    app = create_app(engine_factory=lambda _session_id: IdleEngine(), realtime_factory=realtime_factory)
    http_client = app.test_client()
    options = http_client.get("/api/options")
    csrf = options.json["csrf_token"]
    session_cookie = http_client.get_cookie("pose_session")
    assert session_cookie is not None
    server = make_server("127.0.0.1", 0, app, threaded=True)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    socket = None
    try:
        rejected = Client.connect(
            f"ws://127.0.0.1:{server.server_port}/ws/pose?csrf=invalid",
            headers={"Cookie": f"pose_session={session_cookie.value}"},
        )
        rejection = json.loads(rejected.receive(timeout=2))
        assert rejection["type"] == "error"
        assert rejection["code"] == "csrf_failed"
        try:
            rejected.close()
        except Exception:
            pass

        socket = Client.connect(
            f"ws://127.0.0.1:{server.server_port}/ws/pose?csrf={csrf}",
            headers={"Cookie": f"pose_session={session_cookie.value}"},
        )
        connected = json.loads(socket.receive(timeout=2))
        assert connected["type"] == "connected"
        assert connected["protocol_version"] == 2
        assert connected["session_id"] == session_cookie.value

        socket.send(json.dumps({"type": "start", "settings": {"action": "none"}}))
        started_message = json.loads(socket.receive(timeout=2))
        assert started_message["type"] == "started"
        socket.send(
            make_v2_packet(
                9,
                session_id=connected["session_id"],
                run_id=started_message["state"]["run_id"],
            )
        )
        result = json.loads(socket.receive(timeout=2))
        assert result["type"] == "result"
        assert result["frame_id"] == 9
        assert result["sequence"] == 9
        assert result["metrics"]["backend"] == "fake"

        socket.send(json.dumps({"type": "stop"}))
        assert json.loads(socket.receive(timeout=2))["type"] == "stopped"
    finally:
        if socket is not None:
            try:
                socket.close()
            except Exception:
                pass
        server.shutdown()
        server_thread.join(timeout=2)


def test_report_download_and_session_deletion_are_scoped_to_cookie() -> None:
    def realtime_factory(session_id: str, gate: threading.BoundedSemaphore) -> RealtimePoseSession:
        return RealtimePoseSession(
            session_id,
            backend_factory=lambda _requested, _action: (FakePoseBackend(), "fake"),
            inference_gate=gate,
            max_receive_fps=1000,
        )

    app = create_app(engine_factory=lambda _session_id: IdleEngine(), realtime_factory=realtime_factory)
    client = app.test_client()
    options = client.get("/api/options")
    cookie = client.get_cookie("pose_session")
    assert cookie is not None
    manager = app.extensions["pose_sessions"]
    browser_session, created = manager.get_or_create(cookie.value, "127.0.0.1")
    assert created is False
    browser_session.realtime.start({"action": "none"})
    browser_session.realtime.submit(make_packet(3))
    wait_for_result(browser_session.realtime)
    browser_session.realtime.stop()

    json_report = client.get("/api/report.json")
    csv_report = client.get("/api/report.csv")
    text_report = client.get("/api/report.txt")
    assert json_report.status_code == 200
    assert json_report.json["summary"]["processed_frames"] == 1
    assert "attachment;" in json_report.headers["Content-Disposition"]
    assert csv_report.status_code == 200
    assert csv_report.get_data(as_text=True).startswith("\ufeffsequence,")
    assert text_report.status_code == 200
    assert "HYROX 动作分析文字报告" in text_report.get_data(as_text=True)
    assert "attachment;" in text_report.headers["Content-Disposition"]

    deleted = client.delete("/api/session", headers={"X-CSRF-Token": options.json["csrf_token"]})
    assert deleted.status_code == 200
    assert deleted.json == {"status": "deleted"}
