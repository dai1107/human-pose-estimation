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
        )
    )

    assert isinstance(request, FrameRequest)
    assert request.frame_id == 42
    assert request.session_id == "session-42"
    assert request.run_id == "run-7"
    assert request.action == "lunge"
    assert request.backend == "mediapipe"
    assert request.client_capture_ms == pytest.approx(9876.25)
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
