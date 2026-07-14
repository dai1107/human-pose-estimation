from __future__ import annotations

import io
from typing import Any

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


def test_crowded_lunge_sample_selects_the_largest_foreground_person() -> None:
    assert _backend_plan({"source_mode": "sample", "action": "lunge", "backend": "auto"}) == (
        "yolo-pose",
        "area",
    )
    assert _backend_plan({"source_mode": "camera", "action": "lunge", "backend": "auto"}) == (
        "auto",
        "confidence",
    )


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
