from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

pytest.skip("WHAM/OpenCap integration is deferred", allow_module_level=True)

from src.offline3d import Offline3DManager
from src.offline3d.wham.adapter import build_command, command_template_from_environment
from src.offline3d.wham.backend import WhamBackend
from src.offline3d.wham.parser import parse_wham_payload
from webui.app import create_app


class _FakeEngine:
    def __init__(self) -> None:
        self.started_with: dict[str, object] | None = None

    def snapshot(self) -> dict[str, object]:
        return {"running": False, "status": "idle", "status_text": "等待开始"}

    def start(self, config: dict[str, object]) -> None:
        self.started_with = config

    def stop(self) -> None:
        pass

    def update_settings(self, values: dict[str, object]) -> dict[str, object]:
        return self.snapshot()


def _headers(client: object) -> dict[str, str]:
    options = client.get("/api/options")
    return {"X-CSRF-Token": options.json["csrf_token"]}


def test_upload_mode_ui_defaults_to_fast_and_explains_advanced() -> None:
    page = create_app(_FakeEngine()).test_client().get("/").get_data(as_text=True)

    assert 'name="analysisMode" value="fast" checked' in page
    assert 'name="analysisMode" value="advanced"' in page
    assert "MediaPipe/HYROX + WHAM 3D 辅助" in page
    assert 'uploadAnalysisMode: "fast"' in Path("webui/static/app.js").read_text(encoding="utf-8")
    assert "loadUploadTimeline(playbackReport)" in Path("webui/static/app.js").read_text(encoding="utf-8")


def test_advanced_analysis_hides_all_playback_until_completion() -> None:
    frontend = Path("webui/static/app.js").read_text(encoding="utf-8")
    backend = Path("webui/app.py").read_text(encoding="utf-8")

    assert 'stopUploadPlayback({ hideVideo: true })' in frontend
    assert 'startUploadPlayback({\n        frames: []' not in frontend
    assert '$("#loadingOverlay").hidden = false;' in frontend
    assert '"playback_ready": False' in backend


def test_start_validates_analysis_mode_and_only_allows_advanced_upload() -> None:
    engine = _FakeEngine()
    client = create_app(engine).test_client()
    headers = _headers(client)

    invalid = client.post(
        "/api/start",
        headers=headers,
        json={"source_mode": "camera", "analysis_mode": "advanced"},
    )
    unknown = client.post(
        "/api/start",
        headers=headers,
        json={"source_mode": "camera", "analysis_mode": "turbo"},
    )

    assert invalid.status_code == 400
    assert "仅适用于上传视频" in invalid.json["error"]
    assert unknown.status_code == 400
    assert "无效的分析模式" in unknown.json["error"]


def test_fast_mode_remains_default_for_existing_api_clients() -> None:
    engine = _FakeEngine()
    client = create_app(engine).test_client()
    response = client.post(
        "/api/start",
        headers=_headers(client),
        json={"source_mode": "camera", "camera_index": 0, "action": "lunge"},
    )

    assert response.status_code == 200
    assert engine.started_with is not None
    assert engine.started_with["analysis_mode"] == "fast"


def test_wham_parser_preserves_native_smpl_camera_and_trajectory_fields() -> None:
    result = parse_wham_payload(
        {
            "schema_version": "wham-export-1",
            "source_fps": 25,
            "joint_names": ["pelvis", "left_knee"],
            "coordinate_system": "wham_global",
            "trajectory_confidence": 0.82,
            "frames": [
                {
                    "frame": 2,
                    "joints_3d": [[0, 1, 2], [3, 4, 5]],
                    "smpl_pose": [[0.1, 0.2], [0.3, 0.4]],
                    "body_orientation": [0.3, 0.4, 0.5],
                    "body_translation": [1, 2, 3],
                    "camera_motion": [0.01, 0.02, 0.03],
                    "global_trajectory": [4, 5, 6],
                    "confidence": 0.9,
                }
            ],
        }
    )
    payload = result.as_dict()

    assert result.status == "COMPLETED"
    assert result.frames[0].timestamp_ms == pytest.approx(80.0)
    assert result.frames[0].joints_3d["left_knee"] == (3.0, 4.0, 5.0)
    assert payload["frames"][0]["smpl_pose"] == [[0.1, 0.2], [0.3, 0.4]]
    assert payload["frames"][0]["camera_motion"] == [0.01, 0.02, 0.03]
    assert payload["frames"][0]["global_trajectory"] == [4.0, 5.0, 6.0]
    assert payload["is_ground_truth"] is False


def test_unconfigured_wham_is_explicitly_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POSE_WHAM_COMMAND", raising=False)
    monkeypatch.delenv("POSE_WHAM_COMMAND_JSON", raising=False)
    monkeypatch.setenv("POSE_WHAM_AUTO_WSL", "0")

    result = Offline3DManager.from_environment().analyze("wham", "unused.mp4")

    assert result.status == "UNAVAILABLE"
    assert result.frames == []
    assert "not configured" in result.warnings[0]


def test_wham_command_uses_argv_template_without_shell(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(
        "POSE_WHAM_COMMAND_JSON",
        json.dumps([sys.executable, "runner.py", "{video_path}", "{output_json}"]),
    )
    template = command_template_from_environment()
    command = build_command(
        template,
        video_path=tmp_path / "clip with spaces.mp4",
        output_json=tmp_path / "result.json",
        output_dir=tmp_path,
    )

    assert command[0] == sys.executable
    assert command[-2:] == [str(tmp_path / "clip with spaces.mp4"), str(tmp_path / "result.json")]


def test_external_wham_backend_reads_contract_json(tmp_path: Path) -> None:
    runner = tmp_path / "fake_wham.py"
    runner.write_text(
        "import json, sys\n"
        "json.dump({'frames':[{'timestamp_ms':0,'joints_3d':{'pelvis':[0,0,0]}}]}, open(sys.argv[2], 'w'))\n",
        encoding="utf-8",
    )
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    backend = WhamBackend([sys.executable, str(runner), "{video_path}", "{output_json}"])

    result = backend.analyze(video, output_dir=tmp_path / "result")

    assert result.status == "COMPLETED"
    assert result.frames[0].joints_3d["pelvis"] == (0.0, 0.0, 0.0)
    assert result.metadata["isolated_environment"] is True
