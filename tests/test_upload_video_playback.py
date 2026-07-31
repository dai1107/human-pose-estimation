from __future__ import annotations

from pathlib import Path


def test_uploaded_video_uses_analyze_then_play_timeline() -> None:
    source = Path("webui/static/app.js").read_text(encoding="utf-8")

    assert "URL.createObjectURL(file)" in source
    assert "video.currentTime * 1000" in source
    assert "timelineFrameAt(currentTimestampMs)" in source
    assert "differenceMs <= 150" in source
    assert "startUploadPlayback(playbackReport)" in source
    assert 'const analyzeThenPlay = ui.sourceMode === "upload"' in source
    assert "分析完成后将按原始视频时间正常播放" in source


def test_upload_analysis_is_not_paced_by_source_playback_rate() -> None:
    source = Path("webui/app.py").read_text(encoding="utf-8")

    assert 'if config["source_mode"] == "sample":' in source
    assert "remaining = (1.0 / source_fps) - elapsed" in source
    assert "self._stop_event.wait(remaining)" in source
    pacing_block = source[
        source.index("# Bundled samples still use the legacy MJPEG preview"):source.index(
            "except Exception as exc:"
        )
    ]
    assert 'if config["source_mode"] != "camera":' not in pacing_block
