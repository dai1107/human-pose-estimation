from __future__ import annotations

from pathlib import Path


def test_upload_first_pass_skips_drawing_and_jpeg_encoding() -> None:
    source = Path("webui/app.py").read_text(encoding="utf-8")

    assert 'publish_annotated_frame = config["source_mode"] != "upload"' in source
    assert "if publish_annotated_frame:" in source
    assert "annotated = frame" in source
    assert "jpeg = None" in source
    assert "draw_ms = 0.0" in source
    assert "jpeg_encode_ms = 0.0" in source


def test_full_annotated_video_is_opt_in_and_uses_cached_pose_results() -> None:
    backend = Path("webui/app.py").read_text(encoding="utf-8")
    frontend = Path("webui/static/app.js").read_text(encoding="utf-8")
    template = Path("webui/templates/index.html").read_text(encoding="utf-8")

    assert '"generate_annotated_video": False' in backend
    assert "render_cache.append(" in backend
    assert "_render_cached_annotated_video(" in backend
    assert '"rendered_from_pose_cache": True' in backend
    assert '"pose_inference_count": 0' in backend
    assert "backend.close()" in backend
    assert backend.index("backend.close()") < backend.index(
        "annotated_video_render = _render_cached_annotated_video("
    )
    assert 'ui.sourceMode === "upload"' in frontend
    assert '$("#annotatedVideoToggle").checked' in frontend
    assert 'id="annotatedVideoToggle"' in template
    assert 'id="downloadAnnotated"' in template
