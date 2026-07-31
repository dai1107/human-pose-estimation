from __future__ import annotations

from pathlib import Path


def test_annotated_writer_uses_exact_source_fps() -> None:
    source = Path("webui/app.py").read_text(encoding="utf-8")
    writer_block = source[source.index("writer = cv2.VideoWriter("):]
    writer_block = writer_block[:writer_block.index("if not writer.isOpened()")]

    assert "source_fps" in writer_block
    assert "min(60.0" not in writer_block
    assert "processed_fps" not in writer_block
