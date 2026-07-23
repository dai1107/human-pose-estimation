from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for render tests")
def test_browser_render_performance_module() -> None:
    result = subprocess.run(
        ["node", "--test", str(Path("tests/js/render_performance.test.mjs"))],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert result.returncode == 0, result.stdout + result.stderr
