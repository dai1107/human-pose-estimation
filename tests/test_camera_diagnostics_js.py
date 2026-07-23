from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for camera tests")
def test_browser_camera_diagnostics_module() -> None:
    result = subprocess.run(
        ["node", "--test", str(Path("tests/js/camera_diagnostics.test.mjs"))],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr
