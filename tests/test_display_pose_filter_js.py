from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for Worker filter tests")
def test_browser_display_pose_filter_module() -> None:
    result = subprocess.run(
        ["node", "--test", str(Path("tests/js/display_pose_filter.test.mjs"))],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for predictor tests")
def test_browser_display_pose_predictor_module() -> None:
    result = subprocess.run(
        ["node", "--test", str(Path("tests/js/display_pose_predictor.test.mjs"))],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert result.returncode == 0, result.stdout + result.stderr
