from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import pytest

from tools.ci_preflight import (
    GITHUB_HARD_BLOB_LIMIT,
    OutgoingBlob,
    PreflightError,
    validate_outgoing_blobs,
    verify_distributions,
)


def test_outgoing_git_check_rejects_video_and_github_oversize_blob() -> None:
    with pytest.raises(PreflightError, match="must not be pushed"):
        validate_outgoing_blobs(
            [OutgoingBlob("a" * 40, 1024, "datasets/raw/session.mp4")]
        )

    with pytest.raises(PreflightError, match="100 MB hard limit"):
        validate_outgoing_blobs(
            [
                OutgoingBlob(
                    "b" * 40,
                    GITHUB_HARD_BLOB_LIMIT,
                    "assets/large.bin",
                )
            ]
        )


def test_outgoing_git_check_reports_stable_counts() -> None:
    summary = validate_outgoing_blobs(
        [
            OutgoingBlob("a" * 40, 10, "src/a.py"),
            OutgoingBlob("b" * 40, 20, "src/b.py"),
        ]
    )

    assert summary == {
        "outgoing_blob_count": 2,
        "outgoing_blob_bytes": 30,
        "largest_outgoing_blob_bytes": 20,
    }


def test_distribution_check_requires_exactly_one_wheel_and_sdist(
    tmp_path: Path,
) -> None:
    with pytest.raises(PreflightError, match="Distribution count mismatch"):
        verify_distributions(tmp_path)

    wheel = tmp_path / "pose_estimation_hyrox-0.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in (
            "webui/templates/index.html",
            "configs/product_pose.yaml",
            "models/pose_landmarker_lite.task",
        ):
            archive.writestr(name, b"test")
    sdist = tmp_path / "pose_estimation_hyrox-0.1.tar.gz"
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    payload = fixture_dir / "payload"
    payload.write_bytes(b"test")
    with tarfile.open(sdist, "w:gz") as archive:
        for name in (
            "webui/templates/index.html",
            "configs/product_pose.yaml",
            "models/pose_landmarker_lite.task",
        ):
            archive.add(
                payload,
                arcname=f"pose-estimation-hyrox-0.1/{name}",
            )

    verify_distributions(tmp_path)


def test_distribution_check_rejects_local_vendor_files(tmp_path: Path) -> None:
    wheel = tmp_path / "pose_estimation_hyrox-0.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in (
            "webui/templates/index.html",
            "configs/product_pose.yaml",
            "models/pose_landmarker_lite.task",
            "tools/oni_bridge/vendor/legacy.py",
        ):
            archive.writestr(name, b"test")
    sdist = tmp_path / "pose_estimation_hyrox-0.1.tar.gz"
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    payload = fixture_dir / "payload"
    payload.write_bytes(b"test")
    with tarfile.open(sdist, "w:gz") as archive:
        for name in (
            "webui/templates/index.html",
            "configs/product_pose.yaml",
            "models/pose_landmarker_lite.task",
        ):
            archive.add(payload, arcname=f"package/{name}")

    with pytest.raises(PreflightError, match="forbidden"):
        verify_distributions(tmp_path)
