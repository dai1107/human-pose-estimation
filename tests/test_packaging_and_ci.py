from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_python_cli_and_optional_dependency_groups() -> None:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = payload["project"]

    assert project["requires-python"] == ">=3.10,<3.13"
    assert {"yolo", "rtmw-cpu", "rtmw-gpu", "dev"} <= set(
        project["optional-dependencies"]
    )
    assert {
        "pose-estimation",
        "pose-doctor",
        "pose-web",
        "pose-replay",
        "pose-golden",
        "pose-endurance",
        "pose-baseline",
        "pose-dataset-manifest",
        "pose-oni-audit",
        "pose-oni-export",
        "pose-oni-sync",
        "pose-clean",
    } <= set(project["scripts"])
    assert "configs/hyrox_golden_videos.json" in payload["tool"]["setuptools"]["data-files"]["configs"]
    assert "configs/product_pose.yaml" in payload["tool"]["setuptools"]["data-files"]["configs"]


def test_verified_requirement_files_use_exact_direct_versions() -> None:
    for filename in (
        "requirements-core.txt",
        "requirements-yolo.txt",
        "requirements-dev.txt",
        "requirements-rtmw-cpu.txt",
        "requirements-rtmw-gpu.txt",
    ):
        lines = (ROOT / filename).read_text(encoding="utf-8").splitlines()
        requirements = [
            line.strip()
            for line in lines
            if line.strip() and not line.lstrip().startswith(("#", "-r "))
        ]
        assert requirements
        assert all(
            re.fullmatch(
                r'[A-Za-z0-9_.-]+==[^=<>!~;]+(?:; python_version < "3\.11")?',
                item,
            )
            for item in requirements
        )

    development_requirements = (ROOT / "requirements-dev.txt").read_text(
        encoding="utf-8"
    )
    assert "tomli==" in development_requirements


def test_ci_covers_windows_linux_static_tests_smoke_and_build() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    for required in (
        "ubuntu-latest",
        "windows-latest",
        'python-version: ["3.10", "3.12"]',
        "ci_preflight.py --stage environment",
        "ci_preflight.py --stage static",
        "pytest -q",
        "src.smoke_test",
        "python -m build --no-isolation",
        "ci_preflight.py --stage package --dist-dir dist",
        "python -m pip check",
    ):
        assert required in workflow


def test_repository_pre_push_hook_runs_full_ci_preflight() -> None:
    hook = (ROOT / ".githooks" / "pre-push").read_text(encoding="utf-8")
    installer = (ROOT / "scripts" / "install_git_hooks.ps1").read_text(
        encoding="utf-8"
    )

    assert "ci_preflight.py --stage all" in hook
    assert "core.hooksPath .githooks" in installer


def test_release_and_upgrade_documents_exist() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    releasing = (ROOT / "RELEASING.md").read_text(encoding="utf-8")

    assert "0.1.0.dev0" in changelog
    assert "Semantic Versioning" in releasing
    assert "schema_version" in releasing
    assert "pose-clean" in releasing
