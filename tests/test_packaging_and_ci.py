from __future__ import annotations

import re
import tomllib
from pathlib import Path


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
        "pose-clean",
    } <= set(project["scripts"])


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
        assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[^=<>!~]+", item) for item in requirements)


def test_ci_covers_windows_linux_static_tests_smoke_and_build() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    for required in (
        "ubuntu-latest",
        "windows-latest",
        "compileall",
        "src.import_test",
        "check_text_format.py",
        "pytest -q",
        "src.smoke_test",
        "python -m build",
    ):
        assert required in workflow


def test_release_and_upgrade_documents_exist() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    releasing = (ROOT / "RELEASING.md").read_text(encoding="utf-8")

    assert "0.1.0.dev0" in changelog
    assert "Semantic Versioning" in releasing
    assert "schema_version" in releasing
    assert "pose-clean" in releasing
