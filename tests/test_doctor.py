from __future__ import annotations

from pathlib import Path

from src.doctor import build_parser, run_checks
from src.version import __version__


def test_doctor_parser_supports_machine_readable_and_camera_modes() -> None:
    args = build_parser().parse_args(["--json", "--strict", "--camera", "1", "--camera", "2"])

    assert args.json is True
    assert args.strict is True
    assert args.camera == [1, 2]


def test_doctor_reports_missing_required_project_assets(tmp_path: Path) -> None:
    checks = run_checks(project_root=tmp_path)
    by_name = {check.name: check for check in checks}

    assert by_name["runtime:python"].passed
    assert not by_name["model:pose"].passed
    assert by_name["model:pose"].required
    assert not by_name["config:hyrox"].passed
    assert by_name["output:writable"].passed


def test_program_has_a_nonempty_development_version() -> None:
    assert __version__
    assert __version__.startswith("0.")
