from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from hyrox.config import (
    ConfigValidationError,
    load_burpee_broad_jump_config,
    load_farmers_carry_config,
    load_lunge_config,
    load_observability_config,
    load_rowing_config,
    load_skierg_config,
    load_sled_pull_config,
    load_sled_push_config,
    load_wall_ball_config,
    validate_auxiliary_config,
)
from src.version import __version__
from src.paths import installation_root, runtime_output_root
from src.output_schema import artifact_metadata


PROJECT_ROOT = installation_root()
MINIMUM_PYTHON = (3, 10)


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    required: bool
    message: str

    @property
    def passed(self) -> bool:
        return self.status == "pass"


def _dependency_check(module_name: str, distribution_name: str, *, required: bool) -> CheckResult:
    if importlib.util.find_spec(module_name) is None:
        level = "FAIL" if required else "INFO"
        return CheckResult(
            name=f"dependency:{distribution_name}",
            status="fail" if required else "skip",
            required=required,
            message=f"{level}: not installed",
        )
    try:
        version = importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        version = "installed (version unknown)"
    return CheckResult(f"dependency:{distribution_name}", "pass", required, version)


def _file_check(name: str, path: Path, *, required: bool, minimum_bytes: int = 1) -> CheckResult:
    if not path.is_file():
        return CheckResult(name, "fail" if required else "skip", required, f"missing: {path}")
    size = path.stat().st_size
    if size < minimum_bytes:
        return CheckResult(name, "fail", required, f"file is unexpectedly small ({size} bytes): {path}")
    return CheckResult(name, "pass", required, f"{path.name} ({size / 1024 / 1024:.1f} MiB)")


def _output_check(output_dir: Path) -> CheckResult:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix="doctor_", suffix=".tmp", dir=output_dir, delete=True):
            pass
    except OSError as exc:
        return CheckResult("output:writable", "fail", True, f"cannot write {output_dir}: {exc}")
    return CheckResult("output:writable", "pass", True, str(output_dir))


def _hyrox_config_check(root: Path) -> CheckResult:
    config_dir = root / "configs" / "hyrox"
    loaders = {
        "lunge.yaml": load_lunge_config,
        "wall_ball.yaml": load_wall_ball_config,
        "farmers_carry.yaml": load_farmers_carry_config,
        "rowing.yaml": load_rowing_config,
        "skierg.yaml": load_skierg_config,
        "burpee_broad_jump.yaml": load_burpee_broad_jump_config,
        "sled_push.yaml": load_sled_push_config,
        "sled_pull.yaml": load_sled_pull_config,
        "observability.yaml": load_observability_config,
    }
    expected = set(loaders) | {"contact.yaml", "foot_events.yaml"}
    present = {path.name for path in config_dir.glob("*.yaml")} if config_dir.is_dir() else set()
    missing = sorted(expected - present)
    if missing:
        return CheckResult("config:hyrox", "fail", True, f"missing: {', '.join(missing)}")
    try:
        for filename, loader in loaders.items():
            loader(config_dir / filename)
        validate_auxiliary_config("contact", config_dir / "contact.yaml")
        validate_auxiliary_config("foot_events", config_dir / "foot_events.yaml")
    except ConfigValidationError as exc:
        return CheckResult("config:hyrox", "fail", True, str(exc))
    return CheckResult(
        "config:hyrox",
        "pass",
        True,
        "8 action configs and 3 shared configs are valid",
    )


def _reference_config_check(root: Path) -> CheckResult:
    feature_path = root / "configs" / "reference_features.yaml"
    quality_path = root / "configs" / "reference_quality.yaml"
    missing = [str(path) for path in (feature_path, quality_path) if not path.is_file()]
    if missing:
        return CheckResult(
            "config:reference",
            "fail",
            True,
            f"missing: {', '.join(missing)}",
        )
    try:
        from src.reference.features import load_feature_config
        from src.reference.quality import load_quality_rules

        load_feature_config(feature_path)
        load_quality_rules(quality_path)
    except (ConfigValidationError, ImportError) as exc:
        return CheckResult("config:reference", "fail", True, str(exc))
    return CheckResult("config:reference", "pass", True, "reference configs are valid")


def _camera_check(camera_index: int) -> CheckResult:
    try:
        import cv2

        capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(camera_index)
        try:
            opened = capture.isOpened()
            readable, frame = capture.read() if opened else (False, None)
        finally:
            capture.release()
    except Exception as exc:
        return CheckResult(f"camera:{camera_index}", "fail", True, f"camera check failed: {exc}")
    if not opened:
        return CheckResult(f"camera:{camera_index}", "fail", True, f"camera {camera_index} could not be opened")
    if not readable or frame is None:
        return CheckResult(f"camera:{camera_index}", "fail", True, f"camera {camera_index} opened but returned no frame")
    return CheckResult(f"camera:{camera_index}", "pass", True, f"camera {camera_index}: {frame.shape[1]}x{frame.shape[0]}")


def run_checks(
    *,
    project_root: Path = PROJECT_ROOT,
    camera_indices: Sequence[int] | None = None,
    output_root: Path | None = None,
) -> list[CheckResult]:
    python_ok = sys.version_info[:2] >= MINIMUM_PYTHON
    checks = [
        CheckResult(
            "runtime:python",
            "pass" if python_ok else "fail",
            True,
            f"{platform.python_version()} ({sys.executable})",
        ),
        _dependency_check("numpy", "numpy", required=True),
        _dependency_check("cv2", "opencv-python", required=True),
        _dependency_check("mediapipe", "mediapipe", required=True),
        _dependency_check("PIL", "Pillow", required=True),
        _dependency_check("matplotlib", "matplotlib", required=True),
        _dependency_check("flask", "Flask", required=True),
        _dependency_check("flask_sock", "Flask-Sock", required=True),
        _dependency_check("simple_websocket", "simple-websocket", required=True),
        _dependency_check("ultralytics", "ultralytics", required=False),
        _file_check("model:pose", project_root / "models" / "pose_landmarker_full.task", required=True, minimum_bytes=1024),
        _file_check("model:hand", project_root / "models" / "hand_landmarker.task", required=False, minimum_bytes=1024),
        _file_check("model:yolo-pose", project_root / "yolo11n-pose.pt", required=False, minimum_bytes=1024),
        _hyrox_config_check(project_root),
        _reference_config_check(project_root),
        _output_check(
            output_root
            if output_root is not None
            else (
                runtime_output_root()
                if project_root == PROJECT_ROOT
                else project_root / "outputs"
            )
        ),
    ]
    for camera_index in camera_indices or ():
        checks.append(_camera_check(camera_index))
    return checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check whether this pose-estimation project is ready to run.")
    parser.add_argument("--camera", type=int, action="append", default=[], help="Open and read one frame from this camera index. Repeat to check multiple cameras.")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable JSON report.")
    parser.add_argument("--strict", action="store_true", help="Treat missing optional YOLO/hand components as failures.")
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checks = run_checks(camera_indices=args.camera)
    failed = [check for check in checks if not check.passed and (check.required or args.strict)]
    if args.json:
        payload = {
            **artifact_metadata("doctor_report"),
            "ready": not failed,
            "platform": platform.platform(),
            "checks": [asdict(check) for check in checks],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Pose Estimation Doctor {__version__}")
        for check in checks:
            label = "PASS" if check.passed else ("FAIL" if check.required or args.strict else "SKIP")
            print(f"[{label}] {check.name}: {check.message}")
        print("READY" if not failed else f"NOT READY: {len(failed)} required check(s) failed")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
