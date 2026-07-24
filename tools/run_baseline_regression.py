"""One-command phase-zero regression and baseline freeze."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _run(
    name: str,
    command: list[str],
    *,
    output_dir: Path,
    required: bool,
) -> dict[str, Any]:
    print(f"[baseline] {name}: {' '.join(command)}", flush=True)
    started = datetime.now(timezone.utc)
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except OSError as exc:
        return_code = 127
        stdout = ""
        stderr = str(exc)
    finished = datetime.now(timezone.utc)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{name}.stdout.log").write_text(stdout, encoding="utf-8")
    (log_dir / f"{name}.stderr.log").write_text(stderr, encoding="utf-8")
    return {
        "name": name,
        "required": required,
        "status": "passed" if return_code == 0 else "failed",
        "return_code": return_code,
        "command": command,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": round((finished - started).total_seconds(), 3),
        "stdout_log": f"logs/{name}.stdout.log",
        "stderr_log": f"logs/{name}.stderr.log",
    }


def _write_summary(output_dir: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    failed_required = [
        item for item in checks
        if item["required"] and item["status"] != "passed"
    ]
    payload = {
        "schema_version": 1,
        "artifact_type": "phase_zero_test_summary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not failed_required else "failed",
        "checks": checks,
        "failed_required": [item["name"] for item in failed_required],
        "notes": [
            "Python tests include the existing web/JavaScript wrapper tests.",
            "no_camera_smoke proves startup without ONI/OpenNI and with neural prediction disabled.",
            "A physical camera check cannot be replaced by an offline or mocked result.",
        ],
    }
    (output_dir / "test_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 阶段 0 基线测试汇总",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 生成时间：`{payload['generated_at']}`",
        "",
        "| 检查 | 必需 | 状态 | 耗时（秒） |",
        "|---|---:|---:|---:|",
    ]
    for item in checks:
        lines.append(
            f"| `{item['name']}` | {'是' if item['required'] else '否'} "
            f"| `{item['status']}` | {item['duration_seconds']} |"
        )
    lines.extend(
        [
            "",
            "详细标准输出和错误输出位于 `logs/`。实际显示延迟与 "
            "sensor-to-photon 必须在目标设备上通过 120/240 FPS 外部录像补测。",
            "",
        ]
    )
    (output_dir / "test_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all existing tests, smoke checks, camera check and freeze the phase-zero baseline."
    )
    parser.add_argument("--output-dir", default="reports/baseline")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-duration", type=float, default=1.0)
    parser.add_argument("--camera-warmup", type=float, default=0.25)
    parser.add_argument(
        "--skip-camera",
        action="store_true",
        help="Skip physical camera checks (recorded as skipped, not passed).",
    )
    parser.add_argument(
        "--skip-node",
        action="store_true",
        help="Skip the direct Node test pass; Python wrapper tests still run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    checks: list[dict[str, Any]] = []

    checks.append(
        _run(
            "python_tests",
            [python, "-m", "pytest", "-q"],
            output_dir=output_dir,
            required=True,
        )
    )
    if not args.skip_node:
        node = shutil.which("node")
        js_tests = sorted(str(path) for path in (PROJECT_ROOT / "tests" / "js").glob("*.test.mjs"))
        if node:
            checks.append(
                _run(
                    "web_node_tests",
                    [node, "--test", *js_tests],
                    output_dir=output_dir,
                    required=True,
                )
            )
        else:
            checks.append(
                {
                    "name": "web_node_tests",
                    "required": True,
                    "status": "failed",
                    "return_code": 127,
                    "command": ["node", "--test"],
                    "started_at": None,
                    "finished_at": None,
                    "duration_seconds": 0.0,
                    "stdout_log": None,
                    "stderr_log": None,
                    "reason": "node executable not found",
                }
            )
    checks.append(
        _run(
            "no_camera_smoke",
            [python, "-m", "src.smoke_test"],
            output_dir=output_dir,
            required=True,
        )
    )

    camera_report = output_dir / "camera_report.json"
    camera_benchmark_passed = False
    if args.skip_camera:
        checks.append(
            {
                "name": "physical_camera",
                "required": False,
                "status": "skipped",
                "return_code": None,
                "command": [],
                "started_at": None,
                "finished_at": None,
                "duration_seconds": 0.0,
                "stdout_log": None,
                "stderr_log": None,
                "reason": "--skip-camera",
            }
        )
    else:
        checks.append(
            _run(
                "camera_startup",
                [
                    python,
                    "-m",
                    "src.doctor",
                    "--camera",
                    str(args.camera_index),
                    "--json",
                ],
                output_dir=output_dir,
                required=True,
            )
        )
        camera_check = _run(
                "physical_camera_benchmark",
                [
                    python,
                    "-m",
                    "tools.benchmark_camera_backends",
                    "--camera",
                    str(args.camera_index),
                    "--backends",
                    "default",
                    "--duration",
                    str(args.camera_duration),
                    "--warmup",
                    str(args.camera_warmup),
                    "--output",
                    str(camera_report),
                    "--no-cache-update",
                ],
                output_dir=output_dir,
                required=True,
            )
        checks.append(camera_check)
        camera_benchmark_passed = camera_check["status"] == "passed"

    freeze_command = [
        python,
        "tools/freeze_baseline.py",
        "--output-dir",
        str(output_dir),
    ]
    if camera_benchmark_passed and camera_report.is_file():
        freeze_command.extend(["--camera-report", str(camera_report)])
    checks.append(
        _run(
            "golden_and_latency_baseline",
            freeze_command,
            output_dir=output_dir,
            required=True,
        )
    )
    summary = _write_summary(output_dir, checks)
    print(f"[baseline] {summary['status']}: {output_dir}")
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
