from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..base import Offline3DBackend, Offline3DResult, ProgressCallback
from .adapter import build_command, command_template_from_environment
from .parser import parse_wham_file


class WhamBackend(Offline3DBackend):
    """WHAM adapter for an isolated environment, WSL, Docker or service CLI."""

    name = "wham"

    def __init__(self, command_template: list[str], *, timeout_seconds: float = 7200.0) -> None:
        self.command_template = list(command_template)
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.unavailable_reason = (
            "WHAM external command is not configured; set POSE_WHAM_COMMAND_JSON "
            "to an argv template with {video_path} and {output_json} placeholders"
        )
        self._process_lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._cancel_requested = threading.Event()

    @classmethod
    def from_environment(cls) -> "WhamBackend":
        timeout = float(os.environ.get("POSE_WHAM_TIMEOUT_SECONDS", "7200"))
        try:
            command = command_template_from_environment()
        except (ValueError, TypeError) as exc:
            backend = cls([], timeout_seconds=timeout)
            backend.unavailable_reason = f"invalid WHAM command configuration: {exc}"
            return backend
        if not command and os.name == "nt" and os.environ.get(
            "POSE_WHAM_AUTO_WSL", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}:
            config_path = _wsl_config_path()
            if config_path is not None:
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("wsl_runner.py")),
                    "--config",
                    str(config_path),
                    "--video",
                    "{video_path}",
                    "--output-json",
                    "{output_json}",
                    "--output-dir",
                    "{output_dir}",
                ]
        return cls(command, timeout_seconds=timeout)

    @property
    def available(self) -> bool:
        return bool(self.command_template)

    def analyze(
        self,
        video_path: str | Path,
        *,
        output_dir: str | Path | None = None,
        progress: ProgressCallback | None = None,
    ) -> Offline3DResult:
        source = Path(video_path).resolve()
        if not source.is_file():
            return Offline3DResult.failed(self.name, f"video does not exist: {source}")
        target_dir = Path(output_dir or source.parent / "offline3d_wham").resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        output_json = target_dir / "wham_result.json"
        command = build_command(
            self.command_template,
            video_path=source,
            output_json=output_json,
            output_dir=target_dir,
        )
        started = time.perf_counter()
        self._cancel_requested.clear()
        if progress is not None:
            progress(0.0, "WHAM 3D 分析已启动")
        try:
            stdout_log = target_dir / "wham_stdout.log"
            stderr_log = target_dir / "wham_stderr.log"
            with stdout_log.open("w", encoding="utf-8") as stdout_handle, stderr_log.open(
                "w", encoding="utf-8"
            ) as stderr_handle:
                process = subprocess.Popen(
                    command,
                    cwd=target_dir,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                    shell=False,
                )
                with self._process_lock:
                    self._process = process
                deadline = time.monotonic() + self.timeout_seconds
                while process.poll() is None:
                    if self._cancel_requested.wait(0.2):
                        process.terminate()
                        try:
                            process.wait(timeout=5.0)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5.0)
                        return Offline3DResult(
                            backend=self.name,
                            status="CANCELLED",
                            reference_source="WHAM reconstructed 3D",
                            warnings=["WHAM analysis was cancelled"],
                        )
                    if time.monotonic() >= deadline:
                        process.terminate()
                        try:
                            process.wait(timeout=5.0)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        return Offline3DResult.failed(
                            self.name, f"WHAM timed out after {self.timeout_seconds:g} seconds"
                        )
            if process.returncode != 0:
                stderr = stderr_log.read_text(encoding="utf-8", errors="replace")
                stdout = stdout_log.read_text(encoding="utf-8", errors="replace")
                detail = (stderr or stdout or "").strip()[-2000:]
                return Offline3DResult.failed(
                    self.name,
                    f"WHAM external process exited with {process.returncode}: {detail}",
                )
            if not output_json.is_file():
                return Offline3DResult.failed(
                    self.name, f"WHAM did not create expected output: {output_json}"
                )
            result = parse_wham_file(output_json)
            result.processing_time_ms = (time.perf_counter() - started) * 1000.0
            result.metadata.update(
                {
                    "execution": "external_process",
                    "isolated_environment": True,
                    "output_json": str(output_json),
                    "stdout_log": str(stdout_log),
                    "stderr_log": str(stderr_log),
                }
            )
            if progress is not None:
                progress(100.0, "WHAM 3D 分析完成")
            return result
        except (OSError, ValueError) as exc:
            return Offline3DResult.failed(self.name, str(exc))
        finally:
            with self._process_lock:
                self._process = None

    def cancel(self) -> None:
        self._cancel_requested.set()


def _wsl_config_path() -> Path | None:
    configured = os.environ.get("POSE_WHAM_WSL_CONFIG", "").strip()
    path = (
        Path(configured).expanduser()
        if configured
        else Path(__file__).resolve().parents[3] / "configs" / "wham_wsl.json"
    )
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None
    return path.resolve() if isinstance(payload, dict) and payload.get("enabled") else None
