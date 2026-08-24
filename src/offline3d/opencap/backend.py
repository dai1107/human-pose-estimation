from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from ..alignment import MotionAlignmentResult
from ..base import Offline3DBackend, Offline3DResult, ProgressCallback
from .adapter import build_command, command_template_from_environment
from .parser import parse_opencap_file


class OpenCapBackend(Offline3DBackend):
    """External OpenCap Monocular/OpenSim IK refinement adapter."""

    name = "opencap"

    def __init__(self, command_template: list[str], *, timeout_seconds: float = 14400.0) -> None:
        self.command_template = list(command_template)
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.unavailable_reason = (
            "OpenCap external command is not configured; set "
            "POSE_OPENCAP_COMMAND_JSON to an isolated Ubuntu/WSL/Docker adapter"
        )
        self._process_lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._cancel_requested = threading.Event()

    @classmethod
    def from_environment(cls) -> "OpenCapBackend":
        timeout = float(os.environ.get("POSE_OPENCAP_TIMEOUT_SECONDS", "14400"))
        try:
            command = command_template_from_environment()
        except (ValueError, TypeError) as exc:
            backend = cls([], timeout_seconds=timeout)
            backend.unavailable_reason = f"invalid OpenCap command configuration: {exc}"
            return backend
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
        return Offline3DResult.unavailable(
            self.name,
            "OpenCap refinement requires a completed WHAM result and timestamp alignment bundle",
        )

    def analyze_with_reference(
        self,
        video_path: str | Path,
        *,
        wham_result: Offline3DResult,
        alignment: MotionAlignmentResult,
        output_dir: str | Path | None = None,
        progress: ProgressCallback | None = None,
    ) -> Offline3DResult:
        if wham_result.status != "COMPLETED":
            return Offline3DResult.unavailable(
                self.name, f"OpenCap requires completed WHAM input; got {wham_result.status}"
            )
        if alignment.status != "COMPLETED":
            return Offline3DResult.unavailable(
                self.name, f"OpenCap requires completed timestamp alignment; got {alignment.status}"
            )
        if not self.available:
            return Offline3DResult.unavailable(self.name, self.unavailable_reason)
        source = Path(video_path).resolve()
        if not source.is_file():
            return Offline3DResult.failed(self.name, f"video does not exist: {source}")
        target_dir = Path(output_dir or source.parent / "offline3d_opencap").resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        input_json = target_dir / "opencap_input.json"
        output_json = target_dir / "opencap_result.json"
        with input_json.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": "opencap_refinement_input_v1",
                    "video_path": str(source),
                    "wham": wham_result.as_dict(),
                    "motion_alignment": alignment.as_dict(),
                    "policy": {
                        "formal_hyrox_rule_replacement_allowed": False,
                        "output_role": "biomechanical_reference",
                    },
                },
                handle,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        command = build_command(
            self.command_template,
            video_path=source,
            input_json=input_json,
            output_json=output_json,
            output_dir=target_dir,
        )
        started = time.perf_counter()
        self._cancel_requested.clear()
        if progress is not None:
            progress(0.0, "OpenCap / OpenSim IK 优化已启动")
        try:
            stdout_log = target_dir / "opencap_stdout.log"
            stderr_log = target_dir / "opencap_stderr.log"
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
                            reference_source="OpenCap Monocular / OpenSim IK reference",
                            warnings=["OpenCap analysis was cancelled"],
                        )
                    if time.monotonic() >= deadline:
                        process.terminate()
                        try:
                            process.wait(timeout=5.0)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        return Offline3DResult.failed(
                            self.name, f"OpenCap timed out after {self.timeout_seconds:g} seconds"
                        )
            if process.returncode != 0:
                stderr = stderr_log.read_text(encoding="utf-8", errors="replace")
                stdout = stdout_log.read_text(encoding="utf-8", errors="replace")
                detail = (stderr or stdout or "").strip()[-2000:]
                return Offline3DResult.failed(
                    self.name,
                    f"OpenCap external process exited with {process.returncode}: {detail}",
                )
            if not output_json.is_file():
                return Offline3DResult.failed(
                    self.name, f"OpenCap did not create expected output: {output_json}"
                )
            result = parse_opencap_file(output_json)
            result.processing_time_ms = (time.perf_counter() - started) * 1000.0
            result.metadata.update(
                {
                    "execution": "external_process",
                    "isolated_environment": True,
                    "initial_pose_source": "wham",
                    "alignment_source": "source_timestamp_ms",
                    "input_json": str(input_json),
                    "output_json": str(output_json),
                    "stdout_log": str(stdout_log),
                    "stderr_log": str(stderr_log),
                }
            )
            if progress is not None:
                progress(100.0, "OpenCap / OpenSim IK 优化完成")
            return result
        except (OSError, ValueError) as exc:
            return Offline3DResult.failed(self.name, str(exc))
        finally:
            with self._process_lock:
                self._process = None

    def cancel(self) -> None:
        self._cancel_requested.set()
