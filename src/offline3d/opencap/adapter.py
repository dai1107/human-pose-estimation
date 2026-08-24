from __future__ import annotations

import json
import os
import shlex
from pathlib import Path


COMMAND_JSON_ENV = "POSE_OPENCAP_COMMAND_JSON"
COMMAND_ENV = "POSE_OPENCAP_COMMAND"


def command_template_from_environment() -> list[str]:
    raw_json = os.environ.get(COMMAND_JSON_ENV, "").strip()
    if raw_json:
        parsed = json.loads(raw_json)
        if not isinstance(parsed, list) or not parsed or not all(
            isinstance(value, str) and value for value in parsed
        ):
            raise ValueError(f"{COMMAND_JSON_ENV} must be a non-empty JSON string array")
        return list(parsed)
    raw = os.environ.get(COMMAND_ENV, "").strip()
    return shlex.split(raw, posix=os.name != "nt") if raw else []


def build_command(
    template: list[str],
    *,
    video_path: Path,
    input_json: Path,
    output_json: Path,
    output_dir: Path,
) -> list[str]:
    values = {
        "video_path": str(video_path),
        "input_json": str(input_json),
        "output_json": str(output_json),
        "output_dir": str(output_dir),
    }
    command = [argument.format_map(values) for argument in template]
    if not any("{video_path}" in argument for argument in template):
        command.extend(["--video", str(video_path)])
    if not any("{input_json}" in argument for argument in template):
        command.extend(["--input-json", str(input_json)])
    if not any("{output_json}" in argument for argument in template):
        command.extend(["--output-json", str(output_json)])
    return command
