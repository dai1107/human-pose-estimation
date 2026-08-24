from __future__ import annotations

"""Windows-to-WSL bridge for the isolated official WHAM adapter."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Sequence


def _run_wsl(distro: str, arguments: Sequence[str], *, capture: bool = False) -> str:
    command = ["wsl.exe", "-d", distro, "--", *arguments]
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if completed.returncode != 0:
        detail = ((completed.stderr or completed.stdout or "").strip())[-2000:]
        raise RuntimeError(
            f"WSL command failed with {completed.returncode}: {detail or command[0]}"
        )
    return (completed.stdout or "").strip()


def _wsl_path(distro: str, path: Path) -> str:
    return _run_wsl(
        distro,
        ["wslpath", "-a", str(path.resolve())],
        capture=True,
    )


def _wsl_home(distro: str) -> PurePosixPath:
    value = _run_wsl(distro, ["sh", "-lc", 'printf %s "$HOME"'], capture=True)
    if not value.startswith("/"):
        raise RuntimeError(f"WSL returned an invalid home directory: {value!r}")
    return PurePosixPath(value)


def _resolve_linux_path(home: PurePosixPath, value: str) -> str:
    candidate = PurePosixPath(value)
    return str(candidate if candidate.is_absolute() else home / candidate)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run official WHAM in WSL2")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or not config.get("enabled", False):
        raise RuntimeError("WHAM WSL configuration is disabled")
    distro = str(config.get("distro", "Ubuntu"))
    home = _wsl_home(distro)
    wham_root = _resolve_linux_path(home, str(config["wham_root"]))
    python_path = _resolve_linux_path(home, str(config["python_path"]))

    adapter_windows = Path(__file__).resolve().with_name("official_adapter.py")
    command = [
        python_path,
        _wsl_path(distro, adapter_windows),
        "--video",
        _wsl_path(distro, Path(args.video)),
        "--output-json",
        _wsl_path(distro, Path(args.output_json)),
        "--output-dir",
        _wsl_path(distro, Path(args.output_dir)),
        "--wham-root",
        wham_root,
    ]
    if bool(config.get("estimate_local_only", True)):
        command.append("--estimate-local-only")
    if bool(config.get("run_smplify", False)):
        command.append("--run-smplify")
    _run_wsl(distro, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
