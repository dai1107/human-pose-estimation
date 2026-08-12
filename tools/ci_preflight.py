"""Run deterministic local checks that mirror the GitHub Actions contract."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import py_compile
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_PYTHON_MIN = (3, 10)
SUPPORTED_PYTHON_MAX_EXCLUSIVE = (3, 13)
CI_PYTHON_MINORS = {(3, 10), (3, 12)}
GITHUB_HARD_BLOB_LIMIT = 100_000_000
SAFE_BLOB_LIMIT = 95 * 1024 * 1024
SAFE_PUSH_TOTAL = 500 * 1024 * 1024
BLOCKED_PATH_PREFIXES = (
    "datasets/",
    "outputs/web_uploads/",
    "outputs/recordings/",
)
BLOCKED_SUFFIXES = {
    ".avi",
    ".mkv",
    ".mov",
    ".mp4",
    ".oni",
    ".webm",
}


@dataclass(frozen=True)
class OutgoingBlob:
    object_id: str
    size: int
    path: str


class PreflightError(RuntimeError):
    pass


def _run(
    command: Sequence[str],
    *,
    capture: bool = False,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    print(f"+ {' '.join(command)}", flush=True)
    return subprocess.run(
        list(command),
        cwd=ROOT,
        check=True,
        text=True,
        input=input_text,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def check_environment(*, strict_ci_minor: bool = False) -> None:
    current = sys.version_info[:2]
    if not (SUPPORTED_PYTHON_MIN <= current < SUPPORTED_PYTHON_MAX_EXCLUSIVE):
        raise PreflightError(
            "Unsupported Python "
            f"{sys.version_info.major}.{sys.version_info.minor}; "
            "use Python 3.10-3.12."
        )
    if strict_ci_minor and current not in CI_PYTHON_MINORS:
        raise PreflightError(
            f"CI matrix must use Python 3.10 or 3.12, not {current[0]}.{current[1]}."
        )
    check_pinned_dependencies()


def check_pinned_dependencies() -> None:
    expected: dict[str, str] = {}
    for filename in ("requirements-core.txt", "requirements-dev.txt"):
        for raw_line in (ROOT / filename).read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith(("#", "-r ")):
                continue
            try:
                requirement = Requirement(line)
            except ValueError as exc:
                raise PreflightError(
                    f"{filename} contains a non-pinned requirement: {line}"
                ) from exc
            if requirement.marker is not None and not requirement.marker.evaluate():
                continue
            specifiers = list(requirement.specifier)
            if len(specifiers) != 1 or specifiers[0].operator != "==":
                raise PreflightError(
                    f"{filename} contains a non-pinned requirement: {line}"
                )
            expected[requirement.name] = specifiers[0].version
    errors: list[str] = []
    for name, expected_version in sorted(expected.items()):
        try:
            installed_version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{name}=={expected_version} is not installed")
            continue
        if installed_version != expected_version:
            errors.append(
                f"{name} expected {expected_version}, installed {installed_version}"
            )
    if errors:
        raise PreflightError(
            "Project dependency check failed:\n- " + "\n- ".join(errors)
        )
    print(
        f"Project dependency check passed: {len(expected)} pinned packages."
    )


def _tracked_python_files() -> list[Path]:
    output = _run(
        ["git", "ls-files", "--", "*.py"], capture=True
    ).stdout
    return [ROOT / line for line in output.splitlines() if line.strip()]


def check_static() -> None:
    failures: list[str] = []
    for path in _tracked_python_files():
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(f"{path.relative_to(ROOT)}: {exc.msg}")
    if failures:
        raise PreflightError(
            "Python compilation failed:\n- " + "\n- ".join(failures)
        )
    _run([sys.executable, "-m", "src.import_test"])
    _run([sys.executable, "tools/check_text_format.py"])


def _git_ref_exists(ref: str) -> bool:
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", ref],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _outgoing_revision_args(remote: str) -> list[str]:
    branch = _run(
        ["git", "branch", "--show-current"], capture=True
    ).stdout.strip()
    candidates = []
    if branch:
        candidates.append(f"refs/remotes/{remote}/{branch}")
    candidates.append(f"refs/remotes/{remote}/main")
    for candidate in candidates:
        if _git_ref_exists(candidate):
            return [f"{candidate}..HEAD"]
    return ["HEAD", "--not", f"--remotes={remote}"]


def collect_outgoing_blobs(remote: str = "origin") -> list[OutgoingBlob]:
    object_lines = _run(
        ["git", "rev-list", "--objects", *_outgoing_revision_args(remote)],
        capture=True,
    ).stdout.splitlines()
    paths: dict[str, str] = {}
    object_ids: list[str] = []
    for line in object_lines:
        object_id, _, path = line.partition(" ")
        if not object_id:
            continue
        object_ids.append(object_id)
        if path:
            paths.setdefault(object_id, path.replace("\\", "/"))
    if not object_ids:
        return []
    batch = _run(
        [
            "git",
            "cat-file",
            "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        ],
        capture=True,
        input_text="\n".join(object_ids) + "\n",
    ).stdout
    blobs: list[OutgoingBlob] = []
    for line in batch.splitlines():
        object_id, object_type, raw_size = line.split()
        if object_type == "blob":
            blobs.append(
                OutgoingBlob(
                    object_id=object_id,
                    size=int(raw_size),
                    path=paths.get(
                        object_id, "<historical-or-renamed-blob>"
                    ),
                )
            )
    return blobs


def validate_outgoing_blobs(
    blobs: Iterable[OutgoingBlob],
) -> dict[str, int]:
    items = list(blobs)
    total = sum(item.size for item in items)
    errors: list[str] = []
    for item in items:
        lowered = item.path.lower()
        if item.size >= GITHUB_HARD_BLOB_LIMIT:
            errors.append(
                f"{item.path}: {item.size} bytes exceeds GitHub's "
                "100 MB hard limit"
            )
        elif item.size > SAFE_BLOB_LIMIT:
            errors.append(
                f"{item.path}: {item.size} bytes exceeds the "
                f"{SAFE_BLOB_LIMIT}-byte safety limit"
            )
        if lowered.endswith(tuple(BLOCKED_SUFFIXES)) or lowered.startswith(
            BLOCKED_PATH_PREFIXES
        ):
            errors.append(
                f"{item.path}: generated data/video must not be pushed"
            )
    if total > SAFE_PUSH_TOTAL:
        errors.append(
            f"outgoing blobs total {total} bytes; safety limit is "
            f"{SAFE_PUSH_TOTAL} bytes"
        )
    if errors:
        raise PreflightError(
            "Outgoing Git object check failed:\n- " + "\n- ".join(errors)
        )
    summary = {
        "outgoing_blob_count": len(items),
        "outgoing_blob_bytes": total,
        "largest_outgoing_blob_bytes": max(
            (item.size for item in items), default=0
        ),
    }
    print(
        "Outgoing Git objects passed: "
        f"{summary['outgoing_blob_count']} blobs, "
        f"{summary['outgoing_blob_bytes']} uncompressed bytes."
    )
    return summary


def check_outgoing_git(remote: str) -> None:
    validate_outgoing_blobs(collect_outgoing_blobs(remote))


def verify_distributions(dist_dir: Path) -> None:
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    expected = {*wheels, *sdists}
    unexpected = sorted(
        path.name
        for path in dist_dir.iterdir()
        if path.is_file() and path not in expected
    )
    if len(wheels) != 1 or len(sdists) != 1 or unexpected:
        raise PreflightError(
            "Distribution count mismatch: expected exactly 1 wheel and "
            f"1 sdist; got wheels={len(wheels)}, sdists={len(sdists)}, "
            f"unexpected={unexpected}."
        )
    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_names = set(archive.namelist())
    with tarfile.open(sdists[0], "r:gz") as archive:
        sdist_names = {member.name for member in archive.getmembers()}
    forbidden_parts = (
        "/datasets/",
        "/outputs/",
        "/tools/oni_bridge/vendor/",
    )
    for artifact_name, names in (
        (wheels[0].name, wheel_names),
        (sdists[0].name, sdist_names),
    ):
        forbidden = []
        for name in names:
            normalized = "/" + name.replace("\\", "/")
            if any(marker in normalized for marker in forbidden_parts) or (
                name.lower().endswith(".oni")
            ):
                forbidden.append(name)
        forbidden.sort()
        if forbidden:
            preview = forbidden[:5]
            raise PreflightError(
                f"{artifact_name} contains forbidden local/generated files: "
                f"{preview}"
            )
    required_suffixes = (
        "webui/templates/index.html",
        "configs/product_pose.yaml",
        "models/pose_landmarker_lite.task",
    )
    for suffix in required_suffixes:
        if not any(name.endswith(suffix) for name in wheel_names):
            raise PreflightError(f"Wheel is missing required file: {suffix}")
        if not any(name.endswith(suffix) for name in sdist_names):
            raise PreflightError(
                f"Source distribution is missing required file: {suffix}"
            )
    print(
        f"Distribution check passed: {wheels[0].name} and "
        f"{sdists[0].name}."
    )


def run_full_preflight(remote: str) -> None:
    check_environment()
    check_outgoing_git(remote)
    check_static()
    _run([sys.executable, "-m", "pytest", "-q"])
    _run([sys.executable, "-m", "src.smoke_test"])
    with tempfile.TemporaryDirectory(prefix="pose_ci_dist_") as temporary:
        dist_dir = Path(temporary)
        _run(
            [
                sys.executable,
                "-m",
                "build",
                "--no-isolation",
                "--outdir",
                str(dist_dir),
            ]
        )
        verify_distributions(dist_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run checks shared by local pre-push and GitHub Actions."
    )
    parser.add_argument(
        "--stage",
        choices=("all", "environment", "git", "static", "package"),
        default="all",
    )
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.stage == "all":
            run_full_preflight(args.remote)
        elif args.stage == "environment":
            check_environment(
                strict_ci_minor=bool(os.environ.get("GITHUB_ACTIONS"))
            )
        elif args.stage == "git":
            check_outgoing_git(args.remote)
        elif args.stage == "static":
            check_static()
        elif args.stage == "package":
            verify_distributions(args.dist_dir.resolve())
    except (PreflightError, subprocess.CalledProcessError, OSError) as exc:
        print(f"CI preflight failed: {exc}", file=sys.stderr)
        return 1
    print("CI preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
