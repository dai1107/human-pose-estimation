from __future__ import annotations

import shutil
import uuid
from pathlib import Path


def pytest_configure(config) -> None:
    if config.option.basetemp is None:
        parent = Path(".cache") / "pytest-runs"
        parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(parent / uuid.uuid4().hex)


def pytest_unconfigure(config) -> None:
    basetemp = getattr(config.option, "basetemp", None)
    if not basetemp:
        return
    root = (Path(".cache") / "pytest-runs").resolve()
    target = Path(basetemp).resolve()
    if target.parent != root:
        return
    shutil.rmtree(target, ignore_errors=True)
    try:
        root.rmdir()
    except OSError:
        pass
