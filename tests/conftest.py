from __future__ import annotations

import uuid
from pathlib import Path


def pytest_configure(config) -> None:
    if config.option.basetemp is None:
        parent = Path(".cache") / "pytest-runs"
        parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(parent / uuid.uuid4().hex)
