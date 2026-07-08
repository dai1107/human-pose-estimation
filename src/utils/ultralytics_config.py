from __future__ import annotations

import os
from pathlib import Path


def ensure_ultralytics_config_dir() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    config_root = project_root / ".cache" / "ultralytics"
    config_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_root))
    return config_root
