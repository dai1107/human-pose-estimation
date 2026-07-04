from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    modules = (
        "cv2",
        "mediapipe",
        "numpy",
        "src.realtime_pose",
        "src.biomechanics.types",
        "src.biomechanics.landmarks",
        "src.biomechanics.normalization",
        "src.biomechanics.angles",
        "src.biomechanics.segments",
        "src.biomechanics.velocity",
        "src.biomechanics.stability",
        "src.biomechanics.sequencing",
        "src.biomechanics.session_writer",
        "src.biomechanics.report",
        "src.ui.metrics_overlay",
        "src.utils.time_utils",
    )
    missing: list[str] = []
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            missing.append(f"{module_name}: {exc}")

    if missing:
        print("Import test failed:")
        for item in missing:
            print(f"  - {item}")
        print("Install dependencies with: python -m pip install -r requirements.txt")
        return 1

    print("Import test passed: core dependencies and project modules are importable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
