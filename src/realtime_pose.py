"""Deprecated compatibility facade for the former MediaPipe-only runtime.

The independent realtime loop was retired in the fourth maturity batch.  Old
imports keep working, while execution is forwarded to the consolidated desktop
runtime in :mod:`src.realtime.app`.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

from src.realtime.app import main as _consolidated_main
from src.realtime.legacy_compat import (
    DrawLandmark,
    LandmarkSmoother,
    infer_hand_side,
    parse_args,
    translate_legacy_args,
)

__all__ = [
    "DrawLandmark",
    "LandmarkSmoother",
    "infer_hand_side",
    "main",
    "parse_args",
    "translate_legacy_args",
]


def main(argv: Sequence[str] | None = None) -> int:
    warnings.warn(
        "src.realtime_pose is deprecated; use main.py or the pose-estimation CLI",
        DeprecationWarning,
        stacklevel=2,
    )
    return _consolidated_main(translate_legacy_args(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
