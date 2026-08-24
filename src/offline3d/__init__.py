"""Isolated offline 3D analysis interfaces.

Offline backends are deliberately separate from the real-time pose backends:
they may run in another Python environment or service and must never become a
dependency of the MediaPipe/HYROX product path.
"""

from .base import Offline3DBackend, Offline3DFrame, Offline3DResult
from .manager import Offline3DManager
from .alignment import MotionAlignmentResult, MotionFrame, align_mediapipe_wham

__all__ = [
    "Offline3DBackend",
    "Offline3DFrame",
    "Offline3DManager",
    "Offline3DResult",
    "MotionAlignmentResult",
    "MotionFrame",
    "align_mediapipe_wham",
]
