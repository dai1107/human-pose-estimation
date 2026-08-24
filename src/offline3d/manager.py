from __future__ import annotations

from pathlib import Path

from .base import Offline3DBackend, Offline3DResult, ProgressCallback
from .alignment import MotionAlignmentResult


class Offline3DManager:
    """Small registry that keeps offline backends out of the rule engine."""

    def __init__(self, backends: list[Offline3DBackend] | None = None) -> None:
        self._backends = {backend.name: backend for backend in (backends or [])}

    @classmethod
    def from_environment(cls) -> "Offline3DManager":
        from .opencap.backend import OpenCapBackend
        from .wham.backend import WhamBackend

        return cls([
            WhamBackend.from_environment(),
            OpenCapBackend.from_environment(),
        ])

    def analyze(
        self,
        backend_name: str,
        video_path: str | Path,
        *,
        output_dir: str | Path | None = None,
        progress: ProgressCallback | None = None,
    ) -> Offline3DResult:
        backend = self._backends.get(backend_name)
        if backend is None:
            return Offline3DResult.unavailable(
                backend_name, f"offline 3D backend {backend_name!r} is not registered"
            )
        if not backend.available:
            reason = getattr(backend, "unavailable_reason", "backend is not configured")
            return Offline3DResult.unavailable(backend_name, str(reason))
        return backend.analyze(video_path, output_dir=output_dir, progress=progress)

    def refine_opencap(
        self,
        video_path: str | Path,
        *,
        wham_result: Offline3DResult,
        alignment: MotionAlignmentResult,
        output_dir: str | Path | None = None,
        progress: ProgressCallback | None = None,
    ) -> Offline3DResult:
        backend = self._backends.get("opencap")
        if backend is None:
            return Offline3DResult.unavailable("opencap", "OpenCap backend is not registered")
        analyze = getattr(backend, "analyze_with_reference", None)
        if not callable(analyze):
            return Offline3DResult.failed("opencap", "OpenCap backend lacks refinement interface")
        return analyze(
            video_path,
            wham_result=wham_result,
            alignment=alignment,
            output_dir=output_dir,
            progress=progress,
        )

    def cancel(self) -> None:
        """Best-effort cancellation for a running external backend."""

        for backend in self._backends.values():
            cancel = getattr(backend, "cancel", None)
            if callable(cancel):
                cancel()
