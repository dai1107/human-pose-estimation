from __future__ import annotations

import logging
import sys
from enum import IntEnum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable


class ExitCode(IntEnum):
    SUCCESS = 0
    RUNTIME_ERROR = 1
    CONFIG_ERROR = 2
    INPUT_ERROR = 3
    BACKEND_ERROR = 4
    OUTPUT_ERROR = 5
    INTERRUPTED = 130


class AppError(RuntimeError):
    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        exit_code: ExitCode,
        hint: str = "",
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.exit_code = exit_code
        self.hint = hint


class InputSourceError(AppError):
    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(
            "SRC001",
            message,
            exit_code=ExitCode.INPUT_ERROR,
            hint=hint,
        )


class BackendInitializationError(AppError):
    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(
            "BCK001",
            message,
            exit_code=ExitCode.BACKEND_ERROR,
            hint=hint,
        )


class OutputWriteError(AppError):
    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(
            "OUT001",
            message,
            exit_code=ExitCode.OUTPUT_ERROR,
            hint=hint,
        )


def configure_logging(
    *,
    app_name: str,
    log_dir: str | Path = "outputs/logs",
    debug: bool = False,
) -> logging.Logger:
    """Configure console output and a bounded persistent application log."""

    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pose")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, "_pose_handler", False):
            logger.removeHandler(handler)
            handler.close()

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if debug else logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    console._pose_handler = True  # type: ignore[attr-defined]

    file_handler = RotatingFileHandler(
        directory / f"{app_name}.log",
        maxBytes=2 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    file_handler._pose_handler = True  # type: ignore[attr-defined]
    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger.debug("logging initialized app=%s path=%s", app_name, directory)
    return logger


def report_error(
    logger: logging.Logger,
    error: AppError,
    *,
    debug: bool = False,
) -> None:
    message = f"[{error.error_code}] {error}"
    if error.hint:
        message += f"；建议：{error.hint}"
    if debug and sys.exc_info()[0] is not None:
        logger.exception(message)
    else:
        logger.error(message)


def safe_cleanup(
    logger: logging.Logger,
    label: str,
    callback: Callable[[], object],
    *,
    debug: bool = False,
) -> Exception | None:
    try:
        callback()
    except Exception as exc:
        if debug:
            logger.exception("[REC001] failed to close/save %s", label)
        else:
            logger.error("[REC001] failed to close/save %s: %s", label, exc)
        return exc
    return None


__all__ = [
    "AppError",
    "BackendInitializationError",
    "ExitCode",
    "InputSourceError",
    "OutputWriteError",
    "configure_logging",
    "report_error",
    "safe_cleanup",
]
