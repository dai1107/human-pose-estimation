from __future__ import annotations

from datetime import datetime


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def make_session_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

