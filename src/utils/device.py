from __future__ import annotations


def resolve_torch_device(preferred: str = "auto") -> str:
    normalized = (preferred or "auto").strip().lower()
    if normalized not in {"", "auto"}:
        return preferred.strip()
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "0" if torch.cuda.is_available() else "cpu"
