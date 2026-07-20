from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")
_FLOAT_PATTERN = re.compile(
    r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$"
)


class ConfigValidationError(ValueError):
    """A configuration file is readable but does not match its schema."""

    error_code = "CFG001"

    def __init__(
        self,
        message: str,
        *,
        path: str | Path | None = None,
        key: str | None = None,
        line: int | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.key = key
        self.line = line
        location = str(self.path) if self.path is not None else "configuration"
        if line is not None:
            location += f":{line}"
        if key:
            location += f" [{key}]"
        super().__init__(f"{location}: {message}")


def _without_comment(raw_line: str) -> str:
    quote: str | None = None
    escaped = False
    result: list[str] = []
    for character in raw_line:
        if escaped:
            result.append(character)
            escaped = False
            continue
        if character == "\\" and quote == '"':
            result.append(character)
            escaped = True
            continue
        if character in {"'", '"'}:
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
            result.append(character)
            continue
        if character == "#" and quote is None:
            break
        result.append(character)
    return "".join(result).rstrip()


def _parse_scalar(value: str, *, path: Path, line: int) -> Any:
    text = value.strip()
    if not text:
        raise ConfigValidationError("missing value", path=path, line=line)
    if text.lower() in {"null", "~"}:
        return None
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    if _INTEGER_PATTERN.fullmatch(text):
        return int(text)
    if _FLOAT_PATTERN.fullmatch(text):
        return float(text)
    if text.startswith(("[", "{", "&", "*", "!")):
        raise ConfigValidationError(
            "inline collections, aliases, and YAML tags are not supported",
            path=path,
            line=line,
        )
    return text


def _split_key_value(content: str, *, path: Path, line: int) -> tuple[str, str]:
    if ":" not in content:
        raise ConfigValidationError("expected 'key: value'", path=path, line=line)
    key, value = content.split(":", 1)
    key = key.strip()
    if not _KEY_PATTERN.fullmatch(key):
        raise ConfigValidationError(
            f"invalid field name {key!r}",
            path=path,
            line=line,
        )
    return key, value.strip()


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Load the small YAML subset used by this project.

    Supported documents contain top-level scalars plus one indentation level
    of scalar mappings or scalar lists. Rejecting unsupported YAML features is
    intentional: configuration mistakes must fail visibly instead of being
    partially interpreted.
    """

    config_path = Path(path)
    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ConfigValidationError(
            f"cannot read UTF-8 configuration: {exc}",
            path=config_path,
        ) from exc

    parsed: dict[str, Any] = {}
    parent_key: str | None = None
    parent_line: int | None = None
    for line_number, raw_line in enumerate(lines, start=1):
        if "\t" in raw_line[: len(raw_line) - len(raw_line.lstrip())]:
            raise ConfigValidationError(
                "tabs are not allowed for indentation",
                path=config_path,
                line=line_number,
            )
        line = _without_comment(raw_line)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent not in {0, 2}:
            raise ConfigValidationError(
                "only two-space, one-level indentation is supported",
                path=config_path,
                line=line_number,
            )
        content = line.strip()
        if indent == 0:
            if content.startswith("-"):
                raise ConfigValidationError(
                    "top-level lists are not supported",
                    path=config_path,
                    line=line_number,
                )
            key, value = _split_key_value(
                content,
                path=config_path,
                line=line_number,
            )
            if key in parsed:
                raise ConfigValidationError(
                    "duplicate field",
                    path=config_path,
                    key=key,
                    line=line_number,
                )
            if value:
                parsed[key] = _parse_scalar(
                    value,
                    path=config_path,
                    line=line_number,
                )
                parent_key = None
                parent_line = None
            else:
                parsed[key] = None
                parent_key = key
                parent_line = line_number
            continue

        if parent_key is None:
            raise ConfigValidationError(
                "indented value has no parent field",
                path=config_path,
                line=line_number,
            )
        if content.startswith("-"):
            item_text = content[1:].strip()
            if parsed[parent_key] is None:
                parsed[parent_key] = []
            if not isinstance(parsed[parent_key], list):
                raise ConfigValidationError(
                    "cannot mix list items and mapping fields",
                    path=config_path,
                    key=parent_key,
                    line=line_number,
                )
            parsed[parent_key].append(
                _parse_scalar(item_text, path=config_path, line=line_number)
            )
            continue

        child_key, child_value = _split_key_value(
            content,
            path=config_path,
            line=line_number,
        )
        if not child_value:
            raise ConfigValidationError(
                "nested sections deeper than one level are not supported",
                path=config_path,
                key=f"{parent_key}.{child_key}",
                line=line_number,
            )
        if parsed[parent_key] is None:
            parsed[parent_key] = {}
        if not isinstance(parsed[parent_key], dict):
            raise ConfigValidationError(
                "cannot mix mapping fields and list items",
                path=config_path,
                key=parent_key,
                line=line_number,
            )
        nested = parsed[parent_key]
        if child_key in nested:
            raise ConfigValidationError(
                "duplicate field",
                path=config_path,
                key=f"{parent_key}.{child_key}",
                line=line_number,
            )
        nested[child_key] = _parse_scalar(
            child_value,
            path=config_path,
            line=line_number,
        )

    for key, value in parsed.items():
        if value is None:
            raise ConfigValidationError(
                "empty section",
                path=config_path,
                key=key,
                line=parent_line if key == parent_key else None,
            )
    return parsed


def reject_unknown_fields(
    values: Mapping[str, Any],
    allowed: set[str] | frozenset[str],
    *,
    path: str | Path | None,
    prefix: str = "",
) -> None:
    unknown = sorted(str(key) for key in values if key not in allowed)
    if unknown:
        field = f"{prefix}{unknown[0]}"
        raise ConfigValidationError(
            f"unknown field; allowed fields: {', '.join(sorted(allowed))}",
            path=path,
            key=field,
        )


__all__ = [
    "ConfigValidationError",
    "load_simple_yaml",
    "reject_unknown_fields",
]
