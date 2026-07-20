from __future__ import annotations

from pathlib import Path

import pytest

from hyrox.config import load_lunge_config
from hyrox.registry import create_action_analyzer
from src.configuration import ConfigValidationError, load_simple_yaml
from src.reference.features import load_feature_config
from src.reference.quality import load_quality_rules


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("field", "value", "expected_message"),
    [
        ("visiblity_min", "0.5", "unknown field"),
        ("visibility_min", "high", "expected number"),
        ("visibility_min", "1.2", "between 0 and 1"),
        ("stable_frames", "0", "at least 1"),
        ("action_name", "rowing", "expected 'lunge'"),
    ],
)
def test_action_config_rejects_unknown_type_range_and_action_errors(
    tmp_path: Path,
    field: str,
    value: str,
    expected_message: str,
) -> None:
    path = _write(tmp_path / "invalid.yaml", f"{field}: {value}\n")

    with pytest.raises(ConfigValidationError, match=expected_message):
        load_lunge_config(path)


def test_action_config_rejects_unknown_nested_feedback_field(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "invalid.yaml",
        "feedback_limits:\n  max_message: 2\n",
    )

    with pytest.raises(ConfigValidationError, match="feedback_limits.max_message"):
        load_lunge_config(path)


def test_simple_yaml_rejects_malformed_and_duplicate_lines(tmp_path: Path) -> None:
    malformed = _write(tmp_path / "malformed.yaml", "visibility_min 0.5\n")
    duplicate = _write(
        tmp_path / "duplicate.yaml",
        "visibility_min: 0.5\nvisibility_min: 0.6\n",
    )

    with pytest.raises(ConfigValidationError, match="expected 'key: value'"):
        load_simple_yaml(malformed)
    with pytest.raises(ConfigValidationError, match="duplicate field"):
        load_simple_yaml(duplicate)


def test_registry_validates_in_memory_overrides() -> None:
    with pytest.raises(ConfigValidationError, match="unknown field"):
        create_action_analyzer("lunge", {"stable_frame": 1})


def test_reference_configs_reject_unknown_fields(tmp_path: Path) -> None:
    feature_path = _write(
        tmp_path / "features.yaml",
        "name: custom\nanglez:\n  - left_knee_angle\n",
    )
    quality_path = _write(
        tmp_path / "quality.yaml",
        "minimum_pose_valid_rato: 0.5\n",
    )

    with pytest.raises(ConfigValidationError, match="anglez"):
        load_feature_config(feature_path)
    with pytest.raises(ConfigValidationError, match="minimum_pose_valid_rato"):
        load_quality_rules(quality_path)
