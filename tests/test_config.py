"""Tests for segmentation config resolution and loading."""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.config import (
    CONFIG_ENV_VAR,
    CONFIG_FILENAME,
    UNKNOWN_CLASS_COLOR,
    SegmentationConfig,
    default_config_path,
    load_segmentation_config,
    resolve_config_path,
)

VALID_CONFIG = """
CLASS_COLORS:
  0: "black"
  1: "orange"
  2: "green"
CLASS_LABELS:
  0: "background"
  1: "GIS"
  2: "lamella"
"""


def write_config(directory: Path, content: str = VALID_CONFIG, name: str = CONFIG_FILENAME) -> Path:
    """Write a config file into *directory* and return its path."""
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def clear_config_env(monkeypatch):
    """Keep a developer's own AM_SEG_CONFIG out of the tests."""
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)


# ---------------------------------------------------------------------------
# Resolution order
# ---------------------------------------------------------------------------

def test_explicit_path_wins_over_everything(tmp_path, monkeypatch):
    explicit = write_config(tmp_path, name="custom.yaml")
    scan_root = tmp_path / "scan"
    scan_root.mkdir()
    write_config(scan_root)
    monkeypatch.setenv(CONFIG_ENV_VAR, str(write_config(tmp_path, name="env.yaml")))

    assert resolve_config_path(explicit_path=explicit, start_path=scan_root) == explicit


def test_env_var_used_when_no_explicit_path(tmp_path, monkeypatch):
    env_config = write_config(tmp_path, name="env.yaml")
    scan_root = tmp_path / "scan"
    scan_root.mkdir()
    write_config(scan_root)
    monkeypatch.setenv(CONFIG_ENV_VAR, str(env_config))

    assert resolve_config_path(start_path=scan_root) == env_config


def test_scan_root_used_when_no_explicit_path_or_env(tmp_path):
    scan_root = tmp_path / "scan"
    scan_root.mkdir()
    expected = write_config(scan_root)

    assert resolve_config_path(start_path=scan_root) == expected


def test_falls_back_to_packaged_default(tmp_path):
    empty_scan_root = tmp_path / "scan"
    empty_scan_root.mkdir()

    assert resolve_config_path(start_path=empty_scan_root) == default_config_path()
    assert resolve_config_path() == default_config_path()


def test_missing_explicit_path_raises(tmp_path):
    """A config asked for by name is never silently replaced by a fallback."""
    with pytest.raises(FileNotFoundError):
        resolve_config_path(explicit_path=tmp_path / "does_not_exist.yaml")


def test_missing_env_var_path_raises(tmp_path, monkeypatch):
    monkeypatch.setenv(CONFIG_ENV_VAR, str(tmp_path / "does_not_exist.yaml"))
    with pytest.raises(FileNotFoundError):
        resolve_config_path()


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

def test_load_parses_colors_and_labels(tmp_path):
    config = load_segmentation_config(explicit_path=write_config(tmp_path))

    assert config.class_colors[1] == (255, 165, 0)
    assert config.class_labels[1] == "GIS"
    assert config.source == tmp_path / CONFIG_FILENAME


def test_load_accepts_string_class_indices(tmp_path):
    path = write_config(tmp_path, """
CLASS_COLORS:
  "0": "black"
  "1": "orange"
CLASS_LABELS:
  "0": "background"
  "1": "GIS"
""")
    config = load_segmentation_config(explicit_path=path)

    assert config.class_labels[1] == "GIS"


@pytest.mark.parametrize("section", ["CLASS_COLORS", "CLASS_LABELS"])
def test_missing_section_raises(tmp_path, section):
    # Rename the section so the config parses as YAML but lacks the section.
    path = write_config(tmp_path, VALID_CONFIG.replace(section, "UNUSED_SECTION"))

    with pytest.raises(ValueError, match=section):
        load_segmentation_config(explicit_path=path)


def test_unresolvable_color_names_the_class(tmp_path):
    path = write_config(tmp_path, """
CLASS_COLORS:
  0: "black"
  1: "mauve"
CLASS_LABELS:
  0: "background"
  1: "GIS"
""")
    with pytest.raises(ValueError, match="mauve"):
        load_segmentation_config(explicit_path=path)


def test_labelled_class_without_a_color_raises(tmp_path):
    path = write_config(tmp_path, """
CLASS_COLORS:
  0: "black"
CLASS_LABELS:
  0: "background"
  1: "GIS"
""")
    with pytest.raises(ValueError, match="CLASS_COLORS"):
        load_segmentation_config(explicit_path=path)


def test_non_mapping_section_raises(tmp_path):
    path = write_config(tmp_path, """
CLASS_COLORS:
  - "black"
CLASS_LABELS:
  0: "background"
""")
    with pytest.raises(ValueError, match="mapping"):
        load_segmentation_config(explicit_path=path)


# ---------------------------------------------------------------------------
# Lookup behaviour
# ---------------------------------------------------------------------------

def test_unknown_class_falls_back_rather_than_raising():
    """A label value the config does not define must stay displayable."""
    config = SegmentationConfig(class_colors={0: (0, 0, 0)}, class_labels={0: "background"})

    assert config.rgb(99) == UNKNOWN_CLASS_COLOR
    assert config.name(99) == "class 99"


def test_foreground_classes_excludes_background(tmp_path):
    config = load_segmentation_config(explicit_path=write_config(tmp_path))

    assert config.classes == [0, 1, 2]
    assert config.foreground_classes == [1, 2]


# ---------------------------------------------------------------------------
# The configuration shipped with the package
# ---------------------------------------------------------------------------

def test_packaged_default_config_is_valid():
    """The bundled config must always load; it is the last-resort fallback."""
    config = load_segmentation_config()

    assert config.source == default_config_path()
    assert config.class_labels[0] == "background"
    assert config.name(2) == "lamella"
    assert config.foreground_classes == [1, 2, 3, 4]
    # Every named class resolves to a real colour, not the fallback.
    assert all(config.rgb(index) != UNKNOWN_CLASS_COLOR for index in config.classes)
