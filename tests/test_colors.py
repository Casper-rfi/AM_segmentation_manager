"""Tests for CSS colour-name resolution."""
from __future__ import annotations

import pytest

from am_segmentation_manager.colors import CSS_COLORS, parse_color

# Every colour name used by the segmentation config shipped with the package.
CONFIG_COLOR_NAMES = [
    "black", "orange", "green", "red", "yellow", "magenta", "purple", "white",
    "cyan", "blue", "pink", "gold", "salmon", "olive", "maroon", "navy", "teal",
    "lime", "aqua", "silver", "brown", "indigo", "violet", "turquoise", "tan",
    "orchid", "gray", "khaki", "coral", "crimson", "plum", "lavender",
    "darkgreen", "darkblue", "darkred", "darkgray", "darkorange",
]


def test_parse_named_colors():
    assert parse_color("black") == (0, 0, 0)
    assert parse_color("white") == (255, 255, 255)
    assert parse_color("orange") == (255, 165, 0)
    assert parse_color("darkgreen") == (0, 100, 0)


def test_named_colors_are_case_and_whitespace_insensitive():
    assert parse_color("DarkGreen") == parse_color("darkgreen")
    assert parse_color("  ORANGE  ") == parse_color("orange")


def test_all_config_color_names_resolve():
    """Every colour used by the bundled config must be resolvable."""
    for name in CONFIG_COLOR_NAMES:
        assert parse_color(name), f"{name} did not resolve"


def test_css_table_is_complete():
    """Guard against an accidentally truncated colour table."""
    assert len(CSS_COLORS) == 148


def test_parse_hex_colors():
    assert parse_color("#ff8c00") == (255, 140, 0)
    assert parse_color("ff8c00") == (255, 140, 0)
    assert parse_color("#F80") == (255, 136, 0)


def test_parse_rgb_sequence():
    assert parse_color([255, 140, 0]) == (255, 140, 0)
    assert parse_color((0, 0, 0)) == (0, 0, 0)


@pytest.mark.parametrize(
    "value",
    ["not-a-colour", "", "   ", "#12345", "#gggggg"],
)
def test_unknown_color_strings_raise(value):
    with pytest.raises(ValueError):
        parse_color(value)


@pytest.mark.parametrize(
    "value",
    [[255, 140], [255, 140, 0, 0], [255, 300, 0], [-1, 0, 0], ["255", "140", "0"]],
)
def test_invalid_rgb_sequences_raise(value):
    with pytest.raises(ValueError):
        parse_color(value)


def test_error_message_names_the_offending_value():
    """A bad config entry must be findable from the error alone."""
    with pytest.raises(ValueError, match="mauve"):
        parse_color("mauve")
