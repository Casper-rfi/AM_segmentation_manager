"""Colour-name resolution for segmentation class palettes.

Segmentation configs name their class colours using CSS/X11 colour names (for
example ``"orange"`` or ``"darkgreen"``). This module turns those names -- and
hex strings, and explicit RGB triples -- into plain ``(r, g, b)`` tuples.

The CSS colour table is embedded here as static data rather than obtained from
matplotlib or Qt. It is a fixed, standardised list that will never change, so
carrying ~150 lines of data avoids a heavyweight dependency in the one place
the package needs to draw something, and keeps colour handling testable without
a GUI toolkit installed.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Tuple

RGB = Tuple[int, int, int]

# The CSS Color Module Level 4 named colours (also known as the X11 or SVG
# colour names), mapped to their canonical hex values.
CSS_COLORS: dict[str, str] = {
    "aliceblue":           "#f0f8ff",
    "antiquewhite":        "#faebd7",
    "aqua":                "#00ffff",
    "aquamarine":          "#7fffd4",
    "azure":               "#f0ffff",
    "beige":               "#f5f5dc",
    "bisque":              "#ffe4c4",
    "black":               "#000000",
    "blanchedalmond":      "#ffebcd",
    "blue":                "#0000ff",
    "blueviolet":          "#8a2be2",
    "brown":               "#a52a2a",
    "burlywood":           "#deb887",
    "cadetblue":           "#5f9ea0",
    "chartreuse":          "#7fff00",
    "chocolate":           "#d2691e",
    "coral":               "#ff7f50",
    "cornflowerblue":      "#6495ed",
    "cornsilk":            "#fff8dc",
    "crimson":             "#dc143c",
    "cyan":                "#00ffff",
    "darkblue":            "#00008b",
    "darkcyan":            "#008b8b",
    "darkgoldenrod":       "#b8860b",
    "darkgray":            "#a9a9a9",
    "darkgreen":           "#006400",
    "darkgrey":            "#a9a9a9",
    "darkkhaki":           "#bdb76b",
    "darkmagenta":         "#8b008b",
    "darkolivegreen":      "#556b2f",
    "darkorange":          "#ff8c00",
    "darkorchid":          "#9932cc",
    "darkred":             "#8b0000",
    "darksalmon":          "#e9967a",
    "darkseagreen":        "#8fbc8f",
    "darkslateblue":       "#483d8b",
    "darkslategray":       "#2f4f4f",
    "darkslategrey":       "#2f4f4f",
    "darkturquoise":       "#00ced1",
    "darkviolet":          "#9400d3",
    "deeppink":            "#ff1493",
    "deepskyblue":         "#00bfff",
    "dimgray":             "#696969",
    "dimgrey":             "#696969",
    "dodgerblue":          "#1e90ff",
    "firebrick":           "#b22222",
    "floralwhite":         "#fffaf0",
    "forestgreen":         "#228b22",
    "fuchsia":             "#ff00ff",
    "gainsboro":           "#dcdcdc",
    "ghostwhite":          "#f8f8ff",
    "gold":                "#ffd700",
    "goldenrod":           "#daa520",
    "gray":                "#808080",
    "green":               "#008000",
    "greenyellow":         "#adff2f",
    "grey":                "#808080",
    "honeydew":            "#f0fff0",
    "hotpink":             "#ff69b4",
    "indianred":           "#cd5c5c",
    "indigo":              "#4b0082",
    "ivory":               "#fffff0",
    "khaki":               "#f0e68c",
    "lavender":            "#e6e6fa",
    "lavenderblush":       "#fff0f5",
    "lawngreen":           "#7cfc00",
    "lemonchiffon":        "#fffacd",
    "lightblue":           "#add8e6",
    "lightcoral":          "#f08080",
    "lightcyan":           "#e0ffff",
    "lightgoldenrodyellow": "#fafad2",
    "lightgray":           "#d3d3d3",
    "lightgreen":          "#90ee90",
    "lightgrey":           "#d3d3d3",
    "lightpink":           "#ffb6c1",
    "lightsalmon":         "#ffa07a",
    "lightseagreen":       "#20b2aa",
    "lightskyblue":        "#87cefa",
    "lightslategray":      "#778899",
    "lightslategrey":      "#778899",
    "lightsteelblue":      "#b0c4de",
    "lightyellow":         "#ffffe0",
    "lime":                "#00ff00",
    "limegreen":           "#32cd32",
    "linen":               "#faf0e6",
    "magenta":             "#ff00ff",
    "maroon":              "#800000",
    "mediumaquamarine":    "#66cdaa",
    "mediumblue":          "#0000cd",
    "mediumorchid":        "#ba55d3",
    "mediumpurple":        "#9370db",
    "mediumseagreen":      "#3cb371",
    "mediumslateblue":     "#7b68ee",
    "mediumspringgreen":   "#00fa9a",
    "mediumturquoise":     "#48d1cc",
    "mediumvioletred":     "#c71585",
    "midnightblue":        "#191970",
    "mintcream":           "#f5fffa",
    "mistyrose":           "#ffe4e1",
    "moccasin":            "#ffe4b5",
    "navajowhite":         "#ffdead",
    "navy":                "#000080",
    "oldlace":             "#fdf5e6",
    "olive":               "#808000",
    "olivedrab":           "#6b8e23",
    "orange":              "#ffa500",
    "orangered":           "#ff4500",
    "orchid":              "#da70d6",
    "palegoldenrod":       "#eee8aa",
    "palegreen":           "#98fb98",
    "paleturquoise":       "#afeeee",
    "palevioletred":       "#db7093",
    "papayawhip":          "#ffefd5",
    "peachpuff":           "#ffdab9",
    "peru":                "#cd853f",
    "pink":                "#ffc0cb",
    "plum":                "#dda0dd",
    "powderblue":          "#b0e0e6",
    "purple":              "#800080",
    "rebeccapurple":       "#663399",
    "red":                 "#ff0000",
    "rosybrown":           "#bc8f8f",
    "royalblue":           "#4169e1",
    "saddlebrown":         "#8b4513",
    "salmon":              "#fa8072",
    "sandybrown":          "#f4a460",
    "seagreen":            "#2e8b57",
    "seashell":            "#fff5ee",
    "sienna":              "#a0522d",
    "silver":              "#c0c0c0",
    "skyblue":             "#87ceeb",
    "slateblue":           "#6a5acd",
    "slategray":           "#708090",
    "slategrey":           "#708090",
    "snow":                "#fffafa",
    "springgreen":         "#00ff7f",
    "steelblue":           "#4682b4",
    "tan":                 "#d2b48c",
    "teal":                "#008080",
    "thistle":             "#d8bfd8",
    "tomato":              "#ff6347",
    "turquoise":           "#40e0d0",
    "violet":              "#ee82ee",
    "wheat":               "#f5deb3",
    "white":               "#ffffff",
    "whitesmoke":          "#f5f5f5",
    "yellow":              "#ffff00",
    "yellowgreen":         "#9acd32",
}


def parse_color(value: str | Sequence[int]) -> RGB:
    """Return ``(r, g, b)`` for a colour written in any supported form.

    Accepted forms:

    * a CSS/X11 colour name, case-insensitive (``"orange"``, ``"DarkGreen"``)
    * a hex string, with or without the leading ``#`` (``"#ff8c00"``, ``"f80"``)
    * a sequence of three integers in 0-255 (``[255, 140, 0]``), which is how
      YAML most naturally expresses an explicit colour

    Args:
        value: The colour to resolve.

    Returns:
        The colour as a three-tuple of integers in the range 0-255.

    Raises:
        ValueError: If *value* is not a recognised colour name, is a malformed
            hex string, or is a sequence with the wrong length or out-of-range
            components. The message names the offending value so a bad entry in
            a config file can be found quickly.
    """
    if isinstance(value, str):
        return _parse_color_string(value)
    if isinstance(value, Sequence):
        return _parse_color_sequence(value)
    raise ValueError(f"Unsupported colour value: {value!r}")


def _parse_color_string(value: str) -> RGB:
    """Resolve a colour name or hex string to an RGB tuple."""
    text = value.strip().lower()
    if not text:
        raise ValueError("Colour value is empty")

    hex_text = CSS_COLORS.get(text, text)
    hex_digits = hex_text[1:] if hex_text.startswith("#") else hex_text

    if len(hex_digits) == 3:
        hex_digits = "".join(digit * 2 for digit in hex_digits)
    if len(hex_digits) != 6 or not all(c in "0123456789abcdef" for c in hex_digits):
        raise ValueError(
            f"Unknown colour {value!r}: expected a CSS colour name "
            f"(for example 'orange'), a hex string, or an [r, g, b] triple"
        )

    return (
        int(hex_digits[0:2], 16),
        int(hex_digits[2:4], 16),
        int(hex_digits[4:6], 16),
    )


def _parse_color_sequence(value: Sequence[int]) -> RGB:
    """Validate an explicit ``[r, g, b]`` triple."""
    components = list(value)
    if len(components) != 3:
        raise ValueError(f"Colour {value!r} must have exactly three components")
    if not all(isinstance(c, int) and 0 <= c <= 255 for c in components):
        raise ValueError(f"Colour {value!r} must contain integers in the range 0-255")
    return (components[0], components[1], components[2])
