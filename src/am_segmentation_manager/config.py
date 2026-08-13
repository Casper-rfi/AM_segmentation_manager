"""Segmentation class configuration (class names and display colours).

Segmentation classes are defined in a YAML file (``segmentation_config_AM.yaml``)
holding two mappings from class index to value::

    CLASS_COLORS:
      0: "black"
      1: "orange"
    CLASS_LABELS:
      0: "background"
      1: "GIS"

This module resolves *which* config file to use, loads it into a
:class:`SegmentationConfig`, and validates it. Colour names are resolved to RGB
tuples at load time (see :mod:`am_segmentation_manager.colors`) so a typo in a
colour name is reported once, when the config is read, rather than surfacing
later as a rendering failure.

The config describes how labels are *displayed and named*; it is consumed by
label overlay rendering and by any UI that shows a class legend. It deliberately
holds no filesystem or workflow state.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .colors import RGB, parse_color
from .io.yaml import read_yaml

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "segmentation_config_AM.yaml"
CONFIG_ENV_VAR = "AM_SEG_CONFIG"

#: Colour used when a label contains a class the config does not define. A
#: distinctive colour is preferable to failing: an unexpected class should be
#: visible and obviously unconfigured, not a reason to stop displaying data.
UNKNOWN_CLASS_COLOR: RGB = (255, 0, 255)

#: Class index treated as background. Background is never drawn in overlays.
BACKGROUND_CLASS = 0


@dataclass(frozen=True)
class SegmentationConfig:
    """Display names and colours for the segmentation classes.

    Attributes:
        class_colors: Class index to ``(r, g, b)``, resolved at load time.
        class_labels: Class index to human-readable class name.
        source: Path the configuration was loaded from, for display and
            debugging. ``None`` when constructed directly in code or tests.
    """

    class_colors: Mapping[int, RGB]
    class_labels: Mapping[int, str]
    source: Path | None = None

    def rgb(self, class_index: int) -> RGB:
        """Return the display colour for *class_index*.

        Falls back to :data:`UNKNOWN_CLASS_COLOR` (and logs a warning) for a
        class the config does not define, so that unexpected label values are
        rendered visibly rather than aborting a display operation.
        """
        color = self.class_colors.get(class_index)
        if color is None:
            logger.warning(
                "Class %d has no colour in %s; using the fallback colour",
                class_index,
                self.source or "the segmentation config",
            )
            return UNKNOWN_CLASS_COLOR
        return color

    def name(self, class_index: int) -> str:
        """Return the display name for *class_index*.

        Unnamed classes fall back to ``"class <n>"``. Colour palettes normally
        define many more classes than any one experiment names, so an unnamed
        class is expected rather than an error.
        """
        return self.class_labels.get(class_index, f"class {class_index}")

    @property
    def classes(self) -> list[int]:
        """Return the named class indices, in ascending order."""
        return sorted(self.class_labels)

    @property
    def foreground_classes(self) -> list[int]:
        """Return the named class indices excluding background.

        These are the classes that appear in a label overlay and its legend.
        """
        return [index for index in self.classes if index != BACKGROUND_CLASS]


def resolve_config_path(
    explicit_path: Path | None = None,
    start_path: Path | None = None,
) -> Path:
    """Return the path of the segmentation config file to use.

    Sources are tried in order of how specifically the user asked for them:

    1. *explicit_path*, when given (for example a ``--config`` argument)
    2. the :data:`CONFIG_ENV_VAR` environment variable
    3. ``segmentation_config_AM.yaml`` in *start_path*, the directory being
       scanned, so a dataset collection can carry its own class definitions
    4. the default configuration bundled with the package

    Args:
        explicit_path: A config path supplied directly by the user.
        start_path: The scan root, searched for a config file of the standard
            name.

    Returns:
        The path of the first configuration file found.

    Raises:
        FileNotFoundError: If *explicit_path* or the environment variable names
            a file that does not exist. A config requested by name is never
            silently replaced by a fallback.
    """
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Segmentation config not found: {path}")
        return path

    env_value = os.environ.get(CONFIG_ENV_VAR)
    if env_value:
        path = Path(env_value)
        if not path.exists():
            raise FileNotFoundError(
                f"Segmentation config from ${CONFIG_ENV_VAR} not found: {path}"
            )
        return path

    if start_path is not None:
        path = Path(start_path) / CONFIG_FILENAME
        if path.exists():
            return path

    return default_config_path()


def default_config_path() -> Path:
    """Return the path of the configuration bundled with the package.

    The file is located relative to this module rather than through
    ``importlib.resources`` because the rest of the package works in terms of
    ``Path`` objects, and this package is always installed to a real directory
    (it is not designed to be imported from a zip archive).
    """
    return Path(__file__).parent / "data" / CONFIG_FILENAME


def load_segmentation_config(
    explicit_path: Path | None = None,
    start_path: Path | None = None,
) -> SegmentationConfig:
    """Load and validate the segmentation configuration.

    The file is located with :func:`resolve_config_path`, then parsed and
    checked. Colour values are resolved to RGB tuples immediately so that
    configuration errors are reported here rather than during rendering.

    Args:
        explicit_path: A config path supplied directly by the user.
        start_path: The scan root, searched for a config file.

    Returns:
        The loaded :class:`SegmentationConfig`.

    Raises:
        FileNotFoundError: If a config was explicitly requested but is missing.
        ValueError: If the file is missing a required section, has non-integer
            class indices, contains an unresolvable colour, or names a class
            that has no colour.
    """
    path = resolve_config_path(explicit_path=explicit_path, start_path=start_path)
    data = read_yaml(path)

    raw_colors = _require_section(data, "CLASS_COLORS", path)
    raw_labels = _require_section(data, "CLASS_LABELS", path)

    class_colors = {
        _class_index(key, "CLASS_COLORS", path): _class_color(key, value, path)
        for key, value in raw_colors.items()
    }
    class_labels = {
        _class_index(key, "CLASS_LABELS", path): str(value)
        for key, value in raw_labels.items()
    }

    unpainted = sorted(set(class_labels) - set(class_colors))
    if unpainted:
        raise ValueError(
            f"{path}: CLASS_LABELS defines class(es) {unpainted} with no entry "
            f"in CLASS_COLORS"
        )

    logger.debug("Loaded segmentation config from %s: %d classes", path, len(class_labels))
    return SegmentationConfig(class_colors=class_colors, class_labels=class_labels, source=path)


def _require_section(data: Mapping[str, Any], section: str, path: Path) -> Mapping[Any, Any]:
    """Return *section* from a parsed config, or raise a helpful error."""
    value = data.get(section)
    if value is None:
        raise ValueError(f"{path}: missing required section {section}")
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: section {section} must be a mapping of class index to value")
    return value


def _class_index(key: Any, section: str, path: Path) -> int:
    """Coerce a config key to an integer class index."""
    try:
        return int(key)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {section} key {key!r} is not an integer class index") from exc


def _class_color(key: Any, value: Any, path: Path) -> RGB:
    """Resolve a config colour value, reporting which class failed."""
    try:
        return parse_color(value)
    except ValueError as exc:
        raise ValueError(f"{path}: CLASS_COLORS entry for class {key!r}: {exc}") from exc
