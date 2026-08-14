"""Segmentation class configuration (class names and display colours).

Segmentation classes are defined in a YAML file (``segmentation_config_AM.yaml``)
holding two mappings from class index to value::

    CLASS_COLORS:
      0: "black"
      1: "orange"
    CLASS_LABELS:
      0: "background"
      1: "GIS"

An optional third section configures label quantification::

    MEASUREMENT:
      measured_classes: [1, 2, 3]
      crack_class: 3
      min_crack_area_um2: 0.1

This module resolves *which* config file to use, loads it into a
:class:`SegmentationConfig`, and validates it. Colour names are resolved to RGB
tuples at load time (see :mod:`am_segmentation_manager.colors`) so a typo in a
colour name is reported once, when the config is read, rather than surfacing
later as a rendering failure.

The config describes how labels are *displayed, named and measured*; it is
consumed by label overlay rendering, by any UI that shows a class legend, and by
:mod:`am_segmentation_manager.operations.label_measure`. Measurement settings
live here because they are properties of the segmentation classes: a project
shipping its own class definitions has to be able to say which of its classes is
a crack. It deliberately holds no filesystem or workflow state.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

#: Smallest connected component of the crack class counted as a crack, in um^2.
#: Used when the config file names no threshold. A starting value rather than a
#: physical constant -- see the MEASUREMENT section of the bundled config.
DEFAULT_MIN_CRACK_AREA_UM2 = 0.1

#: Class names recognised as the crack class when ``MEASUREMENT.crack_class`` is
#: absent. Matching a name is evidence; assuming an index is a guess, and a
#: config that renamed its classes would otherwise have its lamella measured as
#: cracks without anything saying so.
CRACK_CLASS_NAMES = frozenset({"crack/hole", "crack", "hole"})


@dataclass(frozen=True)
class SegmentationConfig:
    """Display names and colours for the segmentation classes.

    Attributes:
        class_colors: Class index to ``(r, g, b)``, resolved at load time.
        class_labels: Class index to human-readable class name.
        source: Path the configuration was loaded from, for display and
            debugging. ``None`` when constructed directly in code or tests.
        measured_classes: Class indices label quantification reports areas for.
            ``None`` when the config names none, in which case
            :attr:`classes_to_measure` falls back to every foreground class.
        crack_class: Class index whose connected components are counted as
            cracks. ``None`` when the config neither names one nor labels a
            class with a recognised crack name, which makes crack counting
            unavailable rather than applied to a guessed class.
        min_crack_area_um2: Smallest component counted as a crack, in um^2.
    """

    class_colors: Mapping[int, RGB]
    class_labels: Mapping[int, str]
    source: Path | None = None
    measured_classes: tuple[int, ...] | None = None
    crack_class: int | None = None
    min_crack_area_um2: float = DEFAULT_MIN_CRACK_AREA_UM2

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

    @property
    def classes_to_measure(self) -> list[int]:
        """Return the class indices label quantification reports areas for.

        Falls back to every foreground class when the config names none. That
        is a wider answer than the configured one, never a narrower one: a
        class the config forgot to list is reported rather than silently
        dropped from a summary.
        """
        if self.measured_classes is None:
            return self.foreground_classes
        return list(self.measured_classes)


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
            class indices, contains an unresolvable colour, names a class
            that has no colour, or has an invalid MEASUREMENT section.
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

    measured, crack_class, min_crack_area = _measurement_settings(data, class_labels, path)

    logger.debug("Loaded segmentation config from %s: %d classes", path, len(class_labels))
    return SegmentationConfig(
        class_colors=class_colors,
        class_labels=class_labels,
        source=path,
        measured_classes=measured,
        crack_class=crack_class,
        min_crack_area_um2=min_crack_area,
    )


def _measurement_settings(
    data: Mapping[str, Any], class_labels: Mapping[int, str], path: Path
) -> tuple[tuple[int, ...] | None, int | None, float]:
    """Resolve the optional MEASUREMENT section.

    The section is optional because measurement was added after the config
    format, and an existing config file must keep loading. Every value it can
    omit either has a defined fallback or resolves to ``None``; nothing here
    invents a class index.
    """
    section = data.get("MEASUREMENT")
    if section is None:
        return None, _crack_class_by_name(class_labels), DEFAULT_MIN_CRACK_AREA_UM2
    if not isinstance(section, Mapping):
        raise ValueError(f"{path}: section MEASUREMENT must be a mapping")

    measured: tuple[int, ...] | None = None
    raw_measured = section.get("measured_classes")
    if raw_measured is not None:
        if isinstance(raw_measured, (str, bytes)) or not isinstance(raw_measured, Iterable):
            raise ValueError(
                f"{path}: MEASUREMENT.measured_classes must be a list of class indices"
            )
        measured = tuple(
            _known_class(
                _class_index(value, "MEASUREMENT.measured_classes", path),
                class_labels,
                "MEASUREMENT.measured_classes",
                path,
            )
            for value in raw_measured
        )

    crack_class = section.get("crack_class")
    if crack_class is None:
        crack_class = _crack_class_by_name(class_labels)
    else:
        crack_class = _known_class(
            _class_index(crack_class, "MEASUREMENT.crack_class", path),
            class_labels,
            "MEASUREMENT.crack_class",
            path,
        )

    min_crack_area = section.get("min_crack_area_um2", DEFAULT_MIN_CRACK_AREA_UM2)
    if isinstance(min_crack_area, bool) or not isinstance(min_crack_area, (int, float)):
        raise ValueError(
            f"{path}: MEASUREMENT.min_crack_area_um2 must be a number, "
            f"got {min_crack_area!r}"
        )
    if min_crack_area < 0:
        raise ValueError(
            f"{path}: MEASUREMENT.min_crack_area_um2 must not be negative, "
            f"got {min_crack_area!r}"
        )

    return measured, crack_class, float(min_crack_area)


def _crack_class_by_name(class_labels: Mapping[int, str]) -> int | None:
    """Return the class index named as a crack, or None if no name matches."""
    for index in sorted(class_labels):
        if class_labels[index].strip().lower() in CRACK_CLASS_NAMES:
            return index
    return None


def _known_class(
    index: int, class_labels: Mapping[int, str], setting: str, path: Path
) -> int:
    """Return *index*, or raise if CLASS_LABELS does not define it.

    A measurement setting pointing at an undefined class is a typo whose
    symptom would otherwise be a column of zeroes.
    """
    if index not in class_labels:
        raise ValueError(
            f"{path}: {setting} names class {index}, which CLASS_LABELS does not define"
        )
    return index


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
