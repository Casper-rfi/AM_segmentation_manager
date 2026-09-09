"""What a control opens on, and where those answers are kept.

**A default decides the value a control starts at. It never decides what an
operation does.** The command line and every write path keep the packaged
constants, so nothing set here can change what a run produces -- only what
somebody is shown before they change it. That is what makes these safe to keep
on one machine: they travel with the person, not with the data.

Qt-free, like the controllers. The table and the resolving live here and the
store is passed in, so a preference can be read in a test with a dictionary
and without a display. **A preference is a row in :data:`PREFERENCES`**, as a
descriptive field is a row in `metadata.FIELDS`: adding one is a row here and
a line where the control is built, rather than an edit in five modules.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..display import DEFAULT_ALPHA
from ..layout import SERIES, SERIES_LABELS
from ..operations.csv import DEFAULT_TRAIN_RATIO
from ..operations.label_measure import (
    DEFAULT_SPECK_PIXELS,
    MAX_SPECK_PIXELS,
    MIN_SPECK_PIXELS,
)
from .export_controller import DEFAULT_SERIES

logger = logging.getLogger(__name__)

#: Where these sit in the window's own settings, under one prefix so they can
#: be told from the things the window merely remembers -- the scan root, the
#: model, the geometry. Those record what happened; these are asked for.
SETTINGS_PREFIX = "defaults"

#: The brush's opening width, as a radius in image pixels. Here rather than
#: beside the canvas that paints, which is a Qt module this one may not
#: import; the canvas takes it from here instead. Eight is a stroke wide
#: enough to cover a boundary in one pass at fit, and narrow enough not to
#: bury the structure being corrected.
DEFAULT_BRUSH_RADIUS = 8

#: What the training split's shuffle is seeded with, and whether it is seeded
#: at all. A seed is on by default because an export nobody can reproduce is
#: one nobody can describe in a methods section.
DEFAULT_SEED = 42
DEFAULT_SEEDED = True

#: Whether an export keeps one image per label rather than every imaging
#: condition of it. Off: the wider set is what an export has always meant, and
#: narrowing it silently would disagree with `create-training-csv`.
DEFAULT_ONE_PER_LABEL = False

#: The two boxes the dialog groups these under. A group is the tab a
#: preference is felt on, not the module it reaches.
PROOFREADING = "Proofreading"
EXPORT = "Export"


@dataclass(frozen=True)
class Preference:
    """One value a control opens on.

    Attributes:
        key: Its name under :data:`SETTINGS_PREFIX`.
        group: Which box of the settings dialog it appears in.
        label: Its caption there.
        fallback: What it is worth with nothing stored -- the packaged
            constant, so the dialog and a fresh install agree.
        hint: The tooltip, which says what the value does rather than what it
            is called.
        low: Smallest accepted value, for a number.
        high: Largest accepted value, for a number.
        suffix: Shown after a number, where it has a unit.
    """

    key: str
    group: str
    label: str
    fallback: int | bool
    hint: str
    low: int | None = None
    high: int | None = None
    suffix: str = ""

    @property
    def is_flag(self) -> bool:
        """Return whether this is a switch rather than a number."""
        return isinstance(self.fallback, bool)


PREFERENCES: tuple[Preference, ...] = (
    Preference(
        key="speck_pixels",
        group=PROOFREADING,
        label="Find specks at most",
        fallback=DEFAULT_SPECK_PIXELS,
        low=MIN_SPECK_PIXELS,
        high=MAX_SPECK_PIXELS,
        suffix=" px",
        hint=(
            "The largest component the speck filter rings, on the Curate and "
            "Label tabs. Both open here; either can be moved for one session."
        ),
    ),
    Preference(
        key="brush_radius",
        group=PROOFREADING,
        label="Brush radius",
        fallback=DEFAULT_BRUSH_RADIUS,
        low=1,
        high=200,
        suffix=" px",
        hint=(
            "How wide the proofreading brush opens. The slider offers every "
            "small size and coarser steps above them, so this lands on the "
            "nearest it has."
        ),
    ),
    Preference(
        key="label_opacity",
        group=PROOFREADING,
        label="Label opacity",
        fallback=int(round(DEFAULT_ALPHA * 100)),
        low=0,
        high=100,
        suffix=" %",
        hint=(
            "How strongly a label is drawn over the image, on both tabs that "
            "draw one. Changes nothing on disk and reaches no export."
        ),
    ),
    *(
        Preference(
            key=f"series/{name}",
            group=EXPORT,
            label=SERIES_LABELS[name],
            fallback=name in DEFAULT_SERIES,
            hint=(
                f"Whether the Export tab opens counting {SERIES_LABELS[name]} "
                "images. Every filter but this one opens on everything."
            ),
        )
        for name in SERIES
    ),
    Preference(
        key="one_per_label",
        group=EXPORT,
        label="One image per label",
        fallback=DEFAULT_ONE_PER_LABEL,
        hint=(
            "Whether the Export tab opens keeping one image per label rather "
            "than every imaging condition sharing it."
        ),
    ),
    Preference(
        key="train_share",
        group=EXPORT,
        label="Training share",
        fallback=int(round(DEFAULT_TRAIN_RATIO * 100)),
        low=0,
        high=100,
        suffix=" %",
        hint=(
            "Share of the exportable pairs written to the training file, the "
            "rest to validation. Matches create-training-csv --train-ratio."
        ),
    ),
    Preference(
        key="seeded",
        group=EXPORT,
        label="Seed the split",
        fallback=DEFAULT_SEEDED,
        hint=(
            "Whether the shuffle is reproducible, so the same selection and "
            "seed always give the same two files."
        ),
    ),
    Preference(
        key="seed",
        group=EXPORT,
        label="Seed",
        fallback=DEFAULT_SEED,
        low=0,
        high=999_999,
        hint="The seed that shuffle uses, where it is seeded.",
    ),
)

#: The preferences by key, for a caller that wants one by name.
BY_KEY: Mapping[str, Preference] = {entry.key: entry for entry in PREFERENCES}


def groups() -> list[tuple[str, list[Preference]]]:
    """Return the preferences under their group, in the order declared."""
    ordered: dict[str, list[Preference]] = {}
    for entry in PREFERENCES:
        ordered.setdefault(entry.group, []).append(entry)
    return list(ordered.items())


def _coerce(entry: Preference, stored: Any) -> int | bool:
    """Return *stored* as *entry*'s type, or its fallback where it is not one.

    A store may hand back a string -- an INI file has no types, and a file
    somebody has edited by hand can hold anything at all. A value that will
    not convert, or one outside the range the control offers, is logged and
    replaced rather than raised on: a preference is not worth refusing to open
    the window over.
    """
    if stored is None:
        return entry.fallback
    if entry.is_flag:
        if isinstance(stored, bool):
            return stored
        return str(stored).strip().lower() in {"true", "1", "yes"}
    try:
        number = int(stored)
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring %s=%r in the settings: not a whole number", entry.key, stored
        )
        return entry.fallback
    clamped = max(entry.low, min(entry.high, number))
    if clamped != number:
        logger.warning(
            "Clamping %s=%s in the settings to %s", entry.key, number, clamped
        )
    return clamped


def read(store) -> dict[str, int | bool]:
    """Return every preference resolved against *store*, keyed by name.

    Args:
        store: Anything answering ``value(key, default)`` -- the window's
            ``QSettings``, or a dictionary-backed stand-in in a test.
    """
    return {
        entry.key: _coerce(entry, store.value(f"{SETTINGS_PREFIX}/{entry.key}", None))
        for entry in PREFERENCES
    }


def write(store, values: Mapping[str, int | bool]) -> None:
    """Record *values*, which need not be every preference.

    A value equal to its fallback is written like any other: a preference the
    user has looked at and left alone is still an answer, and one that
    disappeared when the packaged default changed under it would be a setting
    nobody could rely on.
    """
    for key, value in values.items():
        entry = BY_KEY.get(key)
        if entry is None:
            raise KeyError(f"Unknown preference: {key}")
        store.setValue(f"{SETTINGS_PREFIX}/{key}", _coerce(entry, value))


def series_from(values: Mapping[str, int | bool]) -> tuple[str, ...]:
    """Return the series an export should open on, in `layout.SERIES` order.

    Ticked nowhere leaves the Export tab counting nothing, which reads as a
    fault rather than as a setting -- so an empty answer falls back to what
    the package ships with.
    """
    chosen = tuple(name for name in SERIES if values.get(f"series/{name}"))
    return chosen or DEFAULT_SERIES
