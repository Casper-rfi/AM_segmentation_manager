"""What the window and its controls open on, and where those answers are kept.

**A default decides the value a control starts at, never what an operation
does**: the command line and every write path keep the packaged constants, so
nothing set here changes what a run produces, and a preference travels with the
person rather than the data. Holds no Qt -- the store is passed in, so a test
reads one with a dictionary. **A preference is a row in :data:`PREFERENCES`**,
as a descriptive field is a row in ``metadata.FIELDS``.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from ..display import DEFAULT_ALPHA
from ..layout import LABELLED_SERIES, SERIES, SERIES_LABELS
from ..operations.csv import DEFAULT_TRAIN_RATIO
from ..operations.label_measure import (
    DEFAULT_SPECK_PIXELS,
    MAX_SPECK_PIXELS,
    MIN_SPECK_PIXELS,
)
from .export_controller import DEFAULT_SERIES

logger = logging.getLogger(__name__)

#: What a preference can be worth: a number, a switch, or the key of a choice.
Value: TypeAlias = int | bool | str

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

#: The boxes the dialog groups these under. A group is the tab a preference is
#: felt on, not the module it reaches; appearance is felt on every tab.
APPEARANCE = "Appearance"
PROOFREADING = "Proofreading"
EXPORT = "Export"
SHOWN = "Series shown"

#: Which series Curate and Label open showing: every one with labels. `other`
#: is the residue no naming rule reaches -- grid overviews, their tiles,
#: diagnostics -- worth being able to open and not worth listing beside the
#: lamellae by default.
DEFAULT_SHOWN_SERIES: tuple[str, ...] = LABELLED_SERIES

#: The accents the window can be drawn in, as ``(key, caption)``. The keys are
#: those of ``widgets.theme.ACCENTS``, written out because this module holds no
#: Qt (`test_the_accent_preference_offers_every_accent_the_theme_has`).
ACCENT_CHOICES: tuple[tuple[str, str], ...] = (("orange", "Orange"), ("blue", "Blue"))
DEFAULT_ACCENT = "orange"
#: The accent's key in :data:`PREFERENCES`. Named, because it is the one
#: preference the window applies while it is open.
ACCENT = "accent"


@dataclass(frozen=True)
class Preference:
    """One value a control opens on.

    Attributes:
        key: Its name under :data:`SETTINGS_PREFIX`.
        group: Which box of the settings dialog it appears in.
        label: Its caption there.
        fallback: What it is worth with nothing stored -- the packaged
            constant, so the dialog and a fresh install agree. Its type says
            what kind of preference this is: a bool is a switch, an int a
            number, a str one of :attr:`choices`.
        hint: The tooltip, which says what the value does rather than what it
            is called.
        low: Smallest accepted value, for a number.
        high: Largest accepted value, for a number.
        suffix: Shown after a number, where it has a unit.
        choices: The answers a choice offers, as ``(key, caption)``.
    """

    key: str
    group: str
    label: str
    fallback: Value
    hint: str
    low: int | None = None
    high: int | None = None
    suffix: str = ""
    choices: tuple[tuple[str, str], ...] = ()

    @property
    def is_flag(self) -> bool:
        """Return whether this is a switch."""
        return isinstance(self.fallback, bool)

    @property
    def is_choice(self) -> bool:
        """Return whether this is one of a closed set of answers."""
        return isinstance(self.fallback, str)


#: Whether the conflict resolver moves on after a decision. Off, because a
#: view that moves as the key is pressed has to be learnt before it helps.
DEFAULT_ADVANCE = False


PREFERENCES: tuple[Preference, ...] = (
    Preference(
        key=ACCENT,
        group=APPEARANCE,
        label="Accent colour",
        fallback=DEFAULT_ACCENT,
        choices=ACCENT_CHOICES,
        hint=(
            "The colour that marks the chosen tab, row and segment, and the "
            "control in focus. Nothing that carries meaning changes with it."
        ),
    ),
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
    Preference(
        key="advance_after_deciding",
        group=PROOFREADING,
        label="Move on after deciding a conflict",
        fallback=DEFAULT_ADVANCE,
        hint=(
            "Whether a merge's conflict resolver opens going to the next "
            "undecided conflict once one is decided. The switch beside its "
            "viewer changes it for one merge."
        ),
    ),
    *(
        Preference(
            key=f"shown/{name}",
            group=SHOWN,
            label=SERIES_LABELS[name],
            fallback=name in DEFAULT_SHOWN_SERIES,
            hint=(
                f"Whether Curate and Label open showing {SERIES_LABELS[name]} "
                "images. A lamella holding only series switched off leaves the "
                "list, counted in its title. The switches beside the lists "
                "move it for a session."
            ),
        )
        for name in SERIES
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


def _coerce(entry: Preference, stored: Any) -> Value:
    """Return *stored* as *entry*'s type, or its fallback where it is not one.

    A store may hand back a string -- an INI file has no types, and a file
    somebody has edited by hand can hold anything at all. A value that will
    not convert, one outside the range the control offers, or a choice it does
    not offer, is logged and replaced rather than raised on: a preference is not
    worth refusing to open the window over.
    """
    if stored is None:
        return entry.fallback
    if entry.is_choice:
        if str(stored) in {key for key, _caption in entry.choices}:
            return str(stored)
        logger.warning(
            "Ignoring %s=%r in the settings: not one of the choices", entry.key, stored
        )
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
    assert entry.low is not None and entry.high is not None, f"{entry.key} has no range"
    clamped = max(entry.low, min(entry.high, number))
    if clamped != number:
        logger.warning(
            "Clamping %s=%s in the settings to %s", entry.key, number, clamped
        )
    return clamped


def read(store) -> dict[str, Value]:
    """Return every preference resolved against *store*, keyed by name.

    Args:
        store: Anything answering ``value(key, default)`` -- the window's
            ``QSettings``, or a dictionary-backed stand-in in a test.
    """
    return {
        entry.key: _coerce(entry, store.value(f"{SETTINGS_PREFIX}/{entry.key}", None))
        for entry in PREFERENCES
    }


def write(store, values: Mapping[str, Value]) -> None:
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


def accent_from(values: Mapping[str, Value]) -> str:
    """Return the key of the accent the window should open in."""
    return str(values[ACCENT])


def series_from(values: Mapping[str, Value]) -> tuple[str, ...]:
    """Return the series an export should open on, in `layout.SERIES` order.

    Ticked nowhere leaves the Export tab counting nothing, which reads as a
    fault rather than as a setting -- so an empty answer falls back to what
    the package ships with.
    """
    chosen = tuple(name for name in SERIES if values.get(f"series/{name}"))
    return chosen or DEFAULT_SERIES


def shown_series_from(values: Mapping[str, Value]) -> tuple[str, ...]:
    """Return the series Curate and Label should open showing, in `SERIES` order.

    Shown nowhere would list nothing, which reads as a fault rather than as a
    setting -- so an empty answer falls back to what the package ships with.
    """
    chosen = tuple(name for name in SERIES if values.get(f"shown/{name}"))
    return chosen or DEFAULT_SHOWN_SERIES
