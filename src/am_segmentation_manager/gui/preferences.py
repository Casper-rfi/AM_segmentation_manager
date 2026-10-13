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
from dataclasses import dataclass, replace
from typing import Any, TypeAlias

from ..display import DEFAULT_ALPHA
from ..layout import LABELLED_SERIES, SERIES, SERIES_LABELS
from ..operations.label_measure import (
    DEFAULT_SPECK_PIXELS,
    MAX_SPECK_PIXELS,
    MIN_SPECK_PIXELS,
)
from ..operations.training_run import Mode, TrainingSettings, default_settings
from ..operations.training_split import (
    DEFAULT_KEEP_TOGETHER,
    DEFAULT_TRAIN_RATIO,
    KEEP_TOGETHER,
)
from .export_controller import DEFAULT_SERIES, MAX_FOLDS

logger = logging.getLogger(__name__)

#: What a preference can be worth: a whole number, a switch, the key of a
#: choice, or a real number.
Value: TypeAlias = int | bool | str | float

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

#: How many cross-validation folds the Export tab opens on. One is none: a
#: single train/validation split, as `create-training-csv` writes without --folds.
DEFAULT_FOLDS = 1

#: Whether an export keeps one image per label rather than every imaging
#: condition of it. Off: the wider set is what an export has always meant, and
#: narrowing it silently would disagree with `create-training-csv`.
DEFAULT_ONE_PER_LABEL = False

#: The boxes the dialog groups these under. A group is the tab a preference is
#: felt on, not the module it reaches; appearance is how things are drawn,
#: on whichever tab draws them.
APPEARANCE = "Appearance"
PROOFREADING = "Proofreading"
EXPORT = "Export"
SHOWN = "Series shown"
TRAIN_NEW = "Train: new model"
TRAIN_REFINE = "Train: refine a model"

#: The box each way of training has, by mode.
TRAIN_GROUPS: Mapping[Mode, str] = {"new": TRAIN_NEW, "refine": TRAIN_REFINE}

#: The training settings a default can be set for, by
#: :class:`~..operations.training_run.TrainingSettings` field, as ``(field,
#: caption, hint, low, high)``. What each opens at with nothing stored is the
#: published setting for the mode, so the two cannot drift apart.
TRAINING_FIELDS: tuple[tuple[str, str, str, int | float, int | float], ...] = (
    (
        "training_batch_size", "Training batch",
        "Images per training step. Lower it if the GPU runs out of memory.", 1, 256,
    ),
    ("max_epochs", "Maximum epochs", "Maximum number of epochs.", 1, 10_000),
    (
        "frozen_epochs", "Frozen epochs",
        "Number of epochs the encoder stays frozen; fewer than the maximum.", 0, 10_000,
    ),
    (
        "initial_learning_rate", "Initial learning rate",
        "Learning rate at the start of training; no higher than the maximum.", 1e-9, 1.0,
    ),
    ("max_learning_rate", "Maximum learning rate", "Maximum learning rate reached during training.", 1e-9, 1.0),
    (
        "seed", "Seed",
        "Seed for augmentation and shuffling in training. Not the seed of the export split.",
        0, 2**31 - 1,
    ),
)

#: Which series Curate and Label open showing: every one with labels. `other`
#: is the residue no naming rule reaches -- grid overviews, their tiles,
#: diagnostics -- worth being able to open and not worth listing beside the
#: lamellae by default.
DEFAULT_SHOWN_SERIES: tuple[str, ...] = LABELLED_SERIES

#: The accents the window can be drawn in, as ``(key, caption)``. The keys are
#: those of ``widgets.theme.ACCENTS``, written out because this module holds no
#: Qt (`test_the_accent_preference_offers_every_accent_the_theme_has`).
ACCENT_CHOICES: tuple[tuple[str, str], ...] = (
    ("orange", "Orange"), ("blue", "Blue"), ("green", "Green"),
)
DEFAULT_ACCENT = "green"
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
            whole number, a float a real one, a str one of :attr:`choices`.
        hint: The tooltip, which says what the value does rather than what it
            is called.
        low: Smallest accepted value, for a number.
        high: Largest accepted value, for a number.
        suffix: Shown after a number, where it has a unit.
        choices: The answers a choice offers, as ``(key, caption)``.
        viewer: Whether a viewer offers it: only what the Curate view
            itself opens on, a viewer having no other tab to open.
    """

    key: str
    group: str
    label: str
    fallback: Value
    hint: str
    low: int | float | None = None
    high: int | float | None = None
    suffix: str = ""
    choices: tuple[tuple[str, str], ...] = ()
    viewer: bool = False

    @property
    def is_flag(self) -> bool:
        """Return whether this is a switch."""
        return isinstance(self.fallback, bool)

    @property
    def is_choice(self) -> bool:
        """Return whether this is one of a closed set of answers."""
        return isinstance(self.fallback, str)

    @property
    def is_real(self) -> bool:
        """Return whether this is a number that need not be whole."""
        return isinstance(self.fallback, float)


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
            "Colour used to highlight the selected tab, row and control."
        ),
        viewer=True,
    ),
    Preference(
        key="label_opacity",
        group=APPEARANCE,
        label="Label opacity",
        fallback=int(round(DEFAULT_ALPHA * 100)),
        low=0,
        high=100,
        suffix=" %",
        hint=(
            "Default opacity of the label on the Curate and Label tabs."
        ),
        viewer=True,
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
            "Default speck size in pixels on the Curate and Label tabs. Can be changed there for "
            "one session."
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
            "Default brush size for proofreading. Set to the nearest size the slider offers."
        ),
    ),
    Preference(
        key="advance_after_deciding",
        group=PROOFREADING,
        label="Move on after deciding a conflict",
        fallback=DEFAULT_ADVANCE,
        hint=(
            "Go to the next undecided conflict after deciding one during a merge. Can be changed "
            "in the conflict window for one merge."
        ),
    ),
    *(
        Preference(
            key=f"shown/{name}",
            group=SHOWN,
            label=SERIES_LABELS[name],
            fallback=name in DEFAULT_SHOWN_SERIES,
            hint=(
                f"Show {SERIES_LABELS[name]} images by default on the Curate and Label "
                "tabs. Lamellae without selected images are not shown. Can be changed "
                "there for one session."
            ),
            viewer=True,
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
                f"Include {SERIES_LABELS[name]} images by default on the Export tab."
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
            "Keep one image per label by default on the Export tab, instead of every variant "
            "sharing it."
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
            "Default percentage of image-label pairs used for training; the rest is for validation."
        ),
    ),
    Preference(
        key="seeded",
        group=EXPORT,
        label="Seed the split",
        fallback=DEFAULT_SEEDED,
        hint=(
            "Use a fixed seed by default, so that the same selection always gives the same split."
        ),
    ),
    Preference(
        key="seed",
        group=EXPORT,
        label="Seed",
        fallback=DEFAULT_SEED,
        low=0,
        high=999_999,
        hint="Default seed for the split.",
    ),
    Preference(
        key="keep_together",
        group=EXPORT,
        label="Keep together",
        fallback=DEFAULT_KEEP_TOGETHER,
        choices=KEEP_TOGETHER,
        hint=(
            "Default for what is kept together in either training or validation."
        ),
    ),
    Preference(
        key="folds",
        group=EXPORT,
        label="Cross-validation folds",
        fallback=DEFAULT_FOLDS,
        low=1,
        high=MAX_FOLDS,
        hint=(
            "Default number of train/validation splits for cross-validation; 1 writes a single "
            "split."
        ),
    ),
    *(
        Preference(
            key=f"train/{mode}/{name}",
            group=group,
            label=label,
            fallback=getattr(default_settings(mode), name),
            low=low,
            high=high,
            hint=f"{hint} What the Train tab opens at for this way of training.",
        )
        for mode, group in TRAIN_GROUPS.items()
        for name, label, hint, low, high in TRAINING_FIELDS
    ),
)

#: The preferences by key, for a caller that wants one by name.
BY_KEY: Mapping[str, Preference] = {entry.key: entry for entry in PREFERENCES}


def groups(*, viewer: bool = False) -> list[tuple[str, list[Preference]]]:
    """Return the preferences under their group, in the order declared.

    Args:
        viewer: Only those a viewer offers, a group left with none leaving too.
    """
    ordered: dict[str, list[Preference]] = {}
    for entry in PREFERENCES:
        if viewer and not entry.viewer:
            continue
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
    assert entry.low is not None and entry.high is not None, f"{entry.key} has no range"
    number: int | float
    try:
        number = float(stored) if entry.is_real else int(stored)
        if number != number:
            raise ValueError("not a number")
    except (TypeError, ValueError, OverflowError):
        logger.warning(
            "Ignoring %s=%r in the settings: not a %snumber",
            entry.key, stored, "" if entry.is_real else "whole ",
        )
        return entry.fallback
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


def _training_settings(values: Mapping[str, Value], mode: Mode) -> TrainingSettings:
    """Return the published settings for *mode* with whatever *values* holds for it put in."""
    held = {
        name: values[f"train/{mode}/{name}"]
        for name, *_ in TRAINING_FIELDS
        if f"train/{mode}/{name}" in values
    }
    return replace(default_settings(mode), **held)  # type: ignore[arg-type]


def training_problems(values: Mapping[str, Value]) -> list[str]:
    """Return why the training defaults in *values* cannot be trained with, said by their box."""
    return [
        f"{group}: {problem}"
        for mode, group in TRAIN_GROUPS.items()
        for problem in _training_settings(values, mode).problems()
    ]


def training_settings_from(values: Mapping[str, Value], mode: Mode) -> TrainingSettings:
    """Return the settings the Train tab should open at for *mode*.

    Each value is valid alone and a pair of them may not be -- more epochs
    frozen than trained -- so settings that cannot be trained with give way,
    whole, to the published ones.
    """
    settings = _training_settings(values, mode)
    problems = settings.problems()
    if problems:
        logger.warning(
            "Ignoring the training defaults for %s in the settings: %s", mode, "; ".join(problems)
        )
        return default_settings(mode)
    return settings


def recommended(group: str) -> dict[str, Value]:
    """Return what every preference of *group* is worth as the package ships it."""
    return {entry.key: entry.fallback for entry in PREFERENCES if entry.group == group}

