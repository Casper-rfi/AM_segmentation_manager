"""What a control opens on, resolved without a window.

The table and the resolving are Qt-free, so these run with a dictionary in
place of the window's settings and need no display -- which is the point of
keeping them below `gui/widgets/`.
"""
from __future__ import annotations

import pytest

from am_segmentation_manager.display import DEFAULT_ALPHA
from am_segmentation_manager.gui import preferences
from am_segmentation_manager.gui.export_controller import DEFAULT_SERIES
from am_segmentation_manager.layout import SERIES
from am_segmentation_manager.operations.label_measure import (
    DEFAULT_SPECK_PIXELS,
    MAX_SPECK_PIXELS,
)


class Store:
    """A stand-in for the window's QSettings, holding whatever it is given."""

    def __init__(self, held: dict | None = None) -> None:
        self.held = dict(held or {})

    def value(self, key: str, default=None):
        return self.held.get(key, default)

    def setValue(self, key: str, value) -> None:  # noqa: N802 - QSettings naming
        self.held[key] = value


def test_nothing_stored_gives_the_packaged_answers():
    """A fresh install and the dialog have to agree about what a control opens
    at, or the dialog shows a value the window is not using."""
    values = preferences.read(Store())

    assert values["speck_pixels"] == DEFAULT_SPECK_PIXELS
    assert values["label_opacity"] == int(round(DEFAULT_ALPHA * 100))
    assert preferences.series_from(values) == DEFAULT_SERIES


def test_a_preference_survives_being_written_and_read_back():
    store = Store()

    preferences.write(store, {"speck_pixels": 12, "seeded": False})

    values = preferences.read(store)
    assert values["speck_pixels"] == 12
    assert values["seeded"] is False


def test_a_stored_string_is_read_as_the_type_the_table_declares():
    """An INI file has no types, so everything comes back as text."""
    store = Store({
        f"{preferences.SETTINGS_PREFIX}/speck_pixels": "24",
        f"{preferences.SETTINGS_PREFIX}/seeded": "false",
    })

    values = preferences.read(store)

    assert values["speck_pixels"] == 24
    assert values["seeded"] is False


def test_a_value_outside_the_range_is_clamped_rather_than_offered():
    """The settings file can be edited by hand, and a spin box asked for a
    value outside its range takes the nearest without saying so -- which would
    leave the dialog and the window disagreeing."""
    store = Store({f"{preferences.SETTINGS_PREFIX}/speck_pixels": 10_000})

    assert preferences.read(store)["speck_pixels"] == MAX_SPECK_PIXELS


def test_a_value_that_is_not_a_number_falls_back_rather_than_raising():
    """A preference is not worth refusing to open the window over."""
    store = Store({f"{preferences.SETTINGS_PREFIX}/speck_pixels": "large"})

    assert preferences.read(store)["speck_pixels"] == DEFAULT_SPECK_PIXELS


def test_no_series_ticked_falls_back_to_what_the_package_ships():
    """Ticked nowhere leaves the Export tab counting nothing, which reads as a
    fault rather than as a setting."""
    values = {f"series/{name}": False for name in SERIES}

    assert preferences.series_from(values) == DEFAULT_SERIES


def test_the_series_come_back_in_the_order_the_package_states_them():
    values = {f"series/{name}": True for name in SERIES}

    assert preferences.series_from(values) == SERIES


def test_writing_an_unknown_preference_is_refused():
    """A typo would otherwise write a key nothing ever reads back."""
    with pytest.raises(KeyError, match="opacity_of_labels"):
        preferences.write(Store(), {"opacity_of_labels": 50})


def test_every_preference_has_a_group_a_caption_and_a_hint():
    """The dialog is built from the table, so a row missing any of the three
    draws a control nobody can identify."""
    for entry in preferences.PREFERENCES:
        assert entry.group and entry.label and entry.hint, entry.key
        if not entry.is_flag:
            assert entry.low is not None and entry.high is not None, entry.key
            assert entry.low <= entry.fallback <= entry.high, entry.key


def test_the_groups_hold_every_preference_once():
    """Grouping is what the dialog iterates, so one dropped is one no control
    is built for and no test would otherwise notice."""
    grouped = [entry for _title, entries in preferences.groups() for entry in entries]

    assert sorted(entry.key for entry in grouped) == sorted(
        entry.key for entry in preferences.PREFERENCES
    )
