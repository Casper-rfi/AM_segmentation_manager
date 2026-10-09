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
        if entry.is_choice:
            assert entry.fallback in {key for key, _caption in entry.choices}, entry.key
        elif not entry.is_flag:
            assert entry.low is not None and entry.high is not None, entry.key
            assert entry.low <= entry.fallback <= entry.high, entry.key


def test_the_groups_hold_every_preference_once():
    """Grouping is what the dialog iterates, so one dropped is one no control
    is built for and no test would otherwise notice."""
    grouped = [entry for _title, entries in preferences.groups() for entry in entries]

    assert sorted(entry.key for entry in grouped) == sorted(
        entry.key for entry in preferences.PREFERENCES
    )


def test_the_lists_open_without_the_residue():
    """`other` is worth reaching and not worth listing beside the lamellae."""
    values = preferences.read(Store())

    assert preferences.shown_series_from(values) == tuple(
        name for name in SERIES if name != "other"
    )


def test_showing_no_series_falls_back_to_what_the_package_ships():
    values = preferences.read(
        Store({f"{preferences.SETTINGS_PREFIX}/shown/{name}": False for name in SERIES})
    )

    assert preferences.shown_series_from(values) == preferences.DEFAULT_SHOWN_SERIES


def test_a_choice_survives_being_written_and_read_back():
    store = Store()

    preferences.write(store, {"accent": "blue"})

    assert preferences.read(store)["accent"] == "blue"


def test_the_window_opens_green_with_nothing_stored():
    """Green was the scientist's choice for the release."""
    assert preferences.read(Store())["accent"] == "green"


def test_a_choice_not_offered_falls_back_rather_than_raising(caplog):
    """A hand-edited settings file, or one written by a later version offering
    an accent this one does not have."""
    store = Store({f"{preferences.SETTINGS_PREFIX}/accent": "purple"})

    assert preferences.read(store)["accent"] == preferences.DEFAULT_ACCENT
    assert "accent='purple'" in caplog.text


def test_the_accent_opens_the_settings_dialog():
    """Felt on every tab, so it comes before the groups felt on one."""
    assert preferences.groups()[0][0] == preferences.APPEARANCE


def test_a_viewer_offers_only_what_its_one_view_opens_on():
    """The accent, how a label is drawn, and which series the lists cover;
    nothing a viewer has no tab or control for."""
    offered = {entry.key for _title, entries in preferences.groups(viewer=True) for entry in entries}

    assert offered == {"accent", "label_opacity", *(f"shown/{name}" for name in SERIES)}
    assert [title for title, _ in preferences.groups(viewer=True)] == [
        preferences.APPEARANCE, preferences.SHOWN,
    ]


def test_the_full_dialog_still_offers_every_preference_once():
    listed = [entry.key for _title, entries in preferences.groups() for entry in entries]

    assert sorted(listed) == sorted(entry.key for entry in preferences.PREFERENCES)


# -- what training opens at ---------------------------------------------------


def test_training_opens_at_the_published_settings_with_nothing_stored():
    """The fallbacks are read from the published settings, so there is one place they are set."""
    from am_segmentation_manager.operations.training_run import NEW_MODEL, REFINE

    values = preferences.read(Store())

    assert preferences.training_settings_from(values, "new") == NEW_MODEL
    assert preferences.training_settings_from(values, "refine") == REFINE


def test_each_way_of_training_keeps_its_own_defaults():
    store = Store()
    preferences.write(store, {"train/new/max_epochs": 200, "train/refine/training_batch_size": 2})
    values = preferences.read(store)

    new = preferences.training_settings_from(values, "new")
    refine = preferences.training_settings_from(values, "refine")

    assert (new.max_epochs, new.training_batch_size) == (200, 6)
    assert (refine.max_epochs, refine.training_batch_size) == (50, 2)


def test_a_learning_rate_is_read_as_a_real_number_whatever_the_store_holds():
    """An INI file hands back text."""
    values = preferences.read(Store({
        "defaults/train/new/initial_learning_rate": "3e-06",
        "defaults/train/new/max_learning_rate": "nonsense",
        "defaults/train/refine/max_learning_rate": "7",
        "defaults/train/refine/initial_learning_rate": "nan",
    }))

    assert values["train/new/initial_learning_rate"] == 3e-6
    assert values["train/new/max_learning_rate"] == 1e-3, "not a number: the published one"
    assert values["train/refine/max_learning_rate"] == 1.0, "held to the range"
    assert values["train/refine/initial_learning_rate"] == 2e-6


def test_defaults_that_cannot_be_trained_with_give_way_to_the_published_ones(caplog):
    """Each value is valid alone; more epochs frozen than trained is not."""
    from am_segmentation_manager.operations.training_run import NEW_MODEL

    values = preferences.read(Store({
        "defaults/train/new/max_epochs": 10, "defaults/train/new/frozen_epochs": 25,
    }))

    assert preferences.training_problems(values)[0].startswith(preferences.TRAIN_NEW)
    with caplog.at_level("WARNING"):
        assert preferences.training_settings_from(values, "new") == NEW_MODEL
    assert "frozen" in caplog.text
    assert preferences.training_problems(preferences.read(Store())) == []


def test_the_seed_of_training_is_not_the_seed_of_an_export_s_split():
    """Replicates on seeds 1, 2 and 3 must not reshuffle which images validate."""
    store = Store()
    preferences.write(store, {"train/new/seed": 7})
    values = preferences.read(store)

    assert values["seed"] == preferences.DEFAULT_SEED
    assert preferences.training_settings_from(values, "new").seed == 7


def test_a_training_box_can_be_put_back_to_what_the_package_ships():
    from am_segmentation_manager.operations.training_run import REFINE

    shipped = preferences.recommended(preferences.TRAIN_REFINE)

    assert shipped["train/refine/max_epochs"] == REFINE.max_epochs
    assert all(key.startswith("train/refine/") for key in shipped)
