"""Tests for the declared workflow state file.

The file is hand-editable and is the only record of which images are worth
annotating, so what is pinned here is mostly what it refuses: a key that leaves
the stage, a state it does not recognise, a duplicated path, and a write over a
file something else changed first.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.state import (
    AM_STATE_FILENAME,
    FILE_VERSION,
    StageState,
    StaleState,
    StateFileError,
    read_state,
    state_file_path,
    state_key,
    write_state,
)


def written(stage: Path, states: dict[str, str]) -> StageState:
    """Write *states* into a fresh stage directory and return the result."""
    return write_state(read_state(stage), states)


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------

def test_a_key_is_the_label_path_relative_to_the_stage(tmp_path: Path):
    key = state_key(tmp_path / "labels" / "img_000_SEM.tif", tmp_path)
    assert key == "labels/img_000_SEM.tif"


def test_conditions_sharing_a_label_directory_share_one_key(tmp_path: Path):
    """The case the whole design turns on: five images, one entry."""
    labels = tmp_path / "labels"
    keys = {state_key(labels / "img_0004.tif", tmp_path) for _ in range(5)}
    assert keys == {"labels/img_0004.tif"}


def test_a_private_label_directory_gets_its_own_key(tmp_path: Path):
    shared = state_key(tmp_path / "labels" / "img.tif", tmp_path)
    private = state_key(tmp_path / "sem_8_labels" / "img.tif", tmp_path)
    assert shared != private


def test_a_label_outside_the_stage_has_no_key(tmp_path: Path):
    with pytest.raises(StateFileError):
        state_key(tmp_path.parent / "elsewhere" / "img.tif", tmp_path)


@pytest.mark.parametrize(
    "key",
    ["../other/img.tif", "/absolute/img.tif", "labels\\img.tif", "", " labels/img.tif"],
)
def test_a_key_may_not_leave_the_stage(tmp_path: Path, key: str):
    """A hand-edited key must not be able to name a file in another dataset.

    Checked through :func:`read_state`, which is how a hand-edited key really
    arrives, rather than through the validator on its own.
    """
    state_file_path(tmp_path).write_text(
        f"version: 1\nstate:\n  {key!r}: deselected\n", encoding="utf-8"
    )
    with pytest.raises(StateFileError):
        read_state(tmp_path)


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------

def test_a_missing_file_reads_as_everything_selected(tmp_path: Path):
    read = read_state(tmp_path)
    assert read.states == {}
    assert read.fingerprint is None
    assert read.state_of("labels/anything.tif") == "selected"


def test_states_survive_a_round_trip(tmp_path: Path):
    written(tmp_path, {"labels/a.tif": "deselected", "labels/b.tif": "corrected"})
    read = read_state(tmp_path)
    assert dict(read.states) == {"labels/a.tif": "deselected", "labels/b.tif": "corrected"}


def test_an_unlisted_group_is_selected(tmp_path: Path):
    written(tmp_path, {"labels/a.tif": "deselected"})
    assert read_state(tmp_path).state_of("labels/b.tif") == "selected"


def test_the_default_state_is_not_written_out(tmp_path: Path):
    """A state file lists what is *not* selected; the rest is silence."""
    result = written(tmp_path, {"labels/a.tif": "selected", "labels/b.tif": "corrected"})
    assert dict(result.states) == {"labels/b.tif": "corrected"}
    assert "labels/a.tif" not in state_file_path(tmp_path).read_text(encoding="utf-8")


def test_a_stage_with_nothing_to_declare_keeps_its_file(tmp_path: Path):
    """Nothing deletes a file the user may have written in by hand."""
    result = write_state(written(tmp_path, {"labels/a.tif": "corrected"}), {})
    assert state_file_path(tmp_path).exists()
    assert dict(result.states) == {}
    assert read_state(tmp_path).states == {}


def test_the_file_explains_itself(tmp_path: Path):
    """Someone without this software has to be able to read it."""
    written(tmp_path, {"labels/a.tif": "deselected"})
    text = state_file_path(tmp_path).read_text(encoding="utf-8")
    assert text.startswith("# Workflow state for this stage")
    assert "selected | deselected | corrected" in text
    assert f"version: {FILE_VERSION}" in text
    assert "labels/a.tif: deselected" in text


def test_the_file_is_named_beside_the_image_directories(tmp_path: Path):
    assert state_file_path(tmp_path) == tmp_path / AM_STATE_FILENAME


def test_keys_are_written_sorted(tmp_path: Path):
    """Two runs producing the same states produce the same file."""
    written(tmp_path, {"labels/c.tif": "corrected", "labels/a.tif": "corrected"})
    text = state_file_path(tmp_path).read_text(encoding="utf-8")
    assert text.index("labels/a.tif") < text.index("labels/c.tif")


def test_a_filename_holding_a_colon_survives(tmp_path: Path):
    """The body is dumped rather than formatted by hand, so quoting is right."""
    written(tmp_path, {"labels/img: 1.tif": "deselected"})
    assert dict(read_state(tmp_path).states) == {"labels/img: 1.tif": "deselected"}


# ---------------------------------------------------------------------------
# What a file may not hold
# ---------------------------------------------------------------------------

def test_a_duplicated_path_is_an_error(tmp_path: Path):
    """One state per group, or the mapping was pointless."""
    state_file_path(tmp_path).write_text(
        "version: 1\nstate:\n  labels/a.tif: corrected\n  labels/a.tif: deselected\n",
        encoding="utf-8",
    )
    with pytest.raises(StateFileError, match="Duplicate"):
        read_state(tmp_path)


def test_an_unknown_state_is_an_error(tmp_path: Path):
    state_file_path(tmp_path).write_text(
        "version: 1\nstate:\n  labels/a.tif: probably\n", encoding="utf-8"
    )
    with pytest.raises(StateFileError, match="probably"):
        read_state(tmp_path)


def test_a_key_leaving_the_stage_is_an_error_on_read(tmp_path: Path):
    state_file_path(tmp_path).write_text(
        "version: 1\nstate:\n  ../elsewhere/a.tif: corrected\n", encoding="utf-8"
    )
    with pytest.raises(StateFileError):
        read_state(tmp_path)


def test_a_later_file_version_is_refused(tmp_path: Path):
    """Refusing beats misreading a file a later am-seg wrote."""
    state_file_path(tmp_path).write_text(
        f"version: {FILE_VERSION + 1}\nstate:\n  labels/a.tif: corrected\n",
        encoding="utf-8",
    )
    with pytest.raises(StateFileError, match="later version"):
        read_state(tmp_path)


def test_a_file_without_a_version_is_read(tmp_path: Path):
    state_file_path(tmp_path).write_text(
        "state:\n  labels/a.tif: corrected\n", encoding="utf-8"
    )
    assert dict(read_state(tmp_path).states) == {"labels/a.tif": "corrected"}


def test_an_empty_file_reads_as_nothing_declared(tmp_path: Path):
    state_file_path(tmp_path).write_text("", encoding="utf-8")
    assert read_state(tmp_path).states == {}


def test_unparseable_yaml_names_the_file(tmp_path: Path):
    state_file_path(tmp_path).write_text("state:\n  - [\n", encoding="utf-8")
    with pytest.raises(StateFileError, match=AM_STATE_FILENAME):
        read_state(tmp_path)


def test_a_state_block_that_is_not_a_mapping_is_an_error(tmp_path: Path):
    state_file_path(tmp_path).write_text(
        "version: 1\nstate:\n  - labels/a.tif\n", encoding="utf-8"
    )
    with pytest.raises(StateFileError, match="mapping"):
        read_state(tmp_path)


def test_an_unknown_state_is_refused_on_write_too(tmp_path: Path):
    with pytest.raises(StateFileError):
        written(tmp_path, {"labels/a.tif": "probably"})


# ---------------------------------------------------------------------------
# Concurrent writers
# ---------------------------------------------------------------------------

def test_a_write_refuses_a_file_that_changed_since_it_was_read(tmp_path: Path):
    """Two windows on one collection must not discard each other's curation."""
    first = read_state(tmp_path)
    write_state(read_state(tmp_path), {"labels/a.tif": "corrected"})

    with pytest.raises(StaleState):
        write_state(first, {"labels/b.tif": "deselected"})

    assert dict(read_state(tmp_path).states) == {"labels/a.tif": "corrected"}


def test_a_write_returns_a_state_that_can_be_written_again(tmp_path: Path):
    result = written(tmp_path, {"labels/a.tif": "corrected"})
    result = write_state(result, {"labels/a.tif": "deselected"})
    assert dict(read_state(tmp_path).states) == {"labels/a.tif": "deselected"}


def test_a_failed_write_leaves_the_previous_file_intact(tmp_path: Path, monkeypatch):
    """The rename is what makes a write whole or absent."""
    from am_segmentation_manager import state as state_module

    result = written(tmp_path, {"labels/a.tif": "corrected"})
    before = state_file_path(tmp_path).read_text(encoding="utf-8")

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(state_module, "write_atomically", explode)
    with pytest.raises(OSError):
        write_state(result, {"labels/a.tif": "deselected"})

    assert state_file_path(tmp_path).read_text(encoding="utf-8") == before
