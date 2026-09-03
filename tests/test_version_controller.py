"""Tests for the kept versions of the label on screen.

All of it runs without a display, which is what keeping the versions in a
controller buys. What is pinned is mostly what makes the list usable: that a row
says enough to choose from without opening anything, that a version whose pixels
did not travel is shown as absent rather than hidden, and that putting one back
goes through the ordinary save so every guard still applies.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.gui.version_controller import VersionController
from am_segmentation_manager.history import Recorder, history_file_path
from am_segmentation_manager.io.atomic import fingerprint_of
from am_segmentation_manager.operations.label_save import LabelToSave, save_labels

np = pytest.importorskip("numpy")
pytest.importorskip("tifffile")


@pytest.fixture
def recorder() -> Recorder:
    """A recorder attributing everything to a named author."""
    return Recorder(by="casper", by_source="flag")


@pytest.fixture
def label(tmp_path: Path, recorder) -> Path:
    """A label in a project that keeps a history, with one kept version."""
    history_file_path(tmp_path).touch()
    path = tmp_path / "sem_labels" / "img_000.tif"
    _write(path, 2)
    _save(path, _pixels(3), recorder)
    return path


def _pixels(value: int, shape: tuple[int, int] = (16, 16)):
    array = np.zeros(shape, dtype=np.uint8)
    array[2:6, 2:6] = value
    return array


def _write(path: Path, value: int) -> Path:
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, _pixels(value))
    return path


def _save(path: Path, array, recorder: Recorder, **kwargs) -> None:
    save_labels(
        [
            LabelToSave(
                destination=path, label=array, fingerprint=fingerprint_of(path),
                **kwargs,
            )
        ],
        recorder=recorder,
        allow_corrected=kwargs.get("corrected", False),
    )


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def test_a_label_with_no_kept_versions_has_nothing_to_show(tmp_path: Path, recorder):
    history_file_path(tmp_path).touch()
    fresh = _write(tmp_path / "sem_labels" / "new.tif", 1)

    controller = VersionController(recorder)
    controller.set_label(fresh)

    assert not controller.has_versions and controller.rows == ()


def test_a_replaced_version_is_listed(label, recorder):
    controller = VersionController(recorder)
    controller.set_label(label)

    assert controller.has_versions
    (row,) = controller.rows
    assert row.who == "casper"
    assert row.produced_by == "proofreading save"
    assert row.available


def test_a_row_says_how_far_the_version_is_from_the_next(label, recorder):
    """The number that makes ten rows readable without opening any of them."""
    controller = VersionController(recorder)
    controller.set_label(label)

    (row,) = controller.rows
    assert row.changed == pytest.approx(16 / 256)
    assert "6.2% changed" in row.describe()


def test_versions_are_listed_newest_first(label, recorder):
    for value in (4, 5):
        _save(label, _pixels(value), recorder)

    controller = VersionController(recorder)
    controller.set_label(label)

    assert len(controller.rows) == 3
    when = [row.when for row in controller.rows]
    assert when == sorted(when, reverse=True)


def test_a_version_whose_pixels_did_not_travel_is_shown_as_absent(label, recorder):
    """A list that hid the row would be worse than one admitting the gap."""
    from am_segmentation_manager.operations.archive import versions_of

    controller = VersionController(recorder)
    controller.set_label(label)
    (kept,) = versions_of(label.parents[1], "sem_labels/img_000.tif")
    kept.path.unlink()

    controller.refresh()

    (row,) = controller.rows
    assert not row.available
    assert "not in this copy" in row.describe()


def test_a_label_in_a_project_with_no_history_shows_nothing(tmp_path: Path, recorder):
    lonely = _write(tmp_path / "sem_labels" / "img.tif", 2)

    controller = VersionController(recorder)
    controller.set_label(lonely)

    assert not controller.has_versions


# ---------------------------------------------------------------------------
# Comparing
# ---------------------------------------------------------------------------

def test_nothing_is_selected_to_begin_with(label, recorder):
    controller = VersionController(recorder)
    controller.set_label(label)

    assert controller.selected is None
    assert controller.comparison_pixels() is None


def test_selecting_a_version_loads_its_pixels(label, recorder):
    controller = VersionController(recorder)
    controller.set_label(label)

    controller.select(0)

    assert np.array_equal(controller.comparison_pixels(), _pixels(2))


def test_the_difference_marks_only_where_they_disagree(label, recorder):
    controller = VersionController(recorder)
    controller.set_label(label)
    controller.select(0)

    difference = controller.difference(_pixels(3))

    assert difference is not None
    assert difference.sum() == 16
    assert difference[2:6, 2:6].all()


def test_a_version_of_another_size_has_no_difference_to_show(
    tmp_path: Path, recorder
):
    """A conversion writes at model resolution; a mask across a resize would
    mark every pixel, which says nothing."""
    history_file_path(tmp_path).touch()
    path = tmp_path / "sem_labels" / "img.tif"
    _write(path, 2)
    _save(path, _pixels(3, shape=(32, 32)), recorder)

    controller = VersionController(recorder)
    controller.set_label(path)
    controller.select(0)

    assert controller.difference(_pixels(3, shape=(32, 32))) is None


def test_moving_to_another_frame_clears_the_comparison(label, recorder, tmp_path):
    controller = VersionController(recorder)
    controller.set_label(label)
    controller.select(0)

    controller.set_label(_write(tmp_path / "sem_labels" / "other.tif", 1))

    assert controller.selected is None


def test_selecting_out_of_range_changes_nothing(label, recorder):
    controller = VersionController(recorder)
    controller.set_label(label)

    controller.select(9)

    assert controller.selected is None


# ---------------------------------------------------------------------------
# Putting one back
# ---------------------------------------------------------------------------

def test_restoring_puts_the_pixels_back(label, recorder):
    import tifffile

    controller = VersionController(recorder)
    controller.set_label(label)
    controller.select(0)

    controller.restore_selected()

    assert np.array_equal(tifffile.imread(label), _pixels(2))


def test_restoring_keeps_the_version_it_replaced(label, recorder):
    controller = VersionController(recorder)
    controller.set_label(label)
    controller.select(0)

    controller.restore_selected()

    assert len(controller.rows) == 2


def test_restoring_over_proofread_work_needs_confirming(label, recorder):
    """Restoring is the ordinary save, so the ordinary guard applies."""
    controller = VersionController(recorder)
    controller.set_label(label)
    controller.select(0)

    with pytest.raises(PermissionError):
        controller.restore_selected(corrected=True)

    controller.restore_selected(corrected=True, confirmed=True)


def test_restoring_a_version_that_is_not_here_is_refused(label, recorder):
    from am_segmentation_manager.operations.archive import versions_of

    controller = VersionController(recorder)
    controller.set_label(label)
    (kept,) = versions_of(label.parents[1], "sem_labels/img_000.tif")
    kept.path.unlink()
    controller.refresh()
    controller.select(0)

    with pytest.raises(ValueError, match="not in this copy"):
        controller.restore_selected()


def test_restoring_with_nothing_selected_is_refused(label, recorder):
    controller = VersionController(recorder)
    controller.set_label(label)

    with pytest.raises(ValueError, match="No version"):
        controller.restore_selected()


def test_a_restore_is_recorded_as_one(label, recorder):
    from am_segmentation_manager.history import read_log

    controller = VersionController(recorder)
    controller.set_label(label)
    controller.select(0)

    controller.restore_selected()

    kinds = [event.kind for event in read_log(label.parents[1]).events]
    assert "label-restored" in kinds
