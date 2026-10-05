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

from am_segmentation_manager.gui.attribution import Attribution
from am_segmentation_manager.gui.version_controller import VersionController
from am_segmentation_manager.history import Recorder
from am_segmentation_manager.io.atomic import fingerprint_of
from am_segmentation_manager.operations.label_save import LabelToSave, save_labels
from conftest import start_history

np = pytest.importorskip("numpy")
pytest.importorskip("tifffile")


@pytest.fixture
def recorder() -> Recorder:
    """A recorder attributing everything to a named author."""
    return Recorder(by="casper", by_source="flag")


@pytest.fixture
def label(tmp_path: Path, recorder) -> Path:
    """A label in a project that keeps a history, with one kept version."""
    start_history(tmp_path)
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
    start_history(tmp_path)
    fresh = _write(tmp_path / "sem_labels" / "new.tif", 1)

    controller = VersionController(Attribution(recorder))
    controller.set_label(fresh)

    assert not controller.has_versions and controller.rows == ()


def test_a_replaced_version_is_listed(label, recorder):
    controller = VersionController(Attribution(recorder))
    controller.set_label(label)

    assert controller.has_versions
    (row,) = controller.rows
    assert row.who == "casper"
    assert row.produced_by == "proofreading save"
    assert row.available


def test_a_row_says_how_far_the_version_is_from_the_next(label, recorder):
    """The number that makes ten rows readable without opening any of them."""
    controller = VersionController(Attribution(recorder))
    controller.set_label(label)

    (row,) = controller.rows
    assert row.changed == pytest.approx(16 / 256)
    assert "6.2% changed" in row.describe()


def test_versions_are_listed_newest_first(label, recorder):
    for value in (4, 5):
        _save(label, _pixels(value), recorder)

    controller = VersionController(Attribution(recorder))
    controller.set_label(label)

    assert len(controller.rows) == 3
    when = [row.when for row in controller.rows]
    assert when == sorted(when, reverse=True)


def test_a_version_whose_pixels_did_not_travel_is_shown_as_absent(label, recorder):
    """A list that hid the row would be worse than one admitting the gap."""
    from am_segmentation_manager.operations.archive import versions_of

    controller = VersionController(Attribution(recorder))
    controller.set_label(label)
    (kept,) = versions_of(label.parents[1], "sem_labels/img_000.tif")
    kept.path.unlink()

    controller.refresh()

    (row,) = controller.rows
    assert not row.available
    assert "pixels not here" in row.describe()


def test_a_label_in_a_project_with_no_history_shows_nothing(tmp_path: Path, recorder):
    lonely = _write(tmp_path / "sem_labels" / "img.tif", 2)

    controller = VersionController(Attribution(recorder))
    controller.set_label(lonely)

    assert not controller.has_versions


# ---------------------------------------------------------------------------
# Comparing
# ---------------------------------------------------------------------------

def test_nothing_is_selected_to_begin_with(label, recorder):
    controller = VersionController(Attribution(recorder))
    controller.set_label(label)

    assert controller.selected is None
    assert controller.comparison_pixels() is None


def test_selecting_a_version_loads_its_pixels(label, recorder):
    controller = VersionController(Attribution(recorder))
    controller.set_label(label)

    controller.select(0)

    assert np.array_equal(controller.comparison_pixels(), _pixels(2))


def test_the_difference_marks_only_where_they_disagree(label, recorder):
    controller = VersionController(Attribution(recorder))
    controller.set_label(label)
    controller.select(0)

    difference = controller.difference(_pixels(3))

    assert difference is not None
    assert difference.sum() == 16
    assert difference[2:6, 2:6].all()


def test_a_kept_version_of_another_size_is_shown_as_it_is(tmp_path: Path, recorder):
    """Its size is part of what is compared, so nothing resizes it, and a mask
    across a resize would mark every pixel, which says nothing."""
    start_history(tmp_path)
    path = tmp_path / "sem_labels" / "img.tif"
    _write(path, 2)
    _save(path, _pixels(3, shape=(32, 32)), recorder)

    controller = VersionController(Attribution(recorder))
    controller.set_label(path)
    controller.select(0)

    assert controller.comparison_pixels((32, 32)).shape == (16, 16)
    assert controller.difference(_pixels(3, shape=(32, 32))) is None


def test_moving_to_another_frame_clears_the_comparison(label, recorder, tmp_path):
    controller = VersionController(Attribution(recorder))
    controller.set_label(label)
    controller.select(0)

    controller.set_label(_write(tmp_path / "sem_labels" / "other.tif", 1))

    assert controller.selected is None


def test_selecting_out_of_range_changes_nothing(label, recorder):
    controller = VersionController(Attribution(recorder))
    controller.set_label(label)

    controller.select(9)

    assert controller.selected is None


# ---------------------------------------------------------------------------
# Putting one back
# ---------------------------------------------------------------------------

def test_restoring_puts_the_pixels_back(label, recorder):
    import tifffile

    controller = VersionController(Attribution(recorder))
    controller.set_label(label)
    controller.select(0)

    controller.restore_selected()

    assert np.array_equal(tifffile.imread(label), _pixels(2))


def test_restoring_keeps_the_version_it_replaced(label, recorder):
    controller = VersionController(Attribution(recorder))
    controller.set_label(label)
    controller.select(0)

    controller.restore_selected()

    assert len(controller.rows) == 2


def test_restoring_over_proofread_work_needs_confirming(label, recorder):
    """Restoring is the ordinary save, so the ordinary guard applies."""
    controller = VersionController(Attribution(recorder))
    controller.set_label(label)
    controller.select(0)

    with pytest.raises(PermissionError):
        controller.restore_selected(corrected=True)

    controller.restore_selected(corrected=True, confirmed=True)


def test_restoring_a_version_that_is_not_here_is_refused(label, recorder):
    from am_segmentation_manager.operations.archive import versions_of

    controller = VersionController(Attribution(recorder))
    controller.set_label(label)
    (kept,) = versions_of(label.parents[1], "sem_labels/img_000.tif")
    kept.path.unlink()
    controller.refresh()
    controller.select(0)

    with pytest.raises(ValueError, match="but not its pixels"):
        controller.restore_selected()


def test_restoring_with_nothing_selected_is_refused(label, recorder):
    controller = VersionController(Attribution(recorder))
    controller.set_label(label)

    with pytest.raises(ValueError, match="No version"):
        controller.restore_selected()


def test_a_restore_is_recorded_as_one(label, recorder):
    from am_segmentation_manager.history import read_log

    controller = VersionController(Attribution(recorder))
    controller.set_label(label)
    controller.select(0)

    controller.restore_selected()

    kinds = [event.kind for event in read_log(label.parents[1]).events]
    assert "label-restored" in kinds


# ---------------------------------------------------------------------------
# Predictions as versions
# ---------------------------------------------------------------------------

def _prediction(directory: Path, name: str, value: int, model: str, run: str) -> Path:
    """Write a prediction and the provenance naming the model that made it."""
    import yaml

    path = _write(directory / name, value)
    record = directory / "am_seg_predictions.yaml"
    existing = {}
    if record.is_file():
        existing = (yaml.safe_load(record.read_text()) or {}).get("predictions", {})
    existing[name] = {
        "model": model, "run": run,
        "predicted_at": "2026-09-02T11:04:00+01:00",
        "image": name,
    }
    record.write_text(yaml.safe_dump({"predictions": existing}))
    return path


def test_two_models_predictions_both_appear_named_by_model(label, recorder, tmp_path):
    first = _prediction(tmp_path / "sem_predictions" / "am_seg", "img_000.tif", 1, "fpn_v6", "run-6")
    second = _prediction(tmp_path / "sem_predictions" / "raw", "img_000.tif", 3, "fpn_v4", "run-4")

    controller = VersionController(Attribution(recorder))
    controller.set_label(label, [first, second])

    predicted = [row for row in controller.rows if row.source == "prediction"]
    assert [row.who for row in predicted] == ["fpn_v6", "fpn_v4"]
    assert [row.produced_by for row in predicted] == ["run run-6", "run run-4"]


def test_a_prediction_the_pipeline_post_processed_is_not_offered(label, recorder, tmp_path):
    clean = _write(tmp_path / "sem_predictions" / "clean" / "img_000.tif", 1)

    controller = VersionController(Attribution(recorder))
    controller.set_label(label, [clean])

    assert not [row for row in controller.rows if row.source == "prediction"]


def test_a_prediction_is_listed_beside_the_kept_versions(label, recorder, tmp_path):
    prediction = _prediction(
        tmp_path / "sem_predictions" / "am_seg", "img_000.tif", 1, "fpn_v6", "run-6"
    )

    controller = VersionController(Attribution(recorder))
    controller.set_label(label, [prediction])

    sources = [row.source for row in controller.rows]
    assert sources == ["kept", "prediction"], "kept versions come first"
    assert controller.count == 2


def test_comparing_against_a_prediction_writes_nothing(label, recorder, tmp_path):
    prediction = _prediction(
        tmp_path / "sem_predictions" / "am_seg", "img_000.tif", 1, "fpn_v6", "run-6"
    )
    before = {
        path: fingerprint_of(path)
        for path in sorted(tmp_path.rglob("*")) if path.is_file()
    }

    controller = VersionController(Attribution(recorder))
    controller.set_label(label, [prediction])
    controller.select([row.source for row in controller.rows].index("prediction"))
    assert controller.comparison_pixels() is not None
    assert controller.difference(_pixels(3)) is not None

    after = {
        path: fingerprint_of(path)
        for path in sorted(tmp_path.rglob("*")) if path.is_file()
    }
    assert after == before, "comparing may not touch the tree"


def _binned_prediction(tmp_path: Path) -> Path:
    """A prediction at half the label's size, as a model writes one."""
    import tifffile

    path = _prediction(
        tmp_path / "sem_predictions" / "am_seg", "img_000.tif", 1, "fpn_v6", "run-6"
    )
    binned = np.zeros((8, 8), dtype=np.uint8)
    binned[1:3, 1:3] = 1
    tifffile.imwrite(path, binned)
    return path


def test_a_prediction_is_compared_as_converting_it_would_write(label, recorder, tmp_path):
    """The comparison is the label the prediction would become, not the model's grid."""
    from am_segmentation_manager.operations.predictions import as_converted

    prediction = _binned_prediction(tmp_path)
    controller = VersionController(Attribution(recorder))
    controller.set_label(label, [prediction])
    controller.select([row.source for row in controller.rows].index("prediction"))

    shown = controller.comparison_pixels((16, 16))

    import tifffile

    expected = as_converted(tifffile.imread(prediction), (16, 16), prediction)
    assert shown.shape == (16, 16)
    assert np.array_equal(shown, expected)
    assert (shown[2:6, 2:6] == 1).all(), "each model pixel covers two by two"


def test_differences_against_a_binned_prediction_are_drawn(label, recorder, tmp_path):
    prediction = _binned_prediction(tmp_path)
    controller = VersionController(Attribution(recorder))
    controller.set_label(label, [prediction])
    controller.select([row.source for row in controller.rows].index("prediction"))

    mask = controller.difference(_pixels(3))

    assert mask is not None and mask.shape == (16, 16)
    assert mask[2:6, 2:6].all() and mask.sum() == 16


def test_a_prediction_that_holds_no_class_indices_is_shown_as_read(
    label, recorder, tmp_path
):
    """Converting would refuse it, so there is no converted label to show."""
    import tifffile

    prediction = _binned_prediction(tmp_path)
    tifffile.imwrite(prediction, np.full((8, 8), 0.5, dtype=np.float32))
    controller = VersionController(Attribution(recorder))
    controller.set_label(label, [prediction])
    controller.select([row.source for row in controller.rows].index("prediction"))

    assert controller.comparison_pixels((16, 16)).shape == (8, 8)


def test_the_canvas_says_a_prediction_is_shown_as_converted(label, recorder, tmp_path):
    from am_segmentation_manager.gui.version_controller import comparison_badge

    prediction = _binned_prediction(tmp_path)
    controller = VersionController(Attribution(recorder))
    controller.set_label(label, [prediction])
    rows = {row.source: row for row in controller.rows}

    assert comparison_badge(rows["prediction"]) == (
        "showing the prediction as converted, from 2026-09-02 10:04"
    )
    assert comparison_badge(rows["kept"]).startswith("showing the version from ")


def test_a_prediction_is_converted_rather_than_restored(label, recorder, tmp_path):
    prediction = _prediction(
        tmp_path / "sem_predictions" / "am_seg", "img_000.tif", 1, "fpn_v6", "run-6"
    )

    controller = VersionController(Attribution(recorder))
    controller.set_label(label, [prediction])
    controller.select([row.source for row in controller.rows].index("prediction"))

    assert controller.selected.restorable is False
    with pytest.raises(ValueError, match="converting"):
        controller.restore_selected()


def test_a_prediction_that_will_not_open_is_listed_as_unavailable(label, recorder, tmp_path):
    directory = tmp_path / "sem_predictions" / "am_seg"
    prediction = _prediction(directory, "img_000.tif", 1, "fpn_v6", "run-6")
    prediction.unlink()

    controller = VersionController(Attribution(recorder))
    controller.set_label(label, [prediction])

    predicted = [row for row in controller.rows if row.source == "prediction"]
    assert predicted and not predicted[0].available


def test_a_prediction_with_no_provenance_says_so(label, recorder, tmp_path):
    loose = _write(tmp_path / "sem_predictions" / "raw" / "img_000.tif", 1)

    controller = VersionController(Attribution(recorder))
    controller.set_label(label, [loose])

    predicted = [row for row in controller.rows if row.source == "prediction"]
    assert predicted[0].who == "unrecorded model"


# ---------------------------------------------------------------------------
# What the frame path may cost
# ---------------------------------------------------------------------------

def test_only_this_label_s_own_archive_directory_is_read(label, recorder, tmp_path, monkeypatch):
    """Listing versions must not walk the archive: that is linear in the project."""
    from am_segmentation_manager.operations import archive

    other = tmp_path / "sem_labels" / "img_001.tif"
    _write(other, 2)
    _save(other, _pixels(4), recorder)

    monkeypatch.setattr(
        Path, "rglob", lambda self, pattern: pytest.fail("the archive was walked")
    )
    controller = VersionController(Attribution(recorder))
    controller.set_label(label)

    assert len(controller.rows) == 1, "the other label's version is not this label's"
    assert archive.version_directory(tmp_path, "sem_labels/img_000.tif").is_dir()


def test_a_label_whose_stem_prefixes_another_gets_only_its_own(tmp_path, recorder):
    start_history(tmp_path)
    short = _write(tmp_path / "sem_labels" / "img.tif", 2)
    longer = _write(tmp_path / "sem_labels" / "img_000.tif", 2)
    _save(short, _pixels(3), recorder)
    _save(longer, _pixels(3), recorder)

    controller = VersionController(Attribution(recorder))
    controller.set_label(short)

    assert len(controller.rows) == 1
    assert controller.rows[0].path.stem.startswith("img__")


def test_the_log_is_read_again_only_when_it_has_changed(label, recorder, tmp_path):
    from am_segmentation_manager.gui import version_controller as module

    reads = []
    original = module.read_log
    module.read_log = lambda root: (reads.append(root), original(root))[1]
    try:
        controller = VersionController(Attribution(recorder))
        controller.set_label(label)
        controller.set_label(tmp_path / "sem_labels" / "other.tif")
        controller.set_label(label)
        assert len(reads) == 1, "an unchanged log is not re-read"

        _save(label, _pixels(5), recorder)
        controller.refresh()
        assert len(reads) == 2, "an appended log is read again"
    finally:
        module.read_log = original


# ---------------------------------------------------------------------------
# The list reads down its own column
# ---------------------------------------------------------------------------

def test_a_prediction_takes_its_place_in_time_among_the_kept_versions(
    label, recorder, tmp_path
):
    """A prediction made this morning belongs above a version kept last week."""
    prediction = _prediction(
        tmp_path / "sem_predictions" / "am_seg", "img_000.tif", 1, "fpn_v6", "run-6"
    )
    import yaml

    record = prediction.parent / "am_seg_predictions.yaml"
    entry = yaml.safe_load(record.read_text())
    entry["predictions"]["img_000.tif"]["predicted_at"] = "2035-01-01T09:00:00+00:00"
    record.write_text(yaml.safe_dump(entry))

    controller = VersionController(Attribution(recorder))
    controller.set_label(label, [prediction])

    assert controller.rows[0].source == "prediction", "the newest thing is first"


def _newer_prediction(tmp_path: Path) -> Path:
    """A prediction made after every kept version, so it heads the list."""
    import yaml

    prediction = _prediction(
        tmp_path / "sem_predictions" / "am_seg", "img_000.tif", 1, "fpn_v6", "run-6"
    )
    record = prediction.parent / "am_seg_predictions.yaml"
    entry = yaml.safe_load(record.read_text())
    entry["predictions"]["img_000.tif"]["predicted_at"] = "2035-01-01T09:00:00+00:00"
    record.write_text(yaml.safe_dump(entry))
    return prediction


def test_a_comparison_starts_on_the_label_not_a_newer_prediction(
    label, recorder, tmp_path
):
    """What the label held is what is looked back at; a model's guess is a choice."""
    controller = VersionController(Attribution(recorder))
    controller.set_label(label, [_newer_prediction(tmp_path)])

    assert controller.rows[0].source == "prediction"
    assert controller.rows[controller.default_index].source == "kept"


def test_a_comparison_starts_on_a_prediction_only_with_no_kept_version(
    tmp_path: Path, recorder
):
    start_history(tmp_path)
    fresh = _write(tmp_path / "sem_labels" / "img_000.tif", 2)
    controller = VersionController(Attribution(recorder))
    controller.set_label(fresh, [_newer_prediction(tmp_path)])

    assert controller.rows[controller.default_index].source == "prediction"


def test_a_comparison_does_not_start_on_a_version_it_cannot_show(
    label, recorder, tmp_path
):
    from am_segmentation_manager.operations.archive import versions_of

    (kept,) = versions_of(label.parents[1], "sem_labels/img_000.tif")
    kept.path.unlink()
    controller = VersionController(Attribution(recorder))
    controller.set_label(label, [_newer_prediction(tmp_path)])

    assert controller.rows[controller.default_index].source == "prediction"


def test_a_label_with_no_versions_has_nothing_to_start_on(tmp_path: Path, recorder):
    start_history(tmp_path)
    controller = VersionController(Attribution(recorder))
    controller.set_label(_write(tmp_path / "sem_labels" / "new.tif", 1))

    assert controller.default_index == -1


def test_every_row_carries_its_date_in_the_same_place(label, recorder, tmp_path):
    """The date is what the list is ordered by, so it must be findable."""
    prediction = _prediction(
        tmp_path / "sem_predictions" / "am_seg", "img_000.tif", 1, "fpn_v6", "run-6"
    )
    controller = VersionController(Attribution(recorder))
    controller.set_label(label, [prediction])

    from am_segmentation_manager.gui.version_controller import KIND_WIDTH, WHEN_WIDTH

    start = KIND_WIDTH + 2
    columns = [row.describe()[start:start + WHEN_WIDTH] for row in controller.rows]

    assert len(columns) == 2
    assert all(column.strip().count("-") == 2 for column in columns), columns


def test_the_list_is_ordered_newest_first_across_both_clocks(label, recorder, tmp_path):
    """The log writes UTC and a prediction writes local time with its offset."""
    prediction = _prediction(
        tmp_path / "sem_predictions" / "am_seg", "img_000.tif", 1, "fpn_v6", "run-6"
    )
    controller = VersionController(Attribution(recorder))
    controller.set_label(label, [prediction])

    from am_segmentation_manager.gui.version_controller import KIND_WIDTH, WHEN_WIDTH

    start = KIND_WIDTH + 2
    shown = [row.describe()[start:start + WHEN_WIDTH] for row in controller.rows]

    assert shown == sorted(shown, reverse=True)


def test_a_long_training_run_name_is_shortened_at_both_ends(label, recorder, tmp_path):
    run = "Franklin_eBIC_Rosalind_training_absolute_validation_absolute_epoch086"
    prediction = _prediction(
        tmp_path / "sem_predictions" / "am_seg", "img_000.tif", 1, "fpn_v6", run
    )
    controller = VersionController(Attribution(recorder))
    controller.set_label(label, [prediction])

    (row,) = [row for row in controller.rows if row.source == "prediction"]
    line = row.describe()

    assert run not in line, "the whole name would push the rest off the row"
    assert "Franklin" in line and "epoch086" in line, "both ends still identify it"


def test_a_version_whose_change_was_never_measured_says_so(label, recorder):
    """'unknown changed' read as a quantity; it is the absence of one."""
    controller = VersionController(Attribution(recorder))
    controller.set_label(label)
    row = controller.rows[0]

    from dataclasses import replace

    assert "change unrecorded" in replace(row, changed=None).describe()


def test_a_correction_made_in_passing_does_not_read_as_no_change(label, recorder):
    """A few hundred pixels of a few million rounds to 0.0%, which is a lie."""
    from dataclasses import replace

    controller = VersionController(Attribution(recorder))
    controller.set_label(label)
    row = controller.rows[0]

    assert "<0.1% changed" in replace(row, changed=0.00002).describe()
    assert "0.0% changed" in replace(row, changed=0.0).describe(), "nothing is nothing"
