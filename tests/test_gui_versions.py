"""Tests for the earlier-versions strip in the proofreading view.

The controller is tested without a display in ``test_version_controller.py``;
what is left here is what only Qt can answer. Mostly that the strip is not there
until somebody opens it, that looking at an old version cannot be painted on,
and that choosing another one stops showing the last.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

# Must be set before any Qt module is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HAS_QT = importlib.util.find_spec("qtpy") is not None
HAS_IMAGES = all(
    importlib.util.find_spec(name) is not None for name in ("numpy", "tifffile")
)

pytestmark = pytest.mark.skipif(
    not (HAS_QT and HAS_IMAGES), reason="Qt binding or image libs not installed"
)

if HAS_QT and HAS_IMAGES:
    import numpy as np

    from am_segmentation_manager.config import SegmentationConfig
    from am_segmentation_manager.gui.attribution import Attribution
    from am_segmentation_manager.gui.label_controller import LabelController
    from am_segmentation_manager.gui.version_controller import VersionController, VersionRow
    from am_segmentation_manager.gui.widgets.label_panel import LabelPanel
    from am_segmentation_manager.gui.widgets.versions_bar import (
        DIFFERENCE_CLASS,
        VersionsBar,
    )
    from am_segmentation_manager.history import Recorder
    from am_segmentation_manager.io.atomic import fingerprint_of
    from am_segmentation_manager.operations.label_save import LabelToSave, save_labels
    from conftest import destroy, start_history


@pytest.fixture
def config():
    return SegmentationConfig(
        class_colors={0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 128, 0), 3: (0, 0, 255)},
        class_labels={0: "background", 1: "GIS", 2: "lamella", 3: "crack/hole"},
    )


@pytest.fixture
def recorder() -> Recorder:
    """A recorder attributing everything to a named author."""
    return Recorder(by="casper", by_source="flag")


@pytest.fixture
def label(tmp_path: Path, recorder) -> Path:
    """A label with one kept version, in a project that keeps a history."""
    start_history(tmp_path)
    path = tmp_path / "sem_labels" / "img_000.tif"
    path.parent.mkdir(parents=True)
    for value in (2, 3):
        save_labels(
            [
                LabelToSave(
                    destination=path, label=_pixels(value),
                    fingerprint=fingerprint_of(path),
                )
            ],
            recorder=recorder,
        )
    return path


def _pixels(value: int):
    array = np.zeros((16, 16), dtype=np.uint8)
    array[2:6, 2:6] = value
    return array


def _write(path: Path, value: int) -> Path:
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, _pixels(value))
    return path


def _rows(label: Path, recorder: Recorder):
    controller = VersionController(Attribution(recorder))
    controller.set_label(label)
    return controller


# ---------------------------------------------------------------------------
# The strip
# ---------------------------------------------------------------------------

def test_the_strip_is_not_there_until_it_is_opened(qt_app, label, recorder):
    """A control used a few times a year does not sit in a view read every day."""
    bar = VersionsBar()
    try:
        assert not bar.isVisibleTo(bar)

        bar.set_rows(_rows(label, recorder).rows)
        assert not bar.isVisibleTo(bar), "listing versions does not open the strip"

        bar.set_open(True)
        assert bar.isVisibleTo(bar)
    finally:
        destroy(bar)


def test_closing_the_strip_stops_showing_the_version(qt_app, label, recorder):
    """Hiding the strip while a version is up would leave the wrong picture on."""
    bar = VersionsBar()
    shown = []
    try:
        bar.set_rows(_rows(label, recorder).rows)
        bar.set_open(True)
        bar.comparing.connect(shown.append)
        bar.differences.connect(shown.append)
        bar.set_comparing(True)

        bar.set_open(False)

        assert not bar.compare_button.isChecked()
        assert shown[-1] is False
    finally:
        destroy(bar)


def test_a_prediction_is_offered_to_compare_but_not_to_restore(qt_app):
    bar = VersionsBar()
    try:
        bar.set_available(
            True,
            refusal=VersionRow(
                Path("p.tif"), "prediction", "", "", "", None, True
            ).not_restorable_because,
        )

        assert bar.compare_button.isEnabled()
        assert bar.difference_check.isEnabled()
        assert not bar.restore_button.isEnabled()
        assert "converted" in bar.restore_button.toolTip()
    finally:
        destroy(bar)


def test_the_strip_lists_what_is_kept(qt_app, label, recorder):
    bar = VersionsBar()
    try:
        bar.set_rows(_rows(label, recorder).rows)

        assert bar.chooser.count() == 1
        assert "casper" in bar.chooser.itemText(0)
        assert "changed" in bar.chooser.itemText(0)
    finally:
        destroy(bar)


def test_the_strip_starts_on_the_row_it_is_given(qt_app, label, recorder, tmp_path):
    prediction = tmp_path / "sem_predictions" / "am_seg" / "img_000.tif"
    _write(prediction, 1)
    controller = VersionController(Attribution(recorder))
    controller.set_label(label, [prediction])
    bar = VersionsBar()
    chosen = []
    try:
        bar.selected.connect(chosen.append)
        bar.set_rows(controller.rows, 1)

        assert bar.chosen == 1 and chosen == [1]
    finally:
        destroy(bar)


def test_holding_b_starts_on_the_label_not_a_newer_prediction(
    qt_app, config, label, recorder, tmp_path, monkeypatch
):
    """A prediction made after the last save heads the list; B still shows the label."""
    import yaml

    prediction = _write(tmp_path / "sem_predictions" / "am_seg" / "img_000.tif", 1)
    (prediction.parent / "am_seg_predictions.yaml").write_text(yaml.safe_dump({
        "predictions": {"img_000.tif": {"predicted_at": "2035-01-01T09:00:00+00:00"}}
    }))
    panel = LabelPanel(LabelController(config=config), VersionController(Attribution(recorder)))
    try:
        monkeypatch.setattr(LabelController, "current_key", property(lambda self: label))
        monkeypatch.setattr(
            LabelController, "record",
            property(lambda self: type("Record", (), {"predictions": (prediction,)})()),
        )
        panel.refresh_versions()
        assert panel.versions_controller.rows[0].source == "prediction"

        panel._compare_key_held(True)

        assert panel.versions_controller.selected.source == "kept"
        assert np.array_equal(panel._label_to_draw(), _pixels(2))
    finally:
        destroy(panel)


def test_choosing_another_version_stops_showing_the_last(qt_app, label, recorder):
    bar = VersionsBar()
    seen = []
    try:
        bar.set_rows(_rows(label, recorder).rows)
        bar.comparing.connect(seen.append)
        bar.compare_button.setChecked(True)

        bar.set_rows(_rows(label, recorder).rows)

        assert not bar.compare_button.isChecked()
        assert not bar.difference_check.isChecked()
    finally:
        destroy(bar)


def test_a_version_that_is_not_here_cannot_be_looked_at_or_put_back(qt_app):
    bar = VersionsBar()
    try:
        bar.set_available(False)

        assert not bar.compare_button.isEnabled()
        assert not bar.restore_button.isEnabled()
        assert not bar.difference_check.isEnabled()
    finally:
        destroy(bar)


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

def test_the_panel_draws_the_chosen_version_while_comparing(
    qt_app, config, label, recorder
):
    panel = LabelPanel(LabelController(config=config), VersionController(Attribution(recorder)))
    try:
        panel.versions_controller.set_label(label)
        panel.versions.set_rows(panel.versions_controller.rows)
        panel.versions_controller.select(0)

        panel._set_comparing(True)
        drawn = panel._label_to_draw()

        assert drawn is not None and np.array_equal(drawn, _pixels(2))
    finally:
        destroy(panel)


def test_the_panel_marks_only_the_pixels_that_differ(
    qt_app, config, label, recorder, monkeypatch
):
    panel = LabelPanel(LabelController(config=config), VersionController(Attribution(recorder)))
    try:
        panel.versions_controller.set_label(label)
        panel.versions_controller.select(0)
        # The label being proofread, which no frame is loaded to supply here.
        monkeypatch.setattr(panel.controller, "working_label", lambda: _pixels(3))
        panel._showing_differences = True

        drawn = panel._label_to_draw()

        assert set(np.unique(drawn)) == {0, DIFFERENCE_CLASS}
        assert (drawn == DIFFERENCE_CLASS).sum() == 16
        assert (drawn[2:6, 2:6] == DIFFERENCE_CLASS).all()
    finally:
        destroy(panel)


def test_the_panel_draws_a_prediction_at_the_labels_size(
    qt_app, config, label, recorder, tmp_path, monkeypatch
):
    """A model predicts on a binned image; what is drawn is the label it becomes."""
    import tifffile

    prediction = tmp_path / "sem_predictions" / "am_seg" / "img_000.tif"
    prediction.parent.mkdir(parents=True)
    tifffile.imwrite(prediction, np.ones((8, 8), dtype=np.uint8))
    panel = LabelPanel(LabelController(config=config), VersionController(Attribution(recorder)))
    try:
        panel.versions_controller.set_label(label, [prediction])
        sources = [row.source for row in panel.versions_controller.rows]
        panel.versions_controller.select(sources.index("prediction"))
        monkeypatch.setattr(panel.controller, "working_label", lambda: _pixels(3))

        panel._set_comparing(True)
        drawn = panel._label_to_draw()

        assert drawn.shape == (16, 16) and (drawn == 1).all()
    finally:
        destroy(panel)


def test_the_difference_class_is_one_no_configuration_defines(config):
    """So it draws in the colour reserved for a class nothing defines, and what
    is on screen is plainly not a segmentation class."""
    assert DIFFERENCE_CLASS not in config.class_colors


def test_an_earlier_version_on_screen_cannot_be_painted_on(qt_app, config, recorder):
    """A stroke would be recorded against the label underneath, unseen."""
    panel = LabelPanel(LabelController(config=config), VersionController(Attribution(recorder)))
    try:
        panel._comparing = True

        assert panel.showing_a_version

        panel._comparing = False
        panel._showing_differences = True

        assert panel.showing_a_version
    finally:
        destroy(panel)


def test_the_compare_key_does_nothing_without_a_version_chosen(
    qt_app, config, recorder
):
    panel = LabelPanel(LabelController(config=config), VersionController(Attribution(recorder)))
    try:
        panel._compare_key_held(True)

        assert not panel.versions.compare_button.isChecked()
    finally:
        destroy(panel)


def test_the_panel_asks_the_window_to_restore(qt_app, config, recorder):
    """Restoring is a write, so the panel asks and the window confirms."""
    panel = LabelPanel(LabelController(config=config), VersionController(Attribution(recorder)))
    asked = []
    try:
        panel.restore_requested.connect(lambda: asked.append(True))
        panel.versions.restore_button.click()

        assert asked == [True]
    finally:
        destroy(panel)


def test_the_switch_opens_the_strip_and_says_how_many_there_are(
    qt_app, config, label, recorder
):
    """The count is what makes the switch worth reading while the strip is shut."""
    panel = LabelPanel(LabelController(config=config), VersionController(Attribution(recorder)))
    try:
        panel.versions_controller.set_label(label)
        panel.versions.set_rows(panel.versions_controller.rows)
        panel._refresh_versions_switch()

        assert not panel.versions.isVisibleTo(panel)
        assert panel.versions_check.text() == "Compare versions (1)"

        panel.versions_check.setChecked(True)
        assert panel.versions.isVisibleTo(panel)
    finally:
        destroy(panel)


def test_closing_the_switch_puts_the_label_back_on_the_canvas(
    qt_app, config, label, recorder, monkeypatch
):
    panel = LabelPanel(LabelController(config=config), VersionController(Attribution(recorder)))
    try:
        panel.versions_controller.set_label(label)
        panel.versions.set_rows(panel.versions_controller.rows)
        panel.versions_controller.select(0)
        panel.versions_check.setChecked(True)
        monkeypatch.setattr(panel.controller, "working_label", lambda: _pixels(3))
        panel._set_comparing(True)
        assert np.array_equal(panel._label_to_draw(), _pixels(2))

        panel.versions_check.setChecked(False)

        assert np.array_equal(panel._label_to_draw(), _pixels(3))
    finally:
        destroy(panel)


def test_a_label_with_nothing_to_compare_says_so_on_the_switch(qt_app, config, recorder):
    panel = LabelPanel(LabelController(config=config), VersionController(Attribution(recorder)))
    try:
        panel._refresh_versions_switch()
        assert panel.versions_check.text() == "Compare versions"
    finally:
        destroy(panel)
