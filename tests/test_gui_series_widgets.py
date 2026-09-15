"""Widget-level tests for the series control in Curate.

Drives the real widgets offscreen against a 2026 project, whose lamella holds
reference and alignment images around its Adaptive Polishing stage. What the
design turns on is pinned here: the control is always on screen, greyed where
the lamella holds one series, and choosing one replaces the timeline.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from conftest import destroy, settle

# Must be set before any Qt module is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HAS_QT = importlib.util.find_spec("qtpy") is not None
HAS_IMAGE_STACK = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "tifffile", "cv2")
)

pytestmark = pytest.mark.skipif(
    not (HAS_QT and HAS_IMAGE_STACK),
    reason="Qt binding or optional image libs not installed",
)

if HAS_QT and HAS_IMAGE_STACK:
    import numpy as np
    import tifffile
    import yaml
    from qtpy.QtCore import QSettings
    from qtpy.QtWidgets import QApplication

    from am_segmentation_manager.config import SegmentationConfig
    from am_segmentation_manager.gui.cache import FrameCache
    from am_segmentation_manager.gui.controller import CurationController
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow

SEM_VARIANTS = ("SEM_8", "SEM_8_fi_1", "SEM_8_fi_4")


@pytest.fixture(scope="session")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def modern_project(tmp_path: Path) -> Path:
    """A 2026 project: reference and alignment images around one stage."""
    root = tmp_path / "scan"
    project = root / "AutoLamella-2026-07-28-10-31"
    lamella = project / "01-ruling-ibex"
    image = np.full((64, 96), 100, dtype=np.uint8)

    for name in (
        "ref_Polishing_start_eb.tif",
        "ref_Polishing_start_ib.tif",
        "ref_alignment_ib.tif",
    ):
        _write(lamella / name, image)

    stage = lamella / "Milling" / "Polishing" / "adaptive_polish_2026-07-28-11-34-24AM"
    for number in range(2):
        _write(stage / "sem" / f"img_{number:03d}_SEM.tif", image)
        _write(stage / "fib" / f"img_{number:03d}_FIB.tif", image)

    project.mkdir(parents=True, exist_ok=True)
    (project / "experiment.yaml").write_text("name: bitmap\n", encoding="utf-8")
    return root


@pytest.fixture
def mixed_project(tmp_path: Path) -> Path:
    """Two lamellae: one with an Adaptive Polishing stage, one without."""
    root = tmp_path / "scan"
    project = root / "AutoLamella-2026-07-28-10-31"
    image = np.full((64, 96), 100, dtype=np.uint8)

    polished = project / "01-ruling-ibex"
    _write(polished / "ref_Polishing_start_eb.tif", image)
    stage = polished / "Milling" / "Polishing" / "adaptive_polish_2026-07-28-11-34-24AM"
    _write(stage / "sem" / "img_000_SEM.tif", image)

    unpolished = project / "02-ruling-oryx"
    _write(unpolished / "ref_Polishing_start_eb.tif", image)
    _write(unpolished / "ref_alignment_ib.tif", image)

    (project / "experiment.yaml").write_text("name: mixed\n", encoding="utf-8")
    return root


@pytest.fixture
def grid_project(tmp_path: Path) -> Path:
    """A project with a grid overview at its root and two lamellae.

    The overview is written with the header Autolamella gives one, so the
    geometry is read the same way it is off real data; the two positions are
    a little either side of the centre. `02-ruling-oryx` is deliberately left
    out of `positions`, which is the real case of a lamella directory the
    experiment never recorded a position for.
    """
    root = tmp_path / "scan"
    project = root / "AutoLamella-2026-07-28-10-31"
    image = np.full((64, 96), 100, dtype=np.uint8)

    for name in ("01-ruling-ibex", "02-ruling-oryx"):
        stage = project / name / "adaptive_polish_2026-07-28-11-34-24AM"
        _write(stage / "sem" / "img_000_SEM.tif", image)

    described = json.dumps({
        "image": {"beam_type": "ELECTRON", "resolution": [256, 256]},
        "pixel_size": {"x": 1e-6, "y": 1e-6},
        "system": {
            "info": {"model": "Aquilos"},
            "stage": {"shuttle_pre_tilt": 0.0, "rotation_reference": 0.0, "rotation_180": 180.0},
            "electron": {"column_tilt": 0.0},
            "ion": {"column_tilt": 52.0},
        },
        "microscope_state": {
            "stage_position": {"x": 0.0, "y": 0.0, "z": 0.0, "r": 0.0, "t": 0.0},
            "electron_beam": {"scan_rotation": 0.0},
        },
    })
    overview = project / "overview-image-2026-07-28_10-31-00.tif"
    overview.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(overview), np.full((256, 256), 40, dtype=np.uint8), description=described
    )

    (project / "experiment.yaml").write_text(
        yaml.safe_dump({
            "name": "grid",
            "positions": [{
                "petname": "01-ruling-ibex",
                "state": {"microscope_state": {"stage_position": {
                    "x": 20e-6, "y": 0.0, "z": 0.0, "r": 0.0, "t": 0.0,
                }}},
            }],
        }),
        encoding="utf-8",
    )
    return root


@pytest.fixture
def residue_project(tmp_path: Path) -> Path:
    """Two lamellae: one Adaptive Polishing, one holding only unrecognised names.

    The second is what a grid overview, a diagnostic or an Autolamella version
    naming something new looks like once it is indexed.
    """
    root = tmp_path / "scan"
    project = root / "AutoLamella-2026-07-28-10-31"
    image = np.full((64, 96), 100, dtype=np.uint8)

    stage = (
        project / "01-ruling-ibex" / "Milling" / "Polishing"
        / "adaptive_polish_2026-07-28-11-34-24AM"
    )
    _write(stage / "sem" / "img_000_SEM.tif", image)
    _write(project / "02-ruling-oryx" / "something_new_eb.tif", image)

    (project / "experiment.yaml").write_text("name: residue\n", encoding="utf-8")
    return root


@pytest.fixture
def plain_project(tmp_path: Path) -> Path:
    """An ordinary Autolamella project: Adaptive Polishing images only."""
    root = tmp_path / "scan"
    stage = root / "AutoLamella-2026" / "01-alpha" / "adaptive_polish_2026-07-28-11-15-36AM"
    image = np.full((64, 96), 100, dtype=np.uint8)
    for number in range(2):
        _write(stage / "sem" / f"img_{number:03d}_SEM.tif", image)
        _write(stage / "fib" / f"img_{number:03d}_FIB.tif", image)
    (root / "AutoLamella-2026").mkdir(parents=True, exist_ok=True)
    (root / "AutoLamella-2026" / "experiment.yaml").write_text("name: plain\n")
    return root


def _write(path: Path, array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), array)


def _show_other(window) -> None:
    """Switch `other` on, which the lists open without."""
    window.project_panel.shown_boxes["other"].setChecked(True)


def _window(qt_app, root: Path) -> CurationWindow:
    config = SegmentationConfig(
        class_colors={0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 128, 0)},
        class_labels={0: "background", 1: "GIS", 2: "lamella"},
    )
    # Preferences into the scan folder, which is a tmp_path: the default is the
    # real window's, and a test must not read or overwrite that.
    window = CurationWindow(
        CurationController(cache=FrameCache(), config=config),
        settings=QSettings(str(root / "preferences.ini"), QSettings.IniFormat),
    )
    window.scan(str(root))
    settle(
        qt_app,
        lambda: bool(window.controller.frames)
        and window.image_panel.series_combo.count() > 0,
        what="the scan to reach the series control",
    )
    return window


# ---------------------------------------------------------------------------
# The control stays put, and is greyed out without a choice
# ---------------------------------------------------------------------------

def test_the_series_control_is_greyed_without_a_choice(qt_app, plain_project: Path):
    """A project of Adaptive Polishing images alone has nothing to choose --
    but the control still says what is being shown, and the row beside it does
    not move as the reviewer steps onto such a lamella."""
    window = _window(qt_app, plain_project)
    try:
        # isHidden() rather than isVisible(): a child of a window that was
        # never shown is never "visible", so isVisible() would pass here
        # whether or not the control was actually hidden.
        combo = window.image_panel.series_combo
        assert combo.isHidden() is False
        assert window.image_panel.series_label.isHidden() is False
        assert combo.isEnabled() is False
        assert combo.currentText() == "Adaptive Polishing"
    finally:
        destroy(window)


def test_the_series_control_opens_when_a_lamella_holds_more(qt_app, modern_project):
    window = _window(qt_app, modern_project)
    try:
        combo = window.image_panel.series_combo
        assert combo.isEnabled() is True
        offered = [combo.itemData(i) for i in range(combo.count())]
        assert offered == ["adaptive", "reference", "alignment"]
    finally:
        destroy(window)


def test_the_control_shows_the_series_by_name(qt_app, modern_project: Path):
    window = _window(qt_app, modern_project)
    try:
        combo = window.image_panel.series_combo
        captions = [combo.itemText(i) for i in range(combo.count())]
        assert captions == ["Adaptive Polishing", "reference", "alignment"]
    finally:
        destroy(window)


# ---------------------------------------------------------------------------
# A lamella holding none of the chosen series
# ---------------------------------------------------------------------------

def test_a_lamella_without_the_chosen_series_shows_one_it_holds(
    qt_app, mixed_project: Path
):
    """The report this came from: with Adaptive Polishing chosen, a lamella
    that has none showed an empty timeline and -- the control being hidden --
    offered no way back to the series it does hold."""
    window = _window(qt_app, mixed_project)
    try:
        window._step_lamella(1)
        settle(qt_app, lambda: window.controller.series == "reference",
               what="the lamella without Adaptive Polishing to fall back")

        combo = window.image_panel.series_combo
        assert window.controller.series == "reference"
        assert len(window.controller.frames) == 1
        assert combo.currentText() == "reference"
        assert combo.isEnabled() is True, "alignment is still one click away"
    finally:
        destroy(window)


def test_stepping_back_returns_to_the_chosen_series(qt_app, mixed_project: Path):
    """The fallback answers about one lamella rather than changing the choice."""
    window = _window(qt_app, mixed_project)
    try:
        window._step_lamella(1)
        settle(qt_app, lambda: window.controller.series == "reference",
               what="the fallback to the series that lamella holds")
        window._step_lamella(-1)
        settle(qt_app, lambda: window.controller.series == "adaptive",
               what="the chosen series to come back")

        assert window.controller.series == "adaptive"
        assert window.image_panel.series_combo.currentText() == "Adaptive Polishing"
    finally:
        destroy(window)


# ---------------------------------------------------------------------------
# Choosing one replaces the timeline
# ---------------------------------------------------------------------------

def test_the_timeline_opens_on_adaptive_polishing(qt_app, modern_project: Path):
    """So the tab shows what it always showed."""
    window = _window(qt_app, modern_project)
    try:
        assert window.controller.series == "adaptive"
        assert len(window.controller.frames) == 2
    finally:
        destroy(window)


def test_choosing_a_series_rebuilds_the_timeline(qt_app, modern_project: Path):
    window = _window(qt_app, modern_project)
    try:
        combo = window.image_panel.series_combo
        combo.setCurrentIndex(combo.findData("reference"))
        settle(qt_app, lambda: window.controller.series == "reference",
               what="the timeline to be rebuilt")

        assert window.controller.series == "reference"
        assert len(window.controller.frames) == 1
        assert window.controller.frames[0].series == "reference"
    finally:
        destroy(window)


def test_the_beam_buttons_follow_the_chosen_series(qt_app, modern_project: Path):
    """The alignment run holds an ion-beam image alone, so the view moves to
    it -- and the buttons have to say which view that is."""
    window = _window(qt_app, modern_project)
    try:
        assert window.image_panel.mode_buttons["sem"].isChecked() is True

        combo = window.image_panel.series_combo
        combo.setCurrentIndex(combo.findData("alignment"))
        settle(qt_app, lambda: window.controller.target == "FIB",
               what="the view to move to the ion beam")

        assert window.image_panel.mode_buttons["fib"].isChecked() is True
        assert window.controller.target == "FIB"
    finally:
        destroy(window)


def test_the_model_box_follows_the_chosen_series(qt_app, modern_project: Path):
    """A model trained on Adaptive Polishing frames has not seen a reference
    image of the whole lamella, so offering it there is offering the answer to
    a different question."""
    window = _window(qt_app, modern_project)
    try:
        window.operations_panel.set_model("/models/adaptive_sem.pth")

        combo = window.image_panel.series_combo
        combo.setCurrentIndex(combo.findData("reference"))
        settle(qt_app, lambda: window.controller.series == "reference",
               what="the series the model was not trained on")
        assert window.operations_panel.model() == ""

        combo.setCurrentIndex(combo.findData("adaptive"))
        settle(qt_app, lambda: window.controller.series == "adaptive",
               what="the series the model was trained on")
        assert window.operations_panel.model() == "/models/adaptive_sem.pth"
    finally:
        destroy(window)


def test_the_frame_readout_follows_the_chosen_series(qt_app, modern_project: Path):
    window = _window(qt_app, modern_project)
    try:
        assert window.frame_label.text() == "1 / 2"

        combo = window.image_panel.series_combo
        combo.setCurrentIndex(combo.findData("alignment"))
        settle(qt_app, lambda: window.controller.series == "alignment",
               what="the timeline to be rebuilt")

        assert window.frame_label.text() == "1 / 1"
    finally:
        destroy(window)


# ---------------------------------------------------------------------------
# A series no operation can act on
# ---------------------------------------------------------------------------

def test_the_controls_that_change_data_are_offered_for_a_labelled_series(
    qt_app, residue_project: Path
):
    window = _window(qt_app, residue_project)
    try:
        _show_other(window)
        window.controller.set_editing_enabled(True)
        window.state_panel.set_editing_enabled(True)
        window.refresh()

        assert window.controller.series == "adaptive"
        assert window.state_panel.state_box.isEnabled()
        assert window.operations_panel.specks_button.isEnabled()
    finally:
        destroy(window)


def test_the_controls_that_change_data_are_withheld_for_the_residue(
    qt_app, residue_project: Path
):
    """A button that would answer "nothing selected" over a full timeline is
    worse than one plainly unavailable."""
    window = _window(qt_app, residue_project)
    try:
        _show_other(window)
        window.controller.set_editing_enabled(True)
        window.state_panel.set_editing_enabled(True)
        window.controller.select_lamella(1)
        window.refresh()

        assert window.controller.series == "other"
        assert not window.state_panel.state_box.isEnabled()
        assert not window.operations_panel.specks_button.isEnabled()
        assert not window.operations_panel.convert_button.isEnabled()
    finally:
        destroy(window)


def test_the_withheld_controls_say_why(qt_app, residue_project: Path):
    window = _window(qt_app, residue_project)
    try:
        _show_other(window)
        window.controller.select_lamella(1)
        window.refresh()

        assert "no labels" in window.operations_panel.box.toolTip()
        assert "no labels" in window.state_panel.state_box.toolTip()
    finally:
        destroy(window)


def test_switching_editing_back_on_does_not_unlock_the_residue(
    qt_app, residue_project: Path
):
    """Two independent reasons, so lifting one may not lift the other."""
    window = _window(qt_app, residue_project)
    try:
        _show_other(window)
        window.controller.select_lamella(1)
        window.refresh()
        window.state_panel.set_editing_enabled(True)

        assert not window.state_panel.state_box.isEnabled()
    finally:
        destroy(window)


def test_the_timeline_still_shows_the_residue(qt_app, residue_project: Path):
    """Withholding the controls must not withhold the images: being able to
    open one and decide whether the rule or the data is wrong is the point."""
    window = _window(qt_app, residue_project)
    try:
        _show_other(window)
        window.controller.select_lamella(1)
        window.refresh()

        assert [frame.series for frame in window.controller.frames] == ["other"]
    finally:
        destroy(window)


def test_leaving_the_residue_offers_the_controls_again(qt_app, residue_project: Path):
    window = _window(qt_app, residue_project)
    try:
        _show_other(window)
        window.controller.set_editing_enabled(True)
        window.state_panel.set_editing_enabled(True)
        window.controller.select_lamella(1)
        window.refresh()
        window.controller.select_lamella(0)
        window.refresh()

        assert window.state_panel.state_box.isEnabled()
        assert window.operations_panel.specks_button.isEnabled()
        assert window.operations_panel.box.toolTip() == ""
    finally:
        destroy(window)


def test_the_residue_is_badged_view_only_rather_than_unlabelled(
    qt_app, residue_project: Path
):
    """"no label" reports a fault, and an `other` image has nowhere a label
    could go: it could never be otherwise."""
    window = _window(qt_app, residue_project)
    try:
        _show_other(window)
        window.controller.select_lamella(1)
        window.refresh()
        settle(
            qt_app,
            lambda: "view only" in window.image_panel.badges.text(),
            what="the residue frame to be drawn",
        )

        assert "no label" not in window.image_panel.badges.text()
    finally:
        destroy(window)


def test_the_label_tab_opens_on_the_residue(qt_app, residue_project: Path):
    """The Label tab asks where a label lives on every refresh. An `other`
    image has nowhere, and the tab must still open on it, describe it rather
    than the lamella before it, and refuse a stroke."""
    window = _window(qt_app, residue_project)
    try:
        _show_other(window)
        window.controller.select_lamella(1)
        window.refresh()
        window.tabs.setCurrentWidget(window.label_panel)
        window.label_panel.refresh()
        settle(
            qt_app,
            lambda: "view only" in window.label_panel.badges.text(),
            what="the residue frame to be drawn in the Label tab",
        )

        assert window.label_panel.controller.series == "other"
        assert "02-ruling-oryx" in window.label_panel.status.text()
        assert not window.label_panel.canvas.editable
    finally:
        destroy(window)


# ---------------------------------------------------------------------------
# Series switched off
# ---------------------------------------------------------------------------

def test_opening_the_window_raises_nothing(qt_app, tmp_path: Path, monkeypatch):
    """A slot's exception is printed rather than raised, so a failure while
    the window is being built -- where the switches first announce what they
    show -- passes every test that only looks at the result."""
    import sys

    caught = []
    monkeypatch.setattr(sys, "excepthook", lambda *info: caught.append(info))
    window = CurationWindow(
        CurationController(cache=FrameCache()),
        settings=QSettings(str(tmp_path / "preferences.ini"), QSettings.IniFormat),
    )
    try:
        assert not caught, caught[0][1] if caught else None
    finally:
        destroy(window)


def test_the_residue_opens_switched_off_and_counted(qt_app, residue_project: Path):
    """`other` is worth being able to open and not worth listing beside the
    lamellae; leaving it out is said in the title, or it would be silent."""
    window = _window(qt_app, residue_project)
    try:
        assert not window.project_panel.shown_boxes["other"].isChecked()
        assert [lamella.name for lamella in window.controller.lamellae] == [
            "01-ruling-ibex"
        ]
        assert window.project_panel.lamella_box.title() == "Lamellae (1, 1 hidden)"
    finally:
        destroy(window)


def test_switching_a_series_on_lists_its_lamellae(qt_app, residue_project: Path):
    window = _window(qt_app, residue_project)
    try:
        _show_other(window)

        assert len(window.controller.lamellae) == 2
        assert window.project_panel.lamella_box.title() == "Lamellae (2)"
    finally:
        destroy(window)


def test_the_chip_has_a_column_per_series_switched_on(qt_app, residue_project: Path):
    from am_segmentation_manager.gui.widgets.theme import swatch_width

    window = _window(qt_app, residue_project)
    try:
        assert window.project_panel.lamella_list.iconSize().width() == swatch_width(3)

        _show_other(window)

        assert window.project_panel.lamella_list.iconSize().width() == swatch_width(4)
        assert window.project_panel.project_list.iconSize().width() == swatch_width(4)
    finally:
        destroy(window)


def test_the_last_series_cannot_be_switched_off(qt_app, plain_project: Path):
    """None at all would list nothing, which reads as a fault."""
    window = _window(qt_app, plain_project)
    try:
        boxes = window.project_panel.shown_boxes
        for name in ("reference", "alignment", "adaptive"):
            boxes[name].setChecked(False)

        assert boxes["adaptive"].isChecked()
        assert window.controller.shown_series == ("adaptive",)
    finally:
        destroy(window)


def test_the_series_menus_offer_only_what_is_switched_on(
    qt_app, modern_project: Path
):
    window = _window(qt_app, modern_project)
    try:
        window.project_panel.shown_boxes["reference"].setChecked(False)

        combo = window.image_panel.series_combo
        offered = [combo.itemData(index) for index in range(combo.count())]
        assert "reference" not in offered and "adaptive" in offered
        assert "reference" not in window.label_controller.available_series()
    finally:
        destroy(window)


def test_the_settings_decide_which_series_the_lists_open_on(
    qt_app, residue_project: Path
):
    settings = QSettings(str(residue_project / "preferences.ini"), QSettings.IniFormat)
    settings.setValue("defaults/shown/other", True)
    settings.sync()

    window = _window(qt_app, residue_project)
    try:
        assert window.project_panel.shown_boxes["other"].isChecked()
        assert len(window.controller.lamellae) == 2
    finally:
        destroy(window)


# ---------------------------------------------------------------------------
# The grid overview
# ---------------------------------------------------------------------------

def test_the_grid_view_is_offered_where_the_project_has_an_overview(
    qt_app, grid_project: Path
):
    window = _window(qt_app, grid_project)
    try:
        assert "grid" in window.controller.available_modes()
        assert window.image_panel.mode_buttons["grid"].isEnabled()
    finally:
        destroy(window)


def test_the_grid_view_is_withheld_where_the_project_has_none(
    qt_app, modern_project: Path
):
    """Unavailable as the plot view already is for a stage with no figure."""
    window = _window(qt_app, modern_project)
    try:
        assert "grid" not in window.controller.available_modes()
        assert not window.image_panel.mode_buttons["grid"].isEnabled()

        window._set_mode("grid", from_user=True)

        assert window.controller.mode != "grid"
    finally:
        destroy(window)


def test_the_grid_view_marks_every_lamella_it_can_place(qt_app, grid_project: Path):
    window = _window(qt_app, grid_project)
    try:
        window._set_mode("grid", from_user=True)

        placement = window.controller.grid.placement
        assert [marker.name for marker in placement.markers] == ["01-ruling-ibex"]
        assert window.image_panel.stack.currentIndex() == 2
        assert window.image_panel.grid_view._marks
    finally:
        destroy(window)


def test_the_grid_view_always_says_the_positions_are_approximate(
    qt_app, grid_project: Path
):
    """The projection is flat, and a reader arriving here has no other way to
    know that."""
    window = _window(qt_app, grid_project)
    try:
        window._set_mode("grid", from_user=True)

        assert "approximate" in window.image_panel.badges.text()
    finally:
        destroy(window)


def test_a_lamella_the_experiment_never_positioned_is_told_so(
    qt_app, grid_project: Path
):
    """Otherwise the reviewer looks for a ring that is not there."""
    window = _window(qt_app, grid_project)
    try:
        window._set_mode("grid", from_user=True)
        for position, lamella in enumerate(window.controller.lamellae):
            if lamella.name != "02-ruling-oryx":
                continue
            window.controller.select_lamella(position)
            window.refresh()

        assert "no recorded position" in window.image_panel.badges.text()
    finally:
        destroy(window)


def test_a_lamella_the_experiment_did_position_is_not_told_that(
    qt_app, grid_project: Path
):
    window = _window(qt_app, grid_project)
    try:
        window._set_mode("grid", from_user=True)
        for position, lamella in enumerate(window.controller.lamellae):
            if lamella.name != "01-ruling-ibex":
                continue
            window.controller.select_lamella(position)
            window.refresh()

        assert "no recorded position" not in window.image_panel.badges.text()
    finally:
        destroy(window)


def test_the_grid_view_offers_every_overview_in_the_variant_slot(
    qt_app, grid_project: Path
):
    """A retake, or one before and one after coating: the reviewer chooses,
    in the slot the Grid view has no imaging condition for, and leaving the
    view hands the slot back."""
    import shutil

    project = grid_project / "AutoLamella-2026-07-28-10-31"
    shutil.copy(
        project / "overview-image-2026-07-28_10-31-00.tif",
        project / "overview-image-2026-07-28_12-00-00.tif",
    )
    window = _window(qt_app, grid_project)
    try:
        window._set_mode("grid", from_user=True)
        panel = window.image_panel

        assert panel.variant_label.text() == "Overview"
        assert panel.variant_combo.count() == 2
        assert panel.variant_combo.isEnabled()
        assert window.controller.grid.chosen == 0

        panel.variant_combo.setCurrentIndex(1)

        assert window.controller.grid.chosen == 1

        window._set_mode("sem", from_user=True)

        assert panel.variant_label.text() == "Variant"
    finally:
        destroy(window)


def test_an_overview_that_places_nothing_is_shown_saying_why(
    qt_app, grid_project: Path
):
    """The picture is still the grid; a view withheld for want of rings would
    hide it, and one shown without them would not say why."""
    project = grid_project / "AutoLamella-2026-07-28-10-31"
    (project / "experiment.yaml").write_text("name: grid\n", encoding="utf-8")
    window = _window(qt_app, grid_project)
    try:
        assert "grid" in window.controller.available_modes()

        window._set_mode("grid", from_user=True)

        assert "record a position" in window.image_panel.badges.text()
        assert "approximate" not in window.image_panel.badges.text()
    finally:
        destroy(window)


def test_the_grid_view_survives_stepping_between_lamellae(qt_app, grid_project: Path):
    """It shows the project, so moving within it does not take it away."""
    window = _window(qt_app, grid_project)
    try:
        window._set_mode("grid", from_user=True)
        for position in range(len(window.controller.lamellae)):
            window.controller.select_lamella(position)
            window.refresh()
            assert window.controller.mode == "grid", window.controller.lamella.name
    finally:
        destroy(window)


def test_the_current_lamellas_name_is_drawn_above_every_ring(qt_app):
    """A neighbour's ring added after the name, at the same height, was drawn
    over it."""
    from qtpy.QtWidgets import QGraphicsEllipseItem, QGraphicsRectItem

    from am_segmentation_manager.gui.widgets.grid_view import GridView
    from am_segmentation_manager.overview import LamellaMarker

    view = GridView()
    view.set_markers(
        (
            LamellaMarker("05-casual-corgi", 0.5, 0.5, True),
            LamellaMarker("06-next-door", 0.5, 0.45, True),
        ),
        "05-casual-corgi",
        (400, 400),
    )

    rings = [item for item in view._marks if isinstance(item, QGraphicsEllipseItem)]
    names = [item for item in view._marks if isinstance(item, QGraphicsRectItem)]
    assert len(rings) == 2 and len(names) == 1
    assert all(names[0].zValue() > ring.zValue() for ring in rings)
