"""Widget-level tests for the series control in Curate.

Drives the real widgets offscreen against a 2026 project, whose lamella holds
reference and alignment images around its Adaptive Polishing stage. What the
design turns on is pinned here: the control is always on screen, greyed where
the lamella holds one series, and choosing one replaces the timeline.
"""
from __future__ import annotations

import importlib.util
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
        window.controller.select_lamella(1)
        window.refresh()

        assert [frame.series for frame in window.controller.frames] == ["other"]
    finally:
        destroy(window)


def test_leaving_the_residue_offers_the_controls_again(qt_app, residue_project: Path):
    window = _window(qt_app, residue_project)
    try:
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
