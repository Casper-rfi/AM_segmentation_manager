"""Widget-level tests for the series control in Curate.

Drives the real widgets offscreen against a 2026 project, whose lamella holds
reference and alignment images around its Adaptive Polishing stage. What the
design turns on is pinned here: the control is always on screen, greyed where
the lamella holds one series, and choosing one replaces the timeline.
"""
from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

import pytest

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
    _settle(qt_app)
    return window


def _settle(app, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)



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
        window.close()


def test_the_series_control_opens_when_a_lamella_holds_more(qt_app, modern_project):
    window = _window(qt_app, modern_project)
    try:
        combo = window.image_panel.series_combo
        assert combo.isEnabled() is True
        offered = [combo.itemData(i) for i in range(combo.count())]
        assert offered == ["adaptive", "reference", "alignment"]
    finally:
        window.close()


def test_the_control_shows_the_series_by_name(qt_app, modern_project: Path):
    window = _window(qt_app, modern_project)
    try:
        combo = window.image_panel.series_combo
        captions = [combo.itemText(i) for i in range(combo.count())]
        assert captions == ["Adaptive Polishing", "reference", "alignment"]
    finally:
        window.close()


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
        _settle(qt_app)

        combo = window.image_panel.series_combo
        assert window.controller.series == "reference"
        assert len(window.controller.frames) == 1
        assert combo.currentText() == "reference"
        assert combo.isEnabled() is True, "alignment is still one click away"
    finally:
        window.close()


def test_stepping_back_returns_to_the_chosen_series(qt_app, mixed_project: Path):
    """The fallback answers about one lamella rather than changing the choice."""
    window = _window(qt_app, mixed_project)
    try:
        window._step_lamella(1)
        _settle(qt_app)
        window._step_lamella(-1)
        _settle(qt_app)

        assert window.controller.series == "adaptive"
        assert window.image_panel.series_combo.currentText() == "Adaptive Polishing"
    finally:
        window.close()


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
        window.close()


def test_choosing_a_series_rebuilds_the_timeline(qt_app, modern_project: Path):
    window = _window(qt_app, modern_project)
    try:
        combo = window.image_panel.series_combo
        combo.setCurrentIndex(combo.findData("reference"))
        _settle(qt_app)

        assert window.controller.series == "reference"
        assert len(window.controller.frames) == 1
        assert window.controller.frames[0].series == "reference"
    finally:
        window.close()


def test_the_frame_readout_follows_the_chosen_series(qt_app, modern_project: Path):
    window = _window(qt_app, modern_project)
    try:
        assert window.frame_label.text() == "1 / 2"

        combo = window.image_panel.series_combo
        combo.setCurrentIndex(combo.findData("alignment"))
        _settle(qt_app)

        assert window.frame_label.text() == "1 / 1"
    finally:
        window.close()
