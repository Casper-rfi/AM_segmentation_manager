"""Widget-level tests for the variant control and the reach line.

These drive the real widgets offscreen against a project with several imaging
conditions. The behaviour pinned here is what the design turns on: the variant
control is absent unless there is a choice, and the reach line is present
always.
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
def variant_project(tmp_path: Path) -> Path:
    """The legacy layout: SEM conditions sharing a label, FIB with its own."""
    root = tmp_path / "scan"
    project = root / "20240228_adaptive_milling_ai"
    lamella = project / "lam_010"
    image = np.full((64, 96), 100, dtype=np.uint8)
    label = np.zeros((64, 96), dtype=np.uint8)
    label[:20] = 2

    for number in range(2):
        for directory in SEM_VARIANTS:
            _write(lamella / directory / f"img_{number:04d}.tif", image)
        _write(lamella / "FIB" / f"img_{number:04d}.tif", image)
        _write(lamella / "labels" / f"img_{number:04d}.tif", label)
        _write(lamella / "labels_FIB" / f"img_{number:04d}.tif", label)

    project.mkdir(parents=True, exist_ok=True)
    (project / "AM_metadata.yaml").write_text(
        "metadata:\n  sample: HeLa\nvariants:\n  8_fi_1: frame integration 1\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def plain_project(tmp_path: Path) -> Path:
    """An ordinary Autolamella project: one condition per modality."""
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
    window = CurationWindow(CurationController(cache=FrameCache(), config=config))
    window.scan(str(root))
    _settle(qt_app)
    return window


def _settle(app, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)


# ---------------------------------------------------------------------------
# The variant control appears only when there is a choice
# ---------------------------------------------------------------------------

def test_the_variant_control_is_hidden_without_a_choice(qt_app, plain_project: Path):
    """Every Autolamella project so far is in this case."""
    window = _window(qt_app, plain_project)
    try:
        # isHidden() rather than isVisible(): a child of a window that was
        # never shown is never "visible", so isVisible() would pass here
        # whether or not the control was actually hidden.
        assert window.image_panel.variant_combo.isHidden() is True
        assert window.image_panel.variant_label.isHidden() is True
    finally:
        window.close()


def test_the_variant_control_appears_when_conditions_exist(qt_app, variant_project: Path):
    window = _window(qt_app, variant_project)
    try:
        combo = window.image_panel.variant_combo
        assert combo.isHidden() is False
        assert combo.count() == len(SEM_VARIANTS)
        assert [combo.itemData(i) for i in range(combo.count())] == ["8", "8_fi_1", "8_fi_4"]
    finally:
        window.close()


def test_a_described_condition_shows_its_description(qt_app, variant_project: Path):
    """A variant is an opaque directory suffix; metadata can name it."""
    window = _window(qt_app, variant_project)
    try:
        combo = window.image_panel.variant_combo
        captions = [combo.itemText(i) for i in range(combo.count())]
        assert "frame integration 1" in captions
        assert "8_fi_4" in captions  # undescribed, so shown raw
    finally:
        window.close()


def test_choosing_a_condition_changes_only_the_view(qt_app, variant_project: Path):
    window = _window(qt_app, variant_project)
    try:
        before = window.controller.reach()
        window.image_panel.variant_combo.setCurrentIndex(1)
        _settle(qt_app)

        assert window.controller.current_variant("SEM") == "8_fi_1"
        assert window.controller.reach() == before
    finally:
        window.close()


def test_the_control_disappears_when_switching_to_a_single_condition_modality(
    qt_app, variant_project: Path
):
    window = _window(qt_app, variant_project)
    try:
        window._set_mode("fib", from_user=True)
        _settle(qt_app)
        assert window.image_panel.variant_combo.isHidden() is True
    finally:
        window.close()


# ---------------------------------------------------------------------------
# The reach line
# ---------------------------------------------------------------------------

def test_the_reach_line_is_shown_even_when_nothing_is_shared(qt_app, plain_project: Path):
    """Always present, so its absence never has to be interpreted."""
    window = _window(qt_app, plain_project)
    try:
        assert window.state_panel.reach.text() == "Moves 1 image, no label yet"
        assert window.state_panel.reach_shared.isHidden() is True
    finally:
        window.close()


def test_the_reach_line_reports_images_pulled_in_by_a_shared_label(
    qt_app, variant_project: Path
):
    window = _window(qt_app, variant_project)
    try:
        assert window.state_panel.reach.text() == "Moves 3 images + 1 label"
        assert window.state_panel.reach_shared.text() == (
            "2 more share a label with this one"
        )
        assert window.state_panel.reach_shared.isHidden() is False
    finally:
        window.close()


def test_the_reach_line_follows_the_target(qt_app, variant_project: Path):
    window = _window(qt_app, variant_project)
    try:
        window._set_target("FIB")
        _settle(qt_app)

        assert window.state_panel.reach.text() == "Moves 1 image + 1 label"
        assert window.state_panel.reach_shared.isHidden() is True
    finally:
        window.close()


def test_the_reach_line_follows_the_scope(qt_app, variant_project: Path):
    """Widening the scope must re-count, not report the old frame's total."""
    window = _window(qt_app, variant_project)
    try:
        frame_total = window.controller.reach().images
        window._set_scope("lamella")
        _settle(qt_app)

        assert window.controller.scope == "lamella"
        assert window.controller.reach().images > frame_total
        assert str(window.controller.reach().images) in window.state_panel.reach.text()
    finally:
        window.close()


# ---------------------------------------------------------------------------
# Mismatch marking
# ---------------------------------------------------------------------------

def test_a_state_mismatch_is_marked_on_the_timeline(qt_app, variant_project: Path):
    lamella = variant_project / "20240228_adaptive_milling_ai" / "lam_010"
    (lamella / "SEM_8" / "deselected").mkdir()
    (lamella / "SEM_8" / "img_0000.tif").rename(
        lamella / "SEM_8" / "deselected" / "img_0000.tif"
    )

    window = _window(qt_app, variant_project)
    try:
        assert window.controller.mismatched_frames() == {0}
        assert window.timeline._mismatched == {0}
    finally:
        window.close()


def test_a_consistent_project_marks_nothing(qt_app, variant_project: Path):
    window = _window(qt_app, variant_project)
    try:
        assert window.timeline._mismatched == set()
    finally:
        window.close()
