"""Widget-level tests for the variant control and the reach line.

These drive the real widgets offscreen against a project with several imaging
conditions. The behaviour pinned here is what the design turns on: the variant
control is absent unless there is a choice, and the reach line is present
always.
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
    from qtpy.QtCore import QSettings, Qt

    from am_segmentation_manager.config import SegmentationConfig
    from am_segmentation_manager.gui.cache import FrameCache
    from am_segmentation_manager.gui.controller import CurationController
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow

SEM_VARIANTS = ("SEM_8", "SEM_8_fi_1", "SEM_8_fi_4")


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
        "metadata:\n  sample: HeLa\n",
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
        and window.image_panel.variant_combo.count() > 0,
        what="the scan to reach the imaging-condition control",
    )
    return window


# ---------------------------------------------------------------------------
# The variant control stays put, and is greyed out without a choice
# ---------------------------------------------------------------------------

def test_the_variant_control_is_greyed_without_a_choice(qt_app, plain_project: Path):
    """Every Autolamella project so far is in this case, so hiding it moved
    the whole row on nearly every step between projects."""
    window = _window(qt_app, plain_project)
    try:
        # isHidden() rather than isVisible(): a child of a window that was
        # never shown is never "visible", so isVisible() would pass here
        # whether or not the control was actually hidden.
        assert window.image_panel.variant_combo.isHidden() is False
        assert window.image_panel.variant_label.isHidden() is False
        assert window.image_panel.variant_combo.isEnabled() is False
    finally:
        destroy(window)


def test_the_variant_control_opens_when_conditions_exist(qt_app, variant_project: Path):
    window = _window(qt_app, variant_project)
    try:
        combo = window.image_panel.variant_combo
        assert combo.isEnabled() is True
        assert combo.count() == len(SEM_VARIANTS)
        assert [combo.itemData(i) for i in range(combo.count())] == list(SEM_VARIANTS)
    finally:
        destroy(window)


def test_a_condition_is_captioned_by_its_directory_suffix(qt_app, variant_project: Path):
    window = _window(qt_app, variant_project)
    try:
        combo = window.image_panel.variant_combo
        captions = [combo.itemText(i) for i in range(combo.count())]
        assert captions == ["8", "8_fi_1", "8_fi_4"]
    finally:
        destroy(window)


def test_choosing_a_condition_changes_only_the_view(qt_app, variant_project: Path):
    window = _window(qt_app, variant_project)
    try:
        before = window.controller.reach()
        window.image_panel.variant_combo.setCurrentIndex(1)
        settle(qt_app, lambda: window.controller.current_condition("SEM") == "SEM_8_fi_1",
               what="the view to move to the other imaging condition")

        assert window.controller.current_condition("SEM") == "SEM_8_fi_1"
        assert window.controller.reach() == before
    finally:
        destroy(window)


def test_the_control_greys_out_when_switching_to_a_single_condition_modality(
    qt_app, variant_project: Path
):
    window = _window(qt_app, variant_project)
    try:
        window._set_mode("fib", from_user=True)
        settle(qt_app, lambda: window.image_panel.variant_combo.isEnabled() is False,
               what="the control to grey out on a single-condition modality")
        assert window.image_panel.variant_combo.isHidden() is False
        assert window.image_panel.variant_combo.isEnabled() is False
    finally:
        destroy(window)


# ---------------------------------------------------------------------------
# The reach line
# ---------------------------------------------------------------------------

def test_the_reach_line_is_shown_even_when_nothing_is_shared(qt_app, plain_project: Path):
    """Always present, so its absence never has to be interpreted."""
    window = _window(qt_app, plain_project)
    try:
        assert window.state_panel.reach.text() == "Covers 1 image, no label yet, 1 lamella"
        assert window.state_panel.reach_shared.isHidden() is True
    finally:
        destroy(window)


def test_the_reach_line_reports_images_pulled_in_by_a_shared_label(
    qt_app, variant_project: Path
):
    window = _window(qt_app, variant_project)
    try:
        assert window.state_panel.reach.text() == "Covers 3 images, 1 label, 1 lamella"
        assert window.state_panel.reach_shared.text() == (
            "2 more share a label with this one"
        )
        assert window.state_panel.reach_shared.isHidden() is False
        from am_segmentation_manager.gui.widgets.theme import BADGE_STYLES

        assert window.state_panel.reach_shared.styleSheet() == BADGE_STYLES["warning"]
        tip = window.state_panel.reach_shared.toolTip().splitlines()
        assert tip[0] == "Also changed, through the label they share:"
        assert len(tip) == 1 + 2, "one line per image the label pulls in"
    finally:
        destroy(window)


def test_the_reach_line_follows_the_target(qt_app, variant_project: Path):
    window = _window(qt_app, variant_project)
    try:
        window._set_target("FIB")
        settle(qt_app, lambda: window.controller.reach().images == 1,
               what="the reach line to re-count for the ion beam")

        assert window.state_panel.reach.text() == "Covers 1 image, 1 label, 1 lamella"
        assert window.state_panel.reach_shared.isHidden() is True
    finally:
        destroy(window)


def test_the_reach_line_follows_the_selection(qt_app, variant_project: Path):
    """Ticking a lamella must re-count, not report the old frame's total."""
    window = _window(qt_app, variant_project)
    try:
        frame_total = window.controller.reach().images
        window.project_panel.lamella_list.item(0).setCheckState(Qt.Checked)
        settle(qt_app, lambda: window.controller.lamella_checks() == [True],
               what="the tick to be delivered and the reach line re-counted")

        assert window.controller.lamella_checks() == [True]
        assert window.controller.reach().images > frame_total
        assert str(window.controller.reach().images) in window.state_panel.reach.text()
    finally:
        destroy(window)


# ---------------------------------------------------------------------------
# Mismatch marking
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# The family control appears only where there is a family to choose
# ---------------------------------------------------------------------------

@pytest.fixture
def two_family_project(tmp_path: Path) -> Path:
    """One lamella whose SEM images come in two families, sem and sem-5kV."""
    root = tmp_path / "scan"
    project = root / "20260930_adaptive_milling"
    image = np.full((64, 96), 100, dtype=np.uint8)
    for directory in ("sem", "sem_fi4", "sem-5kV", "FIB"):
        _write(project / "lam_010" / directory / "img_0000.tif", image)
    (project / "AM_metadata.yaml").write_text("metadata:\n", encoding="utf-8")
    return root


def test_the_family_control_is_hidden_with_one_family_per_beam(qt_app, variant_project: Path):
    """No project in the collection holds two families of a beam."""
    window = _window(qt_app, variant_project)
    try:
        assert window.state_panel.family_control.isHidden() is True
    finally:
        destroy(window)


def test_choosing_every_family_recounts_the_reach(qt_app, two_family_project: Path):
    window = _window(qt_app, two_family_project)
    try:
        panel = window.state_panel
        assert panel.family_control.isHidden() is False
        assert panel.reach.text().startswith("sem family: covers")

        panel.family_control.buttons["every"].click()

        assert window.controller.family_scope == "every"
        assert panel.reach.text().startswith("Every family: covers")
        window.state_panel.target_control.buttons["FIB"].click()
        assert panel.family_control.isHidden() is True
    finally:
        destroy(window)


# ---------------------------------------------------------------------------
# Several families are offered under a heading each
# ---------------------------------------------------------------------------

def _rows(combo) -> list[tuple[str, bool]]:
    """Return each row's text and whether it can be chosen."""
    enabled = Qt.ItemFlag.ItemIsEnabled
    return [
        (combo.itemText(row), bool(combo.model().index(row, 0).flags() & enabled))
        for row in range(combo.count())
    ]


def test_each_family_is_a_heading_that_cannot_be_chosen(qt_app, two_family_project: Path):
    window = _window(qt_app, two_family_project)
    try:
        window._set_mode("sem")
        combo = window.image_panel.variant_combo
        assert _rows(combo) == [
            ("sem", False), ("base", True), ("fi4", True),
            ("sem-5kV", False), ("base", True),
        ]
        assert combo.count_badge.text() == "3", "the headings are not counted"
        assert combo.currentIndex() == 1
    finally:
        destroy(window)


def test_a_condition_chosen_under_a_heading_is_shown_by_its_whole_name(
    qt_app, two_family_project: Path
):
    window = _window(qt_app, two_family_project)
    try:
        window._set_mode("sem")
        combo = window.image_panel.variant_combo
        combo.resize(300, combo.height())

        combo.setCurrentIndex(4)

        assert window.controller.current_condition("SEM") == "sem-5kV"
        assert combo.shown_text() == "sem-5kV"
        combo.setCurrentIndex(2)
        assert combo.shown_text() == "sem [fi4]"
    finally:
        destroy(window)


def test_one_family_is_offered_without_headings(qt_app, variant_project: Path):
    window = _window(qt_app, variant_project)
    try:
        assert all(enabled for _, enabled in _rows(window.image_panel.variant_combo))
    finally:
        destroy(window)
