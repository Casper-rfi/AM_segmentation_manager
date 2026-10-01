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
    from qtpy.QtGui import QFont, QFontMetrics

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


def test_a_condition_is_captioned_by_its_directory_name(qt_app, variant_project: Path):
    window = _window(qt_app, variant_project)
    try:
        combo = window.image_panel.variant_combo
        captions = [combo.itemText(i) for i in range(combo.count())]
        assert captions == ["SEM_8", "SEM_8_fi_1", "SEM_8_fi_4"]
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
            ("sem", False), ("sem", True), ("sem_fi4", True),
            ("sem-5kV", False), ("sem-5kV", True),
        ]
        assert combo.count_badge.text() == "3", "the headings are not counted"
        assert combo.currentIndex() == 1
    finally:
        destroy(window)


def test_a_condition_is_shown_by_its_folder_name_with_its_family_bold(
    qt_app, two_family_project: Path
):
    window = _window(qt_app, two_family_project)
    try:
        window._set_mode("sem")
        combo = window.image_panel.variant_combo
        combo.resize(300, combo.height())

        combo.setCurrentIndex(4)
        assert window.controller.current_condition("SEM") == "sem-5kV"
        assert combo.shown_runs() == ("sem-5kV", "")
        combo.setCurrentIndex(2)
        assert combo.shown_runs() == ("sem", "_fi4")
    finally:
        destroy(window)


def _condition_combo(names: tuple[str, ...], family: str, primary: str | None = None):
    """Return an ElidingCombo offering *names*, all of *family*, the last one chosen."""
    from am_segmentation_manager.frames import ConditionChoice
    from am_segmentation_manager.gui.widgets.common import fill_condition_combo
    from am_segmentation_manager.gui.widgets.controls import ElidingCombo

    combo = ElidingCombo()
    choices = [
        ConditionChoice(family=family, variant=None, condition=name, primary=name == primary)
        for name in names
    ]
    fill_condition_combo(combo, choices, names[-1])
    return combo


def _starred(combo) -> list[bool]:
    """Return whether each row carries the primary star rather than a blank."""
    from am_segmentation_manager.gui.widgets.common import primary_star

    star = primary_star().pixmap(16, 16).toImage()
    return [
        combo.itemIcon(row).pixmap(16, 16).toImage() == star for row in range(combo.count())
    ]


def test_the_primary_of_a_family_of_several_is_starred(qt_app):
    combo = _condition_combo(("sem", "sem_fi1", "sem_fi4"), "sem", primary="sem_fi1")

    assert _starred(combo) == [False, True, False]
    assert all(not combo.itemIcon(row).isNull() for row in range(3)), "kept in line"
    assert "its primary" in combo.itemData(1, Qt.ItemDataRole.ToolTipRole)
    destroy(combo)


def test_a_family_of_one_carries_no_star(qt_app):
    """Nothing to choose between, so nothing to say which is chosen."""
    combo = _condition_combo(("sem",), "sem", primary="sem")

    assert combo.itemIcon(0).isNull()
    destroy(combo)


def test_the_field_keeps_room_for_the_star_only_where_it_shows_it(qt_app):
    """The blank that lines the open list up would cost a long name its end."""
    from am_segmentation_manager.gui.widgets.common import CONDITION_FIELD_WIDTH

    combo = _condition_combo(("SEM-test", "SEM-test_8_fi_16"), "SEM-test", primary="SEM-test")
    combo.resize(CONDITION_FIELD_WIDTH, combo.sizeHint().height())

    assert combo.shown_text() == "SEM-test_8_fi_16"
    destroy(combo)


def test_a_moved_condition_names_its_family_on_hover(qt_app):
    """sem2kVb does not begin with sem-2kV, so nothing in it is bold to say so."""
    combo = _condition_combo(("sem-2kV", "sem2kVb"), "sem-2kV")

    assert combo.itemData(1, Qt.ItemDataRole.ToolTipRole) == "Family sem-2kV"
    assert combo.shown_runs() == ("", "sem2kVb")
    destroy(combo)


def test_a_long_condition_keeps_its_family_and_its_end(qt_app):
    """Cut at the end, as before, SEM-test_8_fi_16 read SEM-test [8_fi_... and
    could not be told from SEM-test_8_fi_1."""
    from am_segmentation_manager.gui.widgets.controls import condition_runs

    combo = _condition_combo(("SEM-test", "SEM-test_8_fi_1", "SEM-test_8_fi_16"), "SEM-test")
    font = combo.font()
    whole = condition_runs("SEM-test_8_fi_16", 8, font, 1000)
    assert whole == ("SEM-test", "_8_fi_16")

    narrow = QFontMetrics(font)
    bold = QFont(font)
    bold.setBold(True)
    whole_width = QFontMetrics(bold).horizontalAdvance("SEM-test") + narrow.horizontalAdvance(
        "_8_fi_16"
    )
    head, tail = condition_runs("SEM-test_8_fi_16", 8, font, whole_width - 2)
    assert head == "SEM-test"
    assert "…" in tail and tail.endswith("6"), "the middle goes, not the end"

    head, tail = condition_runs("SEM-test_8_fi_16", 8, font, narrow.horizontalAdvance("SE…6"))
    assert head == "" and "…" in tail, "too narrow for the family: nothing bold"
    destroy(combo)


def test_the_variant_field_fits_a_long_condition_whole(qt_app):
    from am_segmentation_manager.gui.widgets.common import CONDITION_FIELD_WIDTH

    combo = _condition_combo(("SEM-test", "SEM-test_8_fi_16"), "SEM-test")
    combo.resize(CONDITION_FIELD_WIDTH, combo.sizeHint().height())

    assert combo.shown_text() == "SEM-test_8_fi_16"
    destroy(combo)


def test_the_open_list_is_as_wide_as_its_widest_condition(qt_app):
    combo = _condition_combo(("SEM-test", "SEM-test_8_fi_16"), "SEM-test")
    combo.resize(60, combo.sizeHint().height())

    combo.showPopup()
    try:
        view = combo.view()
        assert view.width() >= view.sizeHintForColumn(0)
        assert view.sizeHintForColumn(0) > 60
    finally:
        combo.hidePopup()
        destroy(combo)


def test_one_family_is_offered_without_headings(qt_app, variant_project: Path):
    window = _window(qt_app, variant_project)
    try:
        assert all(enabled for _, enabled in _rows(window.image_panel.variant_combo))
    finally:
        destroy(window)
