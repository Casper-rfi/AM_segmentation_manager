"""The Export tab's controls, driven as widgets.

What is pinned here is only the wiring: that a control the user changes reaches
the controller. The arithmetic it then feeds lives in ``test_gui_tabs.py``,
which needs no display.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

# Must be set before any Qt module is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HAS_QT = importlib.util.find_spec("qtpy") is not None

pytestmark = pytest.mark.skipif(not HAS_QT, reason="Qt binding not installed")

if HAS_QT:
    from qtpy.QtCore import Qt
    from qtpy.QtTest import QTest
    from qtpy.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QLabel,
        QListWidget,
        QPushButton,
        QWidget,
    )

    from am_segmentation_manager.aggregate import AREA_BINS_PER_DECADE, LAMELLA_ROOT
    from am_segmentation_manager.gui.export_controller import (
        ANY_LABEL,
        DEFAULT_PREFIX,
        ExportController,
    )
    from am_segmentation_manager.gui.widgets.export_panel import (
        NO_CROSS,
        ExportPanel,
        FilterRow,
    )
    from am_segmentation_manager.layout import SERIES
    from am_segmentation_manager.scanner import Scanner
    from conftest import build_dataset, destroy


#: The top-level widgets built by the test running now; see :func:`_built`.
_BUILT: list = []


def _built(widget):
    """Return *widget*, registered to be destroyed when the test ends.

    Every panel built here goes through this: a panel left alive slows every
    later test that applies the theme, as ``conftest.destroy`` explains.
    """
    _BUILT.append(widget)
    return widget


@pytest.fixture(autouse=True)
def _destroy_built_widgets():
    """Destroy what :func:`_built` registered, once queued calls have run."""
    yield
    if _BUILT:
        QApplication.processEvents()
    while _BUILT:
        destroy(_BUILT.pop())


@pytest.fixture
def panel(qt_app) -> ExportPanel:
    return _built(ExportPanel(ExportController()))


def _tick(widget, row: int, on: bool = True) -> None:
    """Tick a filter row through the widget, as a user would.

    Followed by :func:`_settle`, because acting on a tick is deferred to the
    next turn of the event loop -- see ``ExportPanel._recount``.
    """
    widget.item(row).setCheckState(Qt.Checked if on else Qt.Unchecked)
    _settle()


def _settle() -> None:
    """Let the tab catch up with a tick, as the event loop would."""
    QApplication.processEvents()


def _choose(row: FilterRow, name: str) -> None:
    """Put a filter row on a property, as picking it from the dropdown would.

    Settles for the same reason :func:`_tick` does: the dropdown refills itself
    from inside its own signal, so the row is filled a turn later.
    """
    index = row.property_box.findData(name)
    assert index != -1, f"{name} is not on offer"
    row.property_box.setCurrentIndex(index)
    _settle()


def _is_enabled(row: FilterRow, name: str) -> bool:
    """Return whether a property can be picked from a row's dropdown."""
    model = row.property_box.model()
    index = row.property_box.findData(name)
    assert index != -1, f"{name} is not on offer"
    return bool(model.item(index).flags() & Qt.ItemIsEnabled)


def _loaded_panel(tmp_path, **kwargs) -> ExportPanel:
    """A panel over a real two-lamella scan, as the window builds it."""
    pytest.importorskip("tifffile")
    build_dataset(tmp_path, lamellae=2, acquisitions=2, labels="annotated", **kwargs)
    controller = ExportController()
    controller.set_projects(Scanner(tmp_path).build_index().all_projects(), tmp_path)
    panel = _built(ExportPanel(controller))
    panel.refresh()
    return panel


def _varied_panel(tmp_path) -> ExportPanel:
    """A panel over a scan where a *filterable* property has two values.

    Imaging condition, because it varies within one project and survives being
    scanned: lamella and milling stage are counted on but no longer offered as
    filters, having one row per lamella and per timestamped folder.
    """
    return _loaded_panel(tmp_path, sem_variants=("SEM_8_fi_1",))


def _only_row(panel: ExportPanel) -> FilterRow:
    assert len(panel.filter_rows) == 1
    return panel.filter_rows[0]


def test_the_box_opens_with_one_filter_on_nothing(qt_app, tmp_path):
    """The breakdowns cover far more properties than any one export is chosen
    by, and a column holding every list left too little of each to read."""
    panel = _loaded_panel(tmp_path)

    row = _only_row(panel)
    assert row.property_name == ""
    assert row.values.isHidden(), "an empty box under an empty dropdown reads as broken"
    assert panel.controller.filters.choices == {}


def test_choosing_a_property_shows_its_values(qt_app, tmp_path):
    panel = _varied_panel(tmp_path)
    row = _only_row(panel)

    _choose(row, "variant")

    assert not row.values.isHidden()
    assert row.values.count() == 2


def test_adding_a_filter_leaves_the_first_alone(qt_app, tmp_path):
    panel = _varied_panel(tmp_path)
    _choose(panel.filter_rows[0], "variant")
    _tick(panel.filter_rows[0].values, 0)

    panel.add_button.click()

    assert len(panel.filter_rows) == 2
    assert panel.filter_rows[0].property_name == "variant"
    assert panel.controller.filters.choices["variant"]
    assert panel.filter_rows[1].property_name == ""


def _two_stage_names(tmp_path) -> ExportPanel:
    """A panel over two projects whose stages are named differently."""
    pytest.importorskip("tifffile")
    for name, stage in (("polishing", "adaptive_polish"), ("rough", "MillRoughCut")):
        build_dataset(
            tmp_path / name, acquisitions=2, labels="annotated", stage_parts=(stage,)
        )
    controller = ExportController()
    controller.set_projects(Scanner(tmp_path).build_index().all_projects(), tmp_path)
    panel = _built(ExportPanel(controller))
    panel.refresh()
    return panel


def _values(row: FilterRow) -> list[str]:
    return [row.values.item(index).text() for index in range(row.values.count())]


def _loose_lamella_panel(tmp_path) -> ExportPanel:
    """A 2026-shaped scan: reference images loose in the lamella and in a task."""
    tifffile = pytest.importorskip("tifffile")
    numpy = pytest.importorskip("numpy")
    project = tmp_path / "AutoLamella-2026"
    project.mkdir(parents=True)
    (project / "experiment.yaml").write_text("name: loose\n", encoding="utf-8")
    for lamella_name in ("01-ruling-ibex", "02-tidy-oryx"):
        lamella = project / lamella_name
        for relative in ("ref_alignment_ib.tif", "MillRoughCut/ref_Polishing_start_eb.tif"):
            destination = lamella / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(destination), numpy.zeros((16, 16), dtype=numpy.uint8))

    controller = ExportController()
    controller.set_projects(Scanner(tmp_path).build_index().all_projects(), tmp_path)
    panel = _built(ExportPanel(controller))
    panel.refresh()
    # Through the boxes rather than the filter behind them: the tab opens on
    # Adaptive Polishing alone, and there is none of it here.
    for series in ("reference", "alignment"):
        panel.series_boxes[series].setChecked(True)
    return panel


def test_a_lamella_is_never_offered_as_a_milling_stage(qt_app, tmp_path):
    """Autolamella leaves reference images directly in the lamella, whose
    directory then stands in for a stage -- so every lamella in the collection
    turned up in the list of stage names."""
    panel = _loose_lamella_panel(tmp_path)
    row = _only_row(panel)

    _choose(row, "stage-name")

    assert _values(row) == [LAMELLA_ROOT, "MillRoughCut"]


def test_the_milling_stage_is_offered_by_name(qt_app, tmp_path):
    """A stage folder carries a timestamp, so the qualified stage is one row
    per folder on disk -- hundreds of them, and each of one lamella's run. The
    name is what repeats across lamellae, and what an export is chosen by."""
    panel = _two_stage_names(tmp_path)
    row = _only_row(panel)

    _choose(row, "stage-name")

    assert _values(row) == ["MillRoughCut", "adaptive_polish"]
    assert row.property_box.findData("stage") == -1, "still counted on, not filtered"


def test_filtering_by_stage_name_narrows_the_export(qt_app, tmp_path):
    panel = _two_stage_names(tmp_path)
    row = _only_row(panel)
    before = panel.controller.preview().matched

    _choose(row, "stage-name")
    _tick(row.values, _values(row).index("MillRoughCut"))

    assert panel.controller.filters.choices["stage-name"] == ["MillRoughCut"]
    assert panel.controller.preview().matched == before // 2


def test_a_property_already_filtered_on_is_not_offered_twice(qt_app, tmp_path):
    """One property filtered in one place, so two rows cannot disagree about
    what it keeps."""
    panel = _varied_panel(tmp_path)
    _choose(panel.filter_rows[0], "variant")

    panel.add_button.click()

    second = panel.filter_rows[1]
    assert second.property_box.findData("variant") == -1
    assert second.property_box.findData("project") != -1


def test_removing_a_filter_removes_what_it_was_keeping(qt_app, tmp_path):
    """A filter left behind by a control that has gone narrows every count in
    the tab with nothing on screen to say so."""
    panel = _varied_panel(tmp_path)
    _choose(panel.filter_rows[0], "variant")
    _tick(panel.filter_rows[0].values, 0)
    panel.add_button.click()
    narrowed = sum(panel.controller.breakdown("modality").values())

    panel.filter_rows[0].remove_button.click()

    assert panel.controller.filters.choices == {}
    assert sum(panel.controller.breakdown("modality").values()) > narrowed


def test_removing_the_last_filter_empties_it_rather_than_taking_it_away(
    qt_app, tmp_path,
):
    """A box with no filter in it still says what could be filtered on; no box
    at all leaves the column blank with nothing to explain what is missing."""
    panel = _varied_panel(tmp_path)
    _choose(panel.filter_rows[0], "variant")
    _tick(panel.filter_rows[0].values, 0)

    panel.filter_rows[0].remove_button.click()

    assert len(panel.filter_rows) == 1
    assert panel.filter_rows[0].property_name == ""
    assert panel.controller.filters.choices == {}


def test_moving_a_filter_to_another_property_clears_the_one_it_leaves(
    qt_app, tmp_path,
):
    panel = _varied_panel(tmp_path)
    row = _only_row(panel)
    _choose(row, "variant")
    _tick(row.values, 0)

    _choose(row, "project")

    assert "variant" not in panel.controller.filters.choices


def test_the_acquisition_filters_are_usable_straight_away(qt_app, tmp_path):
    """Their values come from the headers, which the window reads on arrival --
    so nothing here is offered in a state where picking it does nothing, which
    is what a greyed-out row with no explanation amounts to."""
    panel = _loaded_panel(tmp_path)
    panel.controller.load_metadata()
    panel.refresh()
    row = _only_row(panel)

    assert _is_enabled(row, "microscope")


def test_ticking_a_value_narrows_the_table_under_it(qt_app, tmp_path):
    """The filter and the breakdown are one selection, so the table has to move
    when a filter does."""
    panel = _varied_panel(tmp_path)
    _choose(_only_row(panel), "variant")
    before = sum(panel.controller.breakdown("modality").values())

    _tick(_only_row(panel).values, 0)

    assert sum(panel.controller.breakdown("modality").values()) < before


def test_the_count_says_how_much_of_a_list_is_ticked(qt_app, tmp_path):
    """The lists are four rows over values that run to dozens, so what is
    ticked is usually scrolled out of sight -- which is how a filter stays on
    by accident."""
    panel = _varied_panel(tmp_path)
    row = _only_row(panel)
    _choose(row, "variant")

    _tick(row.values, 0)

    assert row.count.text() == "Imaging condition (1 of 2 ticked)"


def test_clearing_unticks_the_values_but_keeps_the_filters(qt_app, tmp_path):
    """Ticks scroll out of sight, so one control has to undo all of them --
    but clearing what a filter keeps and deciding not to filter on that
    property at all are different intentions."""
    panel = _varied_panel(tmp_path)
    row = _only_row(panel)
    _choose(row, "variant")
    _tick(row.values, 0)
    assert panel.controller.filters.choices

    panel.clear_button.click()

    assert panel.controller.filters.choices == {}
    assert row.property_name == "variant", "the filter stays where it was put"
    assert row.ticked_values() == []
    assert row.count.text() == "Imaging condition (2)"


def test_clearing_leaves_the_named_controls_alone(qt_app, tmp_path):
    """They always hold one of their choices, so there is no cleared state to
    put them into -- and "all" is not what somebody clearing filters means to
    lose if they had chosen corrected."""
    panel = _loaded_panel(tmp_path)
    panel.state.setCurrentText("corrected")

    panel.clear_button.click()

    assert panel.state.currentText() == "corrected"


def test_a_long_filter_row_elides_rather_than_scrolling_sideways(qt_app, tmp_path):
    """Project rows differ at the end. A list that scrolls instead of eliding
    shows a column of identical prefixes."""
    panel = _varied_panel(tmp_path)
    row = _only_row(panel)
    _choose(row, "variant")

    assert row.values.textElideMode() == Qt.ElideMiddle
    assert row.values.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert row.values.item(0).toolTip() == row.values.item(0).text()


def test_a_filter_is_labelled_with_its_unit_spelled_properly(
    qt_app, tmp_path,
):
    """The label is the breakdown's own, so a property is named the same way in
    the filter and in the dropdown that counts by it."""
    panel = _loaded_panel(tmp_path)

    assert panel._noun("voltage") == "Accelerating voltage (kV)"
    assert panel._noun("current") == "Beam current (pA)"


def test_the_split_controls_start_where_the_defaults_are(panel: ExportPanel):
    training = panel.controller.training

    assert panel.train_share.value() == training.train_ratio * 100
    assert panel.seed.value() == training.seed
    assert panel.keep_together.currentData() == training.keep_together
    assert panel.prefix.placeholderText() == training.prefix


def test_a_chosen_share_becomes_a_ratio(panel: ExportPanel):
    panel.train_share.setValue(65)

    assert panel.controller.training.train_ratio == 0.65


def test_a_chosen_level_is_what_the_split_keeps_together(panel: ExportPanel):
    panel.keep_together.setCurrentIndex(panel.keep_together.findData("project"))

    assert panel.controller.training.keep_together == "project"


def test_the_split_names_the_groups_it_kept_whole(qt_app, tmp_path):
    panel = _loaded_panel(tmp_path)
    panel.controller.load_measurements()

    panel.keep_together.setCurrentIndex(panel.keep_together.findData("project"))
    assert panel.split_counts.text().endswith("· 1 project")

    panel.keep_together.setCurrentIndex(panel.keep_together.findData("lamella"))
    assert panel.split_counts.text().endswith("· 2 lamellae")


def test_folds_are_off_until_asked_for_and_then_replace_the_share(panel: ExportPanel):
    assert panel.folds.text() == "Off"
    assert panel.controller.training.folds is None

    panel.folds.setValue(3)
    assert panel.controller.training.folds == 3
    assert not panel.train_share.isEnabled()   # cross-validation does not use it

    panel.folds.setValue(1)
    assert panel.controller.training.folds is None
    assert panel.train_share.isEnabled()


def test_the_split_says_how_the_folds_come_out(qt_app, tmp_path):
    panel = _loaded_panel(tmp_path)
    panel.controller.load_measurements()

    panel.folds.setValue(2)
    preview = panel.controller.preview()
    assert panel.split_counts.text() == (
        f"2 folds of {min(preview.fold_sizes)}"
        f"{'' if len(set(preview.fold_sizes)) == 1 else '–' + str(max(preview.fold_sizes))}"
        f" validation · 2 lamellae"
    )

    panel.keep_together.setCurrentIndex(panel.keep_together.findData("project"))
    assert panel.split_counts.text() == "1 project cannot fill 2 folds"


def test_clearing_the_seed_asks_for_an_unrepeatable_shuffle(panel: ExportPanel):
    panel.seeded.setChecked(False)

    assert panel.controller.training.seed is None
    assert not panel.seed.isEnabled()   # a value that no longer applies

    panel.seeded.setChecked(True)

    assert panel.controller.training.seed == 42


def test_a_name_left_empty_is_the_default_one(panel: ExportPanel):
    panel.prefix.setText("hydra_aquilos")
    assert panel.controller.training.prefix == "hydra_aquilos"

    panel.prefix.setText("   ")
    assert panel.controller.training.prefix == DEFAULT_PREFIX


# ---------------------------------------------------------------------------
# Everything that narrows is a filter
# ---------------------------------------------------------------------------

def test_the_filters_box_holds_every_control_that_narrows(panel: ExportPanel):
    """The Export box decides how files are written, never which images. A
    control that shrank the export without shrinking the table would leave two
    numbers on screen describing different collections."""
    filters = panel._filters_box
    for widget in (panel.modality, panel.state, panel.label_state, panel.conditions):
        # An ancestor rather than the parent: the column nests these inside a
        # holder and a splitter to fit them, and which of those a control sits
        # in is arrangement, where being inside the box at all is the rule.
        assert filters.isAncestorOf(widget), f"{widget} narrows from outside Filters"

    narrowing = (QComboBox, QCheckBox, QListWidget)
    for widget in panel._export_box.findChildren(QWidget):
        assert not isinstance(widget, narrowing) or widget in (
            panel.seeded,   # a checkbox, but it only makes the shuffle repeatable
            panel.keep_together,   # a dropdown, but it only decides the side
        ), f"{widget} can narrow, and is in the Export box"


def test_the_label_filter_opens_on_everything(panel: ExportPanel):
    """The tab starts as a view of the whole collection and is narrowed towards
    an export, not the other way about."""
    assert panel.label_state.currentText() == ANY_LABEL
    assert panel.controller.filters.label == ANY_LABEL


def test_choosing_a_label_state_reaches_the_controller(panel: ExportPanel):
    panel.label_state.setCurrentText("empty")

    assert panel.controller.filters.label == "empty"


def test_the_window_offers_no_way_to_export_a_blank_mask(panel: ExportPanel):
    """Deliberate negative examples stay on the command line."""
    assert not hasattr(panel, "include_empty")
    assert not any(
        "empty" in box.text().lower()
        for box in panel._export_box.findChildren(QCheckBox)
    )


# ---------------------------------------------------------------------------
# The breakdown dropdown
# ---------------------------------------------------------------------------

def test_the_measurement_settings_start_where_the_config_says(panel: ExportPanel):
    """The threshold a run with no override applies, so the window and
    quantify-labels agree until the user says otherwise."""
    assert panel.crack_threshold.value() == panel.controller.min_crack_area_um2
    assert panel.area_bins.currentData() == panel.controller.area_bins_per_decade


def test_the_crack_threshold_reaches_the_controller(panel: ExportPanel):
    panel.crack_threshold.setValue(0.25)

    assert panel.controller.min_crack_area_um2 == 0.25


def test_the_bin_control_reaches_the_controller(panel: ExportPanel):
    panel.area_bins.setCurrentIndex(panel.area_bins.findData(3))

    assert panel.controller.area_bins_per_decade == 3
    assert "100 - 200" in panel.controller.breakdowns["area-lamella"].order


def test_the_bin_control_offers_only_supported_resolutions(panel: ExportPanel):
    """The settings are 1, 2, 3 and 6. A spin box spanning them would offer 4
    and 5, which do not subdivide a decade into round edges -- and which
    set_area_bins rejects."""
    offered = [panel.area_bins.itemData(i) for i in range(panel.area_bins.count())]

    assert offered == list(AREA_BINS_PER_DECADE)


def test_the_measurement_settings_sit_beside_the_breakdown_not_the_filters(
    panel: ExportPanel,
):
    """Neither narrows anything: they say how what was measured is read. A
    control in the Filters box promises to change which images are in play."""
    assert panel.crack_threshold in panel._breakdown_box.findChildren(QDoubleSpinBox)
    assert panel.crack_threshold not in panel._filters_box.findChildren(QDoubleSpinBox)


def test_the_breakdown_list_shows_labels_and_carries_names(panel: ExportPanel):
    """The user reads "dwell time (ns)"; the controller is handed "dwell"."""
    rows = {
        panel.breakdown.itemText(index): panel.breakdown.itemData(index)
        for index in range(panel.breakdown.count())
    }

    assert rows["dwell time (ns)"] == "dwell"
    assert rows["microscope serial number"] == "serial"
    assert "dwell" not in rows          # the bare key is never displayed


def test_a_group_heading_cannot_be_chosen(panel: ExportPanel):
    """A heading is there to be read past. Selecting one would ask the
    controller to break down by "Project metadata", which is not a breakdown."""
    model = panel.breakdown.model()
    headings = [
        index
        for index in range(panel.breakdown.count())
        if panel.breakdown.itemData(index) is None
    ]

    assert headings, "the list is not grouped"
    for index in headings:
        assert not model.item(index).isEnabled()
        assert not (model.item(index).flags() & Qt.ItemIsSelectable)


def test_the_breakdown_opens_on_modality(panel: ExportPanel):
    assert panel.breakdown.currentData() == "modality"


def test_the_table_is_headed_with_the_readable_name(qt_app, tmp_path):
    """The heading over the counts is the only thing saying what they are of."""
    pytest.importorskip("tifffile")
    build_dataset(tmp_path, acquisitions=2, labels="annotated")
    controller = ExportController()
    controller.set_projects(Scanner(tmp_path).build_index().all_projects(), tmp_path)
    panel = _built(ExportPanel(controller))

    panel.breakdown.setCurrentIndex(panel.breakdown.findData("variant"))
    panel.refresh()

    assert panel.table.text().startswith("Images by imaging condition")


# ---------------------------------------------------------------------------
# Stepping through the list, and cross-referencing two properties
# ---------------------------------------------------------------------------

def test_the_counts_are_a_table_rather_than_padded_text(qt_app, tmp_path):
    """Padded text lines up only in a fixed-width font and cannot be sorted;
    the text is still what Copy as text gives."""
    panel = measurable(qt_app, tmp_path)
    panel.breakdown.setCurrentIndex(panel.breakdown.findData("modality"))
    panel.refresh()

    grid = panel.table.grid
    headers = [grid.horizontalHeaderItem(column).text() for column in range(grid.columnCount())]
    assert headers[1:] == ["Images", "Share"]
    assert {grid.item(row, 0).text() for row in range(grid.rowCount())} == set(
        panel.controller.breakdown("modality")
    )
    assert panel.table.text().startswith("Images by modality")


def test_the_arrow_keys_step_through_the_list(panel: ExportPanel):
    """Whether Qt's own keys step or open the popup is a style hint's to
    answer, and the binding here is not the one the window ships under.

    Which breakdown sits next to which is not the point, and asserting it
    would make adding one a test failure, so this steps to whatever is there.
    """
    start = panel.breakdown.findData("modality")
    panel.breakdown.setCurrentIndex(start)
    following = panel.breakdown.itemData(start + 1)

    QTest.keyClick(panel.breakdown, Qt.Key_Down)
    assert panel.breakdown.currentData() == following

    QTest.keyClick(panel.breakdown, Qt.Key_Up)
    assert panel.breakdown.currentData() == "modality"


def test_stepping_passes_over_a_group_heading(panel: ExportPanel):
    """A heading is not a breakdown, so landing on one would ask the
    controller to count by "Project metadata"."""
    panel.breakdown.setCurrentIndex(panel.breakdown.findData("state"))

    QTest.keyClick(panel.breakdown, Qt.Key_Down)

    # "workflow state" ends its group, so the next row down is a heading.
    assert panel.breakdown.currentData() == "voltage"


def test_stepping_stops_at_the_ends(panel: ExportPanel):
    """Wrapping from the last row to the first loses your place."""
    # Found rather than named: which breakdown sorts first is the breakdown
    # table's to decide, and this is about the ends, not about that one.
    first = next(
        row for row in range(panel.breakdown.count())
        if panel.breakdown.itemData(row) is not None
    )
    panel.breakdown.setCurrentIndex(first)

    QTest.keyClick(panel.breakdown, Qt.Key_Up)

    assert panel.breakdown.currentIndex() == first


def test_the_cross_reference_starts_switched_off(panel: ExportPanel):
    assert panel.cross.currentData() == NO_CROSS
    assert panel.cross.currentText() == "(none)"


def test_choosing_a_second_property_turns_the_table_into_a_grid(qt_app, tmp_path):
    pytest.importorskip("tifffile")
    build_dataset(tmp_path, acquisitions=2, labels="annotated")
    controller = ExportController()
    controller.set_projects(Scanner(tmp_path).build_index().all_projects(), tmp_path)
    panel = _built(ExportPanel(controller))

    panel.breakdown.setCurrentIndex(panel.breakdown.findData("modality"))
    panel.refresh()
    assert panel.table.text().startswith("Images by modality")

    panel.cross.setCurrentIndex(panel.cross.findData("state"))
    panel.refresh()

    text = panel.table.text()
    assert text.startswith("Images by modality (SEM / FIB) and workflow state")
    assert "total" in text      # the grid carries its margins


# ---------------------------------------------------------------------------
# Measuring the labels
# ---------------------------------------------------------------------------

def measurable(qt_app, tmp_path) -> ExportPanel:
    """A panel over a scan whose labels hold something."""
    pytest.importorskip("tifffile")
    build_dataset(tmp_path, acquisitions=2, labels="annotated")
    controller = ExportController()
    controller.set_projects(Scanner(tmp_path).build_index().all_projects(), tmp_path)
    return _built(ExportPanel(controller))


def test_the_button_says_how_much_it_is_about_to_read(qt_app, tmp_path):
    """The cost scales with the collection and nothing can stop it once it
    starts, so the size of the job belongs on the button."""
    panel = measurable(qt_app, tmp_path)
    panel.refresh()

    assert panel.measure_button.text() == "Measure 4 label(s)"
    assert panel.measure_button.isEnabled()


def test_an_unmeasured_crack_breakdown_asks_rather_than_answering(qt_app, tmp_path):
    """Counting anyway would file every image under "no-label", which reads as
    a fact about the data rather than a question nobody asked yet."""
    panel = measurable(qt_app, tmp_path)
    panel.breakdown.setCurrentIndex(panel.breakdown.findData("crack-count"))
    panel.refresh()

    text = panel.table.text()
    assert "needs the labels measured" in text
    assert "no-label" not in text


def test_measuring_fills_the_table_in_and_spends_the_button(qt_app, tmp_path):
    panel = measurable(qt_app, tmp_path)
    panel.breakdown.setCurrentIndex(panel.breakdown.findData("crack-count"))

    panel.controller.load_measurements()
    panel.refresh()

    assert "needs the labels measured" not in panel.table.text()
    assert panel.measure_button.text() == "Labels measured"
    assert not panel.measure_button.isEnabled()


def test_the_prompt_names_both_halves_of_a_cross_reference(qt_app, tmp_path):
    panel = measurable(qt_app, tmp_path)
    panel.breakdown.setCurrentIndex(panel.breakdown.findData("modality"))
    panel.cross.setCurrentIndex(panel.cross.findData("has-crack"))
    panel.refresh()

    assert "cracked (yes / no)" in panel.table.text()


# ---------------------------------------------------------------------------
# Charting what the table shows
# ---------------------------------------------------------------------------

HAS_PLOTS = importlib.util.find_spec("matplotlib") is not None
needs_plots = pytest.mark.skipif(not HAS_PLOTS, reason="matplotlib not installed")


def test_the_breakdown_opens_as_a_table(qt_app, tmp_path):
    """Numbers first: the chart is the second way of reading them, and a pane
    that opened on a chart would hide the counts the summary line refers to."""
    panel = measurable(qt_app, tmp_path)
    panel.refresh()

    assert panel.as_table.isChecked()
    assert panel.views.currentWidget() is panel.table


def test_only_one_control_is_called_chart(panel: ExportPanel):
    """Regression: a second button called "Chart..." sat next to Measure labels
    and opened a window, so it was what anyone wanting a chart pressed, and the
    tab looked as though it had no inline chart at all. There is now one chart
    and one way to reach it."""
    named = {
        widget.text()
        for widget in panel.findChildren(QWidget)
        if getattr(widget, "text", None) and "chart" in widget.text().lower()
    }

    assert named == {"Chart"}


@needs_plots
def test_switching_to_the_chart_draws_it_in_the_pane(qt_app, tmp_path):
    """The reason this is inline at all: there is room for it beside the
    filters, and a separate window has to be moved out of the way to change one."""
    panel = measurable(qt_app, tmp_path)
    panel.refresh()

    panel.as_chart.setChecked(True)

    assert panel.views.currentWidget() is panel.chart_view
    assert panel.chart_view._canvas is not None


@needs_plots
def test_copying_the_chart_puts_its_image_on_the_clipboard(qt_app, tmp_path):
    """The binding checks the arguments turning the PNG into a QImage only when
    it is called, and a failure inside a slot is printed rather than raised."""
    panel = measurable(qt_app, tmp_path)
    panel.refresh()
    panel.as_chart.setChecked(True)
    QApplication.clipboard().clear()

    panel.chart_view.copy_button.click()

    assert not QApplication.clipboard().image().isNull()


@needs_plots
def test_the_chart_pane_follows_a_filter(qt_app, tmp_path):
    """Same rule as the table: one tab, one set. A chart left showing the
    previous filter would be two collections on screen at once."""
    panel = measurable(qt_app, tmp_path)
    panel.refresh()
    panel.as_chart.setChecked(True)
    first = panel.chart_view._canvas

    panel.modality.setCurrentText("SEM")

    assert panel.chart_view._canvas is not first


def test_the_chart_is_not_drawn_while_the_table_is_showing(qt_app, tmp_path):
    """Building a figure on every keystroke of a filter, for a pane nobody is
    looking at, is what the toggle exists to avoid."""
    panel = measurable(qt_app, tmp_path)
    panel.refresh()

    assert panel.chart_view._canvas is None


@needs_plots
def test_an_unmeasured_breakdown_says_so_in_the_pane_rather_than_drawing(
    qt_app, tmp_path,
):
    """The same answer the detached window gives, in the place being looked at."""
    panel = measurable(qt_app, tmp_path)
    panel.breakdown.setCurrentIndex(panel.breakdown.findData("has-crack"))
    panel.as_chart.setChecked(True)

    assert "measured first" in panel.chart_view.message.text()
    assert panel.chart_view._canvas is None


@needs_plots
def test_one_property_is_charted_as_bars(qt_app, tmp_path):
    panel = measurable(qt_app, tmp_path)
    panel.breakdown.setCurrentIndex(panel.breakdown.findData("modality"))

    panel.as_chart.setChecked(True)

    assert panel.chart_view._canvas is not None
    # Bars have no row to take a share of, so the shading toggle is not offered.
    assert panel.chart_view.shading.isHidden()


@needs_plots
def test_two_properties_are_charted_as_a_shadeable_grid(qt_app, tmp_path):
    panel = measurable(qt_app, tmp_path)
    panel.breakdown.setCurrentIndex(panel.breakdown.findData("modality"))
    panel.cross.setCurrentIndex(panel.cross.findData("state"))

    panel.as_chart.setChecked(True)

    assert not panel.chart_view.shading.isHidden()


@needs_plots
def test_toggling_the_shading_redraws_rather_than_stacking_canvases(qt_app, tmp_path):
    """An old canvas left in the layout keeps its figure alive, and a session
    of charts would then hold every one of them."""
    panel = measurable(qt_app, tmp_path)
    panel.breakdown.setCurrentIndex(panel.breakdown.findData("modality"))
    panel.cross.setCurrentIndex(panel.cross.findData("state"))
    panel.as_chart.setChecked(True)
    first = panel.chart_view._canvas

    panel.chart_view.shading.buttons["row"].click()

    assert panel.chart_view._canvas is not first
    # Taken out of the layout, not merely covered by the new one: a canvas
    # still parented keeps its figure alive.
    assert first.parent() is None


def test_charting_without_matplotlib_explains_rather_than_failing(
    qt_app, tmp_path, monkeypatch
):
    """The pane says so where the chart would be, rather than a dialog that
    has to be dismissed before the table can be read again."""
    panel = measurable(qt_app, tmp_path)
    monkeypatch.setattr("am_segmentation_manager.plots.has_plot_support", lambda: False)

    panel.as_chart.setChecked(True)

    assert "pip install" in panel.chart_view.message.text()
    assert panel.chart_view._canvas is None


def test_the_shading_offers_exactly_what_a_grid_can_be_shaded_by(panel: ExportPanel):
    from am_segmentation_manager.plots import SHADINGS

    assert tuple(panel.chart_view.shading.buttons) == SHADINGS


@needs_plots
def test_the_chart_offers_no_way_to_edit_the_figure(qt_app, tmp_path):
    """Matplotlib's toolbar carries Subplots and Customize, which change a
    figure bound for a paper with nothing recording that they did."""
    from qtpy.QtWidgets import QToolButton

    panel = measurable(qt_app, tmp_path)
    panel.as_chart.setChecked(True)
    view = panel.chart_view

    assert view._toolbar.isHidden()
    captions = {
        button.text() for button in view.findChildren(QToolButton)
        if not view._toolbar.isAncestorOf(button) and not view.shading.isAncestorOf(button)
    }
    assert captions == {"Reset view", "Save figure...", "Copy figure", "Table as CSV..."}, (
        "the wheel zooms and Space pans, so neither has a button"
    )


@needs_plots
def test_what_is_done_with_a_chart_is_under_it_and_how_it_is_shaded_above(qt_app, tmp_path):
    """Above it, the buttons took a row from the chart on a tab short of height."""
    panel = measurable(qt_app, tmp_path)
    panel.as_chart.setChecked(True)
    view = panel.chart_view
    layout = view.layout()

    order = [layout.itemAt(index).widget() for index in range(layout.count())]

    assert order.index(view.shading) < order.index(view._canvas) < order.index(view.bar)
    assert all(view.bar.isAncestorOf(button) for button in view._actions())


@needs_plots
@pytest.mark.parametrize(
    "suffix, magic", [(".png", b"\x89PNG"), (".svg", b"<?xml"), (".pdf", b"%PDF")]
)
def test_a_figure_is_saved_in_the_format_its_name_asks_for(
    qt_app, tmp_path, monkeypatch, suffix, magic
):
    panel = measurable(qt_app, tmp_path)
    output = tmp_path / f"chart{suffix}"
    monkeypatch.setattr(
        "am_segmentation_manager.gui.widgets.chart_view.choose_save_file",
        lambda *args, **kwargs: output,
    )
    panel.as_chart.setChecked(True)

    panel.chart_view.save_button.click()

    assert output.read_bytes().startswith(magic)


@needs_plots
def test_copy_figure_puts_the_chart_on_the_clipboard(qt_app, tmp_path):
    from qtpy.QtWidgets import QApplication

    panel = measurable(qt_app, tmp_path)
    panel.as_chart.setChecked(True)
    QApplication.clipboard().clear()

    panel.chart_view.copy_button.click()

    assert not QApplication.clipboard().image().isNull()


@needs_plots
def test_the_counts_behind_a_chart_are_written_as_stats_writes_them(
    qt_app, tmp_path, monkeypatch
):
    import csv

    panel = measurable(qt_app, tmp_path)
    panel.breakdown.setCurrentIndex(panel.breakdown.findData("modality"))
    output = tmp_path / "counts.csv"
    asked: list[str] = []
    panel.breakdown_export_requested.connect(asked.append)
    monkeypatch.setattr(
        "am_segmentation_manager.gui.widgets.export_panel.choose_save_file",
        lambda *args, **kwargs: output,
    )
    panel.as_chart.setChecked(True)

    panel.chart_view.export_button.click()
    assert asked == [str(output)]

    panel.controller.export_breakdown_csv(output, "modality")
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["modality", "images"]
    assert {row[0]: int(row[1]) for row in rows[1:]} == dict(
        panel.controller.breakdown("modality")
    )


def test_a_breakdown_needing_measured_labels_is_not_written_unmeasured(qt_app, tmp_path):
    """Written anyway, every image would land under "no-label"."""
    panel = measurable(qt_app, tmp_path)

    with pytest.raises(ValueError, match="measured"):
        panel.controller.export_breakdown_csv(tmp_path / "counts.csv", "has-crack")

    assert not (tmp_path / "counts.csv").exists()


def _long_row(qt_app, count: int = 20, ticked=("v3",)) -> FilterRow:
    """A filter row over *count* values, as a pixel size's would be."""
    row = _built(FilterRow(ExportController()))
    row.show_values([f"v{index}" for index in range(count)], set(ticked), "pixel size")
    row.show_filled()
    return row


def _visible(row: FilterRow) -> list[str]:
    return [
        row.values.item(index).text()
        for index in range(row.values.count())
        if not row.values.item(index).isHidden()
    ]


def test_a_long_filter_list_can_be_searched(qt_app):
    row = _long_row(qt_app)

    assert not row.search.isHidden()
    row.search.setText("1")

    assert _visible(row) == ["v1", *(f"v{index}" for index in range(10, 20))]
    assert "hidden" in row.count.text()


def test_a_short_filter_list_has_no_search(qt_app):
    """A box over three values is clutter rather than help."""
    from am_segmentation_manager.gui.widgets.export_panel import SEARCH_FROM

    row = _long_row(qt_app, count=SEARCH_FROM)

    assert row.search.isHidden()


def test_ticking_all_while_searching_ticks_what_is_in_view_and_keeps_the_rest(qt_app):
    """A value ticked and then hidden by a search is still a choice somebody made."""
    row = _long_row(qt_app, ticked=("v3",))
    row.search.setText("1")

    row._tick_all(True)
    row.search.setText("")

    assert row.ticked_values() == ["v1", "v3", *(f"v{index}" for index in range(10, 20))]

    row.search.setText("1")
    row._tick_all(False)
    row.search.setText("")
    assert row.ticked_values() == ["v3"]


def test_the_split_says_what_it_comes_out_as_beside_its_settings(qt_app, tmp_path):
    """The summary line says it too, a column away from the share being set."""
    panel = measurable(qt_app, tmp_path)
    panel.refresh()
    assert panel.split_counts.text() == "Measure labels to see how the pairs split"

    panel.controller.load_measurements()
    panel.train_share.setValue(100)

    preview = panel.controller.preview()
    assert preview.validation == 0
    assert panel.split_counts.text().startswith(f"{preview.train} train / 0 validation · ")


# ---------------------------------------------------------------------------
# Saying what a control does
# ---------------------------------------------------------------------------

def test_every_control_that_needs_explaining_has_a_tooltip(panel: ExportPanel):
    """A control whose name does not explain it and whose tooltip is empty is
    one the user has to guess at -- which is how "One condition per scene" was
    read as the opposite of what it does."""
    for widget in (
        panel.modality, panel.state, panel.label_state, panel.breakdown,
        panel.cross, panel.conditions, panel.measure_button,
        panel.keep_together, panel.train_share, panel.folds, panel.seeded,
        panel.seed, panel.prefix, panel.check_button,
    ):
        assert widget.toolTip(), f"{widget} explains nothing"


#: Longest tooltip worth reading in a hover. Past this it is documentation,
#: and a tooltip nobody finishes explains nothing.
MAX_TOOLTIP = 200


def test_no_tooltip_is_longer_than_anyone_reads(panel: ExportPanel):
    """These grew to five lines apiece. A tooltip says what the control does;
    why the software works that way belongs in docs/decisions.md."""
    too_long = {
        widget.toolTip()
        for widget in panel.findChildren(QWidget)
        if len(widget.toolTip()) > MAX_TOOLTIP
    }

    assert not too_long


def test_every_filter_says_what_it_narrows(panel: ExportPanel):
    """The Filters box decides which images are in play for both the breakdown
    and the exports, and nothing on screen said so."""
    filters = [panel.modality, panel.state, panel.label_state, panel.conditions]
    filters += [row.values for row in panel.filter_rows]

    for widget in filters:
        assert "Narrows the breakdown and every export" in widget.toolTip()


def test_the_cross_reference_is_named_rather_than_conjoined(panel: ExportPanel):
    """"and" is not a name for a control."""
    captions = {widget.text() for widget in panel.findChildren(QLabel)}

    assert "Then by" in captions
    assert "and" not in captions


def test_the_export_buttons_say_what_they_write(panel: ExportPanel):
    buttons = [
        widget
        for widget in panel.findChildren(QPushButton)
        if widget.text().endswith("CSV...")
    ]

    assert len(buttons) == 4   # three exports, plus checking one written earlier
    for button in buttons:
        assert button.toolTip(), f"{button.text()} explains nothing"


def test_a_caption_carries_its_control_s_tooltip(panel: ExportPanel):
    """The word naming a control is what a user hovers, not the control."""
    captions = {
        widget.text(): widget.toolTip()
        for widget in panel.findChildren(QLabel)
    }

    assert captions["Modality"]
    assert captions["State"]
    assert captions["Break down by"]
    assert captions["Training"] == panel.train_share.toolTip()


# ---------------------------------------------------------------------------
# The series control
# ---------------------------------------------------------------------------

def test_the_series_control_opens_on_adaptive_polishing_alone(panel: ExportPanel):
    """The one filter that does not open on everything.

    A scan now holds reference and alignment images too, and a tab that
    silently counted them would disagree with create-training-csv, which
    defaults the same way.
    """
    ticked = {name for name, box in panel.series_boxes.items() if box.isChecked()}

    assert ticked == {"adaptive"}
    assert panel.controller.filters.series == ("adaptive",)


def test_the_series_switches_stand_two_above_two_as_on_curate(panel: ExportPanel):
    """Loose, they wrapped three and one, and the one lined up with nothing."""
    panel.resize(1440, 900)
    panel.show()
    _settle()
    at = {
        name: box.mapTo(panel, box.rect().topLeft())
        for name, box in panel.series_boxes.items()
    }

    assert at["adaptive"].y() == at["reference"].y()
    assert at["alignment"].y() == at["other"].y() > at["adaptive"].y()
    assert at["adaptive"].x() == at["alignment"].x()
    assert at["reference"].x() == at["other"].x() > at["adaptive"].x()


def test_the_series_switches_head_the_column_above_the_projects(panel: ExportPanel):
    """Which series an export covers is asked before anything about those
    images, and it is where Curate asks the same question of its lists."""
    panel.resize(1200, 800)
    panel.show()
    _settle()

    def top(widget) -> int:
        return widget.mapTo(panel, widget.rect().topLeft()).y()

    series = max(top(box) for box in panel.series_boxes.values())
    assert series < top(panel.projects_box) < top(panel.modality)
    assert top(panel.modality) < top(panel.state) < top(panel.label_state)
    assert not panel._filters_box.isAncestorOf(panel.series_block)


def test_every_series_has_a_box(panel: ExportPanel):
    assert set(panel.series_boxes) == set(SERIES)


def test_ticking_a_series_reaches_the_controller(panel: ExportPanel):
    panel.series_boxes["reference"].setChecked(True)

    assert set(panel.controller.filters.series) == {"adaptive", "reference"}


def test_unticking_a_series_reaches_the_controller(panel: ExportPanel):
    panel.series_boxes["adaptive"].setChecked(False)

    assert panel.controller.filters.series == ()


def test_the_series_control_is_not_also_offered_as_a_filter_row(panel: ExportPanel):
    """Three fixed choices, so it is a control rather than a list of values."""
    row = panel.filter_rows[0] if panel.filter_rows else None
    offered = (
        {row.property_box.itemData(i) for i in range(row.property_box.count())}
        if row is not None
        else set()
    )

    assert "series" not in offered


def test_the_buttons_that_move_files_are_kept_off_the_csv_run(panel: ExportPanel):
    """They move images and labels, not a list of paths. Sat among the CSV
    buttons either read as a fifth one of those, which is how a copy of
    gigabytes gets started by accident -- so they have a card of their own, and
    everything in the other writes or reads a list of paths."""
    csv_buttons = [
        button.text() for button in panel.csv_box.findChildren(QPushButton)
        if button.text().endswith("...")
    ]

    assert panel.files_button.text() == "Export images and labels..."
    assert panel.merge_button.text() == "Merge a copy back..."
    assert csv_buttons and all(text.endswith("CSV...") for text in csv_buttons)
    for button in (panel.files_button, panel.merge_button):
        assert panel.copy_box.isAncestorOf(button), button.text()
        assert not panel.csv_box.isAncestorOf(button), button.text()

def test_the_merge_button_says_how_a_merge_is_run(panel: ExportPanel):
    """Which collection to open and which folder to choose, and that it goes one way."""
    tip = panel.merge_button.toolTip()

    assert "open the original" in tip
    assert "AM_export.yaml" in tip
    assert "One way only" in tip


# ---------------------------------------------------------------------------
# The project filter, which has a control of its own
# ---------------------------------------------------------------------------

SESSIONS = [
    "20250428_AM/AutoLamella-2025-04-28-16-59",
    "20250428_AM/AutoLamella-2025-04-29-07-42",
    "20250512_AM_all/AutoLamella-2025-05-12-09-11",
    "Grid21_adaptive_milling",
]


def _projects_in(panel, values=None):
    """Fill the project box directly, as a scan would, and return its tree."""
    from am_segmentation_manager.gui.widgets.export_panel import _folder_of, _name_of

    values = SESSIONS if values is None else values
    panel._project_values = list(values)
    panel.project_tree.set_projects(
        [_folder_of(value) for value in values],
        [_name_of(value) for value in values],
        ticked=[True] * len(values),
        tooltips=values,
    )
    return panel.project_tree


def test_the_project_filter_groups_its_values_by_session(panel):
    """Its values are relative paths: a flat list of them is a run of
    timestamps with the session folder repeated on every row."""
    tree = _projects_in(panel)

    folders = [
        tree.topLevelItem(i).text(0)
        for i in range(tree.topLevelItemCount())
        if tree.project_of(tree.topLevelItem(i)) is None
    ]
    assert folders == ["20250428_AM", "20250512_AM_all"]


def test_a_project_at_the_scan_root_stays_at_the_top_level(panel):
    """Inventing a "(root)" row would name a directory that is not there."""
    tree = _projects_in(panel)

    tops = [
        tree.topLevelItem(i).text(0)
        for i in range(tree.topLevelItemCount())
        if tree.project_of(tree.topLevelItem(i)) is not None
    ]
    assert tops == ["Grid21_adaptive_milling"]


def test_unticking_a_project_keeps_the_whole_path(panel):
    """A row shows the name under its folder; the filter needs the identity,
    which is the relative path the breakdown keys on."""
    from qtpy.QtCore import Qt

    tree = _projects_in(panel)
    tree.project_item(2).setCheckState(0, Qt.Unchecked)
    _settle()

    kept = panel.controller.filters.choices["autolamella-project"]
    assert "20250512_AM_all/AutoLamella-2025-05-12-09-11" not in kept
    assert "20250428_AM/AutoLamella-2025-04-28-16-59" in kept


def test_every_project_ticked_is_stored_as_no_filter_at_all(panel):
    """The values on offer are recomputed from the records the other filters
    pass, so a stored list of all of them would quietly exclude a project that
    appeared afterwards -- from a rescan, or from widening another filter.
    Opening with everything ticked says nothing is excluded; this is what
    keeps that true underneath."""
    from qtpy.QtCore import Qt

    _projects_in(panel)
    panel._projects_changed()

    assert "autolamella-project" not in panel.controller.filters.choices

    panel.project_tree.project_item(0).setCheckState(0, Qt.Unchecked)
    _settle()
    assert panel.controller.filters.choices["autolamella-project"]

    panel.project_tree.project_item(0).setCheckState(0, Qt.Checked)
    _settle()
    assert "autolamella-project" not in panel.controller.filters.choices, (
        "ticking the last one back is not a filter on everything"
    )


def test_clearing_the_filters_puts_every_project_back_rather_than_none(panel):
    """Here every row ticked is what "not filtered" looks like, so clearing
    them would narrow the export to nothing at the press of a button called
    Clear."""
    from qtpy.QtCore import Qt

    tree = _projects_in(panel)
    tree.project_item(0).setCheckState(0, Qt.Unchecked)
    _settle()

    panel._clear_filters()
    _settle()

    assert tree.ticked_rows() == [0, 1, 2, 3]
    assert "autolamella-project" not in panel.controller.filters.choices


def test_the_project_filter_is_not_also_offered_as_a_filter_row(panel):
    """Two controls on one filter would disagree about it, and the row would
    be the one nobody could see was in force."""
    offered = {
        name
        for _heading, rows in panel.controller.filterable_groups()
        for name, _label in rows
    }

    assert "autolamella-project" not in offered


def test_the_project_filter_cannot_be_taken_away(panel):
    """Which projects an export covers is the first question anyone asks of
    one. As a row it was something you had to know to go and add, and once
    added it looked like a filter on the microscope or the sample."""
    from qtpy.QtWidgets import QPushButton

    from am_segmentation_manager.gui.widgets.export_panel import REMOVE_GLYPH

    box = panel.project_tree.parent()
    removes = [
        button
        for button in box.findChildren(QPushButton)
        if button.text() == REMOVE_GLYPH
    ]

    assert not removes


def _project_tree_panel(tmp_path) -> ExportPanel:
    """A panel with records, so the project box is filled from real values.

    Records matter: without them ``refresh`` returns before it refills the
    filters, and the fault these tests are about is in the refilling.
    """
    return _loaded_panel(tmp_path, name="AutoLamella-2025-04-28-16-59")


def test_unticking_a_folder_unticks_every_project_under_it(qt_app, tmp_path):
    """The fault this pins crashed the process.

    Acting on a tick refills the tree, and refilling clears it -- so doing it
    from inside ``itemChanged`` destroyed the item whose signal was being
    emitted while Qt was still propagating the tick to the folder's other
    children. On the reference collection one project of nine came out ticked,
    and the heap was corrupted on the way, which shows as a segfault a few
    gestures later rather than there and then.
    """
    from qtpy.QtCore import Qt

    panel = _project_tree_panel(tmp_path)
    assert panel._project_values, "the tree is empty, so this proves nothing"

    for index in range(panel.project_tree.topLevelItemCount()):
        panel.project_tree.topLevelItem(index).setCheckState(0, Qt.Unchecked)
    _settle()

    assert panel.project_tree.ticked_rows() == []


def test_nothing_is_rebuilt_while_a_tick_is_still_being_delivered(qt_app, tmp_path):
    """The rule the curation lists already keep, now kept here too: a list
    rebuilt from inside its own signal is a use after free."""
    from qtpy.QtCore import Qt

    from am_segmentation_manager.gui.widgets.lists import ProjectTree

    panel = _project_tree_panel(tmp_path)

    inside = []
    original = ProjectTree.set_projects

    def watched(self, *args, **kwargs):
        import traceback

        inside.append(
            any("_item_changed" in frame.name for frame in traceback.extract_stack())
        )
        return original(self, *args, **kwargs)

    ProjectTree.set_projects = watched
    try:
        panel.project_tree.project_item(0).setCheckState(0, Qt.Unchecked)
        _settle()
    finally:
        ProjectTree.set_projects = original

    assert inside, "the tree was never refilled, so this proves nothing"
    assert not any(inside), "the tree was rebuilt from inside its own signal"


def test_a_gesture_that_moves_many_ticks_recounts_once(qt_app, tmp_path):
    """Unticking a folder is one signal per project under it, and each recount
    sweeps every record in the collection."""
    from qtpy.QtCore import Qt

    panel = _project_tree_panel(tmp_path)
    counts = []
    original = panel._controls_changed
    panel._recount.timeout.disconnect(original)
    panel._recount.timeout.connect(lambda: (counts.append(None), original())[1])

    for index in range(panel.project_tree.topLevelItemCount()):
        panel.project_tree.topLevelItem(index).setCheckState(0, Qt.Unchecked)
    _settle()

    assert len(counts) == 1


def test_a_filter_row_is_always_a_flat_list(panel):
    """The one property whose values had a hierarchy has a control of its own
    now, so a row draws a list and nothing else -- where before it carried a
    tree, a set of folds and the flag deciding which to show."""
    row = panel.filter_rows[0]
    row._property = "microscope"

    row.show_values(["Helios 5", "Aquilos 2"], set(), "Microscope model")
    row.show_filled()

    assert not hasattr(row, "tree"), "a row has no tree to keep in step"
    assert row.values.count() == 2
    assert row.ticked_values() == []


def test_unticking_every_project_matches_nothing_rather_than_everything(panel):
    """The button says None and has to mean it.

    An empty choice is how a filter *row* says it has stopped filtering, so
    storing one here would widen the export to every project at the press of a
    button called None -- and the refill would then tick them all back, which
    is a control that visibly undoes itself.
    """
    _projects_in(panel)
    panel._tick_all_projects(False)
    _settle()

    assert panel.project_tree.ticked_rows() == []
    assert panel.controller.filters.choices["autolamella-project"] == []
    assert panel.controller.selected_records() == []


def test_a_project_that_appears_later_joins_an_unfiltered_export(panel):
    """With everything ticked the answer is everything, not the list of names
    that happened to be on screen when it was last looked at."""
    _projects_in(panel, SESSIONS[:2])
    panel._projects_changed()

    _projects_in(panel, SESSIONS)
    panel._projects_changed()

    assert panel.project_tree.ticked_rows() == [0, 1, 2, 3]
    assert "autolamella-project" not in panel.controller.filters.choices


def test_a_filter_row_offers_tick_all_and_none(panel):
    """A property with thirty values is one nobody wants to click through, and
    the two buttons are what the project box and the curation lists already
    offer for the same reason."""
    from qtpy.QtWidgets import QPushButton

    row = panel.filter_rows[0]
    row._property = "microscope"
    row.show_values(["Helios 5", "Aquilos 2"], set(), "Microscope model")
    row.show_filled()

    buttons = row.buttons.findChildren(QPushButton)
    assert [button.text() for button in buttons] == ["Tick all", "None"]

    buttons[0].click()
    assert sorted(row.ticked_values()) == ["Aquilos 2", "Helios 5"]


def test_the_table_sits_in_from_the_edge_and_the_controls_above(panel: ExportPanel):
    """An outline running along the card's edge read as part of the card."""
    margins = panel.views.contentsMargins()

    assert margins.left() > 0 and margins.top() > 0
