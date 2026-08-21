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
    from qtpy.QtGui import QFontMetricsF
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

    from am_segmentation_manager.aggregate import AREA_BINS_PER_DECADE
    from am_segmentation_manager.gui.export_controller import (
        ANY_LABEL,
        DEFAULT_PREFIX,
        ExportController,
    )
    from am_segmentation_manager.gui.widgets.export_panel import NO_CROSS, ExportPanel
    from am_segmentation_manager.scanner import Scanner
    from conftest import build_dataset


@pytest.fixture(scope="session")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def panel(qt_app) -> ExportPanel:
    return ExportPanel(ExportController())


def _tick(widget, row: int, on: bool = True) -> None:
    """Tick a filter row through the widget, as a user would."""
    widget.item(row).setCheckState(Qt.Checked if on else Qt.Unchecked)


def _loaded_panel(tmp_path, **kwargs) -> ExportPanel:
    """A panel over a real two-lamella scan, as the window builds it."""
    pytest.importorskip("tifffile")
    build_dataset(tmp_path, lamellae=2, acquisitions=2, labels="annotated", **kwargs)
    controller = ExportController()
    controller.set_projects(Scanner(tmp_path).build_index().all_projects(), tmp_path)
    panel = ExportPanel(controller)
    panel.refresh()
    return panel


def test_the_acquisition_filters_are_hidden_until_the_headers_are_read(
    qt_app, tmp_path,
):
    """A column of empty boxes reads as broken, and none of them can be used
    before the values exist to choose from."""
    panel = _loaded_panel(tmp_path)

    assert panel.filter_lists["microscope"].isHidden()

    panel._load_metadata()

    # Still hidden, but now because the synthetic TIFFs all say the same thing
    # about it rather than because the headers were never read.
    assert panel.controller.metadata_loaded()


def test_a_filter_that_cannot_narrow_anything_is_not_shown(qt_app, tmp_path):
    """One project and one value means a list that filters nothing. The
    breakdowns cover far more properties than any one dataset varies along."""
    panel = _loaded_panel(tmp_path)

    assert not panel.filter_lists["lamella"].isHidden()   # two of them
    assert panel.filter_lists["project"].isHidden()       # only one


def test_a_ticked_filter_stays_visible_even_when_it_offers_one_value(
    qt_app, tmp_path,
):
    """A list that vanished once used could not be seen or cleared, and would
    go on narrowing everything under it."""
    panel = _loaded_panel(tmp_path)
    lamellae = panel.filter_lists["lamella"]
    _tick(lamellae, 0)

    assert not lamellae.isHidden()
    assert len(panel.controller.filters.choices["lamella"]) == 1


def test_ticking_a_lamella_narrows_the_table_under_it(qt_app, tmp_path):
    """The filter and the breakdown are one selection, so the table has to move
    when a filter does."""
    panel = _loaded_panel(tmp_path)
    before = sum(panel.controller.breakdown("modality").values())

    _tick(panel.filter_lists["lamella"], 0)

    assert sum(panel.controller.breakdown("modality").values()) < before


def test_the_caption_says_how_much_of_a_list_is_ticked(qt_app, tmp_path):
    """The lists are four rows over values that run to dozens, so what is
    ticked is usually scrolled out of sight -- which is how a filter stays on
    by accident."""
    panel = _loaded_panel(tmp_path)

    _tick(panel.filter_lists["lamella"], 0)

    assert panel.filter_captions["lamella"].text() == "Lamella (1 of 2 ticked)"


def test_clearing_the_filters_unticks_every_list(qt_app, tmp_path):
    """Ticks scroll out of sight, so there has to be one control that undoes
    all of them without hunting for which are on."""
    panel = _loaded_panel(tmp_path)
    _tick(panel.filter_lists["lamella"], 0)
    assert panel.controller.filters.choices

    panel.clear_button.click()

    assert panel.controller.filters.choices == {}
    assert panel.filter_lists["lamella"].ticked_values() == []
    assert panel.filter_captions["lamella"].text() == "Lamella (2)"


def test_clearing_leaves_the_named_controls_alone(qt_app, tmp_path):
    """They always hold one of their choices, so there is no cleared state to
    put them into -- and "all" is not what somebody clearing filters means to
    lose if they had chosen corrected."""
    panel = _loaded_panel(tmp_path)
    panel.state.setCurrentText("corrected")

    panel.clear_button.click()

    assert panel.state.currentText() == "corrected"


def test_a_long_filter_row_elides_rather_than_scrolling_sideways(qt_app, tmp_path):
    """Lamella rows are "project / lamella" and differ at the end. A list that
    scrolls instead of eliding shows a column of identical project prefixes."""
    panel = _loaded_panel(tmp_path)
    lamellae = panel.filter_lists["lamella"]

    assert lamellae.textElideMode() == Qt.ElideMiddle
    assert lamellae.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert lamellae.item(0).toolTip() == lamellae.item(0).text()


def test_a_filter_is_labelled_with_its_unit_spelled_properly(panel: ExportPanel):
    """The label is the breakdown's own, so a property is named the same way in
    the filter list and in the dropdown that counts by it."""
    labels = {caption.text() for caption in panel.filter_captions.values()}

    assert "Accelerating voltage (kV)" in labels
    assert "Beam current (pA)" in labels


def test_the_split_controls_start_where_the_defaults_are(panel: ExportPanel):
    training = panel.controller.training

    assert panel.train_share.value() == training.train_ratio * 100
    assert panel.seed.value() == training.seed
    assert panel.prefix.placeholderText() == training.prefix


def test_a_chosen_share_becomes_a_ratio(panel: ExportPanel):
    panel.train_share.setValue(65)

    assert panel.controller.training.train_ratio == 0.65


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
        assert widget.parent() is filters, f"{widget} narrows from outside Filters"

    narrowing = (QComboBox, QCheckBox, QListWidget)
    for widget in panel._export_box.findChildren(QWidget):
        assert not isinstance(widget, narrowing) or widget in (
            panel.seeded,   # a checkbox, but it only makes the shuffle repeatable
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
    panel = ExportPanel(controller)

    panel.breakdown.setCurrentIndex(panel.breakdown.findData("variant"))
    panel.refresh()

    assert panel.table.toPlainText().startswith("Images by imaging condition")


# ---------------------------------------------------------------------------
# Stepping through the list, and cross-referencing two properties
# ---------------------------------------------------------------------------

def test_the_table_is_drawn_in_a_font_its_columns_can_line_up_in(
    panel: ExportPanel,
):
    """reporting.py pads its columns with spaces. In the window's own
    proportional font that padding is not alignment, and the table reads as a
    ragged list rather than as rows."""
    metrics = QFontMetricsF(panel.table.font())

    assert metrics.horizontalAdvance("i") == metrics.horizontalAdvance("W")


def test_the_arrow_keys_step_through_the_list(panel: ExportPanel):
    """Whether Qt's own keys step or open the popup is a style hint's to
    answer, and the binding here is not the one the window ships under."""
    panel.breakdown.setCurrentIndex(panel.breakdown.findData("modality"))

    QTest.keyClick(panel.breakdown, Qt.Key_Down)
    assert panel.breakdown.currentData() == "state"

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
    panel = ExportPanel(controller)

    panel.breakdown.setCurrentIndex(panel.breakdown.findData("modality"))
    panel.refresh()
    assert panel.table.toPlainText().startswith("Images by modality")

    panel.cross.setCurrentIndex(panel.cross.findData("state"))
    panel.refresh()

    text = panel.table.toPlainText()
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
    return ExportPanel(controller)


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

    text = panel.table.toPlainText()
    assert "needs the labels measured" in text
    assert "no-label" not in text


def test_measuring_fills_the_table_in_and_spends_the_button(qt_app, tmp_path):
    panel = measurable(qt_app, tmp_path)
    panel.breakdown.setCurrentIndex(panel.breakdown.findData("crack-count"))

    panel.controller.load_measurements()
    panel.refresh()

    assert "needs the labels measured" not in panel.table.toPlainText()
    assert panel.measure_button.text() == "Labels measured"
    assert not panel.measure_button.isEnabled()


def test_the_prompt_names_both_halves_of_a_cross_reference(qt_app, tmp_path):
    panel = measurable(qt_app, tmp_path)
    panel.breakdown.setCurrentIndex(panel.breakdown.findData("modality"))
    panel.cross.setCurrentIndex(panel.cross.findData("has-crack"))
    panel.refresh()

    assert "cracked (yes / no)" in panel.table.toPlainText()


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
    assert panel.chart_view.normalise.isHidden()


@needs_plots
def test_two_properties_are_charted_as_a_shadeable_grid(qt_app, tmp_path):
    panel = measurable(qt_app, tmp_path)
    panel.breakdown.setCurrentIndex(panel.breakdown.findData("modality"))
    panel.cross.setCurrentIndex(panel.cross.findData("state"))

    panel.as_chart.setChecked(True)

    assert not panel.chart_view.normalise.isHidden()


@needs_plots
def test_toggling_the_shading_redraws_rather_than_stacking_canvases(qt_app, tmp_path):
    """An old canvas left in the layout keeps its figure alive, and a session
    of charts would then hold every one of them."""
    panel = measurable(qt_app, tmp_path)
    panel.breakdown.setCurrentIndex(panel.breakdown.findData("modality"))
    panel.cross.setCurrentIndex(panel.cross.findData("state"))
    panel.as_chart.setChecked(True)
    first = panel.chart_view._canvas

    panel.chart_view.normalise.setChecked(True)

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
        panel.train_share, panel.seeded, panel.seed, panel.prefix,
        panel.load_button, panel.check_button,
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
    filters += list(panel.filter_lists.values())

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
    assert captions["Workflow state"]
    assert captions["Break down by"]
    assert captions["Training"] == panel.train_share.toolTip()
