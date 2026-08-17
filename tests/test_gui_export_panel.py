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
        QLabel,
        QListWidget,
        QPushButton,
        QWidget,
    )

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


def test_the_acquisition_filters_are_hidden_until_the_headers_are_read(
    panel: ExportPanel,
):
    """A column of empty boxes reads as broken, and none of them can be used
    before the values exist to choose from."""
    assert panel.metadata_scroller.isHidden()

    panel._load_metadata()

    assert not panel.metadata_scroller.isHidden()


def test_a_filter_is_labelled_with_its_unit_spelled_properly(panel: ExportPanel):
    labels = {
        panel.metadata_box.layout().itemAt(index).widget().text()
        for index in range(panel.metadata_box.layout().count())
        if isinstance(panel.metadata_box.layout().itemAt(index).widget(), QLabel)
    }

    assert "Acceleration voltage (kV)" in labels
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
    panel.label_state.setCurrentText("empty (placeholder)")

    assert panel.controller.filters.label == "empty (placeholder)"


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
    first = panel.breakdown.findData("variant")
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


@needs_plots
def test_one_property_is_charted_as_bars(qt_app, tmp_path):
    panel = measurable(qt_app, tmp_path)
    panel.breakdown.setCurrentIndex(panel.breakdown.findData("modality"))

    panel._chart()

    assert len(panel._charts) == 1
    # Bars have no row to take a share of, so the shading toggle is not offered.
    assert not panel._charts[0].normalise.isVisible()
    panel._charts[0].close()


@needs_plots
def test_two_properties_are_charted_as_a_shadeable_grid(qt_app, tmp_path):
    panel = measurable(qt_app, tmp_path)
    panel.breakdown.setCurrentIndex(panel.breakdown.findData("modality"))
    panel.cross.setCurrentIndex(panel.cross.findData("state"))

    panel._chart()

    dialog = panel._charts[0]
    assert dialog.normalise.isVisibleTo(dialog)
    dialog.close()


@needs_plots
def test_toggling_the_shading_redraws_rather_than_stacking_canvases(qt_app, tmp_path):
    """An old canvas left in the layout keeps its figure alive, and a session
    of charts would then hold every one of them."""
    panel = measurable(qt_app, tmp_path)
    panel.breakdown.setCurrentIndex(panel.breakdown.findData("modality"))
    panel.cross.setCurrentIndex(panel.cross.findData("state"))
    panel._chart()
    dialog = panel._charts[0]
    first = dialog._canvas

    dialog.normalise.setChecked(True)

    assert dialog._canvas is not first
    assert dialog._layout.count() == 3   # toolbar, canvas, controls
    dialog.close()


@needs_plots
def test_charting_an_unmeasured_breakdown_says_so_rather_than_drawing_nothing(
    qt_app, tmp_path, monkeypatch
):
    panel = measurable(qt_app, tmp_path)
    panel.breakdown.setCurrentIndex(panel.breakdown.findData("crack-count"))
    asked = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.information",
        lambda *args, **kwargs: asked.append(args[2]),
    )

    panel._chart()

    assert panel._charts == []
    assert "measured" in asked[0]


@needs_plots
def test_a_closed_chart_is_let_go_of(qt_app, tmp_path):
    """Held only so a modeless dialog outlives the call that showed it."""
    panel = measurable(qt_app, tmp_path)
    panel._chart()

    panel._charts[0].close()

    assert panel._charts == []


def test_charting_without_matplotlib_explains_rather_than_failing(
    qt_app, tmp_path, monkeypatch
):
    panel = measurable(qt_app, tmp_path)
    monkeypatch.setattr("am_segmentation_manager.plots.has_plot_support", lambda: False)
    told = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.information",
        lambda *args, **kwargs: told.append(args[2]),
    )

    panel._chart()

    assert panel._charts == []
    assert "pip install" in told[0]


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
        panel.chart_button,
        panel.train_share, panel.seeded, panel.seed, panel.prefix,
        panel.load_button, panel.check_button,
    ):
        assert widget.toolTip(), f"{widget} explains nothing"


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
