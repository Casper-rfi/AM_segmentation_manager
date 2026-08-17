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
    from qtpy.QtWidgets import QApplication, QLabel

    from am_segmentation_manager.gui.export_controller import (
        DEFAULT_PREFIX,
        ExportController,
    )
    from am_segmentation_manager.gui.widgets.export_panel import ExportPanel


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
