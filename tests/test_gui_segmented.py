"""The control that picks one of a small closed set.

A run of joined segments answers "which of these", where a checked tool button
elsewhere answers "is this on". The two must not look alike, which is the one
thing here that can only be asserted on pixels -- see `docs/display.md`.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

from conftest import destroy

# Must be set before any Qt module is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HAS_QT = importlib.util.find_spec("qtpy") is not None

pytestmark = pytest.mark.skipif(not HAS_QT, reason="Qt binding not installed")

if HAS_QT:
    from qtpy.QtGui import QColor, QImage
    from qtpy.QtWidgets import QApplication, QToolButton

    from am_segmentation_manager.gui.widgets import theme
    from am_segmentation_manager.gui.widgets.controls import SegmentedControl

SEGMENTS = [("sem", "SEM", "the SEM"), ("fib", "FIB", "the FIB"), ("plot", "Plots", "plots")]


@pytest.fixture(scope="session")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def themed(qt_app):
    """Apply the theme, and put the application back afterwards.

    The QApplication is session-wide, so a test that left the stylesheet on
    would change how every later widget test renders.
    """
    previous = qt_app.styleSheet()
    theme.apply_theme(qt_app)
    yield qt_app
    qt_app.setStyleSheet(previous)


@pytest.fixture
def control(qt_app) -> SegmentedControl:
    control = SegmentedControl(SEGMENTS)
    yield control
    destroy(control)


# ---------------------------------------------------------------------------
# Choosing
# ---------------------------------------------------------------------------

def test_nothing_is_chosen_until_something_is(control):
    assert control.current() is None


def test_choosing_a_segment_reports_its_key(control):
    seen = []
    control.chosen.connect(seen.append)

    control.buttons["fib"].setChecked(True)

    assert seen == ["fib"]
    assert control.current() == "fib"


def test_choosing_one_segment_lets_go_of_the_other(control):
    control.buttons["sem"].setChecked(True)

    control.buttons["plot"].setChecked(True)

    assert not control.buttons["sem"].isChecked()
    assert control.current() == "plot"


def test_reflecting_a_choice_made_elsewhere_reports_nothing(control):
    """Or the panel that set it hears its own change back and round again."""
    seen = []
    control.chosen.connect(seen.append)

    control.set_current("fib")

    assert control.current() == "fib"
    assert not seen


def test_asking_for_a_segment_that_is_not_there_raises(control):
    """A caller with the wrong vocabulary should hear about it here, not show
    a control that quietly did nothing."""
    with pytest.raises(KeyError):
        control.set_current("nothing")


def test_a_segment_can_be_disabled_on_its_own(control):
    """Which is how a view the frame has nothing to show for is greyed out."""
    control.buttons["plot"].setEnabled(False)

    assert not control.buttons["plot"].isEnabled()
    assert control.buttons["sem"].isEnabled()


# ---------------------------------------------------------------------------
# Joined into one run
# ---------------------------------------------------------------------------

def test_the_ends_of_the_run_are_marked_for_the_stylesheet(control):
    names = [button.objectName() for button in control.buttons.values()]

    assert names == ["segmentFirst", "segmentMiddle", "segmentLast"]


def test_a_run_of_two_has_no_middle(qt_app):
    control = SegmentedControl(SEGMENTS[:2])

    assert [b.objectName() for b in control.buttons.values()] == [
        "segmentFirst", "segmentLast",
    ]
    destroy(control)


def test_a_run_of_one_is_rounded_on_both_ends(qt_app):
    control = SegmentedControl(SEGMENTS[:1])

    assert [b.objectName() for b in control.buttons.values()] == ["segmentOnly"]
    destroy(control)


def test_the_segments_are_not_spaced_apart(control):
    """They share their borders, which is what makes the run one control."""
    assert control.layout().spacing() == 0
    assert control.layout().contentsMargins().left() == 0


def test_the_stylesheet_names_every_segment_position(themed):
    """Qt stylesheets have no wildcard, so a name left out of a rule is a
    segment silently drawn as an ordinary button."""
    sheet = theme.stylesheet(theme.active_accent())
    for name in ("segmentOnly", "segmentFirst", "segmentMiddle", "segmentLast"):
        assert sheet.count(f"QToolButton#{name}:checked") == 1, name


# ---------------------------------------------------------------------------
# How it draws
# ---------------------------------------------------------------------------

def _fill(button) -> QColor:
    """Return the colour a laid-out button is filled with, away from its text."""
    image = QImage(button.size(), QImage.Format_ARGB32)
    image.fill(QColor("magenta"))
    button.render(image)
    return image.pixelColor(4, button.height() // 2)


def _laid_out(control):
    control.resize(control.sizeHint())
    control.show()
    return control


def test_the_chosen_segment_is_filled_not_tinted(themed):
    """A checked tool button elsewhere is tinted and means "this is on". A
    chosen segment means "this one of these", and a reader has to be able to
    tell the two apart across the room."""
    control = _laid_out(SegmentedControl(SEGMENTS))
    control.set_current("sem")

    plain = QToolButton()
    plain.setText("SEM")
    plain.setCheckable(True)
    plain.setChecked(True)
    plain.resize(60, 26)
    plain.show()

    assert _fill(control.buttons["sem"]) == QColor(theme.active_accent().colour)
    assert _fill(plain) != QColor(theme.active_accent().colour)
    destroy(control)
    destroy(plain)


def test_the_segments_not_chosen_stay_readable(themed):
    control = _laid_out(SegmentedControl(SEGMENTS))
    control.set_current("sem")

    assert _fill(control.buttons["fib"]) == QColor(theme.BUTTON)
    destroy(control)


def test_a_disabled_segment_is_still_drawn(themed):
    """Greyed out, not blank: it says the view exists and has nothing here."""
    control = _laid_out(SegmentedControl(SEGMENTS))
    control.buttons["plot"].setEnabled(False)

    assert _fill(control.buttons["plot"]) == QColor(theme.BUTTON)
    destroy(control)
