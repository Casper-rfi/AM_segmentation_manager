"""The window's appearance, and the layout it remembers between sessions.

What is worth pinning here is not which colour was chosen -- those are meant to
be changed -- but the two properties a change must not break: that a locked
field stays readable, and that the theme never overrides the palette the rest of
the window reasons about.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

# Must be set before any Qt module is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("qtpy") is None, reason="Qt binding not installed"
)

if importlib.util.find_spec("qtpy") is not None:
    from qtpy.QtCore import QSettings
    from qtpy.QtGui import QColor, QPalette
    from qtpy.QtWidgets import QApplication, QLineEdit, QSplitter, QWidget

    from am_segmentation_manager.gui.widgets import theme


@pytest.fixture(scope="session")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def themed(qt_app):
    """Apply the theme, and put the application back afterwards.

    The QApplication is session-wide, so a test that left the stylesheet on
    would change how every later widget test renders.
    """
    style = qt_app.styleSheet()
    palette = qt_app.palette()
    theme.apply_theme(qt_app)
    yield qt_app
    qt_app.setStyleSheet(style)
    qt_app.setPalette(palette)


# ---------------------------------------------------------------------------
# The theme
# ---------------------------------------------------------------------------

def test_the_window_is_not_left_on_the_platform_grey(themed):
    assert themed.palette().color(QPalette.Window).name() == theme.WINDOW
    assert themed.palette().color(QPalette.Base).name() == theme.BASE


def test_the_style_is_the_same_one_everywhere(themed):
    """PyQt5 here and PySide6 on the user's machine, so the style is chosen
    rather than inherited from the platform.

    The stylesheet has to come off first: with one set, Qt wraps the real style
    in a QStyleSheetStyle and the name underneath is no longer reachable.
    """
    themed.setStyleSheet("")

    assert themed.style().objectName().lower() == "fusion"


def test_the_stylesheet_applies_without_error(themed):
    widget = QWidget()
    widget.setStyleSheet(theme.stylesheet())

    assert widget.styleSheet() == theme.stylesheet()
    assert themed.styleSheet()


def _darkest(widget) -> int:
    """Return the darkest pixel the widget draws, as a grey level."""
    image = widget.grab().toImage()
    return min(
        QColor(image.pixel(x, y)).lightness()
        for y in range(image.height())
        for x in range(image.width())
    )


def test_a_locked_field_still_draws_its_value_darkly(themed):
    """The Add data tab opens locked and is mostly used for reading, so a value
    must not fade into the background.

    Asserted on the pixels rather than on the palette: a stylesheet rule beats
    the palette, so a palette that says the right thing proves nothing about
    what is on screen. This is exactly how the theme first broke it.
    """
    editor = QLineEdit("Rosalind Franklin Institute")
    editor.resize(300, 26)
    editor.setEnabled(False)

    assert _darkest(editor) < 80


def test_a_locked_field_still_reads_as_locked(themed):
    """Readable is not the same as editable-looking: the background dulls."""
    editor = QLineEdit("value")
    editor.resize(300, 26)
    enabled = QColor(editor.grab().toImage().pixel(150, 13))
    editor.setEnabled(False)
    disabled = QColor(editor.grab().toImage().pixel(150, 13))

    assert disabled != enabled


def test_a_badge_says_what_it_means_rather_than_naming_a_colour(themed):
    assert set(theme.BADGE_STYLES) == {"warning", "error", "info"}
    assert all(style.startswith("color:") for style in theme.BADGE_STYLES.values())


# ---------------------------------------------------------------------------
# Remembering the layout
# ---------------------------------------------------------------------------

def test_a_splitter_comes_back_the_size_it_was_left(tmp_path: Path, qt_app):
    """The mechanism the window uses for all three of its splitters."""
    settings = QSettings(str(tmp_path / "preferences.ini"), QSettings.IniFormat)
    splitter = QSplitter()
    splitter.addWidget(QWidget())
    splitter.addWidget(QWidget())
    splitter.setSizes([120, 480])
    settings.setValue("splitter/test", splitter.saveState())

    restored = QSplitter()
    restored.addWidget(QWidget())
    restored.addWidget(QWidget())
    from qtpy.QtCore import QByteArray

    restored.restoreState(settings.value("splitter/test", None, type=QByteArray))

    assert restored.sizes() == splitter.sizes()
