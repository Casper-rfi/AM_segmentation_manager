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
    from qtpy.QtCore import QEvent, QPointF, QSettings, Qt
    from qtpy.QtGui import QColor, QEnterEvent, QPalette
    from qtpy.QtWidgets import (
        QApplication,
        QLineEdit,
        QListWidget,
        QPushButton,
        QSplitter,
        QWidget,
    )

    from am_segmentation_manager.gui.widgets import theme
    from am_segmentation_manager.gui.widgets.common import GripSplitter


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


def test_a_one_character_button_still_draws_its_character(themed):
    """The stylesheet spends 12px a side on button padding, which is wider than
    a button whose whole label is one glyph -- so the label was clipped away
    and the button drew blank. Asserted on the pixels, because the button
    reported its text correctly the whole time it was showing nothing."""
    from am_segmentation_manager.gui.widgets.export_panel import REMOVE_GLYPH

    plain = QPushButton(REMOVE_GLYPH)
    plain.setFixedWidth(26)
    compact = QPushButton(REMOVE_GLYPH)
    compact.setObjectName(theme.COMPACT_BUTTON)
    compact.setFixedWidth(26)

    assert _darkest(compact) < _darkest(plain), "the padding is still clipping it"


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


def _ticked_list(current: int) -> QListWidget:
    """Return a three-row ticked list with *current* selected."""
    widget = QListWidget()
    widget.addItems(["one", "two", "three"])
    for row in range(widget.count()):
        item = widget.item(row)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
    widget.setCurrentRow(current)
    widget.resize(300, 90)
    return widget


def _indicator_darkness(widget, row: int) -> int:
    """Return the darkest pixel in a row's checkbox column."""
    image = widget.grab().toImage()
    top = 4 + row * 22
    return min(
        QColor(image.pixel(x, y)).lightness()
        for y in range(top, top + 20)
        for x in range(4, 24)
    )


def test_a_tick_stays_dark_on_the_row_being_looked_at(themed):
    """The ticks decide what a state change moves, so one that fades on the
    current row is the one row where it matters most.

    A solid accent fill left the indicator at mid grey on blue; the row is
    tinted instead, which keeps Qt's own indicator legible.
    """
    widget = _ticked_list(current=0)

    assert _indicator_darkness(widget, 0) <= _indicator_darkness(widget, 1) + 10


def test_the_row_being_looked_at_is_still_obvious(themed):
    """Legible is not the same as indistinguishable: it still has to stand out."""
    widget = _ticked_list(current=0)
    image = widget.grab().toImage()

    assert QColor(image.pixel(200, 12)) != QColor(image.pixel(200, 34))


def test_a_badge_says_what_it_means_rather_than_naming_a_colour(themed):
    assert set(theme.BADGE_STYLES) == {"warning", "error", "info"}
    assert all(style.startswith("color:") for style in theme.BADGE_STYLES.values())


# ---------------------------------------------------------------------------
# Resizable divisions
# ---------------------------------------------------------------------------

def _handle(app, orientation):
    """Return the handle of a two-pane splitter, laid out and ready to draw.

    Shown, because a splitter gives its handle a size when it lays itself out
    and not before -- an unshown one reports a 6x6 handle wherever it is asked.
    """
    splitter = GripSplitter(orientation)
    splitter.addWidget(QWidget())
    splitter.addWidget(QWidget())
    splitter.resize(200, 200)
    splitter.show()
    app.processEvents()
    return splitter, splitter.handle(1)


def _colour_at(widget, x: int, y: int) -> str:
    return QColor(widget.grab().toImage().pixel(x, y)).name()


@pytest.mark.parametrize("orientation", [Qt.Horizontal, Qt.Vertical])
def test_a_division_that_moves_says_so(themed, orientation):
    """A splitter handle was a hairline like every other rule in the window, so
    the only way to find one was to sweep the pointer along it until the cursor
    changed. Asserted on the pixels, the grip being painted rather than styled.

    Both orientations, because the grip runs along the handle: one turned the
    wrong way is a bar across the division rather than a grip on it, and it
    would still be a coloured mark in the middle of the handle.
    """
    splitter, handle = _handle(themed, orientation)
    middle = (handle.width() // 2, handle.height() // 2)
    along = (
        (handle.width() // 2, 2) if orientation == Qt.Horizontal
        else (2, handle.height() // 2)
    )

    assert _colour_at(handle, *middle) == theme.GRIP
    assert _colour_at(handle, *along) == theme.WINDOW, "the grip runs the whole way"


def test_the_grip_reacts_to_the_pointer(themed):
    """The same accent every other control takes when it is pointed at."""
    splitter, handle = _handle(themed, Qt.Horizontal)
    middle = (handle.width() // 2, handle.height() // 2)
    point = QPointF(handle.rect().center())

    QApplication.sendEvent(handle, QEnterEvent(point, point, point))
    assert _colour_at(handle, *middle) == theme.ACCENT

    QApplication.sendEvent(handle, QEvent(QEvent.Leave))
    assert _colour_at(handle, *middle) == theme.GRIP


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


# ---------------------------------------------------------------------------
# The window icon
# ---------------------------------------------------------------------------

def test_the_icon_ships_inside_the_package(qt_app):
    """It lives in `data/`, not at the repository root, so installing keeps it.

    Loaded rather than merely located: a file Qt cannot decode produces a null
    icon and no error, which looks identical to a correct one until the window
    opens with a blank tile.
    """
    from qtpy.QtGui import QIcon

    from am_segmentation_manager.gui.app import ICON_PATH

    assert ICON_PATH.exists()
    assert not QIcon(str(ICON_PATH)).isNull()


def test_a_missing_icon_does_not_stop_the_window_opening(qt_app, monkeypatch):
    """A source checkout without the asset must still run."""
    from am_segmentation_manager.gui import app

    monkeypatch.setattr(app, "ICON_PATH", Path("/nowhere/absent.png"))
    app._apply_icon(qt_app)


def test_the_icon_is_set_on_the_application_so_dialogs_inherit_it(qt_app):
    from am_segmentation_manager.gui.app import _apply_icon

    _apply_icon(qt_app)

    assert not qt_app.windowIcon().isNull()


def test_the_startup_scan_waits_for_the_window_to_appear(qt_app, tmp_path, monkeypatch):
    """``show()`` only asks for a paint. Nothing is drawn until ``exec()``
    starts delivering events, so a scan started before it ran against a window
    that had never appeared -- seconds of blank grey indistinguishable from a
    hang."""
    from am_segmentation_manager.gui import app

    order: list[str] = []

    class FakeWindow:
        def __init__(self, controller):
            self.project_panel = type("P", (), {"set_path": lambda self, p: None})()

        def show(self):
            order.append("show")

        def scan(self, path):
            order.append("scan")

    monkeypatch.setattr(app, "CurationWindow", FakeWindow, raising=False)
    monkeypatch.setattr(
        "am_segmentation_manager.gui.widgets.main_window.CurationWindow", FakeWindow
    )
    monkeypatch.setattr(
        qt_app, "exec", lambda: order.append("exec") or qt_app.processEvents() or 0
    )
    monkeypatch.setattr(qt_app, "exec_", qt_app.exec, raising=False)

    app.run(start_path=tmp_path)

    assert order == ["show", "exec", "scan"], (
        "the scan must be queued behind the event loop, not run before it"
    )
