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

from conftest import destroy

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
    from am_segmentation_manager.gui.widgets.controls import GripSplitter


#: Every accent the theme offers, by key; parametrised with ``indirect=True``, a
#: test receives the application drawn in each. Written out rather than read from
#: the theme, which is not importable without Qt when the tests are collected.
ACCENT_KEYS = ("orange", "blue", "green")


@pytest.fixture
def themed(qt_app, request):
    """Apply the theme, in the accent a test is parametrised with, and put the
    application back afterwards.

    The QApplication is session-wide, so a test that left the stylesheet on
    would change how every later widget test renders -- and a test that left
    the accent on would change what every later painted widget draws.
    """
    style = qt_app.styleSheet()
    palette = qt_app.palette()
    previous = theme.active_accent()
    key = getattr(request, "param", theme.DEFAULT_ACCENT)
    theme.apply_theme(qt_app, theme.ACCENTS[key])
    yield qt_app
    theme.apply_theme(qt_app, previous)
    qt_app.setStyleSheet(style)
    qt_app.setPalette(palette)


def _accent_colour() -> QColor:
    return QColor(theme.active_accent().colour)


# ---------------------------------------------------------------------------
# The theme
# ---------------------------------------------------------------------------

def test_the_window_is_not_left_on_the_platform_grey(themed):
    assert themed.palette().color(QPalette.Window).name() == theme.WINDOW
    assert themed.palette().color(QPalette.Base).name() == theme.BASE


def _contrast(foreground: str, background: str) -> float:
    """Return the WCAG contrast ratio between two hex colours."""
    def luminance(colour: str) -> float:
        channels = []
        for value in QColor(colour).getRgbF()[:3]:
            channels.append(value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4)
        red, green, blue = channels
        return 0.2126 * red + 0.7152 * green + 0.0722 * blue

    lighter, darker = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def test_the_accents_tested_are_the_accents_offered():
    assert tuple(theme.ACCENTS) == ACCENT_KEYS


def test_the_accent_preference_offers_every_accent_the_theme_has():
    """The preference table holds no Qt, so it names the accents itself; one
    the theme lacks would be stored and then fail to draw."""
    from am_segmentation_manager.gui import preferences

    offered = tuple(key for key, _caption in preferences.ACCENT_CHOICES)
    assert offered == tuple(theme.ACCENTS)
    assert preferences.DEFAULT_ACCENT == theme.DEFAULT_ACCENT


@pytest.mark.parametrize("text", ["TEXT", "HEADING", "MUTED"])
@pytest.mark.parametrize("surface", ["WINDOW", "CARD", "BASE", "ALTERNATE", "HOVER"])
def test_every_text_colour_is_readable_on_every_surface(text, surface):
    """A palette change is the moment a caption quietly drops below legibility."""
    assert _contrast(getattr(theme, text), getattr(theme, surface)) >= 4.5


@pytest.mark.parametrize("text", ["TEXT", "HEADING", "MUTED"])
@pytest.mark.parametrize("key", ACCENT_KEYS)
def test_every_text_colour_is_readable_on_the_row_being_looked_at(text, key):
    """The selected tint is a surface text sits on, and each accent has its own."""
    assert _contrast(getattr(theme, text), theme.ACCENTS[key].selected) >= 4.5


@pytest.mark.parametrize("key", ACCENT_KEYS)
def test_text_on_the_accent_is_readable(key):
    """A chosen segment, a count badge and a dialog's default button write on it."""
    accent = theme.ACCENTS[key]
    assert _contrast(accent.text, accent.colour) >= 4.5


@pytest.mark.parametrize("key", ACCENT_KEYS)
def test_accent_coloured_text_is_readable(key):
    assert _contrast(theme.ACCENTS[key].strong, theme.BASE) >= 4.5


def test_the_window_opens_in_the_default_accent(qt_app):
    """No accent asked for is the packaged one, not whichever was applied last."""
    style, palette, previous = qt_app.styleSheet(), qt_app.palette(), theme.active_accent()
    theme.apply_theme(qt_app, theme.ACCENTS["blue"])
    try:
        theme.apply_theme(qt_app)
        assert theme.active_accent() == theme.ACCENTS[theme.DEFAULT_ACCENT]
    finally:
        theme.apply_theme(qt_app, previous)
        qt_app.setStyleSheet(style)
        qt_app.setPalette(palette)


@pytest.mark.parametrize("themed", ACCENT_KEYS, indirect=True)
def test_an_accent_applied_leaves_no_colour_of_another_behind(themed):
    """A rule still naming a colour of its own would draw one control orange in
    a blue window, and nothing but looking would find it."""
    applied = theme.active_accent()
    sheet = themed.styleSheet().lower()

    for other in theme.ACCENTS.values():
        if other == applied:
            continue
        for colour in (other.colour, other.strong, other.selected):
            assert colour not in sheet
    assert applied.colour in sheet
    assert themed.palette().color(QPalette.Highlight).name() == applied.selected
    assert themed.palette().color(QPalette.Link).name() == applied.strong


@pytest.mark.parametrize("themed", ["blue"], indirect=True)
def test_a_painted_control_takes_the_accent_applied_after_it_was_imported(themed):
    """Widget modules are imported before the theme is applied, so a colour
    read at import would stay the default whatever accent was chosen."""
    from am_segmentation_manager.gui.widgets.controls import ToggleSwitch

    switch = ToggleSwitch("")
    switch.setChecked(True)
    switch.resize(switch.sizeHint())
    track = switch._track()
    image = switch.grab().toImage()

    # The knob sits at the right of a switch that is on, so the left of the
    # track is bare.
    left = image.pixelColor(round(track.left() + 4), round(track.center().y()))
    assert left == QColor(theme.ACCENTS["blue"].colour)


@pytest.mark.parametrize("name", ["ARROW_DOWN", "ARROW_UP"])
def test_the_arrow_images_are_drawn_in_the_muted_text_colour(name):
    """The colour is written into the file, where a palette change cannot reach it."""
    image = Path(getattr(theme, name))

    assert image.exists()
    assert f'fill="{theme.MUTED}"' in image.read_text(encoding="utf-8")


class _Unstyled:
    """An application that takes a palette and stylesheet and draws nothing, so
    a test can change the accent without restyling the session's application."""

    def __getattr__(self, name):
        return lambda *args: None


def test_switching_the_accent_leaves_the_style_alone():
    """Setting Fusion again on a window already open would rebuild its style
    for nothing; only the palette and stylesheet carry the accent."""
    calls: list[str] = []

    class Recorder:
        def __getattr__(self, name):
            return lambda *args: calls.append(name)

    previous = theme.active_accent()
    try:
        theme.apply_accent(Recorder(), theme.ACCENTS["blue"])
    finally:
        theme.apply_accent(_Unstyled(), previous)

    assert "setStyle" not in calls
    assert calls.index("setStyleSheet") < calls.index("setFont")


@pytest.mark.parametrize("key", ACCENT_KEYS)
def test_each_accent_is_shown_by_a_dot_of_its_own_colour(qt_app, key):
    """Offered in Settings, where the chosen fill is the accent already on
    screen and so shows neither answer's colour."""
    dot = theme.accent_swatch(theme.ACCENTS[key]).pixmap(theme.SWATCH_DOT).toImage()
    middle = theme.SWATCH_DOT // 2

    assert dot.pixelColor(middle, middle) == QColor(theme.ACCENTS[key].colour)


def test_a_conflict_mark_takes_an_accent_changed_since_it_was_first_drawn(qt_app):
    """The marks are cached, and a cache keyed on the shape alone kept the
    accent the first one was drawn in."""
    from am_segmentation_manager.gui.conflict_resolution import ALL
    from am_segmentation_manager.gui.widgets.conflict_tree import mark_icon

    def centre(icon) -> QColor:
        image = icon.pixmap(16).toImage()
        return image.pixelColor(image.width() // 2, image.height() // 2)

    previous = theme.active_accent()
    try:
        theme.apply_accent(_Unstyled(), theme.ACCENTS["orange"])
        assert centre(mark_icon(ALL)) == QColor(theme.ACCENTS["orange"].colour)
        theme.apply_accent(_Unstyled(), theme.ACCENTS["blue"])
        assert centre(mark_icon(ALL)) == QColor(theme.ACCENTS["blue"].colour)
    finally:
        theme.apply_accent(_Unstyled(), previous)


@pytest.mark.parametrize("key", ["orange", "green"])
def test_a_mark_still_to_do_is_not_the_accent(key):
    """An amber ring beside the orange accent read as the same colour. Not blue:
    under blue the ring shares the accent's hue and is told apart by shape."""
    from am_segmentation_manager.gui.widgets.frame_strip import MISMATCH

    accent = QColor(theme.ACCENTS[key].colour)
    assert abs(MISMATCH.hueF() * 360 - accent.hueF() * 360) >= 60


@pytest.mark.parametrize("surface", ["WINDOW", "CARD", "BASE"])
def test_warning_text_is_readable(surface):
    assert _contrast(theme.WARNING, getattr(theme, surface)) >= 4.5


@pytest.mark.parametrize("surface", ["CARD", "BASE"])
def test_a_mark_needing_attention_stands_out_from_its_ground(surface):
    """The ring beside a missing field and the completeness bar's fill."""
    from am_segmentation_manager.gui.widgets.frame_strip import MISMATCH

    assert _contrast(MISMATCH.name(), getattr(theme, surface)) >= 3.0


def test_a_table_heading_sits_on_the_grey_ground(themed):
    """The one widget left on Qt's own look, with a dark grid and a flat heading."""
    from qtpy.QtWidgets import QTableWidget

    table = QTableWidget(1, 2)
    table.setHorizontalHeaderLabels(["modality", "images"])
    table.resize(300, 120)
    table.show()
    themed.processEvents()
    header = table.horizontalHeader()
    image = header.grab().toImage()

    assert image.pixelColor(3, header.height() // 2) == QColor(theme.ALTERNATE)
    table.close()


def test_a_segment_caption_is_bold(themed):
    """Qt ignores a font set for the chosen state alone, so every segment is bold."""
    from am_segmentation_manager.gui.widgets.controls import SegmentedControl

    control = SegmentedControl([("sem", "SEM", ""), ("fib", "FIB", "")])
    control.show()
    themed.processEvents()

    assert all(button.font().bold() for button in control.buttons.values())
    destroy(control)


def test_the_tool_button_font_is_levelled_after_the_stylesheet():
    """macOS gives a tool button a font three points smaller than the rest, and
    applying a stylesheet puts it back, so levelling it first draws every run of
    segments in small print. Only the order shows it: offscreen there is no
    smaller font to put back, so no picture here can fail."""
    calls: list[tuple[str, tuple]] = []

    class Recorder:
        def __getattr__(self, name):
            def record(*args):
                calls.append((name, args))
                return "the application font" if name == "font" else None
            return record

    theme.apply_theme(Recorder())
    names = [name for name, _ in calls]

    levelled = calls.index(("setFont", ("the application font", "QToolButton")))
    assert levelled > names.index("setStyleSheet")


def test_a_card_sits_between_the_window_and_the_fields_in_it(themed):
    """Three grounds: the window, a group of controls on it, and a field inside
    that group, each lighter than the last so each reads as its own layer."""
    from qtpy.QtCore import QPoint
    from qtpy.QtWidgets import QGroupBox, QVBoxLayout

    window = QWidget()
    window.resize(300, 200)
    outer = QVBoxLayout(window)
    outer.setContentsMargins(20, 20, 20, 20)
    card = QGroupBox("Card")
    inner = QVBoxLayout(card)
    edit = QLineEdit()
    inner.addWidget(edit)
    inner.addStretch(1)
    outer.addWidget(card)
    window.show()
    themed.processEvents()
    image = window.grab().toImage()

    field = edit.mapTo(window, QPoint(edit.width() // 2, edit.height() // 2))
    group = card.mapTo(window, QPoint(card.width() // 2, card.height() - 10))
    assert image.pixelColor(5, 5) == QColor(theme.WINDOW)
    assert image.pixelColor(group) == QColor(theme.CARD)
    assert image.pixelColor(field) == QColor(theme.BASE)
    window.close()


def test_a_scroll_area_inside_a_card_shows_the_card_behind_it(themed):
    """A scroll area paints its viewport in the window colour, which drew a
    darker block round the fields of the Add data and Export tabs."""
    from qtpy.QtCore import QPoint
    from qtpy.QtWidgets import QGroupBox, QLabel, QScrollArea, QVBoxLayout

    card = QGroupBox("Fields")
    card.resize(300, 200)
    scroller = QScrollArea()
    scroller.setWidgetResizable(True)
    scroller.setFrameShape(QScrollArea.Shape.NoFrame)
    scroller.setWidget(QLabel("value"))
    QVBoxLayout(card).addWidget(scroller)
    card.show()
    themed.processEvents()
    image = card.grab().toImage()

    inside = scroller.mapTo(card, QPoint(scroller.width() - 20, scroller.height() - 10))
    assert image.pixelColor(inside) == QColor(theme.CARD)
    card.close()


@pytest.mark.parametrize("themed", ACCENT_KEYS, indirect=True)
def test_choosing_a_row_marks_its_edge_without_moving_it(themed):
    """Every row reserves the edge the chosen one marks, so the chosen row's
    text stays in line with the rows around it."""
    widget = QListWidget()
    widget.addItems(["same", "same"])
    widget.resize(200, 80)
    widget.setCurrentRow(0)
    widget.show()
    themed.processEvents()
    image = widget.viewport().grab().toImage()
    chosen = widget.visualItemRect(widget.item(0))
    other = widget.visualItemRect(widget.item(1))

    def leftmost_ink(rect) -> int:
        return min(
            x
            for x in range(rect.left() + 4, rect.right())
            for y in range(rect.top(), rect.bottom())
            if QColor(image.pixel(x, y)).lightness() < 100
        )

    assert image.pixelColor(chosen.left() + 1, chosen.center().y()) == _accent_colour()
    assert image.pixelColor(other.left() + 1, other.center().y()) != _accent_colour()
    assert leftmost_ink(chosen) == leftmost_ink(other)
    widget.close()


@pytest.mark.parametrize("themed", ACCENT_KEYS, indirect=True)
def test_a_chosen_class_is_ringed_in_ink_not_the_accent(themed):
    """Every accent is also a class colour: a ring in it round a swatch of the
    same colour hides which class a stroke will write."""
    from qtpy.QtWidgets import QToolButton

    button = QToolButton()
    button.setObjectName(theme.CLASS_BUTTON)
    button.setText("GIS")
    button.setCheckable(True)
    button.setChecked(True)
    button.resize(120, 28)
    image = button.grab().toImage()

    edge = image.pixelColor(0, image.height() // 2)
    assert edge == QColor(theme.TEXT)
    assert edge != _accent_colour()


@pytest.mark.parametrize("themed", ACCENT_KEYS, indirect=True)
def test_the_chosen_tab_is_marked_along_its_top_edge(themed):
    """The tabs are the one control on screen whatever is being done."""
    from qtpy.QtWidgets import QTabWidget

    tabs = QTabWidget()
    tabs.addTab(QWidget(), "Curate")
    tabs.addTab(QWidget(), "Label")
    tabs.resize(400, 200)
    tabs.show()
    bar = tabs.tabBar()
    image = bar.grab().toImage()

    def has_stripe(index: int) -> bool:
        rect = bar.tabRect(index)
        return any(
            image.pixelColor(rect.center().x(), y) == _accent_colour()
            for y in range(rect.top(), rect.top() + 4)
        )

    assert has_stripe(0)
    assert not has_stripe(1)
    tabs.close()


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
    sheet = theme.stylesheet(theme.active_accent())
    widget.setStyleSheet(sheet)

    assert widget.styleSheet() == sheet
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


def test_a_dropdown_with_one_entry_reads_as_text(themed):
    """Greyed alone, a view dropdown with nothing to choose looked all but
    identical to one with several; asserted on the pixels of its edge."""
    from qtpy.QtWidgets import QComboBox

    from am_segmentation_manager.gui.widgets.common import (
        choice_count_badge,
        fill_choice_combo,
    )

    single, several = QComboBox(), QComboBox()
    for combo, entries in (
        (single, [("base", None)]),
        (several, [("base", None), ("8_fi_1", "8_fi_1"), ("8_fi_16", "8_fi_16")]),
    ):
        choice_count_badge(combo)
        combo.resize(160, 26)
        fill_choice_combo(combo, entries, None)

    def edge(combo) -> QColor:
        return QColor(combo.grab().toImage().pixel(0, 13))

    assert edge(several) == QColor(theme.BORDER)
    assert edge(single) != QColor(theme.BORDER)
    assert single.count_badge.isHidden()
    assert not several.count_badge.isHidden() and several.count_badge.text() == "3"


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


@pytest.mark.parametrize(
    "orientation",
    [Qt.Horizontal, Qt.Vertical] if importlib.util.find_spec("qtpy") is not None else [],
)
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


@pytest.mark.parametrize("themed", ACCENT_KEYS, indirect=True)
def test_the_grip_reacts_to_the_pointer(themed):
    """The same accent every other control takes when it is pointed at."""
    splitter, handle = _handle(themed, Qt.Horizontal)
    middle = (handle.width() // 2, handle.height() // 2)
    point = QPointF(handle.rect().center())

    QApplication.sendEvent(handle, QEnterEvent(point, point, point))
    assert _colour_at(handle, *middle) == theme.active_accent().colour

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

    from am_segmentation_manager.gui.widgets.branding import ICON_PATH

    assert ICON_PATH.exists()
    assert not QIcon(str(ICON_PATH)).isNull()


def test_a_missing_icon_does_not_stop_the_window_opening(qt_app, monkeypatch):
    """A source checkout without the asset must still run."""
    from am_segmentation_manager.gui import app
    from am_segmentation_manager.gui.widgets import branding

    monkeypatch.setattr(branding, "ICON_PATH", Path("/nowhere/absent.png"))
    app._apply_icon(qt_app)


def test_a_missing_logo_draws_nothing_rather_than_failing(qt_app, monkeypatch):
    """Every caller of the mark has to survive a checkout without the asset."""
    from am_segmentation_manager.gui.widgets import branding

    monkeypatch.setattr(branding, "ICON_PATH", Path("/nowhere/absent.png"))

    assert branding.logo_pixmap(96).isNull()


def test_the_mark_is_square_at_whatever_size_it_is_asked_for(qt_app):
    """Letting a layout stretch it is the one way this mark looks wrong."""
    from am_segmentation_manager.gui.widgets.branding import logo_pixmap

    for size in (18, 96):
        pixmap = logo_pixmap(size)
        assert pixmap.width() == pixmap.height() == size


def test_the_icon_is_set_on_the_application_so_dialogs_inherit_it(qt_app):
    from am_segmentation_manager.gui.app import _apply_icon

    _apply_icon(qt_app)

    assert not qt_app.windowIcon().isNull()


def test_the_startup_work_waits_for_the_window_to_appear(qt_app, tmp_path, monkeypatch):
    """``show()`` only asks for a paint. Nothing is drawn until ``exec()``
    starts delivering events, so a scan started before it ran against a window
    that had never appeared -- seconds of blank grey indistinguishable from a
    hang. Asking for the keyboard is queued for the same reason, and ahead of
    the scan: the scan puts a modal dialog up, and there is no point activating
    a window behind one."""
    from am_segmentation_manager.gui import app

    order: list[str] = []

    class FakeWindow:
        def __init__(self, controller, settings=None):
            self.scan_bar = type("S", (), {"set_path": lambda self, p: None})()

        def show(self):
            order.append("show")

        def bring_to_front(self):
            order.append("front")

        def scan(self, path):
            order.append("scan")

    monkeypatch.setattr(app, "CurationWindow", FakeWindow, raising=False)
    monkeypatch.setattr(
        "am_segmentation_manager.gui.widgets.main_window.user_settings",
        lambda: QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat),
    )
    monkeypatch.setattr(
        "am_segmentation_manager.gui.widgets.main_window.CurationWindow", FakeWindow
    )
    monkeypatch.setattr(
        qt_app, "exec", lambda: order.append("exec") or qt_app.processEvents() or 0
    )
    monkeypatch.setattr(qt_app, "exec_", qt_app.exec, raising=False)

    app.run(start_path=tmp_path)

    assert order == ["show", "exec", "front", "scan"], (
        "both must be queued behind the event loop, not run before it"
    )


# ---------------------------------------------------------------------------
# A group of controls reads as a group
# ---------------------------------------------------------------------------

def _ink(widget, top: int, bottom: int) -> int:
    """Count the dark pixels in a horizontal band of a rendered widget.

    Asserted on the pixels because that is the only thing that reports whether
    a rule applied: a font-weight Qt ignored still reads back off the widget.
    """
    image = widget.grab().toImage()
    return sum(
        1
        for y in range(top, bottom)
        for x in range(widget.width())
        if image.pixelColor(x, y).lightness() < 150
    )


def _titled_box(qt_app):
    """A group box with one plain caption inside it, laid out and shown."""
    from qtpy.QtWidgets import QGroupBox, QLabel, QVBoxLayout

    box = QGroupBox("Filters")
    column = QVBoxLayout(box)
    column.addWidget(QLabel("Modality"))
    box.resize(220, 90)
    box.show()
    qt_app.processEvents()
    return box


def test_a_group_title_is_heavier_than_the_captions_inside_it(qt_app):
    """A heading quieter than its own contents is the fault being fixed.

    **The weight has to go on `QGroupBox`, not on `QGroupBox::title`.** Qt
    ignores font properties set on a subcontrol, so the obvious rule is no rule
    at all -- and nothing says so except the pixels.
    """
    style = qt_app.styleSheet()
    try:
        # Cleared rather than assumed: the QApplication is session-wide, and a
        # baseline measured under a theme another test left on is the themed
        # rendering compared against itself.
        qt_app.setStyleSheet("")
        plain = _titled_box(qt_app)
        bare_title, bare_caption = _ink(plain, 0, 16), _ink(plain, 20, 44)
        plain.close()

        theme.apply_theme(qt_app)
        themed_box = _titled_box(qt_app)
        title, caption = _ink(themed_box, 0, 16), _ink(themed_box, 20, 44)
        themed_box.close()
    finally:
        qt_app.setStyleSheet(style)

    assert title > bare_title, "the group title did not get its weight"
    assert caption == bare_caption, "the weight reached the controls inside"


# ---------------------------------------------------------------------------
# The macOS application icon
# ---------------------------------------------------------------------------

def test_the_mac_icon_ships_inside_the_package(qt_app):
    """The Dock's icon is a second asset, and Qt must be able to decode it.

    Loaded rather than merely located, for the reason the mark's own test
    gives: an ICNS Qt cannot read is a null icon and no error.
    """
    from qtpy.QtGui import QIcon

    from am_segmentation_manager.gui.widgets.branding import MAC_ICON_PATH

    assert MAC_ICON_PATH.exists()
    assert not QIcon(str(MAC_ICON_PATH)).isNull()


def test_the_mac_icon_is_the_shape_macos_draws_its_own_icons_in(qt_app):
    """The container is 824/1024 of the canvas, centred, with clear corners.

    The one property that makes this icon look native rather than foreign, and
    the one a re-export could silently lose: an icon filling its canvas sits
    visibly larger in the Dock than every icon beside it.
    """
    from qtpy.QtGui import QIcon

    from am_segmentation_manager.gui.widgets.branding import MAC_ICON_PATH

    image = QIcon(str(MAC_ICON_PATH)).pixmap(512, 512).toImage()
    covered = [
        [image.pixelColor(x, y).alpha() > 127 for x in range(512)] for y in range(512)
    ]

    assert not covered[0][0], "the corners carry the shadow's margin"
    assert covered[256][256]

    opaque = [x for x, on in enumerate(covered[256]) if on]
    width = opaque[-1] - opaque[0] + 1
    assert abs(width - round(512 * 824 / 1024)) <= 2
    # Centred: the margin either side of the container is the same.
    assert abs(opaque[0] - (512 - width - opaque[0])) <= 2

    # And the corner is a superellipse rather than the circular arc an rx/ry
    # rounded rectangle draws. The two cover 0.950 and 0.957 of the container,
    # so the fraction tells them apart; only the first is the shape macOS uses.
    assert abs(sum(map(sum, covered)) / (width * width) - 0.9502) < 0.002


def test_the_dock_gets_the_container_and_every_other_desktop_the_mark(qt_app, monkeypatch):
    """Only macOS draws no container of its own, so only macOS needs one."""
    from am_segmentation_manager.gui.widgets import branding

    monkeypatch.setattr(branding.sys, "platform", "darwin")
    assert branding.app_icon_path() == branding.MAC_ICON_PATH

    monkeypatch.setattr(branding.sys, "platform", "win32")
    assert branding.app_icon_path() == branding.ICON_PATH


def test_a_checkout_without_the_container_falls_back_to_the_mark(qt_app, monkeypatch):
    """A missing asset is never an error; the mark is readable everywhere."""
    from am_segmentation_manager.gui import app
    from am_segmentation_manager.gui.widgets import branding

    monkeypatch.setattr(branding.sys, "platform", "darwin")
    monkeypatch.setattr(branding, "MAC_ICON_PATH", Path("/nowhere/absent.icns"))

    assert branding.app_icon_path() == branding.ICON_PATH
    app._apply_icon(qt_app)
