"""How the window looks: the palette, the stylesheet, and the chrome colours.

Everything visual that carries no information is set here, so the window looks
the same under either Qt binding; see `docs/decisions.md` § Display for why the
style is set rather than inherited, and why a stylesheet rule beats a palette.
It is kept short because Qt re-evaluates it whenever a widget is polished.

Colours that carry *meaning* are not here: the workflow states and label lines
live in :mod:`frame_strip`, and the class colours come from the segmentation
config. Those are data, this is chrome.
"""
from __future__ import annotations

from qtpy.QtGui import QColor, QPalette

#: The surface a panel sits on, and the surface a value sits in. Inputs and
#: lists are white against an off-white window, which is what separates "you can
#: type here" from "this is background" without drawing a box around everything.
WINDOW = "#f4f5f7"
BASE = "#ffffff"
ALTERNATE = "#eef0f3"

TEXT = "#1d1f23"
#: Secondary text: group box titles, placeholders, provenance notes.
MUTED = "#6b7280"
DISABLED_TEXT = "#a3a8b0"

BORDER = "#d6d9df"
#: Separators that should be felt rather than seen -- splitter handles, the rule
#: under a heading.
BORDER_LIGHT = "#e6e8ec"
HOVER = "#eaeef6"
PRESSED = "#e2e6ee"

#: One accent, used for selection, focus and hover. A second would have to mean
#: something, and everything that means something is already spoken for by the
#: workflow colours.
ACCENT = "#2f6feb"
ACCENT_TEXT = "#ffffff"
#: The same accent under the pointer or the finger. Not a second colour: a
#: darker cut of the one above, so a filled button can react without the window
#: gaining a hue that would have to mean something.
ACCENT_STRONG = "#2258c4"

#: The row being looked at, in a list whose rows also carry a checkbox and a
#: status chip. A tint rather than a solid accent fill: Qt draws the check
#: indicator in its own colours, so a saturated row left the tick at low
#: contrast on exactly the row the user is on -- and the chips are outlined for
#: legibility on any background, not for surviving white-on-blue.
SELECTED = "#dbe6fb"

#: Text colours for a short annotation beside a value.
WARNING = "#b26a00"
ERROR = "#c0392b"

#: How a short annotation beside a value is coloured. Three levels, so a panel
#: says what it means rather than picking a colour.
BADGE_STYLES = {
    "warning": f"color: {WARNING}; font-weight: bold;",
    "error": f"color: {ERROR}; font-weight: bold;",
    "info": f"color: {MUTED};",
}

#: The shared-label note reports a consequence the user did not ask for, so it
#: is set apart rather than reading as body text.
SHARED_NOTE_STYLE = f"color: {WARNING};"

#: Every chip drawn into a list row gets a dark outline. Without one a chip
#: whose fill is near the selection highlight disappears on exactly the row the
#: user is looking at. An outline makes it legible on any background rather than
#: tuning the fill against one theme.
SWATCH_OUTLINE = QColor("#1a1a1a")

#: A status chip is one band per modality, **SEM above FIB** -- the same
#: arrangement as the timeline's two rows, so the layout means the same thing in
#: both places.
#:
#: Stacking them is what makes a part-finished modality readable. Progress runs
#: left to right within a band, so with the modalities side by side a chip was
#: up to four segments in a row and nothing said which boundary separated the
#: modalities and which separated started from finished -- it read as a gradient.
#: On different axes neither can be mistaken for the other.
SWATCH_WIDTH = 20
SWATCH_BAND = 6
SWATCH_GAP = 1
SWATCH_HEIGHT = SWATCH_BAND * 2 + SWATCH_GAP


def build_palette() -> QPalette:
    """Return the window's colour palette.

    The palette covers everything the stylesheet does not name. Where the two
    disagree the stylesheet wins, which is why a locked editor's text colour is
    settled there rather than here.
    """
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(WINDOW))
    palette.setColor(QPalette.WindowText, QColor(TEXT))
    palette.setColor(QPalette.Base, QColor(BASE))
    palette.setColor(QPalette.AlternateBase, QColor(ALTERNATE))
    palette.setColor(QPalette.Text, QColor(TEXT))
    palette.setColor(QPalette.Button, QColor(BASE))
    palette.setColor(QPalette.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.ToolTipBase, QColor(BASE))
    palette.setColor(QPalette.ToolTipText, QColor(TEXT))
    palette.setColor(QPalette.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor(ACCENT_TEXT))
    palette.setColor(QPalette.Link, QColor(ACCENT))
    if hasattr(QPalette, "PlaceholderText"):  # Qt 5.12+
        palette.setColor(QPalette.PlaceholderText, QColor(MUTED))
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        palette.setColor(QPalette.Disabled, role, QColor(DISABLED_TEXT))
    return palette


def stylesheet() -> str:
    """Return the application stylesheet.

    Backgrounds, borders and spacing only. Nothing here sets a text colour on an
    input, for the reason :func:`build_palette` gives.
    """
    return f"""
    QGroupBox {{
        border: 1px solid {BORDER_LIGHT};
        border-radius: 6px;
        margin-top: 9px;
        padding: 8px 6px 6px 6px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 4px;
        color: {MUTED};
    }}

    QLineEdit, QComboBox, QPlainTextEdit, QListWidget, QAbstractSpinBox {{
        background: {BASE};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 3px 5px;
        selection-background-color: {ACCENT};
        selection-color: {ACCENT_TEXT};
    }}
    QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QListWidget:focus {{
        border-color: {ACCENT};
    }}
    /* A locked field is read, not merely unavailable: the Add data tab opens
       locked and is mostly used for reading, and Qt's disabled grey puts every
       value at low contrast on exactly the tab whose job is showing them. The
       background dulls instead, which is what says "locked"; the buttons and
       radio groups beside it still grey out normally. */
    QLineEdit:disabled, QComboBox:disabled, QPlainTextEdit:disabled {{
        color: {TEXT};
        background: {WINDOW};
    }}
    QListWidget::item {{ padding: 3px 2px; border-radius: 3px; }}
    QListWidget::item:hover:!selected {{ background: {HOVER}; }}
    /* Styling ::item at all stops Qt drawing the selection from the palette,
       which leaves highlighted text painted white on the list's white
       background -- the selected row simply goes blank. Both states are set
       here because an inactive window uses its own. */
    QListWidget::item:selected {{ background: {SELECTED}; color: {TEXT}; }}
    QListWidget::item:selected:!active {{
        background: {SELECTED}; color: {TEXT};
    }}

    QPushButton {{
        background: {BASE};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 4px 12px;
    }}
    QPushButton:hover {{ border-color: {ACCENT}; }}
    QPushButton:pressed {{ background: {PRESSED}; }}
    QPushButton:disabled {{ color: {DISABLED_TEXT}; border-color: {BORDER_LIGHT}; }}
    /* The button a dialog expects you to press. Qt tints it itself, but the
       rule above overrides that -- so a dialog offering two ways to save
       showed them as two identical buttons, with nothing saying which one the
       reviewer almost always wants. */
    QPushButton:default {{
        background: {ACCENT};
        color: {ACCENT_TEXT};
        border-color: {ACCENT};
    }}
    QPushButton:default:hover {{ background: {ACCENT_STRONG}; }}
    QPushButton:default:pressed {{ background: {ACCENT_STRONG}; }}
    QPushButton:default:disabled {{
        background: {BASE};
        color: {DISABLED_TEXT};
        border-color: {BORDER_LIGHT};
    }}

    QToolButton {{
        background: {BASE};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 3px 8px;
    }}
    QToolButton:hover {{ border-color: {ACCENT}; }}
    /* Fusion marks a checked button with a barely darker fill. These carry
       which class a stroke will write and which classes are protected from
       one, so being able to see the answer across the room is worth a rule. */
    QToolButton:checked {{ background: {SELECTED}; border-color: {ACCENT}; }}
    QToolButton:disabled {{ color: {DISABLED_TEXT}; border-color: {BORDER_LIGHT}; }}

    QTabWidget::pane {{
        border: 1px solid {BORDER};
        border-radius: 6px;
        top: -1px;
    }}
    QTabBar::tab {{
        background: transparent;
        border: 1px solid transparent;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        padding: 6px 16px;
        margin-right: 2px;
        color: {MUTED};
    }}
    QTabBar::tab:selected {{
        background: {BASE};
        border-color: {BORDER};
        border-bottom-color: {BASE};
        color: {TEXT};
    }}
    QTabBar::tab:hover:!selected {{ background: {HOVER}; color: {TEXT}; }}

    QSplitter::handle {{ background: {BORDER_LIGHT}; }}
    QSplitter::handle:hover {{ background: {ACCENT}; }}

    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
    QScrollBar::handle {{ background: #c3c8d0; border-radius: 5px; }}
    QScrollBar::handle:vertical {{ min-height: 24px; }}
    QScrollBar::handle:horizontal {{ min-width: 24px; }}
    QScrollBar::handle:hover {{ background: {MUTED}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    QToolTip {{
        background: {BASE};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 4px 6px;
    }}
    """


def apply_theme(application) -> None:
    """Give *application* the window's style, palette and stylesheet.

    Fusion rather than the platform style, so the window looks the same wherever
    it runs -- the development machine and the user's differ in Qt binding, and
    four bugs have already reached the user through that gap.
    """
    application.setStyle("Fusion")
    application.setPalette(build_palette())
    application.setStyleSheet(stylesheet())
