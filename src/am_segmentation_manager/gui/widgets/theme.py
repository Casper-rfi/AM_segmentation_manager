"""How the window looks: the palette, the stylesheet, and the chrome colours.

Everything visual that carries no information is set here, so the window looks
the same wherever it runs; see `docs/display.md` for why the style is set rather
than inherited. It is kept short because Qt re-evaluates it whenever a widget is
polished. The accent is one of :data:`ACCENTS`; every other colour is shared by
all of them. Colours that carry *meaning* -- workflow states, label lines, class
colours -- live in :mod:`frame_strip` and the segmentation config: those are
data, this is chrome.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from qtpy.QtGui import QColor, QIcon, QPainter, QPalette, QPen, QPixmap

from ...layout import MODALITIES, SERIES

#: The arrows in a dropdown and a spin box, drawn in :data:`MUTED`
#: (`test_the_arrow_images_are_drawn_in_the_muted_text_colour`). Files, because
#: a stylesheet can put an image on an arrow but draw no shape of its own.
_DATA = Path(__file__).parent.parent.parent / "data"
ARROW_DOWN = (_DATA / "arrow_down.svg").as_posix()
ARROW_UP = (_DATA / "arrow_up.svg").as_posix()

#: The surface a panel sits on, and the surface a value sits in. Inputs and
#: lists are near-white against a light grey window, which is what separates
#: "you can type here" from "this is background" without drawing a box around
#: everything.
WINDOW = "#e8e7e4"
BASE = "#fdfdfc"
#: The fill of a group of controls: lighter than :data:`WINDOW`, so a group reads
#: as one thing, and darker than :data:`BASE`, so a field inside it still does.
CARD = "#f4f3f1"
ALTERNATE = "#eeedea"

TEXT = "#1f1e1c"
#: Secondary text: placeholders, provenance notes, a caption beside a control.
#: The lightest text that must still be read, so it holds 4.5:1 against every
#: surface text sits on (`test_every_text_colour_is_readable_on_every_surface`).
MUTED = "#66635e"
#: The title of a group of controls. Between :data:`MUTED` and :data:`TEXT`, and
#: set bold below: a heading quieter than the captions under it reads as the
#: least important thing in its own box, which is what a group title must not be.
HEADING = "#4a4844"
DISABLED_TEXT = "#a9a7a2"

BORDER = "#d3d1cc"
#: Separators that should be felt rather than seen -- the rule under a heading.
BORDER_LIGHT = "#e2e0dc"
HOVER = "#ecebe7"
PRESSED = "#e2e0db"
#: The fill of a button: a shade off :data:`BASE`, so a button reads as something
#: to press rather than as a field to type in.
BUTTON = "#f9f8f6"


@dataclass(frozen=True)
class Accent:
    """The colours one accent gives the window, used for selection, focus and hover.

    One accent at a time: a second would have to mean something, and everything
    that means something is spoken for by the workflow colours. Every accent is
    also a class colour, so it stays on controls and off anything drawn beside a
    label or the legend.

    Attributes:
        colour: Filled or drawn on a control: a chosen segment, a tab's edge,
            a switch that is on.
        text: What is written on :attr:`colour`
            (`test_text_on_the_accent_is_readable`).
        strong: A deeper cut of the same accent, not a second colour, for
            accent-coloured text on a light ground: a link
            (`test_accent_coloured_text_is_readable`).
        selected: The tint behind the row being looked at. A tint rather than
            :attr:`colour`: Qt draws a row's check indicator in its own
            colours, so a saturated row left the tick at low contrast on exactly
            the row the user is on. A chip is not outlined, so every fill it
            uses has to hold against this tint as well as against white.
    """

    colour: str
    text: str
    strong: str
    selected: str


#: The accents the window can be drawn in, by the key a preference stores. Orange
#: and green are the logo's, softened light enough that what is written on them
#: is dark, and green greyed to a sage.
ACCENTS: Mapping[str, Accent] = {
    "orange": Accent(colour="#f5a04a", text=TEXT, strong="#b35c12", selected="#fdeedf"),
    "blue": Accent(colour="#2f6feb", text="#ffffff", strong="#2258c4", selected="#dbe6fb"),
    "green": Accent(colour="#83b483", text=TEXT, strong="#3f7a45", selected="#e9f1e9"),
}
DEFAULT_ACCENT = "green"

#: How wide the dot showing an accent beside its name is, in pixels.
SWATCH_DOT = 10

_active: Accent = ACCENTS[DEFAULT_ACCENT]


def active_accent() -> Accent:
    """Return the accent :func:`apply_theme` last applied.

    Read when painting, never at import: widget modules are imported before the
    theme is applied, so a colour bound then would be the default whatever was
    chosen.
    """
    return _active


#: The grip on a splitter handle. Darker than :data:`BORDER`, because it is the
#: one separator that is meant to be *seen*: a division nobody can tell is
#: draggable is a division nobody drags. Under the pointer it takes the accent,
#: like every other control that reacts to being pointed at.
GRIP = "#b1aea8"

#: A scroll bar's handle at rest. Under the pointer it takes :data:`MUTED`.
SCROLL_HANDLE = "#c5c2bc"

#: Text colours for a short annotation beside a value. The warning is a dark
#: amber: text, where the accent never is, so the two are not read as one
#: colour (`test_warning_text_is_readable`).
WARNING = "#955500"
ERROR = "#c0392b"

#: How a short annotation beside a value is coloured. Three levels, so a panel
#: says what it means rather than picking a colour.
BADGE_STYLES = {
    "warning": f"color: {WARNING}; font-weight: bold;",
    "error": f"color: {ERROR}; font-weight: bold;",
    "info": f"color: {MUTED};",
}

#: A chip standing for one value of a field that holds several. Filled rather
#: than outlined, so a row of them reads as several things where a plain
#: comma-separated line reads as one, and filled in grey: the selected tint
#: would read as a chosen row. The cross is the muted text colour, being a
#: control on the chip rather than part of the value.
VALUE_CHIP_STYLE = (
    f"QPushButton {{ background: {ALTERNATE}; border: 1px solid {BORDER}; "
    f"border-radius: 9px; padding: 1px 7px; text-align: left; }}"
    f"QPushButton:hover {{ background: {BORDER_LIGHT}; }}"
    f"QPushButton:disabled {{ background: {WINDOW}; color: {DISABLED_TEXT}; }}"
)

#: Object name marking a button whose label is a single character, so the
#: stylesheet can stop the default padding swallowing it. Set with
#: ``setObjectName``; see the rule of the same name below.
COMPACT_BUTTON = "compactButton"

#: Object name of the mark in the tab row's corner: a button drawn as the mark
#: alone, a frame around it reading as one more control beside the gear.
MARK_BUTTON = "markButton"

#: Object name of the count beside a view dropdown holding several entries.
CHOICE_COUNT = "choiceCount"

#: Object name of a heading over a group of fields inside one card.
SECTION_HEADING = "sectionHeading"

#: Object name of a class's padlock button. The tool-button padding leaves ten
#: pixels of a 28-pixel button for its mark, and the padlock is drawn larger
#: than that on purpose; see `label_panel.MARK`.
PADLOCK_BUTTON = "padlockButton"

#: Object name of a class button on the Label tab. Chosen, it is ringed in
#: :data:`TEXT` rather than the accent, which would merge into the swatch of a
#: class of the same colour and hide which class a stroke will write.
CLASS_BUTTON = "classButton"

#: Object name of the timeline's hover card, so the stylesheet can draw it as it
#: draws a tooltip.
TIMELINE_CARD = "timelineCard"

#: Object name of a legend caption on that card, muted so the frame's own lines
#: above it are what the eye lands on.
TIMELINE_LEGEND = "timelineLegend"

#: The outline round the completeness bar, which is partly filled and so needs
#: its unfilled end drawn to be a bar at all; mid grey, the darkest line in the
#: window otherwise being text. The status chip has none: its cells are filled
#: edge to edge.
SWATCH_OUTLINE = QColor("#9f9c96")

#: How a standalone on/off control is drawn -- the track, the knob on it, and
#: the gap before its caption. Painted by :class:`.common.ToggleSwitch`, so no
#: stylesheet rule below may name ``QCheckBox``: a rule would send the control
#: back through ``QStyle`` and the switch would be drawn twice.
#:
#: The track is *filled* when off rather than outlined. A switch that is off is
#: still a control, and an outline at this size reads as an empty field.
SWITCH_OFF = "#c7c4be"
SWITCH_KNOB = "#ffffff"
SWITCH_WIDTH = 34
SWITCH_HEIGHT = 18
SWITCH_INSET = 2.0
SWITCH_GAP = 8

#: Room kept clear around the track for the focus ring. Reserved rather than
#: drawn over the edge: a ring painted outside the widget is clipped flat on the
#: side it runs off, which reads as a drawing fault rather than as focus.
SWITCH_FOCUS = 3.0

#: Room between one group of view controls and the next, where the groups are
#: `controls.keep_paired` blocks rather than loose widgets. A switch ends at
#: its caption, where a tick box carried padding past one, so a run of them
#: needs a gap to read as separate things -- but the pairing is what separates
#: the groups now, and the width a narrower gap saves is what keeps the whole
#: run inside the pane a laptop screen can offer; `docs/display.md` gives the
#: measurements.
GROUP_GAP = 10

#: How a splitter handle is drawn: how thick the handle is, and the rounded bar
#: down the middle of it. A bar rather than a row of dots, which at this size is
#: three specks that read as dirt on the screen; short enough against the length
#: of the division that it is plainly a grip on it rather than the edge itself.
#: Painted by :class:`.common.GripSplitter` rather than set as a rule below, a
#: stylesheet being able to put only a flat colour or an image file on a handle.
HANDLE_WIDTH = 6
GRIP_THICKNESS = 3.0
GRIP_LENGTH = 28.0

#: A status chip is a grid: one column per series in ``layout.SERIES`` order,
#: **SEM above FIB**. The gap shows the row behind it rather than a border,
#: which is what separates one cell from the next.
#:
#: The width is counted from the series drawn -- see :func:`swatch_width` --
#: rather than written down, so drawing another widens the chip instead of
#: painting a column outside the pixmap, which clips without reporting anything.
#:
#: **The height must stay even**, or the chip lands half a pixel below the tick
#: beside it -- `docs/display.md` § Controls says why.
SWATCH_CELL = 7
SWATCH_GAP = 2


def swatch_width(columns: int) -> int:
    """Return how wide a chip of *columns* series is, in pixels."""
    return columns * SWATCH_CELL + max(columns - 1, 0) * SWATCH_GAP


#: A chip of every series. The lists draw one of the series switched on.
SWATCH_WIDTH = swatch_width(len(SERIES))
SWATCH_HEIGHT = len(MODALITIES) * SWATCH_CELL + (len(MODALITIES) - 1) * SWATCH_GAP


def build_palette(accent: Accent) -> QPalette:
    """Return the window's colour palette, in *accent*.

    The palette covers everything the stylesheet does not name. Where the two
    disagree the stylesheet wins, which is why a locked editor's text colour is
    settled there rather than here.
    """
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(WINDOW))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(BASE))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(ALTERNATE))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(BUTTON))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(BASE))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(accent.selected))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Link, QColor(accent.strong))
    if hasattr(QPalette, "PlaceholderText"):  # Qt 5.12+
        palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(MUTED))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(DISABLED_TEXT))
    return palette


def stylesheet(accent: Accent) -> str:
    """Return the application stylesheet, in *accent*.

    Backgrounds, borders and spacing only. Nothing here sets a text colour on an
    input, for the reason :func:`build_palette` gives.
    """
    return f"""
    /* A group of controls is a card: a title that reads as a heading, and a
       boundary you can see. Both are what say where one cluster of controls
       ends and the next begins, on every tab at once. Its fill sits between the
       window and a field: as white as the fields inside it, it would take away
       the one thing separating "you can type here" from the panel. */
    /* The weight goes on the box, not on ``::title``: Qt ignores font
       properties set on a subcontrol, so the obvious rule is silently no rule
       at all. It reaches the title without reaching the controls inside, the
       stylesheet font not descending the way a CSS one would. */
    QGroupBox {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 6px;
        margin-top: 10px;
        padding: 10px 6px 6px 6px;
        font-weight: bold;
    }}
    /* A scroll area paints its viewport in the window colour, which inside a
       card is a darker block round the fields. The viewport and the widget it
       holds are let through to the card instead. */
    QGroupBox QScrollArea, QGroupBox QScrollArea > QWidget#qt_scrollarea_viewport,
    QGroupBox QScrollArea > QWidget > QWidget {{
        background: transparent;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 4px;
        color: {HEADING};
    }}

    QLineEdit, QComboBox, QPlainTextEdit, QListWidget, QTreeWidget,
    QAbstractSpinBox {{
        background: {BASE};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 3px 5px;
        selection-background-color: {accent.selected};
        selection-color: {TEXT};
    }}
    /* The arrows keep Qt's own drawing, but not its square boxes, which stood
       out as hard corners inside a rounded field. */
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border: none;
    }}
    /* Styling the boxes stops Qt drawing the arrows in them, so each is drawn
       from a small image instead. */
    QComboBox::down-arrow {{ image: url("{ARROW_DOWN}"); width: 8px; height: 5px; }}
    /* The selection above is also the highlighted row of an open dropdown list,
       which takes it from the dropdown rather than from the palette. */
    QAbstractSpinBox {{ padding-right: 18px; }}
    QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
        subcontrol-origin: border;
        width: 16px;
        border: none;
        background: transparent;
    }}
    QAbstractSpinBox::up-button {{ subcontrol-position: top right; margin-top: 2px; }}
    QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; margin-bottom: 2px; }}
    QAbstractSpinBox::up-arrow {{ image: url("{ARROW_UP}"); width: 7px; height: 4px; }}
    QAbstractSpinBox::down-arrow {{ image: url("{ARROW_DOWN}"); width: 7px; height: 4px; }}
    /* An open dropdown and a context menu mark their row as a list does: the
       palette's highlight would fill the whole row in a solid colour. */
    QComboBox QAbstractItemView {{
        background: {BASE};
        border: 1px solid {BORDER};
        selection-background-color: {accent.selected};
        selection-color: {TEXT};
        outline: 0;
    }}
    QMenu {{ background: {BASE}; border: 1px solid {BORDER}; padding: 4px 0; }}
    QMenu::item {{ padding: 4px 20px; }}
    QMenu::item:selected {{ background: {accent.selected}; color: {TEXT}; }}
    QMenu::item:disabled {{ color: {DISABLED_TEXT}; }}
    QMenu::separator {{ height: 1px; background: {BORDER_LIGHT}; margin: 4px 8px; }}
    QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QListWidget:focus,
    QTreeWidget:focus {{
        border-color: {accent.colour};
    }}
    /* A locked field is read, not merely unavailable: the Add data tab opens
       locked and is mostly used for reading, and Qt's disabled grey puts every
       value at low contrast on exactly the tab whose job is showing them. The
       background dulls instead, which is what says "locked"; the buttons and
       radio groups beside it still grey out normally, being controls you
       cannot use rather than values you are reading. */
    QLineEdit:disabled, QComboBox:disabled, QPlainTextEdit:disabled {{
        color: {TEXT};
        background: {WINDOW};
    }}
    /* A view dropdown with one entry has nothing to choose, so it is drawn as
       the text of what is on screen, without a box or an arrow; one with
       several carries its count beside it. See `common.fill_choice_combo`. */
    QComboBox[single="true"]:disabled {{
        border-color: transparent;
        background: transparent;
    }}
    QComboBox[single="true"]::drop-down {{ width: 0px; border: none; }}
    QComboBox[single="true"]::down-arrow {{ image: none; border: none; }}
    QLabel#{CHOICE_COUNT} {{
        background: {accent.colour};
        color: {accent.text};
        border-radius: 8px;
        padding: 0px 5px;
        font-weight: 600;
    }}
    /* A heading inside a card: the card title's colour and weight, one step
       down by standing inside it rather than on its border. */
    QLabel#{SECTION_HEADING} {{ color: {HEADING}; font-weight: bold; padding-top: 6px; }}
    /* Every row reserves the edge the chosen row marks in the accent, so
       choosing a row does not move its tick and text out of line. */
    QListWidget::item, QTreeWidget::item {{
        padding: 3px 2px;
        border-left: 3px solid transparent;
    }}
    QListWidget::item:hover:!selected, QTreeWidget::item:hover:!selected {{
        background: {HOVER};
    }}
    /* Styling ::item at all stops Qt drawing the selection from the palette,
       which leaves highlighted text painted white on the list's white
       background -- the selected row simply goes blank. Both states are set
       here because an inactive window uses its own. */
    QListWidget::item:selected, QTreeWidget::item:selected {{
        background: {accent.selected}; color: {TEXT}; border-left-color: {accent.colour};
    }}
    QListWidget::item:selected:!active, QTreeWidget::item:selected:!active {{
        background: {accent.selected}; color: {TEXT}; border-left-color: {accent.colour};
    }}
    /* A row of several columns is one selection: the bar marks where it
       begins, not every cell of it. */
    QTreeWidget::item:selected:middle, QTreeWidget::item:selected:last,
    QTreeWidget::item:selected:middle:!active, QTreeWidget::item:selected:last:!active {{
        border-left-color: transparent;
    }}

    /* Buttons, fields and segments share one corner radius, nearer the round
       ends of the switches beside them than a square box is. */
    QPushButton {{
        background: {BUTTON};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 5px 12px;
    }}
    /* A button whose whole label is one character. The rule above spends 12px
       a side on padding, which is more than such a button is wide -- so the
       character is clipped away and the button draws blank. Named rather than
       styled where it is built, because chrome is set here and nowhere else. */
    QPushButton#{COMPACT_BUTTON} {{ padding: 5px 2px; }}
    QPushButton:hover {{ background: {HOVER}; border-color: {accent.colour}; }}
    QPushButton:pressed {{ background: {PRESSED}; }}
    QPushButton:disabled {{ color: {DISABLED_TEXT}; border-color: {BORDER_LIGHT}; }}
    /* The button a dialog expects you to press. Qt tints it itself, but the
       rule above overrides that -- so a dialog offering two ways to save
       showed them as two identical buttons, with nothing saying which one the
       reviewer almost always wants. */
    QPushButton:default {{
        background: {accent.colour};
        color: {accent.text};
        border-color: {accent.colour};
    }}
    QPushButton:default:hover {{ background: {accent.colour}; border-color: {TEXT}; }}
    QPushButton:default:pressed {{ background: {accent.colour}; border-color: {TEXT}; }}
    QPushButton:default:disabled {{
        background: {BUTTON};
        color: {DISABLED_TEXT};
        border-color: {BORDER_LIGHT};
    }}

    QToolButton {{
        background: {BUTTON};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 3px 8px;
    }}
    QToolButton:hover {{ border-color: {accent.colour}; }}
    QToolButton#{MARK_BUTTON}, QToolButton#{MARK_BUTTON}:hover {{
        background: transparent; border: none; padding: 0;
    }}
    /* Fusion marks a checked button with a barely darker fill. These carry
       which class a stroke will write and which classes are protected from
       one, so being able to see the answer across the room is worth a rule. */
    QToolButton:checked {{ background: {accent.selected}; border-color: {accent.colour}; }}
    QToolButton:disabled {{ color: {DISABLED_TEXT}; border-color: {BORDER_LIGHT}; }}
    /* Save while edits are held only in this window. Its caption is gone, so
       the button itself says there is something to write. */
    QToolButton[pending="true"] {{ background: {accent.selected}; border-color: {accent.colour}; }}
    QToolButton#{PADLOCK_BUTTON} {{ padding: 1px; }}
    QToolButton#{CLASS_BUTTON}:checked {{ border-color: {TEXT}; }}

    /* A run of segments is one control: they share their borders, the inner
       corners are square, and the chosen one is filled rather than tinted --
       this answers "which of these" where a checked tool button elsewhere
       answers "is this on", and the two must not look alike. Named rather than
       styled where they are built, because chrome is set here and nowhere
       else; `common.SegmentedControl` puts the names on. Every name has to be
       spelled out in every rule, Qt's stylesheets having no wildcard. Every
       caption is bold, Qt ignoring a font set for the chosen state alone. */
    QToolButton#segmentOnly, QToolButton#segmentFirst,
    QToolButton#segmentMiddle, QToolButton#segmentLast {{
        border-radius: 0;
        padding: 4px 12px;
        font-weight: bold;
    }}
    QToolButton#segmentOnly {{ border-radius: 6px; }}
    QToolButton#segmentFirst {{
        border-top-left-radius: 6px;
        border-bottom-left-radius: 6px;
    }}
    QToolButton#segmentLast {{
        border-top-right-radius: 6px;
        border-bottom-right-radius: 6px;
    }}
    /* One border between neighbours, not two abutting ones. */
    QToolButton#segmentMiddle, QToolButton#segmentLast {{ border-left: none; }}
    QToolButton#segmentOnly:checked, QToolButton#segmentFirst:checked,
    QToolButton#segmentMiddle:checked, QToolButton#segmentLast:checked {{
        background: {accent.colour};
        color: {accent.text};
        border-color: {accent.colour};
    }}
    QToolButton#segmentOnly:hover:!checked, QToolButton#segmentFirst:hover:!checked,
    QToolButton#segmentMiddle:hover:!checked, QToolButton#segmentLast:hover:!checked {{
        background: {HOVER};
        border-color: {BORDER};
    }}
    QToolButton#segmentOnly:disabled, QToolButton#segmentFirst:disabled,
    QToolButton#segmentMiddle:disabled, QToolButton#segmentLast:disabled {{
        color: {DISABLED_TEXT};
        background: {BUTTON};
        border-color: {BORDER};
    }}

    /* A table is a list with columns, and is drawn as one: a near-white ground,
       faint rules, a grey heading, and the chosen row tinted rather than filled.
       Styling ::item stops the palette drawing the selection, so both are set. */
    QTableView {{
        background: {BASE};
        border: 1px solid {BORDER};
        border-radius: 6px;
        gridline-color: {BORDER_LIGHT};
        selection-background-color: {accent.selected};
        selection-color: {TEXT};
    }}
    QTableView::item {{ padding: 2px 6px; }}
    QTableView::item:selected {{ background: {accent.selected}; color: {TEXT}; }}
    QHeaderView::section {{
        background: {ALTERNATE};
        color: {HEADING};
        border: none;
        border-right: 1px solid {BORDER_LIGHT};
        border-bottom: 1px solid {BORDER};
        padding: 4px 8px;
    }}
    QTableCornerButton::section {{ background: {ALTERNATE}; border: none; }}

    QTabWidget::pane {{
        border: 1px solid {BORDER};
        border-radius: 6px;
        top: -1px;
    }}
    /* The tabs are the window's top-level navigation, and the only control
       that is on screen whatever is being done. They are sized and weighted to
       say so: a tab the eye passes over is a stage of the workflow nobody
       finds. The folder shape is kept -- it is what joins the tab to the pane
       below it, and an underline would leave the pane belonging to nothing. The
       chosen one carries the accent along its top edge; every tab reserves that
       edge, so choosing one does not move its caption. */
    QTabBar::tab {{
        background: transparent;
        border: 1px solid transparent;
        border-top: 3px solid transparent;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        padding: 8px 22px;
        margin-right: 3px;
        color: {HEADING};
    }}
    QTabBar::tab:selected {{
        background: {BASE};
        border-color: {BORDER};
        border-top-color: {accent.colour};
        border-bottom-color: {BASE};
        color: {TEXT};
        font-weight: bold;
    }}
    QTabBar::tab:hover:!selected {{ background: {HOVER}; color: {TEXT}; }}

    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
    QScrollBar::handle {{ background: {SCROLL_HANDLE}; border-radius: 5px; }}
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
    /* The timeline's hover card is a tooltip holding pictures, and looks like one. */
    QFrame#{TIMELINE_CARD} {{
        background: {BASE};
        border: 1px solid {BORDER};
        border-radius: 4px;
    }}
    QLabel#{TIMELINE_LEGEND} {{ color: {MUTED}; }}
    """


def apply_theme(application, accent: Accent | None = None) -> None:
    """Give *application* the window's style, then its accent.

    Args:
        accent: As :func:`apply_accent`. None is :data:`DEFAULT_ACCENT`.

    Fusion rather than the platform style, so the window looks the same wherever
    it runs -- the development machine and the microscope's differ in platform
    and in Qt, and four bugs have already reached the user through that gap.
    """
    application.setStyle("Fusion")
    apply_accent(application, accent if accent is not None else ACCENTS[DEFAULT_ACCENT])


def apply_accent(application, accent: Accent) -> None:
    """Draw *application* in *accent*, and have :func:`active_accent` answer it.

    Safe on a window already on screen: the palette and stylesheet repolish
    every widget, and the painted ones read the accent when they repaint. What
    stores a colour instead -- a scene item, a cached icon -- has to be redrawn
    by whoever holds it (`test_switching_the_accent_leaves_none_of_the_old_one_on_screen`).

    macOS gives a tool button a font three points smaller than the rest of the
    window, which drew every run of segments in small print, so that class is
    set level with it -- **after** the stylesheet, which puts the platform's
    class fonts back (`test_the_tool_button_font_is_levelled_after_the_stylesheet`).
    """
    global _active
    _active = accent
    application.setPalette(build_palette(accent))
    application.setStyleSheet(stylesheet(accent))
    application.setFont(application.font(), "QToolButton")


def accent_swatch(accent: Accent, size: int = SWATCH_DOT) -> QIcon:
    """Return a dot filled with *accent*, to show an accent where it is offered."""
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor(0, 0, 0, 0)))
    painter.setBrush(QColor(accent.colour))
    painter.drawEllipse(0, 0, size, size)
    painter.end()
    return QIcon(pixmap)
