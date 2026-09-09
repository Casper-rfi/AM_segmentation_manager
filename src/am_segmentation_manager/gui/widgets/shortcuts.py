"""The keys, the list that explains them, and which tab each belongs to.

A key is live only while its tab is showing, so this switches them on and off
as tabs change. The bindings are a table read top to bottom: **the order below
is the order F1 reads in**, headings appearing in the order the bindings first
name them, so grouping the list is a matter of moving a line rather than of
keeping a second table of headings in step with it.

The table names the window's own methods, so this knows the window while the
window knows only that it has keys. Why the list looks and behaves as it does
is `docs/display.md` under Controls.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .common import mark_image
from .frame_strip import timeline_legend
from .keyboard import keyboard_diagram
from .label_panel import CLASS_KEYS
from .metadata_panel import completeness_legend
from .project_panel import status_legend
from .theme import BORDER_LIGHT, HEADING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from qtpy.QtGui import QPixmap

try:  # Qt 6 moved QShortcut out of QtWidgets.
    from qtpy.QtGui import QKeySequence, QShortcut
except ImportError:  # pragma: no cover - depends on the installed binding
    from qtpy.QtGui import QKeySequence
    from qtpy.QtWidgets import QShortcut


#: Which tab a key belongs to. A key is live only while its tab is showing --
#: without that, Up/Down step the lamella behind whichever tab you are actually
#: looking at, and the lists on the other tabs never see an arrow key at all.
#:
#: Each is named after its tab, and the tabs after the step of the workflow they
#: cover. The widget behind "Add data" is the metadata panel, because describing
#: a project is what adding one to the collection currently consists of.
TAB_CURATE = "curate"
TAB_ADD_DATA = "add data"
TAB_LABEL = "label"
TAB_EXPORT = "export"

#: A key that works everywhere, whatever is showing.
ANY_TAB = "any"


#: The headings the F1 list groups its rows under.
#:
#: A row belongs to exactly one, and the *order* the headings appear in is the
#: order the rows below declare them -- so the list is arranged by moving a
#: binding, not by keeping a second table of headings in step with it.
#: :data:`GROUP_HELP` is the exception, and is always last.
GROUP_VIEW = "Change the view"
GROUP_CHOOSE = "Choose what changes"
GROUP_STATE = "Set the segmentation state"
GROUP_TOOLS = "Choose a tool"
GROUP_CLASSES = "Choose a class"
GROUP_SPECKS = "Find specks"
GROUP_EDITS = "Save and undo"
GROUP_ELSEWHERE = "Elsewhere in the window"
#: The two legend headings. Separate rather than one "what the marks mean",
#: because a reader arrives asking about a particular mark and the heading is
#: what tells them which half to read.
GROUP_ROW_MARK = "The mark beside each row"
GROUP_STRIP_MARK = "The bar under the images"
GROUP_HELP = "This list"

#: Mouse gestures worth listing beside the keys, per tab, as
#: ``(heading, what to do, what it does)``.
#:
#: Not bindings: nothing registers these, the widgets read the modifiers
#: themselves. They are here because F1 is the one place that answers "what can
#: I do here?", and a gesture nobody can discover is a gesture nobody uses.
#: They carry a heading for the same reason the keys do -- picking frames out
#: is one idea whether it is done with Esc or with Ctrl-click, and splitting it
#: into a keys half and a mouse half hid that.
TAB_GESTURES: dict[str, tuple[tuple[str, str, str], ...]] = {
    TAB_CURATE: (
        (GROUP_VIEW, "Click or drag the timeline", "Move to a frame"),
        (
            GROUP_CHOOSE, "Ctrl-click a frame",
            "Pick it out for a state change, or drop it again",
        ),
        (
            GROUP_CHOOSE, "Shift-click a frame",
            "Extend the picked run from the last frame you pointed at",
        ),
        (GROUP_CHOOSE, "Ctrl or Shift + drag", "Sweep a run of frames"),
        (
            GROUP_CHOOSE, "Tick a project or lamella",
            "Apply a state change to the whole of it",
        ),
        (
            GROUP_ELSEWHERE, "Right-click a row or an image",
            "Copy its path, or open its folder",
        ),
        (
            GROUP_ELSEWHERE, "Operations",
            "Act on the same selection; each previews before it applies",
        ),
    ),
    TAB_ADD_DATA: (
        (
            GROUP_ELSEWHERE, "Right-click a project",
            "Copy its path, or open its folder",
        ),
    ),
    TAB_LABEL: (
        # First of the tool gestures, so it lands directly under [ and ]: all
        # three resize the brush, and a reader looking for how to do that
        # should not have to step over two other rows to find the third way.
        # The mouse rows stay together either way, this one leading them.
        (GROUP_TOOLS, "Shift + scroll wheel", "Make the brush bigger or smaller"),
        (GROUP_TOOLS, "Drag with the left button", "Paint with the brush"),
        (
            GROUP_TOOLS, "Click with the fill tool",
            "Flood the area of one class you clicked in",
        ),
        (
            GROUP_VIEW, "Hold Space, or drag with the middle button",
            "Pan instead of painting",
        ),
        (GROUP_VIEW, "Scroll wheel", "Zoom about the pointer"),
        (GROUP_VIEW, "Click or drag the timeline", "Move to a frame"),
    ),
}



#: Every left-hand cell of the tables is bold, being the column a reader scans,
#: so bold alone left the headings looking like one more key. They take the
#: colour and the weight a `QGroupBox` title takes over the captions inside its
#: card, plus a size and a rule -- the same idea as the cards, so the list
#: agrees with the window rather than inventing a second hierarchy.
HEADING_STYLE = (
    f"margin-top: 16px; margin-bottom: 2px; color: {HEADING}; "
    f"font-size: 12pt; font-weight: bold; "
    f"border-bottom: 1px solid {BORDER_LIGHT};"
)



def _table(rows) -> str:
    """Return an HTML table of ``(what to do, what it does)`` pairs.

    The left cell also carries the legend's pictures, which is why it is
    middle-aligned: a sample taller than its line otherwise sits on the text
    baseline and drags the row apart.
    """
    body = "".join(
        f"<tr><td style='padding-right: 14px; vertical-align: middle;'>"
        f"<b>{action}</b></td>"
        f"<td style='vertical-align: middle;'>{description}</td></tr>"
        for action, description in rows
    )
    return f"<table>{body}</table>"


def _keys_of(bindings: list) -> set[str]:
    """Return every key the diagram should pick out, chords broken apart.

    ``Ctrl+Shift+Z`` lights Ctrl, Shift and Z: the modifiers are keys somebody
    has to find too, and a diagram that showed only the letter would be saying
    the shortcut is one key.
    """
    return {
        part.strip()
        for binding in bindings
        for part in binding.key.split("+")
        if part.strip()
    }


@dataclass(frozen=True)
class Binding:
    """One keyboard shortcut: what it does, where it applies, and what it says.

    Attributes:
        key: The key sequence, as :class:`QKeySequence` accepts it.
        description: The line shown in the F1 list. Every binding has one.
        action: What pressing the key does.
        tab: The tab this key belongs to, or :data:`ANY_TAB`.
        group: The heading the F1 list files this key under. Required, because
            a key with nowhere to go is one the reader has to place themselves.
        scope: Restrict the key to this widget and its children instead of the
            whole window. Needed where a tab holds editors with arrow-key
            behaviour of their own, which a window-wide key would take first.
        anywhere: Let the key fire from any of the application's windows rather
            than only the one it is bound to. A window-scoped shortcut is dead
            while a second top-level window holds the keyboard, and a dialog --
            even a parented one -- is a second top-level window.
    """

    key: str
    description: str
    action: Callable[[], None]
    tab: str
    group: str
    scope: QWidget | None = None
    anywhere: bool = False


#: Kept clear of the screen edges when the list is capped to fit, so it still
#: reads as a window that can be moved rather than something wedged into the
#: corner.
SCREEN_MARGIN = 64


class ShortcutHelp(QDialog):
    """The keys for the tab in front, listed beside the window.

    Deliberately not modal, and closed by the key that opened it. A reference
    you have to dismiss before you can try what it just told you is a reference
    read once and then guessed at.
    """

    def __init__(self, parent) -> None:
        """Build the dialog against the window it belongs to."""
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._text = QLabel()
        self._text.setTextFormat(Qt.RichText)
        # Two tabs run past 800 pixels now the legend is in the list, so it
        # scrolls rather than being cut short: every row answers a question
        # somebody has had. A dialog merely sized past the screen is worse than
        # either, having no scrollbar the user can reach.
        self._scroll = QScrollArea()
        self._scroll.setWidget(self._text)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        layout.addWidget(self._scroll)
        self._buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self._buttons.rejected.connect(self.close)
        layout.addWidget(self._buttons)

    def fit_to_screen(self) -> None:
        """Size to the whole list, or to the screen where that is smaller.

        Not ``adjustSize``: a scroll area reports a small default hint rather
        than the size of what is inside it, so the dialog would come up a stub
        whatever the list said.
        """
        self._text.adjustSize()
        wanted = self._text.sizeHint()
        margins = self.layout().contentsMargins()
        around = margins.top() + margins.bottom() + self.layout().spacing()
        tall = wanted.height() + around + self._buttons.sizeHint().height()
        wide = wanted.width() + margins.left() + margins.right()

        screen = self.screen() or QApplication.primaryScreen()
        room = screen.availableGeometry()
        height = min(tall, room.height() - SCREEN_MARGIN)
        if height < tall:
            # Room for the scrollbar the cap has just made necessary, or the
            # last column of the table goes under it.
            wide += self._scroll.verticalScrollBar().sizeHint().width()
        self.resize(min(wide, room.width() - SCREEN_MARGIN), height)

    def is_in_front(self) -> bool:
        """Return whether this list is the window a key press is aimed at.

        Not ``isVisible()``: a list left open behind the window is still
        visible, and treating that as "in front" is what made F1 dismiss,
        unseen, the thing it was pressed to see.
        """
        return self.isActiveWindow()

    def show_for(self, tab_name: str, help_text: str) -> None:
        """List *tab_name*'s keys, replacing whatever was listed before."""
        self.setWindowTitle(f"Keyboard shortcuts: {tab_name}")
        self._text.setText(help_text)
        self.fit_to_screen()
        self.show()
        self.raise_()
        # Not just raised: a window that is on top but not activated still
        # leaves the keyboard where it was, and on macOS receives no pointer
        # motion, so the list would not even show its own tooltips.
        self.activateWindow()


def bindings_for(window) -> list[Binding]:
    """Return every key *window* offers, tagged with its tab and its heading.

    A binding scoped to a widget uses ``WidgetWithChildrenShortcut`` and so
    only fires with focus inside it; every other binding belongs to the
    window and is switched on and off as tabs change.

    **The order below is the order F1 reads in.** Headings appear in the
    order the bindings first name them, so grouping the list is a matter of
    moving a line here rather than of maintaining a table of headings
    elsewhere that can fall out of step with it.
    """
    curate = TAB_CURATE
    paint_keys = [
        Binding(
            key,
            f"Paint {name}",
            lambda position=position: window.label_panel.select_class_at(position),
            TAB_LABEL,
            GROUP_CLASSES,
        )
        for position, (key, name) in enumerate(
            zip(CLASS_KEYS, window.label_panel.class_names())
        )
    ]
    return [
        # -- curate --------------------------------------------------
        Binding("Right", "Next frame", lambda: window._step_frame(1), curate, GROUP_VIEW),
        Binding("Left", "Previous frame", lambda: window._step_frame(-1), curate, GROUP_VIEW),
        Binding("Down", "Next lamella", lambda: window._step_lamella(1), curate, GROUP_VIEW),
        Binding("Up", "Previous lamella", lambda: window._step_lamella(-1), curate, GROUP_VIEW),
        Binding(
            "1", "SEM view", lambda: window._set_mode("sem", from_user=True),
            curate, GROUP_VIEW,
        ),
        Binding(
            "2", "FIB view", lambda: window._set_mode("fib", from_user=True),
            curate, GROUP_VIEW,
        ),
        Binding(
            "3", "Plot view", lambda: window._set_mode("plot", from_user=True),
            curate, GROUP_VIEW,
        ),
        Binding(
            "V", "Show / hide the label overlay", window._toggle_overlay,
            curate, GROUP_VIEW,
        ),
        Binding(
            "G", "Fit the view to the pane", window.image_panel.fit, curate, GROUP_VIEW
        ),
        Binding(
            "Esc", "Clear the picked-out frames", window._clear_frame_selection,
            curate, GROUP_CHOOSE,
        ),
        Binding(
            "D", "Set state: deselected", lambda: window.apply_state("deselected"),
            curate, GROUP_STATE,
        ),
        Binding(
            "S", "Set state: selected", lambda: window.apply_state("selected"),
            curate, GROUP_STATE,
        ),
        Binding(
            "C", "Set state: corrected", lambda: window.apply_state("corrected"),
            curate, GROUP_STATE,
        ),
        # -- add data ------------------------------------------------
        # Scoped to the project list: the fields beside it include editable
        # combo boxes, whose own Up/Down a window-wide key would swallow.
        Binding(
            "Down", "Next project", lambda: window.metadata_panel.step_project(1),
            TAB_ADD_DATA, GROUP_VIEW, scope=window.metadata_panel.projects,
        ),
        Binding(
            "Up", "Previous project", lambda: window.metadata_panel.step_project(-1),
            TAB_ADD_DATA, GROUP_VIEW, scope=window.metadata_panel.projects,
        ),
        # -- label ---------------------------------------------------
        Binding(
            "Right", "Next frame", lambda: window.label_panel.step_frame(1),
            TAB_LABEL, GROUP_VIEW,
        ),
        Binding(
            "Left", "Previous frame", lambda: window.label_panel.step_frame(-1),
            TAB_LABEL, GROUP_VIEW,
        ),
        Binding(
            "Down", "Next lamella", lambda: window._step_label_lamella(1),
            TAB_LABEL, GROUP_VIEW,
        ),
        Binding(
            "Up", "Previous lamella", lambda: window._step_label_lamella(-1),
            TAB_LABEL, GROUP_VIEW,
        ),
        Binding(
            "1", "SEM images", lambda: window.label_panel.set_modality("SEM"),
            TAB_LABEL, GROUP_VIEW,
        ),
        Binding(
            "2", "FIB images", lambda: window.label_panel.set_modality("FIB"),
            TAB_LABEL, GROUP_VIEW,
        ),
        Binding(
            "V", "Show / hide the label", window.label_panel.toggle_label,
            TAB_LABEL, GROUP_VIEW,
        ),
        # Held rather than pressed, so the canvas turns it off again on
        # release; this entry is what puts it in the F1 list, the binding
        # machinery having no notion of a key you hold.
        Binding(
            "B", "Hold to show the earlier version chosen below",
            window.label_panel.hold_compare, TAB_LABEL, GROUP_VIEW,
        ),
        Binding(
            "G", "Fit the image to the pane", window.label_panel.canvas_fit,
            TAB_LABEL, GROUP_VIEW,
        ),
        Binding(
            "D", "Brush tool", lambda: window.label_panel.set_tool("brush"),
            TAB_LABEL, GROUP_TOOLS,
        ),
        Binding(
            "F", "Fill tool", lambda: window.label_panel.set_tool("fill"),
            TAB_LABEL, GROUP_TOOLS,
        ),
        Binding(
            "[", "Smaller brush", lambda: window.label_panel.step_brush(-1),
            TAB_LABEL, GROUP_TOOLS,
        ),
        Binding(
            "]", "Larger brush", lambda: window.label_panel.step_brush(1),
            TAB_LABEL, GROUP_TOOLS,
        ),
        *paint_keys,
        Binding(
            "A", "Lock or unlock every class",
            window.label_panel.toggle_all_locks, TAB_LABEL, GROUP_CLASSES,
        ),
        Binding(
            "X", "Find specks: ring them here, count them along the series",
            window.label_panel.toggle_specks, TAB_LABEL, GROUP_SPECKS,
        ),
        Binding(
            "N", "Go to the next speck",
            lambda: window.label_panel.step_speck(1), TAB_LABEL, GROUP_SPECKS,
        ),
        Binding(
            "Ctrl+S", "Save this lamella's labels", window.save_labels,
            TAB_LABEL, GROUP_EDITS,
        ),
        Binding(
            "Ctrl+Z", "Undo the last stroke", window.label_panel.undo,
            TAB_LABEL, GROUP_EDITS,
        ),
        Binding(
            "Ctrl+Shift+Z", "Redo the last stroke", window.label_panel.redo,
            TAB_LABEL, GROUP_EDITS,
        ),
        # The one key that has to work while the list it opens holds the
        # keyboard, since closing it again is what it is for.
        Binding(
            "F1", "Show this list", window.show_shortcuts,
            ANY_TAB, GROUP_HELP, anywhere=True,
        ),
    ]

def _grouped_rows(
    bindings: list[Binding],
    gestures: tuple[tuple[str, str, str], ...],
    marks: tuple[tuple[str, str, str], ...] = (),
) -> dict[str, list[tuple[str, str]]]:
    """Return the rows of each heading, in the order the headings arise.

    Keys first, then the gestures that belong beside them, so a heading
    reads as its keys and then the pointer equivalents; the legend last,
    under headings of its own that nothing else names. :data:`GROUP_HELP`
    is moved to the end: the key that opens this list is the one the reader
    has already found.
    """
    grouped: dict[str, list[tuple[str, str]]] = {}
    for binding in bindings:
        grouped.setdefault(binding.group, []).append(
            (binding.key, binding.description)
        )
    for group, action, description in (*gestures, *marks):
        grouped.setdefault(group, []).append((action, description))
    if GROUP_HELP in grouped:
        grouped[GROUP_HELP] = grouped.pop(GROUP_HELP)
    return grouped

def _tab_marks(tab: str) -> tuple[tuple[str, str, str], ...]:
    """Return the legend for *tab*, as ``(heading, picture, meaning)``.

    Every picture is drawn by the module that draws the real mark, so the
    list cannot explain one thing and the window show another. Only the
    marks on the tab are listed: the white band means frames picked out
    while curating and unsaved edits while proofreading, which is one mark
    with two meanings and no way to say both at once.
    """
    rows: list[tuple[str, list[tuple[QPixmap, str]]]] = []
    if tab == TAB_CURATE:
        rows = [
            (GROUP_ROW_MARK, status_legend()),
            (
                GROUP_STRIP_MARK,
                timeline_legend("Picked out for a state change"),
            ),
        ]
    elif tab == TAB_LABEL:
        rows = [
            (GROUP_STRIP_MARK, timeline_legend("This label has unsaved edits")),
        ]
    elif tab == TAB_ADD_DATA:
        rows = [(GROUP_ROW_MARK, completeness_legend())]
    return tuple(
        (group, mark_image(sample), meaning)
        for group, legend in rows
        for sample, meaning in legend
    )




class Shortcuts:
    """Every key the window offers: bound, switched per tab, and explained.

    Owns the bindings, the :class:`QShortcut` objects made from them and the
    list that explains them, so the window holds one collaborator rather than
    three parallel lists it has to keep in step.

    Built with the window, and calls back into it for the two things only the
    window can answer: which tab is in front, and what that tab is called.
    """

    def __init__(self, window) -> None:
        """Bind every key *window* offers and switch on the tab in front."""
        self._window = window
        #: The table, in the order F1 reads it.
        self.bindings = bindings_for(window)
        #: One per binding, in the same order -- zipped in several places, so
        #: a shortcut that did not line up with its binding would switch the
        #: wrong key on.
        self.shortcuts = [self._install(binding) for binding in self.bindings]
        #: The list, built the first time F1 asks for it.
        self.help: ShortcutHelp | None = None
        self.apply_tab(window._current_tab())

    def _install(self, binding: Binding) -> QShortcut:
        """Return the live shortcut for *binding*, in the context it asked for."""
        shortcut = QShortcut(
            QKeySequence(binding.key),
            binding.scope or self._window,
            activated=binding.action,
        )
        if binding.scope is not None:
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        elif binding.anywhere:
            shortcut.setContext(Qt.ApplicationShortcut)
        return shortcut

    def apply_tab(self, tab: str) -> None:
        """Switch on the keys belonging to *tab*, and switch off the rest."""
        for binding, shortcut in zip(self.bindings, self.shortcuts):
            shortcut.setEnabled(binding.tab in (tab, ANY_TAB))

    def follow_tab(self, tab: str) -> None:
        """Switch the keys over to *tab*, and re-list them if the list is open.

        A list left showing the previous tab's keys is worse than none: it
        answers "what can I press?" with keys that are now switched off.
        """
        self.apply_tab(tab)
        if self.help is not None and self.help.isVisible():
            self.refill()

    def help_html(self, tab: str | None = None) -> str:
        """Return the shortcut table for a tab, as the help dialog's HTML.

        Under a heading per group, keys and mouse gestures together: what a
        reader is looking for is "how do I pick frames out", and answering that
        across a list of keys and a separate list of gestures makes them do the
        grouping the list should have done.

        Separate from showing it so what the dialog says can be tested without
        opening a modal window.
        """
        tab = tab if tab is not None else self._window._current_tab()
        applicable = [b for b in self.bindings if b.tab in (tab, ANY_TAB)]
        grouped = _grouped_rows(
            applicable, TAB_GESTURES.get(tab, ()), _tab_marks(tab)
        )
        help_text = "".join(
            f"<p style='{HEADING_STYLE}'>{group}</p>{_table(rows)}"
            for group, rows in grouped.items()
        )
        own_keys = [b for b in applicable if b.tab != ANY_TAB]
        if own_keys:
            # Above the tables: which of the keys under your hands do anything
            # here is a question a picture answers faster than a list, and what
            # each one does is the list's own job directly below.
            help_text = (
                f"<p>{mark_image(keyboard_diagram(_keys_of(applicable)))}</p>"
                + help_text
            )
        else:
            help_text += (
                "<p style='margin-top: 10px;'>This tab has no keys of its own.</p>"
            )
        if tab == TAB_CURATE:
            help_text += (
                "<p style='margin-top: 10px;'>State changes need <i>Enable editing</i> "
                "switched on.</p>"
            )
        if tab == TAB_LABEL:
            help_text += (
                "<p style='margin-top: 10px;'>Edits are held in this window until "
                "you save them. Marking the saved set corrected needs "
                "<i>Enable editing</i> switched on.</p>"
            )
        return help_text



    def toggle(self) -> None:
        """List the keys for the tab now showing, or put the list away.

        Tooltips only help someone who already suspects a control exists; this
        is the one place that answers "what can I press?". Only the active tab's
        keys are listed, because the others are switched off and pressing one
        does nothing.

        F1 puts the list away only when the list is the window in front. One
        left open behind the main window is still visible, so closing on that
        alone made F1 answer "what can I press?" by dismissing, unseen, the one
        thing that says -- and the fix for a key that appears to do nothing is
        to press it again, which reopened it and made the whole thing look
        intermittent.

        Nothing happens while an operation is running behind its modal dialog.
        The key reaches here from any window now, and a list opened over a
        modal one cannot be read, moved or closed until the operation ends.
        """
        if QApplication.activeModalWidget() is not None:
            return
        if self.help is not None and self.help.is_in_front():
            self.help.close()
            return
        if self.help is None:
            self.help = ShortcutHelp(self._window)
        self.refill()

    def refill(self) -> None:
        """Fill the shortcut list with the keys of the tab in front."""
        self.help.show_for(self._window.current_tab_name(), self.help_html())

