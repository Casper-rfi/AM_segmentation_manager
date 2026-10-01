"""Small widget behaviours shared between panels.

What lands here is something more than one panel *does* and that is too small
to be a module of its own -- a caption, a legend, an event filter, a badge.
The controls this package paints are :mod:`controls`, the two list widgets
:mod:`lists`, asking for a path :mod:`paths`, and the shortcut list's keyboard
:mod:`keyboard`. What the window looks like belongs in :mod:`theme`, anything
that knows about projects, labels or the filesystem belongs in a controller,
and anything only one panel uses belongs in that panel.
"""
from __future__ import annotations

import base64
import functools
import logging
import math
from collections.abc import Sequence

from qtpy.QtCore import (
    QBuffer,
    QByteArray,
    QEvent,
    QObject,
    QPointF,
    Qt,
)
from qtpy.QtGui import (
    QColor,
    QFont,
    QIcon,
    QPainter,
    QPixmap,
    QPolygonF,
    QStandardItem,
    QStandardItemModel,
)
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QCompleter,
    QFrame,
    QLabel,
    QWidget,
)

from ...config import SegmentationConfig
from ...display import DEFAULT_INTENSITY_RANGE
from ...frames import ConditionChoice
from ...layout import family_key
from . import theme

logger = logging.getLogger(__name__)


#: What each label status is called, and how strongly it is said. Shared by the
#: curation and proofreading panels: the two report the same fact about the same
#: file, and a second wording for it would read as a second meaning.
LABEL_BADGES = {
    "missing": ("no label", "warning"),
    "empty": ("empty label - not yet annotated", "warning"),
    "present": ("", "info"),
}

#: What a series with no labels says in place of :data:`LABEL_BADGES`. A scope
#: rather than a fault: an ``other`` image has nowhere a label could be
#: written, so "no label" would report something that could never be otherwise.
VIEW_ONLY_BADGE = ("view only - labelling is not offered for this series", "info")


def badge_markup(badges: Sequence[tuple[str, str]]) -> str:
    """Return *badges*, each ``(text, level)``, as the rich text a label shows them in."""
    return "&nbsp;&nbsp;|&nbsp;&nbsp;".join(
        f'<span style="{theme.BADGE_STYLES[level]}">{text}</span>' for text, level in badges
    )


#: How many filenames a confirmation lists before summarising the rest. Shared
#: because a bulk state change and a batch operation both list what they are
#: about to touch, and two caps would have one dialog stop where the other did
#: not.
MAX_LISTED_FILES = 12


def class_legend_html(config: SegmentationConfig) -> str:
    """Return a swatch and name for each foreground class, as rich text.

    Background is left out: it covers most of a frame and is never drawn, so
    naming it in the legend would describe a colour nothing uses.
    """
    parts = []
    for class_index in config.foreground_classes:
        red, green, blue = config.rgb(class_index)
        parts.append(
            f'<span style="color: rgb({red},{green},{blue});">&#9632;</span> '
            f"{config.name(class_index)}"
        )
    return "&nbsp;&nbsp;".join(parts)


def chip_icon(pixmap: QPixmap) -> QIcon:
    """Return an icon Qt will draw the same way on every row.

    A view draws the icon of the row being looked at in ``QIcon.Selected``
    mode, and the variant Qt generates for that mode blends the pixmap with the
    highlight colour. For a status chip that is wrong twice over: the colour
    *is* the information, and it gets washed out on exactly the row the user is
    on -- a "done" teal came out as a blue-green against a blue-tinted row.

    Supplying the pixmap for every mode and state leaves nothing for Qt to
    generate.
    """
    icon = QIcon()
    for mode in (QIcon.Mode.Normal, QIcon.Mode.Selected, QIcon.Mode.Active, QIcon.Mode.Disabled):
        for state in (QIcon.State.On, QIcon.State.Off):
            icon.addPixmap(pixmap, mode, state)
    return icon


def mark_image(pixmap: QPixmap) -> str:
    """Return *pixmap* as an ``<img>`` tag Qt's rich text will draw inline.

    So that a legend shows the mark itself rather than a description of it: the
    pixmap comes from the same painter that draws the real thing, and the two
    cannot drift apart. The bytes travel in the tag as a ``data:`` URI, there
    being no file to point at and ``QLabel`` offering no document to attach a
    resource to.

    The byte array is held in a name rather than passed as a temporary:
    ``QBuffer(QByteArray())`` lets the array die under the buffer and segfaults
    PySide6 on the write.
    """
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QBuffer.OpenModeFlag.WriteOnly)
    try:
        pixmap.save(buffer, "PNG")
    finally:
        buffer.close()
    encoded = base64.b64encode(data.data()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}">'


#: Monospaced families to try, in the order they are looked for. The style hint
#: below is only consulted for a family that does not exist, so the list has to
#: carry a real name for each platform rather than relying on the hint alone.
_FIXED_WIDTH_FAMILIES = ("Menlo", "Consolas", "DejaVu Sans Mono", "Courier New")


def fixed_width_font() -> QFont:
    """Return a font whose characters are all the same width.

    Anything padded into columns with spaces -- the breakdown table, the
    operation reports -- is only aligned in a font like this.

    Not ``QFontDatabase.systemFont(FixedFont)``, which is the obvious call and
    returns the proportional UI font on macOS: it reports the family it was
    asked for while resolving to something else, so the padding stays padding
    and nothing lines up. Naming families and letting the style hint cover the
    ones that are missing is what actually resolves.
    """
    font = QFont()
    font.setFamilies(list(_FIXED_WIDTH_FAMILIES))
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setFixedPitch(True)
    return font


def horizontal_rule() -> QFrame:
    """Return a hairline for dividing one group of controls from the next.

    Painted rather than left to the style, which draws a sunken frame as two
    lines of its own choosing -- a groove cut across a panel whose every other
    edge is flat. :data:`theme.BORDER` rather than the lighter separator: this
    rule is the only thing saying the groups differ, and against a group box
    the light one cannot be seen at all.
    """
    rule = QFrame()
    rule.setFrameShape(QFrame.Shape.HLine)
    rule.setFrameShadow(QFrame.Shadow.Plain)
    rule.setFixedHeight(1)
    rule.setStyleSheet(f"background: {theme.BORDER}; border: none;")
    return rule


def vertical_rule(height: int) -> QFrame:
    """Return a hairline for dividing one run of controls from the next in a row.

    :func:`horizontal_rule` turned on its side, and for the same reason: the
    style draws a sunken frame as two lines of its own choosing. A height is
    asked for rather than taken from the row, a rule stretched to a layout's
    full height reading as a border rather than as a separator.
    """
    rule = QFrame()
    rule.setFrameShape(QFrame.Shape.VLine)
    rule.setFrameShadow(QFrame.Shadow.Plain)
    rule.setFixedWidth(1)
    rule.setFixedHeight(height)
    rule.setStyleSheet(f"background: {theme.BORDER}; border: none;")
    return rule


def caption_for(text: str, widget: QWidget, tooltip: str) -> QLabel:
    """Return a caption for *widget*, putting *tooltip* on both.

    A user reaches for the word naming a control, not the control itself, so a
    tooltip on the control alone is one nobody finds. Returning the label keeps
    the two impossible to write apart.
    """
    label = QLabel(text)
    label.setToolTip(tooltip)
    widget.setToolTip(tooltip)
    return label


def ticked_caption(noun: str, total: int, ticked: int, hidden: int = 0) -> str:
    """Return a list's caption, saying how many of its rows are ticked.

    Spelled "3 of 12 ticked" rather than as two bare numbers: the ticks decide
    what an operation covers, the lists scroll, and a reader should not have to
    work out which number is which. *hidden* counts rows a filter leaves out,
    which a list that says nothing of them would drop in silence.
    """
    counts = []
    if total:
        counts.append(f"{ticked} of {total} ticked" if ticked else f"{total}")
    if hidden:
        counts.append(f"{hidden} hidden")
    return f"{noun} ({', '.join(counts)})" if counts else noun


def match_anywhere(box: QComboBox) -> None:
    """Have *box* suggest on any part of a value, and let Tab take the suggestion.

    **The popup is not cosmetic.** An editable ``QComboBox`` builds its
    completer in ``InlineCompletion`` mode, which assumes the match is a prefix
    and keeps what was typed while appending the rest of the completion. Set to
    match anywhere and left inline, typing ``a`` against "UltrAuFoil Au R2/2 200
    mesh" writes "altrAuFoil Au R2/2 200 mesh" into the field -- a plausible
    value nobody typed, in metadata that is exported. A popup replaces the whole
    text instead, which is what makes matching anywhere safe.

    **Tab has to be taught to accept**, which is the other half of the same
    fault. A completer's popup answers to Return and to a click; Tab is not its
    key, so it reaches the field, moves the focus, and leaves whatever was
    half-typed behind as the value -- one letter, committed as a chip nobody
    meant. So Tab takes the suggestion on show and stays in the field.

    **The filter goes on the popup, and on the field.** An open popup grabs the
    keyboard, so the key arrives at the popup; ``QCompleter`` then passes it on
    by calling ``widget->event()``, which reaches the field's own handlers but
    **not the event filters installed on it**. Watching only the field is
    therefore watching the one route the key does not take -- except under the
    offscreen platform, which cannot grab a keyboard and so delivers straight to
    the field. Both are watched, because a test that only proves the second
    proves nothing about the window.

    Matching anywhere is worth both: "Quantifoil" and "UltrAuFoil" are told
    apart by their middles, not by how they start.
    """
    completer = box.completer()
    assert completer is not None, "an editable combo box has a completer"
    completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
    completer.setFilterMode(Qt.MatchFlag.MatchContains)
    completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)

    taking = _CompleteOnTab(box)
    line = box.lineEdit()
    if line is not None:
        line.installEventFilter(taking)
    # Creates the popup if it does not exist yet, which is what makes it
    # filterable now rather than on first use. It outlives the suggestions
    # being replaced, so it is installed on once.
    popup = completer.popup()
    assert popup is not None, "asking for the popup creates it"
    popup.installEventFilter(taking)


class _CompleteOnTab(QObject):
    """Makes Tab take what the completer's popup is offering.

    Installed on both the popup and the field, one of which gets the key
    depending on whether the platform grants the popup a keyboard grab. Parented
    to the box, an event filter being kept alive by whoever installed it and by
    nothing else.
    """

    def __init__(self, box: QComboBox) -> None:
        super().__init__(box)
        self._box = box

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt naming
        """Take the suggestion on Tab, and let every other key through."""
        if event.type() != QEvent.Type.KeyPress or event.key() != Qt.Key.Key_Tab:
            return False
        completer = self._box.completer()
        popup = completer.popup() if completer is not None else None
        if completer is None or popup is None or not popup.isVisible():
            return False
        chosen = self._offered(completer, popup)
        if not chosen:
            return False
        popup.hide()
        self._box.setCurrentText(chosen)
        # Hidden again after setting the text: writing to the field is an edit
        # like any other, so the completer answers it by offering the value it
        # has just been given.
        popup.hide()
        line = self._box.lineEdit()
        if line is not None:
            line.setCursorPosition(len(chosen))
        return True

    @staticmethod
    def _offered(completer: QCompleter, popup: QAbstractItemView) -> str:
        """Return the row the popup is on, or the first it would offer.

        A popup that has just appeared has no current row until someone arrows
        into it, and that is exactly when Tab is pressed.
        """
        index = popup.currentIndex()
        if index.isValid():
            return str(index.data())
        return completer.currentCompletion()


#: The item role holding how many leading characters of a condition's name
#: are its family, drawn bold; a row without it is not a condition. See
#: :class:`~.controls.ElidingCombo`.
FAMILY_LENGTH_ROLE = Qt.ItemDataRole.UserRole + 1

#: The item role holding the family a condition's row belongs to.
FAMILY_ROLE = Qt.ItemDataRole.UserRole + 2

#: The item role holding whether a condition's row carries the primary star.
PRIMARY_ROLE = Qt.ItemDataRole.UserRole + 3

#: Side of the primary star's square, in logical pixels: a combo's icon size.
_STAR_SIZE = 16


@functools.cache
def primary_star() -> QIcon:
    """Return the star marking a family's primary: the image predicted, measured and exported.

    Drawn rather than typed, a typed star resting on whichever fonts a
    platform has. The text colour rather than the accent, which a preference
    changes under a list that is not refilled for it.
    """
    pixmap = QPixmap(_STAR_SIZE, _STAR_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(theme.TEXT))
    centre, outer, inner = _STAR_SIZE / 2, _STAR_SIZE * 0.42, _STAR_SIZE * 0.17
    painter.drawPolygon(QPolygonF([
        QPointF(
            centre + radius * math.sin(math.pi * point / 5),
            centre + 0.5 - radius * math.cos(math.pi * point / 5),
        )
        for point in range(10)
        for radius in ((outer if point % 2 == 0 else inner),)
    ]))
    painter.end()
    return chip_icon(pixmap)


@functools.cache
def no_primary_star() -> QIcon:
    """Return a blank the star's size, so a row without it lines up with one that has it."""
    pixmap = QPixmap(_STAR_SIZE, _STAR_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    return chip_icon(pixmap)

#: How wide a Variant dropdown is, in both views. Fixed rather than following
#: its entries, which would move the row on each step to a lamella imaged under
#: other conditions. Wide enough for ``SEM-test_8_fi_16`` whole, and no wider:
#: at 155 the Curate controls wrap on the smallest screen supported
#: (``test_the_view_controls_are_one_block_on_the_smallest_screen_supported``).
CONDITION_FIELD_WIDTH = 150


def fill_choice_combo(
    combo: QComboBox,
    entries: Sequence[tuple[str, object]],
    current: object,
) -> None:
    """Offer *entries* in *combo*, select *current*, and grey it if there is one.

    Rebuilt only when the entries have actually changed. The view controls are
    refilled on every step through a lamella, and a combo cleared that often
    closes its own popup under the pointer.

    Args:
        entries: ``(caption, value)`` pairs, the value being what a reader
            takes from ``itemData`` and never has to map back from a caption.
        current: The value to show. A value not among *entries* falls to the
            first, so a caller need not check.

    A combo given a count by :func:`choice_count_badge` has it kept in step:
    shown with the number of entries where there are several, hidden where
    there is nothing to choose.
    """
    entries = list(entries)
    held = [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())]
    combo.blockSignals(True)
    if held != entries:
        combo.clear()
        for caption, value in entries:
            combo.addItem(caption, value)
    values = [value for _caption, value in entries]
    combo.setCurrentIndex(values.index(current) if current in values else 0)
    combo.blockSignals(False)
    _show_choice_count(combo, len(entries))


def fill_condition_combo(
    combo: QComboBox, choices: Sequence[ConditionChoice], current: str | None
) -> None:
    """Offer imaging conditions by folder name, under a heading per family where there are several.

    A row is the name as on disk, ``SEM-test_8_fi_16``, with its family
    marked by :data:`FAMILY_LENGTH_ROLE`, so the closed field needs no
    other name for it. A family's primary carries :func:`primary_star` where
    the family offers more than one condition
    (``test_the_primary_of_a_family_of_several_is_starred``). Where a beam
    holds several families each is a heading that cannot be chosen, as the
    Export tab's grouped lists have, which is what says which conditions share
    a label. Rebuilt only when the rows change, for the reason
    :func:`fill_choice_combo` gives.

    Args:
        choices: In condition order, so each family's are together.
        current: The condition on screen. One not offered falls to the first.
    """
    choices = list(choices)
    several = len({family_key(choice.family) for choice in choices}) > 1
    sizes: dict[str, int] = {}
    for choice in choices:
        sizes[family_key(choice.family)] = sizes.get(family_key(choice.family), 0) + 1

    # (text, condition, family length, family, starred); a heading has no length.
    rows: list[tuple[str, str | None, int | None, str, bool]] = []
    heading: str | None = None
    for choice in choices:
        if several and (heading is None or family_key(heading) != family_key(choice.family)):
            heading = choice.family
            rows.append((heading, None, None, "", False))
        # A family of one has nothing to choose between, so no star to say which.
        starred = choice.primary and sizes[family_key(choice.family)] > 1
        rows.append(
            (choice.text, choice.condition, choice.family_length, choice.family, starred)
        )
    any_starred = any(row[4] for row in rows)

    enabled = Qt.ItemFlag.ItemIsEnabled
    held = [
        (
            combo.itemText(i), combo.itemData(i), combo.itemData(i, FAMILY_LENGTH_ROLE),
            combo.itemData(i, FAMILY_ROLE) or "", bool(combo.itemData(i, PRIMARY_ROLE)),
        )
        for i in range(combo.count())
    ]
    combo.blockSignals(True)
    if held != rows:
        model = QStandardItemModel(combo)
        for text, condition, family_length, family, starred in rows:
            item = QStandardItem(text)
            if family_length is None:
                item.setFlags(Qt.ItemFlag.NoItemFlags)
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            else:
                item.setData(condition, Qt.ItemDataRole.UserRole)
                item.setData(family_length, FAMILY_LENGTH_ROLE)
                item.setData(starred, PRIMARY_ROLE)
                if any_starred:
                    item.setIcon(primary_star() if starred else no_primary_star())
                if family:
                    # A condition AM_families.yaml moved does not say its family by name.
                    item.setData(family, FAMILY_ROLE)
                    item.setToolTip(
                        f"Family {family}; its primary, predicted, measured and exported from"
                        if starred else f"Family {family}"
                    )
            model.appendRow(item)
        combo.setModel(model)
    selectable = [
        row for row in range(combo.count())
        if combo.model().index(row, combo.modelColumn()).flags() & enabled
    ]
    chosen = next((row for row in selectable if combo.itemData(row) == current), None)
    if chosen is None:
        chosen = selectable[0] if selectable else -1
    combo.setCurrentIndex(chosen)
    combo.blockSignals(False)
    _show_choice_count(combo, len(selectable))


def _show_choice_count(combo: QComboBox, count: int) -> None:
    """Grey *combo* where it offers one entry, and keep its count badge in step."""
    # Greyed rather than hidden where there is nothing to choose: the control
    # still says what is on screen, and the row beside it does not move as the
    # reviewer steps onto a lamella that has only one.
    combo.setEnabled(count > 1)
    single = count <= 1
    if combo.property("single") != single:
        # Re-polished, a stylesheet reading a property only when polished.
        combo.setProperty("single", single)
        combo.style().unpolish(combo)
        combo.style().polish(combo)
    badge = getattr(combo, "count_badge", None)
    if badge is not None:
        badge.setText(str(count))
        badge.setVisible(not single)


def choice_count_badge(combo: QComboBox) -> QLabel:
    """Return the round count shown beside *combo* while it offers several entries.

    Kept in step by :func:`fill_choice_combo`. Its room is held while hidden,
    so the controls beside it do not move as a lamella with one condition gives
    way to one with three.
    """
    badge = QLabel("")
    badge.setObjectName(theme.CHOICE_COUNT)
    badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
    badge.setFixedHeight(16)
    badge.setMinimumWidth(16)
    policy = badge.sizePolicy()
    policy.setRetainSizeWhenHidden(True)
    badge.setSizePolicy(policy)
    badge.setToolTip("How many there are to choose between")
    badge.setVisible(False)
    setattr(combo, "count_badge", badge)  # read back by fill_choice_combo with getattr
    return badge


def fill_grouped_combo(
    combo: QComboBox,
    groups: Sequence[tuple[str, Sequence[tuple[str, str]]]],
    *,
    first: tuple[str, str] | None = None,
) -> None:
    """Fill *combo* with rows gathered under non-selectable headings.

    Each row carries its identifier as item data, so a caller reads
    ``currentData()`` and never has to map a display label back to a name.
    A heading carries none, which is what distinguishes the two.

    Args:
        groups: ``(heading, rows)`` pairs, where each row is
            ``(identifier, label)``.
        first: A row to place above every group, for an entry that belongs to
            no group -- an "any" or "none" option.
    """
    model = QStandardItemModel(combo)
    if first is not None:
        name, label = first
        item = QStandardItem(label)
        item.setData(name, Qt.ItemDataRole.UserRole)
        model.appendRow(item)
    for heading, rows in groups:
        item = QStandardItem(heading)
        # NoItemFlags rather than clearing ItemIsSelectable alone: a heading
        # must also be skipped when stepping through the list with the arrow
        # keys, and Qt skips disabled rows for free.
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        model.appendRow(item)
        for name, label in rows:
            row = QStandardItem(label)
            row.setData(name, Qt.ItemDataRole.UserRole)
            model.appendRow(row)
    combo.setModel(model)


def intensity_badge(limits: tuple[int, int]) -> list[tuple[str, str]]:
    """Return the badge saying the picture differs from the file, or nothing.

    In the warning colour, because the intensity limits are the one control
    that makes what is on screen differ from what was recorded: a reviewer who
    has forgotten they are on is deciding about an image no export contains.
    Nothing at all over the whole range, a line that is always there being a
    line nobody reads.

    Here rather than in either panel: both the curation panes and the
    proofreading canvas carry the control, and two spellings of this would be
    two answers to "is what I am looking at the file".
    """

    if tuple(limits) == tuple(DEFAULT_INTENSITY_RANGE):
        return []
    low, high = limits
    return [(f"shown {low}-{high}", "warning")]
