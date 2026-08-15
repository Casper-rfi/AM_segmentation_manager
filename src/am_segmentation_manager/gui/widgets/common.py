"""Small widget helpers shared between panels.

What lands here is a presentation primitive that more than one panel needs --
a colour vocabulary, a palette adjustment. Anything that knows about projects,
labels or the filesystem belongs in a controller instead, and anything only one
panel uses belongs in that panel.
"""
from __future__ import annotations

import logging

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QListWidget, QMenu, QWidget

logger = logging.getLogger(__name__)

#: How a short annotation beside a value is coloured. Three levels, so a panel
#: says what it means rather than picking a colour.
BADGE_STYLES = {
    "warning": "color: #c08a34; font-weight: bold;",
    "error": "color: #c0504d; font-weight: bold;",
    "info": "color: #888888;",
}

#: Every chip drawn into a list row gets a dark outline. Without one a chip
#: whose fill is near the theme's selection highlight disappears on exactly the
#: row the user is looking at. An outline makes it legible on any background
#: rather than tuning the fill against one theme.
SWATCH_OUTLINE = QColor("#1a1a1a")

SWATCH_WIDTH = 14
SWATCH_HEIGHT = 10


def add_path_menu(
    widget: QListWidget, noun: str, copy_signal, reveal_signal
) -> None:
    """Give a list a right-click menu for the row's filesystem path.

    The row under the cursor is used, not the selected row: right-clicking an
    entry and acting on a different one would be a surprising way to lose track
    of which path you copied. The row number is emitted rather than a path,
    because resolving one is the controller's job, not a widget's.
    """
    widget.setContextMenuPolicy(Qt.CustomContextMenu)

    def show_menu(point) -> None:
        item = widget.itemAt(point)
        if item is None:
            return
        row = widget.row(item)
        menu = QMenu(widget)
        copy = menu.addAction(f"Copy {noun} path")
        reveal = menu.addAction(f"Reveal {noun} in file manager")
        chosen = menu.exec_(widget.viewport().mapToGlobal(point))
        if chosen is copy:
            copy_signal.emit(row)
        elif chosen is reveal:
            reveal_signal.emit(row)

    widget.customContextMenuRequested.connect(show_menu)


def keep_readable_when_disabled(widget: QWidget) -> None:
    """Make *widget* draw its text in the normal colour while disabled.

    The disabled background is left alone, so the widget still reads as locked
    while its value stays legible. Colours are taken from the palette in effect
    now, so this follows the system theme rather than pinning a colour.
    """
    palette = widget.palette()
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        palette.setColor(
            QPalette.Disabled, role, palette.color(QPalette.Active, role)
        )
    widget.setPalette(palette)
