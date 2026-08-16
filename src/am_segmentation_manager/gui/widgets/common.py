"""Small widget behaviours shared between panels.

What lands here is something more than one panel *does* -- a context menu, an
event filter. What the window looks like belongs in :mod:`theme`, anything that
knows about projects, labels or the filesystem belongs in a controller, and
anything only one panel uses belongs in that panel.
"""
from __future__ import annotations

import logging

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QListWidget, QMenu

logger = logging.getLogger(__name__)


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
