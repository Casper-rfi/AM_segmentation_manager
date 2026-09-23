"""How to put the Adaptive Milling mark on screen.

One module, so the window, the application icon and the scan banner agree about
how the mark is scaled; which file it is, and the name beside it, are
:mod:`...identity`'s. A missing asset is never an error: a
caller gets an empty pixmap and draws nothing. The mark is free-form against the
window's own background and set in the system's container as the macOS icon, the
Dock drawing none of its own. **Its orange and green are class colours**, so the
accents softened from them stay on controls; see ``docs/display.md``.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap

# Named here as well as in `identity`, so a test can point this module at a
# missing file without moving the mark for anything else.
from ...identity import ICON_PATH, MAC_ICON_PATH

logger = logging.getLogger(__name__)


def app_icon_path() -> Path:
    """Return the file the application icon should be read from.

    The macOS container on Darwin and the free-form mark everywhere else:
    a Dock icon draws its own container and reads as foreign without one,
    where Windows and the Linux desktops frame what they are given. A checkout
    without the container falls back to the mark rather than to nothing.
    """
    if sys.platform == "darwin" and MAC_ICON_PATH.exists():
        return MAC_ICON_PATH
    return ICON_PATH


def logo_pixmap(size: int) -> QPixmap:
    """Return the mark at *size* logical pixels square, or an empty pixmap.

    Scaled smoothly and kept square: the source is square, and letting a layout
    stretch it is the one way this mark can look wrong.
    """
    if not ICON_PATH.exists():
        logger.debug("No logo at %s", ICON_PATH)
        return QPixmap()
    pixmap = QPixmap(str(ICON_PATH))
    if pixmap.isNull():  # pragma: no cover - a corrupt asset, not a missing one
        logger.warning("Could not read the logo at %s", ICON_PATH)
        return QPixmap()
    return pixmap.scaled(
        size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    )
