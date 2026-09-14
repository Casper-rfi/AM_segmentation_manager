"""Where the Adaptive Milling mark lives, and how to put it on screen.

One module, so the window, the application icon and the scan banner agree about
which file the mark is and how it is scaled. A missing asset is never an error: a
caller gets an empty pixmap and draws nothing. The mark is free-form against the
window's own background and set in the system's container as the macOS icon, the
Dock drawing none of its own. **It is the only place orange and green appear**,
both being segmentation class colours; see ``docs/display.md``.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap

logger = logging.getLogger(__name__)

_DATA = Path(__file__).parent.parent.parent / "data"

#: The mark, as a PNG rather than the ``.ico`` beside it: Qt reads PNG through
#: a built-in plugin on every platform, where ICO buys nothing outside Windows.
#: Located the way :mod:`...config` locates the packaged segmentation config,
#: for the reason given there.
ICON_PATH = _DATA / "AM_icon.png"

#: The same mark set in the macOS icon container, for the Dock. A separate
#: file rather than a scaled :data:`ICON_PATH`, because the container is part
#: of the artwork there and would be a white box anywhere the mark is drawn
#: against the window's own background; `scripts/build_mac_icon.py` builds it.
MAC_ICON_PATH = _DATA / "AM_icon.icns"

#: The name shown beside the mark. Spelled here so the window, the banner and
#: anything later cannot drift apart on capitalisation.
PRODUCT_NAME = "AM segmentation manager"


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
        size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )
