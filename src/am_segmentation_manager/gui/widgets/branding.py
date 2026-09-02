"""Where the Adaptive Milling mark lives, and how to put it on screen.

One module so the window, the application icon and the scan banner cannot
disagree about which file the mark is or how it is scaled. A missing asset is
never an error: a source checkout without it must still open the window, so
every caller here gets an empty pixmap and draws nothing.

The mark is the *only* place this package uses its orange and green. Both are
already spoken for as segmentation class colours, and the timeline draws its
states beside that legend, so a second meaning for either would be read as
data; see `docs/display.md`.
"""
from __future__ import annotations

import logging
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap

logger = logging.getLogger(__name__)

#: The mark, as a PNG rather than the ``.ico`` beside it: Qt reads PNG through
#: a built-in plugin on every platform, where ICO buys nothing outside Windows.
#: Located the way :mod:`...config` locates the packaged segmentation config,
#: for the reason given there.
ICON_PATH = Path(__file__).parent.parent.parent / "data" / "AM_icon.png"

#: The name shown beside the mark. Spelled here so the window, the banner and
#: anything later cannot drift apart on capitalisation.
PRODUCT_NAME = "AM segmentation manager"


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
