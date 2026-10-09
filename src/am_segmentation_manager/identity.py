"""What the application is called, and the files that picture it.

A constant module outside the four layers, reading nothing and importing
nothing of the package: the window, and anything that names the application to
the desktop, take these from here so they cannot drift apart. It holds no Qt,
so naming the application never requires the window's dependencies.
**:data:`APPLICATION_ID` and :data:`VIEWER_ID` never change once shipped**: the
desktop recognises each application by its own, and a new one is a different one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

#: The name shown beside the mark and given to the desktop. Spelled once so
#: the window, the banner and a launcher cannot drift apart on capitalisation.
PRODUCT_NAME: Final[str] = "AM segmentation manager"

#: What the window is called when opened only to look, so a viewer on the
#: microscope is never mistaken for the window that edits.
VIEWER_NAME: Final[str] = "AM segmentation viewer"

#: How the desktop knows the application: the Linux desktop entry's name and
#: the macOS bundle identifier. The distribution's own name, which PyPI keeps
#: unique, rather than a domain that belongs to an employer.
APPLICATION_ID: Final[str] = "adaptive-milling-seg"

#: How the desktop knows the viewer, a launcher of its own beside the window's.
VIEWER_ID: Final[str] = "adaptive-milling-seg-viewer"

#: Where the source is published and newer versions are looked for; the
#: project's Homepage in ``pyproject.toml``.
REPOSITORY_URL: Final[str] = "https://github.com/Casper-rfi/AM_segmentation_manager"

#: Where the packaged assets are, located the way :mod:`.config` locates the
#: packaged segmentation config.
DATA_DIRECTORY: Final[Path] = Path(__file__).parent / "data"

#: The mark, as a PNG: Qt reads PNG through a built-in plugin on every
#: platform, and the Linux desktops take it as an icon.
ICON_PATH: Final[Path] = DATA_DIRECTORY / "AM_icon.png"

#: The mark set in the macOS icon container, for the Dock. A separate file
#: rather than a scaled :data:`ICON_PATH`, because the container is part of the
#: artwork there and would be a white box anywhere the mark is drawn against
#: the window's own background; `scripts/build_mac_icon.py` builds it.
MAC_ICON_PATH: Final[Path] = DATA_DIRECTORY / "AM_icon.icns"

#: The mark as a Windows icon, which is what a Windows shortcut takes.
WINDOWS_ICON_PATH: Final[Path] = DATA_DIRECTORY / "AM_icon.ico"
