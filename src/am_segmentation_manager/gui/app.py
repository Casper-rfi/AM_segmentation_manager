"""Launching the curation window.

Kept separate from the window itself so that importing the widgets does not
create a Qt application, and so the CLI has one obvious entry point.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

from ..config import load_segmentation_config
from .cache import FrameCache
from .controller import CurationController

logger = logging.getLogger(__name__)

QT_BINDINGS = ("PyQt5", "PyQt6", "PySide2", "PySide6")


def _missing_qt_message(error: Exception) -> str:
    """Explain which part of the Qt stack is missing, and what to do about it.

    ``qtpy`` is only an adapter: it selects whichever binding is installed and
    fails if there is none. Those two situations need opposite advice, so they
    are distinguished here rather than guessed at.
    """
    if importlib.util.find_spec("qtpy") is None:
        return (
            "The curation window needs qtpy and a Qt binding, and neither is "
            "installed.\n"
            "  conda install -c conda-forge pyside6 qtpy\n"
            "  or: pip install 'am_segmentation_manager[gui]'"
        )

    installed = [name for name in QT_BINDINGS if importlib.util.find_spec(name) is not None]
    if not installed:
        return (
            "qtpy is installed, but there is no Qt binding for it to use. qtpy "
            "is only an adapter; it needs PySide6 (or PyQt5) underneath.\n"
            "  conda install -c conda-forge pyside6\n"
            "  or: pip install PySide6"
        )

    return (
        f"A Qt binding is installed ({', '.join(installed)}) but could not be "
        f"loaded: {error}\n"
        "If more than one binding is present, remove all but one -- loading two "
        "into the same process is a common cause of this.\n"
        "You can also select one explicitly by setting the QT_API environment "
        f"variable to one of: {', '.join(name.lower() for name in installed)}"
    )


def _enable_high_dpi() -> None:
    """Ask Qt 5 to scale for the display it is on.

    Both attributes have to be set before a QApplication exists, which is why
    this is separate from applying the theme. Qt 6 scales by default and dropped
    them, and setting one on an application that already exists warns and does
    nothing -- so both cases are skipped rather than guessed at.
    """
    from qtpy import QT_VERSION
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QApplication

    if not QT_VERSION.startswith("5") or QApplication.instance() is not None:
        return
    for name in ("AA_EnableHighDpiScaling", "AA_UseHighDpiPixmaps"):
        attribute = getattr(Qt, name, None)
        if attribute is not None:
            QApplication.setAttribute(attribute, True)


def run(start_path: Path | None = None, config_path: Path | None = None) -> int:
    """Open the curation window and run until it is closed.

    Args:
        start_path: Directory to scan on startup. The window opens without
            scanning if this is None.
        config_path: Segmentation config to use, overriding the usual search.

    Returns:
        The Qt application's exit code.

    Raises:
        RuntimeError: If no Qt binding is installed.
    """
    try:
        from qtpy.QtWidgets import QApplication
    except Exception as exc:  # noqa: BLE001 - reported as a clear instruction
        raise RuntimeError(_missing_qt_message(exc)) from exc

    from .widgets.main_window import CurationWindow
    from .widgets.theme import apply_theme

    config = load_segmentation_config(explicit_path=config_path, start_path=start_path)
    logger.info("Using segmentation config: %s", config.source)

    controller = CurationController(cache=FrameCache(), config=config)

    _enable_high_dpi()
    application = QApplication.instance() or QApplication(sys.argv)
    apply_theme(application)
    window = CurationWindow(controller)
    window.show()

    if start_path is not None:
        window.project_panel.set_path(str(start_path))
        window.scan(str(start_path))

    return application.exec_() if hasattr(application, "exec_") else application.exec()
