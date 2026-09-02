"""Launching the curation window, and reporting anything it fails to handle.

Kept separate from the window itself so that importing the widgets does not
create a Qt application, and so the CLI has one obvious entry point.

**No exception raised in the window may reach CPython's own traceback
printer**, which has crashed the process outright on macOS. An exception
escaping a Qt slot propagates nowhere this package can catch it, so
:func:`install_error_reporting` replaces the interpreter's reporter with one
written in Python. See `docs/display.md`.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import threading
import traceback
from pathlib import Path

from ..config import load_segmentation_config
from .cache import FrameCache
from .controller import CurationController

logger = logging.getLogger(__name__)

QT_BINDINGS = ("PyQt5", "PyQt6", "PySide2", "PySide6")


#: Whether a failure is already on screen. A slot that raises once usually
#: raises again on the next repaint, and a dialog per repaint buries the window
#: it is describing. Every failure is logged; this only limits how many are
#: shown at a time.
_reporting = False

#: What a reported failure says above the traceback. The traceback itself is
#: behind "Show Details", because the first thing the reviewer needs is whether
#: their work is safe, not which line raised.
FAILURE_TITLE = "Something went wrong"
FAILURE_ADVICE = (
    "The window is still open, but what is on screen may no longer describe "
    "the collection. Scan the folder again before changing anything.\n\n"
    "The details below are also in the log file."
)


def install_error_reporting() -> None:
    """Report unhandled exceptions through the log and a dialog.

    Replaces :data:`sys.excepthook` and :func:`threading.excepthook`, which is
    what Qt reaches when an exception escapes a slot. Both formats are built
    with :func:`traceback.format_exception` rather than by the interpreter, for
    the reason in the module docstring.

    Called by :func:`run`; a command that is not the window keeps Python's own
    behaviour, a traceback on stderr and a non-zero exit being right there.
    """
    sys.excepthook = _handle_exception
    threading.excepthook = _handle_thread_exception


def _handle_exception(exc_type, exc, tb) -> None:
    """Log an unhandled exception, and offer it to the reviewer."""
    detail = "".join(traceback.format_exception(exc_type, exc, tb))
    logger.error("Unhandled exception in the window\n%s", detail)
    _offer(f"{exc_type.__name__}: {exc}", detail)


def _handle_thread_exception(args) -> None:
    """As :func:`_handle_exception`, naming the thread it came off.

    Reached only by threads Python started. Qt's own worker threads are foreign
    to it and report through :data:`sys.excepthook` like anything else, so both
    hooks are needed to cover the window.
    """
    if args.exc_type is SystemExit:
        return
    detail = "".join(
        traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
    )
    logger.error("Unhandled exception on thread %s\n%s", args.thread, detail)
    _offer(f"{args.exc_type.__name__}: {args.exc_value}", detail)


def _offer(summary: str, detail: str) -> None:
    """Put a failure on screen, unless there is nowhere to put it.

    **Queued rather than shown here.** A hook runs inside whatever Qt was doing
    when the exception was raised -- often part-way through delivering a mouse
    press -- and a modal dialog re-enters the event loop at exactly that point.
    Deferring to the next turn of the loop lets Qt finish the event it is on
    before anything else is asked of it.

    Does nothing off the thread that owns the widgets, where a dialog cannot be
    built at all. The log has it either way.
    """
    global _reporting

    from qtpy.QtCore import QThread, QTimer
    from qtpy.QtWidgets import QApplication

    application = QApplication.instance()
    if application is None or QThread.currentThread() is not application.thread():
        return
    if _reporting:
        return
    _reporting = True
    QTimer.singleShot(0, lambda: _show(summary, detail))


def _show(summary: str, detail: str) -> None:
    """Show one failure and let the next one be reported."""
    global _reporting

    from qtpy.QtWidgets import QApplication, QMessageBox

    try:
        box = QMessageBox(QApplication.activeWindow())
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle(FAILURE_TITLE)
        box.setText(summary)
        box.setInformativeText(FAILURE_ADVICE)
        box.setDetailedText(detail)
        box.setStandardButtons(QMessageBox.Close)
        box.exec_() if hasattr(box, "exec_") else box.exec()
    except Exception:  # noqa: BLE001 - a failure reporting a failure
        logger.exception("Could not show the failure dialog")
    finally:
        _reporting = False


def _apply_icon(application) -> None:
    """Set the application icon, if it is there to set.

    Applied to the application rather than the window so that dialogs and the
    taskbar entry inherit it. A missing file is not an error: a source checkout
    without the asset must still open the window.
    """
    from qtpy.QtGui import QIcon

    from .widgets.branding import ICON_PATH

    if not ICON_PATH.exists():
        logger.debug("No window icon at %s", ICON_PATH)
        return
    application.setWindowIcon(QIcon(str(ICON_PATH)))


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
        from qtpy.QtCore import QTimer
        from qtpy.QtWidgets import QApplication
    except Exception as exc:  # noqa: BLE001 - reported as a clear instruction
        raise RuntimeError(_missing_qt_message(exc)) from exc

    from .widgets.main_window import CurationWindow
    from .widgets.theme import apply_theme

    # Before anything is built, so a failure while the window is being
    # assembled is reported the same way as one after it is on screen.
    install_error_reporting()

    config = load_segmentation_config(explicit_path=config_path, start_path=start_path)
    logger.info("Using segmentation config: %s", config.source)

    controller = CurationController(cache=FrameCache(), config=config)

    _enable_high_dpi()
    application = QApplication.instance() or QApplication(sys.argv)
    apply_theme(application)
    _apply_icon(application)
    window = CurationWindow(controller)
    window.show()
    # Queued for the same reason the scan below is: show() only asks for a
    # window, and asking for the keyboard before there is one to give it to
    # does nothing. Launched from a terminal, the terminal stays in front
    # otherwise -- see CurationWindow.bring_to_front for what that costs.
    QTimer.singleShot(0, window.bring_to_front)

    if start_path is not None:
        window.project_panel.set_path(str(start_path))
        # Held until the event loop is running. show() only asks for a paint;
        # nothing is drawn until exec() below starts delivering events, so a
        # scan started here ran against a window that had never appeared --
        # seconds of blank grey with no way to tell it from a hang. Queued at
        # zero delay, the window is on screen before the first file is read.
        QTimer.singleShot(0, lambda: window.scan(str(start_path)))

    return application.exec_() if hasattr(application, "exec_") else application.exec()
