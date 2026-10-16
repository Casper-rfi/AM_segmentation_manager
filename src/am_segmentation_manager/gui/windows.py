"""Giving the process the application's identity on Windows, through ctypes.

The taskbar groups windows, and finds what a pin starts, by the process's
AppUserModelID. Without one it is derived from the executable, so the window is
taken for ``pythonw.exe`` and a pin made from it starts Python under Python's
icon. Setting the ID before any window exists, the same one the Start-menu
shortcut carries, ties the window to that shortcut instead. Off Windows, or
where the ID cannot be set, nothing happens.
"""
from __future__ import annotations

import ctypes
import logging
import sys

logger = logging.getLogger(__name__)


def set_app_user_model_id(identifier: str) -> bool:
    """Give this process *identifier* as its AppUserModelID, returning whether it could.

    Takes effect only before the process shows a window.
    """
    if sys.platform != "win32":
        return False
    try:
        shell32 = ctypes.windll.shell32
        shell32.SetCurrentProcessExplicitAppUserModelID.argtypes = [ctypes.c_wchar_p]
        shell32.SetCurrentProcessExplicitAppUserModelID.restype = ctypes.c_long
        result = shell32.SetCurrentProcessExplicitAppUserModelID(identifier)
    except (OSError, AttributeError):
        logger.debug("Could not reach shell32", exc_info=True)
        return False
    if result != 0:
        logger.debug("Windows refused the AppUserModelID %s: HRESULT %#x", identifier, result)
        return False
    return True
