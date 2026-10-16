"""The AppUserModelID the window gives its process on Windows."""
from __future__ import annotations

import ctypes
import sys

import pytest

from am_segmentation_manager.gui import windows
from am_segmentation_manager.identity import APPLICATION_ID


def test_nothing_is_set_off_windows(monkeypatch):
    monkeypatch.setattr(windows.sys, "platform", "darwin")

    assert windows.set_app_user_model_id(APPLICATION_ID) is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows' own taskbar")
def test_the_process_takes_the_id_on_windows():
    assert windows.set_app_user_model_id(APPLICATION_ID) is True

    shell32 = ctypes.windll.shell32
    shell32.GetCurrentProcessExplicitAppUserModelID.argtypes = [
        ctypes.POINTER(ctypes.c_void_p)
    ]
    shell32.GetCurrentProcessExplicitAppUserModelID.restype = ctypes.c_long
    pointer = ctypes.c_void_p()
    assert shell32.GetCurrentProcessExplicitAppUserModelID(ctypes.byref(pointer)) == 0
    try:
        assert ctypes.wstring_at(pointer) == APPLICATION_ID
    finally:
        ctypes.windll.ole32.CoTaskMemFree(pointer)
