"""Giving the process the application's name on macOS, through ctypes.

The menu bar names a process by its main bundle's ``CFBundleName``. Run from a
virtual environment the main bundle is the interpreter's directory, which has
none, so the menu bar reads ``python`` however the window was started. Setting
the name before the QApplication exists is what AppKit then reads. Off macOS,
or where the name cannot be changed, nothing happens.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import logging
import sys

logger = logging.getLogger(__name__)

_NAME_KEY = b"CFBundleName"


def _library(name: str) -> str:
    path = ctypes.util.find_library(name)
    if path is None:
        raise OSError(f"No {name} library")
    return path


class _Runtime:
    """The few Objective-C messages needed, each typed exactly.

    ``objc_msgSend`` is not variadic on Apple silicon: a call must be made
    through a prototype matching the method's own arguments, so each shape of
    message gets its own.
    """

    def __init__(self) -> None:
        objc = ctypes.cdll.LoadLibrary(_library("objc"))
        ctypes.cdll.LoadLibrary(_library("Foundation"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        self._objc = objc
        pointer = ctypes.c_void_p
        send = ("objc_msgSend", objc)
        self.send = ctypes.CFUNCTYPE(pointer, pointer, pointer)(send)
        self.send_object = ctypes.CFUNCTYPE(pointer, pointer, pointer, pointer)(send)
        self.send_text = ctypes.CFUNCTYPE(pointer, pointer, pointer, ctypes.c_char_p)(send)
        self.send_pair = ctypes.CFUNCTYPE(None, pointer, pointer, pointer, pointer)(send)
        self.send_flag = ctypes.CFUNCTYPE(ctypes.c_bool, pointer, pointer, pointer)(send)
        self.utf8 = ctypes.CFUNCTYPE(ctypes.c_char_p, pointer, pointer)(send)

    def cls(self, name: bytes) -> int:
        return self._objc.objc_getClass(name)

    def sel(self, name: bytes) -> int:
        return self._objc.sel_registerName(name)

    def string(self, text: bytes) -> int:
        return self.send_text(self.cls(b"NSString"), self.sel(b"stringWithUTF8String:"), text)

    def info_dictionary(self) -> int | None:
        bundle = self.send(self.cls(b"NSBundle"), self.sel(b"mainBundle"))
        return self.send(bundle, self.sel(b"infoDictionary")) if bundle else None


def bundle_name() -> str | None:
    """Return the name the menu bar gives this process, where it has been set."""
    if sys.platform != "darwin":
        return None
    runtime = _Runtime()
    info = runtime.info_dictionary()
    if not info:
        return None
    value = runtime.send_object(info, runtime.sel(b"objectForKey:"), runtime.string(_NAME_KEY))
    if not value:
        return None
    text = runtime.utf8(value, runtime.sel(b"UTF8String"))
    return text.decode("utf-8") if text else None


def set_bundle_name(name: str) -> bool:
    """Name this process *name* in the menu bar, returning whether it could.

    Takes effect only before the QApplication is constructed. A dictionary that
    cannot be changed is left alone: sending it a change would raise an
    Objective-C exception, which ends the process.
    """
    if sys.platform != "darwin":
        return False
    try:
        runtime = _Runtime()
        info = runtime.info_dictionary()
        setter = runtime.sel(b"setObject:forKey:")
        if not info or not runtime.send_flag(info, runtime.sel(b"respondsToSelector:"), setter):
            logger.debug("The main bundle's information cannot be changed; keeping its name")
            return False
        runtime.send_pair(info, setter, runtime.string(name.encode("utf-8")),
                          runtime.string(_NAME_KEY))
    except (OSError, AttributeError):
        logger.debug("Could not reach the Objective-C runtime", exc_info=True)
        return False
    return True
