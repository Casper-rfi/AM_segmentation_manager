"""Whether a path past Windows' old length limit is reachable here, and refusing writes that are not.

Windows reaches a path of 260 characters or more only where its ``LongPathsEnabled``
setting is on **and** the program declares itself long-path aware, so the answer is
found by trying, once per process, and the setting is read only to explain a no.
Everywhere but Windows every path is reachable. **A batch asks before it writes
anything** (:func:`refuse_unreachable`), so a path too long fails the whole batch
rather than the file it reached partway through.
"""
from __future__ import annotations

import functools
import logging
import os
import shutil
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "PathTooLong",
    "WINDOWS_MAX_PATH",
    "explanation",
    "long_paths_work",
    "refuse_unreachable",
    "unreachable",
]

#: The characters, terminator included, a Windows path may run to without long paths.
WINDOWS_MAX_PATH = 260

#: What the name of a temporary a write goes through beside its destination
#: starts with: hidden, so one a hard kill leaves behind is in no listing.
TEMPORARY_PREFIX = ".am-seg-"

#: That name: the prefix, eight random characters, then the destination's suffix.
TEMPORARY_STEM_LENGTH = len(TEMPORARY_PREFIX) + 8

#: Where the setting lives, read only to explain why long paths do not work.
REGISTRY_KEY = r"SYSTEM\CurrentControlSet\Control\FileSystem"
REGISTRY_VALUE = "LongPathsEnabled"

#: What a person is told to do first: a shorter folder needs no administrator,
#: and leaves files every other program can still open, copy and delete.
SHORTER_FOLDER = r"move the collection to a shorter folder, such as D:\data"
HOW_TO_FIX = r"Move the collection to a shorter folder, such as D:\data, as the Troubleshooting guide describes"


class PathTooLong(OSError):
    """A write would reach a path this Windows process cannot."""


@functools.cache
def long_paths_work() -> bool:
    """Return whether this process can write and read a path past the limit.

    True wherever the limit does not apply. On Windows, a file is written and
    read back at a 300-character path in a folder made for it in the temporary
    directory, and that folder removed.
    """
    if sys.platform != "win32":
        return True
    probe = Path(tempfile.mkdtemp(prefix="am-seg-long-path-"))
    try:
        deep = probe
        while len(str(deep)) < 300:
            deep = deep / ("d" * 40)
        deep.mkdir(parents=True)
        target = deep / "probe.txt"
        target.write_text("long", encoding="utf-8")
        return target.read_text(encoding="utf-8") == "long"
    except OSError as error:
        logger.info("A path past %d characters cannot be reached here: %s", WINDOWS_MAX_PATH, error)
        return False
    finally:
        shutil.rmtree(probe, ignore_errors=True)


def explanation() -> str:
    """Return what to do where long paths do not work in this process, for a person to act on.

    Every answer offers the shorter folder; Windows' long-path setting is named
    only where it is already on.
    """
    if sys.platform != "win32":
        return HOW_TO_FIX
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, REGISTRY_KEY) as key:
            enabled, _kind = winreg.QueryValueEx(key, REGISTRY_VALUE)
    except OSError:
        enabled = 0
    if enabled:
        return (
            f"Windows' long-path setting is on, but AM-seg cannot use it: restart "
            f"AM-seg, then Windows if that is not enough, or {SHORTER_FOLDER}"
        )
    return HOW_TO_FIX


def written_length(path: Path) -> int:
    """Return the longest path a write to *path* passes through: itself, or its temporary."""
    absolute = os.path.abspath(path)
    parent = os.path.dirname(absolute)
    temporary = len(parent) + 1 + TEMPORARY_STEM_LENGTH + len(Path(absolute).suffix)
    return max(len(absolute), temporary)


def unreachable(paths: Iterable[Path], works: bool | None = None) -> list[Path]:
    """Return those of *paths* a write cannot reach here, the longest first.

    Args:
        works: Whether long paths work; :func:`long_paths_work` by default.
    """
    if long_paths_work() if works is None else works:
        return []
    return sorted(
        (Path(path) for path in paths if written_length(Path(path)) >= WINDOWS_MAX_PATH),
        key=written_length, reverse=True,
    )


def refuse_unreachable(paths: Iterable[Path], works: bool | None = None) -> None:
    """Raise before anything is written if a write to any of *paths* cannot be reached.

    Raises:
        PathTooLong: Naming how many, the longest, and what to do.
    """
    too_long = unreachable(paths, works)
    if not too_long:
        return
    longest = too_long[0]
    raise PathTooLong(
        f"Refusing to write: {len(too_long)} path(s) would be {WINDOWS_MAX_PATH} "
        f"characters or longer, which Windows cannot reach here, the longest "
        f"{written_length(longest)} ({longest}). {explanation()}."
    )
