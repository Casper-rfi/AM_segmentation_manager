"""Writing a path relative to a base, and checking one that came out of a file.

Two rules the package has to apply the same way everywhere: how a path is
written relative to a base, for CSVs and command-line output, and what a
relative key read out of a hand-editable file may look like. Both live here
rather than beside their first caller, because more than one caller has to
agree about them. **A path written here separates directories with '/' on every
platform**, so a file written on Windows names the same files on Linux.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "NoRelativePath",
    "checked_relative_key",
    "exported_relative_path",
    "make_path_relative",
    "slashed",
]


def slashed(typed: str) -> str:
    """Return a relative path someone typed with '\\' read as '/', as Windows writes it.

    For matching what a person typed against a path this module wrote; never
    for a key read out of a file, which :func:`checked_relative_key` refuses.
    """
    return typed.replace("\\", "/")


class NoRelativePath(ValueError):
    """A path an export must write relative to a folder has no such path."""


def exported_relative_path(path: Path, base_path: Path) -> str:
    """Return *path* relative to *base_path*, for a file another machine reads.

    Outside *base_path* the path climbs with ``..``.

    Raises:
        NoRelativePath: Where no relative path exists, as between two drives on
            Windows. An absolute path in its place would name a file on this
            machine only.
    """
    try:
        return path.resolve().relative_to(base_path.resolve()).as_posix()
    except (ValueError, RuntimeError):
        pass
    try:
        return Path(os.path.relpath(path, base_path)).as_posix()
    except ValueError as error:
        raise NoRelativePath(
            f"{path} has no path relative to {base_path}, which every path in "
            f"this export is written from ({error}). Start from a folder on the "
            f"same drive as the data."
        ) from error


def make_path_relative(path: Path, base_path: Path) -> str:
    """Return *path* relative to *base_path*, for showing.

    As :func:`exported_relative_path`, except that where no relative path exists
    the absolute path is returned: a report naming a file somehow is better than
    one that cannot be shown.
    """
    try:
        return exported_relative_path(path, base_path)
    except NoRelativePath:
        return Path(path).resolve().as_posix()


def checked_relative_key(key: str, base: Path) -> str:
    """Return *key* if it names a path inside *base*, else raise.

    The rule for a relative path read out of a file somebody can edit: it is
    checked rather than joined, because ``../`` or an absolute path would
    otherwise name a file in another dataset. Shared by the state file and the
    history log, which must not drift apart about what a key is.

    Raises:
        ValueError: If *key* is empty, padded, absolute, separated with
            backslashes, or steps outside *base*.
    """
    if not key or key != key.strip():
        raise ValueError(f"{key!r} is not a usable path")
    if "\\" in key:
        raise ValueError(f"{key!r} must separate directories with '/'")
    if any(part in ("", ".", "..") for part in key.split("/")):
        raise ValueError(f"{key!r} must be a relative path inside {base}")
    if Path(key).is_absolute():
        raise ValueError(f"{key!r} must be relative to {base}")
    return key
