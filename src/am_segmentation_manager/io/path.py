"""Writing a path relative to a base, and checking one that came out of a file.

Two rules the package has to apply the same way everywhere: how a path is
written relative to a base, for CSVs and command-line output, and what a
relative key read out of a hand-editable file may look like. Both live here
rather than beside their first caller, because more than one caller has to
agree about them.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["checked_relative_key", "make_path_relative"]


def make_path_relative(path: Path, base_path: Path) -> str:
    """Return a string path for *path* relative to *base_path*.

    Attempts ``Path.resolve().relative_to(base_path.resolve())``. If *path*
    is not a subpath of *base_path*, falls back to ``os.path.relpath``.
    """
    try:
        return str(path.resolve().relative_to(base_path.resolve()))
    except (ValueError, RuntimeError):
        return os.path.relpath(path, base_path)


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
