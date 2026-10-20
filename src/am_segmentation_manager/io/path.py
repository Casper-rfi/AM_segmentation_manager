"""Writing a path relative to a base, and checking one that came out of a file.

Two rules the package has to apply the same way everywhere: how a path is
written relative to a base, for CSVs and command-line output, and what a
relative key read out of a hand-editable file may look like. **A path written
here separates directories with '/' on every platform**, so a file written on
Windows names the same files on Linux. Paths are resolved through
:func:`resolved`, which inside a :func:`resolution_cache` block asks about each
directory once rather than once per file in it.
"""
from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from .directories import listed_as_link

__all__ = [
    "NoRelativePath",
    "checked_relative_key",
    "exported_relative_path",
    "make_path_relative",
    "resolution_cache",
    "resolved",
    "slashed",
]

#: What each path asked about in the current :func:`resolution_cache` block
#: resolved to, or None outside one.
_resolutions: ContextVar[dict[Path, Path] | None] = ContextVar("_resolutions", default=None)


@contextmanager
def resolution_cache() -> Iterator[None]:
    """Resolve each directory at most once for the length of this block, or call.

    For writing many paths that share directories: resolving one asks about
    every directory from the root down, each a round trip on a network
    filesystem. Held for one export the answers cannot go stale in any way that
    matters; held longer they could. Nesting reuses the outer block, and it
    serves as a decorator too.
    """
    if _resolutions.get() is not None:
        yield
        return
    token = _resolutions.set({})
    try:
        yield
    finally:
        _resolutions.reset(token)


def resolved(path: Path) -> Path:
    """Return what ``path.resolve()`` returns, asking about each directory once a block.

    Inside a :func:`resolution_cache` block a path is its resolved directory
    joined to its name, unless the name is a link: so each directory on the way
    down is resolved once, and asked whether it is a link once -- or not at
    all where a :func:`~.directories.directory_cache` listing of its parent is
    held. A link, a name of ``..``, a relative path, and Windows, whose
    resolution also corrects the case of a name, are resolved whole.
    """
    path = Path(path)
    cache = _resolutions.get()
    if cache is None or os.name == "nt" or not path.is_absolute():
        return path.resolve()
    return _resolved_in(cache, path)


def _resolved_in(cache: dict[Path, Path], path: Path) -> Path:
    """Return *path* resolved, from and into *cache*."""
    known = cache.get(path)
    if known is not None:
        return known
    if path.parent == path or path.name in ("", ".", "..") or _resolved_whole(path):
        answer = path.resolve()
    else:
        answer = _resolved_in(cache, path.parent) / path.name
    cache[path] = answer
    return answer


def _resolved_whole(path: Path) -> bool:
    """Return whether *path* must be resolved whole: a link, or something lstat cannot say."""
    listed = listed_as_link(path)
    if listed is not None:
        return listed
    try:
        return stat.S_ISLNK(os.lstat(path).st_mode)
    except FileNotFoundError:
        return False
    except OSError:
        return True

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
        return resolved(path).relative_to(resolved(base_path)).as_posix()
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
        return resolved(Path(path)).as_posix()


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
