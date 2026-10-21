"""Listing a directory once per traversal, and what each entry is.

A scan asks about the same directories many times, so :func:`directory_cache`
holds each listing for one block and answers from it whatever it can: on a network
filesystem a question about something absent is a round trip like any other.
**Nothing that writes files may run inside one.** An unreadable directory reads as
empty rather than raising, so one bad directory cannot abort a survey of a whole
collection. Nothing here knows what a directory or a file is called.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Entry:
    """One directory entry, and what the listing already knew it was.

    ``os.scandir`` reports the kind of an entry from the directory read itself,
    where asking a :class:`~pathlib.Path` costs a ``stat`` apiece -- 73,564 of
    them in one scan of the reference collection. Both flags follow symlinks,
    as ``Path.is_dir`` and ``Path.is_file`` do, and both are False for an entry
    that could not be inspected.
    """

    path: Path
    is_dir: bool
    is_file: bool
    #: Whether the entry is itself a symbolic link, answered by the listing too.
    is_link: bool = False


#: The listings held for the length of one :func:`directory_cache` block, or
#: None outside one. A :class:`~contextvars.ContextVar` rather than a module
#: global so that a scan running on a worker thread gets its own.
_listings: ContextVar[dict[Path, list[Entry]] | None] = ContextVar(
    "_listings", default=None
)


@contextmanager
def directory_cache() -> Iterator[None]:
    """List each directory at most once for the length of this block.

    A scan reads the same directory several times -- once to ask whether it
    marks a project, once for its image sets, once for its loose images, once
    for the children to descend into -- and each listing is a syscall per entry
    on top of the read. Held for one scan the answers cannot go stale, because
    a scan is a snapshot of the filesystem either way; held any longer they
    could, which is why this is a block rather than a permanent memo. **Nothing
    that writes files may run inside one.** Nesting reuses the outer block.
    """
    if _listings.get() is not None:
        yield
        return
    token = _listings.set({})
    try:
        yield
    finally:
        _listings.reset(token)


def list_directory(directory: Path) -> list[Entry]:
    """Return the entries in *directory*, treating unreadable ones as empty.

    A directory that does not exist is silently empty: callers routinely ask
    about optional locations such as state subdirectories. Anything else --
    a permissions problem, or a cloud-storage placeholder that has not synced --
    is warned about but still does not raise, so one bad directory cannot abort
    a survey of an entire collection.

    Ordered by :func:`os.path.normcase` of the name, which is how sorting the
    paths themselves ordered them: case-insensitively where the platform's
    paths are. Inside a :func:`directory_cache` block, a directory its parent's
    listing shows to be absent is empty without being asked about.
    """
    cache = _listings.get()
    if cache is not None:
        cached = cache.get(directory)
        if cached is not None:
            return cached
        if _listed_as_absent(directory, cache):
            cache[directory] = []
            return []
    entries = _read_directory(directory)
    if cache is not None:
        cache[directory] = entries
    return entries


def holds_file(directory: Path, name: str) -> bool:
    """Return whether *directory* holds a file called *name*.

    Answered from a listing held by :func:`directory_cache` where there is one,
    and never by listing a directory to answer it. Where the listing holds the
    name only in another case, the filesystem is asked, which alone knows
    whether it tells the two apart.
    """
    cache = _listings.get()
    if cache is not None:
        listed = cache.get(directory)
        if listed is None and _listed_as_absent(directory, cache):
            return False
        if listed is not None:
            if any(entry.is_file and entry.path.name == name for entry in listed):
                return True
            folded = name.casefold()
            if not any(entry.path.name.casefold() == folded for entry in listed):
                return False
    try:
        return (directory / name).is_file()
    except OSError as exc:  # pragma: no cover - transient filesystem error
        logger.warning("Could not check %s in %s: %s", name, directory, exc)
        return False


def listed_as_file(path: Path) -> bool | None:
    """Return whether *path* is a file, from its directory's listing, or None outside a block.

    Inside a :func:`directory_cache` block the directory is listed once, if it
    has not been, and the answer is :func:`holds_file`'s.
    """
    if _listings.get() is None:
        return None
    list_directory(path.parent)
    return holds_file(path.parent, path.name)


def listed_as_link(path: Path) -> bool | None:
    """Return whether a listing held shows *path* to be a symbolic link, or None if none can say.

    A path its directory's listing does not hold is not a link, there being
    nothing there. Where the listing holds the name only in another case, or
    no listing is held, the answer is None: only the filesystem can give it.
    """
    cache = _listings.get()
    if cache is None:
        return None
    listed = cache.get(path.parent)
    if listed is None:
        return False if _listed_as_absent(path.parent, cache) else None
    folded = path.name.casefold()
    in_another_case = False
    for entry in listed:
        if entry.path.name == path.name:
            return entry.is_link
        in_another_case = in_another_case or entry.path.name.casefold() == folded
    return None if in_another_case else False


def missing_files(paths: Iterable[Path]) -> list[Path]:
    """Return those of *paths* that are not files, in order, as ``Path.is_file`` says.

    Each directory named is listed once and a file the listing shows needs no
    question of its own; on a network filesystem that is one round trip a
    directory rather than one a file. What a listing does not show is asked of
    the filesystem, which alone can see into a directory that cannot be listed.
    """
    paths = list(paths)
    with directory_cache():
        for directory in dict.fromkeys(path.parent for path in paths):
            list_directory(directory)
        return [
            path for path in paths
            if not holds_file(path.parent, path.name) and not path.is_file()
        ]


def _listed_as_absent(directory: Path, cache: dict[Path, list[Entry]]) -> bool:
    """Return whether a listing in *cache* shows *directory* is not there.

    Only its parent's listing can, and only by holding no entry of that name in
    any case: one differing in case may be the same directory.
    """
    parent = directory.parent
    if parent == directory:
        return False
    siblings = cache.get(parent)
    if siblings is None:
        return False
    folded = directory.name.casefold()
    return not any(entry.path.name.casefold() == folded for entry in siblings)


def _read_directory(directory: Path) -> list[Entry]:
    """Return *directory*'s entries, unconditionally reading it."""
    try:
        with os.scandir(directory) as scan:
            entries = [
                Entry(
                    Path(entry.path),
                    _kind(entry, "is_dir"),
                    _kind(entry, "is_file"),
                    _kind(entry, "is_symlink"),
                )
                for entry in scan
            ]
    except (FileNotFoundError, NotADirectoryError):
        return []
    except OSError as exc:
        logger.warning("Could not read directory %s: %s", directory, exc)
        return []
    entries.sort(key=lambda entry: os.path.normcase(entry.path.name))
    return entries


def _kind(entry: os.DirEntry, question: str) -> bool:
    """Return what *entry* answers to ``is_dir``, ``is_file`` or ``is_symlink``, errors being no.

    ``os.DirEntry`` swallows the errors that mean "gone" and raises the rest,
    which here is the same warning a whole unreadable directory gets.
    """
    try:
        return bool(getattr(entry, question)())
    except OSError as exc:  # pragma: no cover - transient filesystem error
        logger.warning("Could not inspect %s: %s", entry.path, exc)
        return False
