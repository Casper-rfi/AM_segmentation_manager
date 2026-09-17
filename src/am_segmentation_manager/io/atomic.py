"""Writing a file whole, and telling whether one changed underneath you.

A write goes to a temporary file in the destination's own directory and is
renamed into place, so an interrupted run leaves the previous file rather than
half of a new one. An append holds an exclusive lock for its one write, so no
other process's append splits or overwrites it. A fingerprint taken before a
write is what lets a caller refuse a destination something else has rewritten.

Nothing here knows what it is writing.
"""
from __future__ import annotations

import errno
import logging
import os
import stat
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

from .long_paths import refuse_unreachable

logger = logging.getLogger(__name__)

__all__ = [
    "Fingerprint",
    "append_whole",
    "file_mode_for",
    "fingerprint_of",
    "write_atomically",
]

#: What a filesystem without locks answers a request for one, on macOS and Linux.
_LOCKS_UNSUPPORTED = frozenset({errno.ENOLCK, errno.ENOTSUP, errno.EOPNOTSUPP})


def write_atomically(
    destination: Path,
    write: Callable[[Path], object],
    *,
    commit_if: Callable[[Path], bool] | None = None,
) -> bool:
    """Have *write* produce a temporary file, then move it onto *destination*.

    The rename is what makes a write whole or absent: an interrupted run leaves
    the file that was already there untouched rather than half-replaced. The
    temporary is created in the destination's own directory, because a rename
    across filesystems is not atomic.

    The temporary is created owner-only, as :func:`tempfile.mkstemp` makes it,
    so its permissions are set before the rename rather than carried onto the
    destination -- see :func:`file_mode_for`.

    Args:
        destination: Where the file should end up. Its directory is created if
            it does not exist.
        write: Called with the temporary path to write.
        commit_if: Asked, with the written temporary, whether to go ahead. A
            caller that can only tell once the file exists -- whether the new
            encoding is actually smaller, whether the destination changed while
            it was being written -- decides here rather than beforehand, which
            is what keeps the decision inside the window the rename closes.
            Returning False leaves *destination* as it was.

    Returns:
        Whether *destination* was replaced.

    Raises:
        PathTooLong: If Windows cannot reach *destination* or its temporary here.
    """
    refuse_unreachable([destination])
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = file_mode_for(destination)
    # The suffix is kept: the encoder OpenCV selects is chosen by extension.
    handle, temporary_name = tempfile.mkstemp(
        dir=destination.parent, suffix=destination.suffix
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        write(temporary)
        # After the write: a writer that recreates the file rather than
        # truncating it brings its own permissions with it.
        os.chmod(temporary, mode)
        if commit_if is not None and not commit_if(temporary):
            return False
        os.replace(temporary, destination)
        return True
    finally:
        if temporary.exists():
            temporary.unlink()


def append_whole(path: Path, data: bytes) -> None:
    """Append *data* to *path* so that no other process's append splits or overwrites it.

    macOS and Linux guarantee that of an append; Windows emulates one as a seek
    to the end and a write, so two processes can land at one offset and the
    second replaces the first. An exclusive lock held for the write closes that
    everywhere. Where a filesystem offers no locks -- some network shares --
    macOS and Linux append without one, their own guarantee still holding.

    Raises:
        OSError: If *path* cannot be opened, locked or written. On Windows a
            lock another process holds is waited for about ten seconds first.
    """
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o666)
    try:
        with _locked(descriptor):
            remaining = memoryview(data)
            while remaining:
                remaining = remaining[os.write(descriptor, remaining):]
    finally:
        os.close(descriptor)


@contextmanager
def _locked(descriptor: int) -> Iterator[None]:
    """Hold an exclusive lock on the open file *descriptor*, released however the body ends."""
    if sys.platform == "win32":
        import msvcrt

        # The lock covers the first byte, which every writer asks for, whatever
        # the file's length; an append then writes at the end regardless.
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except OSError as error:
        if error.errno not in _LOCKS_UNSUPPORTED:
            raise
        yield
        return
    try:
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)


def file_mode_for(destination: Path) -> int:
    """Return the permissions a file written to *destination* should carry.

    An existing file keeps its own, because replacing it is not a decision
    about who may read it. A new one gets what an ordinary write would have
    produced -- ``0o666`` less the process umask -- rather than the owner-only
    mode a temporary file is created with, since a file written into a shared
    archive that only its author can read is a file nobody else can use.
    """
    try:
        return stat.S_IMODE(destination.stat().st_mode)
    except OSError:
        return 0o666 & ~_current_umask()


def _current_umask() -> int:
    """Return the process umask, which can only be read by setting it."""
    mask = os.umask(0o022)
    os.umask(mask)
    return mask


class Fingerprint(NamedTuple):
    """What a file looked like at one moment, compared whole to tell whether it changed."""

    size: int
    modified_ns: int
    #: The inode, or the file ID on Windows, which a file replaced by a rename
    #: does not keep.
    file_id: int


def fingerprint_of(path: Path) -> Fingerprint | None:
    """Return a cheap identity for *path*'s contents, or None if it is absent.

    Enough to tell that something else rewrote a file between a read and a
    write: a replacement by rename, as every write here is, changes the file ID
    whatever the clock says. **One rewrite goes unseen**: in place, at the same
    size, within one step of the modification time's clock -- about 15 ms on
    Windows, 2 s on exFAT -- of the write before it. Absent is itself a
    fingerprint: a file that did not exist and now does has changed.
    """
    try:
        status = path.stat()
    except OSError:
        return None
    return Fingerprint(status.st_size, status.st_mtime_ns, status.st_ino)
