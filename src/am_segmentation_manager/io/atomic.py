"""Writing a file whole, and telling whether one changed underneath you.

A write goes to a temporary file in the destination's own directory and is
renamed into place, so an interrupted run leaves the previous file rather than
half of a new one. A fingerprint taken before a write is what lets a caller
refuse a destination something else has rewritten since it was read.

Nothing here knows what it is writing.
"""
from __future__ import annotations

import logging
import os
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["file_mode_for", "fingerprint_of", "write_atomically"]


def write_atomically(destination: Path, write: Callable[[Path], None]) -> None:
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
    """
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
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def fingerprint_of(path: Path) -> tuple[int, int] | None:
    """Return a cheap identity for *path*'s contents, or None if it is absent.

    Size and modification time, which is enough to tell that something else
    rewrote a file between a read and a write. Absent is itself a fingerprint:
    a file that did not exist and now does has changed.
    """
    try:
        status = path.stat()
    except OSError:
        return None
    return (status.st_size, status.st_mtime_ns)
