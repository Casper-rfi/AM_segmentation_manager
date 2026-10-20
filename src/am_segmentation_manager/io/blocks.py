"""Reading a few blocks of a large file without fetching the rest of it.

A network filesystem can fetch a whole stored object to answer a small read through
the page cache -- on CephFS, megabytes for the kilobytes of a TIFF header. So where
the platform offers ``O_DIRECT`` a file is read around that cache, and only the
blocks asked for are fetched. Without ``O_DIRECT``, or where a filesystem refuses it,
the same blocks are read the ordinary way, so **what is read never depends on how**.
Nothing here knows what a file holds.
"""
from __future__ import annotations

import errno
import io
import mmap
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from _typeshed import WriteableBuffer

#: Bytes fetched at a time: a multiple of every page size ``O_DIRECT`` demands, and
#: enough to hold the header of every image this package has been pointed at.
BLOCK_SIZE = 64 * 1024

#: The flag that bypasses the page cache, or 0 where the platform has none.
_DIRECT = getattr(os, "O_DIRECT", 0)


class BlockReader(io.RawIOBase):
    """A read-only, seekable file that fetches each block it is asked for once.

    Opened at construction and closed as any file is, by :meth:`close` or a
    ``with`` block. It offers no ``fileno``, so a reader handed it cannot read
    the file some other way.

    Raises:
        OSError: If the file cannot be opened; a refusal of ``O_DIRECT`` alone
            is not raised, the file being opened without it.
    """

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.name = str(path)
        self._path = path
        self._position = 0
        self._blocks: dict[int, bytes] = {}
        self._ordinary: io.RawIOBase | None = None
        self._direct: int | None = None
        self._buffer: mmap.mmap | None = None
        #: How many blocks have been fetched from the file.
        self.fetched = 0
        if _DIRECT:
            try:
                self._direct = os.open(path, os.O_RDONLY | _DIRECT)
            except OSError as error:
                if error.errno != errno.EINVAL:
                    raise
        if self._direct is None:
            self._ordinary = open(path, "rb", buffering=0)
        self.size = os.fstat(self._descriptor()).st_size

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        origin = {io.SEEK_SET: 0, io.SEEK_CUR: self._position, io.SEEK_END: self.size}[whence]
        if origin + offset < 0:
            raise ValueError(f"negative seek position {origin + offset}")
        self._position = origin + offset
        return self._position

    def readinto(self, buffer: WriteableBuffer) -> int:
        """Fill *buffer* from the current position, fetching the blocks it spans."""
        view = memoryview(buffer).cast("B")
        end = min(self.size, self._position + len(view))
        written = 0
        while self._position < end:
            index, offset = divmod(self._position, BLOCK_SIZE)
            chunk = self._block(index)[offset:offset + end - self._position]
            if not chunk:
                break
            view[written:written + len(chunk)] = chunk
            written += len(chunk)
            self._position += len(chunk)
        return written

    def close(self) -> None:
        if not self.closed:
            if self._direct is not None:
                os.close(self._direct)
                self._direct = None
            if self._ordinary is not None:
                self._ordinary.close()
            if self._buffer is not None:
                self._buffer.close()
        super().close()

    def _descriptor(self) -> int:
        if self._direct is not None:
            return self._direct
        assert self._ordinary is not None
        return self._ordinary.fileno()

    def _block(self, index: int) -> bytes:
        """Return block *index*, fetching it if this is the first time it is asked for."""
        block = self._blocks.get(index)
        if block is None:
            block = self._fetch(index * BLOCK_SIZE)
            self._blocks[index] = block
            self.fetched += 1
        return block

    def _fetch(self, offset: int) -> bytes:
        """Return the block at *offset*, directly where that is possible."""
        if self._direct is not None:
            block = self._fetch_direct(offset)
            if block is not None:
                return block
            self._read_ordinarily_from_now_on()
        assert self._ordinary is not None
        self._ordinary.seek(offset)
        parts: list[bytes] = []
        remaining = BLOCK_SIZE
        while remaining:
            part = self._ordinary.read(remaining)
            if not part:
                break
            parts.append(part)
            remaining -= len(part)
        return b"".join(parts)

    def _fetch_direct(self, offset: int) -> bytes | None:
        """Return the block at *offset* read around the page cache, or None if refused.

        A short read before the end of the file is a refusal too: the next block
        would start unaligned, which ``O_DIRECT`` does not allow.
        """
        assert self._direct is not None
        if self._buffer is None:
            # An anonymous mapping is page-aligned, as O_DIRECT requires of a buffer.
            self._buffer = mmap.mmap(-1, BLOCK_SIZE)
        try:
            got = os.preadv(self._direct, [self._buffer], offset)
        except OSError as error:
            if error.errno != errno.EINVAL:
                raise
            return None
        if got < BLOCK_SIZE and offset + got < self.size:
            return None
        return self._buffer[:got]

    def _read_ordinarily_from_now_on(self) -> None:
        """Replace the direct descriptor with an ordinary one, for the rest of this file."""
        assert self._direct is not None
        os.close(self._direct)
        self._direct = None
        self._ordinary = open(self._path, "rb", buffering=0)
