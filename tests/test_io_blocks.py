"""Tests for the io/blocks.py module: reading only the blocks of a file asked for.

``O_DIRECT`` is absent on macOS and Windows and refused by tmpfs, where pytest's
temporary directories often live on Linux, so the direct path is exercised with
``O_NONBLOCK`` standing in for it: harmless on a regular file, and it leaves the
aligned ``preadv`` reads the direct path makes as they are.
"""
from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from am_segmentation_manager.io import blocks
from am_segmentation_manager.io.blocks import BLOCK_SIZE, BlockReader

needs_preadv = pytest.mark.skipif(
    not hasattr(os, "preadv"), reason="no preadv on this platform"
)


@pytest.fixture
def data(tmp_path: Path) -> tuple[Path, bytes]:
    """A file of two and a half blocks of distinct bytes."""
    content = os.urandom(BLOCK_SIZE * 5 // 2)
    path = tmp_path / "file.bin"
    path.write_bytes(content)
    return path, content


@pytest.fixture(params=["ordinary", "direct"])
def mode(request, monkeypatch) -> str:
    """Each test once through ordinary reads and once through the direct path."""
    if request.param == "direct":
        if not hasattr(os, "preadv"):
            pytest.skip("no preadv on this platform")
        monkeypatch.setattr(blocks, "_DIRECT", os.O_NONBLOCK)
    else:
        monkeypatch.setattr(blocks, "_DIRECT", 0)
    return request.param


@pytest.mark.parametrize(
    "start, length",
    [(0, 8), (10, 100), (BLOCK_SIZE - 4, 8), (BLOCK_SIZE * 2 + 5, 10**6), (0, 10**6)],
    ids=["start", "inside-a-block", "across-a-boundary", "past-the-end", "whole-file"],
)
def test_reads_what_an_ordinary_read_reads(data, mode, start: int, length: int):
    path, content = data
    with BlockReader(path) as reader:
        reader.seek(start)
        assert reader.read(length) == content[start:start + length]
        assert reader.tell() == min(start + length, len(content))


def test_a_read_fetches_only_the_blocks_it_spans(data, mode):
    """What makes a header cost one block rather than the whole file."""
    path, _ = data
    with BlockReader(path) as reader:
        reader.read(100)
        assert reader.fetched == 1
        reader.seek(BLOCK_SIZE - 10)
        reader.read(20)
        assert reader.fetched == 2
        reader.seek(0)
        reader.read(BLOCK_SIZE + 1)
        assert reader.fetched == 2


@needs_preadv
@pytest.mark.parametrize("refusal", ["open", "read", "short-read"])
def test_a_filesystem_refusing_direct_reads_is_read_the_ordinary_way(
    data, monkeypatch, refusal: str
):
    """tmpfs refuses the open, some filesystems only the read, and a short read
    would leave the next block unaligned."""
    path, content = data
    monkeypatch.setattr(blocks, "_DIRECT", os.O_NONBLOCK)
    real_open, real_preadv = os.open, os.preadv

    def refusing_open(file, flags, *args, **kwargs):
        if refusal == "open" and flags & os.O_NONBLOCK:
            raise OSError(errno.EINVAL, "direct I/O refused")
        return real_open(file, flags, *args, **kwargs)

    def refusing_preadv(fd, buffers, offset, *args):
        if refusal == "read":
            raise OSError(errno.EINVAL, "direct I/O refused")
        return real_preadv(fd, buffers, offset, *args) // 2

    monkeypatch.setattr(blocks.os, "open", refusing_open)
    monkeypatch.setattr(blocks.os, "preadv", refusing_preadv)

    with BlockReader(path) as reader:
        assert reader.read() == content


def test_a_file_that_cannot_be_opened_raises_rather_than_falling_back(tmp_path: Path, mode):
    with pytest.raises(FileNotFoundError):
        BlockReader(tmp_path / "missing.bin")

