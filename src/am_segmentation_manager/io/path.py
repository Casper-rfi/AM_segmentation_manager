"""Path manipulation and resolution utilities.

Centralises path resolution and relative path formatting across scanning,
CSV operations, and CLI output.
"""
from __future__ import annotations

import os
from pathlib import Path


def make_path_relative(path: Path, base_path: Path) -> str:
    """Return a string path for *path* relative to *base_path*.

    Attempts ``Path.resolve().relative_to(base_path.resolve())``. If *path*
    is not a subpath of *base_path*, falls back to ``os.path.relpath``.
    """
    try:
        return str(path.resolve().relative_to(base_path.resolve()))
    except (ValueError, RuntimeError):
        return os.path.relpath(path, base_path)
