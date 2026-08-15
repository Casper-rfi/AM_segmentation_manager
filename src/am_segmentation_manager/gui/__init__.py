"""Curation user interface.

The subpackage is split so only the widget layer depends on a GUI toolkit:
:mod:`~am_segmentation_manager.gui.cache` holds prepared display data, and
:mod:`~am_segmentation_manager.gui.controller` is the view model.

Both are plain Python and testable without a display. Importing this package does
not import any GUI toolkit.
"""
from __future__ import annotations

__all__ = ["cache", "controller"]
