"""Curation user interface.

The subpackage is split so that only the widget layer depends on a GUI toolkit:

* :mod:`am_segmentation_manager.gui.cache` -- holds prepared display data for
  the lamella being reviewed, and decides what to load next.
* :mod:`am_segmentation_manager.gui.controller` -- the view model: what is
  currently selected, what should be shown, and what a state change applies to.

Both are plain Python and can be tested without a display. Importing this
package does not import any GUI toolkit.
"""
from __future__ import annotations

__all__ = ["cache", "controller"]
