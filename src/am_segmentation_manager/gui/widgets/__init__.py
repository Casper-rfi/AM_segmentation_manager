"""Qt widgets for the curation window.

This is the only part of the package that imports a GUI toolkit. Widgets render
what the controller reports and forward user actions back to it; they hold no
knowledge of the filesystem or of the workflow rules.

A tooltip says what a control does, in under 200 characters. Past that it is
documentation, and a tooltip nobody finishes explains nothing; a test holds the
ceiling across the whole window rather than one tab, because enforced narrowly
the ones out of scope grow. Every control in the Export tab's Filters box also
says what it narrows: that it decides the breakdown *and* the exports is not
visible from the box's title.
"""
from __future__ import annotations
