"""Qt widgets for the curation window.

This is the only part of the package that imports a GUI toolkit. Widgets render
what the controller reports and forward user actions back to it; they hold no
knowledge of the filesystem or of the workflow rules.
"""
from __future__ import annotations
