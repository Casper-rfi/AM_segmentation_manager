"""Qt widgets for the curation window.

The only part of the package that imports a GUI toolkit. Widgets render what a
controller reports and forward what the user did; they know nothing of the
filesystem or of the workflow rules. A tooltip says what a control does in under
200 characters, a ceiling a test holds across the whole window, and every control
in the Export tab's Filters box says what it narrows.
"""
from __future__ import annotations
