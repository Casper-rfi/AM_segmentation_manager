"""Guards the rule that keeps the curation UI testable.

Only the widget layer may depend on a GUI toolkit. If the controller or the
cache ever import Qt, they stop being testable without a display and the split
this design relies on has quietly collapsed -- so it is checked rather than
merely documented.

The imports run in a subprocess: once Qt is loaded into this interpreter by any
other means, an in-process check could not tell who loaded it.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

GUI_TOOLKITS = ("PyQt5", "PyQt6", "PySide2", "PySide6", "qtpy")

HEADLESS_MODULES = [
    "am_segmentation_manager.gui",
    "am_segmentation_manager.gui.cache",
    "am_segmentation_manager.gui.controller",
    "am_segmentation_manager.display",
    "am_segmentation_manager.frames",
    "am_segmentation_manager.config",
]


@pytest.mark.parametrize("module", HEADLESS_MODULES)
def test_module_does_not_import_a_gui_toolkit(module):
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(SRC)!r})
        import {module}
        loaded = [name for name in {GUI_TOOLKITS!r} if name in sys.modules]
        print(",".join(loaded))
    """)

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120,
    )

    assert result.returncode == 0, result.stderr
    loaded = result.stdout.strip()
    assert not loaded, f"{module} pulled in a GUI toolkit: {loaded}"


def test_headless_modules_import_without_the_optional_image_stack():
    """A minimal install must still be able to import the package."""
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(SRC)!r})

        # Make the optional image dependencies unimportable.
        import builtins
        blocked = {{"numpy", "cv2", "tifffile"}}
        real_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in blocked:
                raise ImportError(f"blocked: {{name}}")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = guarded
        for name in list(sys.modules):
            if name.split(".")[0] in blocked:
                del sys.modules[name]

        import am_segmentation_manager.gui.controller
        import am_segmentation_manager.gui.cache
        import am_segmentation_manager.display
        print("ok")
    """)

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
