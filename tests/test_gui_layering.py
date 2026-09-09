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

#: The two modules under `gui/` whose whole job is Qt: a worker thread and the
#: loader that drives one. Everything else there is a controller or the data it
#: carries, and is held to importing no toolkit.
GUI_MODULES_THAT_MAY_IMPORT_QT = frozenset({"loader", "worker"})


def _headless_gui_modules() -> list[str]:
    """Return every module directly under `gui/` that must stay toolkit-free.

    Discovered rather than listed. A list has to be extended by whoever adds a
    controller, and three of them were added without it -- so the guard covered
    less each time the window grew. A new Qt-importing module under `gui/`
    fails this instead, which is the question being asked: does it belong in
    `gui/widgets/`?
    """
    package = SRC / "am_segmentation_manager" / "gui"
    found = [
        f"am_segmentation_manager.gui.{path.stem}"
        for path in sorted(package.glob("*.py"))
        if path.stem != "__init__"
        and path.stem not in GUI_MODULES_THAT_MAY_IMPORT_QT
    ]
    return ["am_segmentation_manager.gui", *found]


#: Modules outside `gui/` that the window reads and that must stay drawable
#: without a display. No directory rule reaches these, so they are named.
HEADLESS_SUPPORTING_MODULES = [
    "am_segmentation_manager.display",
    "am_segmentation_manager.frames",
    "am_segmentation_manager.config",
]

HEADLESS_MODULES = [*_headless_gui_modules(), *HEADLESS_SUPPORTING_MODULES]


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
