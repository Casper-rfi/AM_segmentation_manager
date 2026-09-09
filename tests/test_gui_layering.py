"""Guards the rule that keeps the curation UI testable.

Only the widget layer may depend on a GUI toolkit. If the controller or the
cache ever import Qt, they stop being testable without a display and the split
this design relies on has quietly collapsed -- so it is checked rather than
merely documented.

The imports run in a subprocess: once Qt is loaded into this interpreter by any
other means, an in-process check could not tell who loaded it.
"""
from __future__ import annotations

import ast
import importlib.util
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


#: The bases CurationWindow mixes in, in order. Each holds one subject's
#: methods; none holds state of its own.
WINDOW_MIXINS = ("OperationRunner", "OperationActions", "LabelTab")


@pytest.mark.skipif(
    importlib.util.find_spec("qtpy") is None, reason="Qt binding not installed"
)
def test_no_mixin_shadows_another():
    """Two bases defining one name is the way a mixin split goes wrong.

    The window would keep whichever the MRO reached first and lose the other
    silently, so a method moved into the wrong module would go on working until
    somebody edited the copy that is never called. Nothing else checks this:
    the split is by subject, and Python does not mind if a subject repeats.
    """
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow

    bases = [b for b in CurationWindow.__bases__ if b.__name__ in WINDOW_MIXINS]
    assert len(bases) == len(WINDOW_MIXINS), f"expected {WINDOW_MIXINS}, got {bases}"

    seen: dict[str, str] = {}
    clashes: list[str] = []
    for base in bases:
        for name, value in vars(base).items():
            if name.startswith("__") or not callable(value):
                continue
            if name in seen:
                clashes.append(f"{name}: {seen[name]} and {base.__name__}")
            seen[name] = base.__name__
    assert not clashes, "; ".join(clashes)


@pytest.mark.skipif(
    importlib.util.find_spec("qtpy") is None, reason="Qt binding not installed"
)
def test_no_mixin_carries_state_of_its_own():
    """A mixin with an __init__ is an object that should have been one.

    Everything the three reach is built by the window, so none of them has a
    constructor to run -- and a mixin that acquired one would either never be
    called or would have to be threaded into the window's own.
    """
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow

    with_init = [
        base.__name__
        for base in CurationWindow.__bases__
        if base.__name__ in WINDOW_MIXINS and "__init__" in vars(base)
    ]
    assert not with_init, f"{with_init} would need building, so should be objects"


# ---------------------------------------------------------------------------
# The spine
#
# `CLAUDE.md` names four layers -- layout (what things are called), discovery
# (where they are), scanner (what they mean), operations (what you can do to
# them) -- with io/ below all four and gui/, ml/ and cli/ above everything.
# The checks below read *every* import, including the ones inside functions:
# roughly a third of this package's edges are written there, and a rule nobody
# can see is a rule nobody keeps.
# ---------------------------------------------------------------------------

#: Imports that break the spine on purpose. Each names why, and each is a
#: single edge rather than a module-wide exemption.
ALLOWED_UPWARD = {
    ("am_segmentation_manager.operations.predict_process",
     "am_segmentation_manager.ml.environment"),
    ("am_segmentation_manager.operations.predict_process",
     "am_segmentation_manager.ml.model"),
}


def _package_imports() -> list[tuple[str, str, str, int]]:
    """Return every in-package import as (source, target, function, line).

    *function* is empty for an import at module level. Relative imports are
    resolved the way Python resolves them, so a target here is the module that
    would actually be loaded.
    """
    import importlib.util

    root = SRC / "am_segmentation_manager"
    found: list[tuple[str, str, str, int]] = []
    for path in sorted(root.rglob("*.py")):
        parts = list(path.relative_to(root).with_suffix("").parts)
        is_package = parts[-1] == "__init__"
        if is_package:
            parts.pop()
        module = ".".join(["am_segmentation_manager", *parts])
        package = module if is_package else ".".join(
            ["am_segmentation_manager", *parts[:-1]]
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        enclosing: dict[int, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                    enclosing.setdefault(line, node.name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            target = importlib.util.resolve_name(
                "." * node.level + (node.module or ""), package
            )
            found.append((module, target, enclosing.get(node.lineno, ""), node.lineno))
    return found


def _tops(module: str) -> str:
    """Return the part of *module* the spine rules are stated about."""
    name = module.removeprefix("am_segmentation_manager").lstrip(".")
    return name.split(".")[0]


@pytest.mark.parametrize(
    ("layer", "must_not_import"),
    [
        ("io", {"layout", "discovery", "scanner", "operations", "gui", "ml", "cli"}),
        ("layout", {"discovery", "scanner", "operations", "gui", "ml", "cli"}),
        ("discovery", {"scanner", "operations", "gui", "ml", "cli"}),
        ("scanner", {"operations", "gui", "ml", "cli"}),
        ("operations", {"gui", "cli"}),
    ],
)
def test_a_layer_does_not_import_one_above_it(layer, must_not_import):
    """The spine only holds if it is checked where the edges actually are.

    An import written inside a function is invisible to a reader skimming the
    top of the file, and 138 of this package's were, so this reads those too.
    """
    broken = [
        (source, target, function, line)
        for source, target, function, line in _package_imports()
        if _tops(source) == layer
        and _tops(target) in must_not_import
        and (source, target) not in ALLOWED_UPWARD
    ]
    assert not broken, "\n".join(
        f"{s}:{n} {'in ' + f + '()' if f else '(module level)'} imports {t}"
        for s, t, f, n in broken
    )


def test_nothing_below_the_window_imports_it():
    """`gui/` and `cli/` are surfaces: everything may be used by them, and
    nothing below may reach back up into one."""
    reaching = [
        (source, target, function, line)
        for source, target, function, line in _package_imports()
        if _tops(target) in {"gui", "cli"}
        and _tops(source) not in {"gui", "cli"}
    ]
    assert not reaching, reaching
