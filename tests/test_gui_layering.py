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
import functools
import importlib.util
import json
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


#: Run in one child for every headless module. A finder placed first refuses any
#: toolkit and records who asked, so nothing is ever loaded and every module is
#: judged on its own. A module that asked and survived (the request sat in a
#: try) is dropped with the rest imported alongside it, so the next module to
#: import it asks again; one that asked nothing is safe to leave cached.
_TOOLKIT_PROBE = """
import importlib, importlib.abc, json, sys, traceback

config = json.loads(sys.argv[1])
sys.path.insert(0, config["src"])
toolkits = set(config["toolkits"])
requested = []


class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in toolkits:
            requested.append(name)
            raise ModuleNotFoundError(f"toolkit refused: {name}", name=name)
        return None


sys.meta_path.insert(0, Refuse())
results = {}
for module in config["modules"]:
    before = set(sys.modules)
    requested.clear()
    error = None
    try:
        importlib.import_module(module)
    except Exception:
        error = traceback.format_exc()
    results[module] = {"toolkits": sorted(set(requested)), "error": error}
    if requested:
        for name in set(sys.modules) - before:
            if name.split(".")[0] == "am_segmentation_manager":
                del sys.modules[name]
                # `from package import module` reads the attribute before it imports.
                parent, _, child = name.rpartition(".")
                if hasattr(sys.modules.get(parent), child):
                    delattr(sys.modules[parent], child)
print(json.dumps(results))
"""


@pytest.fixture(scope="module")
def toolkit_requests() -> dict[str, dict]:
    """Import every headless module in one child and return what each asked for.

    One interpreter for all of them, since it imports the package once rather
    than once per module. Third-party modules stay cached between modules, so a
    toolkit requested only by a library that guards the import would be charged
    to the first module that loaded it.
    """
    config = {
        "src": str(SRC),
        "modules": HEADLESS_MODULES,
        "toolkits": list(GUI_TOOLKITS),
    }
    result = subprocess.run(
        [sys.executable, "-c", _TOOLKIT_PROBE, json.dumps(config)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("module", HEADLESS_MODULES)
def test_module_does_not_import_a_gui_toolkit(module, toolkit_requests):
    outcome = toolkit_requests[module]

    assert not outcome["toolkits"], (
        f"{module} pulled in a GUI toolkit: {','.join(outcome['toolkits'])}"
    )
    assert outcome["error"] is None, outcome["error"]


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
# them) -- with io/ below all four, training/ above operations/, nothing importing
# gui/, and nothing outside ml/ importing it at module level.
#
# The checks below read *every* import, including the ones inside functions:
# roughly a third of this package's edges are written there, and a rule nobody
# can see is a rule nobody keeps. There is no exception list. The one rule that
# would have needed one is stated so that it does not: what matters about ml/
# is that torch is not *loaded*, which is a rule about module-level imports, so
# the prediction child importing it inside the function that runs there keeps
# the rule rather than breaking it.
# ---------------------------------------------------------------------------


@functools.cache
def _parsed_modules() -> tuple[tuple[tuple[str, ...], ast.Module, dict[int, str]], ...]:
    """Return each module under the package as (path parts, syntax tree, function by line).

    Parsed once for the whole file: reading the package is most of what the
    rules below cost, and none of them changes it. The function map names the
    outermost function enclosing each line; a line at module level has none.
    """
    root = SRC / "am_segmentation_manager"
    parsed = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        enclosing: dict[int, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                    enclosing.setdefault(line, node.name)
        parsed.append((tuple(path.relative_to(root).with_suffix("").parts), tree, enclosing))
    return tuple(parsed)


@functools.cache
def _package_imports() -> tuple[tuple[str, str, str, int], ...]:
    """Return every in-package import as (source, target, function, line).

    *function* is empty for an import at module level. Relative imports are
    resolved the way Python resolves them, so a target here is the module that
    would actually be loaded.
    """
    found: list[tuple[str, str, str, int]] = []
    for path_parts, tree, enclosing in _parsed_modules():
        parts = list(path_parts)
        is_package = parts[-1] == "__init__"
        if is_package:
            parts.pop()
        module = ".".join(["am_segmentation_manager", *parts])
        package = module if is_package else ".".join(
            ["am_segmentation_manager", *parts[:-1]]
        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            target = importlib.util.resolve_name(
                "." * node.level + (node.module or ""), package
            )
            # `from . import parallel` loads the module `parallel`, not the package.
            for alias in node.names:
                submodule = f"{target}.{alias.name}"
                loaded = submodule if submodule in _module_names() else target
                found.append((module, loaded, enclosing.get(node.lineno, ""), node.lineno))
    return tuple(dict.fromkeys(found))


@functools.cache
def _module_names() -> frozenset[str]:
    """Return the dotted name of every module and package under the package."""
    names = set()
    for path_parts, _tree, _enclosing in _parsed_modules():
        parts = [part for part in path_parts if part != "__init__"]
        names.add(".".join(["am_segmentation_manager", *parts]))
    return frozenset(names)


@functools.cache
def _external_imports() -> tuple[tuple[str, str, str, int], ...]:
    """Return every import of another distribution as (source, top-level name, function, line)."""
    found: list[tuple[str, str, str, int]] = []
    for path_parts, tree, enclosing in _parsed_modules():
        module = ".".join(["am_segmentation_manager", *path_parts])
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                found.append((module, name.split(".")[0], enclosing.get(node.lineno, ""), node.lineno))
    return tuple(found)


def _tops(module: str) -> str:
    """Return the part of *module* the spine rules are stated about."""
    name = module.removeprefix("am_segmentation_manager").lstrip(".")
    return name.split(".")[0]


@pytest.mark.parametrize(
    ("layer", "must_not_import"),
    [
        ("io", {"layout", "discovery", "scanner", "operations", "training", "gui", "ml", "cli"}),
        ("layout", {"discovery", "scanner", "operations", "training", "gui", "ml", "cli"}),
        ("discovery", {"scanner", "operations", "training", "gui", "ml", "cli"}),
        ("scanner", {"operations", "training", "gui", "ml", "cli"}),
        ("operations", {"training", "gui", "cli"}),
        ("training", {"gui", "cli"}),
    ],
)
def test_a_layer_does_not_import_one_above_it(layer, must_not_import):
    """The spine only holds if it is checked where the edges actually are.

    An import written inside a function is invisible to a reader skimming the
    top of the file, and 138 of this package's were, so this reads those too.
    Whichever rule runs first pays to parse the package, which can pass a
    second; the rest read what it parsed.
    """
    broken = [
        (source, target, function, line)
        for source, target, function, line in _package_imports()
        if _tops(source) == layer and _tops(target) in must_not_import
    ]
    assert not broken, "\n".join(
        f"{s}:{n} {'in ' + f + '()' if f else '(module level)'} imports {t}"
        for s, t, f, n in broken
    )


def test_nothing_imports_the_window():
    """`gui/` and `cli/` are surfaces: they may use everything, and nothing may
    reach back up into one.

    What this protects is not tidiness. An operation that could open a dialog
    would be unscriptable, unbatchable and untestable even where Qt is
    installed, and the command line would stop being able to do what the window
    does -- which is the whole claim `README.md` makes for it.
    """
    reaching = [
        (source, target, function, line)
        for source, target, function, line in _package_imports()
        if _tops(target) in {"gui", "cli"}
        and _tops(source) not in {"gui", "cli"}
    ]
    assert not reaching, reaching


def test_no_module_imports_one_that_imports_it():
    """Two modules that need each other are one module in two files, and the
    import that closes the loop has to hide inside a function, where nobody
    reading either file's head can see that they are tied."""
    imports: dict[str, list[tuple[str, str, int]]] = {}
    for source, target, function, line in _package_imports():
        if source != target:
            imports.setdefault(source, []).append((target, function, line))

    def cycle_from(start: str) -> list[str]:
        """Return the imports leading from *start* back to it, or [] if none do."""
        path, taken, seen = [start], [], {start}
        stack = [iter(imports.get(start, []))]
        while stack:
            step = next(stack[-1], None)
            if step is None:
                stack.pop()
                path.pop()
                if taken:
                    taken.pop()
                continue
            target, function, line = step
            edge = f"{path[-1]}:{line}{' in ' + function + '()' if function else ''}"
            if target == start:
                return [*taken, edge, start]
            if target not in seen:
                seen.add(target)
                path.append(target)
                taken.append(edge)
                stack.append(iter(imports.get(target, [])))
        return []

    cycles = [cycle for module in sorted(imports) if (cycle := cycle_from(module))]
    assert not cycles, "\n".join(" -> ".join(cycle) for cycle in cycles)


def test_torch_is_not_loaded_outside_the_module_that_runs_it():
    """Nothing outside `ml/` may import it at *module* level.

    The main environment has no torch, so a module-level import anywhere the
    window or the command line reaches would break it outright. Inside a
    function is fine and is how all three callers do it: the cost is paid by
    whoever predicts, and by nobody else.
    """
    loaded = [
        (source, target, line)
        for source, target, function, line in _package_imports()
        if _tops(target) == "ml" and _tops(source) != "ml" and not function
    ]
    assert not loaded, "\n".join(
        f"{s}:{n} imports {t} at module level" for s, t, n in loaded
    )


#: What the train extra brings that nothing else does: the trainer, and the two
#: libraries whose import costs seconds and sets process-wide state.
TRAINING_STACK = {"am_model_training", "monai", "mlflow"}


def test_the_training_stack_is_imported_only_inside_ml_functions():
    """Only `ml/` may import it, and only inside the function that trains.

    Most environments have no train extra, and importing AM-model-training
    loads MONAI and mlflow and sets torch's sharing strategy for the whole
    process: a cost and a side effect only the training child should carry.
    """
    misplaced = [
        f"{source}:{line} imports {name}"
        + (" at module level" if not function else " outside ml/")
        for source, name, function, line in _external_imports()
        if name in TRAINING_STACK and (not function or _tops(source) != "ml")
    ]
    assert not misplaced, "\n".join(misplaced)
