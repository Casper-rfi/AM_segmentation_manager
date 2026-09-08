"""The documents' claims about the suite, checked.

`docs/` keeps a rule where the suite keeps its guard: an entry states what must
stay true and names the test that stops it drifting, rather than restating the
argument the test's own docstring already makes. That only works while the
names still resolve -- a renamed test would leave the document pointing at
nothing, and nothing else would notice.

Names come from the *documents*, so writing one down is what makes it checked.
The same goes for the code a document points at: an entry that says the argument
lives in `some_module.some_function` is worth no more than that name.
"""
from __future__ import annotations

import ast
import collections
import functools
import re
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parent.parent
DOCS = REPOSITORY / "docs"
TESTS = REPOSITORY / "tests"

#: How long the documents may get. `CLAUDE.md` sets this: short enough to read
#: in one sitting, and approaching it means something in there has finished
#: rather than that the limit needs raising. Raised once, from 1000, by the
#: scientist, for the history work -- which is a decision somebody took and not
#: a thing to do again when the next feature lands.
MAX_DOC_LINES = 1050


def _documents() -> list[Path]:
    return sorted(DOCS.glob("*.md"))


def _test_names() -> set[str]:
    """Return every test function the suite defines."""
    names: set[str] = set()
    for path in sorted(TESTS.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                names.add(node.name)
    return names


def _named_in(document: Path) -> set[str]:
    """Return the test names a document points at, in backticks."""
    return set(re.findall(r"`(test_[A-Za-z0-9_]+)`", document.read_text(encoding="utf-8")))


def test_every_test_a_document_names_exists():
    """A document that points at a renamed test points at nothing.

    The pointers are what let an entry be a rule rather than a rehearsal of the
    reasoning, so they have to stay resolvable for the shortening to be honest.
    """
    defined = _test_names()
    missing = {
        f"{document.name}: {name}"
        for document in _documents()
        for name in _named_in(document)
        if name not in defined
    }

    assert not missing, sorted(missing)


def test_the_documents_point_at_some_tests():
    """Or the check above passes by describing nothing."""
    named = {name for document in _documents() for name in _named_in(document)}

    assert len(named) >= 5


@pytest.mark.parametrize("document", _documents(), ids=lambda path: path.name)
def test_a_document_has_no_broken_local_link(document: Path):
    """A cross-reference between the four is followed, not admired."""
    text = document.read_text(encoding="utf-8")
    broken = [
        target
        for target in re.findall(r"\]\(([^)]+)\)", text)
        if not target.startswith(("http://", "https://", "#"))
        and not (document.parent / target.split("#")[0]).exists()
        and not (REPOSITORY / target.split("#")[0]).exists()
    ]

    assert not broken, broken


def test_the_documents_stay_short_enough_to_read():
    """`CLAUDE.md` sets the ceiling, and nothing else counts the lines.

    Failing here is not a licence to raise :data:`MAX_DOC_LINES`: it means
    something in `docs/` has finished and should go, or that an argument now
    lives in a test docstring and the entry can point at it instead.
    """
    counted = {
        document.name: len(document.read_text(encoding="utf-8").splitlines())
        for document in _documents()
    }
    total = sum(counted.values())

    assert total <= MAX_DOC_LINES, f"{total} lines: {counted}"


#: Suffixes that make a dotted name a *file* rather than a reference to code --
#: `display.md`, `theme.py`. Checked against the last segment, so a path with
#: directories in it is caught too.
FILE_SUFFIXES = frozenset(
    {"py", "md", "yaml", "yml", "tif", "png", "csv", "pth", "sh", "ini", "log",
     "ico", "toml"}
)

#: A backticked name that might be code: two or more word segments, no spaces,
#: no `::`, no flags. Anything else the documents put in backticks -- a command,
#: a filename, a value, a Qt method -- is not this shape or is filtered below.
DOTTED_NAME = re.compile(r"^\w+(?:\.\w+)+$")


@functools.lru_cache(maxsize=1)
def _package_symbols() -> tuple[set[str], dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    """Return what the package defines: modules, their names, and class members.

    Held after the first call: every name in every document is resolved against
    it, and parsing the package once a name is most of a second each time.

    Instance attributes count, and are found as ``self.x = ...`` anywhere in a
    class: most of this package's attributes are assigned in ``__init__``, and a
    checker that missed them would call `ExportPanel._recount` a broken pointer.
    """
    modules: set[str] = set()
    top: dict[str, set[str]] = collections.defaultdict(set)
    members: dict[str, set[str]] = collections.defaultdict(set)
    classes: dict[str, set[str]] = collections.defaultdict(set)
    for root in (REPOSITORY / "src", REPOSITORY / "scripts"):
        for path in sorted(root.rglob("*.py")):
            modules.add(path.stem)
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    top[path.stem].add(node.name)
                    classes[path.stem].add(node.name)
                    members[node.name] |= _class_members(node)
                elif isinstance(node, ast.FunctionDef):
                    top[path.stem].add(node.name)
                else:
                    top[path.stem] |= _assigned_names(node)
    return modules, top, members, classes


def _class_members(node: ast.ClassDef) -> set[str]:
    """Return everything reachable on a class, instance attributes included."""
    found: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.FunctionDef):
            found.add(sub.name)
        elif isinstance(sub, ast.Assign):
            for target in sub.targets:
                if isinstance(target, ast.Name):
                    found.add(target.id)
                elif isinstance(target, ast.Attribute) and getattr(
                    target.value, "id", None
                ) == "self":
                    found.add(target.attr)
        elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
            found.add(sub.target.id)
    return found


def _assigned_names(node: ast.stmt) -> set[str]:
    """Return the module-level names one statement binds."""
    if isinstance(node, ast.Assign):
        return {t.id for t in node.targets if isinstance(t, ast.Name)}
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return {node.target.id}
    return set()


def _cited_code(document: Path) -> list[str]:
    """Return the dotted names a document points at that could be this package.

    A name is left alone unless its first segment is one of our modules or
    classes: `collections.Counter` and `system.ion.voltage` are not ours to
    resolve.

    **A Capitalised segment is skipped only under a class**, where the FEI
    header keys `Stage.StageX` and `Stage.CompustageX` collide by accident with
    a dataclass of ours and name nothing on it. Under a *module* it is a class
    of ours -- `controls.SegmentedControl` -- and is exactly what wants
    checking, so the exclusion is not allowed to reach that far.
    """
    modules, _top, members, _classes = _package_symbols()
    cited = []
    for token in re.findall(r"`([^`]+)`", document.read_text(encoding="utf-8")):
        if not DOTTED_NAME.match(token):
            continue
        head, rest = token.split(".", 1)
        last = rest.split(".")[-1]
        if last in FILE_SUFFIXES:
            continue
        if head not in modules and head in members:
            if last[:1].isupper() and not last.isupper():
                continue
        if head in modules or head in members:
            cited.append(token)
    return cited


def _resolves(token: str) -> bool:
    """Return whether *token* names something the package actually defines."""
    modules, top, members, classes = _package_symbols()
    head, rest = token.split(".", 1)
    if head in modules:
        if rest in top[head]:
            return True
        return any(rest in members[name] for name in classes[head])
    return rest in members.get(head, set())


@pytest.mark.parametrize("document", _documents(), ids=lambda path: path.name)
def test_every_code_name_a_document_points_at_exists(document: Path):
    """An entry that stops rehearsing an argument and points at the code making
    it is worth exactly as much as that pointer.

    This found `ProjectTree.apply_filter`, which had never existed under that
    name -- the method is `hide_rows_without`. Nothing else would have: the
    suite reads the documents for *test* names, and a renamed function leaves no
    other mark on them.
    """
    broken = [name for name in _cited_code(document) if not _resolves(name)]

    assert not broken, broken


def test_the_documents_point_at_some_code():
    """Or the check above passes by resolving nothing."""
    cited = [name for document in _documents() for name in _cited_code(document)]

    assert len(cited) >= 20
