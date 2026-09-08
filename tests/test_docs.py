"""The documents' claims about the suite, checked.

`docs/` keeps a rule where the suite keeps its guard: an entry states what must
stay true and names the test that stops it drifting, rather than restating the
argument that test's own docstring already makes.

Two kinds of check. The first asks whether a pointer still resolves, for tests
and for code alike -- a renamed one would leave the document pointing at
nothing, and nothing else would notice. The second asks whether an entry has a
pointer at all, and of which kind: a test says the behaviour still holds, where
a symbol says only that a name was not renamed. Both are ratchets.

Names come from the *documents*, so writing one down is what makes it checked.
:data:`MAX_DOC_LINES` holds how long the four may get, and what may be taken out
of one.
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

#: How long the documents may get, in total. `CLAUDE.md` sets this: short
#: enough to read in one sitting.
#:
#: The number bounds the documents. It does **not** decide what goes. An entry
#: is removed because the code it describes no longer exists, because the fact
#: now sits in a docstring beside that code, because it was a plan and the work
#: has landed, or because the scientist said so -- never to make room for
#: something being added. Over the ceiling with none of those available is a
#: question for the scientist rather than a deletion to be chosen.
MAX_DOC_LINES = 1050

#: The documents holding invariants rather than status. An entry in the roadmap
#: or the future-work sketches describes what is unbuilt or unfinished and so
#: has nothing to point at; only these two are held to naming their evidence.
REFERENCE_DOCUMENTS: tuple[str, ...] = ("decisions.md", "display.md")

#: Entries in :data:`REFERENCE_DOCUMENTS` naming nothing a reader can check,
#: and entries naming a test the suite defines. Both are ratchets, and both
#: move only by a deliberate edit: an unbacked entry may be added only by
#: fixing another, and a test pointer may not be traded for a weaker one.
#: Neither ever forces a deletion -- a backed entry is free at any length.
MAX_UNBACKED_ENTRIES = 43
MIN_TEST_BACKED_ENTRIES = 40


def _documents() -> list[Path]:
    return sorted(DOCS.glob("*.md"))


@functools.lru_cache(maxsize=1)
def _test_names() -> frozenset[str]:
    """Return every test function the suite defines.

    Held after the first call, and frozen so a caller cannot edit what every
    other caller then sees: resolving one name at a time would re-parse the
    whole suite for each entry of each document.
    """
    names: set[str] = set()
    for path in sorted(TESTS.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                names.add(node.name)
    return frozenset(names)


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

    Failing here is not a licence to raise :data:`MAX_DOC_LINES`, and not a
    licence to delete whatever is nearest: what may go, and why, is on that
    constant.
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


# ---------------------------------------------------------------------------
# What an entry offers as evidence
#
# The checks above ask whether a pointer still resolves. These ask whether
# there is one at all, and of which kind: a *test* named beside a claim is
# evidence the behaviour still holds, where a symbol only shows the name has
# not been renamed. `_plots_by_image_number` outlived an entry describing it
# wrongly, which is the failure a symbol pointer cannot catch.
# ---------------------------------------------------------------------------


def _entries(document: Path) -> list[str]:
    """Return the document's top-level bullets, one string each.

    A bullet runs until the next one, a blank line or an unindented line, so
    the entry carries the whole of its own argument and none of its
    neighbour's. Prose outside a bullet is a preamble rather than a claim and
    is not returned.
    """
    entries: list[str] = []
    current: list[str] = []
    for line in document.read_text(encoding="utf-8").splitlines():
        if line.startswith("- "):
            if current:
                entries.append("\n".join(current))
            current = [line]
        elif current and line.startswith("  ") and line.strip():
            current.append(line)
        elif current:
            entries.append("\n".join(current))
            current = []
    if current:
        entries.append("\n".join(current))
    return entries


@functools.lru_cache(maxsize=1)
def _defined_names() -> frozenset[str]:
    """Return every bare identifier the package defines, anywhere."""
    _modules, top, members, _classes = _package_symbols()
    found: set[str] = set()
    for names in top.values():
        found |= names
    for names in members.values():
        found |= names
    return frozenset(found)


def _points_at_code(token: str) -> bool:
    """Return whether *token* names something in the package, however written.

    More permissive than :func:`_resolves`, which answers only for dotted
    names: an entry earns its evidence by naming `FilterRow` or `merge.py` as
    much as by naming `controls.SegmentedControl`. **A bare word counts only
    where it is shaped like code** -- underscored or CamelCase -- so that
    `selected` and `corrected` are read as the English they are rather than as
    whatever happens to share their spelling.
    """
    token = token.rstrip("()").split("/")[-1]
    modules, _top, members, _classes = _package_symbols()
    if token.endswith(".py"):
        return token[:-3] in modules
    if "." in token:
        head = token.split(".", 1)[0]
        return (head in modules or head in members) and _resolves(token)
    if token in modules:
        return True
    shaped = "_" in token or (token[:1].isupper() and any(c.isupper() for c in token[1:]))
    return shaped and token in _defined_names()


def _backing(entry: str) -> str:
    """Return what an entry offers: ``test``, ``code`` or ``nothing``."""
    tokens = re.findall(r"`([^`]+)`", entry)
    defined = _test_names()
    if any(re.fullmatch(r"test_[A-Za-z0-9_]+", t) and t in defined for t in tokens):
        return "test"
    if any(not t.startswith("test_") and _points_at_code(t) for t in tokens):
        return "code"
    return "nothing"


def _reference_entries() -> list[tuple[str, str]]:
    """Return every entry of the two reference documents, with its document."""
    return [
        (document.name, entry)
        for document in _documents()
        if document.name in REFERENCE_DOCUMENTS
        for entry in _entries(document)
    ]


def test_an_entry_that_names_nothing_is_added_only_by_fixing_another():
    """An unbacked entry states a rule with nothing to check it against.

    A ratchet rather than a bar: the count may fall and may not rise, so a new
    entry is free where it names its evidence and costs an old one where it
    does not. Failing here is not a reason to delete anything -- name the test
    or symbol that already guards the claim, or move the fact to a docstring
    beside the code, which is where a fact about behaviour belongs.

    Some entries legitimately point at nothing: the environment to run the
    suite in, the reference collection, a quirk of the microscope's own header.
    Those are facts about the world, which no test of ours can hold.
    """
    unbacked = [
        f"{name}: {' '.join(entry.split())[:70]}"
        for name, entry in _reference_entries()
        if _backing(entry) == "nothing"
    ]

    assert len(unbacked) <= MAX_UNBACKED_ENTRIES, (
        f"{len(unbacked)} entries name nothing checkable, "
        f"was {MAX_UNBACKED_ENTRIES}: {sorted(unbacked)}"
    )


def test_the_evidence_for_an_entry_is_not_traded_for_a_weaker_one():
    """A test pointer says the behaviour still holds; a symbol says less.

    The other ratchet, and the one that matters for staleness: swapping a named
    test for a named function passes every other check in this file while
    quietly removing the only thing that would notice the entry going out of
    date.
    """
    backed = sum(1 for _name, entry in _reference_entries() if _backing(entry) == "test")

    assert backed >= MIN_TEST_BACKED_ENTRIES, (
        f"{backed} entries name a live test, was {MIN_TEST_BACKED_ENTRIES}"
    )


def test_the_backing_check_can_tell_the_three_apart():
    """Or both ratchets hold at whatever the classifier happens to return."""
    assert _backing("- **A rule.** Guarded by `test_every_argument_has_help`.") == "test"
    assert _backing("- **A rule.** Stated in `layout.acquisition_stem`.") == "code"
    assert _backing("- **A rule.** Stated in `merge.py`.") == "code"
    assert _backing("- **A rule.** Nothing but `--dry-run` and `AM_state.yaml`.") == "nothing"
