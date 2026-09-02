"""The documents' claims about the suite, checked.

`docs/` keeps a rule where the suite keeps its guard: an entry states what must
stay true and names the test that stops it drifting, rather than restating the
argument the test's own docstring already makes. That only works while the
names still resolve -- a renamed test would leave the document pointing at
nothing, and nothing else would notice.

Names come from the *documents*, so writing one down is what makes it checked.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parent.parent
DOCS = REPOSITORY / "docs"
TESTS = REPOSITORY / "tests"

#: How long the documents may get. `CLAUDE.md` sets this: short enough to read
#: in one sitting, and approaching it means something in there has finished
#: rather than that the limit needs raising.
MAX_DOC_LINES = 900


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
