"""The words the window and the command line no longer use.

Reads the string literals under ``gui/`` and ``cli/`` -- docstrings and log
messages left out -- so it needs no Qt and no scan. The style these words gave
way to is in ``.claude/skills/ui-text/SKILL.md``.
"""
from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from functools import cache
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "am_segmentation_manager"

#: A retired phrasing, and what is said instead.
RETIRED: tuple[tuple[str, str], ...] = (
    (r"\bput (?:\w+ )?back\b", "restore"),
    (r"[Tt]ick 'Enable editing'", "Enable editing before ..."),
    (r"Show / hide", "Show or hide"),
    (r"\bpicked out\b", "selected, or the selection"),
    (r"Somebody else changed", "... was changed elsewhere"),
)

LOGGERS = {"logger", "log", "logging", "LOGGER"}


def _shown_strings(path: Path) -> Iterator[tuple[int, str]]:
    """Yield each string literal of *path* a user could read, with its line."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skipped: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
        ):
            skipped.add(id(body[0].value))
        if (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id in LOGGERS
        ):
            skipped.update(id(inner) for inner in ast.walk(node))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skipped:
            yield node.lineno, node.value


@cache
def _shown() -> tuple[tuple[str, int, str], ...]:
    """Return every string shown, read once for all the phrasings."""
    paths = sorted((PACKAGE / "gui").rglob("*.py")) + sorted((PACKAGE / "cli").glob("*.py"))
    return tuple(
        (str(path.relative_to(PACKAGE)), line, text)
        for path in paths
        for line, text in _shown_strings(path)
    )


@pytest.mark.parametrize(("retired", "instead"), RETIRED, ids=[r for r, _ in RETIRED])
def test_no_text_shown_uses_a_retired_phrasing(retired: str, instead: str):
    found = [
        f"{path}:{line}: {text[:60]!r}" for path, line, text in _shown() if re.search(retired, text)
    ]

    assert not found, f"say {instead!r} instead:\n" + "\n".join(found)
