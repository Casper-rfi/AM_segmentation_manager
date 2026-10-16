"""The words the window and the command line no longer use.

Reads the string literals under ``gui/`` and ``cli/`` -- docstrings and log
messages left out -- so it needs no Qt and no scan. The style these words gave
way to is in ``.claude/skills/ui-text/SKILL.md``.
"""
from __future__ import annotations

import argparse
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


#: ``am-seg`` in lower case, not part of a longer word or a path.
LOWER_CASE_NAME = re.compile(r"(?<![\w/.'`-])am-seg(?![\w-])")


def test_the_name_is_written_am_seg_except_where_it_is_the_command():
    """`am-seg gui` stays the command; anywhere else it is the name, AM-seg."""
    from am_segmentation_manager.cli.main import build_parser

    commands = {
        name for action in build_parser()._actions
        if isinstance(action, argparse._SubParsersAction) for name in action.choices
    }
    found = []
    for path, line, text in _shown():
        if text.strip() == "am-seg":
            continue
        for match in LOWER_CASE_NAME.finditer(text):
            following = text[match.end():].split(maxsplit=1)
            if following and following[0].strip("'\".,:;)") in commands | {"{__version__}"}:
                continue
            if following and following[0].startswith(("-", "{")):
                continue
            found.append(f"{path}:{line}: {text[:60]!r}")

    assert not found, "say 'AM-seg' instead:\n" + "\n".join(found)
