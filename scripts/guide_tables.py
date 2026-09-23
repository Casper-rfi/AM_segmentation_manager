#!/usr/bin/env python3
"""Write the user guide's generated tables: every command, and each tab's keys.

**A standalone script, not an ``am-seg`` command.** Run it after changing a
command, a key or a mouse gesture; ``tests/test_guides.py`` fails until it has
been. Each table sits between two markers in its page, and only the text
between them is rewritten::

    <!-- generated: commands -->
    ...
    <!-- end generated -->

    uv run python scripts/guide_tables.py           # rewrite the tables
    uv run python scripts/guide_tables.py --check   # exit 1 if any is out of date

The commands come from the parser, and need nothing installed. The keys are
read from the window's own table, the one F1 shows, so they need the ``gui``
extra: the window is built offscreen, with the packaged segmentation config
naming the classes and a throwaway settings file, and never shown or scanned.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent
GUIDES = REPOSITORY / "guides"

#: Which page holds which table.
PAGES = {
    "commands": GUIDES / "08_command_line.md",
    "keys": GUIDES / "09_keys.md",
}

_BLOCK = re.compile(
    r"(<!-- generated: (?P<name>[a-z]+) -->\n)(?P<body>.*?)(<!-- end generated -->)",
    re.DOTALL,
)


def _cell(text: str) -> str:
    """Return *text* safe inside a Markdown table cell."""
    return " ".join(text.split()).replace("|", "\\|")


def commands_markdown() -> str:
    """Return every command under its heading from ``am-seg --help``, as tables."""
    from am_segmentation_manager.cli.main import GROUPS, build_parser

    parser = build_parser()
    subparsers = next(
        action for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    # The parser's own list, in its order: an alias shares its command's row.
    helps = {choice.dest: choice.help or "" for choice in subparsers._choices_actions}
    parts = []
    for group in GROUPS:
        parts.append(f"### {group.TITLE}\n\n| Command | What it does |\n|---|---|\n")
        parts.extend(
            f"| `am-seg {name}` | {_cell(help_line)} |\n"
            for name, help_line in helps.items() if name in group.COMMANDS
        )
        parts.append("\n")
    return "".join(parts)


def tab_rows(window) -> list[tuple[str, dict[str, list[tuple[str, str]]]]]:
    """Return ``(tab name, rows under each heading)`` for every tab, in tab order."""
    tabs = []
    for index in range(window.tabs.count()):
        window.tabs.setCurrentIndex(index)
        tabs.append((
            window.tabs.tabText(index),
            window._keys.reference_rows(window._current_tab()),
        ))
    return tabs


def keys_markdown(window) -> str:
    """Return each tab's keys and gestures under F1's headings, as tables."""
    parts = []
    for name, grouped in tab_rows(window):
        parts.append(f"### {name}\n\n")
        for heading, rows in grouped.items():
            parts.append(f"**{heading}**\n\n| Press or do | What it does |\n|---|---|\n")
            parts.extend(f"| {_cell(action)} | {_cell(what)} |\n" for action, what in rows)
            parts.append("\n")
    return "".join(parts)


def build_window(settings_path: Path):
    """Return the window, built offscreen and unscanned, keeping settings in *settings_path*."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtCore import QSettings
    from qtpy.QtWidgets import QApplication

    from am_segmentation_manager.config import load_segmentation_config
    from am_segmentation_manager.gui.cache import FrameCache
    from am_segmentation_manager.gui.controller import CurationController
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow

    application = QApplication.instance() or QApplication([])
    controller = CurationController(cache=FrameCache(), config=load_segmentation_config())
    window = CurationWindow(
        controller, settings=QSettings(str(settings_path), QSettings.Format.IniFormat)
    )
    return application, window


def replace_generated(text: str, name: str, body: str) -> str:
    """Return *text* with the block named *name* holding *body*.

    Raises:
        ValueError: If *text* has no such block.
    """
    found = False

    def swap(match: re.Match[str]) -> str:
        nonlocal found
        if match["name"] != name:
            return match[0]
        found = True
        return f"{match[1]}\n{body.strip()}\n\n{match[4]}"

    replaced = _BLOCK.sub(swap, text)
    if not found:
        raise ValueError(f"no '<!-- generated: {name} -->' block")
    return replaced


def generated(window=None) -> dict[str, str]:
    """Return every table's Markdown by name; the keys only where *window* is given."""
    tables = {"commands": commands_markdown()}
    if window is not None:
        tables["keys"] = keys_markdown(window)
    return tables


def main(argv: list[str] | None = None) -> int:
    """Rewrite the tables, or with ``--check`` report which are out of date."""
    arguments = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    arguments.add_argument(
        "--check", action="store_true", help="Change nothing; exit 1 if a table is out of date"
    )
    options = arguments.parse_args(argv)

    with tempfile.TemporaryDirectory() as scratch:
        _application, window = build_window(Path(scratch) / "settings.ini")
        try:
            tables = generated(window)
        finally:
            # Stops the window's worker threads, which a process ending
            # around them reports as destroyed while running.
            window.close()

    stale = []
    for name, body in tables.items():
        page = PAGES[name]
        text = page.read_text(encoding="utf-8")
        wanted = replace_generated(text, name, body)
        if wanted == text:
            continue
        stale.append(page)
        if not options.check:
            page.write_text(wanted, encoding="utf-8")
    for page in stale:
        verb = "Out of date" if options.check else "Rewrote"
        print(f"{verb}: {page.relative_to(REPOSITORY)}")
    return 1 if options.check and stale else 0


if __name__ == "__main__":
    sys.exit(main())
