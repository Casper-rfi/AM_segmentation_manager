#!/usr/bin/env python3
"""Write the user guide's generated tables: the metadata fields and the commands.

**A standalone script, not an ``am-seg`` command.** Run it after changing a
command or a metadata field; ``tests/test_guides.py`` fails until it has been.
Each table sits between two markers in its page, and only the text between
them is rewritten::

    <!-- generated: commands -->
    ...
    <!-- end generated -->

    uv run python scripts/guide_tables.py           # rewrite the tables
    uv run python scripts/guide_tables.py --check   # exit 1 if any is out of date

The fields come from ``metadata.FIELDS`` and the commands from the parser, and
need nothing installed. The keys are not tabled: F1 lists them in the window.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent
GUIDES = REPOSITORY / "guides"

#: Which page holds which table.
PAGES = {
    "fields": GUIDES / "user_guide.md",
    "commands": GUIDES / "command_line.md",
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


def fields_markdown() -> str:
    """Return every descriptive field under its heading on the Add data tab, as tables."""
    from am_segmentation_manager.metadata import FIELDS, NUMBER

    parts, section = [], None
    for field in FIELDS:
        if field.section != section:
            section = field.section
            parts.append(f"\n**{section}**\n\n| Field | What to enter | Takes |\n|---|---|---|\n")
        if field.choices:
            takes = "one of: " + ", ".join(field.choices)
        elif field.kind == NUMBER:
            takes = "a number"
        else:
            takes = "text"
        if field.multi:
            takes += ", one or several"
        if field.defaults:
            takes += "; usually " + " or ".join(field.defaults)
        parts.append(f"| {_cell(field.label)} | {_cell(field.comment)} | {_cell(takes)} |\n")
    return "".join(parts)


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


def generated() -> dict[str, str]:
    """Return every table's Markdown by name."""
    return {"fields": fields_markdown(), "commands": commands_markdown()}


def main(argv: list[str] | None = None) -> int:
    """Rewrite the tables, or with ``--check`` report which are out of date."""
    arguments = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    arguments.add_argument(
        "--check", action="store_true", help="Change nothing; exit 1 if a table is out of date"
    )
    options = arguments.parse_args(argv)

    stale = []
    for name, body in generated().items():
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
