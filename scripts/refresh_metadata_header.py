#!/usr/bin/env python3
"""Bring the comments of existing AM_metadata.yaml files up to the current template.

Files written by earlier versions of ``am-seg init-project`` open with a header
inviting hand edits, and end with a commented-out ``variants:`` example that the
package no longer reads. This replaces that header with the current one and
removes the example. **Only comments change**: every file is parsed before and
after, and the run stops before writing anything if a single one would read back
differently. A file whose header or example is not exactly what a template wrote
is reported and left alone, as is one holding a real ``variants:`` section.

**This is a standalone script, not an ``am-seg`` command.** New files get the
current header from ``init-project``; this only catches up files written before.

Previews by default; pass ``--apply`` to rewrite the files, which also writes a
manifest of every file changed.

    python scripts/refresh_metadata_header.py <directory> [--apply]
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from am_segmentation_manager.discovery import find_project_roots
from am_segmentation_manager.io.atomic import write_atomically
from am_segmentation_manager.metadata import AM_METADATA_FILENAME
from am_segmentation_manager.operations.init_project import header_lines

logger = logging.getLogger(__name__)

#: The header lines the old template wrote after the project name and ``#``.
OLD_HEADER: tuple[str, ...] = (
    "# Written by 'am-seg init-project'. Values here override experiment.yaml.",
    "# Replace every TODO with a real value, or delete the line if it",
    "# does not apply. Nothing is inherited from parent directories, so this",
    "# file is the whole story for this project.",
)

#: The commented example the old template ended with, preceded by a blank line.
OLD_VARIANTS_EXAMPLE = (
    "\n"
    "# Descriptions for imaging-condition variants, keyed by the suffix in the\n"
    "# image directory name (sem_5kv -> 5kv). Display only; delete if unused.\n"
    "# variants:\n"
    "#   5kv: 5 kV landing energy\n"
)

NAME_PREFIX = "# Metadata for "


class ValuesWouldChange(RuntimeError):
    """A rewrite would change what a file holds, which means this script is wrong."""


@dataclass(frozen=True)
class Refresh:
    """What becomes of one file.

    Attributes:
        path: The ``AM_metadata.yaml``.
        text: Its contents now.
        new_text: What it would hold, equal to *text* where nothing changes.
        header_replaced: Whether the old header is replaced.
        example_removed: Whether the ``variants:`` example is removed.
        problem: Why the file is left alone, or empty.
    """

    path: Path
    text: str
    new_text: str
    header_replaced: bool = False
    example_removed: bool = False
    problem: str = ""

    @property
    def changes(self) -> bool:
        """Return whether the file would be rewritten."""
        return not self.problem and self.new_text != self.text

    def describe(self, root: Path) -> str:
        """Return one line saying what happens to the file."""
        where = self.path.relative_to(root) if self.path.is_relative_to(root) else self.path
        if self.problem:
            return f"{where}: left alone -- {self.problem}"
        parts = [
            label for label, done in (
                ("header replaced", self.header_replaced),
                ("variants example removed", self.example_removed),
            ) if done
        ]
        return f"{where}: {', '.join(parts) or 'already current'}"


def plan_refresh(path: Path) -> Refresh:
    """Return what refreshing *path* would do, having written nothing.

    Raises:
        ValuesWouldChange: If the rewritten file would parse differently.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    if not lines[0].startswith(NAME_PREFIX):
        return Refresh(path, text, text, problem="its first line does not name the project")
    project_name = lines[0][len(NAME_PREFIX):]
    new_header = header_lines(project_name)

    new_text = text
    header_replaced = False
    old_header = [lines[0], "#", *OLD_HEADER]
    if lines[:len(old_header)] == old_header:
        new_text = "\n".join([*new_header, *lines[len(old_header):]])
        header_replaced = True
    elif lines[:len(new_header)] != new_header:
        return Refresh(path, text, text, problem="its header is not one the template wrote")

    before = _parsed(text)
    if isinstance(before, dict) and "variants" in before:
        return Refresh(path, text, text, problem="it holds a variants section with values")

    example_removed = False
    if new_text.endswith(OLD_VARIANTS_EXAMPLE):
        new_text = new_text[: -len(OLD_VARIANTS_EXAMPLE)]
        example_removed = True
    elif OLD_VARIANTS_EXAMPLE.split("\n")[1] in new_text:
        return Refresh(path, text, text, problem="its variants example has been edited")

    if _parsed(new_text) != before:
        raise ValuesWouldChange(f"Rewriting {path} would change what it holds")
    return Refresh(path, text, new_text, header_replaced, example_removed)


def _parsed(text: str) -> object:
    """Return what *text* holds as YAML."""
    return yaml.load(text, Loader=yaml.SafeLoader)


def plan_all(start_path: Path) -> list[Refresh]:
    """Return a plan for every project's ``AM_metadata.yaml`` under *start_path*."""
    return [
        plan_refresh(root / AM_METADATA_FILENAME)
        for root in find_project_roots(start_path)
        if (root / AM_METADATA_FILENAME).is_file()
    ]


def apply_refresh(plans: list[Refresh]) -> int:
    """Rewrite every file a plan changes, and return how many were rewritten.

    Each file is checked again just before it is replaced, so one edited since
    the plan was made is left as it now is.
    """
    written = 0
    for plan in plans:
        if not plan.changes:
            continue
        replaced = write_atomically(
            plan.path,
            lambda temporary, plan=plan: temporary.write_text(plan.new_text, encoding="utf-8"),
            commit_if=lambda _temporary, plan=plan: (
                plan.path.read_text(encoding="utf-8") == plan.text
            ),
        )
        if replaced:
            written += 1
        else:
            logger.warning("%s changed since it was read; left as it is", plan.path)
    return written


def write_manifest(path: Path, plans: list[Refresh]) -> None:
    """Write one row per file changed, naming what changed in it."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "header_replaced", "variants_example_removed"])
        for plan in plans:
            if plan.changes:
                writer.writerow([plan.path, plan.header_replaced, plan.example_removed])


def main(argv: list[str] | None = None) -> int:
    """Preview or perform the refresh under the given directory."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "start_path", nargs="?", default=".",
        help="Directory to search for projects (default: current directory)",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Rewrite the files. Without this, what would change is only listed",
    )
    parser.add_argument(
        "--manifest", type=Path,
        help="Where to write the list of files changed (default: the current directory)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    start_path = Path(args.start_path)
    try:
        plans = plan_all(start_path)
    except ValuesWouldChange as exc:
        logger.error("%s. Nothing was written.", exc)
        return 2
    if not plans:
        logger.info("No %s found under %s", AM_METADATA_FILENAME, start_path)
        return 0

    for plan in plans:
        logger.info("%s", plan.describe(start_path))
    changing = [plan for plan in plans if plan.changes]
    left_alone = [plan for plan in plans if plan.problem]
    logger.info(
        "%d of %d file(s) to rewrite, %d left alone.",
        len(changing), len(plans), len(left_alone),
    )
    if not args.apply or not changing:
        if changing:
            logger.info("Re-run with --apply to rewrite them.")
        return 0

    manifest = args.manifest or Path(
        f"refresh_metadata_header_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    )
    write_manifest(manifest, plans)
    logger.info("Wrote the manifest to %s", manifest)
    logger.info("Rewrote %d file(s)", apply_refresh(plans))
    return 0


if __name__ == "__main__":
    sys.exit(main())
