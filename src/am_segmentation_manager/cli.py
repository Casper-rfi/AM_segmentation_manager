"""Command line interface for common operations.

The CLI is intentionally lightweight and uses argparse so users can run
``python -m am_segmentation_manager.cli --help`` to see available commands and
``python -m am_segmentation_manager.cli <command> --help`` to get command-specific
usage information.

After installing the package, the ``am-seg`` entry point is also available.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .filtering import iter_matching_records
from .operations.predictions import convert_record_predictions
from .reporting import print_project_summary
from .scanner import Scanner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------

def _add_scope_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the common scope-selection arguments to *parser*."""
    parser.add_argument("paths", nargs="*", help="Explicit image paths")
    parser.add_argument("--start-path", default=".", help="Directory to start scanning from")
    parser.add_argument(
        "--project", action="append", default=[],
        help="Project name or number (for example 001); use 'all' explicitly",
    )
    parser.add_argument(
        "--lamella", action="append", default=[],
        help="Lamella name/number; use 'all' explicitly",
    )
    parser.add_argument(
        "--stage", action="append", default=[],
        help="Stage name/number; use 'all' explicitly",
    )
    parser.add_argument(
        "--image", action="append", default=[],
        help="Image name/number; use 'all' explicitly",
    )
    parser.add_argument(
        "--modality", choices=["SEM", "FIB", "all"], required=True,
        help="Image modality to operate on; specify 'all' for both SEM and FIB",
    )


def _resolve_records(args: argparse.Namespace) -> list:
    """Resolve the target records from *args*, using paths or indexed scope.

    Returns the list of matching :class:`ImageRecord` objects.  Logs an
    error and returns an empty list when scope arguments are incomplete.
    """
    from .operations.selection import record_from_image_path

    if args.paths and not any([args.project, args.lamella, args.stage, args.image]):
        records = [record_from_image_path(Path(path)) for path in args.paths]
        return [r for r in records if args.modality == "all" or r.modality == args.modality]

    missing_scopes = [
        flag
        for flag, values in (
            ("--project", args.project),
            ("--lamella", args.lamella),
            ("--stage", args.stage),
            ("--image", args.image),
        )
        if not values
    ]
    if missing_scopes:
        logger.error(
            "Indexed operations require explicit selectors for every scope. "
            "Use 'all' where appropriate (missing: %s).",
            ", ".join(missing_scopes),
        )
        return []

    index = Scanner(Path(args.start_path)).build_index()
    return list(
        iter_matching_records(
            index,
            project_selectors=args.project,
            lamella_selectors=args.lamella,
            stage_selectors=args.stage,
            image_selectors=args.image,
            modality=args.modality,
        )
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI runner
    """Parse arguments and dispatch to the appropriate operation."""
    parser = argparse.ArgumentParser(
        "am_segmentation_manager",
        description="Manage Autolamella segmentation projects",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    sub = parser.add_subparsers(dest="command")

    # -- list-projects / scan -----------------------------------------------
    list_projects = sub.add_parser(
        "list-projects",
        aliases=["scan"],
        help="Scan a directory and list discovered projects",
        description="Recursively scan for Autolamella projects and print a project summary",
    )
    list_projects.add_argument("start_path", nargs="?", default=".", help="Directory to start scanning from")

    # -- convert-predictions ------------------------------------------------
    convert_predictions = sub.add_parser(
        "convert-predictions",
        help="Convert predictions into labels for explicitly selected images",
    )
    _add_scope_arguments(convert_predictions)
    convert_predictions.add_argument(
        "--kind", choices=["raw", "clean"], required=True,
        help="Prediction set to convert",
    )

    # -- deselect / select / correct ----------------------------------------
    for name, help_text in (
        ("deselect", "Move images and labels into the deselected state"),
        ("select", "Move images and labels into the selected state"),
        ("correct", "Move images and labels into the corrected state"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        _add_scope_arguments(cmd)
        cmd.add_argument(
            "--confirm-corrected", action="store_true",
            help="Confirm moving artifacts out of the corrected state",
        )

    args = parser.parse_args(argv)

    # -- configure logging --------------------------------------------------
    level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # -- dispatch -----------------------------------------------------------
    if args.command in {"list-projects", "scan"}:
        scanner = Scanner(Path(args.start_path))
        index = scanner.build_index()
        for project in index.all_projects():
            print_project_summary(project)
        return 0

    if args.command == "convert-predictions":
        records = _resolve_records(args)
        if not records:
            logger.info("No matching images found to convert")
            return 2
        converted = convert_record_predictions(records, prediction_kind=args.kind)
        logger.info("Total converted: %d", converted)
        return 0

    if args.command in {"deselect", "select", "correct"}:
        from .operations.selection import (
            deselect_records,
            mark_corrected_records,
            select_records,
        )

        records = _resolve_records(args)
        if not records:
            logger.info("No matching images found to %s", args.command)
            return 2

        operation = {
            "deselect": deselect_records,
            "select": select_records,
            "correct": mark_corrected_records,
        }[args.command]
        try:
            changed = operation(records, allow_move_from_corrected=args.confirm_corrected)
        except (FileNotFoundError, FileExistsError, PermissionError, ValueError) as exc:
            logger.error("Could not %s matching images: %s", args.command, exc)
            return 2
        logger.info("Total %s: %d", args.command, changed)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI runner
    raise SystemExit(main())
