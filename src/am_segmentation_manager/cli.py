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
from .io.path import make_path_relative
from .operations.predictions import convert_record_predictions
from .reporting import print_project_summary
from .scanner import Scanner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------

def _add_scope_arguments(
    parser: argparse.ArgumentParser,
    default_modality: str | None = None,
    include_state: bool = True,
) -> None:
    """Add the common scope-selection arguments to *parser*.

    Set *include_state* to False when the caller registers its own
    ``--state`` argument (for example to make it required).
    """
    parser.add_argument("paths", nargs="*", help="Explicit image paths or scan directory")
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
    if include_state:
        parser.add_argument(
            "--state",
            choices=["corrected", "selected", "deselected", "all"],
            default="all",
            help="Restrict to images currently in this workflow state (default: 'all')",
        )
    if default_modality is not None:
        parser.add_argument(
            "--modality", choices=["SEM", "FIB", "all"], default=default_modality,
            help=f"Image modality to operate on (default: {default_modality})",
        )
    else:
        parser.add_argument(
            "--modality", choices=["SEM", "FIB", "all"], required=True,
            help="Image modality to operate on; specify 'all' for both SEM and FIB",
        )


def _resolve_records(args: argparse.Namespace) -> tuple[list, Path]:
    """Resolve target records and scan start_path from CLI arguments.

    Returns a tuple of `(records, start_path)`.
    """
    from .operations.selection import record_from_image_path

    state = getattr(args, "state", "all")
    start_path = Path(args.start_path)
    file_paths: list[Path] = []

    for item in args.paths:
        p = Path(item)
        if p.is_dir():
            if args.start_path == ".":
                start_path = p
        else:
            file_paths.append(p)

    if file_paths and not any([args.project, args.lamella, args.stage, args.image]):
        records = [record_from_image_path(p) for p in file_paths]
        matching = [r for r in records if args.modality == "all" or r.modality == args.modality]
        return matching, start_path

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
        return [], start_path

    index = Scanner(start_path).build_index()
    records = list(
        iter_matching_records(
            index,
            project_selectors=args.project,
            lamella_selectors=args.lamella,
            stage_selectors=args.stage,
            image_selectors=args.image,
            modality=args.modality,
            state=state,
        )
    )
    return records, start_path


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

    # -- create-training-csv -----------------------------------------------
    create_training_csv = sub.add_parser(
        "create-training-csv",
        help="Export dataset image and label pairs to CSV files for ML model training",
        description="Filter project images/labels and export training and validation CSV files",
    )
    _add_scope_arguments(create_training_csv, default_modality="all", include_state=False)
    create_training_csv.add_argument(
        "--state",
        choices=["corrected", "selected", "deselected", "all"],
        required=True,
        help="Workflow state of images/labels to export (required: 'corrected', 'selected', 'deselected', or 'all')",
    )
    create_training_csv.add_argument(
        "-o", "--output-dir",
        default=".",
        help="Directory to save generated CSV files (default: current directory)",
    )
    create_training_csv.add_argument(
        "--output-prefix",
        default="dataset",
        help="Prefix for generated CSV filenames (default: 'dataset')",
    )
    create_training_csv.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Ratio of data assigned to training, with the rest set to validation (default: 0.8)",
    )
    create_training_csv.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/validation split (default: 42)",
    )

    # -- check-training-csv --------------------------------------------------
    check_training_csv = sub.add_parser(
        "check-training-csv",
        help="Check that image/label paths recorded in a training CSV still exist",
        description=(
            "Re-check every image/label path in a CSV exported by "
            "create-training-csv against the current filesystem state, and "
            "optionally write a corrected copy."
        ),
    )
    check_training_csv.add_argument("csv_path", help="Path to the CSV file to check")
    check_training_csv.add_argument(
        "--start-path", default=".",
        help="Scan directory the CSV paths are relative to (default: current directory)",
    )
    check_training_csv.add_argument(
        "--fix", action="store_true",
        help="Write a corrected CSV with updated paths for files that moved between states",
    )
    check_training_csv.add_argument(
        "-o", "--output", default=None,
        help="Output path for the fixed CSV (default: '<name>_checked.csv'); requires --fix",
    )
    check_training_csv.add_argument(
        "--in-place", action="store_true",
        help="Overwrite the original CSV instead of writing a new file; requires --fix",
    )
    check_training_csv.add_argument(
        "--drop-missing", action="store_true",
        help="With --fix, omit rows whose image or label cannot be found in any workflow state",
    )
    check_training_csv.add_argument(
        "--keep-missing", action="store_true",
        help="With --fix, keep rows with unresolved paths unchanged instead of aborting",
    )

    # -- compress-labels ------------------------------------------------------
    compress_labels_cmd = sub.add_parser(
        "compress-labels",
        help="Recompress on-disk label files to reduce storage size",
        description=(
            "Re-encode label TIFFs discovered by the scanner using a compact "
            "encoding, shrinking file size on disk without changing pixel "
            "dimensions or label values."
        ),
    )
    _add_scope_arguments(compress_labels_cmd, default_modality="all")
    compress_labels_cmd.add_argument(
        "--confirm-corrected", action="store_true",
        help="Allow recompressing labels that live in a corrected/ folder",
    )
    compress_labels_cmd.add_argument(
        "--dry-run", action="store_true",
        help="Preview compression savings without modifying any files",
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
        records, start_path = _resolve_records(args)
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

        records, start_path = _resolve_records(args)
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

    if args.command == "create-training-csv":
        from .operations.csv import export_training_csv

        records, start_path = _resolve_records(args)
        if not records:
            logger.info("No matching images found to export CSV")
            return 2

        output_dir = Path(args.output_dir)
        train_csv, val_csv = export_training_csv(
            records=records,
            start_path=start_path,
            output_dir=output_dir,
            prefix=args.output_prefix,
            train_ratio=args.split,
            seed=args.seed,
        )
        logger.info("Exported training CSV: %s", make_path_relative(train_csv, start_path))
        if val_csv:
            logger.info("Exported validation CSV: %s", make_path_relative(val_csv, start_path))
        return 0

    if args.command == "check-training-csv":
        from .operations.csv_check import check_csv_paths, summarize_results, write_fixed_csv

        if args.in_place and args.output:
            logger.error("--in-place and --output are mutually exclusive")
            return 2
        if args.drop_missing and args.keep_missing:
            logger.error("--drop-missing and --keep-missing are mutually exclusive")
            return 2
        if not args.fix and (args.output or args.in_place or args.drop_missing or args.keep_missing):
            logger.error("--output/--in-place/--drop-missing/--keep-missing require --fix")
            return 2

        start_path = Path(args.start_path)
        csv_path = Path(args.csv_path)
        results = check_csv_paths(csv_path, start_path)
        lines, problem_count = summarize_results(results, start_path)
        for line in lines:
            logger.info(line)

        if not args.fix:
            return 0 if problem_count == 0 else 1

        if args.in_place:
            output_path = csv_path
        elif args.output:
            output_path = Path(args.output)
        else:
            output_path = csv_path.with_name(f"{csv_path.stem}_checked{csv_path.suffix}")
        missing_policy = "drop" if args.drop_missing else "keep" if args.keep_missing else "abort"

        try:
            written_path = write_fixed_csv(results, start_path, output_path, missing_policy=missing_policy)
        except ValueError as exc:
            logger.error(str(exc))
            return 2
        logger.info("Wrote fixed CSV: %s", written_path)
        return 0

    if args.command == "compress-labels":
        from .operations.label_compression import compress_labels

        records, start_path = _resolve_records(args)
        if not records:
            logger.info("No matching images found to compress labels for")
            return 2

        try:
            results = compress_labels(
                records,
                allow_corrected=args.confirm_corrected,
                dry_run=args.dry_run,
            )
        except (PermissionError, RuntimeError) as exc:
            logger.error("Could not compress labels: %s", exc)
            return 2

        for result in results:
            rel = make_path_relative(result.label_path, start_path)
            if args.dry_run:
                logger.info("Would compress %s: %d -> %d bytes", rel, result.original_size, result.new_size)
            elif result.applied:
                logger.info("Compressed %s: %d -> %d bytes", rel, result.original_size, result.new_size)
            else:
                logger.info("No savings for %s (%d bytes), left unchanged", rel, result.original_size)

        applied_count = sum(1 for r in results if r.applied or args.dry_run and r.new_size < r.original_size)
        total_before = sum(r.original_size for r in results)
        total_after = sum(r.new_size if (r.applied or args.dry_run) else r.original_size for r in results)
        logger.info(
            "Total: %d label(s) checked, %d %s, %d -> %d bytes",
            len(results), applied_count, "would compress" if args.dry_run else "compressed",
            total_before, total_after,
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI runner
    raise SystemExit(main())
