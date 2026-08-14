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
from .reporting import print_project_summary, print_survey
from .scanner import Scanner
from .survey import DEFAULT_MAX_DEPTH as SURVEY_MAX_DEPTH
from .survey import survey_collection

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
    parser.add_argument(
        "--variant", action="append", default=[],
        help=(
            "Imaging-condition variant, from the image directory name "
            "(sem_5kv -> 5kv); 'base' means the unsuffixed directory and 'all' "
            "every condition. Defaults to every variant"
        ),
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
            variant_selectors=getattr(args, "variant", None) or None,
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

    # -- inspect-layout -----------------------------------------------------
    inspect_layout = sub.add_parser(
        "inspect-layout",
        help="Report how a directory tree is interpreted, without changing anything",
        description=(
            "Survey a collection of datasets and report the projects found, how "
            "their lamella and stage hierarchy was interpreted, which image sets "
            "exist, where each one's labels resolve to and whether they are "
            "shared, plus anything that does not fit the naming conventions. "
            "Read-only: nothing is created, moved or modified. Use it to survey "
            "older datasets, to check bulk filesystem work such as reorganising "
            "labels, and to debug how a directory is being read."
        ),
    )
    inspect_layout.add_argument(
        "start_path", nargs="?", default=".", help="Directory to survey (default: current directory)",
    )
    inspect_layout.add_argument(
        "--detail", action="store_true",
        help="Show the full lamella/stage/image-set tree instead of per-project totals",
    )
    inspect_layout.add_argument(
        "--max-depth", type=int, default=SURVEY_MAX_DEPTH,
        help=(
            "How deep to search for projects and stages "
            f"(default: {SURVEY_MAX_DEPTH}); raise it only for unusually nested trees"
        ),
    )
    inspect_layout.add_argument(
        "--severity", choices=["error", "warning", "info"], default="info",
        help="Lowest finding severity to report (default: info, meaning all)",
    )
    inspect_layout.add_argument(
        "--no-label-check", action="store_true",
        help=(
            "Skip reading labels to tell an annotation from an empty "
            "placeholder. The check costs about 2.5 ms a label; skipping it "
            "makes a survey of a very large collection structure-only"
        ),
    )

    # -- migrate-layout -----------------------------------------------------
    migrate_layout = sub.add_parser(
        "migrate-layout",
        help="Rename pre-convention label directories to the current naming",
        description=(
            "Rename 'labels' to 'sem_labels' and 'labels_fib' to 'fib_labels', "
            "the current convention of naming a label directory after the image "
            "directory it belongs to. Both names are read indefinitely, so this "
            "is optional; it is offered because newly written labels already use "
            "the current name, and a project otherwise ends up with both. "
            "Previews by default -- pass --apply to make the changes. Only "
            "directories are renamed: no file is read, copied or deleted."
        ),
    )
    migrate_layout.add_argument(
        "start_path", nargs="?", default=".", help="Directory to migrate (default: current directory)",
    )
    migrate_layout.add_argument(
        "--apply", action="store_true",
        help="Perform the renames. Without this, they are only listed",
    )

    # -- init-project -------------------------------------------------------
    init_project_cmd = sub.add_parser(
        "init-project",
        help="Write an AM_metadata.yaml template into a project",
        description=(
            "Create an AM_metadata.yaml template holding the descriptive "
            "metadata Autolamella does not record (sample, user, organisation "
            "and so on). Values that can be derived are filled in; the rest are "
            "written as TODO placeholders for you to complete. A dataset with no "
            "experiment.yaml also becomes indexable, because AM_metadata.yaml is "
            "itself a project marker. Existing files are never overwritten."
        ),
    )
    init_project_cmd.add_argument(
        "start_path", nargs="?", default=".",
        help="Project directory, or a directory to search with --recursive",
    )
    init_project_cmd.add_argument(
        "--recursive", action="store_true",
        help="Write a template into every project found under the directory",
    )
    init_project_cmd.add_argument(
        "--defaults", default=None,
        help=(
            "YAML file of field values shared by these projects (for example "
            "organisation and user), copied into each generated file. Fields "
            "that vary between projects, such as sample and description, are "
            "never defaulted and stay as placeholders"
        ),
    )
    init_project_cmd.add_argument(
        "--dry-run", action="store_true",
        help="Report which files would be written without creating anything",
    )

    # -- gui ----------------------------------------------------------------
    gui_cmd = sub.add_parser(
        "gui",
        help="Open the curation window to review images and set their state",
        description=(
            "Open a window for reviewing SEM and FIB images against their "
            "labels, and for setting images to selected, deselected or "
            "corrected. Requires the optional GUI dependencies."
        ),
    )
    gui_cmd.add_argument(
        "start_path", nargs="?", default=None,
        help="Directory to scan on startup (optional; can be chosen in the window)",
    )
    gui_cmd.add_argument(
        "--config", default=None,
        help="Segmentation config file defining class names and colours",
    )

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
        cmd = sub.add_parser(
            name,
            help=help_text,
            description=(
                f"{help_text}. Images sharing a label move together: the label "
                "is the unit of proofreading, so the change applies to every "
                "image of the same scene. Use --dry-run to see exactly which "
                "files would move first."
            ),
        )
        _add_scope_arguments(cmd)
        cmd.add_argument(
            "--confirm-corrected", action="store_true",
            help="Confirm moving artifacts out of the corrected state",
        )
        cmd.add_argument(
            "--dry-run", action="store_true",
            help="Show the files that would move without changing anything",
        )

    # -- repair-state -------------------------------------------------------
    repair_state = sub.add_parser(
        "repair-state",
        help="Move images to agree with the state of the label they share",
        description=(
            "Find images whose workflow state disagrees with the label they "
            "share, and move the images to match. The label is authoritative "
            "because it holds the proofreading work. Groups become inconsistent "
            "when files are moved outside this package, or when an operation is "
            "interrupted partway."
        ),
    )
    _add_scope_arguments(repair_state, default_modality="all")
    repair_state.add_argument(
        "--confirm-corrected", action="store_true",
        help="Confirm moving images out of a corrected folder to match their label",
    )
    repair_state.add_argument(
        "--dry-run", action="store_true",
        help="Show the files that would move without changing anything",
    )

    # -- create-training-csv -----------------------------------------------
    create_training_csv = sub.add_parser(
        "create-training-csv",
        help="Export dataset image and label pairs to CSV files for ML model training",
        description="Filter project images/labels and export training and validation CSV files",
    )
    _add_scope_arguments(create_training_csv, default_modality="all", include_state=False)
    create_training_csv.add_argument(
        "--conditions",
        choices=["primary", "all"],
        default="primary",
        help=(
            "How many imaging conditions of each scene to export. 'primary' "
            "(default) exports one image per scene; 'all' exports every "
            "condition, which are valid but near-identical training pairs "
            "sharing one label"
        ),
    )
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
    create_training_csv.add_argument(
        "--include-empty-labels",
        action="store_true",
        help=(
            "Export images whose label holds only background. Excluded by "
            "default: the segmentation software writes an empty label for every "
            "image it is pointed at, so exporting one pairs an image with a "
            "blank mask"
        ),
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

    if args.command == "inspect-layout":
        survey = survey_collection(
            Path(args.start_path),
            max_depth=args.max_depth,
            check_labels=not args.no_label_check,
        )
        print_survey(survey, detail=args.detail, min_severity=args.severity)
        # A non-zero exit lets the command be used as a check in a script,
        # without making warnings -- which are expected on legacy data -- fail.
        return 1 if survey.finding_counts()["error"] else 0

    if args.command == "migrate-layout":
        from .operations.migrate import (
            MigrationConflict,
            apply_label_migration,
            plan_label_migration,
        )

        start_path = Path(args.start_path)
        try:
            renames = plan_label_migration(start_path)
        except MigrationConflict as exc:
            for conflict in exc.conflicts:
                logger.error("%s", conflict)
            logger.error("Nothing was renamed.")
            return 2

        if not renames:
            logger.info("No pre-convention label directories found under %s", start_path)
            return 0

        for rename in renames:
            logger.info(
                "%s %s", "Renaming" if args.apply else "Would rename",
                rename.describe(start_path),
            )
        if not args.apply:
            logger.info(
                "%d directory rename(s) to make. Re-run with --apply to perform them.",
                len(renames),
            )
            return 0

        changed = apply_label_migration(renames)
        logger.info("Renamed %d label directory(ies)", changed)
        return 0

    if args.command == "init-project":
        from .discovery import find_project_roots, project_markers
        from .operations.init_project import init_projects, read_defaults

        start_path = Path(args.start_path)
        if args.recursive:
            roots = find_project_roots(start_path)
            if not roots:
                logger.error("No projects found under %s", start_path)
                return 2
        else:
            if not start_path.is_dir():
                logger.error("Not a directory: %s", start_path)
                return 2
            roots = [start_path]
            if not project_markers(start_path):
                logger.info(
                    "%s has no project marker yet; writing one makes it a project",
                    start_path,
                )

        try:
            defaults = read_defaults(Path(args.defaults)) if args.defaults else None
        except (FileNotFoundError, ValueError) as exc:
            logger.error("%s", exc)
            return 2

        results = init_projects(roots, defaults=defaults, dry_run=args.dry_run)
        for result in results:
            location = make_path_relative(result.path, start_path)
            if result.skipped_existing:
                logger.info("Kept existing %s", location)
            elif args.dry_run:
                logger.info("Would write %s (%d field(s) to fill in)", location, len(result.todo))
            else:
                logger.info("Wrote %s (%d field(s) to fill in)", location, len(result.todo))

        written = sum(1 for result in results if result.written)
        skipped = sum(1 for result in results if result.skipped_existing)
        remaining = sum(len(result.todo) for result in results if not result.skipped_existing)
        logger.info(
            "Total: %d project(s), %d %s, %d already had one, %d field(s) to fill in",
            len(results), written or (len(results) - skipped if args.dry_run else 0),
            "would be written" if args.dry_run else "written", skipped, remaining,
        )
        return 0

    if args.command == "gui":
        # Imported here so a missing Qt binding cannot break the other
        # commands, or `--help`.
        from .gui.app import run

        try:
            return run(
                start_path=Path(args.start_path) if args.start_path else None,
                config_path=Path(args.config) if args.config else None,
            )
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 2

    if args.command == "convert-predictions":
        records, start_path = _resolve_records(args)
        if not records:
            logger.info("No matching images found to convert")
            return 2
        converted = convert_record_predictions(records, prediction_kind=args.kind)
        logger.info("Total converted: %d", converted)
        return 0

    if args.command in {"deselect", "select", "correct"}:
        from .operations.selection import plan_transition, transition_records

        records, start_path = _resolve_records(args)
        if not records:
            logger.info("No matching images found to %s", args.command)
            return 2

        state = {"deselect": "deselected", "select": "selected", "correct": "corrected"}[
            args.command
        ]
        try:
            plan = plan_transition(
                records, state, allow_move_from_corrected=args.confirm_corrected
            )
        except (FileNotFoundError, FileExistsError, PermissionError, ValueError) as exc:
            logger.error("Could not %s matching images: %s", args.command, exc)
            return 2

        # A shared label makes a state change reach further than the selectors
        # asked for, which the user must be able to see.
        if plan.reaches_beyond_request:
            logger.info(
                "%d further image(s) share a label with your selection and will "
                "also be moved:", len(plan.added),
            )
            for record in plan.added:
                logger.info("  %s", make_path_relative(record.image_path, start_path))

        if args.dry_run:
            for line in plan.describe():
                logger.info("Would move %s", line)
            logger.info(
                "Dry run: %d file(s) would move, affecting %d image(s)",
                len(plan.moves), len(plan.records),
            )
            return 0

        try:
            changed = transition_records(
                records, state, allow_move_from_corrected=args.confirm_corrected
            )
        except (FileNotFoundError, FileExistsError, PermissionError, ValueError) as exc:
            logger.error("Could not %s matching images: %s", args.command, exc)
            return 2
        logger.info("Total %s: %d", args.command, changed)
        return 0

    if args.command == "repair-state":
        from .operations.selection import repair_group_states

        records, start_path = _resolve_records(args)
        if not records:
            logger.info("No matching images found to check")
            return 2

        try:
            repaired = repair_group_states(
                records,
                allow_move_from_corrected=args.confirm_corrected,
                dry_run=args.dry_run,
            )
        except (FileNotFoundError, FileExistsError, PermissionError, ValueError) as exc:
            logger.error("Could not repair state: %s", exc)
            return 2

        if not repaired:
            logger.info("Every image agrees with the label it shares; nothing to repair")
            return 0
        for group, moved in repaired:
            logger.info(
                "%s: %d image(s) moved to %s",
                make_path_relative(group.label_path, start_path) if group.label_path else "(no label)",
                moved, group.state(),
            )
        logger.info(
            "%s: %d group(s)",
            "Would repair" if args.dry_run else "Repaired", len(repaired),
        )
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
            include_empty_labels=args.include_empty_labels,
            conditions=args.conditions,
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
