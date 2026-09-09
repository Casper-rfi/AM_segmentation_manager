"""Counting what the collection holds, and writing it out.

The statistics, the label measurements, and the CSVs -- per-project and
per-image metadata for a repository, file paths for training. These report, so
they cover every series rather than only Adaptive Polishing.
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

from ..aggregate import (
    BREAKDOWNS,
    NEEDS_LABEL_STATES,
    NEEDS_MEASUREMENTS,
    NEEDS_METADATA,
    breakdowns_for,
    count_by,
    cross_tabulate,
    project_lookup,
    resolve_breakdown,
    summarise_lamellae,
)
from ..config import load_segmentation_config
from ..filtering import filter_by_metadata
from ..image_metadata import read_records_metadata
from ..io.path import make_path_relative
from ..operations.copy_export import (
    DestinationInsideSource,
    copy_export,
    plan_copy,
    record_fork_point,
)
from ..operations.csv import export_training_csv
from ..operations.csv_check import check_csv_paths, summarize_results, write_pruned_csv
from ..operations.label_io import label_state
from ..operations.label_measure import measure_records
from ..operations.metadata_csv import (
    export_image_metadata_csv,
    export_label_measurements_csv,
    export_lamella_quantification_csv,
    export_project_metadata_csv,
)
from ..plots import plot_counts, plot_cross_tabulation
from ..reporting import (
    format_bytes,
    format_counts,
    format_cross_tabulation,
    format_label_quantification,
)
from ..scanner import Scanner
from .scope import (
    REPORTING_SERIES,
    _add_metadata_filters,
    _add_scope_arguments,
    _describe_scope,
    _metadata_selectors,
    _resolve_records,
    _resolve_scope,
)

logger = logging.getLogger(__name__)


def add_arguments(sub) -> None:
    """Register the commands that quantify and export."""


    # -- stats ---------------------------------------------------------------
    stats_cmd = sub.add_parser(
        "stats",
        help="Count images by one or two properties",
        description=(
            "Break the selected images down by a property -- microscope, "
            "accelerating voltage, sample, workflow state and so on. Pass --by "
            "twice for a cross-tabulation, which is usually the question worth "
            "asking: not how many Arctis images there are, but how many "
            "corrected FIB images at 30 kV are on each instrument. Read-only. "
            "Breakdowns marked [header] read each image's header, about half a "
            "millisecond per image; [opens labels] reads every label to see "
            "whether it holds anything, tens of milliseconds each; [measures "
            "labels] also labels its connected components. Every count here is a "
            "count of images, so the crack breakdowns count one crack once per "
            "imaging condition sharing the label -- use quantify-labels for "
            "the per-lamella view."
        ),
    )
    _add_scope_arguments(stats_cmd, default_modality="all", default_series=REPORTING_SERIES)
    stats_cmd.add_argument(
        "--by", action="append", default=[], metavar="PROPERTY",
        help=(
            "Property to count by; pass twice to cross-tabulate. Choices, as "
            "<name> = <what it counts>: "
            + ", ".join(
                f"{name} = {BREAKDOWNS[name].label}"
                f"{' [header]' if name in NEEDS_METADATA else ''}"
                f"{' [opens labels]' if name in NEEDS_LABEL_STATES else ''}"
                f"{' [measures labels]' if name in NEEDS_MEASUREMENTS else ''}"
                for name in sorted(BREAKDOWNS)
            )
            + ". Plus area-<class> [measures labels] for each class the "
            "segmentation config measures, which for the bundled one is "
            "area-gis, area-lamella and area-crack-hole"
        ),
    )
    stats_cmd.add_argument(
        "-o", "--output", default=None,
        help="Write the table to a CSV file as well as printing it",
    )
    stats_cmd.add_argument(
        "--plot", default=None, metavar="FILE",
        help=(
            "Draw the breakdown as a bar chart. The suffix chooses the format: "
            ".png to look at, .svg for a figure. Needs the optional matplotlib "
            "extra"
        ),
    )

    # -- quantify-labels -----------------------------------------------------
    quantify = sub.add_parser(
        "quantify-labels",
        help="Measure what the labels contain, in physical units",
        description=(
            "Measure the area of each segmentation class and the cracks in "
            "every selected label, in um^2. The headline is per lamella, "
            "because that is the level the question is asked at: a crack "
            "visible in three frames of a milling sequence is one crack. "
            "Defaults to corrected labels only -- a label derived from a "
            "prediction measures the model rather than the sample -- so pass "
            "--state to widen it. Empty labels are counted separately and "
            "excluded from the totals, since the segmentation software writes "
            "one for every image. Read-only, but it reads every label, which "
            "is far slower than reading headers."
        ),
    )
    _add_scope_arguments(quantify, default_modality="all", include_state=False)
    quantify.add_argument(
        "--state",
        choices=["corrected", "selected", "deselected", "all"],
        default="corrected",
        help=(
            "Workflow state to measure (default: 'corrected'). Widening this "
            "mixes proofread labels with prediction-derived ones"
        ),
    )
    _add_metadata_filters(quantify)
    quantify.add_argument(
        "--min-crack-area", type=float, default=None, metavar="UM2",
        help=(
            "Smallest connected component counted as a crack, in um^2. "
            "Overrides MEASUREMENT.min_crack_area_um2 in the segmentation config"
        ),
    )
    quantify.add_argument(
        "--config", default=None,
        help="Segmentation config file naming the classes and the crack threshold",
    )
    quantify.add_argument(
        "-o", "--output", default=None,
        help=(
            "Write one CSV row per measured image; images with no label are "
            "left out. A shared label appears once per imaging condition, so "
            "deduplicate on the 'label' column before totalling anything"
        ),
    )
    quantify.add_argument(
        "--lamella-output", default=None, metavar="FILE",
        help="Write one CSV row per lamella, matching the printed table",
    )
    quantify.add_argument(
        "--no-detail", action="store_true",
        help="Print only the totals, not the per-lamella table",
    )

    # -- metadata exports ----------------------------------------------------
    export_metadata = sub.add_parser(
        "export-metadata-csv",
        help="Write the acquisition and project metadata of the selected images",
        description=(
            "Export the selected images with everything known about each: where "
            "it sits, its workflow and label state, the acquisition settings "
            "read from its header, and its project's metadata. One file, no "
            "join needed. Read-only. With --modality SEM or FIB this is one row "
            "per image file, which is usually what a repository deposition "
            "wants; with both in play it is one row per acquisition, each "
            "beam's own values under a sem_ or fib_ prefix, matching what "
            "create-training-csv writes."
        ),
    )
    _add_scope_arguments(export_metadata, default_modality="all")
    _add_metadata_filters(export_metadata)
    export_metadata.add_argument("-o", "--output", required=True, help="CSV file to write")
    export_metadata.add_argument(
        "--no-headers", action="store_true",
        help=(
            "Skip reading image headers, leaving the acquisition columns empty. "
            "Faster for a purely structural listing"
        ),
    )

    export_projects = sub.add_parser(
        "export-projects-csv",
        help="Write one CSV row per project, with its metadata and counts",
        description=(
            "Export each project's descriptive metadata, which file supplied "
            "each value, the fields still marked TODO, and how many lamellae, "
            "stages and images it holds. Read-only."
        ),
    )
    export_projects.add_argument("start_path", nargs="?", default=".", help="Directory to scan")
    export_projects.add_argument("-o", "--output", required=True, help="CSV file to write")

    # -- export-files --------------------------------------------------------
    export_files = sub.add_parser(
        "export-files",
        help="Copy the selected images, their labels and the project files out",
        description=(
            "Copy a selection out into a folder of its own, mirroring the "
            "folder hierarchy it came from: each image, the label paired with "
            "it, each stage's AM_state.yaml, and each project's "
            "AM_metadata.yaml, experiment.yaml and protocol.yaml. The result "
            "is a smaller collection of the same shape, so am-seg reads it and "
            "proofreading can continue in the copy. Nothing is ever "
            "overwritten: a destination already holding the same file is "
            "skipped, so an interrupted export resumes, and one holding "
            "anything else refuses the whole run. Free space is checked before "
            "the first file is written. The Export tab narrows a selection by "
            "acquisition settings as well; the selectors here are the "
            "structural ones."
        ),
    )
    _add_scope_arguments(export_files, default_modality="all")
    export_files.add_argument(
        "-o", "--output", required=True, metavar="DIRECTORY",
        help="Directory to copy into. It must not be inside the folder scanned",
    )
    export_files.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be copied, and how much, without writing anything",
    )
    export_files.add_argument(
        "--with-archive", action="store_true",
        help=(
            "Take each project's kept label versions along. Off by default: "
            "the history always travels and is what a merge back needs, while "
            "the archive is a cache of recoverable pixels and can be partial. "
            "Turn it on to compare against what a label used to hold"
        ),
    )

    # -- create-training-csv -----------------------------------------------
    create_training_csv = sub.add_parser(
        "create-training-csv",
        help="Export image and label paths to CSV files for ML model training",
        description=(
            "Filter project images and labels and export training and "
            "validation CSV files. With --modality SEM or FIB each row is one "
            "image and its label; with both in play each row is one "
            "acquisition, as sem_image, sem_label, fib_image, fib_label -- so a "
            "model can be trained on either beam from one file, or on the two "
            "together. A cell is blank where that beam has no image, or no "
            "label worth pairing with one."
        ),
    )
    _add_scope_arguments(create_training_csv, default_modality="all", include_state=False)
    _add_metadata_filters(create_training_csv)
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
            "create-training-csv, and optionally write a copy without the rows "
            "that no longer resolve. Curation cannot move a file, so a path "
            "that does not resolve means the file is gone -- deleted, renamed, "
            "or on a drive that is not mounted. Exits 1 when any row has one."
        ),
    )
    check_training_csv.add_argument("csv_path", help="Path to the CSV file to check")
    check_training_csv.add_argument(
        "--start-path", default=".",
        help="Scan directory the CSV paths are relative to (default: current directory)",
    )
    check_training_csv.add_argument(
        "--prune", action="store_true",
        help=(
            "Write a copy without the rows whose image or label is missing, "
            "keeping the split the CSV already records"
        ),
    )
    check_training_csv.add_argument(
        "-o", "--output", default=None,
        help="Where to write it (default: '<name>_checked.csv'); requires --prune",
    )
    check_training_csv.add_argument(
        "--in-place", action="store_true",
        help="Overwrite the original CSV instead of writing a new file; requires --prune",
    )



# ---------------------------------------------------------------------------
# The commands
# ---------------------------------------------------------------------------

def _cmd_export_projects_csv(args: argparse.Namespace) -> int:
    """Handle ``am-seg export-projects-csv``."""

    start_path = Path(args.start_path)
    projects = Scanner(start_path).build_index().all_projects()
    if not projects:
        logger.warning("No projects found under %s", start_path)
    export_project_metadata_csv(projects, start_path, Path(args.output))
    return 0


def _cmd_export_metadata_csv(args: argparse.Namespace) -> int:
    """Handle ``am-seg export-metadata-csv``."""

    records, start_path = _resolve_records(args)
    if not records:
        logger.warning("No images matched")
        return 0

    selectors = _metadata_selectors(args)
    if selectors and args.no_headers:
        logger.error("Metadata filters need headers; drop --no-headers")
        return 2
    if selectors:
        metadata = read_records_metadata(records)
        records = filter_by_metadata(records, metadata, selectors)
        if not records:
            logger.warning("No images matched the metadata filters")
            return 0

    projects = project_lookup(Scanner(start_path).build_index().all_projects())
    export_image_metadata_csv(
        records, start_path, Path(args.output),
        include_headers=not args.no_headers, projects=projects,
        # The same rule the training export and the window apply.
        paired=args.modality == "all",
    )
    if args.modality == "all":
        logger.info(
            "Both modalities selected: writing one row per acquisition, "
            "with each beam's own values under a sem_ or fib_ prefix. Pass "
            "--modality SEM or FIB for one row per image file."
        )
    return 0



def _cmd_stats(args: argparse.Namespace) -> int:
    """Handle ``am-seg stats``."""
    import csv as csv_module

    # NEEDS_METADATA and BREAKDOWNS are imported at module level; naming
    # them here too would shadow them for the whole of main(), including
    # the parser construction above.

    records, start_path = _resolve_records(args)
    if not records:
        logger.warning("No images matched")
        return 0

    chosen = args.by or ["modality"]
    if len(chosen) > 2:
        logger.error("--by takes at most two properties, got %d", len(chosen))
        return 2

    # The config decides which classes have an area breakdown, so it is
    # resolved before the chosen names can be looked up at all. Loading it
    # opens one small YAML and scans nothing.

    config = load_segmentation_config(start_path=start_path)
    breakdowns = breakdowns_for(config)

    # Headers are only read when a chosen breakdown needs them, so a purely
    # structural question stays a filesystem walk. Measuring needs them
    # too, for the scale.
    def kind_of(name: str) -> str:
        entry = breakdowns.get(name)
        return entry.kind if entry is not None else ""

    measuring = any(kind_of(name) == "measurement" for name in chosen)
    metadata: dict = {}
    if measuring or any(kind_of(name) == "metadata" for name in chosen):

        logger.info("Reading headers for %d image(s)", len(records))
        metadata = read_records_metadata(records)

    measurements: dict = {}
    if measuring:

        logger.info("Measuring labels for %d image(s)", len(records))
        measurements = measure_records(records, config, metadata)

    label_states: dict = {}
    if any(kind_of(name) == "label" for name in chosen):

        logger.info("Reading labels for %d image(s)", len(records))
        label_states = {
            record.image_path: label_state(record) for record in records
        }

    lookup = project_lookup(Scanner(start_path).build_index().all_projects())
    try:
        keys = [
            resolve_breakdown(
                name, metadata, lookup, measurements, label_states, breakdowns
            )
            for name in chosen
        ]
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    if len(keys) == 1:
        entry = breakdowns[chosen[0]]
        counts = count_by(records, keys[0], entry.order, multi=entry.multi)
        for line in format_counts(
            counts,
            f"Images per {chosen[0]}",
            total=len(records),
            multi=entry.multi,
        ):
            print(line)
        rows = [(str(key), count) for key, count in counts.items()]
        header = [chosen[0], "images"]
    else:
        table = cross_tabulate(
            records, keys[0], keys[1],
            row_multi=breakdowns[chosen[0]].multi,
            column_multi=breakdowns[chosen[1]].multi,
        )
        for line in format_cross_tabulation(
            table, chosen[0], chosen[1],
            row_order=breakdowns[chosen[0]].order,
            column_order=breakdowns[chosen[1]].order,
            multi=breakdowns[chosen[0]].multi or breakdowns[chosen[1]].multi,
        ):
            print(line)
        rows = [(str(a), str(b), count) for (a, b), count in table.items()]
        header = [chosen[0], chosen[1], "images"]

    if args.plot:

        try:
            if len(keys) == 1:
                plot_counts(
                    counts, f"Images per {chosen[0]}", Path(args.plot),
                    xlabel=chosen[0],
                )
            else:
                plot_cross_tabulation(table, chosen[0], chosen[1], Path(args.plot))
        except RuntimeError as exc:
            # The counts have already been printed, so a missing optional
            # dependency should not lose them.
            logger.error("%s", exc)
            return 2

    if args.output:
        output = Path(args.output)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv_module.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        logger.info("Wrote %d row(s) to %s", len(rows), output)
    return 0


def _cmd_quantify_labels(args: argparse.Namespace) -> int:
    """Handle ``am-seg quantify-labels``."""

    records, start_path = _resolve_records(args)
    if not records:
        logger.warning("No images matched")
        return 0

    try:
        config = load_segmentation_config(
            explicit_path=Path(args.config) if args.config else None,
            start_path=start_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 2

    if config.crack_class is None:
        logger.error(
            "No crack class configured. Add MEASUREMENT.crack_class to %s, "
            "or name a class 'crack/hole'",
            config.source,
        )
        return 2

    # Headers first: they carry the scale, and the metadata filters work on
    # them. Only then is anything as expensive as a label opened.
    metadata = read_records_metadata(records)
    selectors = _metadata_selectors(args)
    if selectors:
        records = filter_by_metadata(records, metadata, selectors)
        if not records:
            logger.warning("No images matched the metadata filters")
            return 0

    logger.info("Measuring labels for %d image(s)", len(records))
    measurements = measure_records(
        records, config, metadata, min_crack_area_um2=args.min_crack_area
    )
    if not measurements:
        logger.warning("None of the matched images has a label to measure")
        return 0

    lookup = project_lookup(Scanner(start_path).build_index().all_projects())
    summaries = summarise_lamellae(records, measurements, lookup)
    for line in format_label_quantification(
        summaries, config, detail=not args.no_detail
    ):
        print(line)

    if args.output:
        export_label_measurements_csv(
            records, measurements, config, start_path, Path(args.output), lookup
        )
    if args.lamella_output:
        export_lamella_quantification_csv(
            summaries, config, Path(args.lamella_output)
        )
    return 0



def _cmd_create_training_csv(args: argparse.Namespace) -> int:
    """Handle ``am-seg create-training-csv``."""

    records, start_path = _resolve_records(args)
    if not records:
        logger.info("No matching images found to export CSV")
        return 2

    # Metadata filters run after the structural ones, so headers are read
    # only for images that already survived project/lamella/state selection.
    selectors = _metadata_selectors(args)
    if selectors:

        records = filter_by_metadata(records, read_records_metadata(records), selectors)
        if not records:
            logger.info("No images matched the metadata filters")
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
        # The same rule the window applies, so one selection exported
        # either way gives the same two files.
        paired=args.modality == "all",
    )
    if args.modality == "all":
        logger.info(
            "Both modalities selected: writing one row per acquisition, "
            "with SEM and FIB side by side. Pass --modality SEM or FIB for "
            "one row per image."
        )
    logger.info("Exported training CSV: %s", make_path_relative(train_csv, start_path))
    if val_csv:
        logger.info("Exported validation CSV: %s", make_path_relative(val_csv, start_path))
    return 0


def _cmd_export_files(args: argparse.Namespace) -> int:
    """Handle ``am-seg export-files``."""

    records, start_path, projects = _resolve_scope(args)
    if not records:
        logger.info("No matching images found to copy")
        return 2

    try:
        plan = plan_copy(
            records, Path(args.output), start_path,
            projects=projects, with_archive=args.with_archive,
        )
    except DestinationInsideSource as error:
        logger.error("Could not copy: %s", error)
        return 2

    for item in plan.items:
        logger.debug(
            "%s: %s -> %s", item.outcome, item.source, item.destination
        )
    logger.info(
        "%d file(s) to copy, %s; %d already there",
        len(plan.to_copy), format_bytes(plan.bytes_to_copy),
        len(plan.already_there),
    )
    if plan.conflicts:
        logger.error(
            "%d destination(s) already hold a different file, starting with "
            "%s. Nothing was copied; choose an empty directory.",
            len(plan.conflicts), plan.conflicts[0].destination,
        )
        return 2
    if plan.short_by:
        logger.error(
            "Not enough room: %s needed, %s free. Nothing was copied.",
            format_bytes(plan.bytes_to_copy), format_bytes(plan.free_space),
        )
        return 2
    if args.dry_run:
        logger.info("Dry run: nothing was copied")
        return 0
    if not plan.to_copy:
        logger.info("Nothing to copy: %s already holds it all", plan.destination)
        return 0

    report = copy_export(plan)
    logger.info(
        "Copied %d file(s), %s, into %s",
        len(report.copied), format_bytes(report.bytes_copied), plan.destination,
    )
    record = record_fork_point(
        plan.destination, records, projects, start_path,
        stopped=report.stopped,
        scope=_describe_scope(args),
        with_archive=args.with_archive,
        user=getattr(args, "user", None),
    )
    missing = record.without_history if record else ()
    if missing:
        logger.warning(
            "%d project(s) keep no history, so a merge back has no common "
            "ancestor for them: %s. Baseline them with init-history before "
            "exporting, a first change alone recording no state to fork from.",
            len(missing), ", ".join(missing[:3]),
        )
    return 0


#: The selectors that narrow an export, in the order they are asked for. Used
#: to write down what an export covered, so a merge can say what it is merging.
_SCOPE_SELECTORS: tuple[str, ...] = (
    "project", "lamella", "stage", "image", "variant", "series", "state", "modality",
)



def _cmd_check_training_csv(args: argparse.Namespace) -> int:
    """Handle ``am-seg check-training-csv``."""

    if args.in_place and args.output:
        logger.error("--in-place and --output are mutually exclusive")
        return 2
    if not args.prune and (args.output or args.in_place):
        logger.error("--output/--in-place require --prune")
        return 2

    start_path = Path(args.start_path)
    csv_path = Path(args.csv_path)
    results = check_csv_paths(csv_path, start_path)
    lines, problem_count = summarize_results(results, start_path)
    for line in lines:
        logger.info(line)

    if not args.prune:
        return 0 if problem_count == 0 else 1

    if args.in_place:
        output_path = csv_path
    elif args.output:
        output_path = Path(args.output)
    else:
        output_path = csv_path.with_name(f"{csv_path.stem}_checked{csv_path.suffix}")

    written_path, dropped = write_pruned_csv(results, output_path)
    logger.info("Wrote %s, dropping %d row(s)", written_path, dropped)
    return 0




#: What each of this group's subcommands runs. Registered above and
#: dispatched here, so the two cannot name different sets.
COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "export-projects-csv": _cmd_export_projects_csv,
    "export-metadata-csv": _cmd_export_metadata_csv,
    "stats": _cmd_stats,
    "quantify-labels": _cmd_quantify_labels,
    "create-training-csv": _cmd_create_training_csv,
    "export-files": _cmd_export_files,
    "check-training-csv": _cmd_check_training_csv,
}
