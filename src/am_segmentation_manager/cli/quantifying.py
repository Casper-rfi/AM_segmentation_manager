"""Counting what the collection holds, and writing it out.

The statistics, the label measurements, and the CSVs -- per-project and
per-image metadata for a repository, file paths for training. These report, so
they cover every series rather than only Adaptive Polishing.
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from ..aggregate import (
    BALANCEABLE,
    BREAKDOWNS,
    NEEDS_LABEL_STATES,
    NEEDS_MEASUREMENTS,
    NEEDS_METADATA,
    SPLIT,
    SPLIT_BREAKDOWN,
    breakdowns_for,
    by_split,
    count_by,
    cross_tabulate,
    project_lookup,
    resolve_breakdown,
    split_balance,
    split_grouping,
    summarise_lamellae,
)
from ..config import load_segmentation_config
from ..filtering import filter_by_metadata
from ..image_metadata import read_records_metadata
from ..io.path import make_path_relative
from ..io.tables import write_csv
from ..layout import SERIES
from ..operations.copy_export import (
    DestinationInsideSource,
    copy_export,
    plan_copy,
    record_fork_point,
)
from ..operations.label_io import label_state
from ..operations.label_measure import measure_records
from ..operations.metadata_csv import (
    UnreadableMetadataError,
    describe_training_images,
    export_image_metadata_csv,
    export_label_measurements_csv,
    export_lamella_quantification_csv,
    export_project_metadata_csv,
)
from ..operations.training_export import (
    LeftoverFolds,
    TrainingExportRequest,
    divide_for_export,
    export_kfold_csv,
    export_training_csv,
    image_sides,
    images_path,
)
from ..operations.training_export_check import check_csv_paths, summarize_results, write_pruned_csv
from ..operations.training_split import (
    DEFAULT_KEEP_TOGETHER,
    DEFAULT_TRAIN_RATIO,
    KEEP_TOGETHER,
    LEFT_OUT,
    TRAINING,
    VALIDATION,
)
from ..plots import plot_counts, plot_cross_tabulation
from ..reporting import (
    counts_as_rows,
    cross_tabulation_as_rows,
    format_bytes,
    format_counts,
    format_cross_tabulation,
    format_label_quantification,
)
from ..scanner import Scanner
from .scope import (
    REPORTING_SERIES,
    _add_metadata_filters,
    _add_overwrite_argument,
    _add_scope_arguments,
    _describe_scope,
    _metadata_selectors,
    _outputs_refused,
    _projects_in_scope,
    _resolve_scope,
)

logger = logging.getLogger(__name__)


#: The heading ``am-seg --help`` files these under.
TITLE = "Quantify it, and export"


def add_arguments(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the commands that quantify and export."""


    # -- stats ---------------------------------------------------------------
    stats_cmd = sub.add_parser(
        "stats",
        help="Count images by one or two properties",
        description=(
            "Count the selected images by a property, such as microscope, voltage, sample or "
            "workflow state. Pass --by twice to cross-tabulate, for example corrected FIB images "
            "at 30 kV per instrument. Changes nothing. [header] breakdowns read each image header "
            "(under 1 ms per image); [opens labels] reads every label (tens of ms each); [measures "
            "labels] also measures their components. All counts are of images, so a crack is "
            "counted once per imaging condition sharing the label; use quantify-labels for counts "
            "per lamella."
        ),
    )
    _add_scope_arguments(
        stats_cmd, default_modality="all",
        default_series=REPORTING_SERIES, series_choices=SERIES,
    )
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
            "Draw the breakdown as a bar chart. The file suffix sets the format: .png or .svg. "
            "Needs the plots extra"
        ),
    )
    _add_overwrite_argument(stats_cmd)

    # -- quantify-labels -----------------------------------------------------
    quantify = sub.add_parser(
        "quantify-labels",
        help="Measure what the labels contain, in physical units",
        description=(
            "Measure the area of each class and the cracks in every selected label, in um^2. "
            "Totals are per lamella, since a crack seen in three frames is one crack. Only "
            "corrected labels are measured by default, because predicted labels measure the model, "
            "not the sample; use --state to include others. Empty labels are counted separately. "
            "Changes nothing, but reads every label, which is much slower than reading headers."
        ),
    )
    _add_scope_arguments(quantify, default_modality="all", include_state=False)
    quantify.add_argument(
        "--state",
        choices=["corrected", "selected", "deselected", "all"],
        default="corrected",
        help=(
            "Workflow state to measure (default: 'corrected'). Including other states mixes in "
            "prediction-based labels"
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
            "Write one CSV row per measured image; images without a label are left out. A shared "
            "label appears once per imaging condition, so remove duplicates on the 'label' column "
            "before adding up"
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
    _add_overwrite_argument(quantify)

    # -- metadata exports ----------------------------------------------------
    export_metadata = sub.add_parser(
        "export-metadata-csv",
        help="Write the acquisition and project metadata of the selected images",
        description=(
            "Export the selected images with all known information: location, workflow and label "
            "state, acquisition settings from the image header, and project metadata, in one file. "
            "Changes nothing. With --modality SEM or FIB there is one row per image, as usually "
            "needed for a repository deposition; with both, one row per acquisition, with sem_ and "
            "fib_ columns as in create-training-csv."
        ),
    )
    # Counts and describes rather than acting, and an `other` image has a
    # header and a place in the collection like any other; a deposition wants
    # the grid overviews listed beside the lamellae they hold.
    _add_scope_arguments(export_metadata, default_modality="all", series_choices=SERIES)
    _add_metadata_filters(export_metadata)
    export_metadata.add_argument("-o", "--output", required=True, help="CSV file to write")
    export_metadata.add_argument(
        "--no-headers", action="store_true",
        help=(
            "Skip reading image headers, leaving the acquisition columns empty. Faster when only "
            "the structure is needed"
        ),
    )
    _add_overwrite_argument(export_metadata)

    export_projects = sub.add_parser(
        "export-projects-csv",
        help="Write one CSV row per project, with its metadata and counts",
        description=(
            "Export the metadata of each project, the file each value came from, the fields still "
            "marked TODO, and its numbers of lamellae, stages and images. Changes nothing."
        ),
    )
    export_projects.add_argument("start_path", nargs="?", default=".", help="Directory to scan")
    export_projects.add_argument("-o", "--output", required=True, help="CSV file to write")
    _add_overwrite_argument(export_projects)

    # -- export-files --------------------------------------------------------
    export_files = sub.add_parser(
        "export-files",
        help="Copy the selected images, labels and project files to a folder",
        description=(
            "Copy a selection to a new folder, keeping the folder structure: images, their labels, "
            "each stage's AM_state.yaml, and each project's AM_metadata.yaml, experiment.yaml and "
            "protocol.yaml. The result is a smaller collection that AM-seg can open, so "
            "proofreading can continue in the copy. Nothing is overwritten: identical files are "
            "skipped, so an interrupted export resumes, and a different file stops the whole run. "
            "Free space is checked first. The Export tab can also filter by acquisition settings."
        ),
    )
    # Copies each image and whatever label pairs with it, so a series with no
    # labels copies its images and nothing else -- which is what makes the
    # result the same collection, smaller.
    _add_scope_arguments(export_files, default_modality="all", series_choices=SERIES)
    export_files.add_argument(
        "-o", "--output", required=True, metavar="DIRECTORY",
        help="Folder to copy into. It must be outside the scanned folder",
    )
    export_files.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be copied, and how much, without writing anything",
    )
    export_files.add_argument(
        "--with-archive", action="store_true",
        help=(
            "Also copy the kept label versions of each project. Off by default: the history, which "
            "a merge needs, is always copied; the archive is only needed to compare with earlier "
            "versions of a label"
        ),
    )

    # -- create-training-csv -----------------------------------------------
    create_training_csv = sub.add_parser(
        "create-training-csv",
        help="Export image and label paths to CSV files for ML model training",
        description=(
            "Export training and validation CSV files of image and label paths. With --modality "
            "SEM or FIB each row is one image and its label; with both, each row is one "
            "acquisition (sem_image, sem_label, fib_image, fib_label), so a model can be trained "
            "on either beam or both. A cell is empty where a beam has no image or no usable label. "
            "--folds K writes K train/validation pairs for cross-validation. <prefix>_images.csv "
            "lists each image with its side and metadata, and <prefix>_manifest.yaml records how "
            "the files were made. Existing files are only replaced with --overwrite."
        ),
    )
    _add_scope_arguments(create_training_csv, default_modality="all", include_state=False)
    _add_metadata_filters(create_training_csv)
    create_training_csv.add_argument(
        "--conditions",
        choices=["primary", "all"],
        default="primary",
        help=(
            "How many imaging conditions per scene to export. 'primary' (default) exports one "
            "image per scene; 'all' exports every condition, giving near-identical pairs that "
            "share one label"
        ),
    )
    create_training_csv.add_argument(
        "--state",
        choices=["corrected", "selected", "deselected", "all"],
        nargs="+",
        required=True,
        help=(
            "Workflow state of the images to export (required: 'corrected', 'selected', "
            "'deselected' or 'all'). Name several to combine them, as '--state corrected selected'"
        ),
    )
    create_training_csv.add_argument(
        "--validate-on",
        choices=["corrected", "selected", "deselected"],
        default=None,
        help=(
            "Validate only on labels in this state, as in '--state corrected selected "
            "--validate-on corrected', which trains on proofread and predicted labels and "
            "validates on proofread ones. The validation set is the same as '--state corrected' "
            "alone with the same seed. Other rows are used for training, unless they share a "
            "--keep-together group with the validation set; those are left out and counted. Not "
            "with --folds"
        ),
    )
    create_training_csv.add_argument(
        "--balance",
        nargs="+",
        choices=sorted(BALANCEABLE),
        default=[],
        metavar="PROPERTY",
        help=(
            "Give the validation set its share of each value of these properties, as '--balance "
            "lamella-type'. Several properties are combined, which makes small groups, so use as "
            "few as possible. With --validation-from, new lamellae go to the side that is short of "
            "their value. Not with --folds"
        ),
    )
    create_training_csv.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show how the rows would be split and write nothing. With --by, also count the images "
            "on each side by a property. Not with --folds"
        ),
    )
    create_training_csv.add_argument(
        "--by",
        choices=sorted(
            name for name, entry in BREAKDOWNS.items()
            if entry.kind in ("record", "metadata", "project")
        ),
        default=None,
        metavar="PROPERTY",
        help=(
            "With --dry-run, count the images on each side of the split by this property, such as "
            "'lamella-type', 'microscope' or 'voltage'. Takes the same names as 'am-seg stats --by'"
        ),
    )
    create_training_csv.add_argument(
        "--validation-from",
        metavar="MANIFEST",
        default=None,
        help=(
            "Reuse the split of an earlier export of this collection, given by its "
            "<prefix>_manifest.yaml: lamellae (or --keep-together groups) it validated on are "
            "validated on again, including newly proofread images, those it trained on are trained "
            "on again, and only new ones are split by --split. A later model is then never scored "
            "on data an earlier one trained on. Not with --folds"
        ),
    )
    create_training_csv.add_argument(
        "-o", "--output-dir",
        default=".",
        help="Folder to write the CSV files to (default: current folder)",
    )
    create_training_csv.add_argument(
        "--output-prefix",
        default="dataset",
        help="Prefix for generated CSV filenames (default: 'dataset')",
    )
    division = create_training_csv.add_mutually_exclusive_group()
    division.add_argument(
        "--split",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help=(
            "Share of the data used for training; the rest is for validation "
            f"(default: {DEFAULT_TRAIN_RATIO})"
        ),
    )
    division.add_argument(
        "--folds",
        type=_fold_count,
        metavar="K",
        help=(
            "Write K train/validation pairs for cross-validation, <prefix>_fold1_train.csv and "
            "<prefix>_fold1_val.csv up to foldK, instead of one split. Every row is validated in "
            "exactly one fold, and --keep-together applies to each"
        ),
    )
    create_training_csv.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for a reproducible split (default: 42)",
    )
    create_training_csv.add_argument(
        "--keep-together",
        choices=[level for level, _ in KEEP_TOGETHER],
        default=DEFAULT_KEEP_TOGETHER,
        help=(
            "What is kept together in either training or validation: each acquisition, lamella "
            f"or Autolamella project (default: {DEFAULT_KEEP_TOGETHER}). Images of one lamella "
            "are near-duplicates, so a finer split inflates validation scores. Exports made "
            "before this option kept acquisitions together; use 'acquisition' with the same seed "
            "to reproduce one"
        ),
    )
    create_training_csv.add_argument(
        "--include-empty-labels",
        action="store_true",
        help=(
            "Also export images whose label contains only background. Excluded by default: the "
            "segmentation software writes an empty label for every image, so such a pair is "
            "usually not an annotation. Unreadable labels are always left out and counted"
        ),
    )
    create_training_csv.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace the CSV files and manifest of an earlier export with this prefix. Without it, "
            "nothing is written if a file would be replaced"
        ),
    )

    # -- check-training-csv --------------------------------------------------
    check_training_csv = sub.add_parser(
        "check-training-csv",
        help="Check that image/label paths recorded in a training CSV still exist",
        description=(
            "Check that every image and label path in a CSV from create-training-csv still exists, "
            "and optionally write a copy without the rows that do not. AM-seg never moves files, "
            "so a missing path means the file was deleted, renamed, or is on a drive that is not "
            "connected. Exits with 1 if any path is missing."
        ),
    )
    check_training_csv.add_argument("csv_path", help="Path to the CSV file to check")
    check_training_csv.add_argument(
        "--start-path", default=".",
        help="Scanned folder the CSV paths are relative to (default: current folder)",
    )
    check_training_csv.add_argument(
        "--prune", action="store_true",
        help=(
            "Write a copy without the rows whose image or label is missing, keeping the existing "
            "split"
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
    _add_overwrite_argument(check_training_csv)



# ---------------------------------------------------------------------------
# The commands
# ---------------------------------------------------------------------------

def _cmd_export_projects_csv(args: argparse.Namespace) -> int:
    """Handle ``am-seg export-projects-csv``."""

    if _outputs_refused([args.output], args.overwrite):
        return 2
    start_path = Path(args.start_path)
    projects = Scanner(start_path).build_index().all_projects()
    if not projects:
        logger.warning("No projects found under %s", start_path)
    try:
        export_project_metadata_csv(
            projects, start_path, Path(args.output), overwrite=args.overwrite
        )
    except UnreadableMetadataError as error:
        logger.error("Could not export: %s", error)
        return 2
    return 0


def _cmd_export_metadata_csv(args: argparse.Namespace) -> int:
    """Handle ``am-seg export-metadata-csv``."""

    if _outputs_refused([args.output], args.overwrite):
        return 2
    records, start_path, projects = _resolve_scope(args)
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

    lookup = project_lookup(_projects_in_scope(projects, start_path))
    try:
        export_image_metadata_csv(
            records, start_path, Path(args.output),
            include_headers=not args.no_headers, projects=lookup,
            # The same rule the training export and the window apply.
            paired=args.modality == "all",
            overwrite=args.overwrite,
        )
    except UnreadableMetadataError as error:
        logger.error("Could not export: %s", error)
        return 2
    if args.modality == "all":
        logger.info(
            "Both modalities selected: writing one row per acquisition, "
            "with each beam's own values under a sem_ or fib_ prefix. Pass "
            "--modality SEM or FIB for one row per image file."
        )
    return 0



def _cmd_stats(args: argparse.Namespace) -> int:
    """Handle ``am-seg stats``."""

    # NEEDS_METADATA and BREAKDOWNS are imported at module level; naming
    # them here too would shadow them for the whole of main(), including
    # the parser construction above.

    if _outputs_refused([args.output, args.plot], args.overwrite):
        return 2
    records, start_path, projects = _resolve_scope(args)
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

    lookup = project_lookup(_projects_in_scope(projects, start_path))
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
        rows: Sequence[Sequence[object]] = counts_as_rows(counts)
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
        rows = cross_tabulation_as_rows(table)
        header = [chosen[0], chosen[1], "images"]

    if args.plot:

        try:
            if len(keys) == 1:
                plot_counts(
                    counts, f"Images per {chosen[0]}", Path(args.plot),
                    xlabel=chosen[0], overwrite=args.overwrite,
                )
            else:
                plot_cross_tabulation(
                    table, chosen[0], chosen[1], Path(args.plot), overwrite=args.overwrite
                )
        except RuntimeError as exc:
            # The counts have already been printed, so a missing optional
            # dependency should not lose them.
            logger.error("%s", exc)
            return 2

    if args.output:
        write_csv(Path(args.output), header, rows, overwrite=args.overwrite)
        logger.info("Wrote %d row(s) to %s", len(rows), args.output)
    return 0


def _cmd_quantify_labels(args: argparse.Namespace) -> int:
    """Handle ``am-seg quantify-labels``."""

    if _outputs_refused([args.output, args.lamella_output], args.overwrite):
        return 2
    records, start_path, projects = _resolve_scope(args)
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

    lookup = project_lookup(_projects_in_scope(projects, start_path))
    summaries = summarise_lamellae(records, measurements, lookup)
    for line in format_label_quantification(
        summaries, config, detail=not args.no_detail
    ):
        print(line)

    if args.output:
        export_label_measurements_csv(
            records, measurements, config, start_path, Path(args.output), lookup,
            overwrite=args.overwrite,
        )
    if args.lamella_output:
        export_lamella_quantification_csv(
            summaries, config, Path(args.lamella_output), overwrite=args.overwrite
        )
    return 0



def _fold_count(text: str) -> int:
    """Parse ``--folds``, which divides nothing below 2."""
    try:
        folds = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if folds < 2:
        raise argparse.ArgumentTypeError("cross-validation needs at least 2 folds")
    return folds


def _cmd_create_training_csv(args: argparse.Namespace) -> int:
    """Handle ``am-seg create-training-csv``."""

    if args.folds and (
        args.validate_on or args.validation_from or args.dry_run or args.balance
    ):
        logger.error(
            "--validate-on, --validation-from, --balance and --dry-run are for one "
            "split; none can be given with --folds"
        )
        return 2
    if args.by and not args.dry_run:
        logger.error("--by counts what --dry-run previews; give both")
        return 2
    if args.validate_on is not None:
        if "all" not in args.state and args.validate_on not in args.state:
            logger.error(
                "--validate-on %s names a state --state %s leaves out, so there "
                "would be nothing to validate on",
                args.validate_on, " ".join(args.state),
            )
            return 2
    records, start_path, projects = _resolve_scope(args)
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
    scope = " ".join([
        _describe_scope(args),
        *(
            f"--{name} {value}"
            for name, values in selectors.items()
            for value in values
        ),
    ]).strip()
    request = TrainingExportRequest(
        start_path=start_path,
        prefix=args.output_prefix,
        conditions=args.conditions,
        # The same rule the window applies, so one selection exported either
        # way gives the same files.
        paired=args.modality == "all",
        include_empty_labels=args.include_empty_labels,
        grouping=split_grouping(args.keep_together, project_lookup(projects)),
        train_ratio=args.split,
        seed=args.seed,
        validate_on=args.validate_on,
        validation_from=Path(args.validation_from) if args.validation_from else None,
        balance=split_balance(
            args.balance,
            project_lookup(_projects_in_scope(projects, start_path)),
            read_records_metadata(records)
            if any(name in NEEDS_METADATA for name in args.balance) else None,
        ),
        scope=scope,
        user=getattr(args, "user", None),
        describe_images=describe_training_images(
            start_path, project_lookup(_projects_in_scope(projects, start_path))
        ),
    )
    if args.dry_run:
        try:
            _, _, training, validation, left_out = divide_for_export(records, request)
        except ValueError as error:
            logger.error("Could not export: %s", error)
            return 2
        print(
            f"Would write {len(training)} training and {len(validation)} validation "
            f"row(s), leaving {len(left_out)} out. Nothing was written"
        )
        if args.by:
            entry = BREAKDOWNS[args.by]
            key = resolve_breakdown(
                args.by,
                read_records_metadata(records) if entry.kind == "metadata" else None,
                project_lookup(_projects_in_scope(projects, start_path)),
            )
            sides = image_sides(
                [(TRAINING, training), (VALIDATION, validation), (LEFT_OUT, left_out)]
            )
            for line in format_cross_tabulation(
                cross_tabulate(records, key, by_split(sides), row_multi=entry.multi),
                args.by, SPLIT,
                row_order=entry.order, column_order=SPLIT_BREAKDOWN.order,
                multi=entry.multi,
            ):
                print(line)
        return 0
    written: Sequence[tuple[Path, Path | None]]
    try:
        if args.folds:
            written = export_kfold_csv(
                records, request, output_dir, args.folds, overwrite=args.overwrite
            )
        else:
            written = [
                export_training_csv(records, request, output_dir, overwrite=args.overwrite)
            ]
    except LeftoverFolds as error:
        logger.error("Could not export: %s", error)
        return 2
    except FileExistsError as error:
        logger.error(
            "Could not export: %s. Pass --overwrite to replace them, or choose "
            "another --output-prefix", error,
        )
        return 2
    except ValueError as error:
        logger.error("Could not export: %s", error)
        return 2
    if args.modality == "all":
        logger.info(
            "Both modalities selected: writing one row per acquisition, "
            "with SEM and FIB side by side. Pass --modality SEM or FIB for "
            "one row per image."
        )
    for train_csv, val_csv in written:
        logger.info("Exported training CSV: %s", make_path_relative(train_csv, start_path))
        if val_csv:
            logger.info("Exported validation CSV: %s", make_path_relative(val_csv, start_path))
    logger.info(
        "Each image's side and metadata: %s",
        make_path_relative(images_path(output_dir, args.output_prefix), start_path),
    )
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
    "project", "lamella", "stage", "image", "variant", "family", "series", "state",
    "modality",
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
    # --in-place names the file it replaces, which is asking to replace it.
    overwrite = args.overwrite or args.in_place
    output_path: Path | None = None
    if args.prune:
        if args.in_place:
            output_path = csv_path
        elif args.output:
            output_path = Path(args.output)
        else:
            output_path = csv_path.with_name(f"{csv_path.stem}_checked{csv_path.suffix}")
        if _outputs_refused([output_path], overwrite):
            return 2

    results = check_csv_paths(csv_path, start_path)
    lines, problem_count = summarize_results(results, start_path)
    for line in lines:
        logger.info(line)

    if output_path is None:
        return 0 if problem_count == 0 else 1

    written_path, dropped = write_pruned_csv(results, output_path, overwrite=overwrite)
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
