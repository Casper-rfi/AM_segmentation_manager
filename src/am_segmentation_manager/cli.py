"""Command line interface for common operations.

The CLI is intentionally lightweight and uses argparse so users can run
``python -m am_segmentation_manager.cli --help`` to see available commands and
``python -m am_segmentation_manager.cli <command> --help`` to get command-specific
usage information.

After installing the package, the ``am-seg`` entry point is also available.
"""
from __future__ import annotations

import argparse
import faulthandler
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from . import __version__
from .aggregate import (
    BREAKDOWNS,
    NEEDS_LABEL_STATES,
    NEEDS_MEASUREMENTS,
    NEEDS_METADATA,
)
from .filtering import (
    METADATA_FILTERS,
    NO_TASK,
    filter_by_metadata,
    filter_property,
    iter_matching_records,
)
from .history import (
    AM_HISTORY_FILENAME,
    CONFIG_DIRECTORY,
    KINDS,
    USER_ENV_VAR,
    Event,
    Recorder,
    containing_project,
    events_for,
    read_log,
)
from .io.path import make_path_relative
from .layout import SERIES
from .metadata import (
    DESCRIPTIVE_FIELDS,
    TODO_MARKER,
    VALUE_SEPARATOR,
    field_named,
)
from .operations.archive import (
    apply_prune,
    archive_root,
    archived_versions,
    plan_prune,
    total_size,
)
from .operations.export_record import build_export_record, write_export_record
from .operations.init_history import apply_baseline, plan_baseline
from .operations.label_measure import DEFAULT_SPECK_PIXELS, PREDICTION_PIXEL_BLOCK
from .operations.predictions import convert_record_predictions, plan_conversions
from .reporting import format_bytes, print_project_summary, print_survey
from .scanner import Scanner
from .survey import DEFAULT_MAX_DEPTH as SURVEY_MAX_DEPTH
from .survey import survey_collection

logger = logging.getLogger(__name__)

#: The open log file, held for the life of the process. ``faulthandler`` keeps
#: only the file descriptor, so a file closed when the function that opened it
#: returned would send the crash dump nowhere.
_LOG_FILE: TextIO | None = None

#: How a line in the log file is written. Richer than the stdout format, which
#: stays bare because that is the command's output: a crash report needs to say
#: when something happened and on which thread, and Qt's worker threads are
#: foreign to Python, so they appear under their ``Dummy-N`` names rather than
#: the object names Qt gives them.
LOG_FILE_FORMAT = "%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s"

#: Where the window logs when nothing else was asked for. Under the user's home
#: rather than beside the data: a log is this package's own record of a session,
#: not part of any collection being reviewed.
DEFAULT_GUI_LOG = Path.home() / ".am_segmentation_manager" / "gui.log"

#: How large a rolled log may grow before the previous run's file is moved
#: aside. Only the window's own log is capped -- a path the user named is left
#: to accumulate, that being what naming it asks for.
LOG_FILE_MAX_BYTES = 5_000_000


def open_log_file(path: Path, *, max_bytes: int | None = None) -> logging.Handler:
    """Return a handler writing to *path*, with crash dumps going there too.

    The Python log and the native stack trace share one file on purpose: a
    stack trace alone does not say what the process was doing, and the log line
    before it usually does. Records are flushed as they are written and
    ``faulthandler`` writes to the descriptor directly, so the two interleave
    in the order they happened.

    Appends, so several runs accumulate rather than the last one erasing the
    crash before it.

    Args:
        max_bytes: Move the file to ``<name>.1`` once it reaches this size,
            replacing any previous ``.1``. Omit to let it grow without limit,
            which is right for a file the user asked for by name.

    Raises:
        OSError: If *path* cannot be opened for writing.
    """
    global _LOG_FILE

    path.parent.mkdir(parents=True, exist_ok=True)
    _roll_if_large(path, max_bytes)
    _LOG_FILE = path.open("a", buffering=1, encoding="utf-8")
    faulthandler.enable(file=_LOG_FILE, all_threads=True)

    handler = logging.StreamHandler(_LOG_FILE)
    handler.setFormatter(logging.Formatter(LOG_FILE_FORMAT))
    return handler


def _roll_if_large(path: Path, max_bytes: int | None) -> None:
    """Move *path* aside when it has grown past *max_bytes*.

    One generation, not a series: what makes a log worth keeping is the crash
    at the end of it, and two files is enough to survive a session that starts
    fresh. A log that cannot be rolled is left to grow rather than refusing to
    open -- losing the session's log is worse than a large file.
    """
    if max_bytes is None:
        return
    try:
        if not path.exists() or path.stat().st_size < max_bytes:
            return
        path.replace(path.with_name(path.name + ".1"))
    except OSError as error:
        logger.debug("Could not roll %s aside: %s", path, error)


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------

#: What ``--series`` defaults to for a command that acts, exports or measures.
#:
#: Autolamella writes reference and alignment images alongside the Adaptive
#: Polishing ones, and indexing them must not change what an existing command
#: prints or writes. So a command that *reports* on the collection covers every
#: series -- ``inspect-layout`` and ``stats`` -- and every other command stays
#: on Adaptive Polishing until told otherwise. Widening is one flag.
DEFAULT_SERIES: str = "adaptive"

#: What ``--series`` defaults to for a command whose job is to report what is
#: there. Narrowing these silently is the bug this work exists to fix.
REPORTING_SERIES: str = "all"


def _add_scope_arguments(
    parser: argparse.ArgumentParser,
    default_modality: str | None = None,
    include_state: bool = True,
    default_series: str = DEFAULT_SERIES,
) -> None:
    """Add the common scope-selection arguments to *parser*.

    Set *include_state* to False when the caller registers its own
    ``--state`` argument (for example to make it required). Set
    *default_series* to :data:`REPORTING_SERIES` for a command that reports on
    the collection rather than acting on part of it.
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
    parser.add_argument(
        "--series", action="append", default=[],
        metavar="SERIES",
        help=(
            "What the images were taken for: "
            + ", ".join(SERIES)
            + ", or 'all'. Repeat or comma-separate to include several "
            f"(default: {default_series})"
        ),
    )
    parser.set_defaults(default_series=default_series)
    parser.add_argument(
        "--task", action="append", default=[], metavar="TASK",
        help=(
            "Milling task the images belong to, as the project's "
            "experiment.yaml names it (Polishing, Rough Milling, MillRough); "
            "separators and case are ignored. Stages roll up into their task, "
            f"so 'Rough 1' and 'Rough 2' are both Rough Milling. '{NO_TASK}' "
            "selects the images naming no task, 'all' every image. Repeat or "
            "comma-separate to include several (default: every task)"
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



def _format_duration(seconds: float) -> str:
    """Return a rough human-readable duration.

    Rough on purpose: this reports how long a run of thousands of images has
    left, where minutes are the unit anyone acts on.
    """
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _add_metadata_filters(parser: argparse.ArgumentParser) -> None:
    """Add the per-image metadata filters to *parser*.

    These read image headers, so they are applied after the structural
    selectors: only what those keep is ever opened.
    """
    for name in sorted(METADATA_FILTERS):
        parser.add_argument(
            f"--{name}", action="append", default=None, metavar="VALUE",
            help=(
                f"Keep only images whose {filter_property(name)} matches; "
                f"repeatable for alternatives"
            ),
        )


def _metadata_selectors(args: argparse.Namespace) -> dict[str, list[str]]:
    """Return the metadata filters the user asked for, if any."""
    return {
        name: values
        for name in METADATA_FILTERS
        if (values := getattr(args, name.replace("-", "_"), None))
    }


def _resolve_records(args: argparse.Namespace) -> tuple[list, Path]:
    """Resolve target records and scan start_path from CLI arguments.

    Returns a tuple of `(records, start_path)`.
    """
    records, start_path, _projects = _resolve_scope(args)
    return records, start_path


def _resolve_scope(args: argparse.Namespace) -> tuple[list, Path, list]:
    """Resolve the records, the scan root, and the projects they came from.

    The projects are what a command needs when it acts on a *dataset* rather
    than on images -- copying one out takes the files describing it along.
    Empty where explicit image paths were given, there being no scan to take
    them from.

    Returns a tuple of `(records, start_path, projects)`.
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
        return matching, start_path, []

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
        return [], start_path, []

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
            series_selectors=_series_selectors(args),
            task_selectors=getattr(args, "task", None) or None,
        )
    )
    return records, start_path, index.all_projects()


def _series_selectors(args: argparse.Namespace) -> list[str]:
    """Return the series to match, falling back to the command's default.

    A command declares its own default through :func:`_add_scope_arguments`,
    so which series a command covers when nobody says is decided where the
    command is registered rather than here.
    """
    return getattr(args, "series", None) or [
        getattr(args, "default_series", DEFAULT_SERIES)
    ]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the parser for every subcommand.

    Kept separate from :func:`main` so the parser can be inspected without
    running a command, which is what lets the help text be tested.
    """
    parser = argparse.ArgumentParser(
        "am_segmentation_manager",
        description="Manage Autolamella segmentation projects",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--log-file", default=None, metavar="PATH",
        help=(
            "Also write the log to this file, and write a native stack trace "
            "there if the process crashes outright. Appends, so runs "
            f"accumulate. 'gui' already does this in {DEFAULT_GUI_LOG}; name a "
            "path here to send one session somewhere of its own"
        ),
    )
    parser.add_argument(
        "--user", default=None, metavar="NAME",
        help=(
            "Record changes under this name. Otherwise taken from "
            f"{USER_ENV_VAR}, then from {CONFIG_DIRECTORY / 'user.yaml'}, then "
            "from the account running the command"
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"am-seg {__version__}",
        help="Print the installed version and exit",
    )
    # Subcommands are registered in workflow order, because argparse lists them
    # in registration order and that listing is how the commands are found.
    sub = parser.add_subparsers(dest="command")

    # === Adding data to the collection =======================================

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
        help=(
            "Write a template into every project found under the directory, and "
            "into any directory that holds images but is not a project yet"
        ),
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

    # -- show-metadata / edit-metadata ---------------------------------------
    show_metadata = sub.add_parser(
        "show-metadata",
        help="Show each project's resolved metadata, and where each value came from",
        description=(
            "Print the descriptive metadata in effect for each project, with the "
            "file that supplied each value. Values in AM_metadata.yaml override "
            "experiment.yaml per key, so a surprising value is easiest to "
            "explain by seeing which file it came from. Read-only."
        ),
    )
    show_metadata.add_argument("start_path", nargs="?", default=".", help="Directory to scan")
    show_metadata.add_argument(
        "--project", action="append", default=None,
        help="Project name, relative path or scan number; repeatable. Default: all",
    )

    edit_metadata = sub.add_parser(
        "edit-metadata",
        help="Set or clear fields in a project's AM_metadata.yaml",
        description=(
            "Edit the descriptive metadata of one or more projects. Only "
            "AM_metadata.yaml is written; experiment.yaml belongs to Autolamella "
            "and is never touched. The file is edited in place, so comments, key "
            "order and any keys this package does not model are preserved. "
            "Nothing is written unless every requested edit is valid."
        ),
    )
    edit_metadata.add_argument("start_path", nargs="?", default=".", help="Directory to scan")
    edit_metadata.add_argument(
        "--project", action="append", default=None,
        help="Project name, relative path or scan number; repeatable",
    )
    edit_metadata.add_argument(
        "--all", action="store_true",
        help=(
            "Apply to every project found. Values are copied into each file "
            "rather than resolved from a shared one, because nothing is "
            "inherited from parent directories"
        ),
    )
    edit_metadata.add_argument(
        "--set", action="append", default=[], metavar="FIELD=VALUE",
        help=(
            f"Set a field; repeatable. Fields: {', '.join(DESCRIPTIVE_FIELDS)}. "
            f"A field holding several values takes them separated by "
            f"'{VALUE_SEPARATOR.strip()}', or one --set apiece"
        ),
    )
    edit_metadata.add_argument(
        "--clear", action="append", default=[], metavar="FIELD",
        help=(
            "Remove a field from AM_metadata.yaml, so experiment.yaml's value "
            "applies again; repeatable"
        ),
    )
    edit_metadata.add_argument(
        "--todo", action="append", default=[], metavar="FIELD",
        help=(
            "Mark a field as outstanding work. Different from --clear: an "
            "unfilled placeholder is reported by inspect-layout"
        ),
    )
    edit_metadata.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without writing anything",
    )

    # -- list-projects / scan -----------------------------------------------
    list_projects = sub.add_parser(
        "list-projects",
        aliases=["scan"],
        help="Scan a directory and list discovered projects",
        description="Recursively scan for Autolamella projects and print a project summary",
    )
    list_projects.add_argument("start_path", nargs="?", default=".", help="Directory to start scanning from")

    # === Curating data and preparing labels ==================================

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
    # -- predict -------------------------------------------------------------
    predict_cmd = sub.add_parser(
        "predict",
        help="Create predictions for selected images with a trained model",
        description=(
            "Run a segmentation model over the selected images and write the "
            "result to the predictions directory each image set resolves to, "
            "under am_seg/ -- the only prediction directory this package "
            "writes, so a run can never touch the segmentation pipeline's own "
            "raw/ or clean/ output. One prediction is made per label group, "
            "from the group's primary imaging condition, because the other "
            "conditions of a scene share that one label. Predictions are "
            "written at the model's resolution; 'convert-predictions' resizes "
            "them to the image when it makes the label. Needs the inference "
            "dependencies: see scripts/setup_ml.sh."
        ),
    )
    _add_scope_arguments(predict_cmd, default_modality="SEM")
    predict_cmd.add_argument(
        "--model", required=True, metavar="PATH",
        help=(
            "Trained model weights (.pth). Settings that the weights do not "
            "record are read from <PATH>.yaml beside them, if it exists"
        ),
    )
    predict_cmd.add_argument(
        "--device", default="auto",
        help=(
            "Torch device: 'auto' for the fastest available (default), or "
            "'cpu', 'mps', 'cuda:0'. CPU always works and is several times slower"
        ),
    )
    predict_cmd.add_argument(
        "--dry-run", action="store_true",
        help="List the predictions that would be written without loading the model",
    )
    predict_cmd.add_argument(
        "--overwrite", action="store_true",
        help=(
            "Predict again even where this same model already has. Without it a "
            "re-run costs only what an interrupted one did not reach"
        ),
    )

    # -- convert-predictions -------------------------------------------------
    convert_predictions = sub.add_parser(
        "convert-predictions",
        help="Convert predictions into labels for explicitly selected images",
        description=(
            "Write a label file for each selected image from its prediction, "
            "resizing by nearest neighbour where the prediction and the image "
            "have different pixel dimensions. Labels are written to the label "
            "directory the image set resolves to, and an existing label is "
            "kept unless --overwrite is given. Only images explicitly selected by the scope "
            "arguments are converted, because the labels this produces are "
            "pseudo-labels awaiting proofreading, not finished annotations. "
            "Predictions are read from predictions/raw/, or from an undivided "
            "predictions/ directory. The pipeline's post-processed "
            "predictions/clean/ output is never converted: its intensity "
            "values no longer correspond to the segmentation classes."
        ),
    )
    _add_scope_arguments(convert_predictions)
    convert_predictions.add_argument(
        "--dry-run", action="store_true",
        help="List the labels that would be written without writing any",
    )
    convert_predictions.add_argument(
        "--overwrite", action="store_true",
        help=(
            "Replace labels that already exist. A label whose group is "
            "corrected is refused even with this flag; set that group back to "
            "selected first if one really is to be re-converted"
        ),
    )

    # -- deselect / select / correct ----------------------------------------
    for name, help_text in (
        ("deselect", "Declare label groups to be in the deselected state"),
        ("select", "Declare label groups to be in the selected state"),
        ("correct", "Declare label groups to be in the corrected state"),
    ):
        cmd = sub.add_parser(
            name,
            help=help_text,
            description=(
                f"{help_text}. Nothing on disk moves: the state is written "
                "into the stage's AM_state.yaml. Images sharing a label change "
                "together, the label being the unit of proofreading, so the "
                "change applies to every image of the same scene. Use --dry-run "
                "to see exactly which groups would change first."
            ),
        )
        _add_scope_arguments(cmd)
        cmd.add_argument(
            "--confirm-corrected", action="store_true",
            help="Confirm taking label groups out of the corrected state",
        )
        cmd.add_argument(
            "--dry-run", action="store_true",
            help="Show the groups that would change without writing anything",
        )

    # -- find-specks ---------------------------------------------------------
    find_specks_cmd = sub.add_parser(
        "find-specks",
        help="Find labels holding components likely to be too small to be real",
        description=(
            "List every label containing a connected component of a few pixels, "
            "whatever class it is. These are the stray spots that reach a "
            "training set as annotation and teach a model to find nothing -- "
            "and, where the class is background, the holes of that size left "
            "inside an annotation. Read-only, and it reports rather than "
            "repairs: a small structure can be genuine, so which ones to "
            "remove is a judgement made by looking at the image. Open the "
            "flagged labels in the "
            "window to see them in place. Reads every label in scope, which is "
            "far slower than reading headers."
        ),
    )
    _add_scope_arguments(find_specks_cmd, default_modality="all")
    find_specks_cmd.add_argument(
        "--at-most", type=int, default=DEFAULT_SPECK_PIXELS, metavar="PIXELS",
        help=(
            f"Largest component still reported (default: {DEFAULT_SPECK_PIXELS}, "
            "which proofreading settled on and which flags some small "
            f"structures that are real). At {PREDICTION_PIXEL_BLOCK} it reports "
            "only what was painted by hand: a label converted from a prediction "
            "is upscaled from half resolution, so the smallest area one can "
            "hold is a 2x2 block"
        ),
    )
    find_specks_cmd.add_argument(
        "--config", default=None,
        help="Segmentation config file naming the classes",
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
        help="Allow recompressing labels whose group is declared corrected",
    )
    compress_labels_cmd.add_argument(
        "--dry-run", action="store_true",
        help="Preview compression savings without modifying any files",
    )

    # === Quantifying and exporting ===========================================

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

    # -- init-history -------------------------------------------------------
    init_history_cmd = sub.add_parser(
        "init-history",
        help="Start recording a collection's history, for the data already on disk",
        description=(
            "Write each project's first AM_history.jsonl and one baseline entry "
            "per label group that has a label or a state other than selected. "
            "Nothing records until this has run, so this is what a collection "
            "opts in with. Re-runnable: a second run adds only what is new. "
            "A baseline says what was *seen*, never when it was annotated -- "
            "that is not recoverable, the collection having been copied. "
            "Previews by default; pass --apply to write."
        ),
    )
    init_history_cmd.add_argument(
        "--start-path", default=".",
        help="Directory to scan for projects (default: current directory)",
    )
    init_history_cmd.add_argument(
        "--apply", action="store_true",
        help="Write the entries, rather than reporting what would be written",
    )

    # -- archive ------------------------------------------------------------
    archive_cmd = sub.add_parser(
        "archive",
        help="Report the label versions a project has kept, and what they cost",
        description=(
            "List what AM_archive holds. Point it at a project to see the "
            "versions kept per label and the disk they take, or at a label to "
            "see just that label's. Nothing is removed: this is what a pruning "
            "decision is made from."
        ),
    )
    archive_cmd.add_argument(
        "path", nargs="?", default=".",
        help="A project directory, or a label inside one (default: current directory)",
    )

    # -- prune-archive ------------------------------------------------------
    prune_cmd = sub.add_parser(
        "prune-archive",
        help="Remove older kept versions, down to the last few per label",
        description=(
            "Remove archived label versions, keeping the most recent --keep-last "
            "of each. Four are never removed whatever is asked: a label's oldest "
            "kept version, the one taken before a model wrote over it, the one "
            "taken before a merge, and any version the history does not account "
            "for. Previews by default; pass --apply to remove. This is the only "
            "command that deletes a label file."
        ),
    )
    prune_cmd.add_argument(
        "path", nargs="?", default=".", help="The project to prune",
    )
    prune_cmd.add_argument(
        "--keep-last", type=int, required=True, metavar="N",
        help="How many recent versions of each label to keep. Required: there is no default",
    )
    prune_cmd.add_argument(
        "--apply", action="store_true",
        help="Remove them, rather than reporting what would go",
    )

    # -- history ------------------------------------------------------------
    history_cmd = sub.add_parser(
        "history",
        help="Print what has been recorded about a project or one of its labels",
        description=(
            "Read a project's AM_history.jsonl and print what it records. Point "
            "it at a project directory, or at a label to see only that label's "
            "history. A log is a record of what am-seg was told to do, not a "
            "statement about what the files currently hold."
        ),
    )
    history_cmd.add_argument(
        "path", nargs="?", default=".",
        help="A project directory, or a file inside one (default: current directory)",
    )
    history_cmd.add_argument(
        "--target", default=None, metavar="PATH",
        help="Only entries about this path, relative to the project directory",
    )
    history_cmd.add_argument(
        "--kind", default=None, action="append", metavar="KIND",
        help=f"Only entries of this kind. Repeatable. One of: {', '.join(KINDS)}",
    )
    history_cmd.add_argument(
        "--author", default=None, metavar="NAME",
        help="Only entries recorded by this author",
    )
    history_cmd.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Print only the last N entries",
    )

    return parser


# ---------------------------------------------------------------------------
# The commands
#
# One function per subcommand, taking the parsed arguments and returning a
# process exit status. Each is reachable only through COMMANDS below, so a name
# that is registered on the parser and has no handler here is a command that
# silently prints the help -- which `test_every_subcommand_has_a_handler` is
# what stops.
# ---------------------------------------------------------------------------


def _cmd_list_projects(args: argparse.Namespace) -> int:
    """Handle ``am-seg list-projects``."""
    scanner = Scanner(Path(args.start_path))
    index = scanner.build_index()
    for project in index.all_projects():
        print_project_summary(project)
    return 0


def _cmd_inspect_layout(args: argparse.Namespace) -> int:
    """Handle ``am-seg inspect-layout``."""
    survey = survey_collection(
        Path(args.start_path),
        max_depth=args.max_depth,
        check_labels=not args.no_label_check,
    )
    print_survey(survey, detail=args.detail, min_severity=args.severity)
    # A non-zero exit lets the command be used as a check in a script,
    # without making warnings -- which are expected on legacy data -- fail.
    return 1 if survey.finding_counts()["error"] else 0


def _cmd_init_project(args: argparse.Namespace) -> int:
    """Handle ``am-seg init-project``."""
    from .discovery import describe_dataset_candidates, find_project_roots, project_markers
    from .operations.init_project import (
        init_projects,
        project_placement_problem,
        read_defaults,
    )

    start_path = Path(args.start_path)
    if args.recursive:
        existing = find_project_roots(start_path)
        # Datasets predating Autolamella have no marker, so project
        # discovery cannot see them -- which would leave the ones that most
        # need a marker out of the command that writes one.
        candidates = describe_dataset_candidates(start_path)
        # A candidate holding datasets of several layout eras is a folder
        # someone gathered them in, and marking it would make each of them
        # a lamella of one project. Which level is right is not decidable
        # here, so this reports it rather than guessing: name the datasets
        # inside it, or pick them out in the window.
        gathering = [item for item in candidates if item.holds_several_datasets]
        for item in gathering:
            logger.warning(
                "%s holds %d dataset(s) laid out %d different ways; skipped, "
                "because marking it would make each of them a lamella of one "
                "project. Run init-project on each dataset inside it instead.",
                make_path_relative(item.path, start_path),
                len(item.lamellae), len(item.stage_depths),
            )
        proposed = [item for item in candidates if not item.holds_several_datasets]
        for item in proposed:
            logger.info(
                "%s holds images but is not a project yet; a metadata file "
                "makes it one", make_path_relative(item.path, start_path),
            )
        roots = [*existing, *(item.path for item in proposed)]
        if not roots:
            logger.error("No projects or unindexed datasets found under %s", start_path)
            return 2
    else:
        problem = project_placement_problem(start_path)
        if problem is not None:
            logger.error(
                "%s Use --recursive to write a file into every project that "
                "can be identified unambiguously, or name each dataset inside "
                "it in turn.",
                problem,
            )
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


def _cmd_export_projects_csv(args: argparse.Namespace) -> int:
    """Handle ``am-seg export-projects-csv``."""
    from .operations.metadata_csv import export_project_metadata_csv

    start_path = Path(args.start_path)
    projects = Scanner(start_path).build_index().all_projects()
    if not projects:
        logger.warning("No projects found under %s", start_path)
    export_project_metadata_csv(projects, start_path, Path(args.output))
    return 0


def _cmd_export_metadata_csv(args: argparse.Namespace) -> int:
    """Handle ``am-seg export-metadata-csv``."""
    from .aggregate import project_lookup
    from .image_metadata import read_records_metadata
    from .operations.metadata_csv import export_image_metadata_csv

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
    from .aggregate import (
        count_by,
        cross_tabulate,
        project_lookup,
        resolve_breakdown,
    )
    from .reporting import format_counts, format_cross_tabulation

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
    from .aggregate import breakdowns_for
    from .config import load_segmentation_config

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
        from .image_metadata import read_records_metadata

        logger.info("Reading headers for %d image(s)", len(records))
        metadata = read_records_metadata(records)

    measurements: dict = {}
    if measuring:
        from .operations.label_measure import measure_records

        logger.info("Measuring labels for %d image(s)", len(records))
        measurements = measure_records(records, config, metadata)

    label_states: dict = {}
    if any(kind_of(name) == "label" for name in chosen):
        from .operations.label_io import label_state

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
        from .plots import plot_counts, plot_cross_tabulation

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
    from .aggregate import project_lookup, summarise_lamellae
    from .config import load_segmentation_config
    from .image_metadata import read_records_metadata
    from .operations.label_measure import measure_records
    from .operations.metadata_csv import (
        export_label_measurements_csv,
        export_lamella_quantification_csv,
    )
    from .reporting import format_label_quantification

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


def _cmd_find_specks(args: argparse.Namespace) -> int:
    """Handle ``am-seg find-specks``."""
    from .config import load_segmentation_config
    from .operations.csv import find_matching_label
    from .operations.label_measure import records_with_specks
    from .reporting import format_speck_report

    records, start_path = _resolve_records(args)
    if not records:
        logger.warning("No images matched")
        return 0

    if args.at_most < 1:
        logger.error("--at-most must be at least 1, got %d", args.at_most)
        return 2

    try:
        config = load_segmentation_config(
            explicit_path=Path(args.config) if args.config else None,
            start_path=start_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 2

    logger.info("Reading labels for %d image(s)", len(records))
    found = records_with_specks(records, at_most=args.at_most)
    # Counted from the records, not from what was found: a sweep that says
    # "nothing in 248 labels" is a different statement from "nothing".
    labels_read = len({
        label for label in (find_matching_label(record) for record in records)
        if label is not None
    })
    for line in format_speck_report(
        found, start_path, config, at_most=args.at_most, labels_read=labels_read
    ):
        print(line)
    return 0


def _cmd_metadata(args: argparse.Namespace) -> int:
    """Handle ``am-seg edit-metadata``."""
    from .filtering import matches_project
    from .metadata import DESCRIPTIVE_FIELDS as FIELDS
    from .operations.metadata_edit import (
        MetadataValidationError,
        update_project_metadata,
    )

    start_path = Path(args.start_path)
    projects = Scanner(start_path).build_index().all_projects()
    selectors = args.project
    if selectors is not None:
        projects = [p for p in projects if matches_project(p, selectors)]
    if not projects:
        logger.error("No projects matched under %s", start_path)
        return 2

    if args.command == "show-metadata":
        for project in projects:
            resolved = project.experiment_metadata
            print(f"{project.display_name}")
            if resolved is None:
                print("    (no metadata)")
                continue
            for field in FIELDS:
                value = getattr(resolved, field)
                source = resolved.sources.get(field)
                origin = f"  [{source.name}]" if source else ""
                if value is None:
                    state = "TODO" if field in resolved.todo_fields else "-"
                    print(f"    {field:14} {state}")
                else:
                    print(f"    {field:14} {value}{origin}")
        return 0

    if args.project is None and not args.all:
        logger.error("Choose projects with --project, or all of them with --all")
        return 2

    values: dict[str, str | None] = {}
    for item in args.set:
        if "=" not in item:
            logger.error("--set expects FIELD=VALUE, got %r", item)
            return 2
        field, value = item.split("=", 1)
        field, value = field.strip(), value.strip()
        # Repeating --set for a field that holds several values adds to it
        # rather than replacing what the earlier one asked for, so two
        # organisations need no quoting or separator on the command line.
        spec = field_named(field)
        if spec is not None and spec.multi and values.get(field):
            value = f"{values[field]}{VALUE_SEPARATOR}{value}"
        values[field] = value
    for field in args.clear:
        values[field.strip()] = None
    for field in args.todo:
        values[field.strip()] = TODO_MARKER
    if not values:
        logger.error("Nothing to do: pass --set, --clear or --todo")
        return 2

    # Validate against every project before writing to any of them, so a
    # bad field name cannot leave half the collection edited.
    try:
        planned = [
            update_project_metadata(
                project.path, values, dry_run=True, project_name=project.display_name
            )
            for project in projects
        ]
    except MetadataValidationError as exc:
        for problem in exc.problems:
            logger.error("%s", problem)
        return 2

    changed = [change for change in planned if not change.is_empty]
    for change in planned:
        for line in change.describe():
            print(line)
    if not changed:
        logger.info("Nothing to change across %d project(s)", len(projects))
        return 0
    if args.dry_run:
        logger.info(
            "Dry run: %d of %d project(s) would change", len(changed), len(projects)
        )
        return 0

    recorder = _recorder(args)
    for project in projects:
        update_project_metadata(
            project.path, values, project_name=project.display_name, recorder=recorder
        )
    logger.info("Updated %d of %d project(s)", len(changed), len(projects))
    return 0


def _cmd_gui(args: argparse.Namespace) -> int:
    """Handle ``am-seg gui``."""
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


def _cmd_predict(args: argparse.Namespace) -> int:
    """Handle ``am-seg predict``."""
    from .operations.predict import (
        PredictionSource,
        plan_predictions,
        run_predictions,
    )

    records, start_path = _resolve_records(args)
    if not records:
        logger.info("No matching images found to predict")
        return 2

    weights = Path(args.model)
    # The description is read before the model, so a typo in it costs
    # nothing and a mismatch is reported before anything heavy loads.
    try:
        from .ml.model import read_model_description, run_name_of

        description = read_model_description(weights)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    source = PredictionSource(name=weights.name, run_name=run_name_of(weights))

    plan = plan_predictions(records, source, overwrite=args.overwrite)
    if plan.is_refused:
        logger.error(
            "This selection holds %s images. A model is trained on one "
            "modality and nothing here can tell which, so predict one at a "
            "time: pass --modality SEM or --modality FIB",
            " and ".join(plan.modalities),
        )
        return 2

    if plan.represented:
        logger.info(
            "%d image(s) share a label with one being predicted and are "
            "covered by it", plan.represented,
        )
    if plan.up_to_date:
        logger.info(
            "%d prediction(s) were already made by this model and are kept; "
            "pass --overwrite to make them again", len(plan.up_to_date),
        )
    if args.dry_run:
        for item in plan.predictions:
            logger.info(
                "%s %s", item.describe(),
                make_path_relative(item.destination, start_path),
            )
        logger.info(
            "Would predict: %d %s image(s) (dry run, nothing written and no "
            "model loaded)", len(plan.predictions), plan.modality,
        )
        return 0
    if not plan.predictions:
        logger.info("Nothing to predict")
        return 0

    try:
        from .ml.environment import resolve_device
        from .ml.model import load_model

        device = resolve_device(args.device)
        logger.info("Loading %s onto %s", weights.name, device)
        loaded = load_model(weights, description, device=device)
    except (RuntimeError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 2

    total = len(plan.predictions)
    started = time.monotonic()

    def report(index: int, item) -> None:
        elapsed = time.monotonic() - started
        logger.info(
            "[%d/%d] %s (%.2fs each, %s remaining)",
            index, total,
            make_path_relative(item.destination, start_path),
            elapsed / index,
            _format_duration(elapsed / index * (total - index)),
        )

    try:
        written = run_predictions(plan, loaded.predict, source, on_progress=report)
    except KeyboardInterrupt:
        # Each prediction is written and recorded before the next starts, so
        # what finished is kept and the next run resumes from it.
        logger.warning("Interrupted. Re-run to continue where this stopped")
        return 130
    logger.info("Total predicted: %d %s image(s) in %s", written, plan.modality,
                _format_duration(time.monotonic() - started))
    return 0


def _cmd_convert_predictions(args: argparse.Namespace) -> int:
    """Handle ``am-seg convert-predictions``."""
    records, start_path = _resolve_records(args)
    if not records:
        logger.info("No matching images found to convert")
        return 2
    plan = plan_conversions(records, overwrite=args.overwrite)
    if args.dry_run:
        for item in plan.conversions:
            logger.info("%s %s", item.describe(), item.destination)
    for item in plan.refused:
        logger.warning("Refused %s: a proofread label is already there", item.destination)
    if plan.replaceable:
        logger.warning(
            "%d label(s) already exist and were kept; pass --overwrite to replace them",
            len(plan.replaceable),
        )
    if args.dry_run:
        logger.info("Would convert: %d (dry run, nothing written)", len(plan.conversions))
        return 0
    converted = len(convert_record_predictions(
        records, overwrite=args.overwrite, recorder=_recorder(args)
    ))
    logger.info("Total converted: %d", converted)
    return 0


def _cmd_state(args: argparse.Namespace) -> int:
    """Handle ``am-seg correct``."""
    from .operations.label_io import classify_labels
    from .operations.selection import StaleState, apply_plan, plan_transition

    records, start_path = _resolve_records(args)
    if not records:
        logger.info("No matching images found to %s", args.command)
        return 2

    state = {"deselect": "deselected", "select": "selected", "correct": "corrected"}[
        args.command
    ]
    try:
        plan = plan_transition(
            records, state, allow_leaving_corrected=args.confirm_corrected
        )
    except (FileNotFoundError, PermissionError, ValueError) as exc:
        logger.error("Could not %s matching images: %s", args.command, exc)
        return 2

    # A shared label makes a state change reach further than the selectors
    # asked for, which the user must be able to see.
    if plan.reaches_beyond_request:
        logger.info(
            "%d further image(s) share a label with your selection and "
            "change with it:", len(plan.added),
        )
        for record in plan.added:
            logger.info("  %s", make_path_relative(record.image_path, start_path))

    # Corrected is the claim that somebody proofread these labels, and a
    # training export trusts it. Reported rather than refused: the command
    # line is where a deliberate exception is made, and the GUI is where
    # the guard belongs. Read over the expanded set, since a shared label
    # is marked corrected by any image of its scene.
    if state == "corrected":
        readiness = classify_labels(plan.records)
        if readiness:
            logger.warning(
                "%s, which is not proofread work. Marking them corrected "
                "means a training export skips them:", readiness.summary(),
            )
            for record in readiness.records:
                logger.warning(
                    "  %s", make_path_relative(record.image_path, start_path)
                )

    if args.dry_run:
        for line in plan.describe():
            logger.info("Would declare %s", line)
        logger.info(
            "Dry run: %d group(s) would change, affecting %d image(s)",
            len(plan.changes), len(plan.records),
        )
        return 0

    # The plan that was previewed is the one carried out, rather than one
    # worked out again from the same arguments. Its freshness is checked
    # against the files as it is applied.
    try:
        changed = apply_plan(plan, recorder=_recorder(args))
    except (StaleState, OSError, ValueError) as exc:
        logger.error("Could not %s matching images: %s", args.command, exc)
        return 2
    logger.info("Total %s: %d", args.command, changed)
    return 0


def _cmd_create_training_csv(args: argparse.Namespace) -> int:
    """Handle ``am-seg create-training-csv``."""
    from .operations.csv import export_training_csv

    records, start_path = _resolve_records(args)
    if not records:
        logger.info("No matching images found to export CSV")
        return 2

    # Metadata filters run after the structural ones, so headers are read
    # only for images that already survived project/lamella/state selection.
    selectors = _metadata_selectors(args)
    if selectors:
        from .image_metadata import read_records_metadata

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
    from .operations.copy_export import (
        DestinationInsideSource,
        copy_export,
        plan_copy,
    )
    from .reporting import format_bytes

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
    _write_fork_point(args, plan, records, projects, start_path, report.stopped)
    return 0


def _write_fork_point(
    args: argparse.Namespace,
    plan,
    records,
    projects,
    start_path: Path,
    stopped: bool,
) -> None:
    """Record where the copy came from, so it can be merged back.

    Written after the copy rather than before it: a fork point naming a tree
    that was never finished would have a merge compare against an ancestor
    half of it does not have. A stopped run therefore gets none, and says so.
    """
    if stopped:
        logger.warning(
            "The copy was stopped, so no fork point was written. Re-run the "
            "export to finish it; a merge back needs one."
        )
        return

    record = build_export_record(
        _projects_copied(records, projects), start_path,
        scope=_describe_scope(args),
        with_archive=args.with_archive,
        user=getattr(args, "user", None),
    )
    path = write_export_record(plan.destination, record)
    logger.info("Wrote the fork point to %s", path)
    missing = record.without_history
    if missing:
        logger.warning(
            "%d project(s) keep no history, so a merge back has no common "
            "ancestor for them: %s. Run init-history in the collection first.",
            len(missing), ", ".join(missing[:3]),
        )


def _projects_copied(records, projects) -> list:
    """Return the projects the copy actually holds.

    A scan finds every project under the scan root; an export copies the ones
    its selection reached. A fork point naming a project that is not in the
    copy would have a merge look for an ancestor in a tree that does not exist.
    """
    from .operations.copy_export import projects_of

    copied = set(projects_of([record.image_path for record in records], projects))
    return [project for project in projects if project.path in copied]


#: The selectors that narrow an export, in the order they are asked for. Used
#: to write down what an export covered, so a merge can say what it is merging.
_SCOPE_SELECTORS: tuple[str, ...] = (
    "project", "lamella", "stage", "image", "variant", "series", "state", "modality",
)


def _describe_scope(args: argparse.Namespace) -> str:
    """Return what the export was asked for, in the words the caller used."""
    parts = []
    for name in _SCOPE_SELECTORS:
        value = getattr(args, name, None)
        if not value:
            continue
        chosen = value if isinstance(value, list) else [value]
        parts.extend(f"--{name} {item}" for item in chosen)
    if getattr(args, "paths", None):
        parts.extend(str(path) for path in args.paths)
    return " ".join(parts)


def _cmd_check_training_csv(args: argparse.Namespace) -> int:
    """Handle ``am-seg check-training-csv``."""
    from .operations.csv_check import (
        check_csv_paths,
        summarize_results,
        write_pruned_csv,
    )

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


def _cmd_compress_labels(args: argparse.Namespace) -> int:
    """Handle ``am-seg compress-labels``."""
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
            recorder=None if args.dry_run else _recorder(args),
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
        elif result.stale:
            logger.warning(
                "Left %s alone: it changed on disk while being recompressed", rel
            )
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
    stale = [r for r in results if r.stale]
    if stale:
        logger.warning(
            "%d label(s) changed on disk during the run and were left alone; "
            "re-run to compress them", len(stale),
        )
    return 0


#: Which function runs each subcommand. Aliases and the commands that
#: share a handler point at the same one, so the parser and this cannot
#: disagree about how many there are.
def _cmd_init_history(args: argparse.Namespace) -> int:
    """Handle ``am-seg init-history``."""
    start_path = Path(args.start_path)
    index = Scanner(start_path).build_index()
    projects = index.all_projects()
    if not projects:
        logger.error("No projects found under %s", start_path)
        return 2

    plan = plan_baseline(projects)
    for line in plan.describe():
        print(line)
    if not plan.entry_count and not plan.log_count:
        logger.info("Nothing to record across %d project(s)", len(projects))
        return 0

    if not args.apply:
        logger.info(
            "Would start %d log(s) and record %d group(s) across %d project(s). "
            "Re-run with --apply to write.",
            plan.log_count, plan.entry_count, len(projects),
        )
        return 0

    report = apply_baseline(plan, recorder=_recorder(args))
    logger.info(
        "Started %d log(s) and recorded %d group(s)",
        report.logs_started, report.entries_written,
    )
    for path in report.unreadable:
        logger.warning(
            "%s would not open; recorded as a group whose contents are unknown",
            make_path_relative(path, start_path),
        )
    return 0


def _cmd_archive(args: argparse.Namespace) -> int:
    """Handle ``am-seg archive``."""
    path = Path(args.path)
    project = containing_project(path)
    if project is None:
        logger.error("No %s at or above %s", AM_HISTORY_FILENAME, path)
        return 1

    versions = archived_versions(project)
    if path.is_file():
        target = make_path_relative(path, project).replace(os.sep, "/")
        versions = [version for version in versions if version.target == target]

    print(f"{archive_root(project)}")
    if not versions:
        print("No versions kept.")
        return 0

    by_target: dict[str, list] = {}
    for version in versions:
        by_target.setdefault(version.target, []).append(version)
    for target, kept in sorted(by_target.items()):
        print(f"{target}  ({len(kept)} version(s), {format_bytes(total_size(kept))})")
        for version in kept:
            when = version.event.at if version.event else "not recorded"
            who = version.event.by if version.event else "unknown"
            print(f"    {version.path.name}  {when}  {who}")

    print(
        f"\n{len(versions)} version(s) of {len(by_target)} label(s), "
        f"{format_bytes(total_size(versions))}"
    )
    unaccounted = [version for version in versions if version.event is None]
    if unaccounted:
        logger.warning(
            "%d version(s) the history does not account for; they are never pruned",
            len(unaccounted),
        )
    return 0


def _cmd_prune_archive(args: argparse.Namespace) -> int:
    """Handle ``am-seg prune-archive``."""
    project = containing_project(Path(args.path))
    if project is None:
        logger.error("No %s at or above %s", AM_HISTORY_FILENAME, args.path)
        return 1

    try:
        plan = plan_prune(project, keep_last=args.keep_last)
    except ValueError as error:
        logger.error("%s", error)
        return 2

    for version in plan.remove:
        print(f"{version.target}: {version.path.name} ({format_bytes(version.size)})")
    if not plan.remove:
        logger.info("Nothing to remove; %d version(s) kept", plan.kept)
        return 0

    if not args.apply:
        logger.info(
            "Would remove %d version(s), freeing %s, keeping %d. "
            "Re-run with --apply to remove them.",
            len(plan.remove), format_bytes(plan.bytes_freed), plan.kept,
        )
        return 0

    removed = apply_prune(plan)
    logger.info(
        "Removed %d version(s), freeing %s. The history is unchanged: what was "
        "recorded still says what happened.",
        removed, format_bytes(plan.bytes_freed),
    )
    return 0


def _recorder(args: argparse.Namespace) -> Recorder:
    """Return the recorder for this run.

    One per command, so everything a single invocation changes shares a batch.
    A project with no log is left without one, which is what lets recording be
    something a collection opts into rather than something that starts
    happening to it.
    """
    return Recorder.for_run(getattr(args, "user", None))


def _cmd_history(args: argparse.Namespace) -> int:
    """Handle ``am-seg history``."""
    path = Path(args.path)
    if not path.exists():
        logger.error("%s does not exist", path)
        return 1

    project = containing_project(path)
    if project is None:
        logger.error(
            "No %s at or above %s. Nothing has been recorded here yet.",
            AM_HISTORY_FILENAME, path,
        )
        return 1

    target = args.target
    if target is None and path.is_file():
        target = make_path_relative(path, project).replace(os.sep, "/")

    log = read_log(project)
    entries = events_for(log, target) if target else list(log.events)
    if args.kind:
        wanted = set(args.kind)
        entries = [event for event in entries if event.kind in wanted]
    if args.author:
        entries = [event for event in entries if event.by == args.author]
    if args.limit is not None and args.limit >= 0:
        entries = entries[-args.limit:] if args.limit else []

    print(f"{log.path}")
    if target:
        print(f"target: {target}")
    if not entries:
        print("Nothing recorded.")
    for event in entries:
        _print_history_entry(event)

    print(
        f"\n{len(entries)} of {len(log.events)} entr(ies)"
        + (f", {len(log.unrecognised)} written by a newer am-seg" if log.unrecognised else "")
        + (f", {len(log.damaged)} unusable" if log.damaged else "")
    )
    for line in log.damaged:
        logger.warning("line %s is unusable: %s", line.number, line.reason)
    return 0


#: History fields worth a column of their own when printing an entry. Anything
#: else the entry carries is printed after them, so a kind added later shows up
#: without this needing to be taught about it.
_HISTORY_LEAD_FIELDS: tuple[str, ...] = ("old", "new", "field", "changed_fraction")


def _print_history_entry(event: Event) -> None:
    """Print one recorded entry over a line, with its fields after it."""
    attribution = event.by if event.by_source != "os" else f"{event.by} (account)"
    print(f"{event.at}  {event.kind:<18} {attribution}")
    if event.target:
        print(f"    {event.target}")
    shown = [name for name in _HISTORY_LEAD_FIELDS if name in event.data]
    shown += sorted(set(event.data) - set(_HISTORY_LEAD_FIELDS))
    details = "  ".join(f"{name}={event.data[name]}" for name in shown)
    if details:
        print(f"    {details}")


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "list-projects": _cmd_list_projects,
    "scan": _cmd_list_projects,
    "inspect-layout": _cmd_inspect_layout,
    "init-project": _cmd_init_project,
    "export-projects-csv": _cmd_export_projects_csv,
    "export-metadata-csv": _cmd_export_metadata_csv,
    "stats": _cmd_stats,
    "quantify-labels": _cmd_quantify_labels,
    "find-specks": _cmd_find_specks,
    "edit-metadata": _cmd_metadata,
    "show-metadata": _cmd_metadata,
    "gui": _cmd_gui,
    "predict": _cmd_predict,
    "convert-predictions": _cmd_convert_predictions,
    "correct": _cmd_state,
    "deselect": _cmd_state,
    "select": _cmd_state,
    "create-training-csv": _cmd_create_training_csv,
    "export-files": _cmd_export_files,
    "check-training-csv": _cmd_check_training_csv,
    "compress-labels": _cmd_compress_labels,
    "init-history": _cmd_init_history,
    "history": _cmd_history,
    "archive": _cmd_archive,
    "prune-archive": _cmd_prune_archive,
}


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the appropriate operation.

    Takes *argv* so the suite can drive a command the way a shell would, which
    is how all but ``gui`` and ``predict`` are covered.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # -- configure logging --------------------------------------------------
    level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    # The window logs without being asked to. It is the one command whose
    # failures are not visible where it was started: it detaches from the
    # terminal on some platforms, and a crash that kills the process takes
    # anything stderr had not flushed with it. Every other command prints where
    # you can see it, and writes a file only when you name one.
    chosen = getattr(args, "log_file", None)
    log_file = Path(chosen) if chosen else None
    cap = None
    if log_file is None and args.command == "gui":
        log_file, cap = DEFAULT_GUI_LOG, LOG_FILE_MAX_BYTES

    log_failure: OSError | None = None
    if log_file is not None:
        try:
            handlers.append(open_log_file(log_file, max_bytes=cap))
        except OSError as error:
            # Never fatal. A command that will not run because it could not
            # open its own log is worse than one that runs without a log.
            log_failure, log_file = error, None

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=handlers,
        force=True,
    )
    if log_file is not None:
        logger.info("Logging to %s", log_file)
    if log_failure is not None:
        logger.warning("Could not open the log file: %s", log_failure)

    # -- dispatch -----------------------------------------------------------
    handler = COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":  # pragma: no cover - CLI runner
    raise SystemExit(main())
