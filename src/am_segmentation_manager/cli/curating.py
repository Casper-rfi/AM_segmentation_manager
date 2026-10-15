"""Choosing what to label, and getting a label to start from.

The window, prediction and its conversion into labels, the three state changes,
the families of imaging conditions, and the two commands that look at what a
label holds. Everything here acts on
Adaptive Polishing unless told otherwise, for the reason
:data:`~am_segmentation_manager.cli.scope.DEFAULT_SERIES` gives.
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from ..config import load_segmentation_config
from ..families import StaleFamilies
from ..filtering import matches_lamella, matches_project
from ..io.path import make_path_relative
from ..layout import DEFAULT_PREDICTION_ORIGIN, PREDICTION_ORIGINS, other_prediction_origin
from ..models import projects_holding
from ..operations.families_edit import (
    FamiliesValidationError,
    apply_families_change,
    describe_lamella_sections,
    plan_families_change,
    summarise_families,
)
from ..operations.label_compression import compress_labels
from ..operations.label_groups import find_matching_label
from ..operations.label_io import classify_labels
from ..operations.label_measure import (
    DEFAULT_SPECK_PIXELS,
    PREDICTION_PIXEL_BLOCK,
    records_with_specks,
)
from ..operations.predict import (
    PlannedPrediction,
    PredictionSource,
    describe_conditions,
    plan_predictions,
    run_predictions,
)
from ..operations.predictions import (
    ConversionStopped,
    convert_record_predictions,
    plan_conversions,
    what_replacing_leaves,
)
from ..operations.selection import StaleState, apply_plan, plan_transition
from ..reporting import format_speck_report
from ..scanner import Scanner
from ..state import RecordState, StateFileError
from .scope import (
    _add_scope_arguments,
    _format_duration,
    _recorder,
    _resolve_records,
    _resolve_scope,
    _writing_recorder,
)

logger = logging.getLogger(__name__)


#: The heading ``am-seg --help`` files these under.
TITLE = "Curate it, and prepare labels"


def add_arguments(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the commands that curate data and prepare labels."""


    # -- gui ----------------------------------------------------------------
    gui_cmd = sub.add_parser(
        "gui",
        help="Open the curation window to review images and set their state",
        description=(
            "Open the window to review SEM and FIB images and their labels, and to set images to "
            "selected, deselected or corrected. Needs the gui extra."
        ),
    )
    gui_cmd.add_argument(
        "start_path", nargs="?", default=None,
        help="Folder to scan at startup (optional; can be chosen in the window)",
    )
    gui_cmd.add_argument(
        "--config", default=None,
        help="Segmentation config file with class names and colours",
    )
    gui_cmd.add_argument(
        "--viewer", action="store_true",
        help=(
            "Open the viewer: the Curate tab only, for browsing a collection without changing "
            "anything"
        ),
    )

    # -- predict -------------------------------------------------------------
    predict_cmd = sub.add_parser(
        "predict",
        help="Predict labels for the selected images with a trained model",
        description=(
            "Run a segmentation model over the selected images and write the predictions to "
            "predictions/am_seg/. The microscope's own raw/ and clean/ predictions are never "
            "touched. One prediction is made per label group, from its primary imaging condition. "
            "Predictions are written at the model's resolution; convert-predictions resizes them "
            "to the image. Needs the ml extra."
        ),
    )
    _add_scope_arguments(predict_cmd, default_modality="SEM")
    predict_cmd.add_argument(
        "--model", required=True, metavar="PATH",
        help=(
            "Trained model weights (.pth). The model generation is read from <PATH>.yaml beside "
            "the weights, as 'generation: \"1.0\"'; without that file, generation 1.0 is used, as "
            "on the microscope"
        ),
    )
    predict_cmd.add_argument(
        "--device", default="auto",
        help=(
            "Torch device: 'auto' (default) uses the NVIDIA GPU with the most free memory, else "
            "the fastest device; or 'cpu', 'mps', 'cuda:0'. GPU numbers as set by nvidia-smi. The "
            "CPU always works but is several times slower; AM_SEG_WORKERS limits its threads"
        ),
    )
    predict_cmd.add_argument(
        "--dry-run", action="store_true",
        help="List the predictions that would be written, without loading the model",
    )
    predict_cmd.add_argument(
        "--overwrite", action="store_true",
        help=(
            "Predict again where this model already has. Without it, a re-run only predicts what "
            "an interrupted run did not reach"
        ),
    )

    # -- convert-predictions -------------------------------------------------
    convert_predictions = sub.add_parser(
        "convert-predictions",
        help="Convert predictions into labels for the selected images",
        description=(
            "Write a label for each selected image from its prediction, resized to the image by "
            "nearest neighbour. Existing labels are kept unless --overwrite is given. Only "
            "explicitly selected images are converted, because the results are pseudo-labels that "
            "need proofreading. Predictions are read from predictions/am_seg/ (written by predict) "
            "and from the microscope's predictions/raw/ or predictions/. Where an image has both, "
            "--prefer chooses. Post-processed predictions/clean/ are never used."
        ),
    )
    _add_scope_arguments(convert_predictions)
    convert_predictions.add_argument(
        "--dry-run", action="store_true",
        help="List the labels that would be written, without writing any",
    )
    convert_predictions.add_argument(
        "--prefer", choices=PREDICTION_ORIGINS, default=DEFAULT_PREDICTION_ORIGIN,
        help=(
            "Which prediction to convert where an image has both: am-seg's or the microscope's "
            "(default: %(default)s). An image with only one converts that one"
        ),
    )
    convert_predictions.add_argument(
        "--overwrite", action="store_true",
        help=(
            "Replace existing labels. Labels marked corrected are never replaced, even with this "
            "flag; set the group back to selected first to convert it again"
        ),
    )

    # -- deselect / select / correct ----------------------------------------
    for name, help_text in (
        ("deselect", "Set label groups to the deselected state"),
        ("select", "Set label groups to the selected state"),
        ("correct", "Set label groups to the corrected state"),
    ):
        cmd = sub.add_parser(
            name,
            help=help_text,
            description=(
                f"{help_text}, written to the stage's AM_state.yaml; no files move. Images "
                "sharing a label change together. Use --dry-run to see which groups would change."
            ),
        )
        _add_scope_arguments(cmd)
        cmd.add_argument(
            "--confirm-corrected", action="store_true",
            help="Confirm changing groups that are marked corrected",
        )
        cmd.add_argument(
            "--dry-run", action="store_true",
            help="Show the groups that would change without writing anything",
        )

    # -- show-families / edit-families ---------------------------------------
    show_families = sub.add_parser(
        "show-families",
        help="Show each project's families of imaging conditions",
        description=(
            "Print the families of each project: the imaging conditions sharing a label, and how "
            "many stages hold each. * marks a condition placed by AM_families.yaml instead of by "
            "its folder name; the entries of a lamella itself are listed after. Changes nothing."
        ),
    )
    show_families.add_argument("start_path", nargs="?", default=".", help="Directory to scan")
    show_families.add_argument(
        "--project", action="append", default=None,
        help="Project name, relative path or scan number; repeatable. Default: all",
    )

    edit_families = sub.add_parser(
        "edit-families",
        help="Move imaging conditions between families in AM_families.yaml",
        description=(
            "Change which family a condition belongs to, and so which label it shares, without "
            "renaming folders. The change is checked in every stage first, and refused if a label "
            "or prediction would be lost or replaced. A condition joining a family shares its "
            "labels and state; one leaving starts without a label. With --lamella, the change "
            "applies to that lamella only. Shows what would change; nothing is written without "
            "--apply."
        ),
    )
    edit_families.add_argument("start_path", nargs="?", default=".", help="Directory to scan")
    edit_families.add_argument(
        "--project", action="append", default=None, required=True,
        help="Project name, relative path or scan number; repeatable",
    )
    edit_families.add_argument(
        "--set", action="append", default=[], metavar="CONDITION=FAMILY",
        help="Place a condition in a family, e.g. sem2kVb=sem-2kV; repeatable",
    )
    edit_families.add_argument(
        "--clear", action="append", default=[], metavar="CONDITION",
        help="Return a condition to the family given by its folder name; repeatable",
    )
    edit_families.add_argument(
        "--primary", action="append", default=[], metavar="FAMILY=CONDITION",
        help="Set the primary condition a family is predicted from; repeatable",
    )
    edit_families.add_argument(
        "--clear-primary", action="append", default=[], metavar="FAMILY",
        help="Return a family to the default primary condition; repeatable",
    )
    edit_families.add_argument(
        "--lamella", default=None, metavar="LAMELLA",
        help=(
            "Change the entries of one lamella, by name or number, instead of the project's. "
            "--clear and --clear-primary then return it to the project's entries"
        ),
    )
    edit_families.add_argument(
        "--confirm-corrected", action="store_true",
        help="Confirm images leaving a label group marked corrected",
    )
    edit_families.add_argument(
        "--apply", action="store_true",
        help="Write the change. Without it, only show what would change",
    )

    # -- find-specks ---------------------------------------------------------
    find_specks_cmd = sub.add_parser(
        "find-specks",
        help="Find labels with specks: components likely too small to be real",
        description=(
            "List every label with a connected component of a few pixels, of any class. These "
            "stray spots end up in training data as annotation, and holes of that size in "
            "background too. Changes nothing: some small structures are real, so check each one in "
            "the window before removing it. Reads every label, which is much slower than reading "
            "headers."
        ),
    )
    _add_scope_arguments(find_specks_cmd, default_modality="all")
    find_specks_cmd.add_argument(
        "--at-most", type=int, default=DEFAULT_SPECK_PIXELS, metavar="PIXELS",
        help=(
            f"Largest component reported (default: {DEFAULT_SPECK_PIXELS}, which also flags some "
            f"real small structures). At {PREDICTION_PIXEL_BLOCK}, only hand-painted specks are "
            "reported: labels converted from predictions are upscaled from half resolution, so "
            "their smallest area is a 2x2 block"
        ),
    )
    find_specks_cmd.add_argument(
        "--config", default=None,
        help="Segmentation config file naming the classes",
    )

    # -- compress-labels ------------------------------------------------------
    compress_labels_cmd = sub.add_parser(
        "compress-labels",
        help="Compress label files to save disk space",
        description=(
            "Re-encode label TIFFs with a compact encoding, to save disk space without changing "
            "pixel values or dimensions. A label that cannot be read, or would change, is skipped "
            "and listed."
        ),
    )
    _add_scope_arguments(compress_labels_cmd, default_modality="all")
    compress_labels_cmd.add_argument(
        "--confirm-corrected", action="store_true",
        help="Allow compressing labels marked corrected",
    )
    compress_labels_cmd.add_argument(
        "--dry-run", action="store_true",
        help="Show the savings without changing any files",
    )



# ---------------------------------------------------------------------------
# The commands
# ---------------------------------------------------------------------------

def _cmd_gui(args: argparse.Namespace) -> int:
    """Handle ``am-seg gui``."""
    # Imported here so a missing Qt binding cannot break the other
    # commands, or `--help`.
    from ..gui.app import run

    try:
        return run(
            start_path=Path(args.start_path) if args.start_path else None,
            config_path=Path(args.config) if args.config else None,
            viewer=args.viewer,
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 2



def _cmd_predict(args: argparse.Namespace) -> int:
    """Handle ``am-seg predict``."""

    records, start_path = _resolve_records(args)
    if not records:
        logger.info("No matching images found to predict")
        return 2

    weights = Path(args.model)
    # The description is read before the model, so a typo in it costs
    # nothing and a mismatch is reported before anything heavy loads.
    try:
        from ..ml.model import read_model_description, run_name_of

        description = read_model_description(weights)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    source = PredictionSource(
        name=weights.name, run_name=run_name_of(weights),
        generation=description.generation.name,
    )

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
            "Would predict: %d %s image(s)%s (dry run, nothing written and no "
            "model loaded)", len(plan.predictions), plan.modality,
            describe_conditions(plan.conditions),
        )
        return 0
    if not plan.predictions:
        logger.info("Nothing to predict")
        return 0

    try:
        # Both import torch, which costs seconds and is only in the second
        # environment: a command that does not predict must not pay for it.
        from ..ml.environment import limit_cpu_threads, resolve_device
        from ..ml.model import load_model

        # Before the load, which starts CUDA: a GPU is pinned only before then.
        device = resolve_device(args.device)
        threads = limit_cpu_threads()
        logger.info("Loading %s onto %s, with %d CPU thread(s)", weights.name, device, threads)
        loaded = load_model(weights, description, device=device)
    except (RuntimeError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 2

    total = len(plan.predictions)
    started = time.monotonic()

    def report(index: int, item: PlannedPrediction) -> None:
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
    logger.info("Total predicted: %d %s image(s)%s in %s", written, plan.modality,
                describe_conditions(plan.conditions),
                _format_duration(time.monotonic() - started))
    return 0


def _cmd_convert_predictions(args: argparse.Namespace) -> int:
    """Handle ``am-seg convert-predictions``."""
    records, start_path, projects = _resolve_scope(args)
    if not records:
        logger.info("No matching images found to convert")
        return 2
    try:
        plan = plan_conversions(records, overwrite=args.overwrite, prefer=args.prefer)
    except StateFileError as error:
        logger.error("Nothing was converted: %s", error)
        return 2
    if plan.count_fallbacks():
        logger.info(
            "%d image(s) have no %s prediction; converting the %s one",
            plan.count_fallbacks(), args.prefer, other_prediction_origin(args.prefer),
        )
    if args.dry_run:
        for item in plan.conversions:
            logger.info("%s %s", item.describe(), item.destination)
    for item in plan.refused:
        logger.warning("Left untouched, corrected: %s", item.destination)
    if plan.refused:
        logger.warning(
            "%d corrected label(s) left untouched: converting never replaces one",
            len(plan.refused),
        )
    replacing = [
        item for item in (*plan.conversions, *plan.replaceable) if item.existing != "none"
    ]
    if replacing:
        _log_replacements(replacing, _recorder(args), overwrite=args.overwrite)
    if args.dry_run:
        logger.info("Would convert: %d (dry run, nothing written)", len(plan.conversions))
        return 0
    recorder = _writing_recorder(
        args, projects_holding([record.image_path for record in records], projects)
    )
    try:
        converted = len(convert_record_predictions(
            records, overwrite=args.overwrite, recorder=recorder,
            prefer=args.prefer,
        ))
    except ConversionStopped as stopped:
        logger.error("%s", stopped)
        logger.info("Total converted: %d; re-run to convert the rest", len(stopped.written))
        return 1
    logger.info("Total converted: %d", converted)
    return 0



def _log_replacements(replacing, recorder, *, overwrite: bool) -> None:
    """Say what converting over the labels already there does, or would, to them."""
    leaves = Counter(what_replacing_leaves(replacing, recorder).values())
    described = "; ".join(
        f"{count} {text}" for count, text in (
            (leaves["placeholder"], "empty placeholder(s), nothing lost"),
            (leaves["kept"], "holding annotation not marked corrected, kept in AM_archive/"),
            (leaves["reconvertible"], "earlier conversion(s) never edited, not kept"),
        ) if count
    )
    if overwrite:
        logger.info("Replacing %d label(s) already there: %s", len(replacing), described)
    else:
        logger.warning(
            "%d label(s) already there were left as they are (%s); pass --overwrite "
            "to replace them", len(replacing), described,
        )


def _cmd_state(args: argparse.Namespace) -> int:
    """Handle ``am-seg correct``."""

    records, start_path, projects = _resolve_scope(args)
    if not records:
        logger.info("No matching images found to %s", args.command)
        return 2

    states: dict[str, RecordState] = {
        "deselect": "deselected", "select": "selected", "correct": "corrected"
    }
    state = states[args.command]
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
        changed = apply_plan(plan, recorder=_writing_recorder(args, projects))
    except (StaleState, OSError, ValueError) as exc:
        logger.error("Could not %s matching images: %s", args.command, exc)
        return 2
    logger.info("Total %s: %d", args.command, changed)
    return 0



def _cmd_show_families(args: argparse.Namespace) -> int:
    """Handle ``am-seg show-families``."""
    start_path = Path(args.start_path)
    projects = Scanner(start_path).build_index().all_projects()
    if args.project is not None:
        projects = [p for p in projects if matches_project(p, args.project)]
    if not projects:
        logger.error("No projects matched under %s", start_path)
        return 2
    for project in projects:
        print(project.display_name)
        summaries = summarise_families(project)
        if not summaries:
            print("    (no image folders)")
        for summary in summaries:
            for line in summary.describe():
                print(f"  {line}")
        for line in describe_lamella_sections(project):
            print(f"  {line}")
    return 0


def _pairs(items: list[str], flag: str) -> dict[str, str] | None:
    """Return ``NAME=VALUE`` arguments as a mapping, or None if one is malformed."""
    pairs: dict[str, str] = {}
    for item in items:
        name, separator, value = item.partition("=")
        if not separator or not name.strip() or not value.strip():
            logger.error("%s expects NAME=VALUE, got %r", flag, item)
            return None
        pairs[name.strip()] = value.strip()
    return pairs


def _cmd_edit_families(args: argparse.Namespace) -> int:
    """Handle ``am-seg edit-families``."""
    start_path = Path(args.start_path)
    projects = [
        project for project in Scanner(start_path).build_index().all_projects()
        if matches_project(project, args.project)
    ]
    if not projects:
        logger.error("No projects matched under %s", start_path)
        return 2

    families_set = _pairs(args.set, "--set")
    primary_set = _pairs(args.primary, "--primary")
    if families_set is None or primary_set is None:
        return 2
    families: dict[str, str | None] = {**families_set}
    families.update({name.strip(): None for name in args.clear})
    primary: dict[str, str | None] = {**primary_set}
    primary.update({name.strip(): None for name in args.clear_primary})
    if not families and not primary:
        logger.error("Nothing to do: pass --set, --clear, --primary or --clear-primary")
        return 2

    # Every project is planned before any is written, so one refusal leaves
    # the whole collection as it was.
    changes = []
    refused = False
    for project in projects:
        lamella = None
        if args.lamella is not None:
            matched = [
                candidate for candidate in project.lamellae
                if matches_lamella(candidate, [args.lamella])
            ]
            if len(matched) != 1:
                refused = True
                logger.error(
                    "%s: --lamella %s matches %d lamellae, not one",
                    project.display_name, args.lamella, len(matched),
                )
                continue
            lamella = matched[0]
        try:
            changes.append(
                plan_families_change(
                    project, families=families, primary=primary, lamella=lamella,
                    allow_leaving_corrected=args.confirm_corrected,
                )
            )
        except FamiliesValidationError as exc:
            refused = True
            logger.error("%s: refused", project.display_name)
            for problem in exc.problems:
                logger.error("  %s", problem)
    if refused:
        return 2

    for change in changes:
        for line in change.describe():
            print(line)
    changed = [change for change in changes if not change.is_empty]
    if not changed:
        logger.info("Nothing to change across %d project(s)", len(projects))
        return 0
    if not args.apply:
        logger.info(
            "Preview only: %d of %d project(s) would change; pass --apply to write",
            len(changed), len(projects),
        )
        return 0

    roots = {change.project_root for change in changed}
    recorder = _writing_recorder(args, [project for project in projects if project.path in roots])
    try:
        for change in changed:
            apply_families_change(change, recorder=recorder)
    except (StaleFamilies, StaleState) as exc:
        logger.error("%s", exc)
        return 2
    logger.info("Updated %d of %d project(s)", len(changed), len(projects))
    return 0


def _cmd_find_specks(args: argparse.Namespace) -> int:
    """Handle ``am-seg find-specks``."""

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



def _cmd_compress_labels(args: argparse.Namespace) -> int:
    """Handle ``am-seg compress-labels``."""

    records, start_path, projects = _resolve_scope(args)
    if not records:
        logger.info("No matching images found to compress labels for")
        return 2

    try:
        results = compress_labels(
            records,
            allow_corrected=args.confirm_corrected,
            dry_run=args.dry_run,
            recorder=None if args.dry_run else _writing_recorder(args, projects),
        )
    except (PermissionError, RuntimeError) as exc:
        logger.error("Could not compress labels: %s", exc)
        return 2

    for result in results:
        rel = make_path_relative(result.label_path, start_path)
        if result.problem:
            logger.warning("Left %s alone: %s", rel, result.problem)
        elif args.dry_run:
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
    unsafe = [r for r in results if r.problem]
    if unsafe:
        logger.warning(
            "%d label(s) could not be read or re-encoded without altering a "
            "pixel, and were left alone", len(unsafe),
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


#: What each of this group's subcommands runs. Registered above and
#: dispatched here, so the two cannot name different sets.
COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "gui": _cmd_gui,
    "predict": _cmd_predict,
    "convert-predictions": _cmd_convert_predictions,
    "correct": _cmd_state,
    "deselect": _cmd_state,
    "select": _cmd_state,
    "show-families": _cmd_show_families,
    "edit-families": _cmd_edit_families,
    "find-specks": _cmd_find_specks,
    "compress-labels": _cmd_compress_labels,
}
