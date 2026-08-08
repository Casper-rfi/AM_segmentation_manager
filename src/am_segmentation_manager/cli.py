"""Command line interface for common operations.

The CLI is intentionally lightweight and uses argparse so users can run
`python -m am_segmentation_manager.cli --help` to see available commands and
`python -m am_segmentation_manager.cli <command> --help` to get command-specific
usage information.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from .scanner import Scanner
from .operations.predictions import convert_record_predictions

logger = logging.getLogger(__name__)


def _parse_scope_number(value: str | None) -> str | None:
    """Return a zero-padded numeric value for simple scope filters."""
    if value is None:
        return None
    match = re.match(r"^0*(\d+)$", value.strip())
    if not match:
        return None
    return f"{int(match.group(1)):02d}"


def _parse_selector_values(values: list[str] | None) -> tuple[set[int], set[str]]:
    """Parse comma-separated numeric ranges and exact names from CLI selectors."""
    numeric_values: set[int] = set()
    name_values: set[str] = set()
    for raw in values or []:
        for item in raw.split(","):
            value = item.strip()
            if not value:
                continue
            match = re.fullmatch(r"(\d+)(?:-(\d+))?", value)
            if match is None:
                name_values.add(value)
                continue
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) is not None else start
            if end < start:
                start, end = end, start
            for number in range(start, end + 1):
                numeric_values.add(number)
    return numeric_values, name_values


def _matches_lamella(lamella, selectors: list[str] | None) -> bool:
    numeric_values, name_values = _parse_selector_values(selectors)
    if not numeric_values and not name_values:
        return True
    if "all" in name_values or lamella.name in name_values:
        return True
    lamella_number = _parse_scope_number(lamella.number)
    if lamella_number is not None and int(lamella_number) in numeric_values:
        return True
    return False


def _matches_stage(stage, selectors: list[str] | None) -> bool:
    numeric_values, name_values = _parse_selector_values(selectors)
    if not numeric_values and not name_values:
        return True
    if "all" in name_values or stage.name in name_values:
        return True
    stage_number = _parse_scope_number(stage.number)
    if stage_number is not None and int(stage_number) in numeric_values:
        return True
    return False


def _extract_image_number(image_name: str) -> int | None:
    match = re.search(r"AP_img_(\d+)", image_name, re.IGNORECASE)
    if match is not None:
        return int(match.group(1))
    match = re.search(r"img_(\d+)", image_name, re.IGNORECASE)
    if match is not None:
        return int(match.group(1))
    return None


def _matches_image(record, selectors: list[str] | None) -> bool:
    numeric_values, name_values = _parse_selector_values(selectors)
    if not numeric_values and not name_values:
        return True
    if "all" in name_values:
        return True
    image_number = _extract_image_number(record.image_path.name)
    if image_number is not None and image_number in numeric_values:
        return True
    if record.image_path.name in name_values:
        return True
    return False


def _record_matches_paths(record, supplied_paths: list[str] | None) -> bool:
    if not supplied_paths:
        return True
    for supplied in supplied_paths:
        sp = Path(supplied)
        try:
            if sp.resolve() == record.image_path.resolve():
                return True
        except Exception:
            if sp.name == record.image_path.name:
                return True
    return False


def _matches_project(project, selectors: list[str] | None) -> bool:
    """Match projects by their name or zero-padded folder-number prefix."""
    numeric_values, name_values = _parse_selector_values(selectors)
    if "all" in name_values or project.name in name_values:
        return True
    project_number = _parse_scope_number(project.number)
    return project_number is not None and int(project_number) in numeric_values


def _iter_matching_records(index, project_selectors: list[str] | None = None, lamella_selectors: list[str] | None = None, stage_selectors: list[str] | None = None, image_selectors: list[str] | None = None, paths: list[str] | None = None, modality: str = "all"):
    for project in index.all_projects():
        if project_selectors is not None and not _matches_project(project, project_selectors):
            continue
        for lamella in project.lamellae:
            if lamella_selectors is not None and not _matches_lamella(lamella, lamella_selectors):
                continue
            for stage in lamella.stages:
                if stage_selectors is not None and not _matches_stage(stage, stage_selectors):
                    continue
                for record in stage.images:
                    if modality != "all" and record.modality != modality:
                        continue
                    if image_selectors is not None and not _matches_image(record, image_selectors):
                        continue
                    if _record_matches_paths(record, paths):
                        yield record


def _legacy_summarize_lamella(lamella):
    sem_images = 0
    fib_images = 0
    sem_labels = 0
    fib_labels = 0
    sem_selected = 0
    fib_selected = 0
    sem_labels_selected = 0
    fib_labels_selected = 0
    sem_corrected = 0
    fib_corrected = 0
    sem_labels_corrected = 0
    fib_labels_corrected = 0
    for stage in lamella.stages:
        for record in stage.images:
            if record.modality == "SEM":
                sem_images += 1
                # count labels attached to this image
                sem_labels += len(record.labels)
                # count how many of those labels are in a selected location
                for lbl in record.labels:
                    if "deselected" not in str(lbl.parent):
                        sem_labels_selected += 1
                    if "corrected" in str(lbl.parent):
                        sem_labels_corrected += 1
                if record.selected:
                    sem_selected += 1
                if getattr(record, "corrected", False):
                    sem_corrected += 1
            elif record.modality == "FIB":
                fib_images += 1
                fib_labels += len(record.labels)
                for lbl in record.labels:
                    if "deselected" not in str(lbl.parent):
                        fib_labels_selected += 1
                    if "corrected" in str(lbl.parent):
                        fib_labels_corrected += 1
                if record.selected:
                    fib_selected += 1
                if getattr(record, "corrected", False):
                    fib_corrected += 1
    return (
        sem_images,
        fib_images,
        sem_labels,
        fib_labels,
        sem_selected,
        fib_selected,
        sem_labels_selected,
        fib_labels_selected,
        sem_corrected,
        fib_corrected,
        sem_labels_corrected,
        fib_labels_corrected,
    )


def _legacy_print_project_summary(project):
    project_sem = 0
    project_fib = 0
    project_sem_labels = 0
    project_fib_labels = 0
    project_sem_selected = 0
    project_fib_selected = 0
    project_sem_labels_selected = 0
    project_fib_labels_selected = 0
    for lamella in project.lamellae:
        (
            sem,
            fib,
            sem_labels,
            fib_labels,
            sem_selected,
            fib_selected,
            sem_labels_selected,
            fib_labels_selected,
            sem_corrected,
            fib_corrected,
            sem_labels_corrected,
            fib_labels_corrected,
        ) = _legacy_summarize_lamella(lamella)
        project_sem += sem
        project_fib += fib
        project_sem_labels += sem_labels
        project_fib_labels += fib_labels
        project_sem_selected += sem_selected
        project_fib_selected += fib_selected
        project_sem_labels_selected += sem_labels_selected
        project_fib_labels_selected += fib_labels_selected
        project_sem_corrected = project_sem_corrected + sem_corrected if 'project_sem_corrected' in locals() else sem_corrected
        project_fib_corrected = project_fib_corrected + fib_corrected if 'project_fib_corrected' in locals() else fib_corrected
        project_sem_labels_corrected = project_sem_labels_corrected + sem_labels_corrected if 'project_sem_labels_corrected' in locals() else sem_labels_corrected
        project_fib_labels_corrected = project_fib_labels_corrected + fib_labels_corrected if 'project_fib_labels_corrected' in locals() else fib_labels_corrected

    logger.info("Project: %s", project.name)
    logger.info("  Path: %s", project.path)
    logger.info("  Lamellae: %d", len(project.lamellae))
    # Totals including corrected counts (sel/cor/total)
    logger.info(
        "  Totals: SEM images(sel/cor/total)=%d/%d/%d  FIB images(sel/cor/total)=%d/%d/%d  "
        "SEM labels(sel/cor/total)=%d/%d/%d  FIB labels(sel/cor/total)=%d/%d/%d",
        project_sem_selected,
        project_sem_corrected,
        project_sem,
        project_fib_selected,
        project_fib_corrected,
        project_fib,
        project_sem_labels_selected,
        project_sem_labels_corrected,
        project_sem_labels,
        project_fib_labels_selected,
        project_fib_labels_corrected,
        project_fib_labels,
    )
    logger.info("  Lamella breakdown:")
    logger.info(
        "    {:<30} {:>18} {:>18} {:>24} {:>24}".format(
            "Lamella",
            "SEM(sel/cor/total)",
            "FIB(sel/cor/total)",
            "SEM labels(sel/cor/total)",
            "FIB labels(sel/cor/total)",
        )
    )
    logger.info("    " + "-" * 120)
    for lamella in project.lamellae:
        (
            sem,
            fib,
            sem_labels,
            fib_labels,
            sem_selected,
            fib_selected,
            sem_labels_selected,
            fib_labels_selected,
            sem_corrected,
            fib_corrected,
            sem_labels_corrected,
            fib_labels_corrected,
        ) = _legacy_summarize_lamella(lamella)
        sem_str = f"{sem_selected}/{sem_corrected}/{sem}"
        fib_str = f"{fib_selected}/{fib_corrected}/{fib}"
        sem_lbl_str = f"{sem_labels_selected}/{sem_labels_corrected}/{sem_labels}"
        fib_lbl_str = f"{fib_labels_selected}/{fib_labels_corrected}/{fib_labels}"
        logger.info(
            "    {:<30} {:>18} {:>18} {:>24} {:>24}".format(
                lamella.display_name, sem_str, fib_str, sem_lbl_str, fib_lbl_str
            )
        )
        for stage in lamella.stages:
            logger.info(
                "      {:<30} {:>6} image(s)".format(
                    stage.display_name,
                    len(stage.images),
                )
            )
    logger.info("")


_STATUS_ORDER = ("deselected", "selected", "corrected", "all")


def _new_counts() -> dict[str, int]:
    return {status: 0 for status in _STATUS_ORDER}


def _filesystem_state(path: Path) -> str:
    """Return an artifact's workflow state from its containing directory."""
    if path.parent.name == "deselected":
        return "deselected"
    if path.parent.name == "corrected":
        return "corrected"
    return "selected"


def _summarize_lamella(lamella) -> dict[str, dict[str, int]]:
    """Count images and labels in each mutually exclusive workflow state."""
    summary = {name: _new_counts() for name in ("SEM images", "FIB images", "SEM labels", "FIB labels")}
    for stage in lamella.stages:
        for record in stage.images:
            if record.modality not in {"SEM", "FIB"}:
                continue
            image_counts = summary[f"{record.modality} images"]
            image_counts[_filesystem_state(record.image_path)] += 1
            image_counts["all"] += 1
            label_counts = summary[f"{record.modality} labels"]
            for label in record.labels:
                label_counts[_filesystem_state(label)] += 1
                label_counts["all"] += 1
    return summary


def _add_counts(destination: dict[str, int], source: dict[str, int]) -> None:
    for status in _STATUS_ORDER:
        destination[status] += source[status]


def _count_text(counts: dict[str, int]) -> str:
    return "/".join(str(counts[status]) for status in _STATUS_ORDER)


def _print_project_summary(project) -> None:
    """Print a compact, aligned scan summary without logging boilerplate."""
    categories = ("SEM images", "FIB images", "SEM labels", "FIB labels")
    total = {category: _new_counts() for category in categories}
    lamella_summaries = []
    for lamella in project.lamellae:
        summary = _summarize_lamella(lamella)
        lamella_summaries.append((lamella, summary))
        for category in categories:
            _add_counts(total[category], summary[category])

    logger.info("\n%s", project.display_name)
    logger.info("%s", "=" * 72)
    logger.info("Path: %s", project.path)
    logger.info("Counts shown as: deselected / selected / corrected / all")
    logger.info("\nAll lamellae")
    logger.info("%-16s %12s %12s %12s %12s", "", "Deselected", "Selected", "Corrected", "All")
    for category in categories:
        logger.info(
            "%-16s %12d %12d %12d %12d",
            category,
            *(total[category][status] for status in _STATUS_ORDER),
        )
    logger.info("\nLamellae")
    logger.info("Counts: deselected / selected / corrected / all")
    logger.info("%s", "-" * 72)
    for lamella, summary in lamella_summaries:
        logger.info("%s", lamella.display_name)
        for category in categories:
            logger.info("  %-14s %s", category, _count_text(summary[category]))
        for stage in lamella.stages:
            logger.info("  %-14s %d image(s)", stage.display_name, len(stage.images))


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI runner
    parser = argparse.ArgumentParser("am_segmentation_manager", description="Manage Autolamella segmentation projects")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    sub = parser.add_subparsers(dest="command")

    list_projects = sub.add_parser(
        "list-projects",
        aliases=["scan"],
        help="Scan a directory and list discovered projects",
        description="Recursively scan for Autolamella projects and print a project summary",
    )
    list_projects.add_argument("start_path", nargs="?", default=".", help="Directory to start scanning from")

    convert_predictions = sub.add_parser(
        "convert-predictions",
        help="Convert predictions into labels for explicitly selected images",
    )
    convert_predictions.add_argument("paths", nargs="*", help="Explicit image paths")
    convert_predictions.add_argument("--start-path", default=".", help="Directory to start scanning from")
    convert_predictions.add_argument("--project", action="append", default=[], help="Project name or number; use 'all' explicitly")
    convert_predictions.add_argument("--lamella", action="append", default=[], help="Lamella name/number; use 'all' explicitly")
    convert_predictions.add_argument("--stage", action="append", default=[], help="Stage name/number; use 'all' explicitly")
    convert_predictions.add_argument("--image", action="append", default=[], help="Image name/number; use 'all' explicitly")
    convert_predictions.add_argument("--modality", choices=["SEM", "FIB", "all"], required=True)
    convert_predictions.add_argument(
        "--kind",
        choices=["raw", "clean"],
        required=True,
        help="Prediction set to convert",
    )

    def add_state_command(name: str, help_text: str) -> None:
        command = sub.add_parser(name, help=help_text)
        command.add_argument("paths", nargs="*", help="Explicit image paths")
        command.add_argument("--start-path", default=".", help="Directory to start scanning from")
        command.add_argument("--project", action="append", default=[], help="Project name or number (for example 001); use 'all' explicitly")
        command.add_argument("--lamella", action="append", default=[], help="Lamella name/number; use 'all' explicitly")
        command.add_argument("--stage", action="append", default=[], help="Stage name/number; use 'all' explicitly")
        command.add_argument("--image", action="append", default=[], help="Image name/number; use 'all' explicitly")
        command.add_argument(
            "--modality",
            choices=["SEM", "FIB", "all"],
            required=True,
            help="Image modality to change; specify 'all' to change SEM and FIB images",
        )
        command.add_argument(
            "--confirm-corrected",
            action="store_true",
            help="Confirm moving artifacts out of the corrected state",
        )

    add_state_command("deselect", "Move images and labels into the deselected state")
    add_state_command("select", "Move images and labels into the selected state")
    add_state_command("correct", "Move images and labels into the corrected state")

    args = parser.parse_args(argv)

    # configure logging - emit to stdout so CLI output is capturable in tests
    if args and getattr(args, "verbose", False):
        logging.basicConfig(level=logging.DEBUG, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)], force=True)
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)], force=True)

    if args.command in {"list-projects", "scan"}:
        s = Scanner(Path(args.start_path))
        idx = s.build_index()
        for p in idx.all_projects():
            _print_project_summary(p)
        return 0

    if args.command == "convert-predictions":
        if args.paths and not any([args.project, args.lamella, args.stage, args.image]):
            from .operations.selection import record_from_image_path
            records = [record_from_image_path(Path(path)) for path in args.paths]
        else:
            missing_scopes = [flag for flag, values in (("--project", args.project), ("--lamella", args.lamella), ("--stage", args.stage), ("--image", args.image)) if not values]
            if missing_scopes:
                logger.error("Conversion requires explicit selectors; use 'all' where appropriate (missing: %s).", ", ".join(missing_scopes))
                return 2
            records = list(_iter_matching_records(Scanner(Path(args.start_path)).build_index(), project_selectors=args.project, lamella_selectors=args.lamella, stage_selectors=args.stage, image_selectors=args.image, modality=args.modality))
        records = [record for record in records if args.modality == "all" or record.modality == args.modality]
        if not records:
            logger.info("No matching images found to convert")
            return 2
        converted = convert_record_predictions(records, prediction_kind=args.kind)
        logger.info("Total converted: %d", converted)
        return 0

    if args.command in {"deselect", "select", "correct"}:
        # Direct paths identify the exact records to change.  Indexed mode
        # requires every scope level to be explicit; use e.g. ``--lamella all``
        # rather than relying on an omitted filter to mean all lamellae.
        from .operations.selection import (
            deselect_records,
            mark_corrected_records,
            record_from_image_path,
            select_records,
        )

        if args.paths and not any([args.project, args.lamella, args.stage, args.image]):
            records = [
                record
                for record in (record_from_image_path(Path(path)) for path in args.paths)
                if args.modality == "all" or record.modality == args.modality
            ]
        else:
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
                return 2
            index = Scanner(Path(args.start_path)).build_index()
            records = list(
                _iter_matching_records(
                    index,
                    project_selectors=args.project,
                    lamella_selectors=args.lamella,
                    stage_selectors=args.stage,
                    image_selectors=args.image,
                    modality=args.modality,
                )
            )
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

    if args.command == "deselect":
        # Path-only mode: if explicit paths provided and no project/lamella/stage scope,
        # operate directly on the filesystem so absolute paths outside the scan root
        # are supported. This moves the image and any associated labels found under
        # the same stage folder.
        if args.paths and not any([args.project, args.lamella, args.stage, args.image, args.all]):
            from .operations.labels import move_to_deselected
            deselected_count = 0
            for supplied in args.paths:
                sp = Path(supplied)
                if not sp.exists():
                    logger.warning("Path does not exist: %s", sp)
                    continue
                try:
                    move_to_deselected(sp)
                    # attempt to find and move associated labels in the same stage
                    stage_dir = None
                    modality = None
                    for anc in sp.parents:
                        if anc.name == "sem":
                            stage_dir = anc.parent
                            modality = "SEM"
                            break
                        if anc.name == "fib":
                            stage_dir = anc.parent
                            modality = "FIB"
                            break
                    if stage_dir is not None:
                        labels_dir = stage_dir / ("labels" if modality == "SEM" else "labels_fib")
                        lbl_candidate = labels_dir / sp.name
                        if lbl_candidate.exists():
                            move_to_deselected(lbl_candidate)
                    logger.info("Deselected: %s", sp)
                    deselected_count += 1
                except Exception as exc:
                    logger.exception("Failed to deselect path %s: %s", sp, exc)
            logger.info("Total deselected: %d", deselected_count)
            return 0

        # if any scoping args provided, use scanner to find matching records
        if args.all or any([args.project, args.lamella, args.stage, args.image]) or args.paths or not any([args.project, args.lamella, args.stage, args.image, args.all]):
            s = Scanner(Path(args.start_path))
            idx = s.build_index()
            matches = list(
                _iter_matching_records(
                    idx,
                    project_name=args.project,
                    lamella_selectors=args.lamella,
                    stage_selectors=args.stage,
                    image_selectors=args.image,
                    paths=args.paths,
                )
            )
            if not matches:
                logger.info("No matching images found to deselect")
                return 2
            deselected = 0
            from .operations.selection import deselect_records

            for rec in matches:
                try:
                    deselect_records([rec])
                    logger.info("Deselected: %s", rec.image_path)
                    deselected += 1
                except Exception as exc:
                    logger.exception("Failed to deselect %s: %s", rec.image_path, exc)
            logger.info("Total deselected: %d", deselected)
            return 0
        else:
            parser.print_help()
            return 1

    if args.command == "select":
        # Path-only mode: if explicit paths provided and no project/lamella/stage scope,
        # operate directly on the filesystem so absolute paths outside the scan root
        # are supported. This moves the image out of a 'deselected' folder and also
        # restores nearby labels when possible.
        if args.paths and not any([args.project, args.lamella, args.stage, args.image, args.all]):
            from .operations.labels import move_to_selected
            selected_count = 0
            for supplied in args.paths:
                sp = Path(supplied)
                try:
                    # Case: path points directly to a file inside deselected/
                    if sp.exists() and sp.parent.name == "deselected":
                        move_to_selected(sp)
                        # attempt to restore associated label
                        stage_dir = None
                        modality = None
                        for anc in sp.parents:
                            if anc.name == "sem":
                                stage_dir = anc.parent
                                modality = "SEM"
                                break
                            if anc.name == "fib":
                                stage_dir = anc.parent
                                modality = "FIB"
                                break
                        if stage_dir is not None:
                            lbl_deselected = (stage_dir / ("labels" if modality == "SEM" else "labels_fib") / "deselected" / sp.name)
                            if lbl_deselected.exists():
                                move_to_selected(lbl_deselected)
                        logger.info("Selected: %s", sp)
                        selected_count += 1
                        continue

                    # Case: path points to the original location; look for sibling deselected
                    deselected = sp.parent / "deselected" / sp.name
                    if deselected.exists():
                        move_to_selected(deselected)
                        # attempt to restore associated label
                        stage_dir = None
                        modality = None
                        for anc in sp.parents:
                            if anc.name == "sem":
                                stage_dir = anc.parent
                                modality = "SEM"
                                break
                            if anc.name == "fib":
                                stage_dir = anc.parent
                                modality = "FIB"
                                break
                        if stage_dir is not None:
                            lbl_deselected = (stage_dir / ("labels" if modality == "SEM" else "labels_fib") / "deselected" / sp.name)
                            if lbl_deselected.exists():
                                move_to_selected(lbl_deselected)
                        logger.info("Selected: %s", sp)
                        selected_count += 1
                    else:
                        logger.warning("No deselected file found for: %s", sp)
                except Exception as exc:
                    logger.exception("Failed to select path %s: %s", sp, exc)
            logger.info("Total selected: %d", selected_count)
            return 0

        if args.all or any([args.project, args.lamella, args.stage, args.image]) or args.paths or not any([args.project, args.lamella, args.stage, args.image, args.all]):
            s = Scanner(Path(args.start_path))
            idx = s.build_index()
            matches = list(
                _iter_matching_records(
                    idx,
                    project_name=args.project,
                    lamella_selectors=args.lamella,
                    stage_selectors=args.stage,
                    image_selectors=args.image,
                    paths=args.paths,
                )
            )
            if not matches:
                logger.info("No matching images found to select")
                return 2
            from .operations.selection import select_records
            selected = 0
            for rec in matches:
                try:
                    select_records([rec])
                    logger.info("Selected: %s", rec.image_path)
                    selected += 1
                except Exception as exc:
                    logger.exception("Failed to select %s: %s", rec.image_path, exc)
            logger.info("Total selected: %d", selected)
            return 0
        else:
            parser.print_help()
            return 1

    if args.command == "correct":
        # Path-only mode
        if args.paths and not any([args.project, args.lamella, args.stage, args.image, args.all]):
            from .operations.labels import move_to_corrected
            corrected_count = 0
            for supplied in args.paths:
                sp = Path(supplied)
                if not sp.exists():
                    logger.warning("Path does not exist: %s", sp)
                    continue
                try:
                    move_to_corrected(sp)
                    # attempt to find and move associated labels in the same stage
                    stage_dir = None
                    modality = None
                    for anc in sp.parents:
                        if anc.name == "sem":
                            stage_dir = anc.parent
                            modality = "SEM"
                            break
                        if anc.name == "fib":
                            stage_dir = anc.parent
                            modality = "FIB"
                            break
                    if stage_dir is not None:
                        labels_dir = stage_dir / ("labels" if modality == "SEM" else "labels_fib")
                        lbl_candidate = labels_dir / sp.name
                        if lbl_candidate.exists():
                            move_to_corrected(lbl_candidate)
                    logger.info("Marked corrected: %s", sp)
                    corrected_count += 1
                except Exception as exc:
                    logger.exception("Failed to mark corrected path %s: %s", sp, exc)
            logger.info("Total corrected: %d", corrected_count)
            return 0

        # scoped/indexed mode
        if args.all or any([args.project, args.lamella, args.stage, args.image]) or args.paths or not any([args.project, args.lamella, args.stage, args.image, args.all]):
            s = Scanner(Path(args.start_path))
            idx = s.build_index()
            matches = list(
                _iter_matching_records(
                    idx,
                    project_name=args.project,
                    lamella_selectors=args.lamella,
                    stage_selectors=args.stage,
                    image_selectors=args.image,
                    paths=args.paths,
                )
            )
            if not matches:
                logger.info("No matching images found to mark corrected")
                return 2
            corrected = 0
            from .operations.selection import mark_corrected_records

            for rec in matches:
                try:
                    mark_corrected_records([rec])
                    logger.info("Marked corrected: %s", rec.image_path)
                    corrected += 1
                except Exception as exc:
                    logger.exception("Failed to mark corrected %s: %s", rec.image_path, exc)
            logger.info("Total corrected: %d", corrected)
            return 0
        else:
            parser.print_help()
            return 1

    if args.command == "uncorrect":
        # Path-only mode
        if args.paths and not any([args.project, args.lamella, args.stage, args.image, args.all]):
            from .operations.labels import move_to_uncorrected
            uncorrected_count = 0
            for supplied in args.paths:
                sp = Path(supplied)
                try:
                    # Case: path points into corrected/
                    if sp.exists() and sp.parent.name == "corrected":
                        move_to_uncorrected(sp)
                        # attempt to restore associated label
                        stage_dir = None
                        modality = None
                        for anc in sp.parents:
                            if anc.name == "sem":
                                stage_dir = anc.parent
                                modality = "SEM"
                                break
                            if anc.name == "fib":
                                stage_dir = anc.parent
                                modality = "FIB"
                                break
                        if stage_dir is not None:
                            lbl_corrected = (stage_dir / ("labels" if modality == "SEM" else "labels_fib") / "corrected" / sp.name)
                            if lbl_corrected.exists():
                                move_to_uncorrected(lbl_corrected)
                        logger.info("Unmarked corrected: %s", sp)
                        uncorrected_count += 1
                        continue

                    # Case: path points to original location; look for sibling corrected
                    corrected = sp.parent / "corrected" / sp.name
                    if corrected.exists():
                        move_to_uncorrected(corrected)
                        # attempt to restore associated label
                        stage_dir = None
                        modality = None
                        for anc in sp.parents:
                            if anc.name == "sem":
                                stage_dir = anc.parent
                                modality = "SEM"
                                break
                            if anc.name == "fib":
                                stage_dir = anc.parent
                                modality = "FIB"
                                break
                        if stage_dir is not None:
                            lbl_corrected = (stage_dir / ("labels" if modality == "SEM" else "labels_fib") / "corrected" / sp.name)
                            if lbl_corrected.exists():
                                move_to_uncorrected(lbl_corrected)
                        logger.info("Unmarked corrected: %s", sp)
                        uncorrected_count += 1
                    else:
                        logger.warning("No corrected file found for: %s", sp)
                except Exception as exc:
                    logger.exception("Failed to unmark corrected path %s: %s", sp, exc)
            logger.info("Total uncorrected: %d", uncorrected_count)
            return 0

        if args.all or any([args.project, args.lamella, args.stage, args.image]) or args.paths or not any([args.project, args.lamella, args.stage, args.image, args.all]):
            s = Scanner(Path(args.start_path))
            idx = s.build_index()
            matches = list(
                _iter_matching_records(
                    idx,
                    project_name=args.project,
                    lamella_selectors=args.lamella,
                    stage_selectors=args.stage,
                    image_selectors=args.image,
                    paths=args.paths,
                )
            )
            if not matches:
                # Fallback: files may have been moved into 'corrected' folders and the
                # scanner does not index them. Attempt to find corrected files under
                # the stage folders and move them back.
                from .operations.labels import move_to_uncorrected
                uncorrected_fallback = 0
                for p in idx.all_projects():
                    if args.project and p.name != args.project:
                        continue
                    for lam in p.lamellae:
                        if args.lamella and not _matches_lamella(lam, args.lamella):
                            continue
                        for st in lam.stages:
                            if args.stage and not _matches_stage(st, args.stage):
                                continue
                            # check sem/fib and labels folders for corrected/ subfolders
                            for base in [st.path / "sem", st.path / "fib"]:
                                corr_dir = base / "corrected"
                                if corr_dir.exists() and corr_dir.is_dir():
                                    for f in sorted(corr_dir.iterdir()):
                                        if f.is_file():
                                            try:
                                                move_to_uncorrected(f)
                                                uncorrected_fallback += 1
                                            except Exception:
                                                logger.exception("Failed moving corrected file back: %s", f)
                            for base in [st.path / "labels", st.path / "labels_fib"]:
                                corr_dir = base / "corrected"
                                if corr_dir.exists() and corr_dir.is_dir():
                                    for f in sorted(corr_dir.iterdir()):
                                        if f.is_file():
                                            try:
                                                move_to_uncorrected(f)
                                                uncorrected_fallback += 1
                                            except Exception:
                                                logger.exception("Failed moving corrected label back: %s", f)
                if uncorrected_fallback > 0:
                    logger.info("Total uncorrected (fallback): %d", uncorrected_fallback)
                    return 0

                logger.info("No matching images found to unmark corrected")
                return 2
            from .operations.selection import unmark_corrected_records
            uncorrected = 0

            for rec in matches:
                try:
                    unmark_corrected_records([rec])
                    logger.info("Unmarked corrected: %s", rec.image_path)
                    uncorrected += 1
                except Exception as exc:
                    logger.exception("Failed to unmark corrected %s: %s", rec.image_path, exc)
            logger.info("Total uncorrected: %d", uncorrected)
            return 0
        else:
            parser.print_help()
            return 1

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI runner
    raise SystemExit(main())
