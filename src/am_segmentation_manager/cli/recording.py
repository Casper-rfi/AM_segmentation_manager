"""Starting a project's record, naming a moment in it, and reading it back.

``init-history`` is the catch-up that makes a record start complete;
``savepoint`` names a moment the whole collection can be put back to;
``list-savepoints`` and ``history`` say what the record holds. Putting anything
back is :mod:`~am_segmentation_manager.cli.restoring`.

The order is the order somebody meets these: set the record up, name a moment
in it, then read what it says.
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from ..history import AM_HISTORY_FILENAME, KINDS, Event, containing_project, events_for, read_log
from ..io.path import make_path_relative
from ..models import Project
from ..operations.export_record import read_export_record
from ..operations.init_history import apply_baseline, plan_baseline
from ..operations.savepoint import (
    LOST,
    SavepointRefused,
    apply_savepoint,
    find_savepoint,
    list_savepoints,
    not_recording,
    plan_savepoint,
)
from ..scanner import Scanner
from .scope import _recorder

logger = logging.getLogger(__name__)


#: The heading ``am-seg --help`` files these under.
TITLE = "Record what changes, and read it back"


def add_arguments(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the commands that start and read the record."""
    # -- init-history -------------------------------------------------------
    init_history_cmd = sub.add_parser(
        "init-history",
        help="Start recording the history of a collection, for the data already on disk",
        description=(
            "Write the first AM_history.jsonl of each project, with one baseline entry per label "
            "group that has a label or a state other than selected. Without this, a project starts "
            "recording when it is first changed; this adds all projects at once, so a savepoint "
            "covers the whole collection. Can be run again: only new items are added. A baseline "
            "records what was seen, not when it was annotated. It is recorded under the --user or "
            "Settings name, so set that first. Shows a preview; pass --apply to write."
        ),
    )
    init_history_cmd.add_argument(
        "--start-path", default=".",
        help="Folder to scan for projects (default: current folder)",
    )
    init_history_cmd.add_argument(
        "--as", dest="annotator", default=None, metavar="NAME",
        help=(
            "Credit the existing labels to the person who annotated them, for a dataset that "
            "arrived annotated. Starting the history is still credited to you. Point --start-path "
            "at that dataset only, or everything under it is credited to them"
        ),
    )
    init_history_cmd.add_argument(
        "--apply", action="store_true",
        help="Write the entries, instead of showing what would be written",
    )

    # -- savepoint ----------------------------------------------------------
    savepoint_cmd = sub.add_parser(
        "savepoint",
        help="Save the current state of the labels as a savepoint, so a round of work can be undone",
        description=(
            "Save the current state of the labels as a savepoint: one entry per project, sharing "
            "one id. Use it to mark a round of work (proofread, train, predict, convert, proofread "
            "again). It reports which label versions it can restore; --close-gap also keeps "
            "unedited model output, which the next conversion would otherwise overwrite. Shows a "
            "preview by default."
        ),
    )
    savepoint_cmd.add_argument(
        "--start-path", default=".",
        help="Folder to scan for projects (default: current folder)",
    )
    savepoint_cmd.add_argument(
        "--title", default="",
        help="Name of the savepoint, used to find it later",
    )
    savepoint_cmd.add_argument(
        "--description", default="", help="Why the savepoint was created",
    )
    savepoint_cmd.add_argument(
        "--join", metavar="SAVEPOINT",
        help=(
            "Add the projects to an existing savepoint, by id or title, instead of creating a new "
            "one"
        ),
    )
    savepoint_cmd.add_argument(
        "--close-gap", action="store_true",
        help=(
            "Also keep unedited model output. The archive skips it, so the next conversion would "
            "make this state unrestorable"
        ),
    )
    savepoint_cmd.add_argument(
        "--apply", action="store_true",
        help="Write the entries, instead of showing what would be written",
    )

    # -- savepoints ---------------------------------------------------------
    savepoints_cmd = sub.add_parser(
        "list-savepoints",
        help="List the savepoints here, what each covers and what can still be restored",
        description=(
            "One line per savepoint: how many projects here have it, how many it was created for, "
            "and how many of its labels can still be restored. Lower counts are expected on a "
            "computer holding only part of the collection."
        ),
    )
    savepoints_cmd.add_argument(
        "--start-path", default=".",
        help="Folder to scan for projects (default: current folder)",
    )
    savepoints_cmd.add_argument(
        "--savepoint", metavar="SAVEPOINT",
        help="Show one savepoint in full, by id or title, per project",
    )


    # -- history ------------------------------------------------------------
    history_cmd = sub.add_parser(
        "history",
        help="Print the recorded history of a project or one of its labels",
        description=(
            "Print the history recorded in the AM_history.jsonl of a project. Point it at a "
            "project folder, or at a label to see only its history. The history records what "
            "AM-seg was asked to do, not what the files contain now."
        ),
    )
    history_cmd.add_argument(
        "path", nargs="?", default=".",
        help="A project folder, or a file inside one (default: current folder)",
    )
    history_cmd.add_argument(
        "--target", default=None, metavar="PATH",
        help="Only entries about this path, relative to the project folder",
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


# ---------------------------------------------------------------------------
# The commands
# ---------------------------------------------------------------------------

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

    recorder = _recorder(args)
    annotator = recorder.attributed_to(args.annotator) if args.annotator else None
    if annotator is not None:
        logger.info(
            "Crediting %d group(s) across %d project(s) to %r, on your say-so",
            plan.entry_count, len(projects), annotator.by,
        )
    report = apply_baseline(plan, recorder=recorder, annotator=annotator)
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


def _cmd_savepoint(args: argparse.Namespace) -> int:
    """Handle ``am-seg savepoint``."""

    start_path = Path(args.start_path)
    if read_export_record(start_path) is not None:
        logger.error(
            "%s is a copy, which may hold part of a project. A savepoint taken "
            "here would claim a state for labels the copy never had. Take it "
            "in the collection this came from.",
            start_path,
        )
        return 2

    projects = Scanner(start_path).build_index().all_projects()
    if not projects:
        logger.error("No projects found under %s", start_path)
        return 2

    try:
        joining = find_savepoint(projects, args.join) if args.join else None
        plan = plan_savepoint(
            projects, args.title, description=args.description, joining=joining
        )
    except SavepointRefused as error:
        logger.error("Refused: %s", error)
        return 2

    for line in plan.describe():
        print(line)
    if plan.too_late:
        logger.warning(
            "%d project(s) cannot join: their history begins after this "
            "savepoint was taken, so the record knows nothing of what they "
            "held then (%s). Take a new savepoint to name a state they are in.",
            len(plan.too_late),
            ", ".join(project.key for project in plan.too_late[:5]),
        )
    if not plan.projects:
        logger.info(
            "Nothing to write: every project here already carries that savepoint"
            if not plan.too_late else "No project here can join that savepoint"
        )
        return 0

    risky = plan.at_risk
    if risky and not args.close_gap:
        logger.warning(
            "%d label(s) this would name are untouched machine output, which "
            "the archive does not keep: the next conversion would make them "
            "unreachable. Pass --close-gap to keep them now.",
            len(risky),
        )
    if not args.apply:
        logger.info(
            "Would write %d checkpoint entr(ies) %s %r. Re-run with --apply.",
            len(plan.projects), "joining" if plan.joining else "for", plan.title,
        )
        return 0

    report = apply_savepoint(
        plan, recorder=_recorder(args), close_the_gap=args.close_gap
    )
    logger.info(
        "Savepoint %s (%r): %d entr(ies) written, %d version(s) kept",
        report.id, plan.title, report.entries_written, len(report.versions_kept),
    )
    return 0


def _cmd_savepoints(args: argparse.Namespace) -> int:
    """Handle ``am-seg list-savepoints``."""

    start_path = Path(args.start_path)
    projects = Scanner(start_path).build_index().all_projects()
    if not projects:
        logger.error("No projects found under %s", start_path)
        return 2

    if args.savepoint:
        try:
            point = find_savepoint(projects, args.savepoint)
        except SavepointRefused as error:
            logger.error("%s", error)
            return 2
        print(point.describe())
        if point.description:
            print(f"  {point.description}")
        for carried in point.projects:
            unreachable = [
                state for state in carried.targets if state.where == LOST
            ]
            line = (
                f"  {carried.project.key}: {carried.reachable}/"
                f"{len(carried.targets)} reachable"
            )
            if carried.at_risk:
                line += f", {len(carried.at_risk)} only until the next conversion"
            print(line)
            for state in unreachable[:3]:
                print(f"      cannot get back: {state.target}")
        return 0

    savepoints = list_savepoints(projects)
    if savepoints:
        for point in savepoints:
            print(point.describe())
    else:
        logger.info("No savepoint has been taken here")
    _report_silent_projects(projects)
    return 0


def _report_silent_projects(projects: Sequence[Project]) -> None:
    """Say which projects keep no log, and so are in no savepoint.

    The one place that asks what the record covers is the place to learn that
    part of the collection is outside it: a project nothing has changed yet
    records nothing, and nothing else would ever mention it.
    """

    silent = not_recording(projects)
    if not silent:
        return
    logger.warning(
        "%d of %d project(s) here keep no history yet, so no savepoint covers "
        "them: %s. Each joins the record the first time something changes it; "
        "init-history brings them all in now.",
        len(silent), len(projects),
        ", ".join(project.key for project in silent[:5])
        + (f" and {len(silent) - 5} more" if len(silent) > 5 else ""),
    )



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
        target = make_path_relative(path, project)

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
    # No name is the ordinary case, and the line simply does not carry one.
    attribution = f" {event.by}" if event.by else ""
    if event.by and event.by_source == "declared":
        attribution += " (on somebody's say-so)"
    print(f"{event.at}  {event.kind:<18}{attribution}")
    if event.target:
        print(f"    {event.target}")
    shown = [name for name in _HISTORY_LEAD_FIELDS if name in event.data]
    shown += sorted(set(event.data) - set(_HISTORY_LEAD_FIELDS))
    details = "  ".join(f"{name}={event.data[name]}" for name in shown)
    if details:
        print(f"    {details}")


#: What each of this group's subcommands runs. Registered above and dispatched
#: here, so the two cannot name different sets.
COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "init-history": _cmd_init_history,
    "savepoint": _cmd_savepoint,
    "list-savepoints": _cmd_savepoints,
    "history": _cmd_history,
}
