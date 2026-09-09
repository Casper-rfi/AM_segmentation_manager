"""The record: starting it, naming a moment in it, reading it, undoing by it.

Eight commands over one subject. ``init-history`` is the catch-up that makes a
record start complete; ``savepoint`` names a moment the collection can be put
back to; ``revert`` puts back the label a conversion replaced and ``merge``
brings home a copy worked on elsewhere; ``archive`` and ``prune-archive`` report
and reclaim the kept versions those draw on; ``history`` prints what a log holds.

**They register in one block, in the order below**, which is the order
``am-seg --help`` teaches them in. Splitting this module would reorder that.
"""
from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable
from pathlib import Path

from ..history import AM_HISTORY_FILENAME, KINDS, Event, containing_project, events_for, read_log
from ..io.path import make_path_relative
from ..operations.archive import (
    apply_prune,
    archive_root,
    archived_versions,
    plan_prune,
    total_size,
)
from ..operations.init_history import apply_baseline, plan_baseline
from ..operations.merge import (
    CONVERGED,
    KEEP_OURS,
    TAKE_THEIRS,
    MergeRefused,
    apply_merge,
    plan_merge,
)
from ..operations.revert import UNDOABLE, apply_revert, plan_revert, plan_revert_to
from ..reporting import format_bytes
from ..scanner import Scanner
from .scope import _add_scope_arguments, _recorder, _resolve_records

logger = logging.getLogger(__name__)


def add_arguments(sub) -> None:
    """Register the commands that record, restore and reclaim."""

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
        "--as", dest="annotator", default=None, metavar="NAME",
        help=(
            "Credit the labels to whoever annotated them, for a dataset that "
            "arrived already annotated. Recorded as a name stated on their "
            "behalf, which nothing here can verify; starting the log stays "
            "attributed to you. Point --start-path at that dataset, or you "
            "will credit them with everything under it"
        ),
    )
    init_history_cmd.add_argument(
        "--apply", action="store_true",
        help="Write the entries, rather than reporting what would be written",
    )

    # -- savepoint ----------------------------------------------------------
    savepoint_cmd = sub.add_parser(
        "savepoint",
        help="Name the state of the data, so a round of work can be undone",
        description=(
            "Write one checkpoint entry per project, sharing one id: there is "
            "no such thing as 'the collection', so a collection-wide savepoint "
            "is a set of these. Marks a round of model improvement -- proofread "
            "and curate, train, predict, convert, proofread again -- which is "
            "the boundary worth naming. It promises nothing about pixels and "
            "reports what it can reach; --close-gap keeps the versions that "
            "only the next conversion would destroy. Previews by default."
        ),
    )
    savepoint_cmd.add_argument(
        "--start-path", default=".",
        help="Directory to scan for projects (default: current directory)",
    )
    savepoint_cmd.add_argument(
        "--title", default="",
        help="What to call it, and what you will name it by later",
    )
    savepoint_cmd.add_argument(
        "--description", default="", help="Why this state is worth naming",
    )
    savepoint_cmd.add_argument(
        "--join", metavar="SAVEPOINT",
        help=(
            "Add the projects in scope to a savepoint that already exists, by "
            "id or title, rather than starting a new one"
        ),
    )
    savepoint_cmd.add_argument(
        "--close-gap", action="store_true",
        help=(
            "Keep the versions this names that are untouched machine output. "
            "The archive skips those, so the next conversion would make this "
            "state unreachable"
        ),
    )
    savepoint_cmd.add_argument(
        "--apply", action="store_true",
        help="Write the entries, rather than reporting what would be written",
    )

    # -- savepoints ---------------------------------------------------------
    savepoints_cmd = sub.add_parser(
        "list-savepoints",
        help="List the savepoints here, what each covers and what it can reach",
        description=(
            "One line per savepoint: how many projects carrying it are here, "
            "how many it was written across, and how many of the labels it "
            "names can still be got back. A count lower than the one at "
            "creation is the honest answer for a machine holding a subset, "
            "not a fault."
        ),
    )
    savepoints_cmd.add_argument(
        "--start-path", default=".",
        help="Directory to scan for projects (default: current directory)",
    )
    savepoints_cmd.add_argument(
        "--savepoint", metavar="SAVEPOINT",
        help="Show one in full, by id or title, project by project",
    )


    # -- merge --------------------------------------------------------------
    merge_cmd = sub.add_parser(
        "merge",
        help="Bring a worked-on copy back into the collection it came from",
        description=(
            "Merge a copy made by export-files back into its collection. Every "
            "label is re-hashed on both sides and compared against the ancestor "
            "the copy's AM_export.yaml names, so what changed is read from the "
            "files rather than believed from a log. A target only one side "
            "changed is taken; one both changed differently is a conflict, and "
            "one conflict stops the whole merge until it is decided by name. "
            "Nothing is ever deleted: a target the copy does not hold is not a "
            "deletion, and no image is moved. Previews by default."
        ),
    )
    merge_cmd.add_argument("source", help="The copy to merge back")
    merge_cmd.add_argument(
        "--into", required=True, metavar="DIRECTORY",
        help="The collection it came from",
    )
    merge_cmd.add_argument(
        "--resolve", action="append", default=[], metavar="TARGET=SIDE",
        help=(
            "Decide one conflict: '<project>/<label>=theirs' or "
            "'<project>:<field>=ours'. Repeatable. Both versions stay in the "
            "archive whichever way it goes"
        ),
    )
    merge_cmd.add_argument(
        "--apply", action="store_true",
        help="Carry the merge out, rather than reporting what it would do",
    )
    merge_cmd.add_argument(
        "--allow-corrected", action="store_true",
        help=(
            "Take the copy's work over labels this collection has proofread, "
            "and let it drop a corrected claim. The version here is kept "
            "either way, so this can be undone -- but not unasked"
        ),
    )

    # -- revert -------------------------------------------------------------
    revert_cmd = sub.add_parser(
        "revert",
        help="Put a selection of labels back to what they held before an operation",
        description=(
            "Undo a conversion or a merge across a whole selection, putting "
            "each label back to the version that operation replaced. For "
            "trying a newer model over labels you have already proofread: "
            "convert, look at a few, and put the lot back in one command if it "
            "is not good enough. A label whose earlier version was not kept, "
            "or that has been edited since, is reported and left alone. "
            "Previews by default."
        ),
    )
    _add_scope_arguments(revert_cmd)
    revert_cmd.add_argument(
        "--to", metavar="SAVEPOINT",
        help=(
            "Put every label back to what it held at a savepoint, by id or "
            "title -- a whole round of work undone by name. Labels the "
            "savepoint cannot reach are reported and left alone"
        ),
    )
    revert_cmd.add_argument(
        "--undo", choices=sorted(UNDOABLE), default="conversion",
        help="Which operation to undo (default: conversion)",
    )
    revert_cmd.add_argument(
        "--apply", action="store_true",
        help="Put them back, rather than reporting what would be put back",
    )
    revert_cmd.add_argument(
        "--confirm-corrected", action="store_true",
        help=(
            "Put back labels whose group is declared corrected. Required for "
            "those, proofread work being what this is most likely to be undoing"
        ),
    )
    revert_cmd.add_argument(
        "--allow-superseded", action="store_true",
        help=(
            "Put a label back even where it has been edited since the operation "
            "being undone, which discards that editing"
        ),
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
        help="Remove the kept versions no savepoint needs",
        description=(
            "Collapse what lies between savepoints. Every version a savepoint "
            "names is kept, so a state you named stays reachable, and so is "
            "everything kept since the most recent one -- the round still open "
            "is the work in hand. What goes is the ordinary proofreading saves "
            "of rounds already closed, so nothing is removed at all until a "
            "savepoint has been taken. Four more are never "
            "removed: a label's oldest kept version, the one taken before a "
            "model wrote over it, the one taken before a merge, and any version "
            "the history does not account for. Previews by default; pass "
            "--apply to remove. This is the only command that deletes a label "
            "file, and it leaves the history alone -- it costs the pixels, "
            "never the account of what happened."
        ),
    )
    prune_cmd.add_argument(
        "path", nargs="?", default=".", help="The project to prune",
    )
    prune_cmd.add_argument(
        "--start-path", default=None,
        help=(
            "Where to look for the savepoints whose versions must be kept "
            "(default: the project itself). Give the collection root when a "
            "savepoint spans more projects than the one being pruned"
        ),
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
    from ..operations.export_record import read_export_record
    from ..operations.savepoint import (
        SavepointRefused,
        apply_savepoint,
        find_savepoint,
        plan_savepoint,
    )

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
    """Handle ``am-seg savepoints``."""
    from ..operations.savepoint import (
        LOST,
        SavepointRefused,
        find_savepoint,
        list_savepoints,
    )

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


def _report_silent_projects(projects) -> None:
    """Say which projects keep no log, and so are in no savepoint.

    The one place that asks what the record covers is the place to learn that
    part of the collection is outside it: a project nothing has changed yet
    records nothing, and nothing else would ever mention it.
    """
    from ..operations.savepoint import not_recording

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



def _cmd_merge(args: argparse.Namespace) -> int:
    """Handle ``am-seg merge``."""
    try:
        resolutions = _resolutions(args.resolve)
    except ValueError as error:
        logger.error("%s", error)
        return 2

    try:
        plan = plan_merge(
            Path(args.source), Path(args.into), resolutions=resolutions
        )
    except MergeRefused as error:
        logger.error("Refused: %s", error)
        return 2

    for line in plan.describe():
        print(line)

    taken = plan.counted(TAKE_THEIRS)
    logger.info(
        "%d to take from the copy, %d already here, %d converged, %d conflict(s)",
        taken, plan.counted(KEEP_OURS), plan.counted(CONVERGED), len(plan.conflicts),
    )
    if plan.corrected:
        logger.warning(
            "%d piece(s) of proofread work would be written over or unmarked",
            len(plan.corrected),
        )
    if plan.is_refused:
        logger.error(
            "Nothing was merged. Decide each conflict with --resolve "
            "'<target>=theirs' or '<target>=ours'; both versions stay in the "
            "archive either way."
        )
        return 2
    if not taken and not any(project.incoming for project in plan.projects):
        logger.info("Nothing to merge: the collection already holds this work")
        return 0
    if not args.apply:
        logger.info("Re-run with --apply to carry it out")
        return 0

    if plan.corrected and not args.allow_corrected:
        logger.error(
            "%d piece(s) of proofread work would be written over or unmarked: "
            "%s. Nothing was merged. Re-run with --allow-corrected to take it "
            "anyway; the version here is kept either way.",
            len(plan.corrected),
            ", ".join(item.target for item in plan.corrected[:5]),
        )
        return 2

    report = apply_merge(
        plan, recorder=_recorder(args), allow_corrected=args.allow_corrected
    )
    logger.info(
        "Merged %d label(s), %d state(s) and %d field(s); %d version(s) kept, "
        "%d history entr(ies) added",
        report.labels_taken, report.states_taken, report.fields_taken,
        len(report.archived), report.entries_added,
    )
    return 0


def _resolutions(chosen: list[str]) -> dict[str, str]:
    """Return the conflicts decided on the command line.

    Raises:
        ValueError: If one is not ``<target>=theirs`` or ``<target>=ours``.
    """
    resolutions = {}
    for item in chosen:
        target, _, side = item.rpartition("=")
        if not target or side not in ("theirs", "ours"):
            raise ValueError(
                f"{item!r} is not a decision: write '<target>=theirs' or "
                "'<target>=ours'"
            )
        resolutions[target] = side
    return resolutions


def _cmd_revert(args: argparse.Namespace) -> int:
    """Handle ``am-seg revert``."""
    records, start_path = _resolve_records(args)
    if not records:
        logger.info("No matching images found")
        return 2

    from ..operations.savepoint import SavepointRefused, find_savepoint

    try:
        if args.to:
            projects = Scanner(start_path).build_index().all_projects()
            point = find_savepoint(projects, args.to)
            plan = plan_revert_to(records, point)
            logger.info(
                "%s: %d project(s) here, %d of %d label(s) reachable",
                point.title or point.id, point.found, point.reachable, point.targets,
            )
        else:
            plan = plan_revert(
                records, args.undo, allow_superseded=args.allow_superseded
            )
    except SavepointRefused as error:
        logger.error("Refused: %s", error)
        return 2
    except ValueError as error:
        logger.error("%s", error)
        return 2

    for line in plan.describe():
        print(line)
    logger.info(
        "%d label(s) can go back, %d left alone",
        len(plan.revertible), len(plan.blocked),
    )
    if not plan.revertible:
        return 0
    if plan.corrected and not args.confirm_corrected:
        logger.error(
            "%d of them are proofread work. Re-run with --confirm-corrected to "
            "put those back; what is there now is kept either way.",
            len(plan.corrected),
        )
        return 2
    if not args.apply:
        logger.info("Re-run with --apply to put them back")
        return 0

    report = apply_revert(
        plan, recorder=_recorder(args), allow_corrected=args.confirm_corrected
    )
    logger.info("Put %d label(s) back", len(report.restored))
    for path, why in report.failed:
        logger.warning("%s: %s", make_path_relative(path, start_path), why)
    return 1 if report.failed else 0



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

    from ..operations.savepoint import (
        list_savepoints,
        open_round,
        reached_versions,
    )

    root = Path(args.start_path) if args.start_path else project
    savepoints = list_savepoints(Scanner(root).build_index().all_projects())
    needed = reached_versions(savepoints)
    still_open = open_round(project)
    plan = plan_prune(
        project,
        reached=needed.get(project.resolve(), set()),
        open_round=still_open,
    )
    covers = [point for point in savepoints if any(
        carried.project.path.resolve() == project.resolve()
        for carried in point.projects
    )]
    if not covers:
        logger.warning(
            "No savepoint covers %s, so every kept version belongs to the round "
            "still open and none will be removed. Naming a state with "
            "'am-seg savepoint' is what closes a round and makes it collapsible.",
            project.name,
        )
    else:
        logger.info(
            "%d savepoint(s) cover this project; what they name is kept, as are "
            "the %d version(s) kept since the most recent",
            len(covers), len(still_open),
        )

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
    "merge": _cmd_merge,
    "revert": _cmd_revert,
    "archive": _cmd_archive,
    "prune-archive": _cmd_prune_archive,
    "history": _cmd_history,
}
