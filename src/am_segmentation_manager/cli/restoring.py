"""Putting work back, and the store that makes it possible.

``merge`` brings home a copy worked on elsewhere and ``revert`` puts back the
label a conversion replaced; ``archive`` and ``prune-archive`` report and
reclaim the kept versions both of those draw on. Starting and reading the
record itself is :mod:`~am_segmentation_manager.cli.recording`.

The order is the order somebody meets these: put work back first, and reclaim
the space it needed once the round is closed.
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

from ..history import AM_HISTORY_FILENAME, containing_project
from ..io.path import make_path_relative
from ..operations.archive import (
    apply_prune,
    archive_root,
    archived_versions,
    plan_prune,
    total_size,
)
from ..operations.merge import (
    CONVERGED,
    KEEP_OURS,
    OURS,
    TAKE_THEIRS,
    THEIRS,
    MergeRefused,
    apply_merge,
    decisions_for,
    plan_merge,
)
from ..operations.revert import UNDOABLE, apply_revert, plan_revert, plan_revert_to
from ..operations.savepoint import (
    SavepointRefused,
    find_savepoint,
    list_savepoints,
    open_round,
    reached_versions,
)
from ..reporting import format_bytes
from ..scanner import Scanner
from ..state import StateFileError
from .scope import _add_scope_arguments, _projects_in_scope, _recorder, _resolve_scope

logger = logging.getLogger(__name__)


#: The heading ``am-seg --help`` files these under.
TITLE = "Put work back, and reclaim the space"


def add_arguments(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the commands that put work back and manage the archive."""
    # -- merge --------------------------------------------------------------
    merge_cmd = sub.add_parser(
        "merge",
        help="Merge an exported copy back into the collection it came from",
        description=(
            "Merge a copy made by export-files back into its collection. This works one way only: "
            "from the copy into the original. Every label is compared on both sides with the "
            "version recorded in the AM_export.yaml of the copy. A change on one side is taken; a "
            "different change on both sides is a conflict, and the merge stops until each conflict "
            "is decided, by name or by folder, project or lamella. Nothing is deleted and no image "
            "is moved. Shows a preview by default."
        ),
    )
    merge_cmd.add_argument(
        "source", help="The top folder of the copy, with the AM_export.yaml written by export-files"
    )
    merge_cmd.add_argument(
        "--into", required=True, metavar="DIRECTORY",
        help="The collection it came from",
    )
    merge_cmd.add_argument(
        "--resolve", action="append", default=[], metavar="SCOPE=SIDE",
        help=(
            "Decide conflicts, SIDE being 'theirs' or 'ours'. SCOPE is one conflict "
            "('<project>/<label>' or '<project>:<field>'), a folder, project or lamella, or '.' "
            "for all. Repeatable; the narrowest scope decides, and a scope without conflicts is "
            "refused. Both versions are kept in the archive"
        ),
    )
    merge_cmd.add_argument(
        "--apply", action="store_true",
        help="Carry out the merge, instead of showing what it would do",
    )
    merge_cmd.add_argument(
        "--allow-corrected", action="store_true",
        help=(
            "Let the copy overwrite labels marked corrected in this collection, or remove their "
            "corrected state. The current versions are kept, so this can be undone"
        ),
    )

    # -- revert -------------------------------------------------------------
    revert_cmd = sub.add_parser(
        "revert",
        help="Restore a selection of labels to their state before an operation",
        description=(
            "Undo a conversion or a merge for a selection, restoring each label to the version it "
            "replaced. Useful for trying a newer model on proofread labels: convert, check a few, "
            "and undo all at once if needed. Labels without a kept version, or edited since, are "
            "listed and left unchanged. Shows a preview by default."
        ),
    )
    _add_scope_arguments(revert_cmd)
    revert_cmd.add_argument(
        "--to", metavar="SAVEPOINT",
        help=(
            "Restore every label to a savepoint, by id or title, undoing a whole round of work, "
            "and each image's state with its label. Labels the savepoint cannot restore are "
            "listed and left unchanged, with their states"
        ),
    )
    revert_cmd.add_argument(
        "--with-metadata", action="store_true",
        help="With --to, also restore the project metadata fields edited in AM-seg since then",
    )
    revert_cmd.add_argument(
        "--undo", choices=sorted(UNDOABLE), default="conversion",
        help="Which operation to undo (default: conversion)",
    )
    revert_cmd.add_argument(
        "--apply", action="store_true",
        help="Restore the labels, instead of showing what would be restored",
    )
    revert_cmd.add_argument(
        "--confirm-corrected", action="store_true",
        help=(
            "Also restore labels marked corrected. Required for those, as corrected labels are "
            "what this is most likely to undo"
        ),
    )
    revert_cmd.add_argument(
        "--allow-superseded", action="store_true",
        help=(
            "Restore a label even if it was edited after the operation being undone, which "
            "discards those edits"
        ),
    )

    # -- archive ------------------------------------------------------------
    archive_cmd = sub.add_parser(
        "archive",
        help="Report the label versions a project has kept, and their disk usage",
        description=(
            "List the contents of AM_archive. Point it at a project to see the kept versions per "
            "label and their disk usage, or at a label for that label only. Nothing is removed: "
            "use this to decide what to prune."
        ),
    )
    archive_cmd.add_argument(
        "path", nargs="?", default=".",
        help="A project folder, or a label inside one (default: current folder)",
    )

    # -- prune-archive ------------------------------------------------------
    prune_cmd = sub.add_parser(
        "prune-archive",
        help="Remove kept versions that no savepoint needs",
        description=(
            "Remove kept label versions between savepoints. Versions named by a savepoint are "
            "kept, and so is everything since the latest savepoint. What is removed are the "
            "ordinary proofreading saves of earlier rounds, so nothing is removed until a "
            "savepoint exists. Also always kept: the oldest version of a label, the version before "
            "a model overwrote it, the version before a merge, and any version not in the history. "
            "Shows a preview; pass --apply to remove. This is the only command that deletes label "
            "files; the history is never changed."
        ),
    )
    prune_cmd.add_argument(
        "path", nargs="?", default=".", help="The project to prune",
    )
    prune_cmd.add_argument(
        "--start-path", default=None,
        help=(
            "Where to look for savepoints whose versions must be kept (default: the project "
            "itself). Give the collection folder if a savepoint covers more projects than this one"
        ),
    )
    prune_cmd.add_argument(
        "--apply", action="store_true",
        help="Remove them, instead of showing what would be removed",
    )



# ---------------------------------------------------------------------------
# The commands
# ---------------------------------------------------------------------------

def _cmd_merge(args: argparse.Namespace) -> int:
    """Handle ``am-seg merge``."""
    try:
        chosen = _resolutions(args.resolve)
    except ValueError as error:
        logger.error("%s", error)
        return 2

    source, destination = Path(args.source), Path(args.into)
    try:
        plan = plan_merge(source, destination)
        if chosen:
            # A scope names conflicts, so the plan that finds them comes
            # first; the merge itself reads one decision per conflict.
            plan = plan_merge(
                source, destination, resolutions=decisions_for(plan, chosen)
            )
    except MergeRefused as error:
        logger.error("Refused: %s", error)
        return 2
    except ValueError as error:
        logger.error("Nothing was merged: %s", error)
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
            "%d corrected item(s) would be written over or unmarked",
            len(plan.corrected),
        )
    if plan.is_refused:
        logger.error(
            "Nothing was merged. Decide them with --resolve '<scope>=theirs' "
            "or '<scope>=ours', the scope being a conflict, the folder, project "
            "or lamella holding some, or '.' for all; both versions stay in "
            "the archive either way."
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
            "%d corrected item(s) would be written over or unmarked: "
            "%s. Nothing was merged. Re-run with --allow-corrected to take it "
            "anyway; the version here is kept either way.",
            len(plan.corrected),
            ", ".join(item.target for item in plan.corrected[:5]),
        )
        return 2

    try:
        report = apply_merge(
            plan, recorder=_recorder(args), allow_corrected=args.allow_corrected
        )
    except (MergeRefused, StateFileError) as error:
        logger.error("Nothing was merged: %s", error)
        return 2
    except PermissionError as error:
        logger.error("Nothing was merged: %s Re-run with --allow-corrected to go ahead.", error)
        return 2
    logger.info(
        "Merged %d famil(y/ies), %d label(s), %d state(s) and %d field(s); "
        "%d version(s) kept, %d history entr(ies) added",
        report.families_taken, report.labels_taken, report.states_taken,
        report.fields_taken, len(report.archived), report.entries_added,
    )
    return 0


def _resolutions(chosen: list[str]) -> dict[str, str]:
    """Return the scopes decided on the command line, and the side each was given.

    Raises:
        ValueError: If one is not ``<scope>=theirs`` or ``<scope>=ours``.
    """
    resolutions = {}
    for item in chosen:
        scope, _, side = item.rpartition("=")
        if not scope or side not in (THEIRS, OURS):
            raise ValueError(
                f"{item!r} is not a decision: write '<scope>=theirs' or "
                "'<scope>=ours'"
            )
        resolutions[scope] = side
    return resolutions


def _cmd_revert(args: argparse.Namespace) -> int:
    """Handle ``am-seg revert``."""
    records, start_path, projects = _resolve_scope(args)
    if not records:
        logger.info("No matching images found")
        return 2


    if args.with_metadata and not args.to:
        logger.error("--with-metadata restores a savepoint's metadata; name one with --to")
        return 2
    try:
        if args.to:
            point = find_savepoint(_projects_in_scope(projects, start_path), args.to)
            plan = plan_revert_to(records, point, metadata=args.with_metadata)
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
        "%d label(s) can go back, %d left alone; %d state(s) and %d project(s)' metadata",
        len(plan.revertible), len(plan.blocked), len(plan.restorable_states), len(plan.fields),
    )
    if not plan.anything:
        return 0
    if plan.corrected_targets and not args.confirm_corrected:
        logger.error(
            "%d of them are corrected. Re-run with --confirm-corrected to "
            "put those back; what is there now is kept either way.",
            len(plan.corrected_targets),
        )
        return 2
    if not args.apply:
        logger.info("Re-run with --apply to put them back")
        return 0

    report = apply_revert(
        plan, recorder=_recorder(args), allow_corrected=args.confirm_corrected
    )
    logger.info(
        "Put %d label(s), %d state(s) and %d project(s)' metadata back",
        len(report.restored), report.states, report.fields,
    )
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
        target = make_path_relative(path, project)
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
            who = (version.event.by if version.event else None) or "unknown"
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




#: What each of this group's subcommands runs. Registered above and dispatched
#: here, so the two cannot name different sets.
COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "merge": _cmd_merge,
    "revert": _cmd_revert,
    "archive": _cmd_archive,
    "prune-archive": _cmd_prune_archive,
}
