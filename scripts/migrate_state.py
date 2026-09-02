#!/usr/bin/env python3
"""Record in a state file the workflow state a stage's folders currently imply.

Before state was declared in ``AM_state.yaml``, it was implied by which folder
a file sat in: ``deselected/`` or ``corrected/`` beside the images. This
transcribes that into a state file, once, per stage.

**This is a standalone script, not an ``am-seg`` command.** The reference
collection was migrated in full and holds no folder-state stage any more, and
no curation since can create one -- so this is a one-shot job for data that
predates the state file, in the same class as the other scripts here rather
than a step in anyone's workflow. It is kept, rather than deleted, because the
package still *detects* the shape it repairs: ``inspect-layout`` reports an
unmigrated stage, and a state change on one is refused outright, both because
the scanner does not look inside those folders and would otherwise read a
deselected image as selected.

It walks the filesystem itself rather than working from the index, for the same
reason -- and it is the only code anywhere that reads a state from a folder
name.

Nothing moves, nothing is deleted and nothing existing is overwritten, so a run
is undone by deleting the files it wrote. ``--move-files`` then takes the
images and labels back out of the folders they no longer need.

**A stage is migrated whole or not at all.** Writing part of one leaves the
groups it skipped with no entry, and no entry reads as selected -- a deselected
image silently promoted into a training set, which is the one migration failure
with no visible symptom. So a stage holding anything ambiguous is reported and
left alone.

    python scripts/migrate_state.py <collection root> [--dry-run] [--move-files]
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from am_segmentation_manager.io.path import make_path_relative
from am_segmentation_manager.layout import (
    STATE_FOLDER_NAMES,
    ImageSet,
    build_image_sets,
    image_files,
    state_directories,
)
from am_segmentation_manager.models import Project
from am_segmentation_manager.scanner import Scanner
from am_segmentation_manager.state import (
    AM_STATE_FILENAME,
    RecordState,
    StateFileError,
    read_state,
    state_key,
    write_state,
)

logger = logging.getLogger(__name__)

__all__ = [
    "GroupProblem",
    "StageMigration",
    "StageUnnesting",
    "migrate_projects",
    "migrate_stage",
    "stage_group_states",
    "unnest_projects",
    "unnest_stage",
]

#: What a stage needs, independent of whether a dry run carried it out.
MigrationOutcome = Literal[
    "migrated",             # a file should be written, and can be
    "unchanged",            # a file is already there and already says this
    "nothing-to-declare",   # every group is selected, so no file is needed
    "blocked",              # the folders are ambiguous; resolve them first
    "conflict",             # a file is there and says something else
]

#: Why a stage was left alone.
ProblemKind = Literal[
    "inconsistent", "duplicate", "existing", "unreadable", "unmigrated", "collision"
]

#: What un-nesting a stage's files needs.
UnnestingOutcome = Literal[
    "moved",           # files sit in state folders and can come out
    "nothing-nested",  # already flat
    "blocked",         # moving would lose state, or would overwrite
]


@dataclass(frozen=True)
class GroupProblem:
    """One reason a stage cannot be transcribed, or its files moved.

    Attributes:
        key: The group's state-file key, or a filename where the problem is
            not about a group.
        kind: What is wrong.
        detail: What to say to whoever has to resolve it.
    """

    key: str
    kind: ProblemKind
    detail: str


@dataclass(frozen=True)
class StageMigration:
    """What migrating one stage needs, and what was done about it.

    Attributes:
        stage_directory: The directory a state file would sit in.
        declared: The states the file should hold. Empty when every group is
            selected, and when the stage is blocked.
        outcome: What the stage needs, whether or not this run acted on it.
        applied: Whether a file was actually written. Always False for a dry
            run, and for every outcome except ``"migrated"``.
        problems: Why a blocked or conflicting stage was left alone.
    """

    stage_directory: Path
    declared: Mapping[str, RecordState]
    outcome: MigrationOutcome
    applied: bool = False
    problems: tuple[GroupProblem, ...] = ()


def stage_group_states(
    stage_directory: Path,
) -> tuple[dict[str, RecordState], list[GroupProblem]]:
    """Return the state each of the stage's groups is in, and what is ambiguous.

    The label's folder decides where there is one, and the images' own folder
    where there is not. Nothing here resolves a disagreement: a group whose
    images contradict their label, or whose file sits in two state folders at
    once, is returned as a problem instead of as a state.
    """
    image_sets = build_image_sets(stage_directory)
    labels_by_directory = {
        image_set.labels_directory: _locations(image_set.labels_directory)
        for image_set in image_sets
    }

    grouped: dict[str, list[tuple[ImageSet, str, list[Path]]]] = {}
    for image_set in image_sets:
        for name, paths in _locations(image_set.directory).items():
            key = state_key(image_set.labels_directory / name, stage_directory)
            grouped.setdefault(key, []).append((image_set, name, paths))

    states: dict[str, RecordState] = {}
    problems: list[GroupProblem] = []
    for key, members in sorted(grouped.items()):
        image_set, name, _ = members[0]
        label_paths = labels_by_directory[image_set.labels_directory].get(name, [])
        problem = _group_problem(key, members, label_paths)
        if problem is not None:
            problems.append(problem)
            continue
        states[key] = _group_state(members, label_paths)
    return states, problems


def migrate_stage(
    stage_directory: Path, *, dry_run: bool = False
) -> StageMigration:
    """Transcribe one stage's folder states into its state file.

    An existing file is never overwritten: one that already says this is left
    alone, and one that says anything else is reported as a conflict, because
    the only thing that could have written it is somebody's curation.
    """
    states, problems = stage_group_states(stage_directory)
    if problems:
        return StageMigration(stage_directory, {}, "blocked", problems=tuple(problems))

    try:
        existing = read_state(stage_directory)
    except StateFileError as exc:
        return StageMigration(
            stage_directory, {}, "conflict",
            problems=(GroupProblem(AM_STATE_FILENAME, "unreadable", str(exc)),),
        )

    declared = {key: state for key, state in states.items() if state != "selected"}
    if dict(existing.states) == declared:
        return StageMigration(
            stage_directory, declared,
            "unchanged" if existing.fingerprint is not None else "nothing-to-declare",
        )
    if existing.fingerprint is not None:
        return StageMigration(
            stage_directory, declared, "conflict",
            problems=(
                GroupProblem(
                    AM_STATE_FILENAME, "existing",
                    f"{AM_STATE_FILENAME} already declares "
                    f"{len(existing.states)} group(s); leaving it alone",
                ),
            ),
        )

    if dry_run:
        return StageMigration(stage_directory, declared, "migrated")
    write_state(existing, declared)
    logger.debug("Migrated %s: %d group(s) declared", stage_directory, len(declared))
    return StageMigration(stage_directory, declared, "migrated", applied=True)


def migrate_projects(
    projects: Iterable[Project], *, dry_run: bool = False
) -> list[StageMigration]:
    """Transcribe every stage of every project, in index order."""
    return [
        migrate_stage(stage.path, dry_run=dry_run)
        for stage in _stages_of(projects)
    ]


# ---------------------------------------------------------------------------
# Reading state from a folder name -- confined to this module
# ---------------------------------------------------------------------------

def _folder_state(path: Path) -> RecordState:
    """Return the state *path*'s containing folder implies."""
    name = path.parent.name
    return name if name in STATE_FOLDER_NAMES else "selected"  # type: ignore[return-value]


def _locations(directory: Path) -> dict[str, list[Path]]:
    """Return each filename in *directory*, with every state folder holding it.

    A name mapping to more than one path is a file in two states at once.
    """
    found: dict[str, list[Path]] = {}
    for candidate in state_directories(directory):
        for path in image_files(candidate):
            found.setdefault(path.name, []).append(path)
    return found


def _group_state(
    members: list[tuple[ImageSet, str, list[Path]]], label_paths: list[Path]
) -> RecordState:
    """Return a consistent group's state, the label deciding where there is one."""
    if label_paths:
        return _folder_state(label_paths[0])
    return _folder_state(members[0][2][0])


def _group_problem(
    key: str,
    members: list[tuple[ImageSet, str, list[Path]]],
    label_paths: list[Path],
) -> GroupProblem | None:
    """Return why *key* cannot be transcribed, or None if it can."""
    duplicated = [
        paths for _, _, paths in members if len(paths) > 1
    ] + ([label_paths] if len(label_paths) > 1 else [])
    if duplicated:
        folders = sorted(
            {path.parent.name for paths in duplicated for path in paths}
        )
        return GroupProblem(
            key, "duplicate",
            f"the same filename sits in more than one of {', '.join(folders)}",
        )

    state = _group_state(members, label_paths)
    disagreeing = [
        path for _, _, paths in members for path in paths
        if _folder_state(path) != state
    ]
    if disagreeing:
        return GroupProblem(
            key, "inconsistent",
            f"{len(disagreeing)} image(s) are not {state} like their label; "
            "move them to match before migrating",
        )
    return None


def _stages_of(projects: Iterable[Project]):
    """Yield every stage of every project, in index order."""
    for project in projects:
        for lamella in project.lamellae:
            yield from lamella.stages


# ---------------------------------------------------------------------------
# Taking the files back out of the state folders
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StageUnnesting:
    """What un-nesting one stage's files needs, and what was done about it.

    Attributes:
        stage_directory: The stage the files belong to.
        moves: ``(source, destination)`` for every file that would come out of
            a state folder.
        outcome: What the stage needs, whether or not this run acted on it.
        applied: Whether the files were actually moved.
        removed: The state directories cleared, or that a dry run would clear.
        problems: Why a blocked stage was left alone.
    """

    stage_directory: Path
    moves: tuple[tuple[Path, Path], ...]
    outcome: UnnestingOutcome
    applied: bool = False
    removed: tuple[Path, ...] = ()
    problems: tuple[GroupProblem, ...] = ()


def unnest_stage(
    stage_directory: Path,
    *,
    dry_run: bool = False,
    remove_empty_directories: bool = False,
) -> StageUnnesting:
    """Move a stage's files out of ``deselected/`` and ``corrected/``.

    Only once its state is declared. A stage with nested files and no state
    file is refused: there the folders are the only record of what was decided,
    and moving would destroy it rather than un-nest it.

    The whole batch is validated first, so a stage either comes out entirely or
    not at all. A destination that already exists blocks it, since that is a
    file in two states at once and picking one would be a guess.

    *remove_empty_directories* applies whether or not there was anything to
    move, so a stage un-nested by an earlier run can still have its empty
    folders cleared.
    """
    tidy = remove_empty_directories

    moves = _nested_files(stage_directory)
    if not moves:
        return StageUnnesting(
            stage_directory, (), "nothing-nested",
            removed=_empty_state_directories(stage_directory, remove=tidy) if tidy else (),
        )

    if read_state(stage_directory).fingerprint is None:
        return StageUnnesting(
            stage_directory, moves, "blocked",
            problems=(
                GroupProblem(
                    AM_STATE_FILENAME, "unmigrated",
                    "the folders are the only record of this stage's state; "
                    "migrate it before moving anything",
                ),
            ),
        )

    problems = _collisions(moves)
    if problems:
        return StageUnnesting(stage_directory, moves, "blocked", problems=problems)

    if dry_run:
        # The directories a real run would clear are the ones holding nothing
        # but the files about to leave them.
        leaving = {source for source, _ in moves}
        would_remove = (
            _empty_state_directories(stage_directory, remove=False, ignoring=leaving)
            if tidy else ()
        )
        return StageUnnesting(stage_directory, moves, "moved", removed=would_remove)

    for source, destination in moves:
        shutil.move(str(source), str(destination))
    logger.debug("Un-nested %s: %d file(s) moved", stage_directory, len(moves))

    removed = _empty_state_directories(stage_directory, remove=True) if tidy else ()
    return StageUnnesting(
        stage_directory, moves, "moved", applied=True, removed=removed
    )


def unnest_projects(
    projects: Iterable[Project],
    *,
    dry_run: bool = False,
    remove_empty_directories: bool = False,
) -> list[StageUnnesting]:
    """Un-nest every stage of every project, in index order."""
    return [
        unnest_stage(
            stage.path,
            dry_run=dry_run,
            remove_empty_directories=remove_empty_directories,
        )
        for stage in _stages_of(projects)
    ]


def _state_folders(stage_directory: Path) -> list[Path]:
    """Return every state folder of every image and label directory."""
    directories = {
        directory
        for image_set in build_image_sets(stage_directory)
        for directory in (image_set.directory, image_set.labels_directory)
    }
    return [
        candidate
        for directory in sorted(directories)
        for candidate in state_directories(directory)[1:]
    ]


def _nested_files(stage_directory: Path) -> tuple[tuple[Path, Path], ...]:
    """Return every file in a state folder, with where it belongs."""
    return tuple(
        (path, candidate.parent / path.name)
        for candidate in _state_folders(stage_directory)
        for path in image_files(candidate)
    )


def _collisions(moves: tuple[tuple[Path, Path], ...]) -> tuple[GroupProblem, ...]:
    """Return a problem for any destination that is occupied or claimed twice."""
    problems: list[GroupProblem] = []
    claimed: dict[Path, Path] = {}
    for source, destination in moves:
        if destination.exists():
            problems.append(
                GroupProblem(
                    destination.name, "collision",
                    f"{destination} is already there, so {source.parent.name}/ "
                    "holds a second copy",
                )
            )
        elif destination in claimed:
            problems.append(
                GroupProblem(
                    destination.name, "collision",
                    f"{claimed[destination]} and {source} would both become "
                    f"{destination}",
                )
            )
        else:
            claimed[destination] = source
    return tuple(problems)


def _empty_state_directories(
    stage_directory: Path,
    *,
    remove: bool,
    ignoring: set[Path] | None = None,
) -> tuple[Path, ...]:
    """Return the state folders holding nothing, removing them if asked.

    Only empty ones: a directory still holding something is named rather than
    cleared, since nothing here knows what that something is -- on macOS it is
    usually a ``.DS_Store`` the Finder left behind, which is the user's to
    delete, not this package's.

    Args:
        remove: Remove them, rather than only reporting which they are.
        ignoring: Files about to be moved out, so a dry run can say which
            directories a real run would leave empty.
    """
    ignored = ignoring or set()
    empty: list[Path] = []
    for candidate in _state_folders(stage_directory):
        if not candidate.is_dir():
            continue
        remaining = [path for path in candidate.iterdir() if path not in ignored]
        if remaining:
            logger.warning(
                "%s still holds %s; leaving it",
                candidate,
                ", ".join(sorted(path.name for path in remaining[:5]))
                + (" and more" if len(remaining) > 5 else ""),
            )
            continue
        if remove:
            candidate.rmdir()
        empty.append(candidate)
    return tuple(empty)


# ---------------------------------------------------------------------------
# Running it
#
# A stage left alone is named along with the problem that stopped it: "13
# stages migrated" without the other four is the half-answer that sends
# someone looking through the collection by hand.
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for this script."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "start_path", nargs="?", default=".",
        help="Directory to migrate (default: current directory)",
    )
    parser.add_argument(
        "--move-files", action="store_true",
        help=(
            "After the state is declared, move images and labels out of the "
            "deselected/ and corrected/ folders they no longer need. A stage "
            "whose state is not declared yet is left alone, since there the "
            "folders are the only record of it"
        ),
    )
    parser.add_argument(
        "--remove-empty-dirs", action="store_true",
        help=(
            "With --move-files, remove the state folders once they are empty. "
            "One still holding anything is reported, never cleared"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be written and moved without changing anything",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the migration, returning a process exit status.

    Returns 1 when any stage was left alone, so a run that could not finish the
    job is distinguishable from one that had nothing to do.
    """
    args = build_parser().parse_args(argv)
    # On stdout, because a report sent to stderr is one a shell redirect loses;
    # forced, because `basicConfig` does nothing where logging is already set up.
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    start_path = Path(args.start_path)
    projects = Scanner(start_path).build_index().all_projects()
    if not projects:
        logger.error("No projects found under %s", start_path)
        return 2

    results = migrate_projects(projects, dry_run=args.dry_run)
    for result in results:
        location = make_path_relative(result.stage_directory, start_path)
        if result.outcome in ("blocked", "conflict"):
            logger.warning("%s: not migrated", location)
            for problem in result.problems:
                logger.warning("    %s: %s", problem.key, problem.detail)
        elif result.outcome == "migrated":
            logger.info(
                "%s %s: %d group(s) declared",
                "Would migrate" if args.dry_run else "Migrated",
                location, len(result.declared),
            )

    counted = Counter(result.outcome for result in results)
    logger.info(
        "Total: %d stage(s), %d %s, %d already selected throughout, "
        "%d already had a file, %d left alone",
        len(results), counted["migrated"],
        "would be migrated" if args.dry_run else "migrated",
        counted["nothing-to-declare"],
        counted["unchanged"],
        counted["blocked"] + counted["conflict"],
    )
    left_alone = counted["blocked"] + counted["conflict"]

    if args.move_files:
        moved = unnest_projects(
            projects,
            dry_run=args.dry_run,
            remove_empty_directories=args.remove_empty_dirs,
        )
        for result in moved:
            location = make_path_relative(result.stage_directory, start_path)
            if result.outcome == "blocked":
                logger.warning("%s: nothing moved", location)
                for problem in result.problems:
                    logger.warning("    %s: %s", problem.key, problem.detail)
            elif result.outcome == "moved":
                logger.info(
                    "%s %s: %d file(s) out of state folders",
                    "Would move" if args.dry_run else "Moved",
                    location, len(result.moves),
                )
        counted_moves = Counter(result.outcome for result in moved)
        cleared = sum(len(result.removed) for result in moved)
        logger.info(
            "Files: %d stage(s) %s, %d already flat, %d left alone",
            counted_moves["moved"],
            "would be un-nested" if args.dry_run else "un-nested",
            counted_moves["nothing-nested"],
            counted_moves["blocked"],
        )
        if args.remove_empty_dirs:
            logger.info(
                "State folders: %d %s. Any still holding something is "
                "named above and left alone.",
                cleared, "would be removed" if args.dry_run else "removed",
            )
        left_alone += counted_moves["blocked"]

    return 1 if left_alone else 0


if __name__ == "__main__":
    sys.exit(main())
