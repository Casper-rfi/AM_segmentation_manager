"""Recording the workflow state a stage's folders currently imply.

A one-shot transcription. Every label group's state is read from where its
files sit and written into the stage's ``AM_state.yaml``. Nothing moves, nothing
is deleted and nothing existing is overwritten, so a run is undone by deleting
the files it wrote.

**A stage is migrated whole or not at all.** Writing part of one leaves the
groups it skipped with no entry, and no entry reads as selected -- a deselected
image silently promoted into a training set, which is the one migration failure
with no visible symptom. So a stage holding anything ambiguous is reported and
left alone, and its folders stay authoritative until someone resolves it.
"""
from __future__ import annotations

import logging
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..layout import (
    RecordState,
    base_path,
    image_files,
    state_directories,
    state_of_path,
    subdirectories_by_name,
)
from ..models import AdaptiveMillingStage, ImageRecord, Project
from ..state import (
    AM_STATE_FILENAME,
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
    """One reason a stage cannot be transcribed.

    Attributes:
        key: The group's state-file key, or the state file's name where the
            problem is with the file rather than a group.
        kind: ``"inconsistent"`` for images disagreeing with their label,
            ``"duplicate"`` for a file present in more than one state folder,
            ``"existing"`` for a state file already declaring something else,
            ``"unreadable"`` for a state file that will not parse.
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
    stage: AdaptiveMillingStage,
) -> tuple[dict[str, RecordState], list[GroupProblem]]:
    """Return the state each of *stage*'s groups is in, and what is ambiguous.

    The label's location decides where there is one, and the images' own
    location where there is not -- the rule that already governs every read.
    Nothing here resolves a disagreement: a group whose images contradict their
    label, or whose file sits in two state folders at once, is returned as a
    problem instead of as a state.
    """
    labels_by_image_directory = {
        image_set.directory: image_set.labels_directory
        for image_set in stage.image_sets
    }
    duplicated = _duplicated_names(stage, labels_by_image_directory)

    grouped: dict[str, list[ImageRecord]] = {}
    for record in stage.images:
        labels_directory = labels_by_image_directory.get(
            base_path(record.image_path).parent
        )
        if labels_directory is None:
            # An image the stage's image sets do not account for cannot be
            # keyed, and guessing a key would put its state on another group.
            logger.warning("No image set for %s; skipping", record.image_path)
            continue
        key = state_key(labels_directory / record.image_path.name, stage.path)
        grouped.setdefault(key, []).append(record)

    states: dict[str, RecordState] = {}
    problems: list[GroupProblem] = []
    for key, records in grouped.items():
        problem = _group_problem(key, records, duplicated)
        if problem is not None:
            problems.append(problem)
            continue
        states[key] = _group_state(records)
    return states, problems


def migrate_stage(
    stage: AdaptiveMillingStage, *, dry_run: bool = False
) -> StageMigration:
    """Transcribe one stage's folder states into its state file.

    An existing file is never overwritten: one that already says this is left
    alone, and one that says anything else is reported as a conflict, because
    the only thing that could have written it is somebody's curation.
    """
    states, problems = stage_group_states(stage)
    if problems:
        return StageMigration(stage.path, {}, "blocked", problems=tuple(problems))

    try:
        existing = read_state(stage.path)
    except StateFileError as exc:
        return StageMigration(
            stage.path, {}, "conflict",
            problems=(GroupProblem(AM_STATE_FILENAME, "unreadable", str(exc)),),
        )

    declared = {key: state for key, state in states.items() if state != "selected"}
    if dict(existing.states) == declared:
        return StageMigration(
            stage.path, declared,
            "unchanged" if existing.fingerprint is not None else "nothing-to-declare",
        )
    if existing.fingerprint is not None:
        return StageMigration(
            stage.path, declared, "conflict",
            problems=(
                GroupProblem(
                    existing.path.name, "existing",
                    f"{existing.path.name} already declares "
                    f"{len(existing.states)} group(s); leaving it alone",
                ),
            ),
        )

    if dry_run:
        return StageMigration(stage.path, declared, "migrated")
    write_state(existing, declared)
    logger.debug("Migrated %s: %d group(s) declared", stage.path, len(declared))
    return StageMigration(stage.path, declared, "migrated", applied=True)


def migrate_projects(
    projects: Iterable[Project], *, dry_run: bool = False
) -> list[StageMigration]:
    """Transcribe every stage of every project, in index order."""
    return [
        migrate_stage(stage, dry_run=dry_run)
        for project in projects
        for lamella in project.lamellae
        for stage in lamella.stages
    ]


# ---------------------------------------------------------------------------
# What makes a group ambiguous
# ---------------------------------------------------------------------------

def _group_state(records: list[ImageRecord]) -> RecordState:
    """Return a consistent group's state, the label deciding where there is one."""
    label = _located_label(records)
    if label is not None:
        return state_of_path(label)
    return state_of_path(records[0].image_path)


def _located_label(records: list[ImageRecord]) -> Path | None:
    """Return the group's label wherever the scan found it, or None."""
    for record in records:
        if record.labels:
            return record.labels[0]
    return None


def _group_problem(
    key: str, records: list[ImageRecord], duplicated: Mapping[Path, set[str]]
) -> GroupProblem | None:
    """Return why *key* cannot be transcribed, or None if it can."""
    label = _located_label(records)
    copies = [
        directory
        for directory, names in duplicated.items()
        if any(
            base_path(record.image_path).parent == directory
            and record.image_path.name in names
            for record in records
        )
        or (label is not None and base_path(label).parent == directory
            and label.name in names)
    ]
    if copies:
        return GroupProblem(
            key, "duplicate",
            "the same filename sits in more than one state folder of "
            + ", ".join(sorted(directory.name for directory in copies)),
        )

    state = _group_state(records)
    disagreeing = [
        record for record in records if state_of_path(record.image_path) != state
    ]
    if disagreeing:
        return GroupProblem(
            key, "inconsistent",
            f"{len(disagreeing)} image(s) are not {state} like their label; "
            "run repair-state first",
        )
    return None


def _duplicated_names(
    stage: AdaptiveMillingStage, labels_by_image_directory: Mapping[Path, Path]
) -> dict[Path, set[str]]:
    """Return, per directory, the filenames present in more than one of its
    state folders.

    Listed per directory rather than probed per file: the three state folders
    of an image set are three listings, where asking each file where else it
    might be is three stat calls per file.
    """
    directories = {
        *labels_by_image_directory.keys(),
        *labels_by_image_directory.values(),
    }
    duplicated: dict[Path, set[str]] = {}
    for directory in directories:
        seen: dict[str, int] = {}
        for candidate in state_directories(directory):
            for path in image_files(candidate):
                seen[path.name] = seen.get(path.name, 0) + 1
        repeated = {name for name, count in seen.items() if count > 1}
        if repeated:
            duplicated[directory] = repeated
    return duplicated


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
    stage: AdaptiveMillingStage,
    *,
    dry_run: bool = False,
    remove_empty_directories: bool = False,
) -> StageUnnesting:
    """Move *stage*'s files out of ``deselected/`` and ``corrected/``.

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

    moves = _nested_files(stage.path)
    if not moves:
        return StageUnnesting(
            stage.path, (), "nothing-nested",
            removed=_empty_state_directories(stage.path, remove=tidy) if tidy else (),
        )

    if read_state(stage.path).fingerprint is None:
        return StageUnnesting(
            stage.path, moves, "blocked",
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
        return StageUnnesting(stage.path, moves, "blocked", problems=problems)

    if dry_run:
        # The directories a real run would clear are the ones holding nothing
        # but the files about to leave them.
        leaving = {source for source, _ in moves}
        would_remove = (
            _empty_state_directories(stage.path, remove=False, ignoring=leaving)
            if tidy else ()
        )
        return StageUnnesting(stage.path, moves, "moved", removed=would_remove)

    for source, destination in moves:
        shutil.move(str(source), str(destination))
    logger.debug("Un-nested %s: %d file(s) moved", stage.path, len(moves))

    removed = _empty_state_directories(stage.path, remove=True) if tidy else ()
    return StageUnnesting(stage.path, moves, "moved", applied=True, removed=removed)


def unnest_projects(
    projects: Iterable[Project],
    *,
    dry_run: bool = False,
    remove_empty_directories: bool = False,
) -> list[StageUnnesting]:
    """Un-nest every stage of every project, in index order."""
    return [
        unnest_stage(
            stage, dry_run=dry_run, remove_empty_directories=remove_empty_directories
        )
        for project in projects
        for lamella in project.lamellae
        for stage in lamella.stages
    ]


def _nested_files(stage_directory: Path) -> tuple[tuple[Path, Path], ...]:
    """Return every file in a state folder, with where it belongs."""
    moves: list[tuple[Path, Path]] = []
    for directory in sorted(subdirectories_by_name(stage_directory).values()):
        for candidate in state_directories(directory)[1:]:
            for path in image_files(candidate):
                moves.append((path, directory / path.name))
    return tuple(moves)


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
    for directory in sorted(subdirectories_by_name(stage_directory).values()):
        for candidate in state_directories(directory)[1:]:
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
