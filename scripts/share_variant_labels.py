#!/usr/bin/env python3
"""Make an imaging condition share the label of the acquisition it belongs to.

A condition whose files repeat the variant in their filename --
``SEM_TLD/Image_000_TLD.tif`` beside ``SEM_8/Image_000.tif`` -- pairs with no
label, because a label is found by its image's filename. This drops that
suffix, and moves the acquisition's labels from the private label directory
into its family's, ``sem_labels``, which every condition of it resolves to. Only
the family named after the beam is handled, the one every such layout has; a
stage holding another stops the batch. Workflow
state moves with them: a label directory's name is part of its group's key, so
renaming one without rewriting the keys drops every group back to *selected*.

**This is a standalone script, not an ``am-seg`` command.** Two projects in the
reference collection are shaped this way and no naming since repeats it, so the
package reads one identity for an image and its label rather than learning to
strip suffixes.

Previews by default; pass ``--apply``. Nothing is deleted or overwritten: a
rename onto an existing name stops the whole batch, and every rename is written
to a manifest so it can be undone.

    python scripts/share_variant_labels.py <project root> --suffix _TLD [--apply]
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from am_segmentation_manager.families import families_for
from am_segmentation_manager.history import Recorder
from am_segmentation_manager.io.path import make_path_relative
from am_segmentation_manager.layout import (
    classify_image_directory,
    condition_of,
    image_files,
    labels_directory_name,
    shared_labels_directory_name,
    subdirectories_by_name,
)
from am_segmentation_manager.state import RecordState, read_state, write_state

logger = logging.getLogger(__name__)


class ShareConflict(RuntimeError):
    """Every reason the change cannot go ahead, collected so one run reports all.

    Attributes:
        conflicts: One human-readable line per problem.
    """

    def __init__(self, conflicts: Iterable[str]) -> None:
        self.conflicts = list(conflicts)
        super().__init__(f"{len(self.conflicts)} problem(s) stop this change")


@dataclass(frozen=True)
class FileRename:
    """One image file losing the variant suffix its directory already states."""

    source: Path
    destination: Path


@dataclass(frozen=True)
class LabelShare:
    """One private label directory becoming the modality's shared one.

    Attributes:
        source: The directory as it is named now.
        destination: The shared name every condition of the modality resolves to.
        stage_directory: The directory whose state file names these groups.
        states: The declared state per group key, with the moved keys rewritten
            and every other entry carried through untouched.
    """

    source: Path
    destination: Path
    stage_directory: Path
    states: dict[str, RecordState]


@dataclass(frozen=True)
class SharePlan:
    """A validated, not-yet-performed change."""

    renames: tuple[FileRename, ...]
    shares: tuple[LabelShare, ...]

    def describe(self, root: Path) -> list[str]:
        """Return one line per change, for a preview or a log."""
        lines = [
            f"rename {make_path_relative(rename.source, root)} -> {rename.destination.name}"
            for rename in self.renames
        ]
        lines.extend(
            f"share  {make_path_relative(share.source, root)}/ -> "
            f"{share.destination.name}/ ({len(share.states)} state entr"
            f"{'y' if len(share.states) == 1 else 'ies'} rewritten)"
            for share in self.shares
        )
        return lines


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def plan_sharing(project_root: Path, suffix: str) -> SharePlan:
    """Return the renames and directory moves to make, having validated them all.

    Every file in a directory carrying *suffix* must carry it, since a directory
    where only some do is not the shape this handles. A destination that already
    exists stops the batch rather than being written over.

    Raises:
        ShareConflict: If anything at all is wrong, listing all of it.
    """
    conflicts: list[str] = []
    renames: list[FileRename] = []
    stages: dict[Path, set[str]] = {}

    for directory in sorted(path for path in project_root.rglob("*") if path.is_dir()):
        classification = classify_image_directory(directory.name)
        if classification is None:
            continue
        files = image_files(directory)
        carrying = [path for path in files if path.stem.endswith(suffix)]
        if not carrying:
            continue
        if len(carrying) != len(files):
            rest = [path.name for path in files if path not in carrying]
            conflicts.append(
                f"{directory}: only {len(carrying)} of {len(files)} files carry "
                f"{suffix}, starting with {rest[0]}"
            )
            continue

        for path in carrying:
            destination = path.with_name(path.stem[: -len(suffix)] + path.suffix)
            if destination.exists():
                conflicts.append(f"{path}: {destination.name} is already there")
                continue
            renames.append(FileRename(source=path, destination=destination))
        stages.setdefault(directory.parent, set()).add(classification.modality)

    if not renames and not conflicts:
        raise ShareConflict([f"No image file under {project_root} carries {suffix}"])

    shares = _plan_shares(stages, conflicts)
    if conflicts:
        raise ShareConflict(conflicts)
    return SharePlan(renames=tuple(renames), shares=tuple(shares))


def _plan_shares(
    stages: dict[Path, set[str]], conflicts: list[str]
) -> list[LabelShare]:
    """Return the label directories to move, for the stages a rename touched.

    Only those stages: a private label directory elsewhere in the project has no
    second condition to share with, and moving it would be a change nothing here
    asked for. A beam holding a family other than the one named after it stops
    the batch: ``sem_labels`` is that family's, and another's labels moved there
    would join a different segmentation.
    """
    shares: list[LabelShare] = []
    for stage_directory, modalities in sorted(stages.items()):
        siblings = subdirectories_by_name(stage_directory)
        image_directories = [
            directory for name, directory in sorted(siblings.items())
            if classify_image_directory(directory.name) is not None
        ]
        corrections = families_for(stage_directory)
        for modality in sorted(modalities):
            wanted = shared_labels_directory_name(modality)
            others = sorted({
                parsed.family
                for directory in image_directories
                if (parsed := condition_of(directory.name, corrections)) is not None
                and parsed.modality == modality
                and parsed.family.lower() != modality.lower()
            })
            if others:
                conflicts.append(
                    f"{stage_directory}: {modality} holds the family "
                    f"{', '.join(others)} beside {modality}; this shares labels only "
                    f"within the family named after the beam"
                )
                continue
            private = [
                siblings[labels_directory_name(directory.name).lower()]
                for directory in image_directories
                if classify_image_directory(directory.name).modality == modality
                and labels_directory_name(directory.name).lower() in siblings
                and labels_directory_name(directory.name).lower() != wanted
            ]
            if not private:
                continue
            if len(private) > 1:
                listed = ", ".join(path.name for path in private)
                conflicts.append(
                    f"{stage_directory}: several label directories would become "
                    f"{wanted}: {listed}"
                )
                continue
            source = private[0]
            destination = stage_directory / wanted
            if wanted in siblings:
                conflicts.append(
                    f"{stage_directory}: {source.name} and {wanted} both exist; "
                    "merging them is not something this does"
                )
                continue
            states = _rewritten_states(stage_directory, source.name, wanted, conflicts)
            if states is None:
                continue
            shares.append(
                LabelShare(
                    source=source,
                    destination=destination,
                    stage_directory=stage_directory,
                    states=states,
                )
            )
    return shares


def _rewritten_states(
    stage_directory: Path, old: str, new: str, conflicts: list[str]
) -> dict[str, RecordState] | None:
    """Return the stage's declared states with the moved directory's keys renamed.

    A key names its group by the label's path relative to the stage, so the
    directory's name is part of it. Entries naming anything else are carried
    through untouched.
    """
    try:
        declared = read_state(stage_directory)
    except Exception as exc:  # noqa: BLE001 - report alongside everything else
        conflicts.append(f"{stage_directory}: could not read its state file: {exc}")
        return None

    rewritten: dict[str, RecordState] = {}
    for key, value in declared.states.items():
        head, separator, tail = key.partition("/")
        moved = f"{new}/{tail}" if separator and head.lower() == old.lower() else key
        if moved in rewritten:
            conflicts.append(
                f"{stage_directory}: rewriting {key} to {moved} would collide with "
                "an entry already there"
            )
            return None
        rewritten[moved] = value
    return rewritten


# ---------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------

def apply_plan(plan: SharePlan) -> tuple[int, int]:
    """Carry out a validated plan, returning how many files and directories moved.

    Files first, then each label directory with its state file written straight
    after, so an interruption leaves a directory and its state agreeing rather
    than a moved directory whose groups have lost their keys. A move changes
    every group key in the directory, so it is recorded where the project keeps
    a history -- an orphaned history cannot be reattached by hand the way a
    state file can.
    """
    recorder = Recorder.for_run()
    for rename in plan.renames:
        rename.source.rename(rename.destination)
    logger.debug("Renamed %d image file(s)", len(plan.renames))

    for share in plan.shares:
        previous = read_state(share.stage_directory)
        _rename_directory(share.source, share.destination)
        recorder.record_directory_rename(share.source, share.destination)
        # A stage that declared nothing keeps no file: the move changed no
        # group's state, and writing an empty one would invent a declaration.
        if share.states or previous.fingerprint is not None:
            write_state(previous, share.states)
    return len(plan.renames), len(plan.shares)


def _rename_directory(source: Path, destination: Path) -> None:
    """Rename a directory, going via a temporary name for a case-only change.

    A case-insensitive filesystem treats ``FIB_labels`` and ``fib_labels`` as
    one name, so renaming between them directly is either refused or a no-op.
    """
    if source.name != destination.name and source.name.lower() == destination.name.lower():
        staging = source.with_name(f"{source.name}.renaming")
        source.rename(staging)
        staging.rename(destination)
        return
    source.rename(destination)


def write_manifest(path: Path, plan: SharePlan) -> None:
    """Write every rename to a CSV, so the change can be undone."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "from", "to"])
        for rename in plan.renames:
            writer.writerow(["file", str(rename.source), str(rename.destination)])
        for share in plan.shares:
            writer.writerow(["directory", str(share.source), str(share.destination)])


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for this script."""
    parser = argparse.ArgumentParser(
        description=(
            "Drop a variant suffix from image filenames and move the "
            "acquisition's labels into the modality's shared directory, so "
            "every imaging condition of one scene shares one label and one "
            "workflow state."
        ),
        epilog="Previews by default. Nothing is deleted, and nothing is written over.",
    )
    parser.add_argument("project", type=Path, help="Project root to change")
    parser.add_argument(
        "--suffix", required=True,
        help="Filename suffix to drop, before the extension (for example _TLD)",
    )
    parser.add_argument(
        "--manifest", type=Path,
        help=(
            "Where to write the CSV of every rename (default: beside the "
            "working directory, named after the project and the time)"
        ),
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Perform the change; without it the run only reports what it would do",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the change, returning a process exit status."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        plan = plan_sharing(args.project, args.suffix)
    except ShareConflict as exc:
        for conflict in exc.conflicts:
            logger.error("%s", conflict)
        logger.error("Nothing was changed.")
        return 2

    for line in plan.describe(args.project):
        logger.info("%s %s", "" if args.apply else "would", line)
    if not args.apply:
        logger.info(
            "%d file rename(s) and %d label directory move(s). "
            "Re-run with --apply to perform them.",
            len(plan.renames), len(plan.shares),
        )
        return 0

    manifest = args.manifest or Path(
        f"{args.project.name}_share_variant_labels_"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    )
    write_manifest(manifest, plan)
    logger.info("Wrote the manifest to %s", manifest)

    files, directories = apply_plan(plan)
    logger.info("Renamed %d file(s) and moved %d label directory(ies)", files, directories)
    return 0


if __name__ == "__main__":
    sys.exit(main())
