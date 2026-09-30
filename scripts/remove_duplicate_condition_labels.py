#!/usr/bin/env python3
"""Remove the duplicate label and prediction directories splitting a family.

A family's conditions share one label directory, but before families were
resolved as one, predicting each condition wrote each its own: ``SEM_8_labels``
beside ``SEM_8_fi_1_labels``. The survey reports such a family as a conflict.
This keeps one directory per family and removes the others, which leaves the
kept one standing in for the whole family.

**Only what the model can make again is removed.** A label goes only if the last
pixels the project's history records for it were a conversion's, which it still
matches, it is not declared corrected, its image's label survives in the kept
directory, and any state declared for it matches the kept label's. A prediction
goes only if it sits in ``am_seg/`` and that directory's record names it. One
file failing any of this stops the whole run before anything is removed.

The kept label directory is the one holding proofread work; where none does, the
family's own directory, else the one of the family's primary condition. Each
removed label is recorded in the history as ``label-removed``, its state entry
is dropped, and every removal is written to a manifest first.

**This is a standalone script, not an ``am-seg`` command.** The package stopped
writing such directories when it began resolving families; this clears up the
ones written before.

Previews by default; pass ``--apply`` to remove.

    python scripts/remove_duplicate_condition_labels.py <directory> [--apply]
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from am_segmentation_manager.discovery import discover_stages, find_project_roots
from am_segmentation_manager.history import HistoryLog, Recorder, events_for, read_log
from am_segmentation_manager.io.yaml import read_yaml
from am_segmentation_manager.layout import (
    LABELS_SUFFIX,
    OWN_PREDICTION_KIND,
    PREDICTION_PROVENANCE_FILENAME,
    PREDICTIONS_SUFFIX,
    ImageSet,
    build_image_sets,
    conflicting_label_directories,
    conflicting_predictions_directories,
    image_files,
    subdirectories_by_name,
)
from am_segmentation_manager.operations.label_io import label_digest
from am_segmentation_manager.state import read_state, state_key, write_state

logger = logging.getLogger(__name__)

#: Files the operating system leaves in a directory someone opened, and nothing else.
DISPOSABLE_NAMES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})


@dataclass(frozen=True)
class Removal:
    """One file to remove.

    Attributes:
        path: The file.
        kind: ``"label"``, ``"prediction"``, ``"record"`` (an ``am_seg/``
            directory's record of what it predicted) or ``"system"`` (a file
            the operating system left).
        digest: A label's pixel digest when planned, checked again before it goes.
        state_key: A label's declared state entry, dropped with it, or None.
    """

    path: Path
    kind: str
    digest: str | None = None
    state_key: str | None = None


@dataclass
class FamilyRepair:
    """What becomes of one family in one stage.

    Attributes:
        stage: The stage directory.
        family: The family as its base condition spells it.
        kept_labels: The label directory kept, or None where labels were not split.
        kept_predictions: The predictions directory kept, or None likewise.
        removals: Every file to remove, labels first.
        directories: Directories to remove once emptied, innermost first.
        problems: Why this family cannot be repaired, one line each.
    """

    stage: Path
    family: str
    kept_labels: Path | None = None
    kept_predictions: Path | None = None
    removals: list[Removal] = field(default_factory=list)
    directories: list[Path] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def describe(self, root: Path) -> list[str]:
        """Return what the repair would do, one line each."""
        where = self.stage.relative_to(root) if self.stage.is_relative_to(root) else self.stage
        lines = [f"{where}: family {self.family}"]
        lines.extend(
            f"    keep {kept.name}/"
            for kept in (self.kept_labels, self.kept_predictions) if kept is not None
        )
        counts: dict[str, int] = {}
        for removal in self.removals:
            top = removal.path.relative_to(self.stage).parts[0]
            counts[top] = counts.get(top, 0) + 1
        lines.extend(f"    remove {name}/ ({count} file(s))" for name, count in counts.items())
        lines.extend(f"    PROBLEM: {problem}" for problem in self.problems)
        return lines


class RepairRefused(RuntimeError):
    """Some family cannot be repaired, so nothing is removed anywhere."""


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def plan_repairs(start_path: Path) -> list[FamilyRepair]:
    """Return a repair for every family in conflict under *start_path*."""
    recorder = Recorder()
    logs: dict[Path, HistoryLog] = {}
    repairs: list[FamilyRepair] = []
    for project in find_project_roots(start_path):
        for stage in discover_stages(project):
            repairs.extend(_plan_stage(stage.path, recorder, logs))
    return repairs


def _plan_stage(
    stage: Path, recorder: Recorder, logs: dict[Path, HistoryLog]
) -> list[FamilyRepair]:
    """Return a repair for each family in conflict in one stage."""
    siblings = subdirectories_by_name(stage)
    families: dict[tuple[str, str], list[ImageSet]] = {}
    for image_set in build_image_sets(stage):
        if image_set.modality is None or image_set.family is None:
            continue
        families.setdefault((image_set.modality, image_set.family.lower()), []).append(image_set)

    repairs = []
    for (modality, _), members in families.items():
        members.sort(key=lambda member: (member.variant is not None, member.directory.name.lower()))
        primary = members[0]
        labels = conflicting_label_directories(primary.directory, modality, siblings)
        predictions = conflicting_predictions_directories(primary.directory, modality, siblings)
        if not labels and not predictions:
            continue
        repair = FamilyRepair(stage=stage, family=primary.family or primary.directory.name)
        if labels:
            _plan_labels(repair, labels, primary, modality, recorder, logs)
        if predictions:
            _plan_predictions(repair, predictions, primary, modality)
        repairs.append(repair)
    return repairs


def _family_directory_name(primary: ImageSet, modality: str, suffix: str) -> str:
    """Return the lowercased name of the family's own directory of one kind."""
    family = primary.family or primary.directory.name
    return f"{modality if family.lower() == modality.lower() else family}{suffix}".lower()


def _choose(
    candidates: tuple[Path, ...], preferred: list[str]
) -> Path:
    """Return the first of *candidates* named in *preferred*, else the first candidate."""
    by_name = {candidate.name.lower(): candidate for candidate in candidates}
    for name in preferred:
        if name in by_name:
            return by_name[name]
    return candidates[0]


def _untouched(path: Path, recorder: Recorder, logs: dict[Path, HistoryLog]) -> str | None:
    """Return why the label at *path* is not provably untouched model output, or None."""
    root = recorder.project_of(path.resolve())
    if root is None:
        return "its project keeps no history, so where it came from is unknown"
    if root not in logs:
        logs[root] = read_log(root)
    # Only entries recording pixels count: a state change touches none, and
    # state is checked on its own.
    writes = [
        event for event in events_for(logs[root], path.resolve().relative_to(root).as_posix())
        if "digest" in event.data
    ]
    if not writes:
        return "the history records no pixels for it"
    last = writes[-1]
    if last.kind != "label-converted":
        return f"its last recorded write is {last.kind}, not a conversion"
    if last.data.get("digest") != label_digest(path):
        return "its pixels differ from what the conversion wrote"
    return None


def _plan_labels(
    repair: FamilyRepair,
    directories: tuple[Path, ...],
    primary: ImageSet,
    modality: str,
    recorder: Recorder,
    logs: dict[Path, HistoryLog],
) -> None:
    """Choose the label directory to keep and plan removing the others."""
    stage = repair.stage
    declared = read_state(stage).states
    files = {directory: image_files(directory) for directory in directories}
    reasons = {
        path: _untouched(path, recorder, logs)
        for directory in directories for path in files[directory]
    }

    def holds_work(directory: Path) -> bool:
        return any(
            reasons[path] is not None
            or declared.get(state_key(path, stage)) == "corrected"
            for path in files[directory]
        )

    holders = [directory for directory in directories if holds_work(directory)]
    if len(holders) > 1:
        names = ", ".join(f"{directory.name}/" for directory in holders)
        repair.problems.append(f"proofread or unrecorded work in more than one of {names}")
        return
    kept = holders[0] if holders else _choose(
        directories,
        [
            _family_directory_name(primary, modality, LABELS_SUFFIX),
            f"{primary.directory.name}{LABELS_SUFFIX}".lower(),
        ],
    )
    repair.kept_labels = kept
    kept_names = {path.name for path in files[kept]}

    for directory in directories:
        if directory == kept:
            continue
        for path in files[directory]:
            where = f"{directory.name}/{path.name}"
            if reasons[path] is not None:
                repair.problems.append(f"{where}: {reasons[path]}")
                continue
            if path.name not in kept_names:
                repair.problems.append(f"{where}: {kept.name}/ holds no label for its image")
                continue
            key = state_key(path, stage)
            state = declared.get(key)
            if state == "corrected":
                repair.problems.append(f"{where}: declared corrected")
                continue
            kept_state = declared.get(state_key(kept / path.name, stage))
            if state is not None and state != kept_state:
                repair.problems.append(
                    f"{where}: declared {state}, where {kept.name}/{path.name} is "
                    f"{kept_state or 'selected'}"
                )
                continue
            repair.removals.append(
                Removal(path, "label", digest=label_digest(path),
                        state_key=key if state is not None else None)
            )
        _plan_leftovers(repair, directory, set(files[directory]))
        repair.directories.append(directory)


def _plan_predictions(
    repair: FamilyRepair, directories: tuple[Path, ...], primary: ImageSet, modality: str
) -> None:
    """Choose the predictions directory to keep and plan removing the others."""
    preferred = [_family_directory_name(primary, modality, PREDICTIONS_SUFFIX)]
    if repair.kept_labels is not None:
        condition = repair.kept_labels.name[: -len(LABELS_SUFFIX)]
        preferred.insert(0, f"{condition}{PREDICTIONS_SUFFIX}".lower())
    preferred.append(f"{primary.directory.name}{PREDICTIONS_SUFFIX}".lower())
    kept = _choose(directories, preferred)
    repair.kept_predictions = kept

    for directory in directories:
        if directory == kept:
            continue
        own = directory / OWN_PREDICTION_KIND
        if own.is_dir():
            recorded = _recorded_predictions(own)
            for path in sorted(own.iterdir()):
                if path.name in DISPOSABLE_NAMES:
                    repair.removals.append(Removal(path, "system"))
                elif path.is_file() and path.name in recorded:
                    repair.removals.append(Removal(path, "prediction"))
                elif path.name != PREDICTION_PROVENANCE_FILENAME:
                    repair.problems.append(
                        f"{directory.name}/{own.name}/{path.name}: "
                        f"not a prediction its record names"
                    )
            # The record goes last, so an interrupted run leaves it naming what is left.
            if (own / PREDICTION_PROVENANCE_FILENAME).is_file():
                repair.removals.append(Removal(own / PREDICTION_PROVENANCE_FILENAME, "record"))
            repair.directories.append(own)
        _plan_leftovers(repair, directory, {own})
        repair.directories.append(directory)


def _recorded_predictions(own: Path) -> set[str]:
    """Return the filenames an ``am_seg/`` directory's record says it predicted."""
    record = read_yaml(own / PREDICTION_PROVENANCE_FILENAME)
    predictions = record.get("predictions")
    return {str(name) for name in predictions} if isinstance(predictions, dict) else set()


def _plan_leftovers(repair: FamilyRepair, directory: Path, planned: set[Path]) -> None:
    """Plan removing what the system left in *directory*, and refuse anything else."""
    for path in sorted(directory.iterdir()):
        if path in planned:
            continue
        if path.name in DISPOSABLE_NAMES:
            repair.removals.append(Removal(path, "system"))
        else:
            repair.problems.append(f"{directory.name}/{path.name}: not expected here")


# ---------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------

def write_manifest(path: Path, repairs: list[FamilyRepair]) -> None:
    """Write one row per file to remove, naming what was kept in its place."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "family", "kind", "removed", "digest", "kept"])
        for repair in repairs:
            for removal in repair.removals:
                kept = repair.kept_labels if removal.kind == "label" else repair.kept_predictions
                writer.writerow([
                    repair.stage, repair.family, removal.kind, removal.path,
                    removal.digest or "", kept or "",
                ])


def apply_repairs(repairs: list[FamilyRepair], recorder: Recorder) -> int:
    """Carry out every repair and return how many files were removed.

    Raises:
        RepairRefused: If any repair has a problem; nothing is removed then.
    """
    problems = [problem for repair in repairs for problem in repair.problems]
    if problems:
        raise RepairRefused(f"{len(problems)} problem(s); nothing was removed")

    removed = 0
    for repair in repairs:
        dropped: set[str] = set()
        for removal in repair.removals:
            if removal.kind == "label":
                if label_digest(removal.path) != removal.digest:
                    logger.warning("%s changed since it was checked; left as it is", removal.path)
                    continue
                assert repair.kept_labels is not None, "a label is removed only beside a kept one"
                removal.path.unlink()
                recorder.record(
                    removal.path, "label-removed", removed_digest=removal.digest,
                    reason=f"duplicate of {repair.kept_labels.name}/ in family {repair.family}",
                )
                if removal.state_key is not None:
                    dropped.add(removal.state_key)
            else:
                removal.path.unlink()
            removed += 1
        if dropped:
            current = read_state(repair.stage)
            write_state(
                current,
                {key: state for key, state in current.states.items() if key not in dropped},
            )
        for directory in repair.directories:
            try:
                directory.rmdir()
            except OSError:
                logger.warning("%s is not empty; left in place", directory)
    return removed


def main(argv: list[str] | None = None) -> int:
    """Preview or carry out the repair under the given directory."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "start_path", nargs="?", default=".",
        help="Directory to search for projects (default: current directory)",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Remove the duplicates. Without this, what would go is only listed",
    )
    parser.add_argument(
        "--manifest", type=Path,
        help="Where to write the list of files removed (default: the current directory)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    start_path = Path(args.start_path)
    repairs = plan_repairs(start_path)
    if not repairs:
        logger.info("No family under %s has more than one label or predictions directory", start_path)
        return 0
    for repair in repairs:
        for line in repair.describe(start_path):
            logger.info("%s", line)

    problems = sum(len(repair.problems) for repair in repairs)
    files = sum(len(repair.removals) for repair in repairs)
    logger.info("%d family(ies) to repair, %d file(s) to remove, %d problem(s).",
                len(repairs), files, problems)
    if problems:
        logger.error("Nothing can be removed until every problem is resolved.")
        return 2
    if not args.apply:
        logger.info("Re-run with --apply to remove them.")
        return 0

    manifest = args.manifest or Path(
        f"remove_duplicate_condition_labels_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    )
    write_manifest(manifest, repairs)
    logger.info("Wrote the manifest to %s", manifest)
    logger.info("Removed %d file(s)", apply_repairs(repairs, Recorder.for_run()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
