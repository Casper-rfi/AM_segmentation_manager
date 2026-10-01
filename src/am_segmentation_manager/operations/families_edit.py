"""Changing a project's family corrections: the one writer of ``AM_families.yaml``.

A change is worked out in full before anything is written. Every condition in
every stage is resolved under the corrections before and after, and the change
is refused if a directory holding files would be read by no condition, a
condition would read one such directory in place of another, a condition the
change does not name would stop reading one, a family would gain a conflict,
two conditions of a family would share a variant name, or a family a condition
joins would hold two image sizes. Images whose label group changes keep their
curation as :func:`~.selection.plan_regrouping` says.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Final

from ..families import (
    FamiliesFileError,
    StaleFamilies,
    families_file_path,
    read_families,
    write_families,
)
from ..history import Recorder
from ..image_metadata import (
    ImageMetadata,
    condition_disagreements,
    header_reader,
    read_condition_headers,
)
from ..io.atomic import Fingerprint, fingerprint_of
from ..layout import (
    LABELS_SUFFIX,
    PREDICTIONS_SUFFIX,
    FamilyCorrections,
    classify_image_directory,
    condition_of,
    conflicting_label_directories,
    conflicting_predictions_directories,
    correction_problems,
    families_claiming_one_directory,
    image_files_by_name,
    resolve_labels_directory,
    resolve_predictions_directory,
    subdirectories_by_name,
)
from ..models import Project
from ..state import StateFileError
from .selection import (
    Regrouping,
    RegroupingPlan,
    UnmigratedStage,
    apply_regrouping,
    plan_regrouping,
    refuse_stale_regrouping,
)

logger = logging.getLogger(__name__)

__all__ = [
    "FAMILIES_SECTION",
    "PRIMARY_SECTION",
    "ConditionMove",
    "EntryChange",
    "FamiliesChange",
    "FamiliesValidationError",
    "FamilySummary",
    "apply_families_change",
    "plan_corrections",
    "plan_families_change",
    "refuse_stale_change",
    "summarise_families",
    "update_families",
]

#: The two sections of the file, as a ``family-changed`` entry names them.
FAMILIES_SECTION: Final[str] = "families"
PRIMARY_SECTION: Final[str] = "primary"

_Resolver = Callable[..., tuple[Path, bool, bool]]
_Conflicts = Callable[..., tuple[Path, ...]]

#: The two kinds of directory a family shares, and how each resolves.
_SHARED: Final[tuple[tuple[str, _Resolver, _Conflicts], ...]] = (
    ("labels", resolve_labels_directory, conflicting_label_directories),
    ("predictions", resolve_predictions_directory, conflicting_predictions_directories),
)


class FamiliesValidationError(ValueError):
    """Raised when a requested change cannot be made, carrying every reason."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class EntryChange:
    """One entry of the file, before and after.

    Attributes:
        section: :data:`FAMILIES_SECTION` or :data:`PRIMARY_SECTION`.
        key: The condition, or the family, the entry is listed under.
        old: Its value before, or None where there was no entry.
        new: Its value after, or None where the entry is removed.
    """

    section: str
    key: str
    old: str | None
    new: str | None

    def describe(self) -> str:
        """Return a one-line description, for a preview or a log."""
        if self.section == FAMILIES_SECTION:
            before = self.old or "as named"
            after = self.new or "as named"
            return f"{self.key}: family {before} -> {after}"
        return f"primary of {self.key}: {self.old or 'default'} -> {self.new or 'default'}"


@dataclass(frozen=True)
class ConditionMove:
    """One condition, in one stage, whose label directory changes.

    Attributes:
        image_directory: The condition's directory.
        labels_before: The label directory it read before.
        labels_after: The label directory it reads after.
        labels_shared: How many of its images gain a label they did not have,
            by sharing one already in *labels_after*.
    """

    image_directory: Path
    labels_before: Path
    labels_after: Path
    labels_shared: int = 0

    def describe(self, root: Path) -> str:
        """Return a one-line description, relative to the project *root*."""
        where = self.image_directory.relative_to(root).as_posix()
        line = f"{where}: labels {self.labels_before.name}/ -> {self.labels_after.name}/"
        if self.labels_shared:
            line += f" (shares {self.labels_shared} existing label(s))"
        return line


@dataclass(frozen=True)
class FamiliesChange:
    """What a change to a project's families would do, or did.

    Attributes:
        project_root: The project.
        project: Its identity, for reporting.
        before: The corrections as read.
        after: The corrections as they would be written.
        fingerprint: The file as read, so a write over a later edit is refused.
        entries: One per entry that changes.
        moves: One per condition and stage whose label directory changes.
        states: The state entries written so moved images keep their curation.
        warnings: What the change allows but a person should look at: the
            conditions of a family it enlarges disagreeing about pixel size or
            voltage.
        applied: Whether it was written, as opposed to previewed.
    """

    project_root: Path
    project: str
    before: FamilyCorrections
    after: FamilyCorrections
    fingerprint: Fingerprint | None
    entries: tuple[EntryChange, ...] = ()
    moves: tuple[ConditionMove, ...] = ()
    states: RegroupingPlan = field(default_factory=RegroupingPlan)
    warnings: tuple[str, ...] = ()
    applied: bool = False

    @property
    def is_empty(self) -> bool:
        """Return whether nothing would change."""
        return not self.entries

    def describe(self) -> list[str]:
        """Return the change as lines, for a preview."""
        if self.is_empty:
            return [f"{self.project}: no change"]
        lines = [f"{self.project}:"]
        lines.extend(f"  {entry.describe()}" for entry in self.entries)
        lines.extend(f"  {move.describe(self.project_root)}" for move in self.moves)
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        lines.extend(
            f"  {effect.image.relative_to(self.project_root).as_posix()}: "
            f"{effect.old} -> {effect.new}"
            for effect in self.states.effects
        )
        return lines


@dataclass(frozen=True)
class FamilySummary:
    """One family of a scanned project, across its stages.

    Attributes:
        modality: ``"SEM"`` or ``"FIB"``.
        family: The family's name.
        conditions: Each condition's name and how many stages hold it, base
            first.
        moved: The conditions the project's corrections place here.
        primary: The condition the corrections name as primary, or None.
    """

    modality: str
    family: str
    conditions: tuple[tuple[str, int], ...]
    moved: tuple[str, ...] = ()
    primary: str | None = None

    def describe(self) -> list[str]:
        """Return the family as lines, for ``show-families``."""
        listed = ", ".join(
            f"{name}{'*' if name in self.moved else ''} ({count})"
            for name, count in self.conditions
        )
        lines = [f"{self.modality} {self.family}: {listed}"]
        if self.primary is not None:
            lines.append(f"    primary: {self.primary}")
        return lines


def summarise_families(project: Project) -> tuple[FamilySummary, ...]:
    """Return *project*'s families as its scan resolved them, in display order."""
    counts: dict[tuple[str, str], dict[str, int]] = {}
    spelled: dict[tuple[str, str], str] = {}
    base: dict[tuple[str, str], str] = {}
    for lamella in project.lamellae:
        for stage in lamella.stages:
            for image_set in stage.image_sets:
                if image_set.is_loose or image_set.family is None or image_set.modality is None:
                    continue
                key = (image_set.modality, image_set.family.lower())
                spelled.setdefault(key, image_set.family)
                name = image_set.directory.name
                if image_set.variant is None:
                    base[key] = name
                members = counts.setdefault(key, {})
                members[name] = members.get(name, 0) + 1

    summaries = []
    corrections = project.families
    for key in sorted(counts, key=lambda item: (item[0] != "SEM", item[1])):
        members = counts[key]
        ordered = sorted(members, key=lambda name: (name != base.get(key), name.lower()))
        summaries.append(
            FamilySummary(
                modality=key[0],
                family=spelled[key],
                conditions=tuple((name, members[name]) for name in ordered),
                moved=tuple(name for name in ordered if corrections.family_of(name)),
                primary=corrections.primary_of(spelled[key]),
            )
        )
    return tuple(summaries)


def plan_families_change(
    project: Project,
    *,
    families: Mapping[str, str | None] | None = None,
    primary: Mapping[str, str | None] | None = None,
    allow_leaving_corrected: bool = False,
) -> FamiliesChange:
    """Return what changing *project*'s families would do, writing nothing.

    Args:
        project: The project, as scanned; its stages are resolved again from disk.
        families: Condition name to the family it should belong to, or None to
            return it to the family its name gives.
        primary: Family name to its primary condition, or None for the default.
        allow_leaving_corrected: Confirm images leaving a corrected group.

    Raises:
        FamiliesValidationError: Carrying every reason the change is refused.
    """
    before, _ = _read(project.path)
    listings = _listings(project)
    conditions, spellings = _names(listings.values())

    problems: list[str] = []
    after_families = _requested_families(
        before.families, families or {}, conditions, spellings, problems
    )
    after_primary = _requested_primary(
        before.primary, primary or {}, conditions, spellings, problems
    )
    if not problems:
        problems.extend(correction_problems(after_families, after_primary))
    if problems:
        raise FamiliesValidationError(problems)
    return plan_corrections(
        project,
        FamilyCorrections(families=after_families, primary=after_primary),
        allow_leaving_corrected=allow_leaving_corrected,
    )


def plan_corrections(
    project: Project,
    after: FamilyCorrections,
    *,
    allow_leaving_corrected: bool = False,
) -> FamiliesChange:
    """Return what replacing *project*'s corrections with *after* would do.

    What :func:`plan_families_change` does once it has the corrections it was
    asked for, and what a merge does with the corrections it arrived at. A
    condition *after* names that the project does not hold is not refused
    here: in a copy holding part of a project, it is simply not there.

    Raises:
        FamiliesValidationError: Carrying every reason the change is refused.
    """
    root = project.path
    before, fingerprint = _read(root)
    entries = _entry_changes(before, after)
    change = FamiliesChange(
        project_root=root, project=project.display_name,
        before=before, after=after, fingerprint=fingerprint, entries=entries,
    )
    if not entries:
        return change

    listings = _listings(project)
    named = {entry.key.lower() for entry in entries if entry.section == FAMILIES_SECTION}
    problems: list[str] = []
    moves: list[ConditionMove] = []
    regroupings: list[Regrouping] = []
    warnings: list[str] = []
    read = header_reader() if named else None
    if named and read is None:
        warnings.append("image headers were not compared: the image stack is not installed")
    for stage, siblings in listings.items():
        _check_stage(stage, siblings, before, after, named, root, problems, moves, regroupings)
        if read is not None:
            _check_headers(stage, siblings, after, named, root, read, problems, warnings)
    if problems:
        raise FamiliesValidationError(problems)

    try:
        states = plan_regrouping(regroupings)
    except (UnmigratedStage, StateFileError) as error:
        raise FamiliesValidationError([str(error)]) from error
    leaving = [effect for effect in states.effects if effect.old == "corrected"]
    if leaving and not allow_leaving_corrected:
        raise FamiliesValidationError(
            [
                f"{len(leaving)} image(s) would leave a corrected group, starting with "
                f"{leaving[0].image.relative_to(root).as_posix()}; confirm to go ahead"
            ]
        )
    return replace(change, moves=tuple(moves), states=states, warnings=tuple(warnings))


def _read(root: Path) -> tuple[FamilyCorrections, Fingerprint | None]:
    """Return a project's corrections and its file's fingerprint, taken first."""
    fingerprint = fingerprint_of(families_file_path(root))
    try:
        return read_families(root), fingerprint
    except FamiliesFileError as error:
        raise FamiliesValidationError([f"{error}; correct the file by hand first"]) from error


def _listings(project: Project) -> dict[Path, Mapping[str, Path]]:
    """Return each stage of *project* and its subdirectories."""
    return {
        stage.path: subdirectories_by_name(stage.path)
        for lamella in project.lamellae
        for stage in lamella.stages
    }


def apply_families_change(
    change: FamiliesChange, *, recorder: Recorder | None = None
) -> FamiliesChange:
    """Carry out a planned change, or none of it.

    The families file and every stage file are checked against what the plan
    read before either is written.

    Raises:
        StaleFamilies: If the families file changed since the plan read it.
        StaleState: If a stage's state file did.
    """
    if change.is_empty:
        return change
    refuse_stale_change(change)
    path = families_file_path(change.project_root)
    write_families(change.project_root, change.after, change.fingerprint)
    if recorder is not None:
        for entry in change.entries:
            recorder.record(
                path, "family-changed",
                section=entry.section, key=entry.key, old=entry.old, new=entry.new,
            )
    apply_regrouping(change.states, recorder=recorder)
    logger.info(
        "Updated %s: %s", path, "; ".join(entry.describe() for entry in change.entries)
    )
    return replace(change, applied=True)


def refuse_stale_change(change: FamiliesChange) -> None:
    """Raise unless every file *change* writes is as it was read.

    Raises:
        StaleFamilies: If the families file changed since the plan read it.
        StaleState: If a stage's state file did.
    """
    refuse_stale_regrouping(change.states)
    path = families_file_path(change.project_root)
    if fingerprint_of(path) != change.fingerprint:
        raise StaleFamilies(
            f"{path} changed since this change was worked out; look again and repeat it"
        )


def update_families(
    project: Project,
    *,
    families: Mapping[str, str | None] | None = None,
    primary: Mapping[str, str | None] | None = None,
    allow_leaving_corrected: bool = False,
    dry_run: bool = False,
    recorder: Recorder | None = None,
) -> FamiliesChange:
    """Plan a change to *project*'s families and, unless *dry_run*, carry it out.

    Args:
        families: As :func:`plan_families_change`.
        primary: As :func:`plan_families_change`.
        recorder: Where to record the change, or None to record nothing.

    Raises:
        FamiliesValidationError: If the change is refused.
    """
    change = plan_families_change(
        project, families=families, primary=primary,
        allow_leaving_corrected=allow_leaving_corrected,
    )
    if dry_run:
        return change
    return apply_families_change(change, recorder=recorder)


# ---------------------------------------------------------------------------
# Working out the new file
# ---------------------------------------------------------------------------

def _names(listings) -> tuple[dict[str, str], dict[str, str]]:
    """Return the project's condition names, and its family names, by lowercase."""
    conditions: dict[str, str] = {}
    spellings: dict[str, str] = {}
    for siblings in listings:
        for directory in sorted(siblings.values(), key=lambda path: path.name):
            parsed = classify_image_directory(directory.name)
            if parsed is None:
                continue
            conditions.setdefault(directory.name.lower(), directory.name)
            spellings.setdefault(parsed.family.lower(), parsed.family)
    return conditions, spellings


def _without(entries: Mapping[str, str], key: str) -> dict[str, str]:
    """Return *entries* without *key*, compared case-insensitively."""
    return {name: value for name, value in entries.items() if name.lower() != key.lower()}


def _requested_families(
    current: Mapping[str, str],
    requested: Mapping[str, str | None],
    conditions: Mapping[str, str],
    spellings: Mapping[str, str],
    problems: list[str],
) -> dict[str, str]:
    """Return the families section with *requested* applied.

    A condition is named as it is spelled on disk, and a family as the project
    already spells it. An entry giving the family the name already gives is no
    entry.
    """
    result = dict(current)
    for name, family in requested.items():
        condition = conditions.get(name.lower())
        if condition is None:
            problems.append(f"{name} is not an image directory in any stage of this project")
            continue
        result = _without(result, condition)
        if family is None:
            continue
        parsed = classify_image_directory(condition)
        assert parsed is not None, "only image directories are conditions"
        if family.lower() != parsed.family.lower():
            result[condition] = spellings.get(family.lower(), family)
    return result


def _requested_primary(
    current: Mapping[str, str],
    requested: Mapping[str, str | None],
    conditions: Mapping[str, str],
    spellings: Mapping[str, str],
    problems: list[str],
) -> dict[str, str]:
    """Return the primary section with *requested* applied."""
    result = dict(current)
    for family, name in requested.items():
        result = _without(result, family)
        if name is None:
            continue
        condition = conditions.get(name.lower())
        if condition is None:
            problems.append(f"{name} is not an image directory in any stage of this project")
            continue
        result[spellings.get(family.lower(), family)] = condition
    return result


def _entry_changes(before: FamilyCorrections, after: FamilyCorrections) -> tuple[EntryChange, ...]:
    """Return one change per entry that differs, families before primary."""
    changes: list[EntryChange] = []
    for section, old, new in (
        (FAMILIES_SECTION, before.families, after.families),
        (PRIMARY_SECTION, before.primary, after.primary),
    ):
        old_by_key = {key.lower(): (key, value) for key, value in old.items()}
        new_by_key = {key.lower(): (key, value) for key, value in new.items()}
        for lowered in sorted(set(old_by_key) | set(new_by_key)):
            old_key, old_value = old_by_key.get(lowered, (None, None))
            new_key, new_value = new_by_key.get(lowered, (None, None))
            if old_value != new_value:
                key = new_key or old_key
                assert key is not None
                changes.append(EntryChange(section, key, old_value, new_value))
    return tuple(changes)


# ---------------------------------------------------------------------------
# What the change does to one stage
# ---------------------------------------------------------------------------

def _holds_files(directory: Path, cache: dict[Path, bool]) -> bool:
    """Return whether *directory* exists and holds anything at all."""
    if directory not in cache:
        try:
            cache[directory] = directory.is_dir() and any(directory.iterdir())
        except OSError:
            cache[directory] = True
    return cache[directory]


def _check_headers(
    stage: Path,
    siblings: Mapping[str, Path],
    after: FamilyCorrections,
    named: set[str],
    root: Path,
    read: Callable[[Path], ImageMetadata | None],
    problems: list[str],
    warnings: list[str],
) -> None:
    """Compare the headers of each family in *stage* that a named condition joins.

    A different image size is refused, since one label cannot fit both; a
    different pixel size or voltage is allowed and warned about.
    """
    families: dict[tuple[str, str], list[Path]] = {}
    for directory in sorted(siblings.values(), key=lambda path: path.name):
        condition = condition_of(directory.name, after)
        if condition is not None:
            key = (condition.modality, condition.family.lower())
            families.setdefault(key, []).append(directory)
    where = stage.relative_to(root).as_posix() or "."
    for members in families.values():
        if len(members) < 2 or not any(member.name.lower() in named for member in members):
            continue
        readings = read_condition_headers(
            {member.name: image_files_by_name(member) for member in members}, read
        )
        for filename, headers in readings:
            for disagreement in condition_disagreements(headers):
                message = f"{where}: {filename} differs in {disagreement.describe()}"
                if disagreement.blocks_sharing:
                    problems.append(f"{message}; one label cannot fit both")
                else:
                    warnings.append(message)


def _check_stage(
    stage: Path,
    siblings: Mapping[str, Path],
    before: FamilyCorrections,
    after: FamilyCorrections,
    named: set[str],
    root: Path,
    problems: list[str],
    moves: list[ConditionMove],
    regroupings: list[Regrouping],
) -> None:
    """Check one stage under the change, appending what it refuses and what moves."""
    directories = [
        directory for directory in sorted(siblings.values(), key=lambda path: path.name)
        if classify_image_directory(directory.name) is not None
    ]
    if not directories:
        return
    where = stage.relative_to(root).as_posix() or "."
    holds: dict[Path, bool] = {}
    modality = {
        directory: classify_image_directory(directory.name).modality  # type: ignore[union-attr]
        for directory in directories
    }

    labels: dict[str, dict[Path, Path]] = {}
    for kind, resolve, conflicting in _SHARED:
        old_of = {
            directory: resolve(directory, modality[directory], siblings, corrections=before)[0]
            for directory in directories
        }
        new_of = {
            directory: resolve(directory, modality[directory], siblings, corrections=after)[0]
            for directory in directories
        }
        if kind == "labels":
            labels = {"before": old_of, "after": new_of}

        for unread in sorted(set(old_of.values()) - set(new_of.values())):
            if _holds_files(unread, holds):
                problems.append(f"{where}: no condition would read {unread.name}/ any more")
        for directory in directories:
            old, new = old_of[directory], new_of[directory]
            if old == new or not _holds_files(old, holds):
                continue
            if _holds_files(new, holds):
                problems.append(
                    f"{where}: {directory.name} would read {new.name}/ in place of "
                    f"{old.name}/, and both hold {kind}; merging them is not supported"
                )
            elif directory.name.lower() not in named:
                problems.append(
                    f"{where}: {directory.name} would stop reading its {kind} in "
                    f"{old.name}/ because another condition moves"
                )

        reported: set[frozenset[Path]] = set()
        for directory in directories:
            now, was = (
                frozenset(conflicting(directory, modality[directory], siblings, corrections=c))
                for c in (after, before)
            )
            if now and now != was and now not in reported:
                reported.add(now)
                listed = ", ".join(f"{path.name}/" for path in sorted(now))
                problems.append(
                    f"{where}: {directory.name}'s family would have {len(now)} {kind} "
                    f"directories ({listed})"
                )

    claimed_before = {
        stem.lower() for stem, _ in families_claiming_one_directory(directories, corrections=before)
    }
    for stem, families in families_claiming_one_directory(directories, corrections=after):
        if stem.lower() not in claimed_before:
            problems.append(
                f"{where}: families {' and '.join(families)} would both read "
                f"{stem}{LABELS_SUFFIX}/ and {stem}{PREDICTIONS_SUFFIX}/, joining their labels. "
                f"A condition moved out of a family keeps the folders named after it, "
                f"here that family's own: move its other conditions out of it first, "
                f"or rename the folder"
            )

    seen: dict[tuple[str, str, str | None], str] = {}
    for directory in directories:
        condition = condition_of(directory.name, after)
        assert condition is not None
        key = (
            condition.modality,
            condition.family.lower(),
            condition.variant.lower() if condition.variant is not None else None,
        )
        if key in seen:
            problems.append(
                f"{where}: {seen[key]} and {directory.name} would both be "
                f"{condition.family} [{condition.display_variant}]"
            )
        seen[key] = directory.name

    old_of, new_of = labels["before"], labels["after"]
    readers_before: dict[Path, set[Path]] = {}
    readers_after: dict[Path, set[Path]] = {}
    for directory in directories:
        readers_before.setdefault(old_of[directory], set()).add(directory)
        readers_after.setdefault(new_of[directory], set()).add(directory)
    for directory in directories:
        old, new = old_of[directory], new_of[directory]
        if old == new:
            continue
        images = image_files_by_name(directory)
        gained = 0
        if _holds_files(new, holds) and not _holds_files(old, holds):
            gained = len(set(images) & set(image_files_by_name(new)))
        moves.append(
            ConditionMove(
                image_directory=directory, labels_before=old, labels_after=new,
                labels_shared=gained,
            )
        )
        staying = readers_before.get(new, set()) & readers_after.get(new, set())
        regroupings.append(
            Regrouping(
                image_directory=directory,
                image_names=tuple(sorted(images)),
                old_labels=old,
                new_labels=new,
                old_group_remains=bool(readers_after.get(old, set()) - {directory}),
                new_group_exists=bool(staying - {directory}),
            )
        )
