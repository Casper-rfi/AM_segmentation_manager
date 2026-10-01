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
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Final

from ..families import (
    FAMILIES_SECTION,
    FamiliesFileError,
    StaleFamilies,
    families_file_path,
    read_families,
    sections_of,
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
    NO_CORRECTIONS,
    PREDICTIONS_SUFFIX,
    FamilyCorrections,
    FamilyEntries,
    ProjectFamilies,
    classify_image_directory,
    condition_in,
    condition_of,
    correction_problems,
    families_of_stage,
    family_directory_problems,
    family_key,
    image_files_by_name,
    primary_order,
    resolve_labels_directory,
    resolve_predictions_directory,
    subdirectories_by_name,
)
from ..models import Lamella, Project
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
    "ConditionMove",
    "EntryChange",
    "FamiliesChange",
    "FamiliesValidationError",
    "FamilySummary",
    "apply_families_change",
    "describe_lamella_sections",
    "entry_changes",
    "plan_corrections",
    "plan_families_change",
    "refuse_stale_change",
    "requested_corrections",
    "section_key",
    "summarise_families",
    "update_families",
]

#: The two kinds of directory a family shares, and how each resolves.
_SHARED: Final[tuple[tuple[str, Callable[..., tuple[Path, bool, bool]]], ...]] = (
    ("labels", resolve_labels_directory),
    ("predictions", resolve_predictions_directory),
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
        lamella: The lamella whose section holds it, by its path relative to
            the project, or None for the project's own.
    """

    section: str
    key: str
    old: str | None
    new: str | None
    lamella: str | None = None

    def describe(self) -> str:
        """Return a one-line description, for a preview or a log."""
        if self.section == FAMILIES_SECTION:
            absent = "as the project has it" if self.lamella else "as named"
            line = f"{self.key}: family {self.old or absent} -> {self.new or absent}"
        else:
            absent = "the project's" if self.lamella else "default"
            line = f"primary of {self.key}: {self.old or absent} -> {self.new or absent}"
        return f"{line}, in {self.lamella}" if self.lamella else line


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
        before: The corrections as read, lamella sections included.
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
    before: ProjectFamilies
    after: ProjectFamilies
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
        moved: The conditions the corrections place here.
        primary: The condition the corrections name as primary, or None.
        standing: The condition standing for the family: the named primary
            where it is among *conditions*, else the first in
            :func:`~..layout.primary_order`.
    """

    modality: str
    family: str
    conditions: tuple[tuple[str, int], ...]
    moved: tuple[str, ...] = ()
    primary: str | None = None
    standing: str | None = None

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


def summarise_families(
    project: Project,
    corrections: ProjectFamilies | None = None,
    *,
    lamella: str | None = None,
) -> tuple[FamilySummary, ...]:
    """Return *project*'s families under *corrections*, in display order.

    Each stage is read under the corrections in force there, so a condition a
    lamella's section moves is listed in both families. Nothing is read from
    disk: the conditions are the scan's.

    Args:
        corrections: The corrections to read the project under; its own, as
            scanned, by default.
        lamella: The section key whose primaries are shown, or None for the
            project's.
    """
    corrections = project.families if corrections is None else corrections
    counts: dict[tuple[str, str], dict[str, int]] = {}
    spelled: dict[tuple[str, str], str] = {}
    base: dict[tuple[str, str], str] = {}
    for stage in (stage for own in project.lamellae for stage in own.stages):
        in_force = project.families_at(stage.path, corrections)
        for image_set in stage.image_sets:
            if image_set.is_loose or image_set.modality is None:
                continue
            name = image_set.directory.name
            condition = condition_of(name, in_force)
            if condition is None:
                continue
            key = (condition.modality, family_key(condition.family))
            spelled.setdefault(key, condition.family)
            if condition.variant is None:
                base[key] = name
            members = counts.setdefault(key, {})
            members[name] = members.get(name, 0) + 1

    # Placed by the file: by the project's entry, or by a lamella's naming this family.
    sections: list[FamilyCorrections | FamilyEntries] = [
        corrections.project, *corrections.lamellae.values()
    ]
    placed = {
        (name.lower(), family_key(family))
        for section in sections
        for name, family in section.families.items()
    }
    named_in = corrections.at(lamella) if lamella is not None else corrections.project
    summaries = []
    for key in sorted(counts, key=lambda item: (item[0] != "SEM", item[1])):
        members = counts[key]
        ordered = sorted(members, key=lambda name: (name != base.get(key), name.lower()))
        family = spelled[key]
        named = named_in.primary_of(family)
        standing = next(
            (name for name in ordered if named is not None and name.lower() == named.lower()),
            None,
        ) or min(ordered, key=lambda name: _primary_order_in(name, family))
        summaries.append(
            FamilySummary(
                modality=key[0],
                family=family,
                conditions=tuple((name, members[name]) for name in ordered),
                moved=tuple(
                    name for name in ordered
                    if corrections.project.family_of(name) or (name.lower(), key[1]) in placed
                ),
                primary=named,
                standing=standing,
            )
        )
    return tuple(summaries)


def _primary_order_in(name: str, family: str) -> tuple[bool, str]:
    """Return where *name* sorts for *family*'s default primary, as a member of it."""
    condition = condition_in(name, family)
    return primary_order(condition.variant if condition is not None else name)


def describe_lamella_sections(project: Project) -> list[str]:
    """Return each lamella's own entries, a line each, for ``show-families``."""
    lines = []
    for key, entries in sorted(project.families.lamellae.items()):
        listed = [f"{name} in {family}" for name, family in sorted(entries.families.items())]
        listed += [f"primary of {family}: {name}" for family, name in sorted(entries.primary.items())]
        if listed:
            lines.append(f"lamella {key}: {'; '.join(listed)}")
    return lines


def plan_families_change(
    project: Project,
    *,
    families: Mapping[str, str | None] | None = None,
    primary: Mapping[str, str | None] | None = None,
    lamella: Lamella | None = None,
    allow_leaving_corrected: bool = False,
) -> FamiliesChange:
    """Return what changing *project*'s families would do, writing nothing.

    Args:
        project: The project, as scanned; its stages are resolved again from disk.
        families: As :func:`requested_corrections`.
        primary: As :func:`requested_corrections`.
        lamella: One of *project*'s lamellae, to change its entries over the
            project's (``test_a_lamella_entry_is_set_and_moves_that_lamella_alone``);
            None for the project's own.
        allow_leaving_corrected: Confirm images leaving a corrected group.

    Raises:
        FamiliesValidationError: Carrying every reason the change is refused.
    """
    before, fingerprint = _read(project.path)
    names = _condition_names(_listings(project).values())
    if lamella is None:
        key, here, where = None, names, "this project"
    else:
        key, where = section_key(project, lamella), lamella.name
        here = _condition_names(subdirectories_by_name(stage.path) for stage in lamella.stages)
    after = requested_corrections(
        before, names, here, families=families, primary=primary, lamella=key, where=where
    )
    return _plan(project, before, fingerprint, after, allow_leaving_corrected)


def requested_corrections(
    before: ProjectFamilies,
    names: Iterable[str],
    here: Iterable[str],
    *,
    families: Mapping[str, str | None] | None = None,
    primary: Mapping[str, str | None] | None = None,
    lamella: str | None = None,
    where: str = "this project",
) -> ProjectFamilies:
    """Return *before* with a change to one section applied, the project's or a lamella's.

    A condition is named as it is spelled on disk, and a family as the project
    already spells it. An entry giving what the section would give anyway --
    the name's family, or in a lamella the project's -- is no entry. What the
    Families tab shows and what ``edit-families`` writes, so the two agree.

    Args:
        names: Every condition directory's name in the project, which spell
            its families.
        here: The names a change may refer to: the project's, or the
            lamella's.
        families: Condition name to the family it should belong to, or None to
            return it to the family its name gives -- or, in a lamella, the
            one the project gives it.
        primary: Family name to its primary condition, or None for the default
            -- or, in a lamella, the project's.
        lamella: The key of the lamella whose section changes, or None for the
            project's.
        where: How a refusal names the project or lamella.

    Raises:
        FamiliesValidationError: Carrying every reason the change is refused.
    """
    _, spellings = _names(names)
    conditions, _ = _names(here)
    problems: list[str] = []
    if lamella is None:
        after_families = _requested_families(
            before.project.families, families or {}, conditions, spellings, problems
        )
        after_primary = _requested_primary(
            before.project.primary, primary or {}, conditions, spellings, problems
        )
        if not problems:
            problems.extend(correction_problems(after_families, after_primary))
        if problems:
            raise FamiliesValidationError(problems)
        project_after = FamilyCorrections(families=after_families, primary=after_primary)
        lamellae_after = dict(before.lamellae)
    else:
        entries = before.lamellae.get(lamella, FamilyEntries())
        section = FamilyEntries(
            families=_requested_families(
                entries.families, families or {}, conditions, spellings, problems,
                baseline=before.project, where=where,
            ),
            primary=_requested_primary(
                entries.primary, primary or {}, conditions, spellings, problems,
                baseline=before.project, where=where,
            ),
        )
        if problems:
            raise FamiliesValidationError(problems)
        project_after = before.project
        lamellae_after = {k: v for k, v in before.lamellae.items() if k != lamella}
        if section.families or section.primary:
            lamellae_after[lamella] = section
    try:
        return ProjectFamilies(project=project_after, lamellae=lamellae_after)
    except ValueError as error:
        raise FamiliesValidationError([str(error)]) from error


def section_key(project: Project, lamella: Lamella) -> str:
    """Return *lamella*'s key in the file, refusing one that cannot have a section.

    Raises:
        FamiliesValidationError: If *lamella* has no folder of its own or is
            not *project*'s.
    """
    if lamella.synthetic or not any(own is lamella for own in project.lamellae):
        reason = (
            "has no folder of its own; the project's entries already speak for it"
            if lamella.synthetic else "is not one of this project's lamellae"
        )
        raise FamiliesValidationError([f"{lamella.name} {reason}"])
    return lamella.path.relative_to(project.path).as_posix()


def plan_corrections(
    project: Project,
    after: ProjectFamilies,
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
    before, fingerprint = _read(project.path)
    return _plan(project, before, fingerprint, after, allow_leaving_corrected)


def _plan(
    project: Project,
    before: ProjectFamilies,
    fingerprint: Fingerprint | None,
    after: ProjectFamilies,
    allow_leaving_corrected: bool,
) -> FamiliesChange:
    """Return what replacing *before*, read with *fingerprint*, by *after* would do.

    The file is read once per plan, so the fingerprint a write is checked
    against is that of the corrections the change was worked out from
    (``test_a_file_changed_while_a_change_is_planned_is_not_written_over``).
    """
    root = project.path
    entries = entry_changes(before, after)
    change = FamiliesChange(
        project_root=root, project=project.display_name,
        before=before, after=after, fingerprint=fingerprint, entries=entries,
    )
    if not entries:
        return change

    listings = _listings(project)
    named = {entry.key.lower() for entry in entries if entry.section == FAMILIES_SECTION}
    checked = _StageCheck()
    read = header_reader() if named else None
    if named and read is None:
        checked.warnings.append("image headers were not compared: the image stack is not installed")
    for stage, siblings in listings.items():
        relative = stage.relative_to(root).as_posix()
        where = relative or "."
        stage_before, stage_after = before.at(relative), after.at(relative)
        checked.extend(_check_stage(stage, siblings, stage_before, stage_after, named, where))
        if read is not None:
            checked.extend(_check_headers(siblings, stage_after, named, where, read))
    problems, moves, regroupings, warnings = (
        checked.problems, checked.moves, checked.regroupings, checked.warnings
    )
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


def _read(root: Path) -> tuple[ProjectFamilies, Fingerprint | None]:
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
                **({"lamella": entry.lamella} if entry.lamella is not None else {}),
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
    lamella: Lamella | None = None,
    allow_leaving_corrected: bool = False,
    dry_run: bool = False,
    recorder: Recorder | None = None,
) -> FamiliesChange:
    """Plan a change to *project*'s families and, unless *dry_run*, carry it out.

    Args:
        families: As :func:`plan_families_change`.
        primary: As :func:`plan_families_change`.
        lamella: As :func:`plan_families_change`.
        recorder: Where to record the change, or None to record nothing.

    Raises:
        FamiliesValidationError: If the change is refused.
    """
    change = plan_families_change(
        project, families=families, primary=primary, lamella=lamella,
        allow_leaving_corrected=allow_leaving_corrected,
    )
    if dry_run:
        return change
    return apply_families_change(change, recorder=recorder)


# ---------------------------------------------------------------------------
# Working out the new file
# ---------------------------------------------------------------------------

def _condition_names(listings: Iterable[Mapping[str, Path]]) -> list[str]:
    """Return the names of the image directories in *listings*, each stage's in order."""
    return [
        directory.name
        for siblings in listings
        for directory in sorted(siblings.values(), key=lambda path: path.name)
        if classify_image_directory(directory.name) is not None
    ]


def _names(names: Iterable[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Return condition names, and the family names they give, by lowercase, first spelling kept."""
    conditions: dict[str, str] = {}
    spellings: dict[str, str] = {}
    for name in names:
        parsed = classify_image_directory(name)
        if parsed is None:
            continue
        conditions.setdefault(name.lower(), name)
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
    *,
    baseline: FamilyCorrections = NO_CORRECTIONS,
    where: str = "this project",
) -> dict[str, str]:
    """Return a families section with *requested* applied.

    A condition is named as it is spelled on disk, and a family as the project
    already spells it. An entry giving the family *baseline* already gives --
    the name's, or for a lamella the project's -- is no entry.
    """
    result = dict(current)
    for name, family in requested.items():
        condition = conditions.get(name.lower())
        if condition is None:
            problems.append(f"{name} is not an image directory in any stage of {where}")
            continue
        result = _without(result, condition)
        if family is None:
            continue
        parsed = classify_image_directory(condition)
        assert parsed is not None, "only image directories are conditions"
        given = baseline.family_of(condition) or parsed.family
        if family.lower() != given.lower():
            result[condition] = spellings.get(family.lower(), family)
    return result


def _requested_primary(
    current: Mapping[str, str],
    requested: Mapping[str, str | None],
    conditions: Mapping[str, str],
    spellings: Mapping[str, str],
    problems: list[str],
    *,
    baseline: FamilyCorrections = NO_CORRECTIONS,
    where: str = "this project",
) -> dict[str, str]:
    """Return a primary section with *requested* applied, an entry *baseline* already makes being none."""
    result = dict(current)
    for family, name in requested.items():
        result = _without(result, family)
        if name is None:
            continue
        condition = conditions.get(name.lower())
        if condition is None:
            problems.append(f"{name} is not an image directory in any stage of {where}")
            continue
        if (baseline.primary_of(family) or "").lower() != condition.lower():
            result[spellings.get(family.lower(), family)] = condition
    return result


def entry_changes(before: ProjectFamilies, after: ProjectFamilies) -> tuple[EntryChange, ...]:
    """Return one change per entry that differs: the project's, then each lamella's."""
    changes = list(_section_changes(before.project, after.project, None))
    empty = FamilyEntries()
    for key in sorted(set(before.lamellae) | set(after.lamellae)):
        changes += _section_changes(
            before.lamellae.get(key, empty), after.lamellae.get(key, empty), key
        )
    return tuple(changes)


def _section_changes(
    before: FamilyCorrections | FamilyEntries,
    after: FamilyCorrections | FamilyEntries,
    lamella: str | None,
) -> list[EntryChange]:
    """Return one change per entry of one section that differs, families before primary."""
    changes: list[EntryChange] = []
    for (section, old), (_, new) in zip(sections_of(before), sections_of(after), strict=True):
        old_by_key = {key.lower(): (key, value) for key, value in old.items()}
        new_by_key = {key.lower(): (key, value) for key, value in new.items()}
        for lowered in sorted(set(old_by_key) | set(new_by_key)):
            old_key, old_value = old_by_key.get(lowered, (None, None))
            new_key, new_value = new_by_key.get(lowered, (None, None))
            if old_value != new_value:
                key = new_key or old_key
                assert key is not None
                changes.append(EntryChange(section, key, old_value, new_value, lamella))
    return changes


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


@dataclass
class _StageCheck:
    """What checking stages under a change found: refusals, moves, regroupings and warnings."""

    problems: list[str] = field(default_factory=list)
    moves: list[ConditionMove] = field(default_factory=list)
    regroupings: list[Regrouping] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def extend(self, other: _StageCheck) -> None:
        """Add what *other* found to this."""
        self.problems += other.problems
        self.moves += other.moves
        self.regroupings += other.regroupings
        self.warnings += other.warnings


def _check_headers(
    siblings: Mapping[str, Path],
    after: FamilyCorrections,
    named: set[str],
    where: str,
    read: Callable[[Path], ImageMetadata | None],
) -> _StageCheck:
    """Compare the headers of each family in a stage that a named condition joins.

    A different image size is refused, since one label cannot fit both; a
    different pixel size or voltage is allowed and warned about.
    """
    checked = _StageCheck()
    for members in families_of_stage(siblings.values(), corrections=after):
        if len(members) < 2 or not any(member.name.lower() in named for member in members):
            continue
        readings = read_condition_headers(
            {member.name: image_files_by_name(member) for member in members}, read
        )
        for filename, headers in readings:
            for disagreement in condition_disagreements(headers):
                message = f"{where}: {filename} differs in {disagreement.describe()}"
                if disagreement.blocks_sharing:
                    checked.problems.append(f"{message}; one label cannot fit both")
                else:
                    checked.warnings.append(message)
    return checked


def _check_stage(
    stage: Path,
    siblings: Mapping[str, Path],
    before: FamilyCorrections,
    after: FamilyCorrections,
    named: set[str],
    where: str,
) -> _StageCheck:
    """Check one stage under the change: what it refuses, and which conditions move."""
    checked = _StageCheck()
    directories = [
        directory for directory in sorted(siblings.values(), key=lambda path: path.name)
        if classify_image_directory(directory.name) is not None
    ]
    if not directories:
        return checked
    holds: dict[Path, bool] = {}
    modality = {
        directory: classify_image_directory(directory.name).modality  # type: ignore[union-attr]
        for directory in directories
    }

    def read_by(resolve, corrections: FamilyCorrections) -> dict[Path, Path]:
        return {
            directory: resolve(directory, modality[directory], siblings, corrections=corrections)[0]
            for directory in directories
        }

    resolved = {
        kind: (read_by(resolve, before), read_by(resolve, after)) for kind, resolve in _SHARED
    }
    for kind, (old_of, new_of) in resolved.items():
        for unread in sorted(set(old_of.values()) - set(new_of.values())):
            if _holds_files(unread, holds):
                checked.problems.append(f"{where}: no condition would read {unread.name}/ any more")
        for directory in directories:
            old, new = old_of[directory], new_of[directory]
            if old == new or not _holds_files(old, holds):
                continue
            if _holds_files(new, holds):
                checked.problems.append(
                    f"{where}: {directory.name} would read {new.name}/ in place of "
                    f"{old.name}/, and both hold {kind}; merging them is not supported"
                )
            elif directory.name.lower() not in named:
                checked.problems.append(
                    f"{where}: {directory.name} would stop reading its {kind} in "
                    f"{old.name}/ because another condition moves"
                )

    existing = {
        problem.key
        for problem in family_directory_problems(directories, siblings, corrections=before)
    }
    for problem in family_directory_problems(directories, siblings, corrections=after):
        if problem.key in existing:
            continue
        if problem.kind == "shared":
            checked.problems.append(
                f"{where}: families {' and '.join(problem.families)} would both read "
                f"{problem.stem}{LABELS_SUFFIX}/ and {problem.stem}{PREDICTIONS_SUFFIX}/, "
                f"joining their labels. A condition moved out of a family keeps the "
                f"folders named after it, here that family's own: move its other "
                f"conditions out of it first, or rename the folder"
            )
        else:
            listed = ", ".join(f"{path.name}/" for path in sorted(problem.directories))
            checked.problems.append(
                f"{where}: {problem.condition}'s family would have "
                f"{len(problem.directories)} {problem.kind} directories ({listed})"
            )

    seen: dict[tuple[str, str, str | None], str] = {}
    for directory in directories:
        condition = condition_of(directory.name, after)
        assert condition is not None
        key = (
            condition.modality,
            family_key(condition.family),
            condition.variant.lower() if condition.variant is not None else None,
        )
        if key in seen:
            checked.problems.append(
                f"{where}: {seen[key]} and {directory.name} would both be "
                f"{condition.family} [{condition.display_variant}]"
            )
        seen[key] = directory.name

    old_of, new_of = resolved["labels"]
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
        checked.moves.append(
            ConditionMove(
                image_directory=directory, labels_before=old, labels_after=new,
                labels_shared=gained,
            )
        )
        staying = readers_before.get(new, set()) & readers_after.get(new, set())
        checked.regroupings.append(
            Regrouping(
                image_directory=directory,
                image_names=tuple(sorted(images)),
                old_labels=old,
                new_labels=new,
                old_group_remains=bool(readers_after.get(old, set()) - {directory}),
                new_group_exists=bool(staying - {directory}),
            )
        )
    return checked
