"""The families view model: one project's families of imaging conditions, and changes to them.

Wraps :mod:`am_segmentation_manager.operations.families_edit` so a widget can show
which conditions form which family, collect changes, preview and apply them,
without knowing how ``AM_families.yaml`` is read or written. Changes are held here
until saved, and the tree shows them as if they were; every rule about whether one
may be made is the operation's. Families are a project's, so a scope of several
projects shows none. A change is the project's or the lamella in view's own,
one of the two at a time. Qt-free, like the other controllers.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from ..families import FAMILIES_SECTION, FamiliesFileError, read_families
from ..history import Recorder
from ..layout import (
    FamilyCorrections,
    ProjectFamilies,
    classify_image_directory,
    condition_of,
    family_key,
)
from ..models import Lamella, Project
from ..operations.families_edit import (
    FamiliesChange,
    FamiliesValidationError,
    FamilySummary,
    apply_families_change,
    entry_changes,
    plan_families_change,
    requested_corrections,
    summarise_families,
)
from ..operations.init_history import writing_recorder
from .attribution import Attribution


@dataclass(frozen=True)
class ConditionRow:
    """One condition, as the tree lists it under its family.

    Attributes:
        name: The image directory's name.
        stages: How many of the project's stages hold it.
        moved: Whether it sits in another family than its name gives.
        primary: Whether it stands for the family: named so, or chosen by the
            default rule where none is named.
        named_primary: Whether it is named as the primary.
        unsaved: Whether a change made here and not yet saved put it where it
            is, or made it the primary.
        here: Whether the lamella in view holds it; True where none is.
    """

    name: str
    stages: int
    moved: bool = False
    primary: bool = False
    named_primary: bool = False
    unsaved: bool = False
    here: bool = True

    @property
    def note(self) -> str:
        """Return what the tree says about it beyond its name and stages."""
        parts = []
        if self.primary:
            parts.append("primary" if self.named_primary else "primary (default)")
        if self.moved:
            parts.append("moved by AM_families.yaml")
        if self.unsaved:
            parts.append("not saved")
        if not self.here:
            parts.append("not in this lamella")
        return ", ".join(parts)

    @property
    def marks(self) -> str:
        """Return :attr:`note` in the fewest words, the primary being marked otherwise."""
        parts = []
        if self.moved:
            parts.append("moved")
        if self.unsaved:
            parts.append("not saved")
        if not self.here:
            parts.append("not here")
        return ", ".join(parts)


@dataclass(frozen=True)
class FamilyNode:
    """One family of one beam, and its conditions in the order the window lists them.

    Attributes:
        modality: ``"SEM"`` or ``"FIB"``.
        family: The family's name, as its conditions spell it.
        conditions: Base first, then by name.
    """

    modality: str
    family: str
    conditions: tuple[ConditionRow, ...]

    @property
    def primary(self) -> ConditionRow | None:
        """Return the condition standing for the family."""
        return next((row for row in self.conditions if row.primary), None)


class FamiliesController:
    """Drives the Families tab beside the Curate view."""

    def __init__(self, attribution: Attribution | None = None) -> None:
        #: Who changes are recorded as, shared with every other view.
        self.attribution = attribution if attribution is not None else Attribution()
        #: Whether changes may be saved; the tab's one switch sets it.
        self.editing_enabled = False
        self._project: Project | None = None
        #: The lowercased conditions the lamella in view holds, or None for no lamella.
        self._here: frozenset[str] | None = None
        self._in_view: Lamella | None = None
        #: The lamella whose own entries a change is made to, or None for the project's.
        self._target: Lamella | None = None
        self._several = False
        self._problem = ""
        #: Changes not yet saved, as :func:`requested_corrections` takes them:
        #: condition to family, and family to primary. Only entries that
        #: change the file are held.
        self._families: dict[str, str | None] = {}
        self._primary: dict[str, str | None] = {}

    @property
    def recorder(self) -> Recorder:
        """Where this view's changes are recorded."""
        return self.attribution.recorder

    # -- what is shown ------------------------------------------------------

    def set_scope(self, projects: Sequence[Project], lamella: Lamella | None = None) -> None:
        """Show the families of the one project in *projects*, or none where there are several.

        Unsaved changes are kept while the project stays the same one, as it
        does when the folder is read again or another of its lamellae comes
        into view, and dropped when it changes. A change confined to a lamella
        follows the lamella in view while none is held, and stays with its own
        while one is.

        Args:
            lamella: The lamella in view, whose conditions are marked as here.
        """
        self._here = None if lamella is None else frozenset(
            name.lower() for name in _condition_names([lamella])
        )
        project = projects[0] if len(projects) == 1 else None
        if project is None or self._project is None or project.path != self._project.path:
            self.discard()
            self._target = None
        self._several = len(projects) > 1
        self._project = project
        self._in_view = lamella
        if self._target is not None and not self.has_changes:
            self._target = self.lamella_choice
        if self._target is not None and project is not None:
            # A rescan makes new lamellae; the one changes are held for is found again.
            self._target = next(
                (own for own in project.lamellae if own.path == self._target.path), None
            )
        self._problem = ""
        if project is not None:
            try:
                read_families(project.path)
            except FamiliesFileError as error:
                self._problem = (
                    f"Families are shown as given by the folder names. {error}; correct the file "
                    "by hand first"
                )

    @property
    def lamella_choice(self) -> Lamella | None:
        """Return the lamella a change could be confined to: the one in view, with a folder of its own."""
        if self._in_view is None or self._in_view.synthetic:
            return None
        return self._in_view

    @property
    def target(self) -> Lamella | None:
        """Return the lamella whose own entries a change is made to, or None for the project's."""
        return self._target

    def set_target(self, lamella_only: bool) -> None:
        """Make changes the project's, or the lamella in view's own.

        Raises:
            ValueError: If changes are held, which belong to the choice they
                were made under, or there is no lamella to confine them to.
        """
        if self.has_changes:
            raise ValueError("Save or discard the changes first")
        if lamella_only and self.lamella_choice is None:
            raise ValueError("This lamella has no folder of its own")
        self._target = self.lamella_choice if lamella_only else None

    @property
    def project(self) -> Project | None:
        """Return the project whose families are shown, or None."""
        return self._project

    @property
    def editable(self) -> bool:
        """Return whether this project's families can be changed here at all."""
        return self._project is not None and not self._problem

    def status(self) -> str:
        """Return the line above the tree: why nothing is shown, or what is.

        A damaged file is named rather than shown as no corrections: the scan
        read the project by its folder names, which is what the tree shows.
        """
        if self._several:
            return "Select one project to show its families."
        if self._project is None:
            return "No project selected."
        if self._problem:
            return self._problem
        if not self.tree():
            return "This project has no imaging conditions."
        return ""

    def tree(self) -> tuple[FamilyNode, ...]:
        """Return the project's families, with the changes not yet saved, SEM first.

        The families :func:`summarise_families` finds under the corrections a
        save would write, so the tree shows what saving does; a row differing
        from the same summary of the file is not saved.
        """
        if self._project is None:
            return ()
        key = self._section_key()
        saved = summarise_families(self._project, lamella=key)
        shown = summarise_families(self._project, self._after(), lamella=key)
        saved_families = {
            (name.lower(), family_key(summary.family))
            for summary in saved for name, _ in summary.conditions
        }
        saved_primary = {
            (summary.modality, family_key(summary.family)): (summary.primary or "").lower()
            for summary in saved
        }
        return tuple(
            FamilyNode(
                modality=summary.modality,
                family=summary.family,
                conditions=tuple(
                    self._row(summary, name, stages, saved_families, saved_primary)
                    for name, stages in summary.conditions
                ),
            )
            for summary in shown
        )

    def _row(
        self,
        summary: FamilySummary,
        name: str,
        stages: int,
        saved_families: set[tuple[str, str]],
        saved_primary: Mapping[tuple[str, str], str],
    ) -> ConditionRow:
        """Return one condition of *summary* as the tree lists it."""
        family = family_key(summary.family)
        primary = name == summary.standing
        named = primary and summary.primary is not None
        primary_changed = (summary.primary or "").lower() != saved_primary.get(
            (summary.modality, family), ""
        )
        return ConditionRow(
            name=name,
            stages=stages,
            moved=family != family_key(_name_family(name)),
            primary=primary,
            named_primary=named,
            unsaved=(name.lower(), family) not in saved_families or (primary and primary_changed),
            here=self._here is None or name.lower() in self._here,
        )

    def family_of(self, condition: str) -> str:
        """Return the family *condition* is in where a change applies, counting changes not yet saved."""
        parsed = condition_of(condition, self._in_force(self._after()))
        return parsed.family if parsed is not None else condition

    def family_options(self, condition: str) -> list[tuple[str, str]]:
        """Return ``(caption, family)`` for each family *condition* could join.

        Its beam's families, as the tree shows them, and the family its name
        gives, captioned as such, whether or not anything else is in it.
        """
        own = _name_family(condition)
        parsed = classify_image_directory(condition)
        modality = parsed.modality if parsed is not None else None
        families = [node.family for node in self.tree() if node.modality == modality]
        if not any(family_key(family) == family_key(own) for family in families):
            families.append(own)
        return [
            (f"{family} (its name)" if family_key(family) == family_key(own) else family, family)
            for family in families
        ]

    # -- changing -----------------------------------------------------------

    @property
    def has_changes(self) -> bool:
        """Return whether anything is changed here and not saved."""
        return bool(self._families or self._primary)

    def set_family(self, condition: str, family: str) -> None:
        """Place *condition* in *family*, held until saved.

        The family it would have anyway clears any change. A condition named
        as its family's primary gives that up as it leaves, since a primary
        must belong to the family it stands for.

        Raises:
            ValueError: If the change is one :func:`requested_corrections` refuses.
        """
        before = self.family_of(condition)
        if family_key(before) == family_key(family):
            return
        primary = dict(self._primary)
        named = self._in_force(self._after()).primary_of(before)
        if named is not None and named.lower() == condition.lower():
            primary[before] = None
        self._hold({**self._families, condition: family}, primary)

    def set_primary(self, family: str, condition: str | None) -> None:
        """Name *condition* as *family*'s primary, held until saved.

        None returns it to the default, or for a lamella to the project's.

        Raises:
            ValueError: If the change is one :func:`requested_corrections` refuses.
        """
        self._hold(self._families, {**self._primary, family: condition})

    def discard(self) -> None:
        """Forget every change not yet saved."""
        self._families.clear()
        self._primary.clear()

    def _hold(
        self, families: Mapping[str, str | None], primary: Mapping[str, str | None]
    ) -> None:
        """Hold the entries a request changes, refusing one the operation refuses."""
        try:
            after = self._request(families, primary)
        except FamiliesValidationError as error:
            raise ValueError("; ".join(error.problems)) from error
        assert self._project is not None, "a request is made of a project"
        key = self._section_key()
        self.discard()
        for entry in entry_changes(self._project.families, after):
            if entry.lamella == key:
                held = self._families if entry.section == FAMILIES_SECTION else self._primary
                held[entry.key] = entry.new

    def _request(
        self, families: Mapping[str, str | None], primary: Mapping[str, str | None]
    ) -> ProjectFamilies:
        """Return the corrections *families* and *primary* would make where a change applies.

        Raises:
            FamiliesValidationError: As :func:`requested_corrections` does.
        """
        if self._project is None:
            raise FamiliesValidationError(["Select one project to change its families"])
        target = self._target
        return requested_corrections(
            self._project.families,
            _condition_names(self._project.lamellae),
            _condition_names([target] if target is not None else self._project.lamellae),
            families=families,
            primary=primary,
            lamella=self._section_key(),
            where=target.name if target is not None else "this project",
        )

    def _after(self) -> ProjectFamilies:
        """Return the corrections saving would write.

        The file's where the changes held no longer apply, as when a rescan
        no longer finds a condition they name: the plan says why.
        """
        assert self._project is not None, "only asked of a project"
        if not self.has_changes:
            return self._project.families
        try:
            return self._request(self._families, self._primary)
        except FamiliesValidationError:
            return self._project.families

    def _in_force(self, corrections: ProjectFamilies) -> FamilyCorrections:
        """Return *corrections* where a change applies: the lamella chosen, or the project."""
        assert self._project is not None, "only asked of a project"
        if self._target is None:
            return corrections.project
        return self._project.families_at(self._target.path, corrections)

    def _section_key(self) -> str | None:
        """Return the key of the section a change is made to, None for the project's."""
        if self._project is None or self._target is None:
            return None
        return self._target.path.relative_to(self._project.path).as_posix()

    # -- saving -------------------------------------------------------------

    def plan(self) -> FamiliesChange:
        """Return what saving the changes would do, writing nothing.

        Planned with leaving a corrected group allowed: the window counts
        those with :func:`leaving_corrected` and asks before saving.

        Raises:
            FamiliesValidationError: Carrying every reason the change is refused.
            ValueError: If no project is in scope.
        """
        if self._project is None:
            raise ValueError("Select one project to change its families")
        return plan_families_change(
            self._project,
            families=self._families,
            primary=self._primary,
            lamella=self._target,
            allow_leaving_corrected=True,
        )

    def apply(self, change: FamiliesChange) -> FamiliesChange:
        """Write a planned change, and forget the changes it carried.

        Raises:
            PermissionError: If editing has not been enabled.
            StaleFamilies: If the families file changed since the plan read it.
            StaleState: If a stage's state file did.
        """
        if not self.editing_enabled:
            raise PermissionError("Editing is disabled")
        assert self._project is not None, "a change is planned for a project"
        batch = writing_recorder([self._project], self.recorder)
        applied = apply_families_change(change, recorder=batch)
        self.discard()
        return applied


def leaving_corrected(change: FamiliesChange) -> int:
    """Return how many images *change* takes out of a corrected group."""
    return sum(1 for effect in change.states.effects if effect.old == "corrected")


def _condition_names(lamellae: Iterable[Lamella]) -> list[str]:
    """Return the names of the image directories the scan found in *lamellae*."""
    return [
        image_set.directory.name
        for lamella in lamellae
        for stage in lamella.stages
        for image_set in stage.image_sets
        if not image_set.is_loose
    ]


def _name_family(condition: str) -> str:
    """Return the family *condition*'s name gives it."""
    parsed = classify_image_directory(condition)
    return parsed.family if parsed is not None else condition
