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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..families import FamiliesFileError, read_families
from ..history import Recorder
from ..layout import (
    NO_CORRECTIONS,
    FamilyCorrections,
    classify_image_directory,
    condition_of,
    correction_problems,
)
from ..models import Lamella, Project
from ..operations.families_edit import (
    FamiliesChange,
    apply_families_change,
    plan_families_change,
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
        #: Changes not yet saved: condition to family, None for what the
        #: project gives it -- its name's, or for a lamella the project's entry.
        self._families: dict[str, str | None] = {}
        #: Changes not yet saved: family to primary, None for the default, or
        #: for a lamella the project's.
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
        self._here = (
            None if lamella is None else frozenset(
                image_set.directory.name.lower()
                for stage in lamella.stages
                for image_set in stage.image_sets
                if not image_set.is_loose
            )
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
                    f"Shown as the folder names give them. {error}; correct the file "
                    f"by hand first"
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
            return "Choose one project to see its families."
        if self._project is None:
            return "No project selected."
        if self._problem:
            return self._problem
        if not self.tree():
            return "No imaging conditions in this project."
        return ""

    def tree(self) -> tuple[FamilyNode, ...]:
        """Return the project's families, with the changes not yet saved, SEM first."""
        if self._project is None:
            return ()
        members: dict[tuple[str, str], list[tuple[str, int]]] = {}
        spelled: dict[tuple[str, str], str] = {}
        for summary in summarise_families(self._project):
            spelled[(summary.modality, summary.family.lower())] = summary.family
            for name, stages in summary.conditions:
                # Where the scan read it, the file applied; a change not yet saved moves it.
                pending = _lookup(self._families, name) is not None
                family = self.family_of(name) if pending else summary.family
                key = (summary.modality, family.lower())
                spelled.setdefault(key, family)
                members.setdefault(key, []).append((name, stages))

        nodes = []
        for key in sorted(members, key=lambda item: (item[0] != "SEM", item[1])):
            family = spelled[key]
            conditions = sorted(
                members[key],
                key=lambda item: (item[0].lower() != family.lower(), item[0].lower()),
            )
            nodes.append(self._node(key[0], family, conditions))
        return tuple(nodes)

    def _node(
        self,
        modality: str,
        family: str,
        conditions: Sequence[tuple[str, int]],
    ) -> FamilyNode:
        """Return one family, marking the condition that stands for it."""
        names = [name for name, _ in conditions]
        pending = _lookup(self._primary, family)
        named = self._named_primary(family)
        held = next((name for name in names if named and name.lower() == named.lower()), None)
        primary = held if held is not None else min(
            names, key=lambda name: _base_first(self._variant_in(name, family))
        )
        return FamilyNode(
            modality=modality,
            family=family,
            conditions=tuple(
                ConditionRow(
                    name=name,
                    stages=stages,
                    moved=family.lower() != _name_family(name).lower(),
                    primary=name == primary,
                    named_primary=held is not None and name == held,
                    unsaved=_lookup(self._families, name) is not None
                    or (pending is not None and name == primary),
                    here=self._here is None or name.lower() in self._here,
                )
                for name, stages in conditions
            ),
        )

    def family_of(self, condition: str) -> str:
        """Return the family *condition* is in where a change applies, counting changes not yet saved."""
        pending = _lookup(self._families, condition)
        if pending is not None:
            return pending[1] if pending[1] is not None else self._given(condition)
        moved = self._in_force().family_of(condition)
        return moved if moved is not None else _name_family(condition)

    def _given(self, condition: str) -> str:
        """Return the family *condition* has where no entry of the chosen section moves it."""
        if self._target is None:
            return _name_family(condition)
        assert self._project is not None, "a lamella is chosen within a project"
        return self._project.families.project.family_of(condition) or _name_family(condition)

    def _in_force(self) -> FamilyCorrections:
        """Return the corrections on file where a change applies."""
        assert self._project is not None, "only asked of a project's conditions"
        if self._target is None:
            return self._project.families.project
        return self._project.families_at(self._target.path)

    def _section(self) -> tuple[Mapping[str, str], Mapping[str, str]]:
        """Return the chosen section's families and primary entries as on file."""
        assert self._project is not None, "only asked of a project's conditions"
        families = self._project.families
        if self._target is None:
            return families.project.families, families.project.primary
        key = self._target.path.relative_to(self._project.path).as_posix()
        own = families.lamellae.get(key)
        return (own.families, own.primary) if own is not None else ({}, {})

    def family_options(self, condition: str) -> list[tuple[str, str]]:
        """Return ``(caption, family)`` for each family *condition* could join.

        Its beam's families, as the tree shows them, and the family its name
        gives, captioned as such, whether or not anything else is in it.
        """
        own = _name_family(condition)
        modality = _modality(condition)
        families = [node.family for node in self.tree() if node.modality == modality]
        if not any(family.lower() == own.lower() for family in families):
            families.append(own)
        return [
            (f"{family} (its name)" if family.lower() == own.lower() else family, family)
            for family in families
        ]

    # -- changing -----------------------------------------------------------

    @property
    def has_changes(self) -> bool:
        """Return whether anything is changed here and not saved."""
        return bool(self._families or self._primary)

    def family_problem(self, condition: str, family: str) -> str:
        """Return why *condition* cannot join *family*, or an empty string if it can."""
        return "; ".join(correction_problems({condition: family}, {}))

    def set_family(self, condition: str, family: str) -> None:
        """Place *condition* in *family*, held until saved.

        Its name's own family clears any correction. A condition named as its
        family's primary gives that up as it leaves, since a primary must
        belong to the family it stands for.

        Raises:
            ValueError: If *family* is no family *condition* may join.
        """
        problem = self.family_problem(condition, family)
        if problem:
            raise ValueError(problem)
        before = self.family_of(condition)
        if before.lower() == family.lower():
            return
        target = None if family.lower() == self._given(condition).lower() else family
        on_file = _lookup(dict(self._section()[0]), condition)
        _set(
            self._families, condition, target,
            unchanged=_same(on_file[1] if on_file else None, target),
        )

        named = self._named_primary(before)
        if named is not None and named.lower() == condition.lower():
            self.set_primary(before, None)

    def set_primary(self, family: str, condition: str | None) -> None:
        """Name *condition* as *family*'s primary, held until saved.

        None returns it to the default, or for a lamella to the project's.
        """
        on_file = _lookup(dict(self._section()[1]), family)
        _set(
            self._primary, family, condition,
            unchanged=_same(on_file[1] if on_file else None, condition),
        )

    def discard(self) -> None:
        """Forget every change not yet saved."""
        self._families.clear()
        self._primary.clear()

    def _named_primary(self, family: str) -> str | None:
        """Return the condition named as *family*'s primary, counting unsaved changes."""
        pending = _lookup(self._primary, family)
        if pending is not None and pending[1] is not None:
            return pending[1]
        if pending is not None and self._target is None:
            return None
        if pending is not None:
            assert self._project is not None, "a lamella is chosen within a project"
            return self._project.families.project.primary_of(family)
        return self._in_force().primary_of(family)

    def _variant_in(self, condition: str, family: str) -> str | None:
        """Return *condition*'s variant in *family*, as resolution would give it."""
        if family.lower() == _name_family(condition).lower():
            corrections = NO_CORRECTIONS
        else:
            corrections = FamilyCorrections(families={condition: family})
        parsed = condition_of(condition, corrections)
        return parsed.variant if parsed is not None else condition

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
            raise ValueError("Choose one project to change its families")
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


def _name_family(condition: str) -> str:
    """Return the family *condition*'s name gives it."""
    parsed = classify_image_directory(condition)
    return parsed.family if parsed is not None else condition


def _modality(condition: str) -> str | None:
    """Return the beam *condition*'s name gives it."""
    parsed = classify_image_directory(condition)
    return parsed.modality if parsed is not None else None


def _base_first(variant: str | None) -> tuple[bool, str]:
    """Order a family's conditions as the default primary rule does."""
    return variant is not None, (variant or "").lower()


def _same(first: str | None, second: str | None) -> bool:
    """Return whether two names, either absent, are the same whatever the case."""
    return (first or "").lower() == (second or "").lower()


def _lookup(entries: dict[str, str | None], key: str) -> tuple[str, str | None] | None:
    """Return ``(key as held, value)`` for *key* in any case, or None if absent."""
    return next(
        ((held, value) for held, value in entries.items() if held.lower() == key.lower()),
        None,
    )


def _set(entries: dict[str, str | None], key: str, value: str | None, *, unchanged: bool) -> None:
    """Hold *value* for *key*, or forget the key where it matches what is saved."""
    held = _lookup(entries, key)
    if held is not None:
        del entries[held[0]]
    if not unchanged:
        entries[key] = value
