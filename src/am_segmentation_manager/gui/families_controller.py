"""The families view model: one project's families of imaging conditions, as a tree.

Wraps :mod:`am_segmentation_manager.operations.families_edit` so a widget can show
which conditions form which family, and which stands for each, without knowing how
``AM_families.yaml`` is read. Families are a project's, so a scope of several
projects shows none. Qt-free, like the other controllers.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..families import FamiliesFileError, read_families
from ..layout import condition_of
from ..models import Project
from ..operations.families_edit import FamilySummary, summarise_families


@dataclass(frozen=True)
class ConditionRow:
    """One condition, as the tree lists it under its family.

    Attributes:
        name: The image directory's name.
        stages: How many of the project's stages hold it.
        moved: Whether the project's ``AM_families.yaml`` places it in this family.
        primary: Whether it stands for the family: named so in the file, or
            chosen by the default rule where the file names none.
        named_primary: Whether the file names it as the primary.
    """

    name: str
    stages: int
    moved: bool = False
    primary: bool = False
    named_primary: bool = False

    @property
    def note(self) -> str:
        """Return what the tree says about it beyond its name and stages."""
        parts = []
        if self.primary:
            parts.append("primary" if self.named_primary else "primary (default)")
        if self.moved:
            parts.append("moved by AM_families.yaml")
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


class FamiliesController:
    """Drives the Families box of the Add data tab."""

    def __init__(self) -> None:
        self._project: Project | None = None
        self._several = False
        self._problem = ""

    def set_scope(self, projects: Sequence[Project]) -> None:
        """Show the families of the one project in *projects*, or none where there are several."""
        self._several = len(projects) > 1
        self._project = projects[0] if len(projects) == 1 else None
        self._problem = ""
        if self._project is not None:
            try:
                read_families(self._project.path)
            except FamiliesFileError as error:
                self._problem = f"{error}; correct the file by hand first"

    @property
    def project(self) -> Project | None:
        """Return the project whose families are shown, or None."""
        return self._project

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
            return f"Shown as the folder names give them. {self._problem}"
        if not self.tree():
            return "No imaging conditions in this project."
        return ""

    def tree(self) -> tuple[FamilyNode, ...]:
        """Return the project's families as its scan resolved them, SEM first."""
        if self._project is None:
            return ()
        return tuple(
            self._node(summary) for summary in summarise_families(self._project)
        )

    def _node(self, summary: FamilySummary) -> FamilyNode:
        """Return one family, marking the condition that stands for it."""
        names = [name for name, _ in summary.conditions]
        named = summary.primary
        held = next((name for name in names if named and name.lower() == named.lower()), None)
        primary = held if held is not None else self._default_primary(names)
        return FamilyNode(
            modality=summary.modality,
            family=summary.family,
            conditions=tuple(
                ConditionRow(
                    name=name,
                    stages=stages,
                    moved=name in summary.moved,
                    primary=name == primary,
                    named_primary=held is not None and name == held,
                )
                for name, stages in summary.conditions
            ),
        )

    def _default_primary(self, names: Sequence[str]) -> str | None:
        """Return the condition the default rule chooses: the base, else the first variant.

        By the variant each has in its family, as
        :func:`~...frames.primary_among` orders them.
        """
        assert self._project is not None, "only asked of a project's families"
        corrections = self._project.families

        def order(name: str) -> tuple[bool, str]:
            parsed = condition_of(name, corrections)
            variant = parsed.variant if parsed is not None else name
            return variant is not None, (variant or "").lower()

        return min(names, key=order) if names else None
