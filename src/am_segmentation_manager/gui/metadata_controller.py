"""The metadata view model: which project, what it says, what an edit would do.

Wraps :mod:`am_segmentation_manager.operations.metadata_edit` so a widget can
show the resolved values, collect changes, preview them and apply them, without
knowing how the file is written or which of the two YAML files a value came
from. It also works out what to suggest and how described each project is, both
read off the projects it was given rather than from the filesystem.

Qt-free, like the other controllers.
"""
from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..metadata import (
    DESCRIPTIVE_FIELDS,
    FIELD_COMMENTS,
    FIELDS,
    FIELDS_BY_NAME,
    LAMELLA_TYPES,
    TODO_MARKER,
    VALUE_SEPARATOR,
    field_values,
    format_field_value,
)
from ..models import Project
from ..operations.metadata_edit import (
    MetadataChange,
    MetadataValidationError,
    plan_metadata_update,
    update_project_metadata,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DESCRIPTIVE_FIELDS",
    "FIELDS",
    "FIELD_COMMENTS",
    "LAMELLA_TYPES",
    "SUGGESTED_FIELDS",
    "SUGGESTION_LIMIT",
    "TODO_MARKER",
    "VALUE_SEPARATOR",
    "MetadataValidationError",
    "MetadataController",
    "ProjectSummary",
]

#: Fields offered as a list of the values already in use, from the field table.
#: The ones left out are left out deliberately: ``description`` is prose about
#: one experiment, and ``lamella_type`` has a fixed vocabulary of its own.
#: ``project`` names the wider project several datasets belong to, so it repeats
#: across them by definition -- which is exactly what these lists are for.
SUGGESTED_FIELDS: tuple[str, ...] = tuple(
    entry.name for entry in FIELDS if entry.suggested
)

#: How many values a field offers. A collection of several hundred projects
#: would otherwise put a list nobody can read behind every box.
SUGGESTION_LIMIT = 20


@dataclass(frozen=True)
class ProjectSummary:
    """One project's row in the list: what to call it, and how described it is.

    Attributes:
        label: The project's display name, which carries its relative path.
        filled: How many descriptive fields hold a value.
        total: How many there are.
        missing: The fields with no value, in declaration order. A field still
            holding a ``TODO`` placeholder counts as missing, because it is.
        folder: The directory the project sits in, relative to the scan root,
            or empty at the root. What a tree groups the rows under.
        own_name: The label to show beneath that folder, the folder having
            already said the rest. Both are taken from the project rather than
            split out of :attr:`label` here, the split being the model's.
    """

    label: str
    filled: int
    total: int
    missing: tuple[str, ...] = ()
    folder: str = ""
    own_name: str = ""

    @property
    def is_complete(self) -> bool:
        """Return whether every descriptive field has a value."""
        return not self.missing

    def describe(self) -> str:
        """Return a one-line account of what is filled in and what is not."""
        counted = f"{self.filled} of {self.total} fields set"
        if self.is_complete:
            return counted
        return f"{counted} -- missing: {', '.join(self.missing)}"


class MetadataController:
    """Show and edit one project's descriptive metadata."""

    def __init__(self) -> None:
        self._projects: list[Project] = []
        self._position = 0
        self.editing_enabled = False

    # -- selection ---------------------------------------------------------

    def set_projects(self, projects: Sequence[Project]) -> None:
        """Replace the project list, selecting the first."""
        self._projects = list(projects)
        self._position = 0

    @property
    def projects(self) -> list[Project]:
        """Return the projects available to edit."""
        return self._projects

    @property
    def position(self) -> int:
        """Return the index of the selected project."""
        return self._position

    def select(self, position: int) -> None:
        """Select a project by index, clamped to the available range."""
        if self._projects:
            self._position = max(0, min(position, len(self._projects) - 1))

    @property
    def project(self) -> Project | None:
        """Return the selected project, or None when the list is empty."""
        if not self._projects:
            return None
        return self._projects[self._position]

    # -- reading -----------------------------------------------------------

    def values(self) -> dict[str, str]:
        """Return each field's resolved value as text, blank where unset.

        Through :func:`~am_segmentation_manager.metadata.format_field_value`, so
        what an editor shows is what saving would compare against: a number read
        back as ``30.0`` and shown as ``30`` would otherwise register as an edit
        on every project the user merely looked at.
        """
        project = self.project
        resolved = project.experiment_metadata if project else None
        return {
            field: format_field_value(getattr(resolved, field, None)) if resolved else ""
            for field in DESCRIPTIVE_FIELDS
        }

    def sources(self) -> dict[str, str]:
        """Return which file supplied each value, for display beside it.

        Surfacing this is most of the point of a metadata view: a value that
        looks wrong is explained by which of the two files it came from.
        """
        project = self.project
        resolved = project.experiment_metadata if project else None
        if resolved is None:
            return {}
        return {
            field: path.name for field, path in resolved.sources.items()
        }

    def suggestions(self, field: str) -> list[str]:
        """Return values already recorded for *field*, most projects first.

        Typing a value that repeats across projects is how a collection ends up
        with ``HeLa`` and ``Hela`` as separate groups in every breakdown that
        keys on it, so the values in use are worth offering.

        Ties break alphabetically, and the list is capped at
        :data:`SUGGESTION_LIMIT` -- except that the selected project's own value
        always appears, since a list that omits what the box is showing invites
        replacing it by accident. Returns an empty list for a field with no
        suggestions, and for one not in :data:`SUGGESTED_FIELDS`.

        A field's declared defaults come last and always appear, so the usual
        answers are on offer in a collection where nobody has given one yet.
        That is the whole difference between a default and a vocabulary: it is
        offered, not required.
        """
        if field not in SUGGESTED_FIELDS:
            return []
        # Each value of a field that holds several, not the set of them: a
        # project run by two organisations should offer each on its own, or
        # the list fills with pairs nobody would pick as a unit.
        counts = Counter(
            value
            for project in self._projects
            for value in field_values(getattr(project.experiment_metadata, field, None))
        )
        offered = sorted(counts, key=lambda value: (-counts[value], value.lower()))
        offered = offered[:SUGGESTION_LIMIT]
        project = self.project
        current = (
            field_values(getattr(project.experiment_metadata, field, None))
            if project is not None else ()
        )
        offered.extend(value for value in current if value not in offered)
        spec = FIELDS_BY_NAME.get(field)
        if spec is not None:
            offered.extend(
                value for value in spec.defaults if value not in offered
            )
        return offered

    def outstanding(self) -> tuple[str, ...]:
        """Return fields still marked TODO."""
        project = self.project
        resolved = project.experiment_metadata if project else None
        return resolved.todo_fields if resolved else ()

    def project_summaries(self) -> list[ProjectSummary]:
        """Return one row per project, in list order.

        This is what lets the list say which projects still need describing
        without opening each in turn, so it covers every project rather than
        only the selected one.
        """
        return [self._summarise(project) for project in self._projects]

    def summary(self) -> str:
        """Return a one-line description of how complete this project is."""
        project = self.project
        if project is None:
            return "No project selected"
        return f"{project.display_name}: {self._summarise(project).describe()}"

    @staticmethod
    def _summarise(project: Project) -> ProjectSummary:
        """Return how much of *project* is described.

        A project with no metadata file at all is missing every field, which is
        the same situation as one whose fields are all placeholders.
        """
        resolved = project.experiment_metadata
        missing = (
            resolved.missing_fields() if resolved is not None else DESCRIPTIVE_FIELDS
        )
        return ProjectSummary(
            label=project.display_name,
            filled=len(DESCRIPTIVE_FIELDS) - len(missing),
            total=len(DESCRIPTIVE_FIELDS),
            missing=missing,
            folder=project.folder,
            own_name=project.own_name,
        )

    # -- paths, for copying out of the window ------------------------------

    def project_path(self, position: int | None = None) -> Path | None:
        """Return the absolute path of a project, or of the selected one.

        Absolute, because a path pasted into a terminal or a file manager has to
        be for it to be any use, and that is the whole point of copying one.
        """
        index = self._position if position is None else position
        if not (0 <= index < len(self._projects)):
            return None
        return self._projects[index].path.resolve()

    # -- editing -----------------------------------------------------------

    def plan(self, edits: dict[str, str | None]) -> MetadataChange:
        """Return what *edits* would do, writing nothing.

        Raises:
            MetadataValidationError: If any edit is invalid.
        """
        project = self.project
        if project is None:
            raise ValueError("No project selected")
        return plan_metadata_update(
            project.path, edits, project_name=project.display_name
        )

    def apply(self, edits: dict[str, str | None]) -> MetadataChange:
        """Write *edits*, refreshing the project's metadata afterwards.

        Raises:
            MetadataValidationError: If any edit is invalid.
            PermissionError: If editing has not been enabled.
        """
        if not self.editing_enabled:
            raise PermissionError("Editing is disabled")
        project = self.project
        if project is None:
            raise ValueError("No project selected")

        change = update_project_metadata(
            project.path, edits, project_name=project.display_name
        )
        self._refresh(project)
        return change

    def _refresh(self, project: Project) -> None:
        """Re-resolve the project's metadata from disk after a write.

        Without this the view would keep showing what was there before the
        edit, which is the sort of thing that makes a user distrust the tool.
        """
        from ..io.yaml import read_yaml
        from ..metadata import (
            AM_METADATA_FILENAME,
            EXPERIMENT_FILENAME,
            build_experiment_metadata,
        )

        experiment_path = project.path / EXPERIMENT_FILENAME
        am_path = project.path / AM_METADATA_FILENAME
        project.experiment_metadata = build_experiment_metadata(
            read_yaml(experiment_path), read_yaml(am_path),
            experiment_path=experiment_path, am_metadata_path=am_path,
        )
