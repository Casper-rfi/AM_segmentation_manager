"""The metadata view model: which project, what it says, what an edit would do.

Wraps :mod:`am_segmentation_manager.operations.metadata_edit` so a widget can
show the resolved values, collect changes, preview them and apply them, without
knowing how the file is written or which of the two YAML files a value came
from.

Qt-free, like the other controllers.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

from ..metadata import DESCRIPTIVE_FIELDS, LAMELLA_TYPES, TODO_MARKER
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
    "LAMELLA_TYPES",
    "TODO_MARKER",
    "MetadataValidationError",
    "MetadataController",
]


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
        """Return each field's resolved value, blank where unset."""
        project = self.project
        resolved = project.experiment_metadata if project else None
        return {
            field: (getattr(resolved, field, None) or "") if resolved else ""
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

    def outstanding(self) -> tuple[str, ...]:
        """Return fields still marked TODO."""
        project = self.project
        resolved = project.experiment_metadata if project else None
        return resolved.todo_fields if resolved else ()

    def summary(self) -> str:
        """Return a one-line description of how complete this project is."""
        project = self.project
        if project is None:
            return "No project selected"
        values = self.values()
        filled = sum(1 for value in values.values() if value)
        return f"{project.display_name}: {filled} of {len(DESCRIPTIVE_FIELDS)} fields set"

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
