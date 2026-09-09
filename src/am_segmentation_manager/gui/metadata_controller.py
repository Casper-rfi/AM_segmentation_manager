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

from ..history import Recorder
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
from ..operations.init_history import writing_recorder
from ..operations.metadata_edit import (
    MetadataChange,
    MetadataValidationError,
    plan_metadata_update,
    update_project_metadata,
)
from .attribution import Attribution

logger = logging.getLogger(__name__)

__all__ = [
    "DESCRIPTIVE_FIELDS",
    "FIELDS",
    "FIELD_COMMENTS",
    "LAMELLA_TYPES",
    "SUGGESTED_FIELDS",
    "SUGGESTION_LIMIT",
    "MIXED_MARKER",
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


#: What a field shows when the projects being described do not agree on it.
#:
#: Held in the editor rather than beside it, because the panel decides what to
#: write by comparing the box against the resolved value: a field left showing
#: this matches and so is never written, which is the whole rule. The badge
#: beside the field is what *says* it -- this is the editable stand-in, and a
#: value that happened to be typed as ``(mixed)`` would read as untouched, the
#: same trade already made for :data:`TODO_MARKER`.
#:
#: Deliberately not the curation controller's ``MIXED``: that is the state of a
#: frame whose two beams disagree, never shown in a box, and the two sharing a
#: word does not make them the same fact.
MIXED_MARKER = "(mixed)"


class MetadataController:
    """Show and edit the descriptive metadata of a project, or of a folder of them.

    **The scope is a list, and one project is a list of one.** Describing a
    collection is done a session at a time -- a folder holds a dozen Autolamella
    projects sharing a sample, a grid and an operator -- so everything here
    reads the projects in scope rather than a selected one, and the two cases
    are the same code path.
    """

    def __init__(self, attribution: Attribution | None = None) -> None:
        self._projects: list[Project] = []
        self._position = 0
        #: The projects being described. One, or every project under a folder.
        self._scope: list[Project] = []
        #: The folder whose projects are in scope, or None for a single one.
        self._scope_folder: str | None = None
        self.editing_enabled = False
        #: Who changes are recorded as, shared with every other view.
        self.attribution = attribution if attribution is not None else Attribution()

    @property
    def recorder(self) -> Recorder:
        """Where this view's changes are recorded.

        Shared with every other view, so a name adopted in Settings reaches
        this one without anybody remembering to pass it on.
        """
        return self.attribution.recorder



    # -- selection ---------------------------------------------------------

    def set_projects(self, projects: Sequence[Project]) -> None:
        """Replace the project list, selecting the first."""
        self._projects = list(projects)
        self._position = 0
        self.select(0)

    @property
    def projects(self) -> list[Project]:
        """Return the projects available to edit."""
        return self._projects

    @property
    def position(self) -> int:
        """Return the index of the selected project."""
        return self._position

    def select(self, position: int) -> None:
        """Describe one project, by index, clamped to the available range."""
        if not self._projects:
            self._scope, self._scope_folder = [], None
            return
        self._position = max(0, min(position, len(self._projects) - 1))
        self._scope = [self._projects[self._position]]
        self._scope_folder = None

    def select_folder(self, folder: str) -> None:
        """Describe every project under *folder* at once.

        A session's projects share their sample, their grid and their operator,
        and typing that a dozen times is how a collection ends up with ``HeLa``
        beside ``Hela``. Fields the projects disagree on show
        :data:`MIXED_MARKER` and are left alone unless touched.
        """
        self._scope_folder = folder
        self._scope = [
            project for project in self._projects if project.folder == folder
        ]

    @property
    def scope(self) -> list[Project]:
        """Return the projects being described, in list order."""
        return self._scope

    @property
    def scope_folder(self) -> str | None:
        """Return the folder being described, or None for a single project."""
        return self._scope_folder

    @property
    def project(self) -> Project | None:
        """Return the one project in scope, or None where several or none are.

        For the callers that can only mean one -- the path to copy, the
        provenance of a single value. Anything that must work over a folder
        reads :attr:`scope`.
        """
        return self._scope[0] if len(self._scope) == 1 else None

    # -- reading -----------------------------------------------------------

    def values(self) -> dict[str, str]:
        """Return each field's resolved value as text, blank where unset.

        Through :func:`~am_segmentation_manager.metadata.format_field_value`, so
        what an editor shows is what saving would compare against: a number read
        back as ``30.0`` and shown as ``30`` would otherwise register as an edit
        on every project the user merely looked at.

        Over several projects a field they agree on reads as that value, and one
        they do not reads as :data:`MIXED_MARKER` -- which the panel puts in the
        box, so an untouched field compares equal and is never written.
        """
        return {field: self._shared(field) for field in DESCRIPTIVE_FIELDS}

    def _shared(self, field: str) -> str:
        """Return what every project in scope says for *field*, or the marker."""
        seen = {
            format_field_value(getattr(project.experiment_metadata, field, None))
            if project.experiment_metadata is not None
            else ""
            for project in self._scope
        }
        if not seen:
            return ""
        return seen.pop() if len(seen) == 1 else MIXED_MARKER

    def mixed_fields(self) -> tuple[str, ...]:
        """Return the fields the projects in scope disagree on.

        Asked for separately from :meth:`values` so the panel can badge them:
        the marker in the box is editable text and a value typed as that text
        would be indistinguishable, where this is derived and cannot be typed.
        """
        return tuple(
            field
            for field in DESCRIPTIVE_FIELDS
            if self._shared(field) == MIXED_MARKER
        )

    def sources(self) -> dict[str, str]:
        """Return which file supplied each value, for display beside it.

        Surfacing this is most of the point of a metadata view: a value that
        looks wrong is explained by which of the two files it came from.

        Only where every project in scope agrees. A value shared across a folder
        can still have come from ``experiment.yaml`` in one and
        ``AM_metadata.yaml`` in another, and naming one of them would be a
        statement about provenance that is not true.
        """
        named: dict[str, set[str]] = {}
        for project in self._scope:
            resolved = project.experiment_metadata
            for field, path in (resolved.sources if resolved else {}).items():
                named.setdefault(field, set()).add(path.name)
        return {
            field: next(iter(names))
            for field, names in named.items()
            if len(names) == 1 and self._named_by_all(field)
        }

    def _named_by_all(self, field: str) -> bool:
        """Return whether every project in scope names a source for *field*.

        A field one project leaves unset has no shared provenance even when the
        others agree, which for a scope of one is simply "this project has a
        source for it".
        """
        return all(
            project.experiment_metadata is not None
            and field in project.experiment_metadata.sources
            for project in self._scope
        )

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
        """Return fields still marked TODO in any project in scope.

        Any, not all: a placeholder left in one project of a dozen is work
        outstanding, and a badge that appeared only once every project agreed
        would go quiet exactly as the folder was nearly finished.
        """
        return tuple(
            field
            for field in DESCRIPTIVE_FIELDS
            if any(
                project.experiment_metadata is not None
                and field in project.experiment_metadata.todo_fields
                for project in self._scope
            )
        )

    def project_summaries(self) -> list[ProjectSummary]:
        """Return one row per project, in list order.

        This is what lets the list say which projects still need describing
        without opening each in turn, so it covers every project rather than
        only the selected one.
        """
        return [self._summarise(project) for project in self._projects]

    def summary(self) -> str:
        """Return a one-line description of what is being described, and how far.

        For a folder the projects are counted rather than read out: a dozen
        names do not fit and would not help, where how many are still short of
        a field is the reason to be looking at the folder at all.
        """
        if not self._scope:
            return "No project selected"
        if self._scope_folder is None:
            project = self._scope[0]
            return f"{project.display_name}: {self._summarise(project).describe()}"
        summaries = [self._summarise(project) for project in self._scope]
        complete = sum(1 for summary in summaries if summary.is_complete)
        return (
            f"{self._scope_folder}: {len(summaries)} projects, "
            f"{complete} fully described"
        )

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

    def plan(self, edits: dict[str, str | None]) -> list[MetadataChange]:
        """Return what *edits* would do to every project in scope, writing nothing.

        Raises:
            MetadataValidationError: If any edit is invalid for any project.
            ValueError: If nothing is in scope.
        """
        if not self._scope:
            raise ValueError("No project selected")
        return [
            plan_metadata_update(
                project.path, edits, project_name=project.display_name
            )
            for project in self._scope
        ]

    def apply(self, edits: dict[str, str | None]) -> list[MetadataChange]:
        """Write *edits* to every project in scope, refreshing them afterwards.

        **The whole set is planned before any of it is written.** A folder of a
        dozen projects is a batch, and one that stopped half way through on the
        seventh invalid value would leave six described and six not, with
        nothing on screen saying which -- the failure the batch rule exists to
        prevent.

        Raises:
            MetadataValidationError: If any edit is invalid for any project in
                scope. Nothing is written.
            PermissionError: If editing has not been enabled.
            ValueError: If nothing is in scope.
        """
        if not self.editing_enabled:
            raise PermissionError("Editing is disabled")
        self.plan(edits)

        batch = writing_recorder(self._scope, self.recorder)
        changes = [
            update_project_metadata(
                project.path, edits, project_name=project.display_name, recorder=batch
            )
            for project in self._scope
        ]
        for project in self._scope:
            self._refresh(project)
        return changes

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
