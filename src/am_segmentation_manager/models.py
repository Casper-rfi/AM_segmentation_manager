"""The dataclasses the layers pass between them.

One record per image, gathered into stages, lamellae and projects. Nothing here
reads the filesystem: :mod:`~am_segmentation_manager.scanner` fills these in, and
everything above works on them rather than walking the tree again.

**A record carries what its path cannot answer** -- how far a stage sits above an
image depends on the shape of its image set, and a loose image's directory names
no modality -- so ``stage_path`` and the two artefact directories are carried
rather than re-derived. Workflow state is one field, never a pair of flags.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePath
from typing import TYPE_CHECKING

from .state import DEFAULT_STATE, RecordState

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from .layout import ImageSet
    from .metadata import ExperimentMetadata


@dataclass
class ImageRecord:
    """Representation of a single image and associated artifacts.

    Attributes:
        image_path: Path to the image file (SEM or FIB).
        predictions: list of paths to prediction files (raw/clean).
        labels: list of paths to label files.
        project: optional project id/name
        lamella: optional lamella id/name
        stage: optional stage id/name
        timestamp: optional datetime parsed from metadata or filename
        modality: 'SEM' or 'FIB' when known
        variant: imaging-condition suffix of the image directory -- what
            follows the family and ``_`` -- or None for the family's base
            condition.
        family: the family of imaging conditions whose label the image shares,
            as its directory spells it, or None for an image sitting loose.
        series: what the image was taken for -- 'adaptive', 'reference' or
            'alignment'. Orthogonal to modality and variant.
        task: the milling task the image belongs to, spelt as the project's
            ``experiment.yaml`` spells it, or None where neither its directory
            nor its filename names one. Stages roll up into their task, so
            ``Rough 1`` and ``Rough 2`` are both ``Rough Milling``. Coarser
            than the stage directory and orthogonal to it: one task spans
            several stages, and one lamella directory holds every task's
            reference images at once.
        acquired_at: when the instrument recorded taking the image, read from
            its header. Set for a loose image, which has nothing else that
            orders it: many carry no timestamp in the filename, and the epochs
            that do appear are when the milling stage started rather than when
            the image was taken. ``None`` for an Adaptive
            Polishing image, whose filename carries the acquisition number
            that orders it, so its header is never opened.
        stage_path: the stage directory the image belongs to. Carried rather
            than derived from *image_path*, because how far the stage sits
            above an image depends on the shape of its set: an Adaptive
            Polishing image sits one level down in ``sem/``, a reference image
            sits directly in the stage.
        labels_directory: where this record's label lives or would live.
            Carried for the same reason -- it cannot be recovered from a loose
            image's path, whose directory names no modality.
        predictions_directory: where this record's predictions live or would
            live, the resolved directory rather than the ``am_seg`` subdirectory
            inside it that this package writes to. Carried for the same reason
            again, and so that reading a prediction and writing one cannot
            disagree about where it belongs.
        label_shared: whether this record's label directory is shared with
            another image set, so its label also belongs to other images
        state: the workflow state of the label group this image belongs to.
            One field rather than a flag per state, so the three cannot
            contradict each other. Read it through
            :func:`~am_segmentation_manager.operations.selection.current_state`
            and change it through
            :func:`~am_segmentation_manager.operations.selection.transition_records`.
    """

    image_path: Path
    predictions: list[Path] = field(default_factory=list)
    labels: list[Path] = field(default_factory=list)
    project: str | None = None
    lamella: str | None = None
    stage: str | None = None
    timestamp: datetime | None = None
    modality: str | None = None
    variant: str | None = None
    family: str | None = None
    series: str = "adaptive"
    task: str | None = None
    acquired_at: datetime | None = None
    stage_path: Path | None = None
    labels_directory: Path | None = None
    predictions_directory: Path | None = None
    label_shared: bool = False
    state: RecordState = DEFAULT_STATE


@dataclass
class Stage:
    """A directory of a lamella's images, and the state file governing them.

    Not only an Adaptive Polishing stage. A lamella directory holding
    reference images is one, as is a milling task and a single alignment run,
    because what makes a stage is holding images and an ``AM_state.yaml`` --
    not what the images were taken for. Which is which is
    :attr:`ImageRecord.series`, per image set rather than per directory, since
    one directory holds several.

    Attributes:
        name: human friendly stage name (e.g. 'adaptive_polish_...')
        path: filesystem path to the stage folder
        images: list of ImageRecord objects discovered in this stage
        number: zero-padded sequential stage number within the lamella
        plots: summary plot images written during lamella preparation, one per
            acquisition. Fewer plots than images is normal.
        summary_plot: the one plot describing the whole stage, where it has
            one. An alignment run summarises the several images it took to
            converge, so there is nothing to match to a single acquisition.
        synthetic: True when the dataset has no stage directory and the lamella
            directory stands in for one. Older layouts collapse this level, and
            a lamella standing in for a stage of its own reference images is
            the same case.
        image_sets: the image sets in this stage, with the label directory each
            resolves to and whether that label directory is shared with
            another set.
    """

    name: str
    path: Path
    images: list[ImageRecord] = field(default_factory=list)
    number: str | None = None
    plots: list[Path] = field(default_factory=list)
    summary_plot: Path | None = None
    synthetic: bool = False
    image_sets: list[ImageSet] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        """Return a user-facing stage label with its number when available.

        A synthesised stage is marked, so nothing implies a directory that does
        not exist on disk.
        """
        suffix = " (no stage folder)" if self.synthetic else ""
        if self.number:
            return f"{self.number}: {self.name}{suffix}"
        return f"{self.name}{suffix}"


@dataclass
class Lamella:
    """Representation of a lamella inside an Autolamella project.

    Attributes:
        name: lamella folder name
        path: path to lamella folder
        stages: list of Stage
        number: zero-padded lamella number parsed from the folder prefix
        synthetic: True when the dataset has no lamella directories and the
            project directory stands in for one.
    """

    name: str
    path: Path
    stages: list[Stage] = field(default_factory=list)
    number: str | None = None
    synthetic: bool = False

    @property
    def display_name(self) -> str:
        """Return the folder name, which already includes its lamella number."""
        return f"{self.name} (no lamella folder)" if self.synthetic else self.name

    @property
    def image_records(self) -> list[ImageRecord]:
        """Return every image of this lamella's stages, in scan order."""
        return [record for stage in self.stages for record in stage.images]


@dataclass
class Project:
    """Representation of an Autolamella project.

    Attributes:
        name: user-facing project name
        path: path to project root
        relative_path: path to the project root relative to the scan root.
            This is the project's identity: project names are not unique, and
            neither is anything inside experiment.yaml -- a project copied to a
            second location keeps its name, created_at and recorded path.
        metadata: raw merged mapping from experiment.yaml and AM_metadata.yaml,
            for Autolamella keys that are not modelled
        experiment_metadata: the typed view of the descriptive metadata
        lamellae: list of Lamella objects
        metadata_problems: one message for each of experiment.yaml and
            AM_metadata.yaml that is there but would not read; both metadata
            views above then lack that file's values
    """

    name: str
    path: Path
    relative_path: str | None = None
    metadata: dict = field(default_factory=dict)
    experiment_metadata: ExperimentMetadata | None = None
    lamellae: list[Lamella] = field(default_factory=list)
    number: str | None = None
    metadata_problems: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        """Return the project's identity within an index."""
        return self.relative_path or self.name

    @property
    def image_records(self) -> list[ImageRecord]:
        """Return every image of this project's lamellae, in scan order."""
        return [
            record for lamella in self.lamellae for record in lamella.image_records
        ]

    @property
    def display_name(self) -> str:
        """Return a user-facing project label with its number when available.

        The relative path is shown rather than the bare name, so two projects
        of the same name in different directories are distinguishable on sight.
        For a project at the scan root the two are identical.

        Where the folder is shown separately -- a tree groups by it -- that
        distinction is already made, and :attr:`own_name` is the label to use.
        """
        label = self.relative_path or self.name
        if self.number:
            return f"{self.number}: {label}"
        return label

    @property
    def folder(self) -> str:
        """Return the directory holding this project, relative to the scan root.

        Empty for a project sitting at the scan root. This is what groups
        projects for a reader: an Autolamella project is named for the minute it
        started, where the folder somebody put it in says what it was.
        """
        return folder_of(self.relative_path or self.name)

    @property
    def own_name(self) -> str:
        """Return the label for a project shown beneath its own folder.

        :attr:`display_name` without the directory part, which a tree has
        already said once and would otherwise repeat on every row under it.
        """
        name = PurePath(self.relative_path or self.name).name
        return f"{self.number}: {name}" if self.number else name


def folder_of(key: str) -> str:
    """Return the directory holding the project whose identity is *key*.

    Empty for a project at the scan root.
    """
    parent = PurePath(key).parent
    return "" if str(parent) == "." else str(parent)


def projects_holding(paths: Iterable[Path], projects: Sequence[Project]) -> list[Project]:
    """Return the projects of *projects* that *paths* live inside.

    One answer to "whose project is this file in", so that an export deciding
    whose describing files travel and a write deciding whose history to start
    cannot come to different answers about the same file.

    Matched by resolved path rather than by name, identity being a path
    everywhere here: two projects in one collection can be called the same
    thing. Returned in the order *projects* gives, so a caller's report is
    stable.
    """
    roots = {project.path.resolve(): project for project in projects}
    if not roots:
        return []
    found: dict[Path, Project] = {}
    for path in paths:
        resolved = Path(path).resolve()
        for root, project in roots.items():
            if root == resolved or root in resolved.parents:
                found.setdefault(root, project)
                break
    return [project for project in projects if project.path.resolve() in found]
