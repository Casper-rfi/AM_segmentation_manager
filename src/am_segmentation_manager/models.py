"""Data models for am_segmentation_manager.

This module defines the core dataclasses used across the package:
- Project
- Lamella
- Stage
- ImageRecord

The dataclasses are intentionally small and typed so they are easy to test
and reason about. Filesystem scanning is separated from these models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
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
        variant: imaging-condition suffix of the image directory, or None for
            the base directory. An opaque key, never parsed.
        series: what the image was taken for -- 'adaptive', 'reference' or
            'alignment'. Orthogonal to modality and variant.
        stage_path: the stage directory the image belongs to. Carried rather
            than derived from *image_path*, because how far the stage sits
            above an image depends on the shape of its set: an Adaptive
            Polishing image sits one level down in ``sem/``, a reference image
            sits directly in the stage.
        labels_directory: where this record's label lives or would live.
            Carried for the same reason -- it cannot be recovered from a loose
            image's path, whose directory names no modality.
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
    series: str = "adaptive"
    stage_path: Path | None = None
    labels_directory: Path | None = None
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
    """

    name: str
    path: Path
    relative_path: str | None = None
    metadata: dict = field(default_factory=dict)
    experiment_metadata: ExperimentMetadata | None = None
    lamellae: list[Lamella] = field(default_factory=list)
    number: str | None = None

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
        """
        label = self.relative_path or self.name
        if self.number:
            return f"{self.number}: {label}"
        return label
