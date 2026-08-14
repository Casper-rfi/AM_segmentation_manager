"""Data models for am_segmentation_manager.

This module defines the core dataclasses used across the package:
- Project
- Lamella
- AdaptiveMillingStage
- ImageRecord

The dataclasses are intentionally small and typed so they are easy to test
and reason about. Filesystem scanning is separated from these models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

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
        label_shared: whether this record's label directory is shared with
            another image set, so its label also belongs to other images
        selected: whether the image is currently selected for annotation
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
    label_shared: bool = False
    selected: bool = True
    corrected: bool = False


@dataclass
class AdaptiveMillingStage:
    """A stage inside a lamella's milling history.

    Attributes:
        name: human friendly stage name (e.g. 'Adaptive_polish_...')
        path: filesystem path to the stage folder
        images: list of ImageRecord objects discovered in this stage
        number: zero-padded sequential stage number within the lamella
        plots: summary plot images written during lamella preparation, one per
            acquisition. Fewer plots than images is normal.
        synthetic: True when the dataset has no stage directory and the lamella
            directory stands in for one. Older layouts collapse this level.
        image_sets: the image directories in this stage, with the label
            directory each resolves to and whether that label directory is
            shared with another set.
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
        stages: list of AdaptiveMillingStage
        number: zero-padded lamella number parsed from the folder prefix
        synthetic: True when the dataset has no lamella directories and the
            project directory stands in for one.
    """

    name: str
    path: Path
    stages: list[AdaptiveMillingStage] = field(default_factory=list)
    number: str | None = None
    synthetic: bool = False

    @property
    def display_name(self) -> str:
        """Return the folder name, which already includes its lamella number."""
        return f"{self.name} (no lamella folder)" if self.synthetic else self.name


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
