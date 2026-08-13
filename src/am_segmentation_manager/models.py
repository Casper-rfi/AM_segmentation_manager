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
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


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
        selected: whether the image is currently selected for annotation
    """

    image_path: Path
    predictions: List[Path] = field(default_factory=list)
    labels: List[Path] = field(default_factory=list)
    project: Optional[str] = None
    lamella: Optional[str] = None
    stage: Optional[str] = None
    timestamp: Optional[datetime] = None
    modality: Optional[str] = None
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
    """

    name: str
    path: Path
    images: List[ImageRecord] = field(default_factory=list)
    number: Optional[str] = None
    plots: List[Path] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        """Return a user-facing stage label with its number when available."""
        if self.number:
            return f"{self.number}: {self.name}"
        return self.name


@dataclass
class Lamella:
    """Representation of a lamella inside an Autolamella project.
    
    Attributes:
        name: lamella folder name
        path: path to lamella folder
        stages: list of AdaptiveMillingStage
        number: zero-padded lamella number parsed from the folder prefix
    """

    name: str
    path: Path
    stages: List[AdaptiveMillingStage] = field(default_factory=list)
    number: Optional[str] = None

    @property
    def display_name(self) -> str:
        """Return the folder name, which already includes its lamella number."""
        return self.name


@dataclass
class Project:
    """Representation of an Autolamella project.

    Attributes:
        name: user-facing project name
        path: path to project root
        metadata: raw metadata mapping loaded from experiment.yaml and AM_metadata.yaml
        lamellae: list of Lamella objects
    """

    name: str
    path: Path
    metadata: Dict = field(default_factory=dict)
    lamellae: List[Lamella] = field(default_factory=list)
    number: Optional[str] = None

    @property
    def display_name(self) -> str:
        """Return a user-facing project label with its number when available."""
        if self.number:
            return f"{self.number}: {self.name}"
        return self.name
