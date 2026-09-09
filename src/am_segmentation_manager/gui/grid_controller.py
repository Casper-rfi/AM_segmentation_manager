"""What the Curate tab's Grid view shows, and holds no Qt.

One question: for the project being looked at, is there a grid overview, and
where does each of its lamellae fall on it. Answered from what the scan already
indexed plus one header and one ``experiment.yaml`` per project, held for as
long as that project is on screen.

The placement is approximate -- :func:`~..overview.reproject` says what the
projection ignores -- so whatever draws it says so.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..io.yaml import read_yaml
from ..metadata import EXPERIMENT_FILENAME
from ..models import Project
from ..overview import (
    LamellaMarker,
    Overview,
    overview_records,
    place_lamellae,
    read_positions,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = logging.getLogger(__name__)

#: The longest edge an overview is prepared at, in pixels. A grid overview is
#: 3072 or 4096 square, which as a pixmap is several times what the pane can
#: show; this is the same trade the plot cache makes, and the picture is a map
#: rather than something anyone inspects at full resolution.
MAX_EDGE: int = 2048


@dataclass(frozen=True)
class GridPlacement:
    """An overview and where a project's lamellae fall on it.

    Attributes:
        overview: The picture and the geometry that placed the markers.
        markers: One per lamella whose position could be placed, in the order
            ``experiment.yaml`` records them.
    """

    overview: Overview
    markers: tuple[LamellaMarker, ...]


@dataclass(frozen=True)
class GridImage:
    """A prepared overview, or why it could not be prepared."""

    image: np.ndarray | None = None
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


class GridController:
    """Resolves a project's overview and holds it while that project is shown.

    One project at a time, which is what the Curate tab shows: moving to
    another drops the previous one rather than growing a cache, an overview
    being several megabytes.
    """

    def __init__(self) -> None:
        self._key: str | None = None
        self._placement: GridPlacement | None = None
        self._image: GridImage | None = None

    def set_project(self, project: Project | None) -> None:
        """Resolve the overview for *project*, or clear it for None.

        Cheap to call repeatedly: a project already resolved is left alone, so
        the window may ask on every refresh.
        """
        key = project.key if project is not None else None
        if key == self._key:
            return
        self._key = key
        self._image = None
        self._placement = _resolve(project) if project is not None else None

    @property
    def placement(self) -> GridPlacement | None:
        """Return the overview and markers for the project on screen, if any.

        None is the ordinary answer for most projects: only the 2025 layout
        and later writes an overview, and one saved by hand carries no header
        to place anything with.
        """
        return self._placement

    def image(self) -> GridImage | None:
        """Return the overview picture, reading it once per project.

        Read here rather than by the frame loader: an overview belongs to the
        project rather than to any frame, so nothing about stepping through a
        timeline decides when it is wanted or when it should be dropped.
        """
        if self._placement is None:
            return None
        if self._image is None:
            self._image = _read(self._placement.overview.path)
        return self._image

    def marker_named(self, name: str | None) -> LamellaMarker | None:
        """Return the marker for the lamella called *name*, if it has one."""
        if name is None or self._placement is None:
            return None
        for marker in self._placement.markers:
            if marker.name == name:
                return marker
        return None


def _resolve(project: Project) -> GridPlacement | None:
    """Return the first usable overview of *project* and its markers."""
    from ..image_metadata import read_image_metadata
    from ..overview import overview_from_metadata

    records = overview_records(project)
    if not records:
        return None
    positions = read_positions(read_yaml(project.path / EXPERIMENT_FILENAME))
    if not positions:
        logger.debug("%s records no lamella positions to place", project.name)
        return None

    for record in records:
        try:
            metadata = read_image_metadata(record.image_path)
        except Exception as exc:  # noqa: BLE001 - a missing image stack included
            logger.info("Could not read %s: %s", record.image_path.name, exc)
            continue
        overview = overview_from_metadata(record.image_path, metadata)
        if overview is None:
            continue
        markers = place_lamellae(overview, positions)
        if markers:
            return GridPlacement(overview=overview, markers=tuple(markers))
    logger.debug("No overview of %s can place its lamellae", project.name)
    return None


def _read(path: Path) -> GridImage:
    """Read an overview reduced to :data:`MAX_EDGE`, or say why it would not."""
    try:
        from ..display import prepare_plot
    except Exception as exc:  # noqa: BLE001 - the optional image stack
        return GridImage(error=str(exc))
    try:
        return GridImage(image=prepare_plot(path, max_size=(MAX_EDGE, MAX_EDGE)))
    except Exception as exc:  # noqa: BLE001 - an unreadable file is reportable
        logger.warning("Could not prepare overview %s: %s", path, exc)
        return GridImage(error=str(exc))
