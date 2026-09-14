"""What the Curate tab's Grid view shows, and holds no Qt.

For the project being looked at: which grid overviews it holds, where its
lamellae fall on the one chosen, and why nothing is placed where nothing is.
Answered from what the scan already indexed plus image headers and one
``experiment.yaml``, held for as long as that project is on screen.

The placement is approximate -- :func:`~..overview.reproject` says what the
projection ignores -- so whatever draws it says so.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ..io.yaml import read_yaml_or_problem
from ..metadata import EXPERIMENT_FILENAME
from ..models import Project
from ..overview import (
    LamellaMarker,
    Overview,
    OverviewImage,
    StagePosition,
    place_lamellae,
    positions_from_headers,
    project_overviews,
    read_positions,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

    from ..image_metadata import ImageMetadata

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
class OverviewChoice:
    """One overview on offer, and what could be placed on it.

    Attributes:
        overview: The picture, and the file drawn for it.
        caption: What the picker calls it: when, by which beam, and how large.
        acquired_at: When its header says it was taken, where it says.
        placement: The geometry and the markers, or None where nothing can be
            placed on it.
        reason: Why nothing is placed, where nothing is -- said to the reviewer
            rather than leaving a picture with no rings to explain itself.
        scanned_height: How many of the file's rows the scan fills, as its
            header counts them, where it says. The microscope's own software
            writes an information bar below the scan, in rows it does not count.
    """

    overview: OverviewImage
    caption: str
    acquired_at: datetime | None = None
    placement: GridPlacement | None = None
    reason: str | None = None
    scanned_height: int | None = None


@dataclass(frozen=True)
class GridImage:
    """A prepared overview, or why it could not be prepared."""

    image: np.ndarray | None = None
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


class GridController:
    """Resolves a project's overviews and holds the one shown while it is on screen.

    One project at a time, which is what the Curate tab shows: moving to
    another drops the previous one rather than growing a cache, an overview
    being several megabytes.
    """

    def __init__(self) -> None:
        self._key: str | None = None
        self._choices: tuple[OverviewChoice, ...] = ()
        self._chosen = 0
        self._image: GridImage | None = None

    def set_project(self, project: Project | None) -> None:
        """Resolve the overviews of *project*, or clear them for None.

        Cheap to call repeatedly: a project already resolved is left alone, so
        the window may ask on every refresh. Opens on the earliest overview
        that places anything: taken before the lamellae were milled or coated,
        it is usually the clearest picture of where they are.
        """
        key = project.key if project is not None else None
        if key == self._key:
            return
        self._key = key
        self._image = None
        self._choices = tuple(_resolve(project)) if project is not None else ()
        self._chosen = next(
            (
                index for index, choice in enumerate(self._choices)
                if choice.placement is not None
            ),
            0,
        )

    @property
    def choices(self) -> tuple[OverviewChoice, ...]:
        """Return every overview of the project on screen, earliest first."""
        return self._choices

    @property
    def chosen(self) -> int:
        """Return the position among :attr:`choices` of the overview shown."""
        return self._chosen

    def choose(self, index: int) -> None:
        """Show the overview at *index* instead; its picture is read when asked."""
        if not self._choices:
            return
        index = max(0, min(index, len(self._choices) - 1))
        if index == self._chosen:
            return
        self._chosen = index
        self._image = None

    @property
    def current(self) -> OverviewChoice | None:
        """Return the overview shown, or None for a project holding none."""
        return self._choices[self._chosen] if self._choices else None

    @property
    def placement(self) -> GridPlacement | None:
        """Return the geometry and markers of the overview shown, if it places any."""
        current = self.current
        return current.placement if current is not None else None

    def image(self) -> GridImage | None:
        """Return the picture of the overview shown, reading it once per choice.

        Read here rather than by the frame loader: an overview belongs to the
        project rather than to any frame, so nothing about stepping through a
        timeline decides when it is wanted or when it should be dropped.
        """
        current = self.current
        if current is None:
            return None
        if self._image is None:
            self._image = _read(current.overview.display_path, current.scanned_height)
        return self._image

    def marker_named(self, name: str | None) -> LamellaMarker | None:
        """Return the marker for the lamella called *name*, if it has one."""
        placement = self.placement
        if name is None or placement is None:
            return None
        for marker in placement.markers:
            if marker.name == name:
                return marker
        return None


def _resolve(project: Project) -> list[OverviewChoice]:
    """Return every overview of *project*, earliest first, with what it places."""
    overviews = project_overviews(project, _header)
    if not overviews:
        return []
    positions = _positions(project)
    choices = [_choice(overview, positions) for overview in overviews]
    # Undated ones after the dated, keeping path order, which the sort keeps.
    return sorted(
        choices,
        key=lambda choice: (choice.acquired_at is None, choice.acquired_at or datetime.min),
    )


def _positions(project: Project) -> list[StagePosition]:
    """Return each lamella's position: ``experiment.yaml``'s, or where its images were taken.

    Headers are read only for the lamellae ``experiment.yaml`` records nothing
    for -- few in an Autolamella project, every one in a dataset predating it.
    """
    # A file that would not read was reported by the scan; its lamellae are
    # placed from their headers instead.
    recorded = read_positions(read_yaml_or_problem(project.path / EXPERIMENT_FILENAME)[0])
    named = {position.name for position in recorded}
    unrecorded = {
        lamella.name: [
            header for record in lamella.image_records
            if (header := _header(record.image_path)) is not None
        ]
        for lamella in project.lamellae
        if lamella.path != project.path and lamella.name not in named
    }
    return recorded + positions_from_headers(unrecorded)


def _header(path: Path) -> ImageMetadata | None:
    """Return the header of the image at *path*, or None where it cannot be read."""
    from ..image_metadata import read_image_metadata

    try:
        return read_image_metadata(path)
    except Exception as exc:  # noqa: BLE001 - a missing image stack included
        logger.info("Could not read %s: %s", path.name, exc)
        return None


def _choice(overview: OverviewImage, positions: list[StagePosition]) -> OverviewChoice:
    """Return *overview* as the picker offers it, placing *positions* where it can."""
    from ..overview import overview_from_metadata

    path = overview.record.image_path
    metadata = _header(path)
    caption = _caption(path, metadata)
    acquired_at = metadata.acquired_at if metadata is not None else None
    scanned_height = metadata.image_height if metadata is not None else None

    def unplaced(reason: str) -> OverviewChoice:
        return OverviewChoice(
            overview, caption, acquired_at, reason=reason, scanned_height=scanned_height
        )

    if metadata is None:
        return unplaced("no header to place positions with")
    geometry = overview_from_metadata(path, metadata)
    if geometry is None:
        return unplaced("its header states no stage geometry to place positions with")
    if not positions:
        return unplaced("neither experiment.yaml nor the lamellae's images record a position")
    markers = place_lamellae(geometry, positions)
    if not markers:
        return unplaced("no recorded position could be placed on this overview")
    return OverviewChoice(
        overview, caption, acquired_at, GridPlacement(geometry, tuple(markers)),
        scanned_height=scanned_height,
    )


def _caption(path: Path, metadata: ImageMetadata | None) -> str:
    """Return what the picker calls an overview: when, by which beam, how large.

    Taken from the header rather than the name, which only Autolamella's own
    overviews date; the name stands in where the header says nothing. The size
    tells a retake apart from the overview it replaced a minute later.
    """
    from ..image_metadata import BEAM_MODALITY

    when = metadata.acquired_at if metadata is not None else None
    stamp = f"{when:%Y-%m-%d %H:%M}" if when is not None else path.stem
    if when is not None and metadata is not None and metadata.acquired_at_timezone:
        stamp = f"{stamp} {metadata.acquired_at_timezone}"
    parts = [stamp]
    if metadata is not None and metadata.beam_type is not None:
        parts.append(BEAM_MODALITY.get(metadata.beam_type, metadata.beam_type))
    if metadata is not None and metadata.image_width:
        parts.append(f"{metadata.image_width} px")
    return " · ".join(parts)


def _read(path: Path, rows: int | None = None) -> GridImage:
    """Read an overview reduced to :data:`MAX_EDGE`, or say why it would not.

    Only its top *rows* where given: a mark is placed as a fraction of the
    scan, so a picture with the microscope's information bar still below it
    would put every mark too low, by up to the bar's height.
    """
    try:
        from ..display import fit_within, read_color_array, require_image_stack, to_rgb
    except Exception as exc:  # noqa: BLE001 - the optional image stack
        return GridImage(error=str(exc))
    try:
        require_image_stack("Displaying overviews")
        picture = to_rgb(read_color_array(path))
        if rows:
            picture = picture[:rows]
        return GridImage(image=fit_within(picture, (MAX_EDGE, MAX_EDGE)))
    except Exception as exc:  # noqa: BLE001 - an unreadable file is reportable
        logger.warning("Could not prepare overview %s: %s", path, exc)
        return GridImage(error=str(exc))
