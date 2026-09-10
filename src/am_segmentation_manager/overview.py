"""Placing a project's lamellae on the grid overview they were milled from.

An Autolamella project writes a stitched overview of the whole grid at its
root, and records every lamella's stage coordinates in ``experiment.yaml``. The
overview's own header states where the stage was when it was taken and how
large a pixel is, which is enough to say where each lamella falls in it.

Reads what :mod:`~am_segmentation_manager.image_metadata` already parsed and
what :mod:`~am_segmentation_manager.scanner` already indexed; it opens no image
and walks no directory of its own. **The position is approximate** -- see
:func:`reproject`, which says what the projection ignores -- so anything
showing one says so rather than implying a measurement.

The projection is derived from fibsemOS; :func:`reproject` names the function
and version it came from, and ``THIRD_PARTY_NOTICES.md`` carries the licence.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .image_metadata import ImageMetadata
from .models import ImageRecord, Project

logger = logging.getLogger(__name__)

#: How far a stage rotation may differ from another and still count as the
#: same, in degrees. fibsemOS's own tolerance for deciding that two positions
#: face the same way; below it a position is projected directly, and above it
#: the stage was rotated between the two and :func:`reproject` declines.
ROTATION_TOLERANCE_DEG: float = 2.0

#: How far a stage tilt may be from -180 degrees and still count as a
#: compustage mounted upside down, in degrees. fibsemOS's own tolerance.
INVERTED_TILT_TOLERANCE_DEG: float = 5.0


@dataclass(frozen=True)
class StagePosition:
    """Where the stage was, in the units the instrument records.

    Attributes:
        x: Metres along the stage's x axis.
        y: Metres along the stage's y axis.
        z: Metres along the stage's z axis.
        rotation: Stage rotation in radians.
        tilt: Stage tilt in radians.
        name: What the position is called, where anything names it -- a
            lamella's petname, which is also its directory.
    """

    x: float
    y: float
    z: float
    rotation: float
    tilt: float
    name: str | None = None


@dataclass(frozen=True)
class Overview:
    """A grid overview and the geometry that places anything on it.

    Attributes:
        path: The image file.
        width: Width in pixels.
        height: Height in pixels.
        pixel_size_m: The edge of one pixel, in metres.
        centre: Where the stage was when it was taken. For a stitched
            overview this is the centre tile's position, which is the centre
            of the stitch.
        scan_rotation: The beam's scan rotation in radians, which says whether
            the image is written the way the stage moves or turned about.
    """

    path: Path
    width: int
    height: int
    pixel_size_m: float
    centre: StagePosition
    scan_rotation: float


def overview_from_metadata(path: Path, metadata: ImageMetadata) -> Overview | None:
    """Return the geometry *metadata* states, or None if it does not state it.

    None is an ordinary answer, not a defect: an overview saved by hand from
    the microscope's own interface carries no header at all, and one is
    unusable rather than wrong. Every field is required, since a projection
    missing any of them would be a guess.
    """
    fields = (
        metadata.image_width,
        metadata.image_height,
        metadata.pixel_size_nm,
        metadata.stage_x_um,
        metadata.stage_y_um,
        metadata.stage_z_um,
        metadata.stage_rotation_deg,
        metadata.stage_tilt_deg,
        metadata.scan_rotation_deg,
    )
    if any(field is None for field in fields):
        logger.debug("No usable geometry in the header of %s", path)
        return None
    width, height, pixel_nm, x_um, y_um, z_um, r_deg, t_deg, scan_deg = fields
    if not pixel_nm or width <= 0 or height <= 0:
        logger.debug("Unusable geometry in the header of %s", path)
        return None
    return Overview(
        path=path,
        width=int(width),
        height=int(height),
        # The header states nanometres and micrometres; the projection works in
        # metres, which is what the instrument itself records.
        pixel_size_m=float(pixel_nm) * 1e-9,
        centre=StagePosition(
            x=float(x_um) * 1e-6,
            y=float(y_um) * 1e-6,
            z=float(z_um) * 1e-6,
            rotation=math.radians(float(r_deg)),
            tilt=math.radians(float(t_deg)),
        ),
        scan_rotation=math.radians(float(scan_deg)),
    )


def reproject(overview: Overview, position: StagePosition) -> tuple[float, float]:
    """Return where *position* falls on *overview*, in pixels from its top left.

    **Derived from fibsemOS**, whose acquisition software wrote both the
    overview and the coordinates: ``calculate_reprojected_stage_position`` in
    ``fibsem/imaging/tiled.py`` of fibsem 0.5.1 (MIT, Copyright (c) 2025
    fibsemOS). ``THIRD_PARTY_NOTICES.md`` carries the licence,
    ``docs/decisions.md`` says why it is restated here rather than imported,
    and ``test_overview_matches_fibsem`` holds it to fibsemOS's own answers.

    **The result is approximate, and callers must present it as one.** Two
    approximations, both fibsemOS's: the projection is flat, correcting for
    neither the perspective of a tilted stage nor the curvature of the grid,
    so the error grows with distance from the centre; and it assumes the
    overview was taken flat to its beam.

    Returns:
        ``(x, y)`` in pixels, which may fall outside the image -- a lamella
        milled off the overview is a real answer. :func:`is_inside` says which.

    Raises:
        ValueError: If *position* faces the other way from *overview*, more
            than :data:`ROTATION_TOLERANCE_DEG` from it. fibsemOS transforms
            such a position through a calibration measured on one microscope,
            which is not a constant this package can carry: nothing in the
            reference collection needs it, every overview there being taken at
            the rotation its lamellae were recorded at.
    """
    turned = abs(math.degrees(overview.centre.rotation - position.rotation))
    if turned > ROTATION_TOLERANCE_DEG:
        raise ValueError(
            f"{position.name or 'The position'} was recorded {turned:.0f} degrees "
            f"from {overview.path.name}, so it cannot be placed on it: a stage "
            f"rotated between the two needs a per-microscope calibration this "
            f"package does not carry"
        )

    delta_x = position.x - overview.centre.x
    delta_y = position.y - overview.centre.y
    delta_z = position.z - overview.centre.z

    # The stage moves in three dimensions and the image has two. y and z are
    # combined into the one distance the beam sees, signed by y: raising the
    # stage and moving it away look alike from above.
    seen_y = math.hypot(delta_y, delta_z)
    seen_y = seen_y if delta_y < 0 else -seen_y

    offset_x = delta_x / overview.pixel_size_m
    offset_y = seen_y / overview.pixel_size_m

    # A scan rotation of half a turn writes the image upside down and
    # backwards, so the offsets are measured the other way.
    if math.isclose(overview.scan_rotation, math.pi, abs_tol=1e-6):
        offset_x = -offset_x
        offset_y = -offset_y

    # A compustage holds the grid upside down, which mirrors y and not x.
    if math.isclose(
        math.degrees(overview.centre.tilt), -180.0,
        abs_tol=INVERTED_TILT_TOLERANCE_DEG,
    ):
        offset_y = -offset_y

    return (overview.width / 2 + offset_x, overview.height / 2 + offset_y)


@dataclass(frozen=True)
class LamellaMarker:
    """Where one lamella falls on an overview, as a fraction of it.

    Fractions rather than pixels so that whoever draws it may reduce the
    picture to whatever size it is shown at without the marks having to be
    recomputed, and so that nothing has to agree with anything else about the
    resolution it was drawn at.

    Attributes:
        name: The lamella directory the position was recorded for.
        x: Distance from the left edge, 1.0 being the right edge.
        y: Distance from the top edge, 1.0 being the bottom edge.
        inside: Whether it falls within the picture at all. A lamella milled
            after the grid was moved does not, which is a real answer rather
            than an error.
    """

    name: str
    x: float
    y: float
    inside: bool


def place_lamellae(
    overview: Overview, positions: Iterable[StagePosition]
) -> list[LamellaMarker]:
    """Return where each of *positions* falls on *overview*.

    A position :func:`reproject` refuses is left out rather than placed
    wrongly, with the reason logged: the whole value of the picture is that a
    mark on it is where the lamella is.
    """
    markers: list[LamellaMarker] = []
    for position in positions:
        if position.name is None:
            continue
        try:
            x, y = reproject(overview, position)
        except ValueError as exc:
            logger.info("Not placing %s on %s: %s", position.name, overview.path.name, exc)
            continue
        markers.append(
            LamellaMarker(
                name=position.name,
                x=x / overview.width,
                y=y / overview.height,
                inside=is_inside(overview, (x, y)),
            )
        )
    return markers


def is_inside(overview: Overview, point: tuple[float, float]) -> bool:
    """Return whether *point* falls within *overview*'s bounds."""
    x, y = point
    return 0 <= x < overview.width and 0 <= y < overview.height


def read_positions(experiment: Mapping[str, Any] | None) -> list[StagePosition]:
    """Return each lamella's recorded stage position from one ``experiment.yaml``.

    Named by ``petname``, which is the directory Autolamella writes that
    lamella's images into, so a caller can match a position to the lamella the
    scanner indexed. A position missing coordinates is skipped rather than
    defaulted: the whole point is where it was, and a lamella the software
    created but never positioned has no answer.

    An experiment with no ``positions`` -- every dataset predating
    ``experiment.yaml`` -- yields nothing, which is why a caller checks the
    list rather than assuming one per lamella.
    """
    positions = (experiment or {}).get("positions")
    if not isinstance(positions, list):
        return []

    found: list[StagePosition] = []
    for entry in positions:
        if not isinstance(entry, Mapping):
            continue
        name = _lamella_name(entry)
        state = entry.get("state")
        recorded = _stage_position(state)
        if recorded is None:
            logger.debug("No stage position recorded for %s", name or "a lamella")
            continue
        found.append(
            StagePosition(
                x=recorded[0], y=recorded[1], z=recorded[2],
                rotation=recorded[3], tilt=recorded[4], name=name,
            )
        )
    return found


def _lamella_name(entry: Mapping[str, Any]) -> str | None:
    """Return the directory name a position's lamella was written into.

    ``petname`` where there is one, and the last part of the recorded ``path``
    otherwise -- the same rule :func:`~.tasks.read_vocabularies` uses, because
    it is the same question about the same file.
    """
    petname = entry.get("petname")
    if isinstance(petname, str) and petname.strip():
        return petname.strip()
    path = entry.get("path")
    if isinstance(path, str) and path.strip():
        return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or None
    return None


def _stage_position(
    state: Any,
) -> tuple[float, float, float, float, float] | None:
    """Return ``(x, y, z, rotation, tilt)`` from a position's recorded state."""
    if not isinstance(state, Mapping):
        return None
    microscope_state = state.get("microscope_state")
    if not isinstance(microscope_state, Mapping):
        return None
    recorded = microscope_state.get("stage_position")
    if not isinstance(recorded, Mapping):
        return None
    values = []
    for key in ("x", "y", "z", "r", "t"):
        value = recorded.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        values.append(float(value))
    return (values[0], values[1], values[2], values[3], values[4])


#: What a display copy of an overview is called. Autolamella writes the same
#: picture twice, once as acquired and once with a gamma curve applied for
#: looking at. The copy carries the same header, so it is folded into the
#: overview it copies and drawn in its place, rather than offered as a second.
GAMMA_SUFFIX: str = "-autogamma"


def is_overview_name(image_name: str) -> bool:
    """Return whether a filename names a grid overview rather than its copy."""
    stem = Path(image_name).stem.lower()
    return "overview" in stem and not stem.endswith(GAMMA_SUFFIX)


@dataclass(frozen=True)
class OverviewImage:
    """One grid overview a project holds, as acquired and as it is looked at.

    Attributes:
        record: The overview as acquired, whose header places anything on it.
        display_path: The file to draw: the gamma copy where there is one --
            the same pixels and the same header, made for looking at -- and
            the acquired file otherwise.
    """

    record: ImageRecord
    display_path: Path


def project_overviews(project: Project) -> list[OverviewImage]:
    """Return a project's grid overviews, each once, from what the scan indexed.

    Read off the :class:`~.models.Project` rather than by looking in the
    directory: the overviews are already records, indexed at the project root
    as the ``other`` series, and a second traversal could disagree with the
    first about what is there.

    An overview belongs to the project rather than to any lamella, which is
    exactly why it sits at the root -- so this looks only at the synthetic
    lamella standing in for it, never inside a real one. A gamma copy is folded
    into the overview it copies, and a tile is part of a stitch rather than an
    overview of its own. In path order, which for Autolamella's timestamped
    names is the order they were taken.
    """
    records = [
        record
        for lamella in project.lamellae
        if lamella.path == project.path
        for record in lamella.image_records
        if record.image_path.parent == project.path
    ]
    present = {record.image_path for record in records}
    found = []
    for record in records:
        path = record.image_path
        if not is_overview_name(path.name):
            continue
        gamma = path.with_name(f"{path.stem}{GAMMA_SUFFIX}{path.suffix}")
        found.append(
            OverviewImage(record=record, display_path=gamma if gamma in present else path)
        )
    return sorted(found, key=lambda overview: overview.record.image_path)
