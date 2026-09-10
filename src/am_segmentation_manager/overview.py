"""Placing a project's lamellae on the grid overview they were milled from.

An overview is an image named as one or as wide as one, at the project root or
where fibsemOS's grid tasks write it; a lamella is where ``experiment.yaml``
records it or, failing that, where its own images were taken. The overview's
header states where the stage was and how large a pixel is, which is enough to
say where each lamella falls in it.

Reads what :mod:`~am_segmentation_manager.image_metadata` parsed and what
:mod:`~am_segmentation_manager.scanner` indexed, headers through a reader its
caller passes; it walks no directory of its own. **The position is approximate** -- see
:func:`reproject`, which says what the projection ignores -- so anything
showing one says so rather than implying a measurement.

The projection is derived from fibsemOS; :func:`reproject` names the function
and version it came from, and ``THIRD_PARTY_NOTICES.md`` carries the licence.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .image_metadata import ELECTRON, ION, ImageMetadata
from .models import ImageRecord, Project

logger = logging.getLogger(__name__)

#: How far a stage rotation may differ from another and still count as the
#: same, in degrees. fibsemOS's own tolerance for deciding that two positions
#: face the same way; below it a position is projected directly, and above it
#: the stage was rotated between the two and :func:`reproject` declines.
ROTATION_TOLERANCE_DEG: float = 2.0

#: How far a scan rotation may be from a whole number of half turns and still
#: count as one, in radians: the rounding a header's decimals introduce.
SCAN_ROTATION_TOLERANCE_RAD: float = 1e-3

#: How far a compustage may be from the pose facing the ion column and still
#: face it: fibsemOS's own tolerances, on the tilt in radians and on the
#: rotation in degrees.
FACING_TILT_TOLERANCE_RAD: float = 0.1
FACING_ROTATION_TOLERANCE_DEG: float = 5.0


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
        coordinate_system: ``RAW`` or ``SPECIMEN``, in whatever case the
            recording wrote it, or None where it does not say. The same stage
            has different coordinates in each.
    """

    x: float
    y: float
    z: float
    rotation: float
    tilt: float
    name: str | None = None
    coordinate_system: str | None = None


@dataclass(frozen=True)
class Overview:
    """A grid overview and the geometry that places anything on it.

    The defaults are a flat stage seen by the electron column, which is what a
    test builds by hand; a header states every one of them.

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
        beam: ``"ELECTRON"`` or ``"ION"``: which column took it, and so which
            angle it saw the stage at.
        ion_column_tilt: The ion column's angle from the electron column, in
            radians, which is what an ion-beam overview is seen through.
        compustage: Whether the stage is a compustage, which holds the grid
            upside down and does not rotate. fibsemOS knows one by the
            microscope's model name.
        shuttle_pretilt: The angle the shuttle holds the grid at, in radians.
            Read only for a compustage, for the tilt that faces it to the ion
            column.
        rotation_reference: The stage rotation facing the grid to the electron
            column, in radians. Read only for a compustage, which faces the
            ion column only at this rotation.
    """

    path: Path
    width: int
    height: int
    pixel_size_m: float
    centre: StagePosition
    scan_rotation: float
    beam: str = ELECTRON
    ion_column_tilt: float = math.radians(52)
    compustage: bool = False
    shuttle_pretilt: float = 0.0
    rotation_reference: float = 0.0


#: The model name by which fibsemOS recognises a compustage.
COMPUSTAGE_MODEL: str = "Arctis"

#: What fibsemOS configures an Arctis with, standing in for what a header saved
#: by the microscope's own software leaves out: a compustage cannot rotate,
#: holds the grid with no pre-tilt, and meets the ion column at 52 degrees.
COMPUSTAGE_DEFAULTS: Mapping[str, float] = {
    "stage_rotation_deg": 0.0,
    "shuttle_pretilt_deg": 0.0,
    "stage_rotation_reference_deg": 0.0,
    "ion_column_tilt_deg": 52.0,
}

#: The coordinate system a compustage's position is taken to be in where its
#: header does not say: the only one fibsemOS records an Arctis in. Assumed of
#: the microscope's own headers, no file in the reference collection carrying
#: both kinds of header to compare.
COMPUSTAGE_FRAME: str = "SPECIMEN"


def overview_from_metadata(path: Path, metadata: ImageMetadata) -> Overview | None:
    """Return the geometry *metadata* states, or None if it does not state it.

    None is an ordinary answer, not a defect: a flat stage's overview saved
    from the microscope's own interface may state no rotation, and is unusable
    rather than wrong. A compustage's takes :data:`COMPUSTAGE_DEFAULTS` for
    what it leaves out; anything else the projection reads is required, since a
    projection missing it would be a guess.
    """
    compustage = _is_compustage(metadata)
    ion_column_tilt = _stated(metadata, "ion_column_tilt_deg", compustage)
    fields = (
        metadata.image_width,
        metadata.image_height,
        metadata.pixel_size_nm,
        metadata.stage_x_um,
        metadata.stage_y_um,
        metadata.stage_z_um,
        _stated(metadata, "stage_rotation_deg", compustage),
        metadata.stage_tilt_deg,
        metadata.scan_rotation_deg,
        metadata.beam_type,
    )
    if any(field is None for field in fields) or (
        metadata.beam_type == ION and ion_column_tilt is None
    ):
        logger.debug("No usable geometry in the header of %s", path)
        return None
    width, height, pixel_nm, x_um, y_um, z_um, r_deg, t_deg, scan_deg, beam = fields
    if not pixel_nm or width <= 0 or height <= 0:
        logger.debug("Unusable geometry in the header of %s", path)
        return None
    # The header states nanometres, micrometres and degrees; the projection
    # works in metres and radians, which is what the instrument records.
    column_geometry: dict[str, float] = {}
    if ion_column_tilt is not None:
        column_geometry["ion_column_tilt"] = math.radians(float(ion_column_tilt))
    if compustage:
        column_geometry["shuttle_pretilt"] = math.radians(
            float(_stated(metadata, "shuttle_pretilt_deg", compustage))
        )
        column_geometry["rotation_reference"] = math.radians(
            float(_stated(metadata, "stage_rotation_reference_deg", compustage))
        )
    return Overview(
        path=path,
        width=int(width),
        height=int(height),
        pixel_size_m=float(pixel_nm) * 1e-9,
        centre=StagePosition(
            x=float(x_um) * 1e-6,
            y=float(y_um) * 1e-6,
            z=float(z_um) * 1e-6,
            rotation=math.radians(float(r_deg)),
            tilt=math.radians(float(t_deg)),
            coordinate_system=metadata.stage_coordinate_system,
        ),
        scan_rotation=math.radians(float(scan_deg)),
        beam=beam,
        compustage=compustage,
        **column_geometry,
    )


def _is_compustage(metadata: ImageMetadata) -> bool:
    """Return whether *metadata* was recorded on a compustage."""
    return COMPUSTAGE_MODEL in (metadata.microscope_model or "")


def _stated(metadata: ImageMetadata, field: str, compustage: bool) -> Any:
    """Return *field* of *metadata*, or what fibsemOS configures a compustage with."""
    value = getattr(metadata, field)
    if value is None and compustage:
        return COMPUSTAGE_DEFAULTS.get(field)
    return value


def reproject(overview: Overview, position: StagePosition) -> tuple[float, float]:
    """Return where *position* falls on *overview*, in pixels from its top left.

    **Derived from fibsemOS**, whose acquisition software wrote both the
    overview and the coordinates: ``calculate_reprojected_stage_position2`` and
    the ``inverse_view_corrected_dy`` it reaches, in
    ``fibsem/imaging/tiling/reprojection.py`` and ``fibsem/transformations.py``
    as fibsemOS's FIB-766 left them, first released in fibsem 0.5.2rc1 (MIT,
    Copyright (c) 2025 fibsemOS) -- the projection fibsemOS's own minimap
    draws with. ``THIRD_PARTY_NOTICES.md`` carries the licence,
    ``docs/decisions.md`` says why it is restated here rather than imported,
    ``test_overview_matches_fibsem`` holds it to fibsemOS's own answers, and
    ``test_each_overview_places_its_lamellae_to_within_tens_of_microns`` to
    where a person placed the lamellae.

    **The result is approximate, and callers must present it as one.** The
    stage is treated as rigid, so a move is seen foreshortened by the angle
    between the stage and the beam, its height included, and the grid's shape
    does not enter. What is ignored: the tilt a position was recorded at, the
    overview's own being used; and beam shift.

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
        ValueError: If *overview* was scanned at a rotation other than a whole
            number of half turns, which fibsemOS does not model either.
        ValueError: If *position* and *overview* are known to be in different
            coordinate systems, which put the same stage in different places.
    """
    frame, recorded = _frame(overview, overview.centre), _frame(overview, position)
    if frame is not None and recorded is not None and frame != recorded:
        raise ValueError(
            f"{position.name or 'The position'} was recorded in {recorded} "
            f"coordinates and {overview.path.name} in {frame}, which put the same "
            f"stage in different places"
        )
    half_turns = _half_turns(overview.scan_rotation)
    if half_turns is None:
        raise ValueError(
            f"{overview.path.name} was scanned at "
            f"{math.degrees(overview.scan_rotation):.1f} degrees of rotation, so nothing "
            f"can be placed on it: only a scan square to the stage or half a turn "
            f"from it is modelled"
        )
    turned = abs(math.degrees(overview.centre.rotation - position.rotation))
    if turned > ROTATION_TOLERANCE_DEG:
        raise ValueError(
            f"{position.name or 'The position'} was recorded {turned:.0f} degrees "
            f"from {overview.path.name}, so it cannot be placed on it: a stage "
            f"rotated between the two needs a per-microscope calibration this "
            f"package does not carry"
        )

    offset_x = (position.x - overview.centre.x) / overview.pixel_size_m
    offset_y = -_seen_along_y(
        overview, position.y - overview.centre.y, position.z - overview.centre.z
    ) / overview.pixel_size_m

    # A scan rotation of half a turn writes the image upside down and
    # backwards, so the offsets are measured the other way.
    if half_turns % 2:
        offset_x = -offset_x
        offset_y = -offset_y

    return (overview.width / 2 + offset_x, overview.height / 2 + offset_y)


def _half_turns(scan_rotation: float) -> int | None:
    """Return how many half turns *scan_rotation* is, or None if not a whole number."""
    turns = round(scan_rotation / math.pi)
    if abs(scan_rotation - turns * math.pi) > SCAN_ROTATION_TOLERANCE_RAD:
        return None
    return turns


def _frame(overview: Overview, position: StagePosition) -> str | None:
    """Return the coordinate system *position* is in, on the stage *overview* was taken on."""
    if position.coordinate_system:
        return position.coordinate_system.upper()
    return COMPUSTAGE_FRAME if overview.compustage else None


def _seen_along_y(overview: Overview, delta_y: float, delta_z: float) -> float:
    """Return how far along the image's y axis a stage move is seen, in metres.

    The stage's y and z axes tilt with it, so both are seen through the angle
    between the stage and the beam, measured from the electron column. The
    shuttle's pre-tilt does not enter: it decides where the grid lies, not how
    the stage carrying it moves.
    """
    stage_tilt = overview.centre.tilt
    compustage_sign = 1.0
    if overview.compustage:
        # A compustage images the back of the grid, half a turn over, and is
        # seen the other way up except when it faces the ion column.
        compustage_sign = 1.0 if _faces_the_ion_column(overview) else -1.0
        stage_tilt += math.pi

    seen_at = stage_tilt - (overview.ion_column_tilt if overview.beam == ION else 0.0)
    return (delta_y * math.cos(seen_at) - delta_z * math.sin(seen_at)) * compustage_sign


def _faces_the_ion_column(overview: Overview) -> bool:
    """Return whether a compustage was posed facing the ion column, as fibsemOS decides it.

    That pose is tilted by the ion column's angle less the pre-tilt and a half
    turn, at the reference rotation, a compustage having no other.
    """
    facing_tilt = overview.ion_column_tilt - overview.shuttle_pretilt - math.pi
    if abs(overview.centre.tilt - facing_tilt) > FACING_TILT_TOLERANCE_RAD:
        return False
    full = 2 * math.pi
    first, second = overview.centre.rotation % full, overview.rotation_reference % full
    large, small = max(first, second), min(first, second)
    return min(large - small, full + small - large) < math.radians(FACING_ROTATION_TOLERANCE_DEG)


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
    defaulted: the whole point is where it was, and for a lamella the software
    created but never positioned :func:`positions_from_headers` answers instead.

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
        recorded = _stage_position(entry.get("state"), name)
        if recorded is None:
            logger.debug("No stage position recorded for %s", name or "a lamella")
            continue
        found.append(recorded)
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


def _stage_position(state: Any, name: str | None) -> StagePosition | None:
    """Return the stage position a lamella's recorded state holds, if it holds one."""
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
    frame = recorded.get("coordinate_system")
    return StagePosition(
        x=values[0], y=values[1], z=values[2], rotation=values[3], tilt=values[4],
        name=name, coordinate_system=frame if isinstance(frame, str) else None,
    )


#: How close two of a lamella's images must be to count as taken at one place,
#: in micrometres. A lamella's images sit within a few of each other; a
#: lift-out lamella that moves goes far further.
PLACE_TOLERANCE_UM: float = 30.0


def positions_from_headers(
    images: Mapping[str, Iterable[ImageMetadata]],
) -> list[StagePosition]:
    """Return where each lamella's own images were taken, from their headers.

    For the lamellae ``experiment.yaml`` records no position for, which in a
    dataset predating it is all of them: the place most of a lamella's images
    were taken, those within :data:`PLACE_TOLERANCE_UM` of each other being one
    place and the earlier of equal places winning, at the first image taken
    there. An image as wide as an overview says nothing about a lamella and is
    passed over, so a folder of overviews or of a stitch's tiles places nothing.

    Args:
        images: Each lamella's name, with the headers of its images in scan order.
    """
    found: list[StagePosition] = []
    for name, headers in images.items():
        places: list[list[StagePosition]] = []
        for metadata in headers:
            position = _position_of(metadata, name)
            if position is None:
                continue
            for place in places:
                if math.hypot(position.x - place[0].x, position.y - place[0].y) * 1e6 <= (
                    PLACE_TOLERANCE_UM
                ):
                    place.append(position)
                    break
            else:
                places.append([position])
        if places:
            found.append(max(places, key=len)[0])
    return found


def _position_of(metadata: ImageMetadata, name: str) -> StagePosition | None:
    """Return where the stage was for one of a lamella's images, if its header says."""
    if _is_wide(metadata):
        return None
    values = (
        metadata.stage_x_um,
        metadata.stage_y_um,
        metadata.stage_z_um,
        _stated(metadata, "stage_rotation_deg", _is_compustage(metadata)),
        metadata.stage_tilt_deg,
    )
    if any(value is None for value in values):
        return None
    x_um, y_um, z_um, r_deg, t_deg = (float(value) for value in values)
    return StagePosition(
        x=x_um * 1e-6, y=y_um * 1e-6, z=z_um * 1e-6,
        rotation=math.radians(r_deg), tilt=math.radians(t_deg),
        name=name, coordinate_system=metadata.stage_coordinate_system,
    )


#: What a display copy of an overview is called. Autolamella writes the same
#: picture twice, once as acquired and once with a gamma curve applied for
#: looking at. The copy carries the same header, so it is folded into the
#: overview it copies and drawn in its place, rather than offered as a second.
GAMMA_SUFFIX: str = "-autogamma"


#: The narrowest field of view that makes an image an overview whatever it is
#: called, in micrometres. Every image a lamella's own workflow takes in the
#: reference collection is under 200 wide, and every wider one beside them 500
#: or more -- a grid, several of its squares, or a stitch's tile.
OVERVIEW_MIN_FIELD_UM: float = 500.0

#: How many path parts below the project root fibsemOS's grid tasks write an
#: overview: ``<grid>/<task>/<image>``. A stitch's tiles sit a folder further.
GRID_TASK_DEPTH: int = 3


def is_overview_name(image_name: str) -> bool:
    """Return whether a filename names a grid overview rather than its copy."""
    stem = Path(image_name).stem.lower()
    return "overview" in stem and not stem.endswith(GAMMA_SUFFIX)


def _is_wide(metadata: ImageMetadata | None) -> bool:
    """Return whether *metadata* shows a field of view only an overview reaches."""
    return (
        metadata is not None
        and metadata.hfw_um is not None
        and metadata.hfw_um >= OVERVIEW_MIN_FIELD_UM
    )


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


def _unread(path: Path) -> ImageMetadata | None:
    return None


def project_overviews(
    project: Project,
    read_header: Callable[[Path], ImageMetadata | None] = _unread,
) -> list[OverviewImage]:
    """Return a project's grid overviews, each once, from what the scan indexed.

    Read off the :class:`~.models.Project` rather than by looking in the
    directory: the overviews are already records, indexed as the ``other``
    series, and a second traversal could disagree with the first about what is
    there.

    An overview belongs to the project rather than to any lamella, so it is
    looked for at the root, named as one or as wide as one, and where
    fibsemOS's grid tasks write it, :data:`GRID_TASK_DEPTH` parts down, as wide
    as one -- never directly inside a lamella, and never among a stitch's tiles.
    *read_header* is asked only about ``other`` images whose name does not
    already decide. A gamma copy is folded into the overview it copies. In path
    order, which for Autolamella's timestamped names is the order they were taken.
    """
    records = [record for lamella in project.lamellae for record in lamella.image_records]
    present = {record.image_path for record in records}
    found = []
    for record in records:
        path = record.image_path
        if not _is_project_overview(record, project.path, read_header):
            continue
        gamma = path.with_name(f"{path.stem}{GAMMA_SUFFIX}{path.suffix}")
        found.append(
            OverviewImage(record=record, display_path=gamma if gamma in present else path)
        )
    return sorted(found, key=lambda overview: overview.record.image_path)


def _is_project_overview(
    record: ImageRecord, root: Path, read_header: Callable[[Path], ImageMetadata | None]
) -> bool:
    """Return whether *record* is one of the project's overviews rather than a lamella's."""
    path = record.image_path
    if path.stem.lower().endswith(GAMMA_SUFFIX):
        return False
    try:
        depth = len(path.relative_to(root).parts)
    except ValueError:
        return False
    if depth == 1 and is_overview_name(path.name):
        return True
    if depth not in (1, GRID_TASK_DEPTH) or record.series != "other":
        return False
    return _is_wide(read_header(path))
