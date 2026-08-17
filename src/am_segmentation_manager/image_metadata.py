"""Acquisition metadata read from image headers.

Two dialects are read: the fibsem JSON in TIFF tag 270, where SEM and FIB share
one schema, and the older FEI vendor tag 34682, symmetric between beams with
``Beam.Beam`` naming which fired. Where each value lives is declared once, in
:data:`FIELDS`.

**A missing value is None, never a default. Headers are never read during
scanning** -- metadata is a mapping keyed by image path, built by the commands
that need it, so a scan stays a filesystem walk.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from datetime import datetime
from pathlib import Path
from typing import Any

from .io.image import require_image_stack

logger = logging.getLogger(__name__)

#: TIFF tag holding the fibsem JSON document.
FIBSEM_TAG = 270

#: TIFF tag holding the FEI/Thermo section mapping.
FEI_TAG = 34682

#: Canonical beam types. The instrument states this itself, which makes it a
#: cross-check on the directory the image was found in.
ELECTRON = "ELECTRON"
ION = "ION"

#: Beam type to the modality this package organises data by.
BEAM_MODALITY = {ELECTRON: "SEM", ION: "FIB"}

#: Keys that identify a tag-270 document as a real acquisition header.
#:
#: tifffile writes ``{"shape": [...]}`` into the description of any array it
#: saves, and Autolamella's own autofocus iteration images carry exactly that
#: and nothing else. Accepting a bare shape as a header would report an image
#: width and height for files that record no acquisition settings at all, which
#: reads as "metadata present but mostly empty" rather than "no metadata".
FIBSEM_MARKER_KEYS = frozenset({"microscope_state", "system", "pixel_size", "image"})


@dataclass(frozen=True)
class ImageMetadata:
    """Acquisition settings read from one image header.

    Every field is optional. ``None`` means the header did not record it, which
    is never the same as a default: see the module docstring.

    Attributes:
        image_width: Width in pixels.
        image_height: Height in pixels.
        pixel_size_nm: Pixel edge length in nanometres.
        dwell_time_ns: Dwell time per pixel in nanoseconds.
        image_integrations: Frames integrated. ``1`` where the header records
            no integration, including where it records an explicit null: no
            frame integration means a single frame, not an unknown number.
        line_integrations: Lines integrated, on the same basis. If there is an
            image the probe passed over each line at least once, so one is the
            physical floor rather than a guess.
        microscope_model: Instrument model, e.g. ``"Arctis"``.
        microscope_serial: Instrument serial number. Distinguishes two
            instruments of the same model, which is what "fine-tune for one
            microscope" actually needs.
        acceleration_voltage_kv: Accelerating voltage in kilovolts.
        beam_current_pa: Beam current in picoamps. Amps in both dialects and
            both beams, so one conversion covers everything.
        beam_type: ``"ELECTRON"`` or ``"ION"``, as stated by the instrument.
        detector_type: e.g. ``"ETD"``.
        detector_mode: e.g. ``"SecondaryElectrons"``.
        plasma_gas: The ion column's plasma gas, e.g. ``"Argon"``. Only ever
            read for an image the ion beam took, and ``None`` on a gallium
            column, which has no plasma and records no gas.
        electron_source: The electron column's emitter, e.g. ``"FEG"``. Only
            ever read for an image the electron beam took; the fibsem dialect
            records none, so it is ``None`` there.
        hfw_um: Horizontal field width in micrometres. With pixel size this
            pins down magnification.
        acquired_at: When the image was taken.
        source: The header dialect the values came from.
    """

    image_width: int | None = None
    image_height: int | None = None
    pixel_size_nm: float | None = None
    dwell_time_ns: float | None = None
    image_integrations: int | None = None
    line_integrations: int | None = None
    microscope_model: str | None = None
    microscope_serial: str | None = None
    acceleration_voltage_kv: float | None = None
    beam_current_pa: float | None = None
    beam_type: str | None = None
    detector_type: str | None = None
    detector_mode: str | None = None
    plasma_gas: str | None = None
    electron_source: str | None = None
    hfw_um: float | None = None
    acquired_at: datetime | None = None
    source: str | None = None

    @property
    def modality(self) -> str | None:
        """Return ``"SEM"``/``"FIB"`` from the beam the instrument reported."""
        return BEAM_MODALITY.get(self.beam_type) if self.beam_type else None

    @property
    def is_empty(self) -> bool:
        """Return whether no value at all was recovered."""
        return all(
            getattr(self, field.name) is None
            for field in dataclass_fields(self)
            if field.name != "source"
        )


@dataclass(frozen=True)
class HeaderField:
    """Where one value lives in each dialect, and what it needs multiplying by.

    Paths are key sequences. ``{beam}`` in a fibsem path expands to
    ``electron_beam``/``ion_beam``, ``{system_beam}`` to ``electron``/``ion``;
    in a FEI path ``{Beam}`` expands to ``EBeam``/``IBeam`` and ``{Scan}`` to
    ``EScan``/``IScan``. Several paths may be given, tried in order, because
    both dialects record some values in more than one place.

    Attributes:
        name: The :class:`ImageMetadata` field to fill.
        fibsem: Candidate paths in the tag-270 document.
        fei: Candidate paths in the tag-34682 mapping.
        scale: Multiplier applied to a numeric value, e.g. 1e9 for metres to
            nanometres.
        search: Keys to fall back to a recursive search for, when no declared
            path matches. This is what lets an unanticipated variant still
            produce a value; it logs when it fires, which is how a new dialect
            announces itself.
        beam_specific: Whether the value describes one beam. **An image header
            carries settings for both beams**, so a search for "voltage" across
            the whole document can return the other beam's. For these fields the
            search is confined to the subtrees belonging to the beam that took
            the image, and returns nothing rather than something plausible from
            the wrong column.
        beams: The beams the value exists for at all. For any other beam -- and
            for an image whose beam the header does not state -- the field stays
            None rather than reading a key that means nothing for the column
            that fired. Empty means every beam.
        floor: A value to use when the header records none. Set only where
            absence has a defined physical meaning -- the integration counts,
            where "not enabled" is one pass. Everywhere else absence stays
            None, because a plausible number is worse than an admitted gap.
    """

    name: str
    fibsem: tuple[tuple[str, ...], ...] = ()
    fei: tuple[tuple[str, ...], ...] = ()
    scale: float | None = None
    search: tuple[str, ...] = ()
    beam_specific: bool = False
    beams: tuple[str, ...] = ()
    floor: int | None = None


#: The field map. Adding a metadata field is a row here; supporting a new
#: dialect is a column.
FIELDS: tuple[HeaderField, ...] = (
    HeaderField(
        "image_width",
        fibsem=(("image", "resolution", 0), ("shape", 1)),
        fei=(("Image", "ResolutionX"),),
    ),
    HeaderField(
        "image_height",
        fibsem=(("image", "resolution", 1), ("shape", 0)),
        fei=(("Image", "ResolutionY"),),
    ),
    HeaderField(
        "pixel_size_nm",
        fibsem=(("pixel_size", "x"), ("pixel_size", "y")),
        fei=(("Scan", "PixelWidth"), ("{Scan}", "PixelWidth")),
        scale=1e9,
        search=("pixelwidth",),
    ),
    # image.* describes *this acquisition*; microscope_state.* is the beam's
    # state and system.* its configured default. One real header carries
    # 200ns, 300ns and 1000ns in those three places, so the order matters and
    # the image's own settings win.
    HeaderField(
        "dwell_time_ns",
        fibsem=(("image", "dwell_time"), ("microscope_state", "{beam}", "dwell_time")),
        fei=(("Scan", "Dwelltime"), ("{Scan}", "Dwell")),
        scale=1e9,
        search=("dwell_time", "dwelltime"),
        beam_specific=True,
    ),
    # Recorded by both dialects, but only sometimes by fibsem: the 2024 headers
    # carry image.frame_integration, later ones often write an explicit null.
    #
    # A null here is not "unknown" -- it is the field saying integration was not
    # enabled, and not enabled is one pass. An image exists, so the probe
    # crossed each line at least once. `floor` therefore resolves absence to 1,
    # which is decoding the field rather than defaulting it. Nothing else in
    # this table has a floor, because nothing else has a physical one: a missing
    # voltage stays None.
    HeaderField(
        "image_integrations",
        fibsem=(("image", "frame_integration"),),
        fei=(("Scan", "Integrate"), ("Image", "Integrate")),
        floor=1,
    ),
    HeaderField(
        "line_integrations",
        fibsem=(("image", "line_integration"),),
        fei=(("{Scan}", "LineIntegration"),),
        beam_specific=True,
        floor=1,
    ),
    HeaderField(
        "microscope_model",
        fibsem=(("system", "info", "model"),),
        fei=(("System", "SystemType"),),
        search=("systemtype",),
    ),
    HeaderField(
        "microscope_serial",
        fibsem=(("system", "info", "serial_number"),),
        fei=(("System", "Dnumber"),),
    ),
    # Only microscope_state, never system.*. The latter holds both beams'
    # *configured defaults* whichever beam took the image, so it is neither this
    # image's setting nor necessarily this beam's -- a plausible wrong number is
    # worse than None.
    HeaderField(
        "acceleration_voltage_kv",
        fibsem=(("microscope_state", "{beam}", "voltage"),),
        fei=(("{Beam}", "HV"), ("Beam", "HV")),
        scale=1e-3,
        search=("voltage", "hv"),
        beam_specific=True,
    ),
    HeaderField(
        "beam_current_pa",
        fibsem=(("microscope_state", "{beam}", "beam_current"),),
        fei=(("{Beam}", "BeamCurrent"),),
        scale=1e12,
        search=("beam_current", "beamcurrent"),
        beam_specific=True,
    ),
    HeaderField(
        "detector_type",
        fibsem=(("microscope_state", "{detector}", "type"),),
        fei=(("Detectors", "Name"),),
        beam_specific=True,
    ),
    HeaderField(
        "detector_mode",
        fibsem=(("microscope_state", "{detector}", "mode"),),
        fei=(("Detectors", "Mode"),),
        beam_specific=True,
    ),
    # Both dialects record these once per beam -- FEI as `Source` under each
    # column, fibsem as `plasma_gas` under each -- and write the key for the
    # column that did not fire as well, so a fibsem SEM header carries
    # system.electron.plasma_gas. `beams` is what keeps each value on the beam
    # it exists for; see docs/decisions.md § Metadata.
    HeaderField(
        "plasma_gas",
        fibsem=(("system", "ion", "plasma_gas"),),
        fei=(("IBeam", "Source"),),
        beams=(ION,),
    ),
    HeaderField(
        "electron_source",
        fei=(("EBeam", "Source"),),
        beams=(ELECTRON,),
    ),
    HeaderField(
        "hfw_um",
        fibsem=(("image", "hfw"), ("microscope_state", "{beam}", "hfw")),
        fei=(("{Beam}", "HFW"), ("Scan", "HorFieldsize")),
        scale=1e6,
        beam_specific=True,
    ),
)


def _walk(document: Any, path: tuple[Any, ...]) -> Any:
    """Return the value at *path*, or None if any step is missing."""
    node = document
    for key in path:
        if isinstance(key, int):
            if not isinstance(node, (list, tuple)) or key >= len(node):
                return None
            node = node[key]
            continue
        if not isinstance(node, Mapping):
            return None
        if key in node:
            node = node[key]
            continue
        # Case-insensitive fallback: the FEI sections are consistently cased,
        # but the legacy label directories taught us not to rely on that.
        lowered = {str(existing).lower(): existing for existing in node}
        actual = lowered.get(str(key).lower())
        if actual is None:
            return None
        node = node[actual]
    return node


def _search(node: Any, keys: tuple[str, ...], _depth: int = 0) -> Any:
    """Find the first value under any of *keys*, ignoring case and separators.

    The fallback for a dialect variant nobody anticipated. Carried over from the
    standalone extraction script, where it was what made one reader work across
    generations.
    """
    if _depth > 6 or not isinstance(node, Mapping):
        return None
    wanted = {key.lower().replace("-", "").replace("_", "") for key in keys}
    for key, value in node.items():
        if str(key).lower().replace("-", "").replace("_", "") in wanted:
            return value
    for value in node.values():
        found = _search(value, keys, _depth + 1)
        if found is not None:
            return found
    return None


def _number(value: Any, scale: float | None) -> float | int | None:
    """Return *value* as a number, scaled, or None if it is not one."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str) and (
        not value.strip() or value.strip().lower() in _NULL_STRINGS
    ):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if scale is not None:
        number *= scale
    return number


#: Strings that mean "not recorded". The fibsem writer serialises a Python
#: ``None`` into the JSON as the *string* ``"None"`` in places, and ``"null"``
#: in others -- ``user.email`` is ``"null"``, and 141 FIB images in the
#: reference collection carry ``plasma_gas: "None"``. Taking those literally
#: reports a beam source called "None", which is the reader inventing a value
#: out of a header saying it has none.
_NULL_STRINGS = frozenset({"none", "null", "nan", "n/a", "na", "-"})


def _text(value: Any) -> str | None:
    """Return *value* as a non-empty string, or None."""
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text or text.lower() in _NULL_STRINGS:
        return None
    return text


#: Fields whose value is a count or a size in pixels rather than a measurement.
_INTEGER_FIELDS = {
    "image_width", "image_height", "image_integrations", "line_integrations",
}
#: Fields that are labels rather than numbers.
_TEXT_FIELDS = {
    "microscope_model", "microscope_serial", "detector_type", "detector_mode",
    "plasma_gas", "electron_source",
}


def _beam_substitutions(beam_type: str | None) -> dict[str, str]:
    """Return the placeholder expansions for a beam type."""
    if beam_type == ION:
        return {
            "{beam}": "ion_beam", "{system_beam}": "ion", "{detector}": "ion_detector",
            "{Beam}": "IBeam", "{Scan}": "IScan",
        }
    return {
        "{beam}": "electron_beam", "{system_beam}": "electron",
        "{detector}": "electron_detector", "{Beam}": "EBeam", "{Scan}": "EScan",
    }


def _expand(path: tuple[Any, ...], substitutions: Mapping[str, str]) -> tuple[Any, ...]:
    return tuple(substitutions.get(part, part) if isinstance(part, str) else part for part in path)


#: Where a beam-specific search is allowed to look, per dialect. Everything
#: outside these subtrees may describe the *other* beam.
_BEAM_SUBTREES: dict[str, tuple[tuple[str, ...], ...]] = {
    "fibsem": (("microscope_state", "{beam}"), ("microscope_state", "{detector}")),
    "fei": (("{Beam}",), ("{Scan}",), ("Beam",), ("Scan",), ("Detectors",)),
}


def _search_roots(
    document: Mapping[str, Any], dialect: str, substitutions: Mapping[str, str]
) -> list[Any]:
    """Return the subtrees a beam-specific search may look in."""
    roots = []
    for path in _BEAM_SUBTREES.get(dialect, ()):
        node = _walk(document, _expand(path, substitutions))
        if node is not None:
            roots.append(node)
    return roots


def _read_tags(path: Path) -> tuple[Mapping[str, Any] | None, str | None]:
    """Return the parsed header document and which dialect it is."""
    import json

    import tifffile

    with tifffile.TiffFile(str(path)) as handle:
        page = handle.pages[0]
        fei = page.tags.get(FEI_TAG)
        if fei is not None and isinstance(fei.value, Mapping):
            return fei.value, "fei"
        description = page.tags.get(FIBSEM_TAG)
        if description is not None and isinstance(description.value, str):
            text = description.value.strip()
            if text.startswith("{"):
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    return None, None
                if isinstance(parsed, Mapping) and FIBSEM_MARKER_KEYS & set(parsed):
                    return parsed, "fibsem"
    return None, None


def _detect_beam(document: Mapping[str, Any], dialect: str) -> str | None:
    """Return the beam the instrument says produced the image."""
    if dialect == "fibsem":
        stated = _text(_walk(document, ("image", "beam_type")))
        if stated:
            return ION if stated.upper().startswith("ION") else ELECTRON
        # Older documents may only populate one of the two beam objects.
        if _walk(document, ("microscope_state", "ion_beam")) is not None:
            return ION
        if _walk(document, ("microscope_state", "electron_beam")) is not None:
            return ELECTRON
        return None
    stated = _text(_walk(document, ("Beam", "Beam")))
    if stated:
        return ION if stated.upper().startswith("I") else ELECTRON
    if _walk(document, ("IBeam",)) is not None:
        return ION
    if _walk(document, ("EBeam",)) is not None:
        return ELECTRON
    return None


def _acquired_at(document: Mapping[str, Any], dialect: str) -> datetime | None:
    """Return when the image was taken, from whichever form the dialect uses."""
    if dialect == "fibsem":
        stamp = _text(_walk(document, ("microscope_state", "timestamp")))
        for pattern in ("%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            if stamp:
                try:
                    return datetime.strptime(stamp, pattern)
                except ValueError:
                    continue
        return None
    date = _text(_walk(document, ("User", "Date")))
    time = _text(_walk(document, ("User", "Time")))
    if not date or not time:
        return None
    return _parse_datetime(date, time)


#: How the FEI dialect writes a date, most likely first. US ordering, because
#: that is what the instrument software writes regardless of where it is.
_DATE_TIME_PATTERNS = ("%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S")


def _parse_datetime(date: str, time: str) -> datetime | None:
    """Parse a date and a possibly 12-hour time, independently of the locale.

    The AM/PM marker is stripped and applied by hand rather than parsed with
    ``%p``. ``%p`` matches whatever the *current locale* calls the two halves
    of the day, so it reads "PM" under a C or English locale and fails under a
    German or French one -- while the header always says "PM" whatever machine
    it was written on. That made acquisition times parse on one developer's
    machine and silently come back as None on another's.
    """
    text = f"{date} {time}".strip()
    hour_offset = 0
    upper = text.upper()
    for marker, offset in (("AM", 0), ("PM", 12)):
        if upper.endswith(marker):
            text = text[: -len(marker)].strip()
            hour_offset = offset
            break
    else:
        marker = None

    for pattern in _DATE_TIME_PATTERNS:
        try:
            parsed = datetime.strptime(text, pattern)
        except ValueError:
            continue
        if marker is None:
            return parsed
        # 12 AM is midnight and 12 PM is noon, so the hour wraps rather than
        # simply gaining twelve.
        hour = parsed.hour % 12 + hour_offset
        return parsed.replace(hour=hour)

    logger.debug("Unparsed acquisition time: %r %r", date, time)
    return None


def extract_metadata(document: Mapping[str, Any], dialect: str) -> ImageMetadata:
    """Build :class:`ImageMetadata` from an already-parsed header document.

    Separated from file reading so the mapping can be tested directly, without
    writing TIFFs, and so a caller holding a header from elsewhere can reuse it.
    """
    beam = _detect_beam(document, dialect)
    substitutions = _beam_substitutions(beam)
    values: dict[str, Any] = {"beam_type": beam, "source": dialect}
    floors: dict[str, int] = {}

    for field in FIELDS:
        if field.beams and beam not in field.beams:
            continue
        candidates = field.fibsem if dialect == "fibsem" else field.fei
        raw = None
        for path in candidates:
            raw = _walk(document, _expand(path, substitutions))
            if raw is not None and raw != "":
                break
            raw = None
        if raw is None and field.search:
            # A beam-specific value is searched for only under the beam that
            # took the image. Searching the whole document would happily return
            # the other beam's voltage, which looks entirely plausible.
            if field.beam_specific:
                for root in _search_roots(document, dialect, substitutions):
                    raw = _search(root, field.search)
                    if raw is not None:
                        break
            else:
                raw = _search(document, field.search)
            if raw is not None:
                logger.debug(
                    "%s found by search rather than a declared path; the header "
                    "dialect may have changed", field.name,
                )
        if raw is None:
            if field.floor is not None:
                floors[field.name] = field.floor
            continue

        if field.name in _TEXT_FIELDS:
            values[field.name] = _text(raw)
        elif field.name in _INTEGER_FIELDS:
            number = _number(raw, field.scale)
            values[field.name] = int(round(number)) if number is not None else None
        else:
            values[field.name] = _number(raw, field.scale)

    values["acquired_at"] = _acquired_at(document, dialect)

    # Floors apply only to a header that described an acquisition. Inferring
    # "one pass" from a document that yielded nothing at all would be the
    # invention the floor exists to avoid -- there is no evidence of an
    # acquisition to reason about.
    if any(value is not None for key, value in values.items() if key != "source"):
        for name, floor in floors.items():
            values.setdefault(name, floor)

    return ImageMetadata(**{key: value for key, value in values.items() if value is not None})


def read_image_metadata(path: Path) -> ImageMetadata | None:
    """Read one image's acquisition metadata.

    Returns None when the file has no recognised header, or cannot be read at
    all. A batch over thousands of images must not stop because one file is
    truncated -- 12 of 2680 in the reference collection are -- so failure is
    logged and reported as absence.

    Raises:
        RuntimeError: If the optional image stack is not installed.
    """
    require_image_stack("Reading image metadata")
    try:
        document, dialect = _read_tags(Path(path))
    except Exception as error:  # noqa: BLE001 - one bad file must not stop a batch
        logger.warning("Could not read header of %s: %s", path, error)
        return None
    if document is None or dialect is None:
        logger.debug("No recognised metadata header in %s", path)
        return None
    return extract_metadata(document, dialect)


def read_records_metadata(records: Iterable) -> dict[Path, ImageMetadata]:
    """Read metadata for many records, keyed by image path.

    Returned as a mapping rather than attached to the records: metadata must
    stay out of the scan, and a mapping keeps the cost where it was asked for.
    Records whose header could not be read are absent from the result.
    """
    result: dict[Path, ImageMetadata] = {}
    for record in records:
        metadata = read_image_metadata(record.image_path)
        if metadata is not None:
            result[record.image_path] = metadata
    logger.info("Read metadata for %d image(s)", len(result))
    return result


def modality_disagreements(
    records: Iterable, metadata: Mapping[Path, ImageMetadata]
) -> list[tuple[Path, str, str]]:
    """Return records whose header disagrees with the directory they are in.

    The directory is what organises the data and stays authoritative; the
    instrument's own statement of which beam it used is a cross-check. A
    disagreement means a file was probably copied into the wrong place, which
    is worth reporting and never worth silently correcting.

    Returns:
        ``(image path, directory modality, header modality)`` per disagreement.
    """
    found = []
    for record in records:
        entry = metadata.get(record.image_path)
        if entry is None or entry.modality is None or record.modality is None:
            continue
        if entry.modality != record.modality:
            found.append((record.image_path, record.modality, entry.modality))
    return found
