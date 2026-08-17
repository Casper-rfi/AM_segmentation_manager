"""Record filtering and scope-matching utilities.

This module provides functions to match projects, lamellae, stages and
images against user-supplied selectors (names, numbers, ranges).  The
utilities are used by the CLI and can be reused by any future feature
that needs to select a subset of indexed records (e.g. filtered CSV
export, metadata display).
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from pathlib import Path

from .models import ImageRecord
from .scanner import ProjectIndex

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Selector parsing
# ---------------------------------------------------------------------------

def parse_scope_number(value: str | None) -> str | None:
    """Return a zero-padded numeric value for simple scope filters.

    Strips leading zeros and re-pads to two digits so that ``"1"``,
    ``"01"`` and ``"001"`` all match lamella / stage ``"01"``.
    """
    if value is None:
        return None
    match = re.match(r"^0*(\d+)$", value.strip())
    if not match:
        return None
    return f"{int(match.group(1)):02d}"


def parse_selector_values(values: list[str] | None) -> tuple[set[int], set[str]]:
    """Parse comma-separated numeric ranges and exact names from CLI selectors.

    Returns a tuple of ``(numeric_values, name_values)`` where each element
    in *values* is split on commas and classified:

    * Pure integers and ``start-end`` ranges are added to *numeric_values*.
    * Everything else (e.g. lamella folder names) is added to *name_values*.
    """
    numeric_values: set[int] = set()
    name_values: set[str] = set()
    for raw in values or []:
        for item in raw.split(","):
            value = item.strip()
            if not value:
                continue
            match = re.fullmatch(r"(\d+)(?:-(\d+))?", value)
            if match is None:
                name_values.add(value)
                continue
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) is not None else start
            if end < start:
                start, end = end, start
            for number in range(start, end + 1):
                numeric_values.add(number)
    return numeric_values, name_values


# ---------------------------------------------------------------------------
# Scope matchers
# ---------------------------------------------------------------------------

def matches_project(project, selectors: list[str] | None) -> bool:
    """Return whether *project* matches the user-supplied selectors.

    Matches by relative path, project name, or zero-padded scan-order number.
    The relative path is the project's identity, and is the only selector that
    can distinguish two projects sharing a name; the scan-order number is the
    convenient equivalent.
    """
    numeric_values, name_values = parse_selector_values(selectors)
    if "all" in name_values:
        return True
    identifiers = {project.name}
    if getattr(project, "relative_path", None):
        identifiers.add(project.relative_path)
    if identifiers & name_values:
        return True
    project_number = parse_scope_number(project.number)
    return project_number is not None and int(project_number) in numeric_values


def matches_lamella(lamella, selectors: list[str] | None) -> bool:
    """Return whether *lamella* matches the user-supplied selectors."""
    numeric_values, name_values = parse_selector_values(selectors)
    if not numeric_values and not name_values:
        return True
    if "all" in name_values or lamella.name in name_values:
        return True
    lamella_number = parse_scope_number(lamella.number)
    if lamella_number is not None and int(lamella_number) in numeric_values:
        return True
    return False


def matches_stage(stage, selectors: list[str] | None) -> bool:
    """Return whether *stage* matches the user-supplied selectors."""
    numeric_values, name_values = parse_selector_values(selectors)
    if not numeric_values and not name_values:
        return True
    if "all" in name_values or stage.name in name_values:
        return True
    stage_number = parse_scope_number(stage.number)
    if stage_number is not None and int(stage_number) in numeric_values:
        return True
    return False


def extract_image_number(image_name: str) -> int | None:
    """Extract the image number from an Autolamella image filename.

    Recognises patterns like ``AP_img_013`` or ``img_013``.
    """
    match = re.search(r"AP_img_(\d+)", image_name, re.IGNORECASE)
    if match is not None:
        return int(match.group(1))
    match = re.search(r"img_(\d+)", image_name, re.IGNORECASE)
    if match is not None:
        return int(match.group(1))
    return None


def matches_image(record, selectors: list[str] | None) -> bool:
    """Return whether *record* matches image selectors (number or filename)."""
    numeric_values, name_values = parse_selector_values(selectors)
    if not numeric_values and not name_values:
        return True
    if "all" in name_values:
        return True
    image_number = extract_image_number(record.image_path.name)
    if image_number is not None and image_number in numeric_values:
        return True
    if record.image_path.name in name_values:
        return True
    return False


def record_matches_paths(record, supplied_paths: list[str] | None) -> bool:
    """Return whether *record* matches any of the supplied filesystem paths."""
    if not supplied_paths:
        return True
    for supplied in supplied_paths:
        sp = Path(supplied)
        try:
            if sp.resolve() == record.image_path.resolve():
                return True
        except Exception:
            if sp.name == record.image_path.name:
                return True
    return False


#: Selector meaning "the unsuffixed image directory of each modality".
BASE_VARIANT = "base"


def matches_variant(record: ImageRecord, selectors: list[str] | None) -> bool:
    """Return whether *record* matches the requested imaging-condition variants.

    Selectors are matched against the variant string exactly, since a variant
    is opaque to this package. ``"base"`` selects the unsuffixed image
    directory, and ``"all"`` selects everything. ``None`` matches everything,
    so callers that do not care about variants are unaffected.
    """
    if not selectors:
        return True
    values = {value.strip() for raw in selectors for value in raw.split(",") if value.strip()}
    if not values or "all" in values:
        return True
    if record.variant is None:
        return BASE_VARIANT in values
    return record.variant in values


def matches_state(record: ImageRecord, state: str) -> bool:
    """Return whether *record* matches the requested workflow state.

    States:
    * ``"corrected"``: record was marked corrected
    * ``"selected"``: record is selected and not corrected
    * ``"deselected"``: record is deselected and not corrected
    * ``"all"``: matches any workflow state
    """
    if state == "all":
        return True
    if state == "corrected":
        return record.corrected
    if state == "selected":
        return record.selected and not record.corrected
    if state == "deselected":
        return not record.selected and not record.corrected
    return False


# ---------------------------------------------------------------------------
# Compound iterator
# ---------------------------------------------------------------------------

def iter_matching_records(
    index: ProjectIndex,
    project_selectors: list[str] | None = None,
    lamella_selectors: list[str] | None = None,
    stage_selectors: list[str] | None = None,
    image_selectors: list[str] | None = None,
    paths: list[str] | None = None,
    modality: str = "all",
    state: str = "all",
    variant_selectors: list[str] | None = None,
) -> Iterator[ImageRecord]:
    """Yield every :class:`ImageRecord` that passes the supplied filters.

    Filters combine with AND: a record must satisfy every one supplied.

    Args:
        index: The scanned project index to walk.
        project_selectors: Match these projects; ``None`` matches all. The
            lamella, stage and image selectors behave the same way.
        paths: Match records at these paths.
        modality: ``"SEM"``, ``"FIB"`` or ``"all"``.
        state: ``"corrected"``, ``"selected"``, ``"deselected"`` or ``"all"``.
        variant_selectors: Imaging-condition variants, where ``"base"`` is the
            unsuffixed image directory.
    """
    for project in index.all_projects():
        if project_selectors is not None and not matches_project(project, project_selectors):
            continue
        for lamella in project.lamellae:
            if lamella_selectors is not None and not matches_lamella(lamella, lamella_selectors):
                continue
            for stage in lamella.stages:
                if stage_selectors is not None and not matches_stage(stage, stage_selectors):
                    continue
                for record in stage.images:
                    if modality != "all" and record.modality != modality:
                        continue
                    if not matches_variant(record, variant_selectors):
                        continue
                    if state != "all" and not matches_state(record, state):
                        continue
                    if image_selectors is not None and not matches_image(record, image_selectors):
                        continue
                    if record_matches_paths(record, paths):
                        yield record


# ---------------------------------------------------------------------------
# Metadata filters
#
# Applied *after* the structural selectors above, deliberately. Filtering by
# project and lamella first means headers are read only for what survives, so
# the cost of a metadata query stays proportional to what it asks about rather
# than to the size of the archive.
# ---------------------------------------------------------------------------

#: Metadata filters, mapped to the ImageMetadata attribute they test. Keeping
#: this a table rather than a chain of if-statements means a new filter is a
#: row, matching how image_metadata.FIELDS declares where values come from.
METADATA_FILTERS: dict[str, str] = {
    "microscope": "microscope_model",
    "serial": "microscope_serial",
    "voltage": "acceleration_voltage_kv",
    "current": "beam_current_pa",
    "dwell": "dwell_time_ns",
    "pixel-size": "pixel_size_nm",
    "plasma-gas": "plasma_gas",
    "electron-source": "electron_source",
    "detector": "detector_type",
}

#: How the unit a field name ends in is spelled for a reader. A filter takes the
#: value in the unit the field is stored in, so the unit has to be visible
#: wherever the filter is offered -- "acceleration voltage kv" invites volts.
FIELD_UNITS: dict[str, str] = {
    "kv": "kV", "nm": "nm", "pa": "pA", "ns": "ns", "um": "µm",
}


def filter_property(name: str) -> str:
    """Return what a metadata filter matches on, with its unit, for a reader.

    Raises:
        KeyError: If *name* is not a metadata filter.
    """
    words = METADATA_FILTERS[name].split("_")
    unit = FIELD_UNITS.get(words[-1])
    if unit is None:
        return " ".join(words)
    return f"{' '.join(words[:-1])} ({unit})"


def _matches_value(actual, wanted: str) -> bool:
    """Compare a metadata value to a requested one, tolerantly.

    Numbers are compared numerically where both sides parse, so ``30000``
    matches a stored ``30000.0``; otherwise the comparison is a
    case-insensitive string match, so ``arctis`` finds ``Arctis``.
    """
    if actual is None:
        return False
    try:
        return abs(float(actual) - float(wanted)) < 1e-9
    except (TypeError, ValueError):
        return str(actual).strip().lower() == wanted.strip().lower()


def matches_metadata(metadata, selectors: dict[str, list[str]] | None) -> bool:
    """Return whether one image's metadata satisfies every filter.

    Filters combine with AND across properties and OR within one, so
    ``--voltage 2 --voltage 2.5 --microscope Arctis`` means "either voltage, on
    that microscope".

    An image whose header could not be read matches nothing once a metadata
    filter is in play. That is deliberate: including it would put images of
    unknown provenance into a set defined by provenance.
    """
    if not selectors:
        return True
    if metadata is None:
        return False
    for name, wanted in selectors.items():
        attribute = METADATA_FILTERS[name]
        actual = getattr(metadata, attribute, None)
        if not any(_matches_value(actual, value) for value in wanted):
            return False
    return True


def filter_by_metadata(
    records: list,
    metadata: dict,
    selectors: dict[str, list[str]] | None,
) -> list:
    """Return the records whose metadata satisfies *selectors*.

    Args:
        records: Records already narrowed by the structural selectors.
        metadata: Image path to :class:`~am_segmentation_manager.image_metadata.ImageMetadata`.
        selectors: Filter name to accepted values, from
            :data:`METADATA_FILTERS`.
    """
    if not selectors:
        return records
    kept = [
        record for record in records
        if matches_metadata(metadata.get(record.image_path), selectors)
    ]
    if len(kept) < len(records):
        logger.info(
            "Metadata filters kept %d of %d image(s)", len(kept), len(records)
        )
    return kept
