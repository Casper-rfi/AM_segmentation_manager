"""Choosing a subset of an index, by where a record is and by what it is.

The selectors a caller supplies -- names, numbers, ranges like ``2-5`` -- are
answered against projects, lamellae, stages and images, then against what an
image header records. **Structural selectors run first**: they compare strings,
so headers are read only for what survives, and a caller passes in what it read
-- nothing here opens one.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Collection, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from .io.path import slashed
from .layout import SERIES, extract_image_number, normalise_task_name
from .models import ImageRecord, Lamella, Project, Stage
from .operations.selection import current_state
from .scanner import ProjectIndex

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .image_metadata import ImageMetadata

logger = logging.getLogger(__name__)

#: Re-exported: an acquisition number is a naming rule, so it lives in
#: :mod:`.layout` with the rest of them, but the image selectors below are
#: the reason it exists and callers reach for it here.
__all__ = ["extract_image_number"]


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

def matches_project(project: Project, selectors: list[str] | None) -> bool:
    """Return whether *project* matches the user-supplied selectors.

    Matches by relative path, project name, or zero-padded scan-order number.
    The relative path is the project's identity, and is the only selector that
    can distinguish two projects sharing a name; the scan-order number is the
    convenient equivalent. A relative path may separate directories with '\\'.
    """
    numeric_values, name_values = parse_selector_values(selectors)
    if "all" in name_values:
        return True
    if project.name in name_values:
        return True
    if project.relative_path and project.relative_path in {
        slashed(value) for value in name_values
    } | name_values:
        return True
    project_number = parse_scope_number(project.number)
    return project_number is not None and int(project_number) in numeric_values


def matches_lamella(lamella: Lamella, selectors: list[str] | None) -> bool:
    """Return whether *lamella* matches the user-supplied selectors."""
    return _matches_name_or_number(lamella.name, lamella.number, selectors)


def matches_stage(stage: Stage, selectors: list[str] | None) -> bool:
    """Return whether *stage* matches the user-supplied selectors."""
    return _matches_name_or_number(stage.name, stage.number, selectors)


def _matches_name_or_number(
    name: str, number: str | None, selectors: list[str] | None
) -> bool:
    """Return whether a folder named *name*, numbered *number*, matches *selectors*.

    No selector at all matches everything, as does ``"all"``.
    """
    numeric_values, name_values = parse_selector_values(selectors)
    if not numeric_values and not name_values:
        return True
    if "all" in name_values or name in name_values:
        return True
    padded = parse_scope_number(number)
    return padded is not None and int(padded) in numeric_values


def matches_image(record: ImageRecord, selectors: list[str] | None) -> bool:
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


def record_matches_paths(record: ImageRecord, supplied_paths: list[str] | None) -> bool:
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


def matches_family(record: ImageRecord, selectors: list[str] | None) -> bool:
    """Return whether *record* is in one of the requested families.

    Matched whatever the case, as family names are; ``"all"`` or ``None``
    matches everything. An image sitting loose belongs to no family.
    """
    if not selectors:
        return True
    values = {
        value.strip().lower() for raw in selectors for value in raw.split(",") if value.strip()
    }
    if not values or "all" in values:
        return True
    return record.family is not None and record.family.lower() in values


def matches_series(record: ImageRecord, selectors: list[str] | None) -> bool:
    """Return whether *record* is in one of the requested series.

    ``"all"`` selects every series, and ``None`` matches everything so that a
    caller which does not care is unaffected. Unlike a variant, a series is a
    closed vocabulary, so an unknown name is a mistake worth reporting rather
    than a selector that quietly matches nothing.

    Raises:
        ValueError: If a selector names no known series.
    """
    if not selectors:
        return True
    values = {value.strip() for raw in selectors for value in raw.split(",") if value.strip()}
    if not values or "all" in values:
        return True
    unknown = sorted(values - set(SERIES))
    if unknown:
        raise ValueError(
            f"Unknown series: {', '.join(unknown)}; expected one of "
            f"{', '.join(SERIES)} or 'all'"
        )
    return record.series in values


#: Selector meaning "the images whose task nothing records".
NO_TASK = "none"


def matches_task(record: ImageRecord, selectors: list[str] | None) -> bool:
    """Return whether *record* belongs to one of the requested milling tasks.

    Task names are set in the protocol somebody wrote, so this is an open
    vocabulary like a variant rather than a closed one like a series: an
    unrecognised name matches nothing rather than raising, because it may well
    be a task some other project used. Matching is on the normalised name --
    see :func:`~.layout.normalise_task_name` -- so ``--task rough-milling``
    finds ``Rough Milling`` and ``--task millpolishing`` finds
    ``MillPolishing``.

    ``"all"`` selects every task, :data:`NO_TASK` selects the images naming
    none, and ``None`` matches everything so a caller that does not care is
    unaffected.
    """
    if not selectors:
        return True
    values = {value.strip() for raw in selectors for value in raw.split(",") if value.strip()}
    if not values or "all" in values:
        return True
    wanted = {normalise_task_name(value) for value in values}
    if record.task is None:
        return normalise_task_name(NO_TASK) in wanted
    return normalise_task_name(record.task) in wanted


def matches_state(record: ImageRecord, state: str | Collection[str]) -> bool:
    """Return whether *record* matches the requested workflow state.

    An equality test rather than a set of rules: the three states are mutually
    exclusive and a record carries exactly one, so ``"all"`` is the only
    special case. Anything that is not a state matches nothing.

    Args:
        state: One state, ``"all"``, or several states of which any matches.
    """
    if isinstance(state, str):
        return state == "all" or current_state(record) == state
    return "all" in state or current_state(record) in state


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
    state: str | Collection[str] = "all",
    variant_selectors: list[str] | None = None,
    series_selectors: list[str] | None = None,
    task_selectors: list[str] | None = None,
    family_selectors: list[str] | None = None,
) -> Iterator[ImageRecord]:
    """Yield every :class:`ImageRecord` that passes the supplied filters.

    Filters combine with AND: a record must satisfy every one supplied.

    Args:
        index: The scanned project index to walk.
        project_selectors: Match these projects; ``None`` matches all. The
            lamella, stage and image selectors behave the same way.
        paths: Match records at these paths.
        modality: ``"SEM"``, ``"FIB"`` or ``"all"``.
        state: ``"corrected"``, ``"selected"``, ``"deselected"`` or ``"all"``,
            or several of them.
        variant_selectors: Imaging-condition variants, where ``"base"`` is a
            family's unsuffixed image directory.
        series_selectors: What the images were taken for -- ``"adaptive"``,
            ``"reference"``, ``"alignment"`` or ``"all"``. ``None`` matches
            every series, so the caller decides the default rather than
            inheriting one from here.
        task_selectors: Milling task names, where :data:`NO_TASK` selects the
            images naming none.
        family_selectors: Families of imaging conditions, in any case.
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
                    if not matches_series(record, series_selectors):
                        continue
                    if not matches_task(record, task_selectors):
                        continue
                    if modality != "all" and record.modality != modality:
                        continue
                    if not matches_variant(record, variant_selectors):
                        continue
                    if not matches_family(record, family_selectors):
                        continue
                    if not matches_state(record, state):
                        continue
                    if image_selectors is not None and not matches_image(record, image_selectors):
                        continue
                    if record_matches_paths(record, paths):
                        yield record


# ---------------------------------------------------------------------------
# Metadata filters -- applied after the structural selectors above, deliberately
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


def _matches_value(actual: str | float | None, wanted: str) -> bool:
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


def matches_metadata(
    metadata: ImageMetadata | None, selectors: dict[str, list[str]] | None
) -> bool:
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
    records: list[ImageRecord],
    metadata: Mapping[Path, ImageMetadata],
    selectors: dict[str, list[str]] | None,
) -> list[ImageRecord]:
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
