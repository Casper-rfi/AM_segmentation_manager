"""Counting records by one or two properties.

Three modules already walk the project tree independently --
:func:`~am_segmentation_manager.filtering.iter_matching_records`,
:func:`~am_segmentation_manager.survey.survey_collection` and
:func:`~am_segmentation_manager.reporting.summarize_lamella`. Statistics would
have been a fourth, and the GUI's curation status a fifth. This module is the
shared counting step so that does not happen: it takes records another layer
selected and groups them.

**Scope.** It counts. It does not filter (:mod:`am_segmentation_manager.filtering`
does), does not read files, and does not format output
(:mod:`am_segmentation_manager.reporting` does). If it starts growing a query
language, that is the signal it has outgrown its purpose.

**No dataframe.** :class:`collections.Counter` does all of this in a few lines
with no dependency, and the collections involved are thousands of records, not
millions. pandas was removed from the requirements precisely because nothing
used it; it should return only if something genuinely needs it.

A key function returns :data:`MISSING` rather than ``None`` for an unknown
value, so "not recorded" is a visible row in the output. A large one is a
finding -- it says the archive cannot answer the question being asked of it.
"""
from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .image_metadata import ImageMetadata
from .models import ImageRecord, Project
from .operations.label_measure import (
    EMPTY,
    MEASURED,
    NO_PIXEL_SIZE,
    LabelMeasurement,
    measurement_status,
)
from .operations.selection import current_state

logger = logging.getLogger(__name__)

#: Stands in for a value the data does not record, so it counts as its own
#: group rather than disappearing from the total.
MISSING = "(not recorded)"

#: A function from a record to the group it belongs in.
KeyFunction = Callable[[ImageRecord], Hashable]


def _present(value: object) -> Hashable:
    """Return *value*, or :data:`MISSING` when there is none."""
    if value is None or value == "":
        return MISSING
    return value  # type: ignore[return-value]


def count_by(records: Iterable[ImageRecord], key: KeyFunction) -> dict[Hashable, int]:
    """Return how many records fall into each group.

    Ordered by count descending, then by the group name, so the largest group
    is first and the order is stable between runs.
    """
    counts = Counter(key(record) for record in records)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], str(item[0]))))


def cross_tabulate(
    records: Iterable[ImageRecord], row: KeyFunction, column: KeyFunction
) -> dict[tuple[Hashable, Hashable], int]:
    """Return counts for each ``(row, column)`` pair.

    This is what "is the training set balanced" usually means: not how many
    Arctis images there are, but how many *corrected FIB* images at 30 kV there
    are on each instrument.
    """
    counts = Counter((row(record), column(record)) for record in records)
    return dict(sorted(counts.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))))


def totals(table: Mapping[tuple[Hashable, Hashable], int]) -> tuple[list, list]:
    """Return the distinct row and column labels of a cross-tabulation."""
    rows = sorted({key[0] for key in table}, key=str)
    columns = sorted({key[1] for key in table}, key=str)
    return rows, columns


# ---------------------------------------------------------------------------
# Keys read straight off the record
# ---------------------------------------------------------------------------

def by_modality(record: ImageRecord) -> Hashable:
    """Group by ``SEM``/``FIB``, as the directory organises them."""
    return _present(record.modality)


def by_state(record: ImageRecord) -> Hashable:
    """Group by workflow state: selected, deselected or corrected."""
    return current_state(record)


def by_variant(record: ImageRecord) -> Hashable:
    """Group by imaging condition, with the unsuffixed directory as ``base``."""
    return record.variant or "base"


def by_stage(record: ImageRecord) -> Hashable:
    """Group by adaptive-milling stage name."""
    return _present(record.stage)


def by_lamella(record: ImageRecord) -> Hashable:
    """Group by lamella name, synthetic ones included."""
    return _present(record.lamella)


# ---------------------------------------------------------------------------
# Keys needing per-image metadata
#
# Each closes over the mapping read by image_metadata.read_records_metadata,
# so this module never opens a file and stays testable with plain records.
# ---------------------------------------------------------------------------

def _metadata_key(
    metadata: Mapping[Path, ImageMetadata], attribute: str
) -> KeyFunction:
    def key(record: ImageRecord) -> Hashable:
        entry = metadata.get(record.image_path)
        return _present(getattr(entry, attribute) if entry is not None else None)

    return key


def by_microscope_model(metadata: Mapping[Path, ImageMetadata]) -> KeyFunction:
    """Group by instrument model, e.g. ``Arctis``."""
    return _metadata_key(metadata, "microscope_model")


def by_microscope_serial(metadata: Mapping[Path, ImageMetadata]) -> KeyFunction:
    """Group by instrument serial number.

    Two instruments of one model share a model name and are different
    machines, so this rather than the model is what "fine-tune for one
    microscope" needs.
    """
    return _metadata_key(metadata, "microscope_serial")


def by_voltage(metadata: Mapping[Path, ImageMetadata]) -> KeyFunction:
    """Group by accelerating voltage in volts."""
    return _metadata_key(metadata, "acceleration_voltage_v")


def by_beam_current(metadata: Mapping[Path, ImageMetadata]) -> KeyFunction:
    """Group by beam current in picoamps."""
    return _metadata_key(metadata, "beam_current_pa")


def by_dwell_time(metadata: Mapping[Path, ImageMetadata]) -> KeyFunction:
    """Group by pixel dwell time in nanoseconds."""
    return _metadata_key(metadata, "dwell_time_ns")


def by_pixel_size(metadata: Mapping[Path, ImageMetadata]) -> KeyFunction:
    """Group by pixel size in nanometres."""
    return _metadata_key(metadata, "pixel_size_nm")


def by_beam_source(metadata: Mapping[Path, ImageMetadata]) -> KeyFunction:
    """Group by ion source or electron emitter."""
    return _metadata_key(metadata, "beam_source")


def by_detector(metadata: Mapping[Path, ImageMetadata]) -> KeyFunction:
    """Group by detector type."""
    return _metadata_key(metadata, "detector_type")


def by_header_modality(metadata: Mapping[Path, ImageMetadata]) -> KeyFunction:
    """Group by the beam the *instrument* reported, not the directory."""
    return _metadata_key(metadata, "modality")


# ---------------------------------------------------------------------------
# Keys needing label measurements
#
# Same shape as the metadata keys: they close over a mapping built elsewhere,
# so this module still opens no files. Measuring is far more expensive than
# reading a header, which is why the CLI only does it when one of these is
# asked for.
# ---------------------------------------------------------------------------

def by_has_crack(measurements: Mapping[Path, LabelMeasurement]) -> KeyFunction:
    """Group by whether the label holds a crack at or above the threshold.

    A label that is missing, empty or unmeasurable groups under its status
    rather than under "intact": each is its own finding, and calling an
    unread label intact would understate the cracking.
    """

    def key(record: ImageRecord) -> Hashable:
        measurement = measurements.get(record.image_path)
        status = measurement_status(measurement)
        if status != MEASURED:
            return status
        return "cracked" if measurement.has_crack else "intact"

    return key


def by_crack_count(measurements: Mapping[Path, LabelMeasurement]) -> KeyFunction:
    """Group by how many cracks the label holds."""

    def key(record: ImageRecord) -> Hashable:
        measurement = measurements.get(record.image_path)
        status = measurement_status(measurement)
        return measurement.crack_count if status == MEASURED else status

    return key


# ---------------------------------------------------------------------------
# Keys needing the record's project
#
# ImageRecord holds its project's *name*, and names are not unique -- two
# projects in the reference collection share one. So grouping by a project
# property goes through a lookup keyed by image path, built by walking the
# index once.
# ---------------------------------------------------------------------------

def project_lookup(projects: Iterable[Project]) -> dict[Path, Project]:
    """Map every image path to the project it belongs to."""
    lookup: dict[Path, Project] = {}
    for project in projects:
        for lamella in project.lamellae:
            for stage in lamella.stages:
                for record in stage.images:
                    lookup[record.image_path] = project
    return lookup


def _project_metadata_key(lookup: Mapping[Path, Project], attribute: str) -> KeyFunction:
    def key(record: ImageRecord) -> Hashable:
        project = lookup.get(record.image_path)
        metadata = project.experiment_metadata if project is not None else None
        return _present(getattr(metadata, attribute) if metadata is not None else None)

    return key


def by_sample(lookup: Mapping[Path, Project]) -> KeyFunction:
    """Group by biological sample, from the project's metadata."""
    return _project_metadata_key(lookup, "sample")


def by_lamella_type(lookup: Mapping[Path, Project]) -> KeyFunction:
    """Group by preparation type: cells-on-grid, waffle, lift-out, other."""
    return _project_metadata_key(lookup, "lamella_type")


def by_user(lookup: Mapping[Path, Project]) -> KeyFunction:
    """Group by the user recorded in the project's metadata."""
    return _project_metadata_key(lookup, "user")


def by_organisation(lookup: Mapping[Path, Project]) -> KeyFunction:
    """Group by the organisation recorded in the project's metadata."""
    return _project_metadata_key(lookup, "organisation")


def by_project(lookup: Mapping[Path, Project]) -> KeyFunction:
    """Group by the project's identity, which is its relative path."""

    def key(record: ImageRecord) -> Hashable:
        project = lookup.get(record.image_path)
        return _present(project.key if project is not None else None)

    return key


#: Breakdowns available by name, for the CLI's ``--by``. Values are either a
#: key function or a factory needing metadata (``m``) or a project lookup
#: (``p``); the CLI supplies whichever the chosen breakdown asks for.
BREAKDOWNS: dict[str, tuple[str, object]] = {
    "modality": ("record", by_modality),
    "state": ("record", by_state),
    "variant": ("record", by_variant),
    "stage": ("record", by_stage),
    "lamella": ("record", by_lamella),
    "microscope": ("metadata", by_microscope_model),
    "serial": ("metadata", by_microscope_serial),
    "voltage": ("metadata", by_voltage),
    "current": ("metadata", by_beam_current),
    "dwell": ("metadata", by_dwell_time),
    "pixel-size": ("metadata", by_pixel_size),
    "beam-source": ("metadata", by_beam_source),
    "detector": ("metadata", by_detector),
    "sample": ("project", by_sample),
    "lamella-type": ("project", by_lamella_type),
    "user": ("project", by_user),
    "organisation": ("project", by_organisation),
    "project": ("project", by_project),
    "has-crack": ("measurement", by_has_crack),
    "crack-count": ("measurement", by_crack_count),
}

#: Which breakdowns need per-image headers read. Reading them costs about half
#: a millisecond each, so a command asking only for structural breakdowns
#: should not pay for it.
NEEDS_METADATA = frozenset(
    name for name, (kind, _) in BREAKDOWNS.items() if kind == "metadata"
)

#: Which breakdowns need every label read and its components labelled. That is
#: milliseconds to seconds per label rather than half a millisecond, so it is
#: worth keeping separate from NEEDS_METADATA: a command should pay for this
#: only when asked a question that needs it. These also need the headers, for
#: the scale.
NEEDS_MEASUREMENTS = frozenset(
    name for name, (kind, _) in BREAKDOWNS.items() if kind == "measurement"
)


def resolve_breakdown(
    name: str,
    metadata: Mapping[Path, ImageMetadata] | None = None,
    lookup: Mapping[Path, Project] | None = None,
    measurements: Mapping[Path, LabelMeasurement] | None = None,
) -> KeyFunction:
    """Return the key function for a breakdown name.

    Raises:
        ValueError: If *name* is not a known breakdown, listing the ones that
            are.
    """
    entry = BREAKDOWNS.get(name)
    if entry is None:
        raise ValueError(
            f"Unknown breakdown {name!r}; choose from {', '.join(sorted(BREAKDOWNS))}"
        )
    kind, factory = entry
    if kind == "record":
        return factory  # type: ignore[return-value]
    if kind == "metadata":
        return factory(metadata or {})  # type: ignore[operator]
    if kind == "measurement":
        return factory(measurements or {})  # type: ignore[operator]
    return factory(lookup or {})  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Rolling measurements up to the lamella
#
# "What proportion of lamellae crack" is a question about lamellae, and a crack
# visible in three frames of a milling sequence is one crack. So the roll-up
# deduplicates by label file before counting anything.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LamellaQuantification:
    """What the labels of one lamella hold.

    Attributes:
        project: The project's identity (its relative path where known).
        lamella: The lamella's name.
        labels: Distinct labels found, however they turned out.
        measured: Labels that were annotated and had a known scale. This is the
            denominator of :attr:`cracked_share`.
        empty: Labels holding nothing but background. Excluded from *measured*
            because the segmentation software writes one for every image, so
            they are not evidence that anyone annotated anything.
        unmeasurable: Labels whose image header gave no pixel size, so their
            areas are unknown.
        cracked: Measured labels holding at least one crack over the threshold.
        cracks: Total qualifying components across the measured labels.
        crack_area_um2: Their total area.
        largest_crack_um2: The largest single crack.
        areas_um2: Area per class index, summed over the measured labels.
    """

    project: str
    lamella: str
    labels: int = 0
    measured: int = 0
    empty: int = 0
    unmeasurable: int = 0
    cracked: int = 0
    cracks: int = 0
    crack_area_um2: float = 0.0
    largest_crack_um2: float = 0.0
    areas_um2: dict[int, float] = field(default_factory=dict)

    @property
    def has_crack(self) -> bool:
        """Return whether any of this lamella's labels holds a crack."""
        return self.cracked > 0

    @property
    def cracked_share(self) -> float | None:
        """Return the fraction of measured labels holding a crack.

        ``None`` when nothing could be measured, rather than 0.0: no data and
        no cracks are different answers.
        """
        if not self.measured:
            return None
        return self.cracked / self.measured


def summarise_lamellae(
    records: Iterable[ImageRecord],
    measurements: Mapping[Path, LabelMeasurement],
    lookup: Mapping[Path, Project] | None = None,
) -> list[LamellaQuantification]:
    """Roll per-label measurements up to one row per lamella.

    Args:
        records: The images the measurements were taken for.
        measurements: Image path to measurement, from
            :func:`~am_segmentation_manager.operations.label_measure.measure_records`.
        lookup: Image path to project, so lamellae of two projects sharing a
            name stay apart. Project names are not unique; relative paths are.

    Returns:
        One entry per lamella, ordered by project then lamella.
    """
    lookup = lookup or {}
    grouped: dict[tuple[str, str], list[LabelMeasurement | None]] = {}
    seen: dict[tuple[str, str], set[Path]] = {}

    for record in records:
        project = lookup.get(record.image_path)
        key = (
            str(project.key) if project is not None else str(_present(record.project)),
            str(_present(record.lamella)),
        )
        bucket = grouped.setdefault(key, [])
        measurement = measurements.get(record.image_path)
        if measurement is None:
            # No label at all. Counted once per image, since nothing groups
            # these: without a label there is no shared unit to collapse them
            # onto.
            bucket.append(None)
            continue
        labels_seen = seen.setdefault(key, set())
        if measurement.path in labels_seen:
            continue  # a condition sharing a label already counted it
        labels_seen.add(measurement.path)
        bucket.append(measurement)

    return [
        _summarise_one(project, lamella, bucket)
        for (project, lamella), bucket in sorted(grouped.items())
    ]


def _summarise_one(
    project: str, lamella: str, measurements: list[LabelMeasurement | None]
) -> LamellaQuantification:
    """Total one lamella's deduplicated measurements."""
    areas: dict[int, float] = {}
    labels = measured = empty = unmeasurable = cracked = cracks = 0
    crack_area = largest = 0.0

    for measurement in measurements:
        if measurement is None:
            continue
        labels += 1
        status = measurement_status(measurement)
        if status == EMPTY:
            empty += 1
            continue
        if status == NO_PIXEL_SIZE:
            unmeasurable += 1
            continue
        measured += 1
        for index, area in measurement.areas_um2.items():
            areas[index] = areas.get(index, 0.0) + area
        if measurement.crack_count:
            cracked += 1
            cracks += measurement.crack_count
            crack_area += sum(measurement.crack_areas_um2)
            largest = max(largest, measurement.crack_areas_um2[0])

    return LamellaQuantification(
        project=project,
        lamella=lamella,
        labels=labels,
        measured=measured,
        empty=empty,
        unmeasurable=unmeasurable,
        cracked=cracked,
        cracks=cracks,
        crack_area_um2=crack_area,
        largest_crack_um2=largest,
        areas_um2=areas,
    )
