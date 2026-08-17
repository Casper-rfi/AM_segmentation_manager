"""Counting records by one or two properties.

The shared counting step: it groups records another layer selected, so
statistics did not become a fourth traversal of the project tree.

**Scope.** It counts. It does not filter
(:mod:`am_segmentation_manager.filtering` does), read files, or format output
(:mod:`am_segmentation_manager.reporting` does). A key function returns
:data:`MISSING` rather than ``None`` for an unknown value, so "not recorded" is
a visible row rather than a silent omission.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .config import SegmentationConfig
from .image_metadata import ImageMetadata
from .models import ImageRecord, Project
from .operations.label_measure import (
    EMPTY,
    MEASURED,
    NO_LABEL,
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


def _ranked(order: Sequence[Hashable]) -> Callable[[Hashable], tuple[int, str]]:
    """Return a sort key placing *order*'s values first, in that order.

    Anything not listed sorts after everything listed, by name.
    """
    rank = {value: index for index, value in enumerate(order)}
    return lambda value: (rank.get(value, len(rank)), str(value))


def count_by(
    records: Iterable[ImageRecord],
    key: KeyFunction,
    order: Sequence[Hashable] = (),
) -> dict[Hashable, int]:
    """Return how many records fall into each group.

    Ordered by count descending, then by the group name, so the largest group
    is first and the order is stable between runs.

    Args:
        order: Group names to keep in a declared order instead, largest-first
            being wrong for a scale: bins of increasing area shuffled by
            population no longer read as a distribution.
    """
    counts = Counter(key(record) for record in records)
    if not order:
        return dict(sorted(counts.items(), key=lambda item: (-item[1], str(item[0]))))
    ranked = _ranked(order)
    return dict(sorted(counts.items(), key=lambda item: ranked(item[0])))


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


def totals(
    table: Mapping[tuple[Hashable, Hashable], int],
    row_order: Sequence[Hashable] = (),
    column_order: Sequence[Hashable] = (),
) -> tuple[list, list]:
    """Return the distinct row and column labels of a cross-tabulation.

    Args:
        row_order: Labels to place first, in this order, for an axis that is a
            scale rather than a set of names. The column order likewise.
    """
    rows = sorted({key[0] for key in table}, key=_ranked(row_order))
    columns = sorted({key[1] for key in table}, key=_ranked(column_order))
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
    """Group by accelerating voltage in kV, to the nearest tenth.

    Rounded because a header sometimes records a measured voltage rather than
    the set point: 29.98 kV and 30.0 kV are one beam setting, and left exact
    each becomes its own row of a table about beam settings.
    """
    def key(record: ImageRecord) -> Hashable:
        entry = metadata.get(record.image_path)
        voltage = entry.acceleration_voltage_kv if entry is not None else None
        return _present(None if voltage is None else round(voltage, 1))

    return key


def by_beam_current(metadata: Mapping[Path, ImageMetadata]) -> KeyFunction:
    """Group by beam current in picoamps."""
    return _metadata_key(metadata, "beam_current_pa")


def by_dwell_time(metadata: Mapping[Path, ImageMetadata]) -> KeyFunction:
    """Group by pixel dwell time in nanoseconds."""
    return _metadata_key(metadata, "dwell_time_ns")


def by_pixel_size(metadata: Mapping[Path, ImageMetadata]) -> KeyFunction:
    """Group by pixel size in nanometres."""
    return _metadata_key(metadata, "pixel_size_nm")


def by_plasma_gas(metadata: Mapping[Path, ImageMetadata]) -> KeyFunction:
    """Group by the ion column's plasma gas, e.g. ``Argon``."""
    return _metadata_key(metadata, "plasma_gas")


def by_electron_source(metadata: Mapping[Path, ImageMetadata]) -> KeyFunction:
    """Group by the electron column's emitter, e.g. ``FEG``."""
    return _metadata_key(metadata, "electron_source")


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

def by_label_state(states: Mapping[Path, str]) -> KeyFunction:
    """Group by what the image's label turned out to be.

    Its own kind rather than a measurement: answering it opens each label but
    needs no segmentation config and does no component labelling, so it is
    milliseconds where a measurement is tens of them.
    """
    def key(record: ImageRecord) -> Hashable:
        return _present(states.get(record.image_path))

    return key


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


#: Decade edges for a class-area breakdown, in um^2. Decades rather than bins
#: fitted to one class, because the classes differ by orders of magnitude -- in
#: the reference collection a lamella is hundreds of um^2 and a crack hundredths
#: -- and one scale is what lets the three be read against each other. Fixed
#: rather than derived from the data, so a bin means the same thing after a
#: filter changes.
AREA_BIN_EDGES = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)

#: The class is not in this label at all. Distinct from a small area and from
#: an unmeasurable label: "no GIS was deposited here" is a finding.
ABSENT = "absent"


def area_bins() -> tuple[str, ...]:
    """Return the area bin labels, smallest first."""
    edges = AREA_BIN_EDGES
    labels = [f"< {edges[0]:g}"]
    labels += [f"{low:g} - {high:g}" for low, high in zip(edges, edges[1:])]
    labels.append(f">= {edges[-1]:g}")
    return tuple(labels)


def _area_bin(area: float) -> str:
    """Return which bin *area* falls in."""
    if area <= 0:
        return ABSENT
    for index, edge in enumerate(AREA_BIN_EDGES):
        if area < edge:
            return area_bins()[index]
    return area_bins()[-1]


#: Every row a class-area breakdown can produce, in the order they mean
#: something: absent, then increasing area, then the three ways a label can
#: fail to yield a number at all.
AREA_ORDER: tuple[str, ...] = (
    ABSENT, *area_bins(), EMPTY, NO_PIXEL_SIZE, NO_LABEL,
)


def by_class_area(class_index: int):
    """Return a factory grouping by how much of one class a label holds.

    Binned because a count needs discrete groups and an area is continuous.
    The areas themselves are free once a label is measured -- they come from
    the same pixel counts the emptiness check already needs -- so this costs
    nothing beyond the measurement.
    """

    def factory(measurements: Mapping[Path, LabelMeasurement]) -> KeyFunction:
        def key(record: ImageRecord) -> Hashable:
            measurement = measurements.get(record.image_path)
            status = measurement_status(measurement)
            if status != MEASURED:
                return status
            return _area_bin(measurement.areas_um2.get(class_index, 0.0))

        return key

    return factory


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


@dataclass(frozen=True)
class Breakdown:
    """One way of grouping records, and what to call it.

    Attributes:
        kind: What the key function needs before it can be used --
            ``record`` (nothing), ``metadata``, ``project`` or ``measurement``.
            It is also where the value comes from, which is what
            :data:`GROUP_TITLES` turns into a heading.
        factory: The key function itself for a ``record`` breakdown, or a
            factory taking whatever *kind* names for the rest.
        label: What the breakdown is called where there is room to say it
            properly, carrying the unit the values are counted in.
        order: The rows this produces, in the order they mean something. Empty
            where the groups are names rather than a scale, which is when
            largest-first is the more useful order.
    """

    kind: str
    factory: object
    label: str
    order: tuple[Hashable, ...] = ()


#: Heading for each kind of breakdown, in the order they are offered. The kind
#: says what a value needs to be computed, which is also what it means: a
#: microscope model read from a header and a crack area measured off a label
#: are answers to different sorts of question, and a flat list of names does
#: not say which is which.
GROUP_TITLES: dict[str, str] = {
    "record": "Image and workflow",
    "metadata": "Acquisition, from the image header",
    "project": "Project metadata",
    "label": "Label state",
    "measurement": "Label measurements",
}


#: Breakdowns available by name, for the CLI's ``--by`` and the window's
#: dropdown. The name is the stable identifier a command line takes; the label
#: is what a reader is shown.
BREAKDOWNS: dict[str, Breakdown] = {
    "modality": Breakdown("record", by_modality, "modality (SEM / FIB)"),
    "state": Breakdown("record", by_state, "workflow state"),
    "variant": Breakdown("record", by_variant, "imaging condition"),
    "stage": Breakdown("record", by_stage, "milling stage"),
    "lamella": Breakdown("record", by_lamella, "lamella"),
    "microscope": Breakdown("metadata", by_microscope_model, "microscope model"),
    "serial": Breakdown("metadata", by_microscope_serial, "microscope serial number"),
    "voltage": Breakdown("metadata", by_voltage, "accelerating voltage (kV)"),
    "current": Breakdown("metadata", by_beam_current, "beam current (pA)"),
    "dwell": Breakdown("metadata", by_dwell_time, "dwell time (ns)"),
    "pixel-size": Breakdown("metadata", by_pixel_size, "pixel size (nm)"),
    "plasma-gas": Breakdown("metadata", by_plasma_gas, "plasma gas"),
    "electron-source": Breakdown("metadata", by_electron_source, "electron source"),
    "detector": Breakdown("metadata", by_detector, "detector type"),
    "sample": Breakdown("project", by_sample, "sample"),
    "lamella-type": Breakdown("project", by_lamella_type, "lamella preparation type"),
    "user": Breakdown("project", by_user, "user"),
    "organisation": Breakdown("project", by_organisation, "organisation"),
    "project": Breakdown("project", by_project, "project"),
    "label": Breakdown("label", by_label_state, "label state"),
    "has-crack": Breakdown("measurement", by_has_crack, "cracked (yes / no)"),
    "crack-count": Breakdown("measurement", by_crack_count, "number of cracks"),
}

#: Which breakdowns need per-image headers read. Reading them costs about half
#: a millisecond each, so a command asking only for structural breakdowns
#: should not pay for it.
NEEDS_METADATA = frozenset(
    name for name, entry in BREAKDOWNS.items() if entry.kind == "metadata"
)

#: Which breakdowns need every label read and its components labelled. That is
#: tens of milliseconds per label rather than half a millisecond, so it is
#: worth keeping separate from NEEDS_METADATA: a command should pay for this
#: only when asked a question that needs it. These also need the headers, for
#: the scale.
NEEDS_MEASUREMENTS = frozenset(
    name for name, entry in BREAKDOWNS.items() if entry.kind == "measurement"
)

#: Which breakdowns need each label opened but not measured. Between the two
#: above in cost, and worth its own set for the same reason: a command should
#: pay only for what it was asked.
NEEDS_LABEL_STATES = frozenset(
    name for name, entry in BREAKDOWNS.items() if entry.kind == "label"
)


def _slug(name: str) -> str:
    """Return a class name as a command-line-safe breakdown name."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def breakdowns_for(config: SegmentationConfig) -> dict[str, Breakdown]:
    """Return every breakdown, including one area breakdown per measured class.

    The per-class ones cannot live in :data:`BREAKDOWNS`: which classes exist
    is the segmentation config's to say, and a table fixed in code would either
    hardcode this project's classes or drop a project that renamed them.
    """
    table = dict(BREAKDOWNS)
    for index in config.classes_to_measure:
        name = f"area-{_slug(config.name(index))}"
        if name in table:
            logger.warning(
                "Class %d is called %r, which collides with the breakdown "
                "%r; not offering its area", index, config.name(index), name
            )
            continue
        table[name] = Breakdown(
            "measurement",
            by_class_area(index),
            f"{config.name(index)} area (um^2)",
            order=AREA_ORDER,
        )
    return table


def grouped_breakdowns(
    table: Mapping[str, Breakdown] | None = None
) -> list[tuple[str, list[tuple[str, str]]]]:
    """Return each group's heading and its ``(name, label)`` rows.

    Groups follow :data:`GROUP_TITLES`; rows within one are ordered by label,
    which is what a reader scans. A group with no breakdowns is omitted rather
    than shown empty.
    """
    table = BREAKDOWNS if table is None else table
    grouped = []
    for kind, title in GROUP_TITLES.items():
        rows = sorted(
            (
                (name, entry.label)
                for name, entry in table.items()
                if entry.kind == kind
            ),
            key=lambda row: row[1],
        )
        if rows:
            grouped.append((title, rows))
    return grouped


def resolve_breakdown(
    name: str,
    metadata: Mapping[Path, ImageMetadata] | None = None,
    lookup: Mapping[Path, Project] | None = None,
    measurements: Mapping[Path, LabelMeasurement] | None = None,
    label_states: Mapping[Path, str] | None = None,
    table: Mapping[str, Breakdown] | None = None,
) -> KeyFunction:
    """Return the key function for a breakdown name.

    Args:
        table: The breakdowns to resolve against, from :func:`breakdowns_for`
            where the per-class area breakdowns are wanted. Defaults to the
            config-independent ones.

    Raises:
        ValueError: If *name* is not a known breakdown, listing the ones that
            are.
    """
    table = BREAKDOWNS if table is None else table
    entry = table.get(name)
    if entry is None:
        raise ValueError(
            f"Unknown breakdown {name!r}; choose from {', '.join(sorted(table))}"
        )
    if entry.kind == "record":
        return entry.factory  # type: ignore[return-value]
    if entry.kind == "metadata":
        return entry.factory(metadata or {})  # type: ignore[operator]
    if entry.kind == "measurement":
        return entry.factory(measurements or {})  # type: ignore[operator]
    if entry.kind == "label":
        return entry.factory(label_states or {})  # type: ignore[operator]
    return entry.factory(lookup or {})  # type: ignore[operator]


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
