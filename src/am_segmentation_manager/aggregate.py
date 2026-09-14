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
import math
import re
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from .config import SegmentationConfig
from .image_metadata import ImageMetadata
from .layout import is_alignment_directory, stage_base_name
from .metadata import FIELDS as PROJECT_FIELDS
from .metadata import field_values
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

#: What images sitting directly in a lamella directory are counted under by
#: :func:`by_stage_name`. There is no milling stage to name: the reference
#: images Autolamella takes outside any task live there, and a layout with no
#: stage level puts everything there. Grouping them by the directory would
#: report one row per lamella, which is what the qualified `stage` breakdown is
#: for and what this one exists to avoid.
LAMELLA_ROOT = "(lamella directory)"

#: A function from a record to the group it belongs in -- or, for a breakdown
#: declared ``multi``, to the tuple of every group it belongs in.
KeyFunction = Callable[[ImageRecord], Hashable]


def groups_of(
    key: KeyFunction, record: ImageRecord, *, multi: bool
) -> tuple[Hashable, ...]:
    """Return every group *record* belongs in under *key*.

    The one place either kind of key is read, so counting, cross-tabulating and
    filtering cannot come to different conclusions about what a record is in.
    Whether a key returns one group or several is declared by
    ``Breakdown.multi``, never guessed from what it returned.
    """
    return cast("tuple[Hashable, ...]", key(record)) if multi else (key(record),)


def _present(value: object) -> Hashable:
    """Return *value*, or :data:`MISSING` when there is none."""
    if value is None or value == "":
        return MISSING
    return value  # type: ignore[return-value]


def value_order(value: Hashable) -> tuple[int, float, str]:
    """Return a sort key putting numbers in numeric order, then names, then :data:`MISSING`.

    Takes a group or its text alike, a filter list holding strings: sorted as
    text, a pixel size of 8.431 lands after 19.53 and 10 kV before 2 kV. The one
    rule for every ordering of values, so a list and a table cannot disagree.
    """
    if value == MISSING:
        return (2, 0.0, "")
    if not isinstance(value, bool):
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
        else:
            if math.isfinite(number):
                return (0, number, "")
    return (1, 0.0, str(value))


def _ranked(order: Sequence[Hashable]) -> Callable[[Hashable], tuple]:
    """Return a sort key placing *order*'s values first, in that order.

    Anything not listed sorts after everything listed, by :func:`value_order`.
    """
    rank = {value: index for index, value in enumerate(order)}
    return lambda value: (rank.get(value, len(rank)), value_order(value))


def count_by(
    records: Iterable[ImageRecord],
    key: KeyFunction,
    order: Sequence[Hashable] = (),
    *,
    multi: bool = False,
) -> dict[Hashable, int]:
    """Return how many records fall into each group.

    Ordered by count descending, then by :func:`value_order`, so the largest
    group is first and the order is stable between runs.

    Args:
        order: Group names to keep in a declared order instead, largest-first
            being wrong for a scale: bins of increasing area shuffled by
            population no longer read as a distribution.
        multi: Whether *key* returns every group a record is in rather than
            one, from ``Breakdown.multi``. The counts then sum to more than the
            records counted: an image of a sample that is both infected and
            differentiated is one of each, and the alternative -- a bucket per
            combination -- answers "how many are infected" with nothing.
    """
    counts = Counter(
        group for record in records for group in groups_of(key, record, multi=multi)
    )
    if not order:
        return dict(
            sorted(counts.items(), key=lambda item: (-item[1], value_order(item[0])))
        )
    ranked = _ranked(order)
    return dict(sorted(counts.items(), key=lambda item: ranked(item[0])))


def cross_tabulate(
    records: Iterable[ImageRecord],
    row: KeyFunction,
    column: KeyFunction,
    *,
    row_multi: bool = False,
    column_multi: bool = False,
) -> dict[tuple[Hashable, Hashable], int]:
    """Return counts for each ``(row, column)`` pair.

    This is what "is the training set balanced" usually means: not how many
    Arctis images there are, but how many *corrected FIB* images at 30 kV there
    are on each instrument.

    An axis that holds several values puts a record in every cell it belongs
    to, on the same reasoning as :func:`count_by`: one image of an infected,
    differentiated sample is one infected image and one differentiated one.
    """
    counts = Counter(
        (row_group, column_group)
        for record in records
        for row_group in groups_of(row, record, multi=row_multi)
        for column_group in groups_of(column, record, multi=column_multi)
    )
    return dict(
        sorted(
            counts.items(),
            key=lambda item: (value_order(item[0][0]), value_order(item[0][1])),
        )
    )


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


def by_series(record: ImageRecord) -> Hashable:
    """Group by what the image was taken for.

    Returns the series slug rather than its display label, so that a value
    counted here is a value ``--series`` accepts. The label belongs on the
    breakdown, which is where a reader sees it.
    """
    return record.series


def by_task(record: ImageRecord) -> Hashable:
    """Group by the milling task, with the stages of one task rolled into it.

    A different question from :func:`by_stage_name`, which answers "which
    directory": one task spans several stage directories (``Rough 1``,
    ``Rough 2``), and one lamella directory holds every task's reference
    images at once. Counting by task is what narrows an export to *the
    reference images of the polishing task*.

    An image naming no task counts under :data:`MISSING` -- beam-shift
    alignment names none, and neither does an Adaptive Polishing stage in a
    layout predating ``Milling/<Task>/``.
    """
    return _present(record.task)




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


def _significant(value: float, digits: int) -> float:
    """Return *value* rounded to *digits* significant figures."""
    if not value:
        return 0.0
    return float(f"{value:.{digits}g}")


def _rounded_metadata_key(
    metadata: Mapping[Path, ImageMetadata], attribute: str, digits: int
) -> KeyFunction:
    """Return a key grouping a measured value to *digits* significant figures.

    Some header values are derived rather than set: a pixel size comes out of
    the magnification, so one field of view yields 13.020833333333336 and
    another 13.020833333333332. Left exact, each becomes its own row of a table
    whose question is which magnifications were used. Significant figures
    rather than decimal places, because these span decades -- rounding a
    0.5 nm pixel and a 500 nm one to the same decimal place loses one of them.
    """
    def key(record: ImageRecord) -> Hashable:
        entry = metadata.get(record.image_path)
        value = getattr(entry, attribute, None) if entry is not None else None
        return _present(None if value is None else _significant(value, digits))

    return key


#: Significant figures kept by the breakdowns that group a derived measurement.
#: A grouping figure only, and far coarser than the rounding a value gets when
#: it is read: this one collapses values the instrument really did measure
#: differently, which an export must keep apart.
_GROUPING_DIGITS = 4


def by_beam_current(metadata: Mapping[Path, ImageMetadata]) -> KeyFunction:
    """Group by beam current in picoamps, to four significant figures."""
    return _rounded_metadata_key(metadata, "beam_current_pa", _GROUPING_DIGITS)


def by_dwell_time(metadata: Mapping[Path, ImageMetadata]) -> KeyFunction:
    """Group by pixel dwell time in nanoseconds."""
    return _metadata_key(metadata, "dwell_time_ns")


def by_pixel_size(metadata: Mapping[Path, ImageMetadata]) -> KeyFunction:
    """Group by pixel size in nanometres, to four significant figures."""
    return _rounded_metadata_key(metadata, "pixel_size_nm", _GROUPING_DIGITS)


def by_plasma_gas(metadata: Mapping[Path, ImageMetadata]) -> KeyFunction:
    """Group by the ion column's plasma gas, e.g. ``Argon``."""
    return _metadata_key(metadata, "plasma_gas")


def by_electron_source(metadata: Mapping[Path, ImageMetadata]) -> KeyFunction:
    """Group by the electron column's emitter, e.g. ``FEG``."""
    return _metadata_key(metadata, "electron_source")


def by_detector(metadata: Mapping[Path, ImageMetadata]) -> KeyFunction:
    """Group by detector type."""
    return _metadata_key(metadata, "detector_type")


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
        if status != MEASURED or measurement is None:
            return status
        return "cracked" if measurement.has_crack else "intact"

    return key


#: Powers of ten the area bins span, covering the reference collection from a
#: hundredth of a um^2 to a thousand.
_AREA_DECADES = range(-2, 4)

#: How each decade is subdivided, per "bins per decade" setting. 1-3, 1-2-5 and
#: the E6 series are the standard log-scale subdivisions, and they keep every
#: edge a round number: raising the resolution splits a bin rather than moving
#: its edges, so a count taken at one setting still means something at another.
#:
#: Six is here because a class can span less than one decade and still want a
#: distribution: in the reference collection the corrected lamellae run 58 to
#: 495 um^2 -- 0.93 decades -- so at three per decade they fall in two bins
#: however finely the rest of the scale is cut. Cracks span 3.6 decades and are
#: read perfectly well at one.
_BIN_MANTISSAS: dict[int, tuple[float, ...]] = {
    1: (1.0,),
    2: (1.0, 3.0),
    3: (1.0, 2.0, 5.0),
    6: (1.0, 1.5, 2.0, 3.0, 5.0, 7.0),
}

#: The settings :func:`area_bin_edges` accepts.
AREA_BINS_PER_DECADE: tuple[int, ...] = tuple(sorted(_BIN_MANTISSAS))


def bin_mantissas(per_decade: int) -> tuple[float, ...]:
    """Return the values one decade is cut at, for a *per_decade* setting.

    What the setting actually means, for a control that has to say so: at 3 it
    is 1, 2 and 5, which is the scale a reader recognises rather than a number
    they have to look up.

    Raises:
        ValueError: If *per_decade* is not one of :data:`AREA_BINS_PER_DECADE`.
    """
    if per_decade not in _BIN_MANTISSAS:
        raise ValueError(
            f"Bins per decade must be one of {AREA_BINS_PER_DECADE}, "
            f"got {per_decade!r}"
        )
    return _BIN_MANTISSAS[per_decade]

#: Decades, which is the coarsest setting. The classes differ by orders of
#: magnitude -- a lamella is hundreds of um^2 and a crack hundredths -- so one
#: log scale is what lets them be read against each other, and the useful knob
#: is how finely it is cut rather than where it sits.
DEFAULT_AREA_BINS_PER_DECADE = 1

#: The class is not in this label at all. Distinct from a small area and from
#: an unmeasurable label: "no GIS was deposited here" is a finding.
ABSENT = "absent"


def area_bin_edges(
    per_decade: int = DEFAULT_AREA_BINS_PER_DECADE,
) -> tuple[float, ...]:
    """Return the area bin edges in um^2, smallest first.

    Fixed physical values rather than derived from the data, so a bin means the
    same thing after a filter changes and across the classes being compared.
    *per_decade* chooses how finely the scale is cut, which is a different
    question from where it sits: one class can fill a single decade and still
    want a distribution inside it.

    Raises:
        ValueError: If *per_decade* is not one of :data:`AREA_BINS_PER_DECADE`.
    """
    if per_decade not in _BIN_MANTISSAS:
        raise ValueError(
            f"Bins per decade must be one of {AREA_BINS_PER_DECADE}, "
            f"got {per_decade!r}"
        )
    # Snapped to two significant figures, which every mantissa here has. The
    # product is not exact -- 3.0 * 10**-1 is 0.30000000000000004 -- and an
    # edge a hair above its round value puts an area of exactly 0.3 in the bin
    # below, against the rule that an edge belongs to the bin above it.
    return tuple(
        float(f"{mantissa * (10.0 ** decade):.2g}")
        for decade in _AREA_DECADES
        for mantissa in _BIN_MANTISSAS[per_decade]
    )


def area_bins(edges: Sequence[float]) -> tuple[str, ...]:
    """Return the label of each bin *edges* defines, smallest first."""
    labels = [f"< {edges[0]:g}"]
    labels += [f"{low:g} - {high:g}" for low, high in zip(edges, edges[1:])]
    labels.append(f">= {edges[-1]:g}")
    return tuple(labels)


def area_order(edges: Sequence[float]) -> tuple[str, ...]:
    """Return every row a class-area breakdown can produce, in meaningful order.

    Absent, then increasing area, then the three ways a label can fail to yield
    a number at all.
    """
    return (ABSENT, *area_bins(edges), EMPTY, NO_PIXEL_SIZE, NO_LABEL)


def _area_bin(area: float, edges: Sequence[float], labels: Sequence[str]) -> str:
    """Return which bin *area* falls in."""
    if area <= 0:
        return ABSENT
    for index, edge in enumerate(edges):
        if area < edge:
            return labels[index]
    return labels[-1]


def by_class_area(class_index: int, edges: Sequence[float]):
    """Return a factory grouping by how much of one class a label holds.

    Binned because a count needs discrete groups and an area is continuous.
    The areas themselves are free once a label is measured -- they come from
    the same pixel counts the emptiness check already needs -- so this costs
    nothing beyond the measurement.
    """
    # Built once rather than per record: this key runs over every image in the
    # selection, on every refresh.
    labels = area_bins(edges)

    def factory(measurements: Mapping[Path, LabelMeasurement]) -> KeyFunction:
        def key(record: ImageRecord) -> Hashable:
            measurement = measurements.get(record.image_path)
            status = measurement_status(measurement)
            if status != MEASURED or measurement is None:
                return status
            return _area_bin(
                measurement.areas_um2.get(class_index, 0.0), edges, labels
            )

        return key

    return factory


def by_crack_count(measurements: Mapping[Path, LabelMeasurement]) -> KeyFunction:
    """Group by how many cracks the label holds."""

    def key(record: ImageRecord) -> Hashable:
        measurement = measurements.get(record.image_path)
        status = measurement_status(measurement)
        if status != MEASURED or measurement is None:
            return status
        return measurement.crack_count

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


def _project_metadata_key(
    lookup: Mapping[Path, Project], attribute: str, *, multi: bool = False
) -> KeyFunction:
    def key(record: ImageRecord) -> Hashable:
        project = lookup.get(record.image_path)
        metadata = project.experiment_metadata if project is not None else None
        value = getattr(metadata, attribute) if metadata is not None else None
        # Each value formatted rather than taken as it is: a group is named by
        # its value, and a deposition time of 30 seconds heading a column as
        # "30.0" is one nobody typed. The Export filter matches the same string.
        groups = tuple(_present(text) for text in field_values(value)) or (MISSING,)
        return groups if multi else groups[0]

    return key


def by_project_field(
    name: str, *, multi: bool = False
) -> Callable[[Mapping[Path, Project]], KeyFunction]:
    """Return the factory grouping records by the project metadata field *name*.

    One function for every described field rather than one per field: which
    fields exist is for :data:`~am_segmentation_manager.metadata.FIELDS` to say,
    and a breakdown written out per field is a breakdown a new field does not
    get. The outer call names the field and the inner takes the project lookup,
    which is the shape a ``project`` breakdown is built from.

    Args:
        multi: Whether the field holds several values, in which case the key
            returns every one of them and a record is counted under each.
    """

    def factory(lookup: Mapping[Path, Project]) -> KeyFunction:
        return _project_metadata_key(lookup, name, multi=multi)

    return factory


def _project_of(lookup: Mapping[Path, Project], record: ImageRecord) -> str:
    """Return the identity of *record*'s project, falling back to its name.

    The same fallback :func:`summarise_lamellae` uses: the relative path is the
    identity, and a record whose project is not in the lookup still has to land
    somewhere named rather than under "(not recorded)".
    """
    project = lookup.get(record.image_path)
    if project is not None:
        return str(project.key)
    return str(_present(record.project))


def by_lamella(lookup: Mapping[Path, Project]) -> KeyFunction:
    """Group by lamella, named as ``project / lamella``.

    Qualified by the project because lamella names repeat between projects --
    almost every project has an ``01-``. An unqualified row cannot be found
    again, and two projects' lamellae would be counted as one.
    """
    def key(record: ImageRecord) -> Hashable:
        return f"{_project_of(lookup, record)} / {_present(record.lamella)}"

    return key


def by_stage(lookup: Mapping[Path, Project]) -> KeyFunction:
    """Group by adaptive-milling stage, named as ``project / lamella / stage``.

    Qualified for the same reason as :func:`by_lamella`, and more so: a stage
    name is either shared by every lamella (``adaptive_polish``) or a bare
    timestamp, and neither says which lamella it belongs to.
    """
    def key(record: ImageRecord) -> Hashable:
        return (
            f"{_project_of(lookup, record)} / {_present(record.lamella)}"
            f" / {_present(record.stage)}"
        )

    return key


def _milling_stage_of(record: ImageRecord) -> str:
    """Return the directory naming the milling stage *record* was taken for.

    Its own stage directory, except for an alignment run: Autolamella collects
    those in an ``Alignment/`` directory under the task they align for, and
    names each run after the lamella, the task and the time it ran. That is a
    row per run -- and one starting with the lamella's name -- where what the
    images belong to is the task the ``Alignment/`` directory sits in.
    """
    path = record.stage_path
    if path is not None and is_alignment_directory(path.parent.name):
        return path.parent.parent.name
    return str(_present(record.stage))


def by_stage_name(record: ImageRecord) -> Hashable:
    """Group by what a stage is *called*, with its timestamp taken out.

    Unqualified, unlike :func:`by_stage`, and that is the point: what is left
    of ``adaptive_polish_2026-07-28-02-43-32PM`` is shared by every stage of
    every lamella, and the milling task a set of reference images belongs to
    is named the same way across a project. So this counts and filters in a
    handful of rows where the qualified stage runs to hundreds.

    A record with no milling stage to name counts under :data:`LAMELLA_ROOT`
    rather than putting every lamella in the collection into a list of stages.
    That is a stage directory synthesised from the lamella, which carries the
    lamella's name, and an alignment run directly under a lamella.
    """
    name = _milling_stage_of(record)
    if record.lamella and name == record.lamella:
        return LAMELLA_ROOT
    return stage_base_name(name)


def by_autolamella_project(lookup: Mapping[Path, Project]) -> KeyFunction:
    """Group by the project directory's identity, which is its relative path.

    Distinct from the ``project`` breakdown, which groups by the *wider* project
    a dataset was collected for. One is where the files are and the other is
    what the work was part of, and several directories usually belong to one
    wider project.
    """

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
        factory: The key function itself for a ``record`` breakdown, or a
            factory taking whatever *kind* names for the rest.
        label: What the breakdown is called where there is room to say it
            properly, carrying the unit the values are counted in.
        order: The rows this produces, in the order they mean something. Empty
            where the groups are names rather than a scale, which is when
            largest-first is the more useful order.
        group: Which heading of :data:`GROUP_TITLES` it is listed under, which
            is where the value comes from. Usually the same as *kind* and
            defaults to it; a lamella needs the project lookup to be *named*
            but is not project metadata, which is where the two come apart.
        multi: Whether a record can be in more than one of this breakdown's
            groups, its key returning every group rather than one. The counts
            then sum to more than the records counted, which is a real answer
            to a real question and has to be said out loud wherever they are
            shown. Declared here rather than guessed from what a key returns.
    """

    kind: str
    factory: object
    label: str
    order: tuple[Hashable, ...] = ()
    group: str = ""
    multi: bool = False

    def __post_init__(self) -> None:
        if not self.group:
            object.__setattr__(self, "group", self.kind)


#: One breakdown per project metadata field that can be counted by, keyed and
#: labelled from the field table. Generated rather than written out: adding a
#: field to :data:`~am_segmentation_manager.metadata.FIELDS` gives it a count, a
#: ``--by`` selector and an Export filter row without editing this module, and
#: they are offered in the order the Add data tab asks for them.
PROJECT_BREAKDOWNS: dict[str, Breakdown] = {
    entry.breakdown_name: Breakdown(
        "project",
        by_project_field(entry.name, multi=entry.multi),
        entry.breakdown_label,
        multi=entry.multi,
    )
    for entry in PROJECT_FIELDS
    if entry.breakdown_label
}


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
    "series": Breakdown("record", by_series, "series (adaptive / reference / alignment)"),
    "autolamella-project": Breakdown(
        "project", by_autolamella_project, "Autolamella project", group="record"
    ),
    "stage": Breakdown("project", by_stage, "milling stage", group="record"),
    "stage-name": Breakdown("record", by_stage_name, "milling stage name"),
    "task": Breakdown("record", by_task, "milling task"),
    "lamella": Breakdown("project", by_lamella, "lamella", group="record"),
    "microscope": Breakdown("metadata", by_microscope_model, "microscope model"),
    "serial": Breakdown("metadata", by_microscope_serial, "microscope serial number"),
    "voltage": Breakdown("metadata", by_voltage, "accelerating voltage (kV)"),
    "current": Breakdown("metadata", by_beam_current, "beam current (pA)"),
    "dwell": Breakdown("metadata", by_dwell_time, "dwell time (ns)"),
    "pixel-size": Breakdown("metadata", by_pixel_size, "pixel size (nm)"),
    "plasma-gas": Breakdown("metadata", by_plasma_gas, "plasma gas"),
    "electron-source": Breakdown("metadata", by_electron_source, "electron source"),
    "detector": Breakdown("metadata", by_detector, "detector type"),
    **PROJECT_BREAKDOWNS,
    "label": Breakdown("label", by_label_state, "label state"),
    "has-crack": Breakdown("measurement", by_has_crack, "cracked (yes / no)"),
    "crack-count": Breakdown("measurement", by_crack_count, "number of cracks"),
}

#: Which breakdowns need per-image headers read. Reading them costs under a
#: millisecond each, so a command asking only for structural breakdowns
#: should not pay for it.
NEEDS_METADATA = frozenset(
    name for name, entry in BREAKDOWNS.items() if entry.kind == "metadata"
)

#: Which breakdowns need every label read and its components labelled. That is
#: tens of milliseconds per label rather than under one, so it is
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


def breakdowns_for(
    config: SegmentationConfig,
    area_bins_per_decade: int = DEFAULT_AREA_BINS_PER_DECADE,
) -> dict[str, Breakdown]:
    """Return every breakdown, including one area breakdown per measured class.

    The per-class ones cannot live in :data:`BREAKDOWNS`: which classes exist
    is the segmentation config's to say, and a table fixed in code would either
    hardcode this project's classes or drop a project that renamed them.

    Args:
        area_bins_per_decade: How finely the area scale is cut; see
            :func:`area_bin_edges`. Rebuilding the table is how a caller
            changes it, because the bins are closed over by the key functions.
    """
    edges = area_bin_edges(area_bins_per_decade)
    order = area_order(edges)
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
            by_class_area(index, edges),
            f"{config.name(index)} area (um^2)",
            order=order,
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
                if entry.group == kind
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
