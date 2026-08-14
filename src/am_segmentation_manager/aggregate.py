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
from pathlib import Path

from .image_metadata import ImageMetadata
from .models import ImageRecord, Project
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
    return _present(record.stage)


def by_lamella(record: ImageRecord) -> Hashable:
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
    return _metadata_key(metadata, "beam_current_pa")


def by_dwell_time(metadata: Mapping[Path, ImageMetadata]) -> KeyFunction:
    return _metadata_key(metadata, "dwell_time_ns")


def by_pixel_size(metadata: Mapping[Path, ImageMetadata]) -> KeyFunction:
    return _metadata_key(metadata, "pixel_size_nm")


def by_beam_source(metadata: Mapping[Path, ImageMetadata]) -> KeyFunction:
    """Group by ion source or electron emitter."""
    return _metadata_key(metadata, "beam_source")


def by_detector(metadata: Mapping[Path, ImageMetadata]) -> KeyFunction:
    return _metadata_key(metadata, "detector_type")


def by_header_modality(metadata: Mapping[Path, ImageMetadata]) -> KeyFunction:
    """Group by the beam the *instrument* reported, not the directory."""
    return _metadata_key(metadata, "modality")


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
    return _project_metadata_key(lookup, "user")


def by_organisation(lookup: Mapping[Path, Project]) -> KeyFunction:
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
}

#: Which breakdowns need per-image headers read. Reading them costs about half
#: a millisecond each, so a command asking only for structural breakdowns
#: should not pay for it.
NEEDS_METADATA = frozenset(
    name for name, (kind, _) in BREAKDOWNS.items() if kind == "metadata"
)


def resolve_breakdown(
    name: str,
    metadata: Mapping[Path, ImageMetadata] | None = None,
    lookup: Mapping[Path, Project] | None = None,
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
    return factory(lookup or {})  # type: ignore[operator]
