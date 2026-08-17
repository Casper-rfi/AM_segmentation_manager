"""The export view model: what is selected, and what it would export.

Choosing a training set is iterative -- apply a filter, look at what survives,
adjust -- which is why this moved into the window at all. That loop only works
if the count is cheap to refresh, so the controller holds the scanned records
and re-counts in memory rather than re-scanning.

Qt-free, like :mod:`am_segmentation_manager.gui.controller`. Everything here is
decisions; the widgets render them.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..aggregate import BREAKDOWNS, NEEDS_METADATA, count_by, project_lookup, resolve_breakdown
from ..filtering import METADATA_FILTERS, filter_by_metadata, matches_state
from ..image_metadata import ImageMetadata, read_records_metadata
from ..models import ImageRecord, Project
from ..operations.csv import (
    find_matching_label,
    keep_one_condition_per_scene,
    split_pairs,
)
from ..operations.label_io import is_label_file_empty

logger = logging.getLogger(__name__)

#: Workflow states offered as a filter, plus "any".
STATES = ("all", "selected", "deselected", "corrected")

#: Modalities offered as a filter, plus "any".
MODALITIES = ("all", "SEM", "FIB")

#: What the two training files are called when the user names nothing. The
#: command line's default, so the same selection exported either way lands in
#: the same filenames.
DEFAULT_PREFIX = "dataset"


@dataclass(frozen=True)
class ExportPreview:
    """What the current selection would export.

    The two exclusions are the point of showing this at all: both
    ``keep_one_condition_per_scene`` and the empty-label rule drop images
    silently on the command line, reported only in the log. Seeing them before
    exporting is what makes the composition of a training set visible.

    Attributes:
        matched: Images passing the structural and metadata filters.
        corrected: How many of those are corrected.
        annotated: How many have a label holding an annotation.
        empty_labels: Images excluded because their label is a blank placeholder.
        missing_labels: Images excluded because they have no label at all.
        after_conditions: How many remain once one condition per scene is kept.
        exportable: What a training export would actually write.
        train: How many of those the split would put in the training set.
        validation: How many it would put in the validation set. The two are
            whole label groups, so the achieved ratio is only as close to the
            requested one as the grouping allows -- which is the reason to show
            it before writing rather than after.
    """

    matched: int = 0
    corrected: int = 0
    annotated: int = 0
    empty_labels: int = 0
    missing_labels: int = 0
    after_conditions: int = 0
    exportable: int = 0
    train: int = 0
    validation: int = 0

    def summary(self) -> str:
        """Return the one-line description shown above the export buttons."""
        if not self.matched:
            return "Nothing matches these filters"
        parts = [f"{self.matched} image(s) match", f"{self.corrected} corrected"]
        if self.missing_labels:
            parts.append(f"{self.missing_labels} without a label")
        if self.empty_labels:
            parts.append(f"{self.empty_labels} with an empty label")
        if self.after_conditions != self.matched:
            parts.append(f"{self.after_conditions} after one condition per scene")
        parts.append(f"{self.exportable} exportable")
        if self.exportable:
            parts.append(f"{self.train} train / {self.validation} validation")
        return " · ".join(parts)


@dataclass
class ExportFilters:
    """What the user has chosen to keep.

    Structural filters are cheap. Metadata filters need headers, so the
    controller reads them once and reuses them for the rest of the session.
    """

    modality: str = "all"
    state: str = "all"
    metadata: dict[str, list[str]] = field(default_factory=dict)
    conditions: str = "primary"
    include_empty_labels: bool = False


@dataclass
class TrainingExport:
    """How a training export is divided and named.

    Separate from :class:`ExportFilters` because none of it decides which images
    are exported, only how they are split and what the files are called. The
    defaults are the command line's, so the window and ``create-training-csv``
    produce the same two files from the same selection.

    Attributes:
        train_ratio: Share of pairs assigned to training. ``1.0`` writes no
            validation file.
        seed: Seed for a reproducible shuffle, or None to leave the order alone.
        prefix: Filename stem, written as ``<prefix>_train.csv`` and
            ``<prefix>_val.csv``.
    """

    train_ratio: float = 0.8
    seed: int | None = 42
    prefix: str = DEFAULT_PREFIX


class ExportController:
    """Holds a scan, applies filters to it, and reports what would be exported."""

    def __init__(self, start_path: Path | None = None) -> None:
        self.start_path = start_path
        self.filters = ExportFilters()
        self.training = TrainingExport()
        self._projects: list[Project] = []
        self._records: list[ImageRecord] = []
        self._metadata: dict[Path, ImageMetadata] = {}
        # Tracked separately from the mapping being non-empty: a collection
        # whose headers are all unreadable would otherwise be re-read on every
        # filter change, since "loaded but found nothing" and "not loaded yet"
        # look identical from the mapping alone.
        self._metadata_loaded = False
        self._label_empty: dict[Path, bool | None] = {}

    # -- loading -----------------------------------------------------------

    def set_projects(self, projects: Sequence[Project], start_path: Path | None = None) -> None:
        """Take the projects from an existing scan, rather than scanning again.

        The curation window has already scanned; a second scan for the export
        view would double the slowest thing the application does.
        """
        self._projects = list(projects)
        if start_path is not None:
            self.start_path = start_path
        self._records = [
            record
            for project in self._projects
            for lamella in project.lamellae
            for stage in lamella.stages
            for record in stage.images
        ]
        self._metadata = {}
        self._metadata_loaded = False
        self._label_empty = {}
        logger.debug("Export view holding %d record(s)", len(self._records))

    @property
    def records(self) -> list[ImageRecord]:
        return self._records

    @property
    def projects(self) -> list[Project]:
        return self._projects

    def metadata_loaded(self) -> bool:
        """Return whether the headers have been read, however few were found."""
        return self._metadata_loaded

    def load_metadata(self) -> int:
        """Read every record's header, once per scan.

        Returns:
            How many headers yielded metadata. Fewer than the record count
            simply means some files carry none.
        """
        if not self._metadata_loaded:
            self._metadata = read_records_metadata(self._records)
            self._metadata_loaded = True
        return len(self._metadata)

    # -- filtering ---------------------------------------------------------

    def selected_records(self, ignoring: str | None = None) -> list[ImageRecord]:
        """Return the records passing the current filters.

        Args:
            ignoring: A metadata filter to leave out, which is what lets that
                filter offer the values still reachable rather than only the
                ones it already keeps.
        """
        kept = [
            record for record in self._records
            if (self.filters.modality == "all" or record.modality == self.filters.modality)
            and matches_state(record, self.filters.state)
        ]
        metadata_filters = {
            name: values
            for name, values in self.filters.metadata.items()
            if name != ignoring
        }
        if metadata_filters:
            self.load_metadata()
            kept = filter_by_metadata(kept, self._metadata, metadata_filters)
        return kept

    def _is_empty_label(self, path: Path) -> bool | None:
        """Read a label's emptiness once, then remember it.

        The preview refreshes on every filter change, and reading a label costs
        milliseconds; without this the view would stutter on each keystroke.
        """
        if path not in self._label_empty:
            self._label_empty[path] = is_label_file_empty(path)
        return self._label_empty[path]

    def preview(self) -> ExportPreview:
        """Return what the current filters would export."""
        from ..operations.selection import current_state

        matched = self.selected_records()
        if not matched:
            return ExportPreview()

        after = (
            keep_one_condition_per_scene(matched)
            if self.filters.conditions == "primary" else matched
        )

        empty = missing = annotated = 0
        # Built in the order the export builds them, and split by the same
        # function, so the counts shown cannot disagree with the files written.
        pairs: list[tuple[str, str]] = []
        for record in after:
            label = find_matching_label(record)
            if label is None:
                missing += 1
                continue
            is_empty = self._is_empty_label(label)
            if is_empty is True:
                empty += 1
                if self.filters.include_empty_labels:
                    pairs.append((str(record.image_path), str(label)))
                continue
            annotated += 1
            pairs.append((str(record.image_path), str(label)))

        train, validation = split_pairs(
            pairs,
            train_ratio=self.training.train_ratio,
            seed=self.training.seed,
        )

        return ExportPreview(
            matched=len(matched),
            corrected=sum(1 for record in matched if current_state(record) == "corrected"),
            annotated=annotated,
            empty_labels=empty,
            missing_labels=missing,
            after_conditions=len(after),
            exportable=len(pairs),
            train=len(train),
            validation=len(validation),
        )

    # -- breakdowns --------------------------------------------------------

    def available_breakdowns(self) -> list[str]:
        """Return the breakdown names, in a stable order."""
        return sorted(BREAKDOWNS)

    def breakdown(self, name: str) -> dict:
        """Return counts of the current selection by *name*."""
        if name in NEEDS_METADATA:
            self.load_metadata()
        lookup = project_lookup(self._projects)
        key = resolve_breakdown(name, self._metadata, lookup)
        return count_by(self.selected_records(), key)

    def metadata_values(self, name: str) -> list[str]:
        """Return the values of a metadata property the selection still holds.

        Offering what the data actually contains, rather than a free-text box,
        is the difference between a filter you can explore and one you have to
        already know the answer to. The values are those of the records passing
        every *other* filter, so a FIB-only property is not offered while
        looking at SEM images -- and so that narrowing one filter does not empty
        the list of the one you are using.

        Values already chosen for *name* stay offered whatever else excludes
        them, because a filter that vanished from its own list could not be
        seen or cleared.
        """
        if name not in METADATA_FILTERS:
            raise ValueError(f"Unknown metadata filter: {name}")
        self.load_metadata()
        attribute = METADATA_FILTERS[name]
        values = {
            getattr(self._metadata.get(record.image_path), attribute, None)
            for record in self.selected_records(ignoring=name)
        }
        offered = {str(value) for value in values if value is not None}
        return sorted(offered | set(self.filters.metadata.get(name, ())), key=str)

    def set_metadata_filter(self, name: str, values: Sequence[str]) -> None:
        """Set or clear one metadata filter."""
        if name not in METADATA_FILTERS:
            raise ValueError(f"Unknown metadata filter: {name}")
        if values:
            self.filters.metadata[name] = list(values)
        else:
            self.filters.metadata.pop(name, None)

    @property
    def metadata_map(self) -> Mapping[Path, ImageMetadata]:
        return self._metadata
