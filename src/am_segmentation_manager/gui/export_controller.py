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

from ..aggregate import (
    DEFAULT_AREA_BINS_PER_DECADE,
    GROUP_TITLES,
    NEEDS_LABEL_STATES,
    NEEDS_METADATA,
    Breakdown,
    KeyFunction,
    breakdowns_for,
    count_by,
    cross_tabulate,
    grouped_breakdowns,
    project_lookup,
    resolve_breakdown,
)
from ..config import SegmentationConfig, load_segmentation_config
from ..filtering import matches_state
from ..image_metadata import ImageMetadata, read_records_metadata
from ..models import ImageRecord, Project
from ..operations.csv import (
    find_matching_label,
    keep_one_condition_per_scene,
    split_pairs,
)
from ..operations.label_io import (
    ANNOTATED,
    EMPTY,
    MISSING,
    UNREADABLE,
    LabelReadiness,
    classify_labels,
    is_label_file_empty,
    label_state,
)
from ..operations.label_measure import LabelMeasurement

logger = logging.getLogger(__name__)

#: Workflow states offered as a filter, plus "any".
STATES = ("all", "selected", "deselected", "corrected")

#: Modalities offered as a filter, plus "any".
MODALITIES = ("all", "SEM", "FIB")

#: What each label filter keeps, as the states
#: :func:`~am_segmentation_manager.operations.label_io.label_state` returns.
#: Named rather than multi-select, because the useful questions are these five
#: and "select nothing to mean everything" is too quiet a default for a control
#: that decides whether you are looking at 2,000 images or 200.
#:
#: A label this package could not read is kept only by the two options that
#: claim nothing about what is inside a label. Filing it under "empty" would
#: assert a file is blank that nobody managed to open -- the distinction
#: :func:`~am_segmentation_manager.operations.label_io.is_label_file_empty`
#: exists to preserve.
#:
#: "empty" says what the file holds, and nothing about how it came to. The
#: segmentation software's blank and a label somebody proofread and found
#: nothing in are the same bytes; only where the label sits tells them apart,
#: and that is the workflow state filter's question rather than this one's.
LABEL_FILTERS: dict[str, frozenset[str]] = {
    "any": frozenset({ANNOTATED, EMPTY, MISSING, UNREADABLE}),
    "has a label": frozenset({ANNOTATED, EMPTY, UNREADABLE}),
    "annotated": frozenset({ANNOTATED}),
    "empty": frozenset({EMPTY}),
    "no label yet": frozenset({MISSING}),
}

#: Breakdowns with a control of their own in the Filters box, so they are not
#: also offered as a list of values. Modality and workflow state are three
#: fixed choices each rather than whatever the data holds, and the label filter
#: is five named questions rather than four states -- "has a label" is not a
#: value any record carries.
DEDICATED_CONTROLS = frozenset({"modality", "state", "label"})


#: What the label filter starts on: everything. The tab opens on the whole
#: collection and is narrowed towards an export, rather than opening on the
#: export and hiding most of the archive until the filter is found.
ANY_LABEL = "any"

#: What the two training files are called when the user names nothing. The
#: command line's default, so the same selection exported either way lands in
#: the same filenames.
DEFAULT_PREFIX = "dataset"


def _sortable(value: str) -> tuple[int, float, str]:
    """Return a sort key putting numbers in numeric order, names after them.

    The values are strings because that is what a list widget holds, and
    sorting a pixel size as one puts 8.431 after 19.53.
    """
    try:
        return (0, float(value), "")
    except ValueError:
        return (1, 0.0, value)


@dataclass(frozen=True)
class ExportPreview:
    """What the current selection would export.

    The exclusions are the point of showing this at all: the empty-label rule
    drops images silently on the command line, reported only in the log, and
    the label filter can admit images no export could write. Seeing the gap
    before exporting is what makes the composition of a training set visible.

    Attributes:
        matched: Images passing every filter, dedup included.
        corrected: How many of those are corrected.
        deselected: How many of the *exportable* ones were decided against.
            Shown because the workflow-state filter opens on "all", so a
            training set can quietly include images somebody rejected -- the
            empty-label rule catches most of them, but not one that was
            labelled and deselected afterwards.
        annotated: How many have a label holding an annotation.
        empty_labels: Images excluded because their label holds nothing but
            background.
        missing_labels: Images excluded because they have no label at all.
        exportable: What a training export would actually write.
        train: How many of those the split would put in the training set.
        validation: How many it would put in the validation set. The two are
            whole label groups, so the achieved ratio is only as close to the
            requested one as the grouping allows -- which is the reason to show
            it before writing rather than after.
    """

    matched: int = 0
    corrected: int = 0
    deselected: int = 0
    annotated: int = 0
    empty_labels: int = 0
    missing_labels: int = 0
    exportable: int = 0
    train: int = 0
    validation: int = 0

    def summary(self) -> str:
        """Return the one-line description shown above the export buttons."""
        if not self.matched:
            return "Nothing matches these filters"
        parts = [f"{self.matched} image(s) match", f"{self.corrected} corrected"]
        if self.deselected:
            parts.append(f"{self.deselected} deselected")
        if self.missing_labels:
            parts.append(f"{self.missing_labels} without a label")
        if self.empty_labels:
            parts.append(f"{self.empty_labels} with an empty label")
        parts.append(f"{self.exportable} exportable")
        if self.exportable:
            parts.append(f"{self.train} train / {self.validation} validation")
        return " · ".join(parts)


@dataclass(frozen=True)
class Sifted:
    """The exportable images, and how many each rule dropped.

    One traversal feeding both the summary line and the breakdown below it,
    because two counts on one screen that disagree are worse than either alone.

    Attributes:
        pairs: Each exportable image with the label it is paired with.
        annotated: How many of those have a label holding an annotation.
        empty_labels: Images whose label holds nothing but background.
        missing_labels: Images with no label at all.
    """

    pairs: list[tuple[ImageRecord, Path]]
    annotated: int = 0
    empty_labels: int = 0
    missing_labels: int = 0


@dataclass
class ExportFilters:
    """What the user has chosen to keep.

    Everything that decides *which* images are in play, so that one place
    answers it: the count, the breakdown and the export all read this, and a
    control that narrowed the export without narrowing the table would be a
    tab showing two different collections at once.

    Structural filters are cheap. Metadata filters need headers, and the label
    filter needs each label opened, so the controller does both once and
    reuses them for the rest of the session.

    Attributes:
        choices: Breakdown name to the values kept, for every filter that
            offers the values the data actually holds. A filter and a
            breakdown are the same key function used two ways, so there is one
            table of them rather than two that can disagree.
        conditions: ``"all"`` keeps every imaging condition of a scene,
            ``"primary"`` keeps one image per label. Starts at ``"all"``,
            unlike ``create-training-csv``: the tab opens on the whole
            collection and is narrowed towards an export.
        label: A key of :data:`LABEL_FILTERS`.
    """

    modality: str = "all"
    state: str = "all"
    choices: dict[str, list[str]] = field(default_factory=dict)
    conditions: str = "all"
    label: str = ANY_LABEL


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

    def __init__(
        self, start_path: Path | None = None, config: SegmentationConfig | None = None
    ) -> None:
        self.start_path = start_path
        # Which class is a crack, and how small a component still counts as one.
        # Loaded here when nobody supplies it so the controller stands alone in
        # a test; the window hands over the one the overlays already use.
        self.config = config if config is not None else load_segmentation_config()
        # Resolved once: which area breakdowns exist is the config's to say,
        # and rebuilding the table per refresh would rebuild the closures
        # the breakdowns are made of.
        self._area_bins_per_decade = DEFAULT_AREA_BINS_PER_DECADE
        self.breakdowns: dict[str, Breakdown] = breakdowns_for(
            self.config, self._area_bins_per_decade
        )
        # Starts at the config's, which is also what quantify-labels applies
        # when nothing overrides it.
        self._min_crack_area_um2 = self.config.min_crack_area_um2
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
        self._label_states: dict[Path, str] = {}
        self._measurements: dict[Path, LabelMeasurement] = {}
        self._measurements_loaded = False
        # Built by walking every record, and asked for once per filter list on
        # every refresh. Rebuilt only when the scan changes.
        self._lookup: dict[Path, Project] = {}

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
            record for project in self._projects for record in project.image_records
        ]
        self._metadata = {}
        self._metadata_loaded = False
        self._label_empty = {}
        self._label_states = {}
        self._measurements = {}
        self._measurements_loaded = False
        self._lookup = project_lookup(self._projects)
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

    def measurements_loaded(self) -> bool:
        """Return whether the labels have been measured."""
        return self._measurements_loaded

    def needs_measurements(self, name: str) -> bool:
        """Return whether *name* cannot be counted until the labels are measured.

        Asked before counting rather than measuring on demand: a breakdown
        that quietly opened every label would fire on each press of an arrow
        key stepping down the list.
        """
        entry = self.breakdowns.get(name)
        return entry is not None and entry.kind == "measurement"

    def measurable_labels(self) -> int:
        """Return how many distinct labels a measurement would open.

        What the button says before it is pressed. The label is the unit, not
        the image, so several imaging conditions of one scene cost one read
        between them.
        """
        labels = {
            found
            for record in self._records
            if (found := find_matching_label(record)) is not None
        }
        return len(labels)

    @property
    def min_crack_area_um2(self) -> float:
        """Return the smallest component currently counted as a crack, in um^2."""
        return self._min_crack_area_um2

    def set_crack_threshold(self, value: float) -> None:
        """Set the smallest component counted as a crack, in um^2.

        Re-measures immediately when the labels have already been read, rather
        than spending the button again: the threshold is applied to component
        areas that :func:`read_label_pixels` has already cached *without* it, so
        a new one costs arithmetic rather than a second pass over the archive.

        Raises:
            ValueError: If *value* is negative.
        """
        if value < 0:
            raise ValueError(f"A crack threshold cannot be negative: {value!r}")
        if value == self._min_crack_area_um2:
            return
        self._min_crack_area_um2 = float(value)
        if self._measurements_loaded:
            self._measurements_loaded = False
            self.load_measurements()

    @property
    def area_bins_per_decade(self) -> int:
        """Return how finely the class-area scale is currently cut."""
        return self._area_bins_per_decade

    def set_area_bins(self, per_decade: int) -> None:
        """Set how finely the class-area breakdowns cut each decade.

        Rebuilds the breakdown table, because the bins are closed over by the
        key functions. The names are unchanged, so a dropdown holding them
        needs no rebuilding. Nothing is re-measured: the areas do not depend on
        how they are grouped.

        Raises:
            ValueError: If *per_decade* is not a supported setting.
        """
        if per_decade == self._area_bins_per_decade:
            return
        self.breakdowns = breakdowns_for(self.config, per_decade)
        self._area_bins_per_decade = per_decade

    def load_measurements(self) -> int:
        """Measure every label, once per scan.

        Tens of milliseconds per label against half a millisecond for a header,
        which is why nothing calls this implicitly: it is the one thing in this
        tab that has to be asked for.

        The headers are read first and kept. Measuring needs them anyway, for
        the scale; reading them into the mapping rather than letting
        ``measure_records`` read and discard its own is what makes a later
        change of threshold free.

        Returns:
            How many images have a measured label.
        """
        if not self._measurements_loaded:
            from ..operations.label_measure import measure_records

            self.load_metadata()
            self._measurements = measure_records(
                self._records,
                self.config,
                self._metadata,
                min_crack_area_um2=self._min_crack_area_um2,
            )
            self._measurements_loaded = True
        return len(self._measurements)

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

    def label_state(self, record: ImageRecord) -> str:
        """Return *record*'s label state, reading each label at most once.

        The filter and the breakdown both ask this of every record on every
        refresh, and answering it opens a file.
        """
        path = record.image_path
        if path not in self._label_states:
            self._label_states[path] = label_state(record)
        return self._label_states[path]

    def selected_records(self, ignoring: str | None = None) -> list[ImageRecord]:
        """Return the records passing the current filters.

        The dedup rule is applied here with the rest, not left to the export:
        it decides which images are in play, so the table has to see it.

        Args:
            ignoring: A metadata filter to leave out, which is what lets that
                filter offer the values still reachable rather than only the
                ones it already keeps.
        """
        if self.filters.label not in LABEL_FILTERS:
            raise ValueError(f"Unknown label filter: {self.filters.label!r}")
        # None rather than the full set, so the default filter never opens a
        # label to decide it is keeping it anyway.
        wanted = (
            None if self.filters.label == ANY_LABEL
            else LABEL_FILTERS[self.filters.label]
        )
        kept = [
            record for record in self._records
            if (self.filters.modality == "all" or record.modality == self.filters.modality)
            and matches_state(record, self.filters.state)
            and (wanted is None or self.label_state(record) in wanted)
        ]
        # Each remaining filter keeps the records whose group is one of the
        # chosen ones -- the breakdown's own key function, read as a predicate.
        # Compared as the string the list offered, so what was picked and what
        # is matched cannot drift: a pixel size is grouped to four significant
        # figures, and the filter has to agree with the table about that.
        chosen = {
            name: set(values)
            for name, values in self.filters.choices.items()
            if name != ignoring and values
        }
        if chosen:
            for name, key in zip(chosen, self._keys(list(chosen))):
                allowed = chosen[name]
                kept = [record for record in kept if str(key(record)) in allowed]
        if self.filters.conditions == "primary":
            kept = keep_one_condition_per_scene(kept)
        return kept

    def _is_empty_label(self, path: Path) -> bool | None:
        """Read a label's emptiness once, then remember it.

        The preview refreshes on every filter change, and reading a label costs
        milliseconds; without this the view would stutter on each keystroke.
        """
        if path not in self._label_empty:
            self._label_empty[path] = is_label_file_empty(path)
        return self._label_empty[path]

    def _sift(self, matched: Sequence[ImageRecord]) -> Sifted:
        """Separate the images that can be exported from those that cannot.

        The filters have already run, dedup included. What is left is the rule
        no filter can express: a training pair needs a label with something in
        it, so an image whose label is missing or blank is reported here rather
        than written.
        """
        empty = missing = annotated = 0
        pairs: list[tuple[ImageRecord, Path]] = []
        for record in matched:
            label = find_matching_label(record)
            if label is None:
                missing += 1
                continue
            if self._is_empty_label(label) is True:
                empty += 1
                continue
            annotated += 1
            pairs.append((record, label))

        return Sifted(
            pairs=pairs,
            annotated=annotated,
            empty_labels=empty,
            missing_labels=missing,
        )

    def exportable_records(self) -> list[ImageRecord]:
        """Return the images a training export would actually write.

        Narrower than what the filters match whenever the label filter admits
        an image with no label or a blank one: those are countable, and looking
        at them is the point of the filter, but they cannot be exported.
        """
        return [record for record, _ in self._sift(self.selected_records()).pairs]

    def unexportable_corrected(self) -> LabelReadiness:
        """Return the corrected images the filters match that cannot be exported.

        The one case worth stopping for. An unlabelled image in any other state
        is the ordinary condition of an archive being worked through, and a
        dialog about it would fire on every export until it was ignored;
        corrected is the claim that the label holds proofread work, so an
        export dropping one means either the proofreading is missing or the
        label is.
        """
        from ..operations.selection import current_state

        corrected = [
            record for record in self.selected_records()
            if current_state(record) == "corrected"
        ]
        return classify_labels(corrected, self.label_state)

    def preview(self) -> ExportPreview:
        """Return what the current filters would export."""
        from ..operations.selection import current_state

        matched = self.selected_records()
        if not matched:
            return ExportPreview()

        sifted = self._sift(matched)
        train, validation = split_pairs(
            [(str(record.image_path), str(label)) for record, label in sifted.pairs],
            train_ratio=self.training.train_ratio,
            seed=self.training.seed,
        )

        return ExportPreview(
            matched=len(matched),
            corrected=sum(1 for record in matched if current_state(record) == "corrected"),
            deselected=sum(
                1 for record, _ in sifted.pairs
                if current_state(record) == "deselected"
            ),
            annotated=sifted.annotated,
            empty_labels=sifted.empty_labels,
            missing_labels=sifted.missing_labels,
            exportable=len(sifted.pairs),
            train=len(train),
            validation=len(validation),
        )

    # -- breakdowns --------------------------------------------------------

    def available_breakdowns(self) -> list[tuple[str, list[tuple[str, str]]]]:
        """Return the breakdowns as headed groups, for the dropdown.

        Grouped by where the value comes from, because that is what tells a
        reader whether a row was measured off a label, read from a header or
        typed into the project's metadata.
        """
        return grouped_breakdowns(self.breakdowns)

    def _keys(self, names: Sequence[str]) -> list[KeyFunction]:
        """Return the key function for each name, reading headers once.

        Resolved together rather than one at a time: the project lookup walks
        the whole index to build itself, and a cross-tabulation needs two keys
        from the one walk.
        """
        if any(name in NEEDS_METADATA for name in names):
            self.load_metadata()
        states: dict[Path, str] = {}
        if any(name in NEEDS_LABEL_STATES for name in names):
            states = {
                record.image_path: self.label_state(record)
                for record in self._records
            }
        return [
            resolve_breakdown(
                name, self._metadata, self._lookup, self._measurements, states,
                self.breakdowns,
            )
            for name in names
        ]

    def breakdown(self, name: str) -> dict:
        """Return counts of the filtered images, grouped by *name*.

        Counted over everything the filters match, so the tab answers "what is
        in this collection" as well as "what would I export" -- which of the
        two you get is the label filter's to decide, not this method's. Every
        control that narrows is a filter, so the table still moves whenever any
        of them does.
        """
        (key,) = self._keys([name])
        return count_by(
            self.selected_records(), key, self.breakdowns[name].order
        )

    def cross_tabulation(self, row: str, column: str) -> dict:
        """Return counts of what would be exported by two properties at once.

        The question a training set actually raises: not how many Arctis
        images there are, but how many of them are corrected FIB at 30 kV --
        and, more usefully, which combinations have nothing in them at all.
        """
        row_key, column_key = self._keys([row, column])
        return cross_tabulate(self.selected_records(), row_key, column_key)

    def filterable_breakdowns(self) -> list[tuple[str, str]]:
        """Return the ``(name, label)`` of every breakdown offered as a filter.

        A filter and a breakdown are the same key function used two ways, so
        the list of one is derived from the other rather than declared beside
        it. Ordered as the breakdown dropdown is, so a property is found in
        the same place in both.

        Measurement breakdowns are left out: they cannot answer until the
        labels have been measured, and a filter that quietly keeps nothing
        until a button elsewhere is pressed is worse than no filter.
        """
        headings = list(GROUP_TITLES)
        rows = [
            (name, entry)
            for name, entry in self.breakdowns.items()
            if entry.kind != "measurement" and name not in DEDICATED_CONTROLS
        ]
        rows.sort(key=lambda row: (headings.index(row[1].group), row[1].label))
        return [(name, entry.label) for name, entry in rows]

    def needs_metadata(self, name: str) -> bool:
        """Return whether *name* cannot be offered until the headers are read."""
        entry = self.breakdowns.get(name)
        return entry is not None and entry.kind == "metadata"

    def choice_values(self, name: str) -> list[str]:
        """Return the values of a property the rest of the selection still holds.

        Offering what the data actually contains, rather than a free-text box,
        is the difference between a filter you can explore and one you have to
        already know the answer to. The values are those of the records passing
        every *other* filter, so a FIB-only property is not offered while
        looking at SEM images -- and so that narrowing one filter does not empty
        the list of the one you are using.

        Values already chosen for *name* stay offered whatever else excludes
        them, because a filter that vanished from its own list could not be
        seen or cleared.

        Raises:
            ValueError: If *name* is not a breakdown offered as a filter.
        """
        self._require_filter(name)
        (key,) = self._keys([name])
        values = {
            str(key(record)) for record in self.selected_records(ignoring=name)
        }
        return sorted(
            values | set(self.filters.choices.get(name, ())), key=_sortable
        )

    def set_choice_filter(self, name: str, values: Sequence[str]) -> None:
        """Set or clear one filter.

        Raises:
            ValueError: If *name* is not a breakdown offered as a filter.
        """
        self._require_filter(name)
        if values:
            self.filters.choices[name] = list(values)
        else:
            self.filters.choices.pop(name, None)

    def _require_filter(self, name: str) -> None:
        """Raise unless *name* is a breakdown that can be filtered on."""
        if name not in self.breakdowns or name in DEDICATED_CONTROLS:
            raise ValueError(f"Unknown filter: {name}")
        if self.breakdowns[name].kind == "measurement":
            raise ValueError(f"Cannot filter on a measured property: {name}")

    @property
    def metadata_map(self) -> Mapping[Path, ImageMetadata]:
        return self._metadata
