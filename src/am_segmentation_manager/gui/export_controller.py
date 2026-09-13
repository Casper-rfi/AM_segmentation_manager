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
from collections.abc import Callable, Iterable, Mapping, Sequence
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
    groups_of,
    project_lookup,
    resolve_breakdown,
    value_order,
)
from ..config import SegmentationConfig, load_segmentation_config
from ..filtering import matches_state
from ..image_metadata import ImageMetadata, read_records_metadata
from ..models import ImageRecord, Project
from ..operations.csv import (
    DEFAULT_TRAIN_RATIO,
    build_training_rows,
    export_training_csv,
    find_matching_label,
    keep_one_condition_per_scene,
    split_rows,
)
from ..operations.label_io import (
    ANNOTATED,
    EMPTY,
    MISSING,
    UNREADABLE,
    LabelReadiness,
    classify_label,
    classify_labels,
    is_label_file_empty,
)
from ..operations.label_measure import LabelMeasurement, measure_records, measurement_is_empty
from ..operations.metadata_csv import export_image_metadata_csv
from ..operations.selection import current_state
from ..reporting import counts_as_rows, cross_tabulation_as_rows, write_table_csv

logger = logging.getLogger(__name__)

#: Tells "never read" apart from a cached ``None``, which is what an
#: unreadable label records.
_ABSENT = object()

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

#: The one filter whose values are the projects themselves. Named so the
#: window can find its dedicated control's values without knowing the string,
#: and so `PROJECT_FILTER not in DEDICATED_CONTROLS` cannot silently become
#: true and leave the tab with two controls on one filter.
PROJECT_FILTER = "autolamella-project"

#: Breakdowns with a control of their own in the Filters box, so they are not
#: also offered as a list of values. Modality, workflow state and series are
#: fixed sets of choices rather than whatever the data holds; the label filter
#: is five named questions rather than four states -- "has a label" is not a
#: value any record carries -- and the projects are a tree that every tab
#: shows, not a row somebody has to go and add.
DEDICATED_CONTROLS = frozenset({"modality", "state", "label", "series", PROJECT_FILTER})

#: Of those, the ones that are not choice filters underneath. Each has a field
#: of its own on :class:`Filters`, so a value written into ``choices`` for one
#: would be stored and never read -- which is what :meth:`_require_filter`
#: refuses. The project filter is a dedicated *control* over an ordinary
#: choice filter, so it is deliberately not here: the two ideas were one set
#: until something needed to be both.
FIXED_CONTROLS = frozenset({"modality", "state", "label", "series"})

#: What the series control starts on where the settings name nothing -- see
#: `gui/preferences.py`. **The one filter that does not open on everything**,
#: and deliberately: a scan holds reference and alignment images
#: as well, and an export that silently grew to include them would disagree
#: with ``create-training-csv``, which defaults the same way. Ticking another
#: is one click, and the summary line says what is counted.
DEFAULT_SERIES: tuple[str, ...] = ("adaptive",)

#: Breakdowns counted on, but never offered as a filter in the window. Each
#: would produce a list nobody could work with: a lamella is one row per
#: lamella and a milling stage one per timestamped folder, both running to
#: hundreds, and an electron source does not vary across this collection at
#: all. They stay breakdowns, and the command line still filters on
#: ``--electron-source``; what is dropped is the value list, the window's
#: scarcest space being the column those lists sit in.
UNFILTERABLE = frozenset({"lamella", "stage", "electron-source"})

#: What the label filter starts on: everything. The tab opens on the whole
#: collection and is narrowed towards an export, rather than opening on the
#: export and hiding most of the archive until the filter is found.
ANY_LABEL = "any"

#: What the two training files are called when the user names nothing. The
#: command line's default, so the same selection exported either way lands in
#: the same filenames.
DEFAULT_PREFIX = "dataset"


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
        exportable: What a training export would actually write -- images
            where one modality is chosen, rows where both are.
        train: How many of those the split would put in the training set.
        validation: How many it would put in the validation set. The two are
            whole groups, so the achieved ratio is only as close to the
            requested one as the grouping allows -- which is the reason to show
            it before writing rather than after.
        paired: Whether a row carries both beams, which both modalities being
            in play is what decides.
        sem_only: Paired rows with no FIB label to go beside the SEM one.
        fib_only: Paired rows with no SEM label. Shown because a paired export
            is chosen in order to have both, and how much of it is actually
            paired is not otherwise visible until the file is open.
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
    paired: bool = False
    sem_only: int = 0
    fib_only: int = 0
    measured: bool = True

    def summary(self) -> str:
        """Return the one-line description shown above the export buttons."""
        if not self.matched:
            return "Nothing matches these filters"
        parts = [f"{self.matched} image(s) match", f"{self.corrected} corrected"]
        if not self.measured:
            # Everything below this line is a question about what is inside a
            # label, which is a read per label and the one thing in this tab
            # slow enough to be asked for rather than assumed.
            parts.append("press Measure labels to count what can be exported")
            return " · ".join(parts)
        if self.deselected:
            parts.append(f"{self.deselected} deselected")
        if self.missing_labels:
            parts.append(f"{self.missing_labels} without a label")
        if self.empty_labels:
            parts.append(f"{self.empty_labels} with an empty label")
        if self.paired:
            parts.append(f"{self.exportable} paired row(s)")
            if self.sem_only:
                parts.append(f"{self.sem_only} SEM only")
            if self.fib_only:
                parts.append(f"{self.fib_only} FIB only")
        else:
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
        series: What the images were taken for, as a set of ticked values.
            Starts at :data:`DEFAULT_SERIES` rather than at everything, which
            is the one place this tab does not open on the whole collection.
        conditions: ``"all"`` keeps every imaging condition of a scene,
            ``"primary"`` keeps one image per label. Starts at ``"all"``,
            unlike ``create-training-csv``: the tab opens on the whole
            collection and is narrowed towards an export.
        label: A key of :data:`LABEL_FILTERS`.
    """

    modality: str = "all"
    state: str = "all"
    series: tuple[str, ...] = DEFAULT_SERIES
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

    train_ratio: float = DEFAULT_TRAIN_RATIO
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
        #: Whether each label holds anything, **keyed by the label** -- one
        #: entry per file, conditions of a scene sharing it.
        self._label_empty: dict[Path, bool | None] = {}
        #: What each image's label turned out to be, **keyed by the image**,
        #: as the two below are: that is what lets :mod:`..aggregate` close
        #: over them the way it closes over the header mapping. Two key spaces
        #: for one subject, so anything dropping an entry has to say which.
        self._label_states: dict[Path, str] = {}
        self._measurements: dict[Path, LabelMeasurement] = {}
        #: The labels a measurement has covered, **keyed by the label**. Apart
        #: from ``_label_empty``, which the label filter also fills: a label
        #: opened to ask whether it is blank has not been measured.
        self._measured_labels: set[Path] = set()
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
        self._measured_labels = set()
        self._measurements_loaded = False
        self._lookup = project_lookup(self._projects)
        logger.debug("Export view holding %d record(s)", len(self._records))

    def forget_labels(self, paths: Iterable[Path]) -> int:
        """Drop what was read from the labels at *paths*, so they are read again.

        A conversion replaces labels this view may already have opened,
        classified and measured, and the path is the key to all three -- so a
        replaced label would go on reporting the state and the area of the one
        it replaced. That used to be reset by the scan after converting, which
        handed the projects over again.

        **The three are not keyed alike**, so *paths* cannot simply be popped
        from each: ``_label_empty`` is keyed by the label, while
        ``_label_states`` and ``_measurements`` are keyed by the *image*, which
        is what lets :mod:`..aggregate` close over them exactly as it does the
        header mapping. The images are found from what each record carries
        rather than through ``find_matching_label``, which stats the disk once
        per record: this is on the path that exists to avoid reading the
        collection, and the file was just written, so whether it is there is
        not in question.

        The header cache is untouched: a label is not an acquisition, and
        nothing here changes what an image's header says.

        Returns:
            How many of *paths* this view had actually read.
        """
        wanted = {Path(path) for path in paths}
        if not wanted:
            return 0
        forgotten = sum(
            1 for path in wanted
            if self._label_empty.pop(path, _ABSENT) is not _ABSENT
        )
        self._measured_labels -= wanted
        for record in self._records:
            if any(label in wanted for label in record.labels):
                self._label_states.pop(record.image_path, None)
                self._measurements.pop(record.image_path, None)
        return forgotten

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
        """Return whether everything the filters match has been measured.

        Asked of the selection rather than remembered as one flag: widening a
        filter brings in labels nothing has opened, and a tab that went on
        answering from what it measured before would be answering about a
        different set than the one on screen.
        """
        return bool(self._records) and not self._unmeasured_labels()

    def needs_measurements(self, name: str) -> bool:
        """Return whether *name* cannot be counted until the labels are measured.

        Asked before counting rather than measuring on demand: a breakdown
        that quietly opened every label would fire on each press of an arrow
        key stepping down the list.
        """
        entry = self.breakdowns.get(name)
        return entry is not None and entry.kind == "measurement"

    def measurable_labels(self) -> int:
        """Return how many distinct labels a measurement would still open.

        What the button says before it is pressed. Counted over what the
        filters match rather than the whole collection -- an export is chosen
        by narrowing, and measuring an archive to answer a question about one
        project of it is work nobody asked for. The label is the unit, not the
        image, so several imaging conditions of one scene cost one read
        between them, and one already measured costs nothing.
        """
        return len(self._unmeasured_labels())

    def unread_labels(self) -> int:
        """Return how many distinct labels of the selection nobody has opened.

        Fewer than :meth:`measurable_labels` once the label filter has looked
        inside them: asking whether a label is blank opens it without
        measuring it.
        """
        return len(self._selected_labels() - self._label_empty.keys())

    def _unmeasured_labels(self) -> set[Path]:
        """Return the labels of the current selection not yet measured."""
        return self._selected_labels() - self._measured_labels

    def _selected_labels(self) -> set[Path]:
        """Return the distinct labels the current selection pairs with."""
        return {
            found
            for record in self.selected_records()
            if (found := find_matching_label(record)) is not None
        }

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

    def load_measurements(
        self, on_progress: Callable[[int, int], None] | None = None
    ) -> int:
        """Measure every label, once per scan.

        Tens of milliseconds per label against half a millisecond for a header,
        which is why nothing calls this implicitly: it is the one thing in this
        tab that has to be asked for.

        The headers are read first and kept. Measuring needs them anyway, for
        the scale; reading them into the mapping rather than letting
        ``measure_records`` read and discard its own is what makes a later
        change of threshold free.

        Args:
            on_progress: Called as ``on_progress(done, total)`` while measuring,
                counted in label groups. Reading the headers first is not
                counted: it is half a millisecond an image against tens of
                milliseconds a label, so a counter covering both would sit at
                zero and then move.

        Returns:
            How many images have a measured label.
        """

        self.load_metadata()
        records = self.selected_records()
        measured = measure_records(
            records,
            self.config,
            self._metadata,
            min_crack_area_um2=self._min_crack_area_um2,
            on_progress=on_progress,
        )
        self._measurements.update(measured)
        # Every label the pass reached, including one it could not read: that
        # is an answer too, and asking again would read it again.
        self._measured_labels.update(
            label for record in records
            if (label := find_matching_label(record)) is not None
        )
        self._remember_label_states(measured)
        self._measurements_loaded = True
        return len(self._measurements)

    def _remember_label_states(self, measured: Mapping[Path, LabelMeasurement]) -> None:
        """Record what each measured label holds, without opening it again.

        Measuring reads the pixels, and whether there is anything in them is
        the same read -- so the exportable count, the label filter and the
        label-state breakdown are all answered by the one pass the user asked
        for rather than by a second one they did not.
        """

        for image_path, measurement in measured.items():
            self._label_empty[measurement.path] = measurement_is_empty(measurement)
            self._label_states[image_path] = classify_label(
                measurement.path, self._label_empty[measurement.path]
            )
        # An image whose label is missing is absent from the measurements, and
        # that is an answer too: there is nothing to open and nothing to wait
        # for. Recorded here so it is not asked again image by image.
        for record in self.selected_records():
            if record.image_path not in self._label_states:
                label = find_matching_label(record)
                if label is None:
                    self._label_states[record.image_path] = classify_label(None, None)

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
        refresh, and answering it opens a file. Answered from the same cache
        the exportable count reads, because they are the same question about
        the same file: keeping two caches read every label in the collection
        twice.
        """
        path = record.image_path
        if path not in self._label_states:
            label = find_matching_label(record)
            self._label_states[path] = classify_label(
                label, self._is_empty_label(label) if label is not None else None
            )
        return self._label_states[path]

    def warm_labels(
        self, on_progress: Callable[[int, int], None] | None = None
    ) -> int:
        """Open every label once, filling the caches the whole tab counts from.

        The tab cannot show a single number without knowing what is inside each
        label -- a blank label is not an annotation -- and over a real
        collection that is half a minute of reading. Done here, in one pass
        that can report progress and be run off the interface thread, rather
        than inside the first refresh, where it stopped the window dead with
        nothing on screen to say why.

        Counted in label files rather than images, because several imaging
        conditions of one scene share one and cost one read between them, and
        over what the filters match rather than the whole collection -- reading
        an archive to write an export chosen from one project of it is work
        nobody asked for.

        Cheaper than :meth:`load_measurements`, which opens the same files and
        also counts what is in them; this only asks whether anything is. What
        an export needs, in other words, and not what a breakdown does.

        Args:
            on_progress: Called as ``on_progress(done, total)`` in label files.

        Returns:
            How many label files were read.
        """
        records = self.selected_records()
        paired = [(record, find_matching_label(record)) for record in records]
        labels = sorted({label for _, label in paired if label is not None})
        for done, label in enumerate(labels, start=1):
            self._is_empty_label(label)
            if on_progress is not None:
                on_progress(done, len(labels))
        for record, label in paired:
            self._label_states.setdefault(
                record.image_path,
                classify_label(
                    label, self._label_empty[label] if label is not None else None
                ),
            )
        return len(labels)

    def describe_filters(self) -> str:
        """Return what the filters narrow to, as one line of the tab's own words.

        For the fork point an export writes, whose ``scope`` records what a
        copy was asked for so that whoever merges it back months later can see
        what it was meant to hold. **The tab's vocabulary, not the command
        line's**: a copy made here was narrowed by these controls, and writing
        it as flags would describe a run nobody made.

        Everything left at its default is left out, so an unnarrowed export
        says so rather than listing the whole vocabulary back.
        """
        parts = []
        if self.filters.modality != "all":
            parts.append(f"modality {self.filters.modality}")
        if self.filters.state != "all":
            parts.append(f"state {self.filters.state}")
        if self.filters.series:
            parts.append(f"series {', '.join(sorted(self.filters.series))}")
        if self.filters.conditions != "all":
            parts.append(f"conditions {self.filters.conditions}")
        if self.filters.label != ANY_LABEL:
            parts.append(f"label {self.filters.label}")
        for name, values in sorted(self.filters.choices.items()):
            parts.append(
                f"{name} {', '.join(sorted(values))}" if values else f"{name} none"
            )
        return "; ".join(parts) if parts else "everything in the scan"

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
            and (not self.filters.series or record.series in self.filters.series)
            and matches_state(record, self.filters.state)
            and (wanted is None or self.label_state(record) in wanted)
        ]
        # Each remaining filter keeps the records whose group is one of the
        # chosen ones -- the breakdown's own key function, read as a predicate.
        # Compared as the string the list offered, so what was picked and what
        # is matched cannot drift: a pixel size is grouped to four significant
        # figures, and the filter has to agree with the table about that.
        # Present with no values is a filter that matches nothing, not an
        # absent one: see :meth:`set_choice_filter`. So membership decides
        # here, never whether the list is empty.
        chosen = {
            name: set(values)
            for name, values in self.filters.choices.items()
            if name != ignoring
        }
        if chosen:
            for name, key in zip(chosen, self._keys(list(chosen))):
                allowed = chosen[name]
                multi = self.breakdowns[name].multi
                # Any of them, for a property a record has several of: an
                # export narrowed to infected samples wants every image of one,
                # not only those of samples that are nothing else.
                kept = [
                    record for record in kept
                    if any(
                        str(group) in allowed
                        for group in groups_of(key, record, multi=multi)
                    )
                ]
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

    def labels_known(self, matched: Sequence[ImageRecord] | None = None) -> bool:
        """Return whether every matched label has already been opened.

        What decides whether the counts below the first two are answerable.
        Nothing here opens one to find out: that is the cost this asks about.
        """
        records = self.selected_records() if matched is None else matched
        return all(
            (label := find_matching_label(record)) is None
            or label in self._label_empty
            for record in records
        )

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

        corrected = [
            record for record in self.selected_records()
            if current_state(record) == "corrected"
        ]
        return classify_labels(corrected, self.label_state)

    @property
    def paired(self) -> bool:
        """Return whether an export writes both beams on one row.

        Both modalities being in play is the whole of the question. A file
        mixing SEM and FIB rows under one ``image`` column is not something
        anyone trains on, so choosing neither modality means choosing to
        export them side by side.
        """
        return self.filters.modality == "all"

    def training_rows(self, matched: Sequence[ImageRecord] | None = None) -> list:
        """Return the rows a training export would write, from one builder.

        The preview and the file come from the same call, so a count on screen
        that disagreed with what was written cannot happen. The emptiness
        answers come from this controller's cache, so building the preview
        opens no label the tab has not already read.
        """
        records = self.selected_records() if matched is None else matched
        return build_training_rows(
            records,
            self.start_path or Path(),
            paired=self.paired,
            is_empty=self._is_empty_label,
        )

    def export_training(
        self, output_dir: Path, *, overwrite: bool = False
    ) -> tuple[Path, Path | None]:
        """Write the training files and their manifest for the current selection.

        Here rather than in the window because every part of it is a decision
        -- which records, which shape, how split, what named -- and the window's
        share is only asking where to put them. The emptiness answers come from
        this controller's cache, so writing opens no label the tab has already
        read.

        Returns:
            ``(train_csv_path, val_csv_path)``, the second None at a full
            training share.

        Raises:
            FileExistsError: If an earlier export's files are in the way and
                *overwrite* is False.
        """

        return export_training_csv(
            records=self.selected_records(),
            start_path=self.start_path or output_dir,
            output_dir=output_dir,
            prefix=self.training.prefix,
            train_ratio=self.training.train_ratio,
            seed=self.training.seed,
            # The window never writes blank masks. Deliberate negative examples
            # are a considered thing, and the command line's
            # --include-empty-labels is where they stay.
            include_empty_labels=False,
            # Already applied: the filters decide which images are in play, and
            # "one image per label" is one of them.
            conditions="all",
            paired=self.paired,
            is_empty=self._is_empty_label,
            overwrite=overwrite,
            scope=self.describe_filters(),
        )

    def export_image_metadata(self, output: Path) -> Path:
        """Write the per-image metadata for the current selection.

        Paired when both modalities are in play, matching the training export:
        one row per acquisition, each beam's own values prefixed with which
        beam. Choosing a single modality gives one row per image file, which is
        what a repository deposition wants -- and two such exports append,
        since they carry the same columns.
        """

        return export_image_metadata_csv(
            self.selected_records(),
            self.start_path or output.parent,
            output,
            projects=project_lookup(self._projects),
            paired=self.paired,
        )

    def preview(self) -> ExportPreview:
        """Return what the current filters would export."""

        matched = self.selected_records()
        if not matched:
            return ExportPreview()
        if not self.labels_known(matched):
            # Cheap by construction: the two counts above the line are the
            # record's own, and everything below one needs a label opened.
            return ExportPreview(
                matched=len(matched),
                corrected=sum(
                    1 for record in matched if current_state(record) == "corrected"
                ),
                paired=self.paired,
                measured=False,
            )

        sifted = self._sift(matched)
        rows = self.training_rows(matched)
        train, validation = split_rows(
            rows,
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
            exportable=len(rows),
            train=len(train),
            validation=len(validation),
            paired=self.paired,
            # Counted off the built rows rather than re-derived, so what the
            # line says is a fact about the file rather than a second opinion.
            sem_only=sum(1 for row in rows if row.values[1] and not row.values[3])
            if self.paired else 0,
            fib_only=sum(1 for row in rows if row.values[3] and not row.values[1])
            if self.paired else 0,
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
        entry = self.breakdowns[name]
        return count_by(
            self.selected_records(), key, entry.order, multi=entry.multi
        )

    def cross_tabulation(self, row: str, column: str) -> dict:
        """Return counts of what would be exported by two properties at once.

        The question a training set actually raises: not how many Arctis
        images there are, but how many of them are corrected FIB at 30 kV --
        and, more usefully, which combinations have nothing in them at all.
        """
        row_key, column_key = self._keys([row, column])
        return cross_tabulate(
            self.selected_records(), row_key, column_key,
            row_multi=self.breakdowns[row].multi,
            column_multi=self.breakdowns[column].multi,
        )

    def export_breakdown_csv(
        self, output: Path, name: str, other: str | None = None
    ) -> Path:
        """Write the breakdown on screen as CSV, as ``am-seg stats --output`` does.

        Raises:
            ValueError: If a property it counts by needs the labels measured and
                they are not -- which would write every image under "no-label".
        """
        unmeasured = [
            chosen for chosen in (name, other)
            if chosen and self.needs_measurements(chosen)
        ]
        if unmeasured and not self.measurements_loaded():
            raise ValueError(
                f"Counting by {' and '.join(unmeasured)} needs the labels measured first"
            )
        if other:
            return write_table_csv(
                output, [name, other, "images"],
                cross_tabulation_as_rows(self.cross_tabulation(name, other)),
            )
        return write_table_csv(
            output, [name, "images"], counts_as_rows(self.breakdown(name))
        )

    def filterable_breakdowns(self) -> list[tuple[str, str]]:
        """Return the ``(name, label)`` of every breakdown offered as a filter.

        A filter and a breakdown are the same key function used two ways, so
        the list of one is derived from the other rather than declared beside
        it. Ordered as the breakdown dropdown is, so a property is found in
        the same place in both.

        Measurement breakdowns are left out: they cannot answer until the
        labels have been measured, and a filter that quietly keeps nothing
        until a button elsewhere is pressed is worse than no filter. So are
        the :data:`UNFILTERABLE` ones, which can be counted on but not usefully
        chosen from.
        """
        headings = list(GROUP_TITLES)
        rows = [
            (name, entry)
            for name, entry in self.breakdowns.items()
            if entry.kind != "measurement"
            and name not in DEDICATED_CONTROLS
            and name not in UNFILTERABLE
        ]
        rows.sort(key=lambda row: (headings.index(row[1].group), row[1].label))
        return [(name, entry.label) for name, entry in rows]

    def filterable_groups(self) -> list[tuple[str, list[tuple[str, str]]]]:
        """Return the filterable breakdowns as headed groups, for a dropdown.

        Grouped and ordered as the breakdown dropdown is, so a property sits in
        the same place whether it is being counted by or filtered on.
        """
        offered = dict(self.filterable_breakdowns())
        grouped = []
        for heading, rows in grouped_breakdowns(self.breakdowns):
            kept = [row for row in rows if row[0] in offered]
            if kept:
                grouped.append((heading, kept))
        return grouped

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
        multi = self.breakdowns[name].multi
        # Each value on its own rather than the set a record holds: a filter
        # offering "Diamond; RFI" as one entry could not be used to find every
        # project either of them touched, which is the point of the split.
        values = {
            str(group)
            for record in self.selected_records(ignoring=name)
            for group in groups_of(key, record, multi=multi)
        }
        return sorted(
            values | set(self.filters.choices.get(name, ())), key=value_order
        )

    def set_choice_filter(
        self, name: str, values: Sequence[str], *, empty_keeps_everything: bool = True
    ) -> None:
        """Set or clear one filter.

        Args:
            empty_keeps_everything: What an empty *values* means. True clears
                the filter, which is what a filter row wants -- ticking none
                of a property's values is how you stop filtering on it. False
                keeps an empty choice, which matches no record at all: a
                control that is always on screen and starts with everything
                ticked has no other way to say "none of them", and unticking
                its last row would otherwise widen the export to everything.

        Raises:
            ValueError: If *name* is not a breakdown offered as a filter.
        """
        self._require_filter(name)
        if values or not empty_keeps_everything:
            self.filters.choices[name] = list(values)
        else:
            self.filters.choices.pop(name, None)

    def _require_filter(self, name: str) -> None:
        """Raise unless *name* is a breakdown that can be filtered on."""
        if name not in self.breakdowns or name in FIXED_CONTROLS:
            raise ValueError(f"Unknown filter: {name}")
        if self.breakdowns[name].kind == "measurement":
            raise ValueError(f"Cannot filter on a measured property: {name}")

    @property
    def metadata_map(self) -> Mapping[Path, ImageMetadata]:
        return self._metadata
