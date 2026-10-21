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
    SPLIT,
    SPLIT_BREAKDOWN,
    Breakdown,
    KeyFunction,
    breakdowns_for,
    count_by,
    cross_tabulate,
    grouped_breakdowns,
    groups_of,
    project_lookup,
    resolve_breakdown,
    split_balance,
    split_grouping,
    value_order,
)
from ..config import SegmentationConfig, load_segmentation_config
from ..filtering import matches_state
from ..image_metadata import ImageMetadata, read_records_metadata
from ..io.directories import directory_cache
from ..io.tables import write_csv
from ..models import ImageRecord, Project
from ..operations.label_groups import find_matching_label
from ..operations.label_io import (
    ANNOTATED,
    EMPTY,
    MISSING,
    UNREADABLE,
    LabelReadiness,
    classify_labels,
)
from ..operations.label_measure import (
    LabelMeasurement,
    apply_crack_threshold,
    measure_records,
    measurement_is_empty,
)
from ..operations.metadata_csv import (
    describe_training_images,
    export_image_metadata_csv,
    export_project_metadata_csv,
)
from ..operations.selection import current_state
from ..operations.training_export import (
    TrainingExportRequest,
    TrainingRow,
    build_training_rows,
    divide_rows,
    earlier_sides,
    export_kfold_csv,
    export_training_csv,
    image_sides,
    keep_one_condition_per_scene,
)
from ..operations.training_split import (
    DEFAULT_KEEP_TOGETHER,
    DEFAULT_TRAIN_RATIO,
    LEFT_OUT,
    TRAINING,
    VALIDATION,
    Balance,
    Grouping,
    assign_folds,
)
from ..reporting import counts_as_rows, cross_tabulation_as_rows
from .batch_scope import counted
from .known_labels import KnownLabels

logger = logging.getLogger(__name__)

#: What a training set of proofread and predicted labels is filtered to.
LABELLED_STATES = "corrected + selected"

#: Workflow states offered as a filter, plus "any" and the two a model trains on.
STATES = ("all", "selected", "deselected", "corrected", LABELLED_STATES)

#: What the window offers to balance a validation set by, as ``(breakdown, caption)``;
#: the command line's ``--balance`` takes any, and several.
BALANCE_BY: tuple[tuple[str, str], ...] = (
    ("", "Nothing"),
    ("lamella-type", "Lamella type"),
    ("microscope", "Microscope"),
    ("voltage", "Voltage"),
)

#: The state validated on where a training export validates on one alone.
VALIDATED_STATE = "corrected"

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
#: `gui/preferences.py`. A scan holds reference and alignment images as well,
#: and an export that silently grew to include them would disagree with
#: ``create-training-csv``, which defaults the same way. Ticking another is one
#: click, and the summary line says what is counted.
DEFAULT_SERIES: tuple[str, ...] = ("adaptive",)

#: What the beam and state filters start on: **proofread work from one beam**,
#: what a training run reads. Opened on everything, a training set would
#: include images somebody deselected, and pair both beams on one row, which no
#: run can train on. Either is one choice away, and narrows the breakdown too.
DEFAULT_MODALITY = "SEM"
DEFAULT_STATE = "corrected"

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

#: The most cross-validation folds the window offers: enough for leaving out
#: one lamella at a time across the whole collection.
MAX_FOLDS = 999


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
            Shown because a workflow-state filter set to include them lets a
            training set quietly hold images somebody rejected -- the
            empty-label rule catches most of them, but not one that was
            labelled and deselected afterwards.
        annotated: How many have a label holding an annotation.
        empty_labels: Images excluded because their label holds nothing but
            background.
        missing_labels: Images excluded because they have no label at all.
        unreadable_labels: Images excluded because their label could not be
            read, so could not be told blank or annotated.
        exportable: What a training export would actually write -- images
            where one modality is chosen, rows where both are.
        train: How many of those the split would put in the training set.
        validation: How many it would put in the validation set. The two are
            whole groups, so the achieved ratio is only as close to the
            requested one as the grouping allows -- which is the reason to show
            it before writing rather than after.
        left_out: Rows in neither set: not proofread, and of a group that is
            validated on. Zero unless only corrected labels are validated on.
        split_problem: Why the rows cannot be divided as asked, which an
            export would be refused for; empty where they can.
        groups: How many groups the split kept whole.
        keep_together: What a group is: a key of
            :data:`~am_segmentation_manager.operations.training_split.KEEP_TOGETHER`.
        folds: How many cross-validation folds were asked for; 0 for a single
            split, which *train* and *validation* then describe.
        fold_sizes: The rows each fold validates on, empty where there are
            fewer groups than folds.
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
    unreadable_labels: int = 0
    exportable: int = 0
    train: int = 0
    validation: int = 0
    left_out: int = 0
    split_problem: str = ""
    groups: int = 0
    keep_together: str = DEFAULT_KEEP_TOGETHER
    folds: int = 0
    fold_sizes: tuple[int, ...] = ()
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
        if self.unreadable_labels:
            parts.append(f"{self.unreadable_labels} with an unreadable label")
        if self.paired:
            parts.append(f"{self.exportable} paired row(s)")
            if self.sem_only:
                parts.append(f"{self.sem_only} SEM only")
            if self.fib_only:
                parts.append(f"{self.fib_only} FIB only")
        else:
            parts.append(f"{self.exportable} exportable")
        if self.exportable:
            parts.append(self.split_text())
        return " · ".join(parts)

    def split_text(self) -> str:
        """Return how the exportable rows divide, for the summary and the split's own line."""
        if self.split_problem:
            return self.split_problem
        if not self.folds:
            text = f"{self.train} train / {self.validation} validation"
            return f"{text} / {self.left_out} left out" if self.left_out else text
        if not self.fold_sizes:
            groups = counted(self.groups, *GROUP_NOUNS[self.keep_together])
            return f"{groups} cannot fill {self.folds} folds"
        low, high = min(self.fold_sizes), max(self.fold_sizes)
        sizes = str(low) if low == high else f"{low}–{high}"
        return f"{self.folds} folds of {sizes} validation"


#: What a group of the training split is called, one and several, by level.
GROUP_NOUNS: dict[str, tuple[str, str]] = {
    "acquisition": ("acquisition", "acquisitions"),
    "lamella": ("lamella", "lamellae"),
    "project": ("project", "projects"),
}


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
        unreadable_labels: Images whose label could not be read, so could not
            be told blank or annotated.
    """

    pairs: list[tuple[ImageRecord, Path]]
    annotated: int = 0
    empty_labels: int = 0
    missing_labels: int = 0
    unreadable_labels: int = 0


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
        modality: ``"all"``, ``"SEM"`` or ``"FIB"``. Starts at
            :data:`DEFAULT_MODALITY`.
        state: A workflow state, :data:`LABELLED_STATES` or ``"all"``. Starts at
            :data:`DEFAULT_STATE`.
        series: What the images were taken for, as a set of ticked values.
            Starts at :data:`DEFAULT_SERIES` rather than at everything.
        conditions: ``"all"`` keeps every imaging condition of a scene,
            ``"primary"`` keeps one image per label. Starts at ``"all"``,
            unlike ``create-training-csv``, where Settings (F2) name nothing.
        label: A key of :data:`LABEL_FILTERS`.
    """

    modality: str = DEFAULT_MODALITY
    state: str = DEFAULT_STATE
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
        keep_together: What the split never divides, a key of
            :data:`~am_segmentation_manager.operations.training_split.KEEP_TOGETHER`.
        folds: How many cross-validation folds to write instead of one split,
            or None for one split. *train_ratio* then plays no part.
        validate_on_corrected: Validate on corrected labels only, the rest
            training or left out as
            :func:`~am_segmentation_manager.operations.training_export.divide_rows` says.
            Plays no part in folds.
        balance: The breakdown the validation side is given its share of, a
            key of :data:`BALANCE_BY`; empty for none. Plays no part in folds.
        validation_from: The manifest of an earlier export whose split to keep
            to, or None. Set through :meth:`ExportController.set_validation_from`.
        prefix: Filename stem, written as ``<prefix>_train.csv`` and
            ``<prefix>_val.csv``.
    """

    train_ratio: float = DEFAULT_TRAIN_RATIO
    seed: int | None = 42
    keep_together: str = DEFAULT_KEEP_TOGETHER
    folds: int | None = None
    validate_on_corrected: bool = False
    balance: str = ""
    validation_from: Path | None = None
    prefix: str = DEFAULT_PREFIX


class ExportController:
    """Holds a scan, applies filters to it, and reports what would be exported."""

    def __init__(
        self,
        start_path: Path | None = None,
        config: SegmentationConfig | None = None,
        *,
        known_labels: KnownLabels | None = None,
    ) -> None:
        self.start_path = start_path
        #: What each label holds, shared with the window's other views.
        #: The side each image of the export `training.validation_from` names was on.
        self._earlier: dict[str, str] | None = None
        self.known_labels = known_labels if known_labels is not None else KnownLabels()
        # Which class is a crack, and how small a component still counts as one.
        # Loaded here when nobody supplies it so the controller stands alone in
        # a test; the window hands over the one the overlays already use.
        self.config = config if config is not None else load_segmentation_config()
        # Resolved once: which area breakdowns exist is the config's to say,
        # and rebuilding the table per refresh would rebuild the closures
        # the breakdowns are made of.
        self._area_bins_per_decade = DEFAULT_AREA_BINS_PER_DECADE
        self.breakdowns: dict[str, Breakdown] = self._breakdown_table()
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
        #: The headers the scan read, None where it found none; see
        #: :meth:`set_projects`.
        self._scan_headers: dict[Path, ImageMetadata | None] = {}
        #: Each image's measurement, **keyed by the image**, which is what lets
        #: :mod:`..aggregate` close over it the way it closes over the headers.
        self._measurements: dict[Path, LabelMeasurement] = {}
        #: The labels a measurement has covered, **keyed by the label**. Apart
        #: from what :attr:`known_labels` knows, which a label opened on its own
        #: to ask whether it is blank also fills: that label has not been measured.
        self._measured_labels: set[Path] = set()
        # Built by walking every record, and asked for once per filter list on
        # every refresh. Rebuilt only when the scan changes.
        self._lookup: dict[Path, Project] = {}

    # -- loading -----------------------------------------------------------

    def set_projects(
        self,
        projects: Sequence[Project],
        start_path: Path | None = None,
        headers: Mapping[Path, ImageMetadata | None] | None = None,
    ) -> None:
        """Take the projects from an existing scan, rather than scanning again.

        The curation window has already scanned; a second scan for the export
        view would double the slowest thing the application does. *headers* are
        what that scan read, :attr:`~..scanner.ProjectIndex.headers`, which
        :meth:`load_metadata` does not read again.
        """
        self._projects = list(projects)
        if start_path is not None:
            self.start_path = start_path
        self._records = [
            record for project in self._projects for record in project.image_records
        ]
        self._scan_headers = dict(headers or {})
        self._metadata = {}
        self._metadata_loaded = False
        self.known_labels.clear()
        self._measurements = {}
        self._measured_labels = set()
        self._lookup = project_lookup(self._projects)
        logger.debug("Export view holding %d record(s)", len(self._records))

    def forget_labels(self, paths: Iterable[Path]) -> int:
        """Drop what was read from the labels at *paths*, so they are read again.

        A conversion replaces labels this view may already have opened,
        classified and measured, and the path is the key to all three -- so a
        replaced label would go on reporting the state and the area of the one
        it replaced.

        **They are not keyed alike**, so *paths* cannot simply be popped from
        each: what a label holds is kept by the label, in the shared
        :attr:`known_labels`, while ``_measurements`` is keyed by the *image*,
        which is what lets :mod:`..aggregate` close over it exactly as it does
        the header mapping. The images are found from what each record carries
        rather than through ``find_matching_label``, which outside a listing
        block stats the disk once per record: this is on the path that exists to avoid reading the
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
        forgotten = self.known_labels.forget(labels=wanted)
        self._measured_labels -= wanted
        for record in self._records:
            if any(label in wanted for label in record.labels):
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
        return sum(
            1 for label in self._selected_labels(label_filter=False)
            if not self.known_labels.knows(label)
        )

    def labels_pending(self, breakdowns: Sequence[str | None] = ()) -> int:
        """Return how many labels must be opened before the view can be counted.

        Nonzero only where the label filter or one of *breakdowns* asks what a
        label holds, and some label has not been opened yet. Answered without
        opening one, so the window can read them behind a dialog that counts
        rather than stalling in the middle of a redraw.
        """
        # The side of the split is asked of the rows, which no label that is
        # blank or unread is in.
        asked = self.filters.label != ANY_LABEL or any(
            name in NEEDS_LABEL_STATES or name == SPLIT for name in breakdowns
        )
        return self.unread_labels() if asked else 0

    def _unmeasured_labels(self) -> set[Path]:
        """Return the labels of the current selection not yet measured."""
        return self._selected_labels() - self._measured_labels

    def _selected_labels(self, *, label_filter: bool = True) -> set[Path]:
        """Return the distinct labels the current selection pairs with."""
        with directory_cache():
            return {
                found
                for record in self.selected_records(label_filter=label_filter)
                if (found := find_matching_label(record)) is not None
            }

    @property
    def min_crack_area_um2(self) -> float:
        """Return the smallest component currently counted as a crack, in um^2."""
        return self._min_crack_area_um2

    def set_crack_threshold(self, value: float) -> None:
        """Set the smallest component counted as a crack, in um^2.

        Applied at once to every measurement held, those the filters hide
        included, and opening no label: each measurement carries the component
        pixel counts the threshold is applied to. A label not yet measured still
        waits for :meth:`load_measurements`.

        Raises:
            ValueError: If *value* is negative.
        """
        if value < 0:
            raise ValueError(f"A crack threshold cannot be negative: {value!r}")
        if value == self._min_crack_area_um2:
            return
        self._min_crack_area_um2 = float(value)
        # Keyed by identity while the old mapping keeps every old object alive:
        # records sharing a label must go on sharing one measurement.
        applied: dict[int, LabelMeasurement] = {}
        self._measurements = {
            image_path: applied.setdefault(
                id(measurement), apply_crack_threshold(measurement, self._min_crack_area_um2)
            )
            for image_path, measurement in self._measurements.items()
        }

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
        self._area_bins_per_decade = per_decade
        self.breakdowns = self._breakdown_table()

    def _breakdown_table(self) -> dict[str, Breakdown]:
        """Return every breakdown this tab counts by, the side of its own split among them."""
        return {
            **breakdowns_for(self.config, self._area_bins_per_decade),
            SPLIT: SPLIT_BREAKDOWN,
        }

    def load_measurements(
        self, on_progress: Callable[[int, int], None] | None = None
    ) -> int:
        """Measure every label the filters match, once per scan.

        Tens of milliseconds per label against under a millisecond for a header,
        which is why nothing calls this implicitly: it is the one thing in this
        tab that has to be asked for.

        The headers are read first and kept. Measuring needs them anyway, for
        the scale; reading them into the mapping rather than letting
        ``measure_records`` read and discard its own is what lets a label a
        widened filter brings in be measured without reading a header again.

        Args:
            on_progress: Called as ``on_progress(done, total)`` in label files
                read. Reading the headers first is not counted: it is under a
                millisecond an image against tens of milliseconds a label, so a
                counter covering both would sit at zero and then move.

        Returns:
            How many images have a measured label.
        """
        self._measure(self.selected_records(), on_progress)
        return len(self._measurements)

    def _measure(
        self,
        records: Sequence[ImageRecord],
        on_progress: Callable[[int, int], None] | None,
    ) -> int:
        """Measure the labels of *records*, remembering what each one holds.

        Returns:
            How many distinct label files the records pair with.
        """
        self.load_metadata()
        measured = measure_records(
            records,
            self.config,
            self._metadata,
            min_crack_area_um2=self._min_crack_area_um2,
            on_progress=on_progress,
        )
        self._measurements.update(measured)
        labels = {
            record.image_path: label
            for record in records
            if (label := find_matching_label(record)) is not None
        }
        # Every label the pass reached, including one it could not read: that
        # is an answer too, and asking again would read it again.
        self._measured_labels.update(labels.values())
        self._remember_label_states(records, labels, measured)
        return len(set(labels.values()))

    def _remember_label_states(
        self,
        records: Sequence[ImageRecord],
        labels: Mapping[Path, Path],
        measured: Mapping[Path, LabelMeasurement],
    ) -> None:
        """Record what each record's label holds, opening no measured label again.

        Measuring reads the pixels, and whether there is anything in them is
        the same read -- so the exportable count, the label filter and the
        label-state breakdown are all answered by the one pass rather than by
        a second. A label the measurement could not read is asked of
        :func:`is_label_file_empty` instead, which accepts masks measuring
        refuses, such as one of non-integer values.
        """
        for image_path, measurement in measured.items():
            self.known_labels.remember(
                image_path, measurement.path, measurement_is_empty(measurement)
            )
        self.known_labels.prefetch(
            record for record in records if record.image_path not in measured
        )
        for record in records:
            if record.image_path in measured:
                continue
            label = labels.get(record.image_path)
            self.known_labels.remember(
                record.image_path,
                label,
                self.known_labels.emptiness(label) if label is not None else None,
            )

    def load_metadata(
        self, on_progress: Callable[[int, int], None] | None = None
    ) -> int:
        """Read every record's header once per scan, opening none the scan read.

        Args:
            on_progress: Called as ``on_progress(done, total)`` in headers read.

        Returns:
            How many headers yielded metadata. Fewer than the record count
            simply means some files carry none.
        """
        if not self._metadata_loaded:
            self._metadata = {
                record.image_path: header
                for record in self._records
                if (header := self._scan_headers.get(record.image_path)) is not None
            }
            self._metadata.update(
                read_records_metadata(self._records_without_headers(), on_progress)
            )
            self._metadata_loaded = True
        return len(self._metadata)

    def headers_to_read(self) -> int:
        """Return how many headers :meth:`load_metadata` would still open."""
        return 0 if self._metadata_loaded else len(self._records_without_headers())

    def _known_headers(self) -> dict[Path, ImageMetadata | None]:
        """Return every header read so far, the scan's and this tab's, None where one had none."""
        known = dict(self._scan_headers)
        if self._metadata_loaded:
            known.update(
                (record.image_path, self._metadata.get(record.image_path))
                for record in self._records
            )
        return known

    def _records_without_headers(self) -> list[ImageRecord]:
        """Return the records whose header the scan did not read."""
        return [
            record for record in self._records if record.image_path not in self._scan_headers
        ]

    # -- filtering ---------------------------------------------------------

    def label_state(self, record: ImageRecord) -> str:
        """Return *record*'s label state, reading each label at most once.

        The filter and the breakdown both ask this of every record on every
        refresh, and answering it opens a file. Answered from the same cache
        the exportable count reads, because they are the same question about
        the same file: keeping two caches read every label in the collection
        twice.
        """
        return self.known_labels.state(record)

    def warm_labels(
        self, on_progress: Callable[[int, int], None] | None = None
    ) -> int:
        """Open every label once, filling the caches the whole tab counts from.

        The tab cannot show a single number without knowing what is inside each
        label -- a blank label is not an annotation. Done here, in one pass
        that can report progress and be run off the interface thread, rather
        than inside the first refresh, where it stopped the window dead with
        nothing on screen to say why.

        Counted in label files rather than images, because several imaging
        conditions of one scene share one and cost one read between them, and
        over what the filters match rather than the whole collection -- reading
        an archive to write an export chosen from one project of it is work
        nobody asked for. The label filter itself is left out, being what this
        answers.

        **Each label is measured while it is open**: counting its pixels is the
        read that says whether there are any, so a breakdown by area or crack
        chosen afterwards opens no label again.

        Args:
            on_progress: Called as ``on_progress(done, total)`` in label files.

        Returns:
            How many label files the selection pairs with.
        """
        return self._measure(self.selected_records(label_filter=False), on_progress)

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

    def selected_records(
        self, ignoring: str | None = None, *, label_filter: bool = True
    ) -> list[ImageRecord]:
        """Return the records passing the current filters.

        The dedup rule is applied here with the rest, not left to the export:
        it decides which images are in play, so the table has to see it.

        Args:
            ignoring: A metadata filter to leave out, which is what lets that
                filter offer the values still reachable rather than only the
                ones it already keeps.
            label_filter: Clear to leave the label filter out, which is the one
                filter that opens files. What remains is found without opening
                a label and holds every record the full set does -- the scene's
                conditions share a label, so dedup keeps the same ones.
        """
        if self.filters.label not in LABEL_FILTERS:
            raise ValueError(f"Unknown label filter: {self.filters.label!r}")
        # None rather than the full set, so the default filter never opens a
        # label to decide it is keeping it anyway.
        wanted = (
            None if self.filters.label == ANY_LABEL or not label_filter
            else LABEL_FILTERS[self.filters.label]
        )
        states = (
            ("corrected", "selected") if self.filters.state == LABELLED_STATES
            else self.filters.state
        )
        kept = [
            record for record in self._records
            if (self.filters.modality == "all" or record.modality == self.filters.modality)
            and (not self.filters.series or record.series in self.filters.series)
            and matches_state(record, states)
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

    def labels_known(self, matched: Sequence[ImageRecord] | None = None) -> bool:
        """Return whether every matched label has already been opened.

        What decides whether the counts below the first two are answerable.
        Nothing here opens one to find out: that is the cost this asks about.
        """
        records = (
            self.selected_records(label_filter=False) if matched is None else matched
        )
        with directory_cache():
            return all(
                (label := find_matching_label(record)) is None
                or self.known_labels.knows(label)
                for record in records
            )

    def _sift(self, matched: Sequence[ImageRecord]) -> Sifted:
        """Separate the images that can be exported from those that cannot.

        The filters have already run, dedup included. What is left is the rule
        no filter can express: a training pair needs a label with something in
        it, so an image whose label is missing or blank is reported here rather
        than written, and so is one whose label could not be read.
        """
        empty = missing = unreadable = annotated = 0
        pairs: list[tuple[ImageRecord, Path]] = []
        for record in matched:
            label = find_matching_label(record)
            if label is None:
                missing += 1
                continue
            emptiness = self.known_labels.emptiness(label)
            if emptiness is None:
                unreadable += 1
                continue
            if emptiness:
                empty += 1
                continue
            annotated += 1
            pairs.append((record, label))

        return Sifted(
            pairs=pairs,
            annotated=annotated,
            empty_labels=empty,
            missing_labels=missing,
            unreadable_labels=unreadable,
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
        self.known_labels.prefetch(records)
        return build_training_rows(
            records,
            self.start_path or Path(),
            paired=self.paired,
            is_empty=self.known_labels.emptiness,
            grouping=self.grouping(),
        )

    def grouping(self) -> Grouping:
        """Return what the split keeps together, keyed as the breakdowns key it."""
        return split_grouping(self.training.keep_together, self._lookup)

    def can_validate_on_corrected(self) -> bool:
        """Return whether the state filter lets corrected images through to validate on."""
        return self.filters.state in ("all", VALIDATED_STATE, LABELLED_STATES)

    @property
    def validate_on(self) -> str | None:
        """Return the one state a single split validates on, or None for any.

        None wherever the state filter leaves out corrected images, whatever the switch says.
        """
        if self.training.validate_on_corrected and self.can_validate_on_corrected():
            return VALIDATED_STATE
        return None

    def balance(self) -> Balance | None:
        """Return what the validation side is balanced by, reading headers where that needs them."""
        if not self.training.balance:
            return None
        if self.training.balance in NEEDS_METADATA:
            self.load_metadata()
        return split_balance([self.training.balance], self._lookup, self._metadata)

    def set_validation_from(self, manifest: Path | None) -> None:
        """Keep to the split of the export *manifest* records, or to none.

        Its lists are read here, once, so a recount opens no file.

        Raises:
            ValueError: If it is not a manifest an export can keep to; the
                choice is then left as it was.
        """
        self._earlier = earlier_sides(manifest) if manifest is not None else None
        self.training.validation_from = manifest

    def export_training(
        self,
        output_dir: Path,
        *,
        overwrite: bool = False,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> Sequence[tuple[Path, Path | None]]:
        """Write the training files and their manifest for the current selection.

        Here rather than in the window because every part of it is a decision
        -- which records, which shape, how split, what named -- and the window's
        share is only asking where to put them. The emptiness answers and the
        headers come from this controller's caches, so writing opens no label or
        header the tab has already read.

        Args:
            on_progress: Called as ``on_progress(done, total)`` in headers read
                for the images file.

        Returns:
            ``(train_csv_path, val_csv_path)`` per fold, or once for a single
            split, whose second is None at a full training share.

        Raises:
            FileExistsError: If an earlier export's files are in the way and
                *overwrite* is False.
            LeftoverFiles: If an earlier export's files would outlive this
                one, whatever *overwrite* says.
            ValueError: If there are fewer groups than folds, or nothing
                corrected to validate on where only that is validated on.
        """
        start_path = self.start_path or output_dir
        request = TrainingExportRequest(
            start_path=start_path,
            prefix=self.training.prefix,
            # Already applied: the filters decide which images are in play, and
            # "one image per label" is one of them.
            conditions="all",
            paired=self.paired,
            # The window never writes blank masks. Deliberate negative examples
            # are a considered thing, and the command line's
            # --include-empty-labels is where they stay.
            include_empty_labels=False,
            is_empty=self.known_labels.emptiness,
            grouping=self.grouping(),
            train_ratio=self.training.train_ratio,
            seed=self.training.seed,
            validate_on=self.validate_on,
            validation_from=self.training.validation_from,
            balance=self.balance(),
            scope=self.describe_filters(),
            describe_images=describe_training_images(
                start_path, project_lookup(self._projects),
                known=self._known_headers(), state_of=self.known_labels.state,
                on_progress=on_progress,
            ),
        )
        records = self.selected_records()
        self.known_labels.prefetch(records)
        if self.training.folds:
            return export_kfold_csv(
                records, request, output_dir, self.training.folds, overwrite=overwrite
            )
        return [export_training_csv(records, request, output_dir, overwrite=overwrite)]

    def export_image_metadata(
        self,
        output: Path,
        *,
        overwrite: bool = False,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Write the per-image metadata for the current selection.

        Paired when both modalities are in play, matching the training export:
        one row per acquisition, each beam's own values prefixed with which
        beam. Choosing a single modality gives one row per image file, which is
        what a repository deposition wants -- and two such exports append,
        since they carry the same columns. Opens no header the tab has read.

        Args:
            on_progress: Called as ``on_progress(done, total)`` in headers read.

        Raises:
            FileExistsError: If *output* exists and *overwrite* is False.
        """
        records = self.selected_records()
        self.known_labels.prefetch(records)
        return export_image_metadata_csv(
            records,
            self.start_path or output.parent,
            output,
            projects=project_lookup(self._projects),
            paired=self.paired,
            overwrite=overwrite,
            known=self._known_headers(),
            state_of=self.known_labels.state,
            on_progress=on_progress,
        )

    def preview(self) -> ExportPreview:
        """Return what the current filters would export.

        Every label directory is listed once for it, rather than the disk asked
        once per record.
        """
        with directory_cache():
            return self._preview()

    def _preview(self) -> ExportPreview:
        """Return what :meth:`preview` does, inside its listing block."""

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
        groups = len({row.group for row in rows})
        folds = self.training.folds or 0
        split_problem = ""
        if folds:
            train: list[TrainingRow] = []
            validation: list[TrainingRow] = []
            left_out: list[TrainingRow] = []
            fold_sizes = tuple(
                len(fold) for fold in assign_folds(
                    rows, lambda row: row.group, folds, self.training.seed
                )
            ) if groups >= folds else ()
        else:
            try:
                train, validation, left_out = self._divided(rows)
            except ValueError as error:
                train, validation, left_out = [], [], []
                split_problem = str(error)
            fold_sizes = ()

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
            unreadable_labels=sifted.unreadable_labels,
            exportable=len(rows),
            train=len(train),
            validation=len(validation),
            left_out=len(left_out),
            split_problem=split_problem,
            groups=groups,
            keep_together=self.training.keep_together,
            folds=folds,
            fold_sizes=fold_sizes,
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
            # Over what the filters could still match rather than the whole
            # scan: every counted record is among these, and a label outside
            # them is one :meth:`labels_pending` never asked to have read.
            states = {
                record.image_path: self.label_state(record)
                for record in self.selected_records(label_filter=False)
            }
        sides = self.split_sides() if SPLIT in names else {}
        return [
            resolve_breakdown(
                name, self._metadata, self._lookup, self._measurements, states,
                self.breakdowns, sides,
            )
            for name in names
        ]

    def split_sides(self) -> dict[Path, str]:
        """Return the side of the training split each exportable image is on.

        From the rows and the division the preview and the files come from: a
        fold's name where folds are written, and nothing where the rows cannot
        be divided as asked, which the split's own line says.
        """
        rows = self.training_rows()
        if self.training.folds:
            groups = len({row.group for row in rows})
            if groups < self.training.folds:
                return {}
            folds = assign_folds(
                rows, lambda row: row.group, self.training.folds, self.training.seed
            )
            return image_sides(
                [(f"fold {number}", fold) for number, fold in enumerate(folds, start=1)]
            )
        try:
            train, validation, left_out = self._divided(rows)
        except ValueError:
            return {}
        return image_sides(
            [(TRAINING, train), (VALIDATION, validation), (LEFT_OUT, left_out)]
        )

    def _divided(self, rows: Sequence[TrainingRow]) -> tuple[list, list, list]:
        """Divide *rows* as a single split is set to be, for the preview and the table alike."""
        return divide_rows(
            rows,
            train_ratio=self.training.train_ratio,
            seed=self.training.seed,
            validate_on=self.validate_on,
            earlier=self._earlier,
            balance=self.balance(),
        )

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
        self, output: Path, name: str, other: str | None = None, *, overwrite: bool = False
    ) -> Path:
        """Write the breakdown on screen as CSV, as ``am-seg stats --output`` does.

        Raises:
            ValueError: If a property it counts by needs the labels measured and
                they are not -- which would write every image under "no-label".
            FileExistsError: If *output* exists and *overwrite* is False.
        """
        unmeasured = [
            chosen for chosen in (name, other)
            if chosen and self.needs_measurements(chosen)
        ]
        if unmeasured and not self.measurements_loaded():
            raise ValueError(
                f"Counting by {' and '.join(unmeasured)} needs the labels measured first"
            )
        rows: Sequence[Sequence[object]]
        if other:
            header = [name, other, "images"]
            rows = cross_tabulation_as_rows(self.cross_tabulation(name, other))
        else:
            header = [name, "images"]
            rows = counts_as_rows(self.breakdown(name))
        write_csv(output, header, rows, overwrite=overwrite)
        return output

    def export_project_metadata(self, output: Path, *, overwrite: bool = False) -> Path:
        """Write one row per project, as ``am-seg export-projects-csv`` does.

        Raises:
            FileExistsError: If *output* exists and *overwrite* is False.
            UnreadableMetadataError: If a project's metadata file could not be read.
        """
        return export_project_metadata_csv(
            self.projects, self.start_path or output.parent, output, overwrite=overwrite
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
            if entry.kind not in ("measurement", "split")
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
