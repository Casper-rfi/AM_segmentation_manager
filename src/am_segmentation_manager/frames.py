"""Per-lamella chronological view over scanned image records.

The scanner produces one :class:`~am_segmentation_manager.models.ImageRecord` per
*file*, so an acquisition's SEM and FIB images are separate records. Reviewing a
lamella needs the opposite arrangement: one entry per *acquisition*, holding both
modalities and the summary plot, in the order taken.

:class:`LamellaFrame` is that arrangement. This module performs no filesystem
access -- it is a pure transformation of the scanned model.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .layout import (
    NO_CORRECTIONS,
    SERIES,
    UNSEQUENCED,
    FamilyCorrections,
    acquisition_stem,
    alignment_run_order,
    condition_of,
    condition_order,
    extract_image_number,
)
from .models import ImageRecord, Lamella, Stage

logger = logging.getLogger(__name__)

MODALITIES = ("SEM", "FIB")

# Identity of an acquisition within a stage. Adaptive Polishing images are
# grouped by the number embedded in their filename, because the SEM and FIB
# files of one acquisition do not reliably share a stem: older projects name
# the file inside ``fib/`` with the same ``_SEM`` suffix as the electron image.
# Reference and alignment filenames carry no number and differ only by their
# beam suffix, so those group by the stem with the suffix removed.
FrameKey = tuple[int | None, str]

#: Identity of an acquisition within its *lamella*, which spans several stages
#: and so needs the stage as well as :data:`FrameKey`. Stable across a rescan,
#: unlike a frame's position; see :attr:`LamellaFrame.key`.
FrameIdentity = tuple[str, int | None, str]


@dataclass(frozen=True)
class LamellaFrame:
    """One acquisition point in a lamella's chronological milling history.

    A frame holds the SEM and FIB images of that acquisition, plus the summary
    plot when one was written. Any of them may be absent, and callers are
    expected to handle that rather than assume a complete set.

    Each modality is a *tuple* rather than a single record, because one
    acquisition can be captured under several imaging conditions -- five SEM
    images of one scene at different frame-integration settings, for example.
    They are in :func:`~.layout.condition_order`: by family, each base before
    its variants. What a view opens on is the first family's primary, which the
    project's corrections may name; see :meth:`shown_record`.

    Attributes:
        lamella: Name of the lamella the frame belongs to.
        stage: Name of the stage folder the images came from.
        stage_number: Zero-padded stage number within the lamella.
        image_number: Acquisition number parsed from the filename, when present.
        sem: The SEM image records, in condition order. Empty if there are none.
        fib: The FIB image records, in condition order. Empty if there are none.
        plot_path: The summary plot for this acquisition, when one exists.
        stage_plot: The plot describing the whole stage, when it wrote one.
            Shared by every frame of that stage; see :attr:`displayed_plot`.
        series: What the images were taken for. One value for the frame: see
            :attr:`Acquisition.series`.
        corrections: The project's family corrections, which say what family
            a condition chosen by name belongs to and which condition is each
            family's primary.
    """

    lamella: str
    stage: str
    stage_number: str | None
    image_number: int | None
    sem: tuple[ImageRecord, ...] = ()
    fib: tuple[ImageRecord, ...] = ()
    plot_path: Path | None = None
    stage_plot: Path | None = None
    series: str = "adaptive"
    corrections: FamilyCorrections = field(
        default=NO_CORRECTIONS, compare=False, repr=False
    )

    @property
    def displayed_plot(self) -> Path | None:
        """Return the plot to show for this frame, of either kind.

        Its own where the stage wrote one per acquisition, which is every
        Adaptive Polishing stage that has plots at all. Otherwise the stage's,
        which is what an alignment run has: one figure summarising the several
        images it took to converge, with nothing to match to a single
        acquisition. :attr:`plot_is_shared` says which this is, because a
        figure describing five frames should not be read as describing one.
        """
        return self.plot_path or self.stage_plot

    @property
    def plot_is_shared(self) -> bool:
        """Return whether the plot on show describes the stage, not this frame."""
        return self.plot_path is None and self.stage_plot is not None

    @property
    def acquired_at(self) -> datetime | None:
        """Return when this frame was taken, if its images say.

        The earliest of its records', so the answer does not depend on which
        beams happen to be present. ``None`` for an Adaptive Polishing frame,
        whose images are ordered by the number in their filenames instead.
        """
        taken = [r.acquired_at for r in self.records if r.acquired_at is not None]
        return min(taken) if taken else None

    @property
    def display_name(self) -> str:
        """Return a short user-facing label, e.g. 'stage 02 - img 013'."""
        stage_part = f"stage {self.stage_number}" if self.stage_number else self.stage
        if self.image_number is None:
            primary = next(iter(self.records), None)
            image_part = primary.image_path.stem if primary is not None else "?"
        else:
            image_part = f"img {self.image_number:03d}"
        return f"{stage_part} - {image_part}"

    @property
    def key(self) -> FrameIdentity:
        """Return what identifies this acquisition within its lamella.

        What a frame *is*, rather than where it sits. A position is only
        meaningful against one build of the list: an image added or removed
        shifts every index after it, so anything that has to name a frame
        across a rescan names it with this.
        """
        primary = next(iter(self.records), None)
        return (
            self.stage,
            self.image_number,
            primary.image_path.stem if primary is not None else "",
        )

    @property
    def records(self) -> list[ImageRecord]:
        """Return every image record in this frame, SEM first."""
        return [*self.sem, *self.fib]

    @property
    def base_records(self) -> list[ImageRecord]:
        """Return one record per modality, SEM first: the first family's primary.

        What a single-image view opens on, and what an operation means by "this
        frame's SEM image" when it is not condition-aware.
        """
        return [
            record
            for record in (self.primary("SEM"), self.primary("FIB"))
            if record is not None
        ]

    def modality_records(self, modality: str) -> tuple[ImageRecord, ...]:
        """Return every record for *modality*, in condition order."""
        normalised = modality.upper()
        if normalised == "SEM":
            return self.sem
        if normalised == "FIB":
            return self.fib
        raise ValueError(f"Unknown modality: {modality}")

    def record(self, modality: str, condition: str | None = None) -> ImageRecord | None:
        """Return one record for *modality*.

        Args:
            modality: ``'SEM'`` or ``'FIB'``.
            condition: The imaging condition to return, by its directory's name
                in any case. ``None``, the default, returns the first family's
                primary.

        Returns:
            The matching record, or None when the frame has no such image.
        """
        if condition is None:
            return self.primary(modality)
        wanted = condition.lower()
        return next(
            (
                record
                for record in self.modality_records(modality)
                if (record.condition or "").lower() == wanted
            ),
            None,
        )

    def families(self, modality: str) -> list[str]:
        """Return the families *modality* was imaged under here, in condition order.

        Each once, however its conditions spell it; ``""`` for records naming
        no family.
        """
        seen: dict[str, str] = {}
        for record in self.modality_records(modality):
            seen.setdefault(family_key(record.family), record.family or "")
        return list(seen.values())

    def primary(self, modality: str, family: str | None = None) -> ImageRecord | None:
        """Return the record standing for one family of *modality*.

        Args:
            family: The family, in any case. ``None`` means the first family.

        Returns:
            The family's primary, as :func:`primary_among` chooses it, or None
            when the frame holds no image of that family.
        """
        records = self.modality_records(modality)
        if not records:
            return None
        wanted = family_key(family if family is not None else records[0].family)
        members = [record for record in records if family_key(record.family) == wanted]
        return primary_among(members, self.corrections) if members else None

    def shown_record(self, modality: str, condition: str | None = None) -> ImageRecord | None:
        """Return the record a view shows for *modality* with *condition* chosen.

        **The one fallback, for every view and the cache.** The chosen
        condition; where this frame lacks it, the primary of the family it
        belongs to, so stepping past an acquisition taken under fewer
        conditions stays in the family; where the frame lacks the family too,
        the first family's primary, so the pane is never blank while the
        frame has an image of the beam.
        """
        if condition is not None:
            exact = self.record(modality, condition)
            if exact is not None:
                return exact
            parsed = condition_of(condition, self.corrections)
            if parsed is not None and parsed.family is not None:
                same_family = self.primary(modality, parsed.family)
                if same_family is not None:
                    return same_family
        return self.primary(modality)

    def condition_choices(self, modality: str) -> list[ConditionChoice]:
        """Return each condition of *modality* a view can show, in condition order."""
        return [
            ConditionChoice(
                family=record.family or "", variant=record.variant, condition=record.condition
            )
            for record in self.modality_records(modality)
        ]


@dataclass(frozen=True)
class ConditionChoice:
    """One imaging condition a view can show, as a control offers it.

    Attributes:
        family: The family it belongs to, as its records spell it; ``""`` for
            an image of none.
        variant: Its variant within the family, or None for the base.
        condition: Its image directory's name, which choosing it passes on;
            None for an image sitting directly in its stage.
    """

    family: str
    variant: str | None
    condition: str | None

    @property
    def label(self) -> str:
        """Return what it is called within its family: its variant, or ``base``."""
        return self.variant if self.variant is not None else "base"

    @property
    def name(self) -> str:
        """Return what it is called among several families, as ``sem-5kV [fi4]``.

        The family alone for its base, as two families would otherwise each
        offer a ``base``.
        """
        if not self.family:
            return self.label
        return self.family if self.variant is None else f"{self.family} [{self.variant}]"


def family_key(family: str | None) -> str:
    """Return what identifies a family in comparisons: its name in lower case.

    Family names compare case-insensitively, as a stage's directories do.
    """
    return (family or "").lower()


def series_present(lamella: Lamella | None) -> list[str]:
    """Return the series *lamella* holds images of, in display order.

    Read off the records rather than from the vocabulary, so a view offering
    these never offers one that would show an empty timeline.
    """
    if lamella is None:
        return []
    present = {record.series for record in lamella.image_records}
    return [name for name in SERIES if name in present]


def resolve_series(wanted: str, present: Sequence[str]) -> str:
    """Return the series to show: *wanted* where the lamella holds it.

    **What was chosen and what a lamella can show are two things**, and this
    answers only the second. A lamella holding no Adaptive Polishing images
    falls back to one it does hold; *wanted* is untouched, which is what lets
    the next lamella that has the chosen series open on it. A caller that
    stored the answer back would leave the reviewer on whatever the emptiest
    lamella in the collection happened to hold.

    Args:
        wanted: The series chosen, returned unchanged when *present* is empty:
            a lamella with no images at all settles nothing.
        present: What the lamella holds, in display order.
    """
    if not present or wanted in present:
        return wanted
    return present[0]


def primary_among(
    records: Sequence[ImageRecord], corrections: FamilyCorrections
) -> ImageRecord:
    """Return the record standing for one family's *records*: its primary condition.

    The condition *corrections* name for the family where it is among
    *records*; otherwise the base condition, otherwise the first variant in
    case-insensitive string order, which is what the oldest layouts need,
    having no base: their SEM sets are ``SEM_8``, ``SEM_8_fi_1`` and so on.

    Raises:
        ValueError: If *records* is empty.
    """
    if not records:
        raise ValueError("A family with no records has no primary")
    family = next((record.family for record in records if record.family), None)
    named = corrections.primary_of(family) if family is not None else None
    if named is not None:
        for record in records:
            if (record.condition or "").lower() == named.lower():
                return record
    return min(records, key=_base_first)


def _base_first(record: ImageRecord) -> tuple[bool, str]:
    """Order one family's records: the base, then variants by name."""
    return record.variant is not None, (record.variant or "").lower()


def build_lamella_frames(
    lamella: Lamella,
    series: str | None = None,
    *,
    corrections: FamilyCorrections,
) -> list[LamellaFrame]:
    """Return the lamella's acquisitions as one chronologically ordered list.

    Frames follow the order of ``lamella.stages`` -- which the scanner has
    already sorted by the timestamp in each stage folder name -- and are ordered
    by acquisition number within each stage. Acquisitions whose filenames carry
    no number are placed after the numbered ones, ordered by filename.

    Args:
        lamella: A lamella from a scanned project index.
        series: Keep only acquisitions of this series. ``None`` keeps every
            one. A 2026 lamella holds its reference images, its alignment runs
            and its Adaptive Polishing stages, which is one timeline of three
            unrelated things; anything reviewing it wants one at a time.
        corrections: The corrections of the project holding *lamella*, which
            every frame carries.

    Returns:
        One :class:`LamellaFrame` per acquisition, across all of the lamella's
        stages.
    """
    frames: list[LamellaFrame] = []
    for stage in lamella.stages:
        frames.extend(_build_stage_frames(lamella, stage, series, corrections))
    frames = _in_acquisition_order(frames)
    logger.debug("Built %d frames for lamella %s", len(frames), lamella.name)
    return frames


def _in_acquisition_order(frames: list[LamellaFrame]) -> list[LamellaFrame]:
    """Return the frames in the order they were taken.

    **Where every frame knows when it was taken, that is the order** -- which
    is every reference and alignment timeline, and the only thing that puts one
    right. Their stages interleave in time: a lamella directory holds reference
    images spanning the whole session while the milling tasks below it hold
    theirs in the middle, so no position for that stage in a stage-by-stage
    list is the correct one.

    Adaptive Polishing frames carry no acquisition time, because their
    filenames carry the number that orders them within a stage and their stage
    folders are timestamped -- both free, where a time costs opening the image.
    Those keep the stage-and-number order they were built in, which is already
    the order they were taken in.
    """
    if not all(frame.acquired_at is not None for frame in frames):
        return frames
    return sorted(
        frames,
        key=lambda frame: (frame.acquired_at, frame.stage, _sequence(frame), frame.key),
    )


def _sequence(frame: LamellaFrame) -> int:
    """Return a frame's place in its alignment run, for breaking a tie.

    A header records whole seconds and an alignment run takes several images
    within one, so acquisition time alone leaves them in filename order --
    which reads ``final_image`` before the last ``new_image``.
    """
    primary = next(iter(frame.records), None)
    return (
        alignment_run_order(primary.image_path.name)
        if primary is not None
        else UNSEQUENCED
    )


def _build_stage_frames(
    lamella: Lamella,
    stage: Stage,
    series: str | None,
    corrections: FamilyCorrections,
) -> list[LamellaFrame]:
    """Return the frames for a single stage, ordered by acquisition number."""
    plots = _plots_by_image_number(stage.plots)
    return [
        LamellaFrame(
            lamella=lamella.name,
            stage=stage.name,
            stage_number=stage.number,
            image_number=acquisition.image_number,
            sem=acquisition.sem,
            fib=acquisition.fib,
            plot_path=(
                plots.get(acquisition.image_number)
                if acquisition.image_number is not None
                else None
            ),
            stage_plot=stage.summary_plot,
            series=acquisition.series,
            corrections=corrections,
        )
        for acquisition in group_by_acquisition(stage.images)
        if series is None or acquisition.series == series
    ]


@dataclass(frozen=True)
class Acquisition:
    """One acquisition: both beams' images of a single moment, every condition.

    What :class:`LamellaFrame` is for the timeline, without the names a lamella
    view needs -- so the same grouping serves an export over records drawn from
    anywhere in the collection.

    Attributes:
        stage_directory: The stage the images sit in. Part of the identity
            because acquisition numbers restart in every stage.
        key: The acquisition's identity within that stage.
        sem: SEM records, base condition first. Empty if there are none.
        fib: FIB records, base condition first. Empty if there are none.
    """

    stage_directory: Path
    key: FrameKey
    sem: tuple[ImageRecord, ...] = ()
    fib: tuple[ImageRecord, ...] = ()

    @property
    def image_number(self) -> int | None:
        """Return the acquisition number, or None where the name carries none."""
        return self.key[0]

    @property
    def identity(self) -> str:
        """Return a stable name for this acquisition, across the collection.

        What keeps an acquisition's rows on one side of a training split: its
        beams and every imaging condition of them are one scene, and splitting
        them would put near-identical images in both sets.
        """
        number, stem = self.key
        return f"{self.stage_directory}|{number if number is not None else stem}"

    @property
    def records(self) -> list[ImageRecord]:
        """Return every record in this acquisition, SEM first."""
        return [*self.sem, *self.fib]

    @property
    def acquired_at(self) -> datetime | None:
        """Return when this acquisition was taken, if its images say.

        The earliest of its records', because the two beams of one moment are
        seconds apart and either would do; taking the earliest makes the answer
        independent of which beams happen to be present.
        """
        taken = [r.acquired_at for r in self.records if r.acquired_at is not None]
        return min(taken) if taken else None

    @property
    def series(self) -> str:
        """Return what this acquisition's images were taken for.

        One value for the whole acquisition: a series is decided by filename,
        and the beam suffix plays no part in that decision, so both beams and
        every imaging condition of one scene classify the same way.
        """
        first = next(iter(self.records), None)
        return first.series if first is not None else "adaptive"


def group_by_acquisition(records: Iterable[ImageRecord]) -> list[Acquisition]:
    """Return one entry per acquisition, holding both beams' images.

    **The one statement of what makes an SEM and a FIB image the same
    acquisition.** They are separate records and do not reliably share a
    filename stem -- older projects name the file inside ``fib/`` with the same
    ``_SEM`` suffix as the electron image -- so they are matched by the number
    in the filename, within one stage directory. Records with no parseable
    number fall back to their filename stem.

    Takes records from anywhere rather than one stage, so an export can group a
    filtered selection spanning the collection. Ordered by stage directory,
    then by acquisition number with the unnumbered ones after, so a CSV written
    twice from one selection comes out the same both times.

    Records of an unrecognised modality are dropped: a frame is a moment seen
    by one beam or both, and nothing here can say which one these belong to.
    An image with **no** beam is the expected case and is logged at debug --
    the ``other`` series holds every image whose beam neither its filename nor
    its header states, and warning once per image on every rebuild would teach
    a reader to ignore this log. A modality that is a *value* and not one of
    :data:`~.layout.MODALITIES` is a defect somewhere above and still warns.
    """
    grouped: dict[tuple[Path, FrameKey], dict[str, list[ImageRecord]]] = {}
    for record in records:
        modality = (record.modality or "").upper()
        if modality not in MODALITIES:
            report = logger.debug if record.modality is None else logger.warning
            report(
                "Skipping %s: unknown modality %r", record.image_path, record.modality
            )
            continue
        # A scanned record carries its stage. Falling back to the image
        # directory's parent covers a record built by hand, and is right for
        # exactly the case it still applies to: an Adaptive Polishing image
        # sits one level below its stage, in a directory naming a modality, so
        # conditions of one scene group together. A loose image sits directly
        # in its stage, where the same derivation would name the lamella's
        # parent and scatter one stage's images across the collection.
        stage = record.stage_path or record.image_path.parent.parent
        identity = (stage, _frame_key(record.image_path))
        grouped.setdefault(identity, {}).setdefault(modality, []).append(record)

    acquisitions = [
        Acquisition(
            stage_directory=directory,
            key=key,
            sem=tuple(sorted(modalities.get("SEM", []), key=_condition_sort_key)),
            fib=tuple(sorted(modalities.get("FIB", []), key=_condition_sort_key)),
        )
        for (directory, key), modalities in grouped.items()
    ]
    acquisitions.sort(key=_acquisition_sort_key)
    return acquisitions


def _condition_sort_key(record: ImageRecord) -> tuple[tuple[str, bool, str], str]:
    """Order a beam's records by family, base first, then by variant."""
    return condition_order(record.family, record.variant), (record.condition or "").lower()


def _plots_by_image_number(plots: Iterable[Path]) -> dict[int, Path]:
    """Map acquisition number to summary plot for the plots of one stage."""
    by_number: dict[int, Path] = {}
    for plot in plots:
        number = extract_image_number(plot.name)
        if number is None:
            logger.debug("Ignoring plot with no acquisition number: %s", plot)
            continue
        by_number.setdefault(number, plot)
    return by_number


def _frame_key(image_path: Path) -> FrameKey:
    """Return the grouping key identifying an image's acquisition.

    The acquisition number where the filename carries one, which is every
    Adaptive Polishing image. Outside those there is no number, and the two
    beams differ only by the suffix -- ``ref_Polishing_start_eb`` against
    ``ref_Polishing_start_ib`` -- so the stem is taken with that removed.
    """
    number = extract_image_number(image_path.name)
    if number is not None:
        return (number, "")
    return (None, acquisition_stem(image_path.name))


def _acquisition_sort_key(
    acquisition: Acquisition,
) -> tuple[str, bool, int, datetime, str]:
    """Order acquisitions within a stage, stages by directory.

    A numbered acquisition is placed by its number: that is what an Adaptive
    Polishing filename carries, it is already acquisition order, and it costs
    nothing to read. An unnumbered one is placed by **when it was taken**,
    which is the only thing that orders a reference or alignment image --
    alphabetical order put ``Setup Lamella Position`` last of four tasks when
    it is the first, and read an alignment run as ``final, new_00..02,
    reference`` when it was ``reference, new_00..02, final``.

    The filename is the last resort, for an acquisition whose images carry
    neither a number nor a readable header.
    """
    number, stem = acquisition.key
    return (
        str(acquisition.stage_directory),
        number is None,
        number if number is not None else 0,
        acquisition.acquired_at or datetime.max,
        stem,
    )
