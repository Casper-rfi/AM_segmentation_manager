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
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .filtering import extract_image_number
from .models import AdaptiveMillingStage, ImageRecord, Lamella

logger = logging.getLogger(__name__)

MODALITIES = ("SEM", "FIB")

# Identity of an acquisition within a stage. Images are grouped by the number
# embedded in their filename, because the SEM and FIB files of one acquisition
# do not reliably share a stem: older projects name the file inside ``fib/``
# with the same ``_SEM`` suffix as the electron image. Files with no parseable
# number fall back to grouping by filename stem.
FrameKey = tuple[int | None, str]


@dataclass(frozen=True)
class LamellaFrame:
    """One acquisition point in a lamella's chronological milling history.

    A frame holds the SEM and FIB images of that acquisition, plus the summary
    plot when one was written. Any of them may be absent, and callers are
    expected to handle that rather than assume a complete set.

    Each modality is a *tuple* rather than a single record, because one
    acquisition can be captured under several imaging conditions -- five SEM
    images of one scene at different frame-integration settings, for example.
    The base image directory comes first, followed by variants in name order,
    so ``sem[0]`` is the primary image and :meth:`record` returns it by default.

    Attributes:
        lamella: Name of the lamella the frame belongs to.
        stage: Name of the stage folder the images came from.
        stage_number: Zero-padded stage number within the lamella.
        image_number: Acquisition number parsed from the filename, when present.
        sem: The SEM image records, base variant first. Empty if there are none.
        fib: The FIB image records, base variant first. Empty if there are none.
        plot_path: The summary plot for this acquisition, when one exists.
    """

    lamella: str
    stage: str
    stage_number: str | None
    image_number: int | None
    sem: tuple[ImageRecord, ...] = ()
    fib: tuple[ImageRecord, ...] = ()
    plot_path: Path | None = None

    @property
    def display_name(self) -> str:
        """Return a short user-facing label, e.g. ``'stage 02 - img 013'``."""
        stage_part = f"stage {self.stage_number}" if self.stage_number else self.stage
        if self.image_number is None:
            primary = next(iter(self.records), None)
            image_part = primary.image_path.stem if primary is not None else "?"
        else:
            image_part = f"img {self.image_number:03d}"
        return f"{stage_part} - {image_part}"

    @property
    def records(self) -> list[ImageRecord]:
        """Return every image record in this frame, SEM first."""
        return [*self.sem, *self.fib]

    @property
    def base_records(self) -> list[ImageRecord]:
        """Return one record per modality: the base image of each, SEM first.

        This is what a single-image view shows, and what an operation means by
        "this frame's SEM image" when it is not variant-aware.
        """
        return [records[0] for records in (self.sem, self.fib) if records]

    def modality_records(self, modality: str) -> tuple[ImageRecord, ...]:
        """Return every record for *modality*, base variant first."""
        normalised = modality.upper()
        if normalised == "SEM":
            return self.sem
        if normalised == "FIB":
            return self.fib
        raise ValueError(f"Unknown modality: {modality}")

    def record(self, modality: str, variant: str | None = None) -> ImageRecord | None:
        """Return one record for *modality*.

        Args:
            modality: ``'SEM'`` or ``'FIB'``.
            variant: The imaging-condition variant to return. ``None``, the
                default, returns the base image -- the first record, which is
                the unsuffixed image directory when there is one.

        Returns:
            The matching record, or None when the frame has no such image.
        """
        records = self.modality_records(modality)
        if variant is None:
            return records[0] if records else None
        return next((record for record in records if record.variant == variant), None)

    def variants(self, modality: str) -> list[str | None]:
        """Return the variants present for *modality*, base first."""
        return [record.variant for record in self.modality_records(modality)]


def build_lamella_frames(lamella: Lamella) -> list[LamellaFrame]:
    """Return the lamella's acquisitions as one chronologically ordered list.

    Frames follow the order of ``lamella.stages`` -- which the scanner has
    already sorted by the timestamp in each stage folder name -- and are ordered
    by acquisition number within each stage. Acquisitions whose filenames carry
    no number are placed after the numbered ones, ordered by filename.

    Args:
        lamella: A lamella from a scanned project index.

    Returns:
        One :class:`LamellaFrame` per acquisition, across all of the lamella's
        stages.
    """
    frames: list[LamellaFrame] = []
    for stage in lamella.stages:
        frames.extend(_build_stage_frames(lamella, stage))
    logger.debug("Built %d frames for lamella %s", len(frames), lamella.name)
    return frames


def _build_stage_frames(lamella: Lamella, stage: AdaptiveMillingStage) -> list[LamellaFrame]:
    """Return the frames for a single stage, ordered by acquisition number."""
    grouped = _group_records_by_acquisition(stage.images)
    plots = _plots_by_image_number(stage.plots)

    frames: list[LamellaFrame] = []
    for key in sorted(grouped, key=_frame_sort_key):
        image_number, _ = key
        modalities = grouped[key]
        frames.append(
            LamellaFrame(
                lamella=lamella.name,
                stage=stage.name,
                stage_number=stage.number,
                image_number=image_number,
                sem=tuple(sorted(modalities.get("SEM", []), key=_variant_sort_key)),
                fib=tuple(sorted(modalities.get("FIB", []), key=_variant_sort_key)),
                plot_path=plots.get(image_number) if image_number is not None else None,
            )
        )
    return frames


def _group_records_by_acquisition(
    records: Iterable[ImageRecord],
) -> dict[FrameKey, dict[str, list[ImageRecord]]]:
    """Group image records by acquisition, then by modality.

    One acquisition can have several records of the same modality, one per
    imaging condition, so every record is kept.
    """
    grouped: dict[FrameKey, dict[str, list[ImageRecord]]] = {}
    for record in records:
        modality = (record.modality or "").upper()
        if modality not in MODALITIES:
            logger.warning(
                "Skipping %s: unknown modality %r", record.image_path, record.modality
            )
            continue
        key = _frame_key(record.image_path)
        grouped.setdefault(key, {}).setdefault(modality, []).append(record)
    return grouped


def _variant_sort_key(record: ImageRecord) -> tuple[bool, str]:
    """Order the base image first, then variants by name."""
    return (record.variant is not None, (record.variant or "").lower())


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
    """Return the grouping key identifying an image's acquisition."""
    number = extract_image_number(image_path.name)
    if number is not None:
        return (number, "")
    return (None, image_path.stem)


def _frame_sort_key(key: FrameKey) -> tuple[bool, int, str]:
    """Order numbered acquisitions first, then unnumbered ones by filename."""
    number, stem = key
    return (number is None, number if number is not None else 0, stem)
