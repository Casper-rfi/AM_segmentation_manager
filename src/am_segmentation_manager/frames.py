"""Per-lamella chronological view over scanned image records.

The scanner produces one :class:`~am_segmentation_manager.models.ImageRecord`
per *file*: the SEM image and the FIB image of the same acquisition are separate
records, spread across the stages of a lamella. Reviewing a lamella needs the
opposite arrangement -- one entry per *acquisition*, holding both modalities and
the summary plot, ordered as the images were taken.

:class:`LamellaFrame` is that arrangement, and :func:`build_lamella_frames`
derives it from an already-scanned :class:`~am_segmentation_manager.models.Lamella`.
This module performs no filesystem access: it is a pure transformation of the
scanned model, so it stays cheap to call and straightforward to test.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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

    A frame holds whichever of the SEM and FIB images exist for that
    acquisition, plus the summary plot when one was written. Any of the three
    may be absent, and callers are expected to handle that rather than assume a
    complete set.

    Attributes:
        lamella: Name of the lamella the frame belongs to.
        stage: Name of the stage folder the images came from.
        stage_number: Zero-padded stage number within the lamella.
        image_number: Acquisition number parsed from the filename, when present.
        sem: The SEM image record, or None if this acquisition has no SEM image.
        fib: The FIB image record, or None if this acquisition has no FIB image.
        plot_path: The summary plot for this acquisition, when one exists.
    """

    lamella: str
    stage: str
    stage_number: str | None
    image_number: int | None
    sem: ImageRecord | None = None
    fib: ImageRecord | None = None
    plot_path: Path | None = None

    @property
    def display_name(self) -> str:
        """Return a short user-facing label, e.g. ``'stage 02 - img 013'``."""
        stage_part = f"stage {self.stage_number}" if self.stage_number else self.stage
        if self.image_number is None:
            name = self.sem or self.fib
            image_part = name.image_path.stem if name is not None else "?"
        else:
            image_part = f"img {self.image_number:03d}"
        return f"{stage_part} - {image_part}"

    @property
    def records(self) -> list[ImageRecord]:
        """Return the image records present in this frame, SEM first."""
        return [record for record in (self.sem, self.fib) if record is not None]

    def record(self, modality: str) -> ImageRecord | None:
        """Return the record for *modality* (``'SEM'`` or ``'FIB'``), if any."""
        normalised = modality.upper()
        if normalised == "SEM":
            return self.sem
        if normalised == "FIB":
            return self.fib
        raise ValueError(f"Unknown modality: {modality}")


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
                sem=modalities.get("SEM"),
                fib=modalities.get("FIB"),
                plot_path=plots.get(image_number) if image_number is not None else None,
            )
        )
    return frames


def _group_records_by_acquisition(
    records: Iterable[ImageRecord],
) -> dict[FrameKey, dict[str, ImageRecord]]:
    """Group image records by acquisition, then by modality."""
    grouped: dict[FrameKey, dict[str, ImageRecord]] = {}
    for record in records:
        modality = (record.modality or "").upper()
        if modality not in MODALITIES:
            logger.warning(
                "Skipping %s: unknown modality %r", record.image_path, record.modality
            )
            continue
        key = _frame_key(record.image_path)
        slot = grouped.setdefault(key, {})
        if modality in slot:
            # Legacy layouts can hold several images of one acquisition taken
            # under different conditions. Showing the first is better than
            # dropping the acquisition, but the extras must not pass silently.
            logger.warning(
                "Multiple %s images for acquisition %s in this stage; showing %s and ignoring %s",
                modality, key[0] if key[0] is not None else key[1],
                slot[modality].image_path.name, record.image_path.name,
            )
            continue
        slot[modality] = record
    return grouped


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
