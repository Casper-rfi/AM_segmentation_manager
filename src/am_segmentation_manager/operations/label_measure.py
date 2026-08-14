"""Measuring what a label contains, in physical units.

This is the first thing in the package that depends on both a label's pixels
and its image's header: a pixel count is not comparable between two images at
different magnifications, and comparing across the archive is the point. Areas
are reported in um^2.

**Labels carry no header of their own.** A label is written by segmentation
software from an image, and inherits nothing from it, so the scale comes from
the corresponding image's :class:`~am_segmentation_manager.image_metadata.ImageMetadata`
and is rescaled by the label's own pixel dimensions -- see
:func:`label_pixel_size_nm`. A label converted from a prediction may be smaller
than its image, and getting that scaling wrong produces areas that are wrong by
a constant factor, which is the hardest kind of error to notice in a
distribution.

**Two layers, deliberately.** :func:`read_label_pixels` does the expensive,
unit-free half -- read the file, count class pixels, label the crack class's
connected components -- and :func:`measure_label` converts the result to
physical units and applies the size threshold. Only the first half is cached,
so changing the threshold does not invalidate anything, and the threshold stays
ordinary arithmetic over numbers already computed.

**Scope.** Per-class area, crack count, crack size distribution. Shape
descriptors, morphology and position are analysis, and analysis belongs in a
script reading the exported CSV.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import BACKGROUND_CLASS, SegmentationConfig
from ..io.image import has_image_stack_support

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from collections.abc import Iterable, Mapping

    from ..image_metadata import ImageMetadata
    from ..models import ImageRecord

logger = logging.getLogger(__name__)

__all__ = [
    "LabelMeasurement",
    "LabelPixels",
    "clear_cache",
    "label_pixel_size_nm",
    "measure_label",
    "measure_records",
    "measurement_status",
    "read_label_pixels",
]

#: Pixel counts already computed, keyed by the file's identity *and* the
#: property of the file that was measured. Labelling components in a 3072x2048
#: image is milliseconds to seconds, and a command that groups by two crack
#: statistics would otherwise read the archive twice.
#:
#: The key carries mtime and size so an edited label is re-read: proofreading
#: happens while this process may still be running, and a stale area is worse
#: than a slow one. It is in memory only -- a cache on disk would be exactly the
#: hidden persistent state this package avoids, and for repeated analysis the
#: answer is to export the measurements once and work from the CSV.
_CACHE: dict[tuple[Path, int, int, int | None], LabelPixels] = {}


@dataclass(frozen=True)
class LabelPixels:
    """What one label holds, counted in pixels.

    Free of physical units and of the crack threshold, which is what makes it
    cacheable: nothing here changes when the user passes a different
    ``--min-crack-area``.

    Attributes:
        width: Label width in pixels.
        height: Label height in pixels.
        class_pixels: Pixel count per class index, for every class present.
            Includes background, so the counts sum to the label's area.
        component_pixels: Size in pixels of each connected component of the
            crack class, largest first. Empty when the config names no crack
            class, which is distinct from a label having no cracks -- the
            distinction is preserved by :attr:`crack_class`.
        crack_class: The class index components were found for, or ``None``
            when the config names none.
    """

    width: int
    height: int
    class_pixels: dict[int, int]
    component_pixels: tuple[int, ...] = ()
    crack_class: int | None = None

    @property
    def pixels(self) -> int:
        """Return the label's total area in pixels."""
        return self.width * self.height

    @property
    def is_empty(self) -> bool:
        """Return whether the label holds nothing but background.

        The segmentation software writes an empty label for every image it is
        pointed at, so this is not the same as "nothing was found": it is the
        difference between a proofread label with no cracks in it and a
        placeholder nobody ever annotated. Averaging the two together would
        understate how much of the archive holds real annotation.
        """
        return not set(self.class_pixels) - {BACKGROUND_CLASS}


@dataclass(frozen=True)
class LabelMeasurement:
    """What one label holds, in physical units.

    Attributes:
        path: The label file measured.
        pixels: The underlying pixel counts, kept so a caller can see the raw
            numbers without re-reading the file.
        pixel_size_nm: Edge length of one *label* pixel in nanometres, already
            rescaled from the image's. ``None`` when the image header did not
            record enough to derive it, in which case no area could be
            computed -- see :func:`label_pixel_size_nm`.
        areas_um2: Area per class index. Empty when *pixel_size_nm* is None.
        crack_count: Components of the crack class at or above the threshold.
            ``None`` when the areas are unknown (the threshold is physical, so
            it cannot be applied to pixels) or when no crack class is
            configured. Zero means measured, and no crack found.
        crack_areas_um2: Area of each qualifying component, largest first.
        min_crack_area_um2: The threshold that was applied, recorded so an
            exported measurement says what it counted.
    """

    path: Path
    pixels: LabelPixels
    pixel_size_nm: float | None
    areas_um2: dict[int, float]
    crack_count: int | None
    crack_areas_um2: tuple[float, ...]
    min_crack_area_um2: float

    @property
    def has_crack(self) -> bool:
        """Return whether at least one component met the threshold.

        False when the label could not be measured, so a caller that asks only
        this question never mistakes an unmeasurable label for an intact one.
        Check :attr:`crack_count` for ``None`` where the difference matters.
        """
        return bool(self.crack_count)


#: The states a label's measurement can be in. Kept as one vocabulary because
#: three callers need it -- the CSV export writes it as a column, the
#: breakdowns group by it, and the roll-up counts each separately -- and they
#: must not drift. Each is a different finding: an archive with no labels, one
#: full of placeholders the segmentation software wrote, and one this package
#: cannot put a scale on are three different problems.
MEASURED = "measured"
EMPTY = "empty"
NO_PIXEL_SIZE = "no-pixel-size"
NO_LABEL = "no-label"


def measurement_status(measurement: LabelMeasurement | None) -> str:
    """Return which of the four states a label's measurement is in.

    ``None`` stands for a record whose label does not exist, which is why this
    is a function rather than a property.
    """
    if measurement is None:
        return NO_LABEL
    if measurement.pixels.is_empty:
        return EMPTY
    if measurement.crack_count is None:
        return NO_PIXEL_SIZE
    return MEASURED


def clear_cache() -> None:
    """Discard cached pixel counts.

    Only needed by tests and by a long-running process that has moved on to a
    different collection; ordinary CLI runs are short enough not to care.
    """
    _CACHE.clear()


def read_label_pixels(
    path: Path, crack_class: int | None = None, *, use_cache: bool = True
) -> LabelPixels | None:
    """Count class pixels and crack components in one label file.

    Args:
        path: The label file to read.
        crack_class: Class index to find connected components of. ``None``
            skips component labelling, which is the expensive part.
        use_cache: Whether to consult and populate the module cache.

    Returns:
        The counts, or ``None`` when the answer cannot be determined: the
        optional image stack is not installed, or the file is missing or
        unreadable. ``None`` is deliberately distinct from an all-background
        result, so an unreadable label is never counted as an empty one.
    """
    if not has_image_stack_support():
        logger.debug("Cannot measure %s: tifffile/numpy/cv2 not installed", path)
        return None

    key: tuple[Path, int, int, int | None] | None = None
    if use_cache:
        key = _cache_key(path, crack_class)
        if key is not None:
            cached = _CACHE.get(key)
            if cached is not None:
                return cached

    counted = _count_pixels(path, crack_class)
    if counted is not None and key is not None:
        _CACHE[key] = counted
    return counted


def _cache_key(path: Path, crack_class: int | None) -> tuple[Path, int, int, int | None] | None:
    """Return a cache key identifying this file's current contents."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return (path, stat.st_mtime_ns, stat.st_size, crack_class)


def _count_pixels(path: Path, crack_class: int | None) -> LabelPixels | None:
    """Read *path* and count it, or return None if it cannot be read."""
    import numpy as np

    from ..io.image import read_array, to_2d

    try:
        array = to_2d(read_array(path))
    except Exception as exc:  # noqa: BLE001 - an unreadable label must not raise here
        logger.warning("Could not read label %s: %s", path, exc)
        return None

    if not np.issubdtype(array.dtype, np.integer):
        logger.warning(
            "Label %s has non-integer values (%s); class indices are discrete, "
            "so this is not a label mask",
            path,
            array.dtype,
        )
        return None
    if np.issubdtype(array.dtype, np.signedinteger) and array.min() < 0:
        logger.warning("Label %s holds negative class indices; not measuring it", path)
        return None

    height, width = int(array.shape[0]), int(array.shape[1])
    counts = np.bincount(array.reshape(-1).astype(np.int64, copy=False))
    class_pixels = {
        int(index): int(count) for index, count in enumerate(counts) if count
    }

    component_pixels: tuple[int, ...] = ()
    if crack_class is not None:
        component_pixels = _components(array == crack_class)

    return LabelPixels(
        width=width,
        height=height,
        class_pixels=class_pixels,
        component_pixels=component_pixels,
        crack_class=crack_class,
    )


def _components(mask) -> tuple[int, ...]:
    """Return the pixel size of each connected component of *mask*, largest first.

    Eight-connected, so a component is not split by a one-pixel diagonal --
    a crack is a physical feature, and a diagonal step across it is a rendering
    artefact of the pixel grid rather than a second crack.
    """
    import cv2
    import numpy as np

    if not mask.any():
        return ()
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    # Row 0 is the background of the mask, which is everything that is not the
    # class being measured.
    areas = [int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)]
    return tuple(sorted(areas, reverse=True))


def label_pixel_size_nm(metadata: ImageMetadata | None, label_width: int) -> float | None:
    """Return the edge length of one label pixel, in nanometres.

    A label has no header. It covers the same field of view as the image it was
    drawn from, so its pixel size is the image's scaled by the ratio of their
    widths -- a label half as wide has pixels twice the size.

    Two routes, in order of directness:

    1. ``pixel_size_nm x (image_width / label_width)``.
    2. ``hfw_um x 1000 / label_width``, when the header records a field width
       but no image width. This needs nothing from the image's pixel grid.

    Returns:
        The label's pixel size, or ``None`` when the header records neither
        route. ``None`` rather than the image's own pixel size: assuming a
        label matches its image would silently scale every area by a constant
        factor whenever it does not.
    """
    if metadata is None or label_width <= 0:
        return None

    if metadata.pixel_size_nm and metadata.image_width:
        return metadata.pixel_size_nm * (metadata.image_width / label_width)

    if metadata.hfw_um:
        return metadata.hfw_um * 1000.0 / label_width

    return None


def measure_label(
    path: Path,
    metadata: ImageMetadata | None,
    config: SegmentationConfig,
    *,
    min_crack_area_um2: float | None = None,
    use_cache: bool = True,
) -> LabelMeasurement | None:
    """Measure one label in physical units.

    Args:
        path: The label file.
        metadata: Header of the image the label belongs to, which is where the
            scale comes from. ``None`` yields a measurement with no areas
            rather than no measurement at all, so an unreadable header is
            visible in the output.
        config: Supplies the crack class and the default threshold.
        min_crack_area_um2: Overrides the configured threshold, for a
            ``--min-crack-area`` flag.
        use_cache: Whether to consult and populate the module cache.

    Returns:
        The measurement, or ``None`` when the label itself could not be read.
    """
    threshold = (
        config.min_crack_area_um2 if min_crack_area_um2 is None else float(min_crack_area_um2)
    )
    counts = read_label_pixels(path, config.crack_class, use_cache=use_cache)
    if counts is None:
        return None

    pixel_size = label_pixel_size_nm(metadata, counts.width)
    if pixel_size is None:
        logger.debug("No pixel size for %s; reporting counts without areas", path)
        return LabelMeasurement(
            path=path,
            pixels=counts,
            pixel_size_nm=None,
            areas_um2={},
            crack_count=None,
            crack_areas_um2=(),
            min_crack_area_um2=threshold,
        )

    # (nm)^2 to um^2 is a factor of 1e-6, applied once here rather than at each
    # use of the result.
    area_per_pixel_um2 = (pixel_size * pixel_size) * 1e-6
    areas = {
        index: count * area_per_pixel_um2 for index, count in counts.class_pixels.items()
    }

    crack_count: int | None = None
    crack_areas: tuple[float, ...] = ()
    if counts.crack_class is not None:
        crack_areas = tuple(
            area
            for area in (size * area_per_pixel_um2 for size in counts.component_pixels)
            if area >= threshold
        )
        crack_count = len(crack_areas)

    return LabelMeasurement(
        path=path,
        pixels=counts,
        pixel_size_nm=pixel_size,
        areas_um2=areas,
        crack_count=crack_count,
        crack_areas_um2=crack_areas,
        min_crack_area_um2=threshold,
    )


# ---------------------------------------------------------------------------
# Measuring a set of records
#
# The label is the unit, not the image: several imaging conditions of one
# acquisition share a single label, because they are a single segmentation.
# Measuring per image would read that label once per condition and, worse,
# count one crack several times in any total.
# ---------------------------------------------------------------------------

def measure_records(
    records: Iterable[ImageRecord],
    config: SegmentationConfig,
    metadata: Mapping[Path, ImageMetadata] | None = None,
    *,
    min_crack_area_um2: float | None = None,
    use_cache: bool = True,
) -> dict[Path, LabelMeasurement]:
    """Measure the label of every record, once per label.

    Keyed by image path, mirroring
    :func:`~am_segmentation_manager.image_metadata.read_records_metadata`, so
    :mod:`am_segmentation_manager.aggregate` can close over it the same way and
    never open a file itself. Records sharing a label map to the *same*
    measurement object, which is how a caller deduplicates: group by
    :attr:`LabelMeasurement.path`, or compare by identity.

    A record whose label does not exist is absent from the mapping. That is
    distinct from a label that could not be read, which is present with no
    areas -- "never annotated" and "unreadable" are different findings.

    Args:
        records: The images to measure, already filtered.
        config: Supplies the crack class and the default threshold.
        metadata: Image path to header, from ``read_records_metadata``. Read
            here for the images that need it when not supplied.
        min_crack_area_um2: Overrides the configured threshold.
        use_cache: Whether to consult and populate the module cache.

    Returns:
        Image path to the measurement of that image's label.
    """
    from .csv import find_matching_label
    from .label_groups import group_key, primary_record

    records = list(records)
    groups: dict[str, list[ImageRecord]] = {}
    for record in records:
        groups.setdefault(group_key(record), []).append(record)

    if metadata is None:
        from ..image_metadata import read_records_metadata

        # Only the primary of each group is ever consulted for scale, so only
        # those headers are read. On a collection with five conditions per
        # scene that is a fifth of the reads.
        primaries = [primary_record(members) for members in groups.values()]
        metadata = read_records_metadata(primaries)

    measurements: dict[Path, LabelMeasurement] = {}
    for members in groups.values():
        primary = primary_record(members)
        label = find_matching_label(primary)
        if label is None:
            continue

        _warn_on_inconsistent_scale(members, metadata)
        measured = measure_label(
            label,
            metadata.get(primary.image_path),
            config,
            min_crack_area_um2=min_crack_area_um2,
            use_cache=use_cache,
        )
        if measured is None:
            continue
        for record in members:
            measurements[record.image_path] = measured

    return measurements


def _warn_on_inconsistent_scale(
    members: list[ImageRecord], metadata: Mapping[Path, ImageMetadata]
) -> None:
    """Warn when images sharing a label disagree about their pixel size.

    They are conditions of one acquisition, so they should agree, and the
    measurement takes the primary's. A disagreement means the group's areas are
    right for one condition and wrong for the others -- worth saying out loud
    rather than resolving silently, since which one is correct is a question
    about the data, not about this code.
    """
    sizes = {
        metadata[record.image_path].pixel_size_nm
        for record in members
        if record.image_path in metadata
        and metadata[record.image_path].pixel_size_nm is not None
    }
    if len(sizes) > 1:
        logger.warning(
            "Images sharing one label disagree about pixel size (%s nm); "
            "measuring at the primary condition's scale: %s",
            ", ".join(f"{size:g}" for size in sorted(sizes)),
            members[0].image_path.parent.parent,
        )
