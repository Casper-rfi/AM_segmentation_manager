"""Measuring what a label contains.

Areas are reported in um^2, a pixel count not being comparable across
magnifications; :func:`find_specks` alone counts pixels, a tiny component being
suspect because of how it was made. **Labels carry no header**: the scale comes
from the image's metadata, rescaled by the label's own dimensions in
:func:`label_pixel_size_nm`, a label converted from a prediction not necessarily
sharing its image's size.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import BACKGROUND_CLASS, SegmentationConfig
from ..image_metadata import read_records_metadata
from ..io.image import has_image_stack_support, read_array, to_2d
from ..parallel import map_in_processes
from .csv import find_matching_label
from .label_groups import group_key, primary_record

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from collections.abc import Callable, Collection, Iterable, Mapping

    from ..image_metadata import ImageMetadata
    from ..models import ImageRecord

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SPECK_PIXELS",
    "MAX_SPECK_PIXELS",
    "MIN_SPECK_PIXELS",
    "count_specks_in",
    "PREDICTION_PIXEL_BLOCK",
    "LabelMeasurement",
    "LabelPixels",
    "clear_cache",
    "label_pixel_size_nm",
    "measure_label",
    "Speck",
    "find_specks",
    "measure_records",
    "measurement_status",
    "records_with_specks",
    "read_label_pixels",
    "read_many_label_pixels",
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


#: Components at or below this many pixels are suspect, wherever the question
#: is asked -- the report and the proofreading view share it.
#:
#: **Set from proofreading, not from reasoning about the data.** It began at
#: :data:`PREDICTION_PIXEL_BLOCK`, which is what could only have been painted
#: by hand; extended use of the speck filter found the components worth a
#: second glance run a good deal larger, and that some false positives are
#: worth the ones this catches.
DEFAULT_SPECK_PIXELS = 64

#: The smallest area a label converted from a prediction can hold, in pixels.
#:
#: Predictions are made at half resolution and upscaled by nearest neighbour,
#: so every component of such a label covers a multiple of four pixels -- which
#: is why anything below this was put there by hand. Not a threshold any more,
#: but the fact that explains what the low end of the slider means.
PREDICTION_PIXEL_BLOCK = 4

#: The range the size may be set to, in pixels. One pixel is the smallest thing
#: there is; the top is where "small component" stops meaning anything, which
#: proofreading has put further out than it first appeared. Here rather than on
#: a widget: two controls now offer this size, and a range owned by one of them
#: would be a dependency between two panels with nothing else in common.
MIN_SPECK_PIXELS = 1
MAX_SPECK_PIXELS = 128


@dataclass(frozen=True)
class Speck:
    """One connected component small enough to be worth questioning.

    Either a stray annotation, or -- where its class is background -- a hole of
    that size inside one. Not a finding of fact: a small structure can be real,
    which is why nothing here removes one. It is somewhere to look.

    Attributes:
        class_index: The class the component belongs to.
        pixels: How many pixels it covers.
        x: Column of its centroid, for bringing it into view.
        y: Row of its centroid.
    """

    class_index: int
    pixels: int
    x: int
    y: int


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


#: Fewer labels than this are counted in this process. One takes about 37 ms to
#: read and count, so sixteen cost about twice what starting the worker
#: processes does.
LABEL_POOL_FILES = 16


def read_many_label_pixels(
    paths: Iterable[Path],
    crack_class: int | None = None,
    *,
    use_cache: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[Path, LabelPixels | None]:
    """Count many label files, in worker processes where there are enough.

    :func:`read_label_pixels` for a batch, answering from and filling the same
    cache, and returning ``None`` for a label on the same terms.

    Args:
        on_progress: Called as ``on_progress(done, total)`` over the files
            actually read; one the cache answers is not counted.
    """
    paths = list(dict.fromkeys(paths))
    if not has_image_stack_support():
        logger.debug("Cannot measure %d label(s): tifffile/numpy/cv2 not installed", len(paths))
        return dict.fromkeys(paths)

    counted: dict[Path, LabelPixels | None] = {}
    keys: dict[Path, tuple[Path, int, int, int | None] | None] = {}
    for path in paths:
        key = _cache_key(path, crack_class) if use_cache else None
        cached = _CACHE.get(key) if key is not None else None
        if cached is not None:
            counted[path] = cached
        else:
            keys[path] = key

    unread = list(keys)
    results = map_in_processes(
        partial(_count_pixels, crack_class=crack_class),
        unread,
        chunk_size=1,
        min_items=LABEL_POOL_FILES,
        on_progress=on_progress,
    )
    for path, result in zip(unread, results):
        counted[path] = result
        if result is not None and (key := keys[path]) is not None:
            _CACHE[key] = result
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
    class_pixels = _class_pixels(array)

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


#: Largest pixel count a float32 histogram still represents exactly. cv2
#: returns float32, whose mantissa is 24 bits, so a label with more pixels than
#: this could come back with a count off by one -- and a pixel count that is
#: quietly approximate is worse than a slow one.
_EXACT_IN_FLOAT32 = 2 ** 24


def _class_pixels(array) -> dict[int, int]:
    """Return how many pixels each class index covers.

    Counting is the single most expensive step of measuring a label -- more
    than reading the file, and more than finding its connected components --
    because ``np.bincount`` widens the array to ``intp`` first, which allocates
    eight bytes a pixel before counting any of them. cv2's histogram counts the
    bytes where it can, which is a labelled TIFF's usual dtype.
    """
    import cv2
    import numpy as np

    if array.dtype == np.uint8 and array.size <= _EXACT_IN_FLOAT32:
        # calcHist will not take a view, and to_2d may well have handed us one.
        counted = cv2.calcHist(
            [np.ascontiguousarray(array)], [0], None, [256], [0, 256]
        ).ravel()
    else:
        counted = np.bincount(array.reshape(-1).astype(np.int64, copy=False))
    return {int(index): int(count) for index, count in enumerate(counted) if count}


def _components(mask) -> tuple[int, ...]:
    """Return the pixel size of each connected component of *mask*, largest first."""
    return tuple(sorted((size for size, _, _ in _component_stats(mask)), reverse=True))


def _component_stats(mask, connectivity: int = 8) -> list[tuple[int, int, int]]:
    """Return ``(pixels, x, y)`` for each connected component of *mask*.

    Eight-connected by default, so a component is not split by a one-pixel
    diagonal -- a crack is a physical feature, and a diagonal step across it is
    a rendering artefact of the pixel grid rather than a second crack. Holes
    are counted four-connected instead; see :func:`find_specks`.

    The centroid comes back with the size because the two callers want
    different halves of the same cv2 call: the crack statistics need only how
    big each component is, and finding specks needs somewhere to look.
    """
    import cv2
    import numpy as np

    if not mask.any():
        return []
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=connectivity
    )
    # Row 0 is the background of the mask, which is everything that is not the
    # class being measured.
    return [
        (
            int(stats[index, cv2.CC_STAT_AREA]),
            int(round(float(centroids[index][0]))),
            int(round(float(centroids[index][1]))),
        )
        for index in range(1, count)
    ]


def find_specks(
    label,
    *,
    at_most: int = DEFAULT_SPECK_PIXELS,
    ignore: Collection[int] = (),
) -> tuple[Speck, ...]:
    """Return every component of *label* covering at most *at_most* pixels.

    Background is included, and a small background component is a *hole*: the
    background of a frame is one enormous component, so the only way a few
    background pixels can stand alone is by being enclosed in an annotation.
    Both are errors and neither is more forgivable -- a stray pixel of GIS in
    the vacuum and a stray pixel of vacuum in the GIS are the same mistake seen
    from either side.

    Holes are counted **four-connected** while annotations are counted
    eight-connected, which is the pairing that makes an enclosed region stay
    enclosed. With both eight-connected, a diagonal staircase of single-pixel
    holes joins up and leaks out into the background around it, and stops being
    reported: in one real label that was the difference between finding seven
    and finding four.

    Takes the array rather than a path, so it can be asked of a label being
    edited as well as one on disk, and so it never consults the cache above --
    which is keyed by what a *file* looked like.

    Args:
        label: 2D array of class indices.
        at_most: Largest component still reported, in pixels.
        ignore: Classes not worth reporting. Nothing by default.

    Returns:
        The specks found, smallest first, then by position. Ordered so that
        stepping through them is stable and starts with the least defensible.
    """
    import numpy as np

    ignored = set(ignore)
    found = [
        Speck(class_index=int(class_index), pixels=pixels, x=x, y=y)
        for class_index in np.unique(label)
        if int(class_index) not in ignored
        for pixels, x, y in _component_stats(
            label == class_index,
            connectivity=4 if int(class_index) == BACKGROUND_CLASS else 8,
        )
        if pixels <= at_most
    ]
    return tuple(sorted(found, key=lambda speck: (speck.pixels, speck.y, speck.x)))


def count_specks_in(path: Path, *, at_most: int = DEFAULT_SPECK_PIXELS) -> int:
    """Return how many specks one label file holds, reading only that file.

    For marking a whole series at once, where the question is "how many" per
    frame rather than where each one is. A label that will not open counts
    none: the label line already reports it as unreadable, and a count is not
    the place to say so a second time.
    """

    try:
        array = to_2d(read_array(path))
    except Exception as exc:  # noqa: BLE001 - one bad label must not stop a sweep
        logger.warning("Could not read label %s: %s", path, exc)
        return 0
    return len(find_specks(array, at_most=at_most))


def records_with_specks(
    records: Iterable[ImageRecord],
    *,
    at_most: int = DEFAULT_SPECK_PIXELS,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[Path, tuple[Speck, ...]]:
    """Return the specks in every label the records point at, keyed by label.

    Keyed by the label rather than by the image, so a label shared by several
    imaging conditions is read once and reported once -- it is one
    segmentation, and it is the file somebody has to go and fix.

    Labels that do not exist, and labels that cannot be read, are absent: this
    reports what was found, and :func:`classify_labels` is what reports what is
    missing.

    Args:
        on_progress: Called as ``on_progress(done, total)`` after each label,
            counted in labels rather than in records: the conditions of one
            scene share a file and cost one read between them, so a counter
            over records would sit still for several at a time.
    """

    records = list(records)
    labels = [
        label for label in dict.fromkeys(
            find_matching_label(record) for record in records
        ) if label is not None
    ]
    found: dict[Path, tuple[Speck, ...]] = {}
    for done, label in enumerate(labels, start=1):
        try:
            array = to_2d(read_array(label))
        except Exception as exc:  # noqa: BLE001 - an unreadable label must not stop the sweep
            logger.warning("Could not read label %s: %s", label, exc)
        else:
            specks = find_specks(array, at_most=at_most)
            if specks:
                found[label] = specks
        if on_progress is not None:
            # Reported for a label that would not open as well as one that did:
            # it was looked at, and a counter that skipped it would never reach
            # its total.
            on_progress(done, len(labels))
    return found


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
    counts = read_label_pixels(path, config.crack_class, use_cache=use_cache)
    if counts is None:
        return None
    return _measurement_of(path, counts, metadata, config, min_crack_area_um2)


def _measurement_of(
    path: Path,
    counts: LabelPixels,
    metadata: ImageMetadata | None,
    config: SegmentationConfig,
    min_crack_area_um2: float | None,
) -> LabelMeasurement:
    """Return what the pixel counts of the label at *path* amount to in physical units."""
    threshold = (
        config.min_crack_area_um2 if min_crack_area_um2 is None else float(min_crack_area_um2)
    )
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

def measurement_is_empty(measurement: LabelMeasurement) -> bool:
    """Return whether a measured label holds nothing but background.

    The same question :func:`~am_segmentation_manager.display.is_empty_label`
    answers off the pixels, answered instead off the counts a measurement
    already carries. Measuring opens the label anyway, so a caller that has
    measured need never read it a second time to find out whether there is
    anything in it.
    """

    return not any(
        count
        for index, count in measurement.pixels.class_pixels.items()
        if index != BACKGROUND_CLASS
    )


def measure_records(
    records: Iterable[ImageRecord],
    config: SegmentationConfig,
    metadata: Mapping[Path, ImageMetadata] | None = None,
    *,
    min_crack_area_um2: float | None = None,
    use_cache: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
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
        on_progress: Called as ``on_progress(done, total)`` over the label
            files read. Counted in files, not records, because the conditions
            of one scene share a label and are measured once.

    Returns:
        Image path to the measurement of that image's label.
    """

    records = list(records)
    groups: dict[str, list[ImageRecord]] = {}
    for record in records:
        groups.setdefault(group_key(record), []).append(record)

    if metadata is None:

        # Only the primary of each group is ever consulted for scale, so only
        # those headers are read. On a collection with five conditions per
        # scene that is a fifth of the reads.
        primaries = [primary_record(members) for members in groups.values()]
        metadata = read_records_metadata(primaries)

    labels = {
        key: find_matching_label(primary_record(members)) for key, members in groups.items()
    }
    counts = read_many_label_pixels(
        (label for label in labels.values() if label is not None),
        config.crack_class,
        use_cache=use_cache,
        on_progress=on_progress,
    )

    measurements: dict[Path, LabelMeasurement] = {}
    for key, members in groups.items():
        label = labels[key]
        if label is None or (pixels := counts.get(label)) is None:
            continue
        primary = primary_record(members)
        _warn_on_inconsistent_scale(members, metadata)
        measured = _measurement_of(
            label, pixels, metadata.get(primary.image_path), config, min_crack_area_um2
        )
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
