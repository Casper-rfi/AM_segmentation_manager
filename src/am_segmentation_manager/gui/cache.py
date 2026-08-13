"""Display data held in memory for the lamella currently being reviewed.

Reading an acquisition from disk takes ~200 ms cold and a summary plot about
the same, so navigating a lamella is only smooth if frames are already in memory
by the time they are asked for. This module holds what has been prepared and
decides what to prepare next; it does not know how work is scheduled, so the
same object serves synchronous tests and a background worker thread.

Sizing follows measurements on real data. A prepared acquisition and its label
cost ~1.4 MB, so a lamella's images fit comfortably in memory. Plots are
prepared at the display's physical resolution -- which is what keeps their text
crisp -- so they cost between 2.3 MB and 9 MB each depending on the screen, and
their limit is derived from a memory budget rather than fixed.

Loading is delegated to injected callables. Tests supply fakes and never touch
the filesystem; the GUI supplies the real readers from
:mod:`am_segmentation_manager.display`.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generic, Iterable, Sequence, TypeVar

from ..display import (
    DEFAULT_PLOT_MAX_EDGE,
    DisplayImage,
    prepare_display_image,
    prepare_plot,
)
from ..frames import LamellaFrame

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")

#: Prepared acquisitions to keep. At ~1.4 MB each this bounds the cache at
#: roughly 170 MB, which covers a normal lamella outright and keeps the
#: neighbourhood of the current frame for an unusually long one.
DEFAULT_IMAGE_LIMIT = 120

#: Starting number of prepared plots to keep, before a display size is known.
#: Large enough to hold a whole lamella: a plot takes ~200 ms to decode, which
#: no amount of prefetching can hide from someone holding down an arrow key, so
#: the only way to navigate plots smoothly is to have the lamella already in
#: memory. Replaced by :func:`_plot_limit_for` once the pane size is known.
DEFAULT_PLOT_LIMIT = 64

#: How far either side of the current frame to prepare ahead.
DEFAULT_RADIUS = 3

#: Radius used for plots while the plot view is on: the whole lamella. Plots
#: are otherwise not prepared at all, so this costs nothing when they are not
#: being looked at.
PLOT_RADIUS = None

#: Roughly how much memory prepared plots may occupy. A plot is prepared at the
#: display's physical resolution, so its cost varies fourfold between a normal
#: and a high-resolution screen; the entry limit is derived from this budget
#: rather than fixed, so a typical lamella stays cached on either.
PLOT_MEMORY_BUDGET_BYTES = 320 * 1024 * 1024

#: Bounds on the derived plot limit: enough to navigate comfortably at the
#: bottom, and never so many that an unusually long lamella exhausts memory.
MIN_PLOT_LIMIT = 8
MAX_PLOT_LIMIT = 64

#: Loads one acquisition and its label into display form.
ImageLoader = Callable[[Path, "Path | None"], DisplayImage]

#: Loads one summary plot into display form.
PlotLoader = Callable[[Path], "np.ndarray"]


@dataclass(frozen=True)
class FrameImage:
    """A frame's prepared acquisition, or why it could not be prepared.

    A file that cannot be read -- a corrupt TIFF, or a cloud placeholder that
    never materialised -- is recorded rather than raised, so that one bad file
    does not stop the user navigating past it. The failure is cached too: a slow
    failing read should not be retried on every repaint.
    """

    display: DisplayImage | None = None
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


@dataclass(frozen=True)
class FramePlot:
    """A frame's prepared summary plot, or why it could not be prepared."""

    plot: "np.ndarray | None" = None
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


@dataclass(frozen=True)
class LoadRequest:
    """A single unit of preparation work.

    ``modality`` names the acquisition to prepare, or is None to request the
    frame's summary plot.
    """

    frame_index: int
    modality: str | None = None


class BoundedCache(Generic[K, V]):
    """A least-recently-used cache with a maximum number of entries.

    Reading or writing an entry marks it as recently used, so eviction removes
    whatever is furthest from where the user is working.

    Every operation is guarded by a lock. Reading marks an entry as used, and
    so mutates the ordering: with a background thread preparing frames while
    the interface reads them, unguarded access could corrupt the ordering. The
    lock covers only the bookkeeping, never a load.
    """

    def __init__(self, max_entries: int | None = None) -> None:
        """Create a cache holding at most *max_entries* items (None: unbounded)."""
        if max_entries is not None and max_entries < 1:
            raise ValueError(f"max_entries must be at least 1, got {max_entries}")
        self.max_entries = max_entries
        self._entries: OrderedDict[K, V] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: K) -> V | None:
        """Return the cached value for *key*, or None, marking it as used."""
        with self._lock:
            if key not in self._entries:
                return None
            self._entries.move_to_end(key)
            return self._entries[key]

    def put(self, key: K, value: V) -> None:
        """Store *value*, evicting the least recently used entry if needed."""
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while self.max_entries is not None and len(self._entries) > self.max_entries:
                evicted, _ = self._entries.popitem(last=False)
                logger.debug("Evicted %s from cache", evicted)

    def clear(self) -> None:
        """Discard every entry."""
        with self._lock:
            self._entries.clear()

    def keys(self) -> list[K]:
        """Return the cached keys, least recently used first."""
        with self._lock:
            return list(self._entries)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._entries

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def _plot_limit_for(width: int, height: int) -> int:
    """Return how many plots of this size fit within the memory budget.

    Prepared plots are three bytes a pixel, and their size follows the screen:
    the same pane costs four times as much on a high-resolution display. A
    fixed entry count would therefore mean either wasting memory on an ordinary
    screen or exhausting it on a good one.
    """
    per_plot = max(1, width * height * 3)
    limit = PLOT_MEMORY_BUDGET_BYTES // per_plot
    return int(max(MIN_PLOT_LIMIT, min(limit, MAX_PLOT_LIMIT)))


def prefetch_order(current: int, total: int, radius: int | None = None) -> list[int]:
    """Return frame indices to prepare, nearest to *current* first.

    The order alternates forward and back -- ``current``, ``current + 1``,
    ``current - 1``, ``current + 2`` ... -- because a user moving through a
    lamella may reverse at any point, and the frame just behind them is as
    likely to be wanted as the one ahead.

    Args:
        current: The frame being viewed.
        total: Number of frames in the lamella.
        radius: How far either side to include; None includes every frame.

    Returns:
        Valid frame indices, without duplicates, nearest first.
    """
    if total <= 0:
        return []
    current = max(0, min(current, total - 1))
    reach = total if radius is None else radius

    order = [current]
    for distance in range(1, reach + 1):
        for index in (current + distance, current - distance):
            if 0 <= index < total:
                order.append(index)
    return order


class FrameCache:
    """Prepared images and plots for the lamella currently being reviewed.

    The cache is scoped to one lamella: :meth:`set_frames` discards everything
    held for the previous one, because moving to another lamella makes all of it
    irrelevant at once.
    """

    def __init__(
        self,
        load_image: ImageLoader | None = None,
        load_plot: PlotLoader | None = None,
        *,
        image_limit: int | None = DEFAULT_IMAGE_LIMIT,
        plot_limit: int | None = DEFAULT_PLOT_LIMIT,
    ) -> None:
        """Create a cache, optionally with custom loaders.

        Args:
            load_image: Called as ``load_image(image_path, label_path)``.
                Defaults to the real display pipeline.
            load_plot: Called as ``load_plot(plot_path)``.
            image_limit: Maximum prepared acquisitions to keep.
            plot_limit: Maximum prepared plots to keep.
        """
        self._load_image: ImageLoader = load_image or prepare_display_image
        self._load_plot: PlotLoader = load_plot or self._prepare_plot_at_display_size
        self.plot_max_size = (DEFAULT_PLOT_MAX_EDGE, DEFAULT_PLOT_MAX_EDGE)
        self._frames: tuple[LamellaFrame, ...] = ()
        self._images: BoundedCache[tuple[int, str], FrameImage] = BoundedCache(image_limit)
        self._plots: BoundedCache[int, FramePlot] = BoundedCache(plot_limit)
        self._generation = 0

    def _prepare_plot_at_display_size(self, path: Path) -> "np.ndarray":
        """Read a plot at whatever size it is currently being displayed."""
        return prepare_plot(path, max_size=self.plot_max_size)

    def set_plot_max_size(self, size: tuple[int, int], *, tolerance: float = 0.2) -> bool:
        """Set the size plots are prepared at, discarding any prepared otherwise.

        *size* is in physical screen pixels. A plot prepared at any other size
        has to be resampled again when it is drawn, which softens the small text
        these figures are full of.

        Changes smaller than *tolerance* are ignored, so that dragging a window
        edge does not throw the whole lamella away repeatedly.

        Returns:
            Whether the size changed and prepared plots were discarded.
        """
        width, height = size
        if width <= 0 or height <= 0:
            return False
        current_width, current_height = self.plot_max_size
        change = max(
            abs(width - current_width) / current_width,
            abs(height - current_height) / current_height,
        )
        if change < tolerance:
            return False
        logger.debug("Plot display size %s -> %s", self.plot_max_size, size)
        self.plot_max_size = (width, height)
        self._plots.max_entries = _plot_limit_for(width, height)
        self._plots.clear()
        return True

    @property
    def frames(self) -> tuple[LamellaFrame, ...]:
        """Return the frames this cache is currently serving."""
        return self._frames

    @property
    def generation(self) -> int:
        """Return a counter identifying the current set of frames.

        Incremented by :meth:`set_frames`. A load that began before the frames
        changed is discarded rather than stored, so switching lamella while
        preparation is in flight cannot leave the previous lamella's data
        indexed against the new one.
        """
        return self._generation

    def set_frames(self, frames: Sequence[LamellaFrame]) -> None:
        """Serve *frames*, discarding everything prepared for the previous set."""
        self._frames = tuple(frames)
        self._generation += 1
        self.clear()
        logger.debug("Cache now serving %d frames", len(self._frames))

    def clear(self) -> None:
        """Discard all prepared data, keeping the current frames."""
        self._images.clear()
        self._plots.clear()

    def image(self, frame_index: int, modality: str) -> FrameImage | None:
        """Return the prepared acquisition, reading it if not already held.

        Returns None when the frame has no image for *modality*, which is a
        normal outcome rather than a failure -- not every acquisition has both.
        """
        record = self._record(frame_index, modality)
        if record is None:
            return None

        key = (frame_index, modality)
        cached = self._images.get(key)
        if cached is not None:
            return cached

        label_path = record.labels[0] if record.labels else None
        generation = self._generation
        try:
            prepared = FrameImage(display=self._load_image(record.image_path, label_path))
        except Exception as exc:  # noqa: BLE001 - a bad file must not block navigation
            logger.warning("Could not prepare %s: %s", record.image_path, exc)
            prepared = FrameImage(error=str(exc))
        if generation == self._generation:
            self._images.put(key, prepared)
        return prepared

    def plot(self, frame_index: int) -> FramePlot | None:
        """Return the prepared summary plot, reading it if not already held.

        Returns None when the frame has no plot, which is normal: stages
        routinely hold fewer plots than acquisitions.
        """
        frame = self._frame(frame_index)
        if frame is None or frame.plot_path is None:
            return None

        cached = self._plots.get(frame_index)
        if cached is not None:
            return cached

        generation = self._generation
        try:
            prepared = FramePlot(plot=self._load_plot(frame.plot_path))
        except Exception as exc:  # noqa: BLE001 - as above
            logger.warning("Could not prepare plot %s: %s", frame.plot_path, exc)
            prepared = FramePlot(error=str(exc))
        if generation == self._generation:
            self._plots.put(frame_index, prepared)
        return prepared

    def cached_image(self, frame_index: int, modality: str) -> FrameImage | None:
        """Return a prepared acquisition only if it is already held.

        Lets a caller distinguish "not ready yet" from "nothing to show", so a
        view can say *loading* rather than blocking on a slow read.
        """
        return self._images.get((frame_index, modality))

    def cached_plot(self, frame_index: int) -> FramePlot | None:
        """Return a prepared plot only if it is already held."""
        return self._plots.get(frame_index)

    def ensure(self, request: LoadRequest) -> None:
        """Prepare the data named by *request* if it is not already held."""
        if request.modality is None:
            self.plot(request.frame_index)
        else:
            self.image(request.frame_index, request.modality)

    def pending(
        self,
        current: int,
        modalities: Iterable[str],
        *,
        radius: int | None = DEFAULT_RADIUS,
        include_plots: bool = False,
    ) -> list[LoadRequest]:
        """Return outstanding work, in the order it should be done.

        The current frame comes first, then its neighbours outwards. Anything
        already held -- including a previous failure -- is left out.

        Args:
            current: The frame being viewed.
            modalities: Which acquisitions to prepare, in priority order.
            radius: How far either side of *current* to prepare ahead.
            include_plots: Also prepare summary plots. Off by default: plots
                cost several times more than images and are only worth
                preparing while the plot view is on.

        Returns:
            Requests for data that is not yet held.
        """
        wanted = list(modalities)
        requests: list[LoadRequest] = []
        for frame_index in prefetch_order(current, len(self._frames), radius):
            for modality in wanted:
                if self._record(frame_index, modality) is None:
                    continue
                if (frame_index, modality) in self._images:
                    continue
                requests.append(LoadRequest(frame_index, modality))
            if include_plots:
                frame = self._frame(frame_index)
                if frame is not None and frame.plot_path is not None:
                    if frame_index not in self._plots:
                        requests.append(LoadRequest(frame_index, None))
        return requests

    def _frame(self, frame_index: int) -> LamellaFrame | None:
        """Return the frame at *frame_index*, or None if out of range."""
        if 0 <= frame_index < len(self._frames):
            return self._frames[frame_index]
        return None

    def _record(self, frame_index: int, modality: str):
        """Return the image record for a frame and modality, if it exists."""
        frame = self._frame(frame_index)
        if frame is None:
            return None
        return frame.record(modality)
