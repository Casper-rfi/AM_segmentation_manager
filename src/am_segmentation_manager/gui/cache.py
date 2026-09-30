"""Display data held in memory for the lamella currently being reviewed.

Holds what has been prepared and decides what to prepare next, so frames are in
memory before they are asked for. It does not know how work is scheduled, so the
same object serves synchronous tests and a background worker thread.

Plot limits are derived from a memory budget rather than fixed, because plots
are prepared at the display's physical resolution and so cost between 2.3 MB and
9 MB each depending on the screen.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

from ..display import (
    DEFAULT_PLOT_MAX_EDGE,
    DisplayImage,
    prepare_display_image,
    prepare_plot,
)
from ..frames import LamellaFrame
from ..io.atomic import Fingerprint, fingerprint_of

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")

#: Prepared acquisitions to keep. At ~1.4 MB each this bounds the cache at
#: roughly 170 MB, which covers a normal lamella outright and keeps the
#: neighbourhood of the current frame for an unusually long one.
DEFAULT_IMAGE_LIMIT = 120

#: Starting number of prepared plots to keep, before one has been read. Large
#: enough to hold a whole lamella: a plot decodes more slowly than an arrow key
#: repeats, which no amount of prefetching can hide from someone holding it
#: down, so the only way to navigate plots smoothly is to have the lamella
#: already in memory.
#: Replaced by :func:`_plot_limit_for_bytes` once a plot has been read and its
#: real cost is known.
DEFAULT_PLOT_LIMIT = 64

#: How far either side of the current frame to prepare ahead.
DEFAULT_RADIUS = 3

#: Plots are not prepared at all unless the plot view is on, so reading ahead
#: for them costs nothing when they are not being looked at. How far ahead is
#: :attr:`FrameCache.plot_radius`, which is as far as the cache can keep.
#:
#: A plot is as slow to read however it is asked for -- a PNG has to be
#: decompressed whole, so the reduced-decode flags buy nothing -- and slower
#: than a held-down arrow key. Nothing can be read fast enough to keep
#: up, so every plot has to be there already. That is what bounds these entries
#: to the pane rather than to the file.

#: Roughly how much memory prepared plots may occupy. A plot is held at the
#: resolution it was rendered at, which the package does not control -- these
#: run to several thousand pixels across, and one collection's are not another's
#: -- so the entry limit is derived from this budget rather than fixed.
PLOT_MEMORY_BUDGET_BYTES = 320 * 1024 * 1024

#: Bounds on the derived plot limit: enough to navigate comfortably at the
#: bottom, and never so many that an unusually long lamella exhausts memory.
MIN_PLOT_LIMIT = 8
MAX_PLOT_LIMIT = 64

#: Plots kept at the resolution they were rendered at. Only what is being
#: looked at and what was just left, because these are near 40 MB each: they
#: exist so that stopping on a frame and zooming in shows the figure rather
#: than a magnified copy of the pane, which is a different job from stepping.
DETAIL_PLOT_LIMIT = 2

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
    #: What the label file looked like just before its pixels were read, or
    #: None where there was none to read. Taken before rather than after, so a
    #: rewrite during the read counts as a change.
    label_fingerprint: Fingerprint | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


@dataclass(frozen=True)
class FramePlot:
    """A frame's prepared summary plot, or why it could not be prepared."""

    plot: np.ndarray | None = None
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


@dataclass(frozen=True)
class LoadRequest:
    """A single unit of preparation work.

    ``modality`` names the acquisition to prepare, or is None to request the
    frame's summary plot. ``condition`` names the imaging condition, or is None
    for the frame's first of that modality.

    ``detail`` asks for the plot at the resolution it was rendered at rather
    than at the size of the pane. Only ever asked for the frame being looked
    at, and always last: it is the slowest thing here and nothing is waiting
    on it.
    """

    frame_index: int
    modality: str | None = None
    condition: str | None = None
    detail: bool = False


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

    def drop(self, keys: Iterable[K]) -> int:
        """Discard the entries at *keys*, and return how many went."""
        with self._lock:
            gone = 0
            for key in list(keys):
                if self._entries.pop(key, None) is not None:
                    gone += 1
            return gone

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


def _plot_limit_for_bytes(per_plot: int) -> int:
    """Return how many plots of this size fit within the memory budget.

    Measured from one that was actually read rather than from the size they
    were asked for: they are held as rendered, and how large that is belongs to
    whatever wrote them. A fixed entry count would mean either wasting memory
    on small figures or exhausting it on large ones.
    """
    limit = PLOT_MEMORY_BUDGET_BYTES // max(1, per_plot)
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
        self._load_plot: PlotLoader = load_plot or self._prepare_plot_for_the_pane
        #: The size plots are prepared at for stepping through, in physical
        #: screen pixels. Set by the window from the pane it draws them in.
        self.plot_max_size = (DEFAULT_PLOT_MAX_EDGE, DEFAULT_PLOT_MAX_EDGE)
        self._frames: tuple[LamellaFrame, ...] = ()
        self._images: BoundedCache[tuple[int, str, str | None], FrameImage] = BoundedCache(image_limit)
        self._plots: BoundedCache[int, FramePlot] = BoundedCache(plot_limit)
        self._details: BoundedCache[int, FramePlot] = BoundedCache(DETAIL_PLOT_LIMIT)
        self._generation = 0

    @property
    def plot_radius(self) -> int | None:
        """Return how far either side of a frame plots are worth preparing.

        As far as this cache can keep, and no further. Asking for a whole
        lamella that does not fit is the cache working against itself: the read
        that finishes last evicts the one that finished first, so a step in
        either direction lands on something that was prepared and then thrown
        away. None where nothing is evicted at all.
        """
        held = self._plots.max_entries
        return None if held is None else max(1, (held - 1) // 2)

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
        self._details.clear()

    def relabel(self, written: Mapping[Path, np.ndarray]) -> int:
        """Put the labels in *written* into the frames already prepared.

        A prepared frame holds the pixels its label had when it was read, so
        one written since would go on being drawn as what it replaced. The
        array is put in rather than the frame dropped, because dropping it
        leaves nothing to draw until a background read catches up -- a blank
        canvas where a save had just been made.

        Each frame takes the label's fingerprint as it now is, so that what it
        holds and what it says it read stay one moment.

        Several imaging conditions of one scene share a label, so one written
        label can reach more than one prepared frame. A frame whose label had
        to be resampled to be displayed is left alone: its array is not the
        one on disk, which is also why such a frame cannot be edited.

        Returns:
            How many prepared frames were brought up to date.
        """
        wanted = {Path(path).resolve(): array for path, array in written.items()}
        if not wanted:
            return 0
        fingerprints = {path: fingerprint_of(path) for path in wanted}
        changed = 0
        for key in self._images.keys():
            prepared = self._images.get(key)
            record = self._record(*key)
            if prepared is None or record is None or prepared.display is None:
                continue
            if prepared.display.label_resized:
                continue
            for label in record.labels:
                array = wanted.get(label.resolve())
                if array is None:
                    continue
                self._images.put(
                    key,
                    replace(
                        prepared,
                        display=replace(prepared.display, label=array),
                        label_fingerprint=fingerprints[label.resolve()],
                    ),
                )
                changed += 1
                break
        return changed

    def image(
        self, frame_index: int, modality: str, condition: str | None = None
    ) -> FrameImage | None:
        """Return the prepared acquisition, reading it if not already held.

        Returns None when the frame has no image for *modality* and *condition*,
        which is a normal outcome rather than a failure -- not every
        acquisition has both modalities, and not every one has every condition.

        Entries are keyed by condition as well as modality, so switching between
        two imaging conditions to compare them does not evict either.
        """
        record = self._record(frame_index, modality, condition)
        if record is None:
            return None

        key = (frame_index, modality, record.condition)
        cached = self._images.get(key)
        if cached is not None:
            return cached

        label_path = record.labels[0] if record.labels else None
        fingerprint = fingerprint_of(label_path) if label_path is not None else None
        generation = self._generation
        try:
            prepared = FrameImage(
                display=self._load_image(record.image_path, label_path),
                label_fingerprint=fingerprint,
            )
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
        if frame is None or frame.displayed_plot is None:
            return None

        cached = self._plots.get(frame_index)
        if cached is not None:
            return cached

        generation = self._generation
        try:
            prepared = FramePlot(plot=self._load_plot(frame.displayed_plot))
        except Exception as exc:  # noqa: BLE001 - as above
            logger.warning("Could not prepare plot %s: %s", frame.displayed_plot, exc)
            prepared = FramePlot(error=str(exc))
        else:
            self._budget_for(prepared)
        if generation == self._generation:
            self._plots.put(frame_index, prepared)
        return prepared

    def _prepare_plot_for_the_pane(self, path: Path) -> np.ndarray:
        """Read a plot reduced to the size it is displayed at."""
        return prepare_plot(path, max_size=self.plot_max_size)

    def set_plot_max_size(self, size: tuple[int, int], *, tolerance: float = 0.2) -> bool:
        """Set the size plots are prepared at, discarding any prepared otherwise.

        *size* is in physical screen pixels. A plot prepared at any other size
        has to be resampled again when it is drawn, which softens the small
        text these figures are full of.

        Changes smaller than *tolerance* are ignored, so that dragging a window
        edge does not throw the whole lamella away repeatedly. The
        full-resolution copies are untouched: what the pane is does not change
        what the file holds.

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
        self._plots.max_entries = _plot_limit_for_bytes(width * height * 3)
        self._plots.clear()
        return True

    def _budget_for(self, prepared: FramePlot) -> None:
        """Set how many plots to keep, from the size of one that was read.

        Their size is not this package's to choose, so it is measured rather
        than assumed -- and measured on every read, because one lamella's plots
        need not be the size of another's.
        """
        # Asked of the array rather than required of it: the loader is
        # injectable, and this is bookkeeping rather than something the cache
        # needs to understand about what it is holding.
        nbytes = getattr(prepared.plot, "nbytes", 0)
        if not nbytes:
            return
        limit = _plot_limit_for_bytes(int(nbytes))
        if limit != self._plots.max_entries:
            logger.debug(
                "Plots are %.1f MB each; keeping %d", nbytes / (1024 * 1024), limit
            )
            self._plots.max_entries = limit

    def cached_image(
        self, frame_index: int, modality: str, condition: str | None = None
    ) -> FrameImage | None:
        """Return a prepared acquisition only if it is already held.

        Lets a caller distinguish "not ready yet" from "nothing to show", so a
        view can say *loading* rather than blocking on a slow read.
        """
        key = self._key(frame_index, modality, condition)
        return self._images.get(key) if key is not None else None

    def drop_image(
        self, frame_index: int, modality: str, condition: str | None = None
    ) -> bool:
        """Discard a prepared acquisition, so it is read again; return whether one was held."""
        key = self._key(frame_index, modality, condition)
        return key is not None and self._images.drop([key]) > 0

    def cached_plot(self, frame_index: int) -> FramePlot | None:
        """Return a prepared plot only if it is already held."""
        return self._plots.get(frame_index)

    def plot_detail(self, frame_index: int) -> FramePlot | None:
        """Return the plot at the resolution it was rendered at, reading it if needed.

        Separate from :meth:`plot` because the two answer different questions.
        Stepping through a lamella needs every plot small enough to be held;
        stopping on one and looking into it needs that one whole. Reading is
        as slow either way, so this is asked for only where it will be
        looked at.
        """
        frame = self._frame(frame_index)
        if frame is None or frame.displayed_plot is None:
            return None

        cached = self._details.get(frame_index)
        if cached is not None:
            return cached

        generation = self._generation
        try:
            prepared = FramePlot(plot=prepare_plot(frame.displayed_plot))
        except Exception as exc:  # noqa: BLE001 - as above
            logger.warning("Could not read %s whole: %s", frame.displayed_plot, exc)
            prepared = FramePlot(error=str(exc))
        if generation == self._generation:
            self._details.put(frame_index, prepared)
        return prepared

    def cached_plot_detail(self, frame_index: int) -> FramePlot | None:
        """Return the full-resolution plot only if it is already held."""
        return self._details.get(frame_index)

    def holds(
        self, frame_index: int, modality: str, condition: str | None = None
    ) -> bool:
        """Return whether a prepared acquisition is already held.

        Asked by a caller building its own read-ahead order, so that it can
        leave out what is in memory without disturbing the recency ordering the
        way a lookup would.
        """
        key = self._key(frame_index, modality, condition)
        return key is not None and key in self._images

    def ensure(self, request: LoadRequest) -> None:
        """Prepare the data named by *request* if it is not already held."""
        if request.detail:
            self.plot_detail(request.frame_index)
        elif request.modality is None:
            self.plot(request.frame_index)
        else:
            self.image(request.frame_index, request.modality, request.condition)

    def pending(
        self,
        current: int,
        modalities: Iterable[str],
        *,
        radius: int | None = DEFAULT_RADIUS,
        include_plots: bool = False,
        conditions: dict[str, str | None] | None = None,
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
            conditions: The imaging condition being viewed for each modality.
                Only the viewed condition is prepared ahead -- preparing every
                condition would multiply the read-ahead cost by however many a
                stage happens to have.

        Returns:
            Requests for data that is not yet held.
        """
        wanted = list(modalities)
        chosen = conditions or {}
        requests: list[LoadRequest] = []
        for frame_index in prefetch_order(current, len(self._frames), radius):
            for modality in wanted:
                condition = chosen.get(modality)
                key = self._key(frame_index, modality, condition)
                if key is None or key in self._images:
                    continue
                requests.append(LoadRequest(frame_index, modality, condition))
            if include_plots:
                frame = self._frame(frame_index)
                if frame is not None and frame.displayed_plot is not None:
                    if frame_index not in self._plots:
                        requests.append(LoadRequest(frame_index, None))
        # Last, and only where it will be looked at: it is the slowest read
        # here, and putting it anywhere else would delay a step through the
        # lamella for a resolution nobody is using yet.
        if include_plots and current not in self._details:
            frame = self._frame(current)
            if frame is not None and frame.displayed_plot is not None:
                requests.append(LoadRequest(current, None, detail=True))
        return requests

    def _frame(self, frame_index: int) -> LamellaFrame | None:
        """Return the frame at *frame_index*, or None if out of range."""
        if 0 <= frame_index < len(self._frames):
            return self._frames[frame_index]
        return None

    def _record(self, frame_index: int, modality: str, condition: str | None = None):
        """Return the image record for a frame, modality and condition, if any.

        Falls back to the frame's first image when the requested condition is
        not present: navigating through a lamella where only some acquisitions
        were captured under a given condition should keep showing something
        rather than blanking the pane.
        """
        frame = self._frame(frame_index)
        if frame is None:
            return None
        record = frame.record(modality, condition)
        if record is None and condition is not None:
            return frame.record(modality)
        return record

    def _key(
        self, frame_index: int, modality: str, condition: str | None
    ) -> tuple[int, str, str | None] | None:
        """Return what the image shown for this request is held under, or None.

        The condition of the record actually shown, so an image is held once
        however it was asked for -- by name, as the first, or by a fallback.
        """
        record = self._record(frame_index, modality, condition)
        return (frame_index, modality, record.condition) if record is not None else None
