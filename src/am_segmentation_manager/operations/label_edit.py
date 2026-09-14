"""Reversible edits to a label array.

Brush, flood fill, class locking and undo as operations on numpy arrays, testable
without a display. Arrays are modified in place and every edit returns the
:class:`EditPatch` that reverses it, or None for one that would change nothing.
Class values are categorical, so nothing interpolates and no operation changes a
label's dtype. Points are ``(x, y)`` -- column then row -- as in OpenCV and Qt.
"""
from __future__ import annotations

import logging
from collections import deque
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..config import BACKGROUND_CLASS

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = logging.getLogger(__name__)

#: How many edits an :class:`UndoStack` keeps before the oldest is forgotten.
#: Patches are bounding-box sized, so this costs little; it exists to bound a
#: long proofreading session rather than to save a measurable amount of memory.
DEFAULT_UNDO_DEPTH = 32

#: Connectivity :func:`fill` uses by default. Four rather than eight so a
#: one-pixel diagonal boundary holds the fill in: a fill that leaks through a
#: line the user can see is a fill they have to undo.
DEFAULT_CONNECTIVITY = 4

__all__ = [
    "DEFAULT_CONNECTIVITY",
    "DEFAULT_UNDO_DEPTH",
    "EditPatch",
    "UndoStack",
    "fill",
    "new_label",
    "paint",
    "patch_between",
]


@dataclass(frozen=True, eq=False)
class EditPatch:
    """One reversible change to a rectangular region of a label.

    The region is the tight bounding box of the pixels that actually changed,
    and *before* and *after* hold every pixel in it, changed or not. Restoring
    the whole box is correct because edits are undone in the order they were
    made; it is also what keeps a patch a plain array slice rather than a mask.

    Attributes:
        bbox: ``(top, left, height, width)`` of the region, in label pixels.
        before: The region's values before the edit.
        after: The region's values after it.

    Patches compare by identity: two patches holding equal arrays are still
    separate edits.
    """

    bbox: tuple[int, int, int, int]
    before: np.ndarray
    after: np.ndarray

    @property
    def pixels(self) -> int:
        """Return how many pixels the patched region covers."""
        _, _, height, width = self.bbox
        return height * width


def new_label(shape: tuple[int, ...], dtype=None) -> np.ndarray:
    """Return an all-background label array matching *shape*.

    What an image with no label file yet starts from. Only the first two axes
    of *shape* are used, so an image array can be passed straight in.

    Args:
        shape: The label's ``(height, width)``, or any shape beginning with it.
        dtype: Integer dtype to allocate; defaults to ``uint8``, which holds
            every class the segmentation config defines.
    """
    import numpy as np

    return np.full(shape[:2], BACKGROUND_CLASS, dtype=dtype or np.uint8)


def paint(
    label: np.ndarray,
    points: Sequence[tuple[int, int]],
    radius: int,
    value: int,
    *,
    protected: Collection[int] = (),
) -> EditPatch | None:
    """Draw a brush stroke through *points*, writing *value*.

    Successive points are joined, so a drag reported as a handful of widely
    spaced cursor positions leaves a continuous stroke rather than separate
    dabs. One call is one stroke and therefore one undo step.

    Args:
        label: The array to draw on, modified in place.
        points: The cursor positions of the stroke, in order. A single point
            is a dab.
        radius: Brush radius in label pixels; 0 paints a single pixel.
        value: The class to write.
        protected: Classes that must not be painted over.

    Returns:
        The patch reversing the stroke, or None if nothing changed.

    Raises:
        ValueError: If *radius* is negative, or *value* does not fit the
            label's dtype.
    """
    import cv2
    import numpy as np

    if radius < 0:
        raise ValueError(f"Brush radius cannot be negative, got {radius}")
    if not points:
        return None
    _check_value(label, value)

    height, width = label.shape[:2]
    xs = [int(x) for x, _ in points]
    ys = [int(y) for _, y in points]
    left = max(0, min(xs) - radius)
    right = min(width - 1, max(xs) + radius)
    top = max(0, min(ys) - radius)
    bottom = min(height - 1, max(ys) + radius)
    if left > right or top > bottom:
        logger.debug("Stroke lies entirely outside the label")
        return None

    region = label[top : bottom + 1, left : right + 1]
    stroke = np.zeros(region.shape, dtype=np.uint8)
    shifted = [(x - left, y - top) for x, y in zip(xs, ys)]

    for start, end in zip(shifted, shifted[1:]):
        cv2.line(stroke, start, end, 1, 2 * radius + 1)
    for point in shifted:
        cv2.circle(stroke, point, radius, 1, -1)
        # The disc of a zero radius is drawn inconsistently across OpenCV
        # versions, and a one-pixel pencil has to work.
        x, y = point
        if 0 <= y < stroke.shape[0] and 0 <= x < stroke.shape[1]:
            stroke[y, x] = 1

    _clear_protected(stroke, region, protected)
    return _apply(label, top, left, (stroke != 0) & (region != value), value)


def fill(
    label: np.ndarray,
    seed: tuple[int, int],
    value: int,
    *,
    protected: Collection[int] = (),
    connectivity: int = DEFAULT_CONNECTIVITY,
) -> EditPatch | None:
    """Flood the region of one class containing *seed*, writing *value*.

    The region is the connected run of pixels sharing the seed's current class,
    so a fill stops wherever the class changes -- which is what makes an area
    enclosed by an annotated boundary fillable without any boundary tracing.

    Args:
        label: The array to fill, modified in place.
        seed: Where the fill starts. A seed outside the array fills nothing.
        value: The class to write.
        protected: Classes that must not be filled over. Seeding on one of them
            fills nothing.
        connectivity: 4 or 8; see :data:`DEFAULT_CONNECTIVITY`.

    Returns:
        The patch reversing the fill, or None if nothing changed.

    Raises:
        ValueError: If *connectivity* is neither 4 nor 8, or *value* does not
            fit the label's dtype.
    """
    import cv2
    import numpy as np

    if connectivity not in (4, 8):
        raise ValueError(f"Connectivity must be 4 or 8, got {connectivity}")
    _check_value(label, value)

    height, width = label.shape[:2]
    x, y = int(seed[0]), int(seed[1])
    if not (0 <= x < width and 0 <= y < height):
        logger.debug("Fill seed (%d, %d) lies outside the label", x, y)
        return None

    # floodFill will not take a uint16 image, and labels may be either depth.
    # Filling a boolean "same class as the seed" image instead is independent of
    # the label's dtype, and asks the same question of it.
    same = (label == label[y, x]).astype(np.uint8)
    mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(
        same, mask, (x, y), 0,
        flags=connectivity | cv2.FLOODFILL_MASK_ONLY | (1 << 8),
    )
    filled = mask[1:-1, 1:-1]

    _clear_protected(filled, label, protected)
    return _apply(label, 0, 0, (filled != 0) & (label != value), value)


def patch_between(before: np.ndarray, after: np.ndarray) -> EditPatch | None:
    """Return the patch that turns *before* into *after*, or None if they match.

    How a brush stroke becomes one undo step: the tools are applied segment by
    segment as the pointer moves, so that the drawing keeps up with it, and the
    whole stroke is reduced to a single reversible edit when the button is
    released.

    Raises:
        ValueError: If the two arrays are different shapes, which would mean
            the label was replaced rather than edited.
    """
    import numpy as np

    if before.shape != after.shape:
        raise ValueError(
            f"Cannot compare a {before.shape} label with a {after.shape} one"
        )

    changed = before != after
    if not changed.any():
        return None

    rows = np.flatnonzero(changed.any(axis=1))
    cols = np.flatnonzero(changed.any(axis=0))
    top, bottom = int(rows[0]), int(rows[-1])
    left, right = int(cols[0]), int(cols[-1])
    region = (slice(top, bottom + 1), slice(left, right + 1))
    return EditPatch(
        bbox=(top, left, bottom - top + 1, right - left + 1),
        before=before[region].copy(),
        after=after[region].copy(),
    )


class UndoStack:
    """The edits made to one label, most recent last.

    Bounded, so a long session cannot grow without limit; pushing past
    :attr:`max_entries` forgets the oldest edit, which can then no longer be
    undone. Pushing a new edit discards anything that had been undone, as
    every editor does -- the redone future no longer follows from the present.
    """

    def __init__(self, max_entries: int = DEFAULT_UNDO_DEPTH) -> None:
        """Create a stack keeping at most *max_entries* edits."""
        if max_entries < 1:
            raise ValueError(f"max_entries must be at least 1, got {max_entries}")
        self.max_entries = max_entries
        self._done: deque[EditPatch] = deque(maxlen=max_entries)
        self._undone: list[EditPatch] = []

    def push(self, patch: EditPatch | None) -> None:
        """Record *patch* as the most recent edit, ignoring None."""
        if patch is None:
            return
        self._done.append(patch)
        self._undone.clear()

    @property
    def can_undo(self) -> bool:
        """Return whether there is an edit to undo."""
        return bool(self._done)

    @property
    def can_redo(self) -> bool:
        """Return whether an undone edit can be reapplied."""
        return bool(self._undone)

    def undo(self, label: np.ndarray) -> EditPatch | None:
        """Reverse the most recent edit on *label*, returning what was undone."""
        if not self._done:
            return None
        patch = self._done.pop()
        _write(label, patch.bbox, patch.before)
        self._undone.append(patch)
        return patch

    def redo(self, label: np.ndarray) -> EditPatch | None:
        """Reapply the most recently undone edit, returning what was redone."""
        if not self._undone:
            return None
        patch = self._undone.pop()
        _write(label, patch.bbox, patch.after)
        self._done.append(patch)
        return patch

    def clear(self) -> None:
        """Forget every edit, undone or not."""
        self._done.clear()
        self._undone.clear()

    def __len__(self) -> int:
        """Return how many edits can still be undone."""
        return len(self._done)


def _check_value(label: np.ndarray, value: int) -> None:
    """Raise if *value* cannot be stored in *label* without wrapping."""
    import numpy as np

    info = np.iinfo(label.dtype)
    if not info.min <= value <= info.max:
        raise ValueError(
            f"Class {value} does not fit a {label.dtype} label "
            f"({info.min} to {info.max})"
        )


def _clear_protected(
    mask: np.ndarray, region: np.ndarray, protected: Collection[int]
) -> None:
    """Remove the locked classes' pixels from a tool's *mask*, in place.

    Brush and fill lock classes the same way, so that the two cannot drift
    apart in what "protected" means.
    """
    import numpy as np

    if not protected:
        return
    mask[np.isin(region, np.asarray(tuple(protected)))] = 0


def _apply(
    label: np.ndarray, top: int, left: int, changed: np.ndarray, value: int
) -> EditPatch | None:
    """Write *value* where *changed*, returning the patch that reverses it.

    *changed* covers the region of *label* starting at ``(top, left)``. The
    patch is tightened to the pixels that really changed, so a stroke near the
    edge of a large fill's bounding box does not carry the whole box around.
    """
    import numpy as np

    if not changed.any():
        return None

    rows = np.flatnonzero(changed.any(axis=1))
    cols = np.flatnonzero(changed.any(axis=0))
    first_row, last_row = int(rows[0]), int(rows[-1])
    first_col, last_col = int(cols[0]), int(cols[-1])

    view = label[
        top + first_row : top + last_row + 1,
        left + first_col : left + last_col + 1,
    ]
    before = view.copy()
    view[changed[first_row : last_row + 1, first_col : last_col + 1]] = value
    return EditPatch(
        bbox=(top + first_row, left + first_col, view.shape[0], view.shape[1]),
        before=before,
        after=view.copy(),
    )


def _write(label: np.ndarray, bbox: tuple[int, int, int, int], values: np.ndarray) -> None:
    """Restore a patched region of *label* to *values*."""
    top, left, height, width = bbox
    label[top : top + height, left : left + width] = values
