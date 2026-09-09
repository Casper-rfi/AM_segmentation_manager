"""Preparing images, labels and plots for viewing.

Turns the large files on disk into small arrays suitable for display, since a
viewer shows a 2048x3072 acquisition at a fraction of its size.

Deliberately free of any GUI toolkit: it works in numpy arrays a widget layer can
wrap, and every function here is testable with synthetic arrays.

A :class:`DisplayImage` holds a frame's *data* rather than a finished picture, so
composing the coloured overlay stays a separate, cheap step.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .config import BACKGROUND_CLASS, SegmentationConfig
from .io.image import read_array, read_color_array, require_image_stack, to_2d

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = logging.getLogger(__name__)

#: Longest edge, in pixels, that acquisitions are reduced to for display.
DEFAULT_MAX_EDGE = 1024

#: Size the window prepares a plot at before a pane has been measured. Only a
#: starting point: :meth:`FrameCache.set_plot_max_size` replaces it with the
#: pane's own size, and reading a plot *whole* passes no size at all.
DEFAULT_PLOT_MAX_EDGE = 1600

#: Opacity of the label colour drawn over the image.
DEFAULT_ALPHA = 0.4

#: What stands in for the acquisition where it is hidden, as an RGB triple.
#:
#: **Not black**, which is also the colour of the pane behind the picture: with
#: both black, the label's transparent background and the emptiness past the
#: frame's edge become one surface, and a row of pixels missing along an edge
#: cannot be seen at all -- which is exactly what somebody checking a label
#: runs to the edge is looking for. Dark enough that an annotation of a few
#: pixels is still far and away the brightest thing on screen, that being what
#: hiding the image is for.
HIDDEN_IMAGE_GREY = (32, 32, 32)

#: Percentiles used by the optional contrast stretch. Only applied when a
#: caller explicitly asks for it.
DEFAULT_PERCENTILES = (0.5, 99.5)

#: What is known about a frame's label.
#:
#: ``"missing"`` -- no label file exists.
#: ``"empty"``   -- a label file exists but contains only background. The
#:                  external segmentation software creates one of these for
#:                  every image it sees, so this means "not yet annotated".
#: ``"present"`` -- the label contains at least one foreground class.
LabelStatus = Literal["missing", "empty", "present"]


@dataclass(frozen=True)
class DisplayImage:
    """A single acquisition reduced to display size, with its label.

    Attributes:
        image: 2D uint8 greyscale array, downscaled to fit.
        label: Label array matching *image* in shape, or None when no label
            file exists. Class values are preserved, not remapped.
        label_status: What is known about the label; see :data:`LabelStatus`.
        label_resized: Whether the label had different pixel dimensions from
            the image and had to be resampled to match. Worth surfacing: it
            usually means the label came from a prediction at another scale.
        original_shape: ``(height, width)`` of the image file on disk.
        scale: Factor applied to reach the display size, 1.0 if unchanged.
        source: Path the image was read from.
    """

    image: np.ndarray
    label: np.ndarray | None = None
    label_status: LabelStatus = "missing"
    label_resized: bool = False
    original_shape: tuple[int, int] = (0, 0)
    scale: float = 1.0
    source: Path | None = None

    @property
    def has_label(self) -> bool:
        """Return whether a label exists, empty or not."""
        return self.label is not None


# ---------------------------------------------------------------------------
# Array preparation
# ---------------------------------------------------------------------------

def to_display_8bit(
    image: np.ndarray,
    *,
    stretch: bool = False,
    percentiles: tuple[float, float] = DEFAULT_PERCENTILES,
) -> np.ndarray:
    """Convert an image array to 8-bit for display, without altering contrast.

    SEM and FIB acquisitions already have their intensity range set correctly
    during acquisition, so this performs no contrast adjustment by default:

    * 8-bit input is returned unchanged -- newer acquisitions are already
      display-ready and must be shown exactly as recorded.
    * Higher bit depths, which older projects use, are linearly rescaled onto
      0-255 across the image's full value range. This is a format conversion
      required to display the data at all, not a contrast adjustment: the
      mapping is linear and nothing is clipped. Using the observed range
      rather than the dtype maximum also displays correctly when a detector
      writes, say, 12-bit values into a 16-bit file.

    Set *stretch* to apply a percentile stretch as well, for the rare image
    whose contrast really is too low to review. This clips the tails, so it is
    off unless asked for.

    Args:
        image: 2D image array of any dtype.
        stretch: Clip to *percentiles* before rescaling.
        percentiles: Low and high percentiles mapped to 0 and 255.

    Returns:
        A 2D uint8 array. A constant image returns all zeros, since it has no
        range to map onto.
    """
    import numpy as np

    if image.dtype == np.uint8 and not stretch:
        return image

    if stretch:
        low, high = (float(value) for value in np.percentile(image, percentiles))
    else:
        low, high = float(image.min()), float(image.max())

    if high <= low:
        logger.debug("Image has a single value; displaying as zeros")
        return np.zeros(image.shape, dtype=np.uint8)

    scaled = (image.astype(np.float32) - low) * (255.0 / (high - low))
    return np.rint(np.clip(scaled, 0, 255)).astype(np.uint8)


#: The intensity limits a frame is shown between until anybody moves them: the
#: whole 8-bit range, which is the display array's own, so a frame opens
#: showing exactly what was recorded.
DEFAULT_INTENSITY_RANGE = (0, 255)


def apply_intensity_range(
    image: np.ndarray, low: int, high: int
) -> np.ndarray:
    """Return *image* rescaled so *low* maps to black and *high* to white.

    A view control, applied where the picture is drawn rather than where it is
    prepared: :class:`DisplayImage` holds a uint8 array already in memory, and
    folding this into the preparation would invalidate that cache on every
    movement of a slider and re-read the file behind it. It is a lookup table
    over 256 values whatever the size of the frame.

    Everything below *low* comes out black and everything above *high* white,
    which is the point: an acquisition whose data sits in a narrow band is
    reviewed by throwing away the empty ends of the range. That clipping is why
    it is a *view*, and why nothing here reaches an export --
    :func:`to_display_8bit` is what a written file goes through, and it clips
    nothing.

    Args:
        image: 2D uint8 array, as :attr:`DisplayImage.image` holds.
        low: Intensity shown as black, 0-255.
        high: Intensity shown as white. Held at least one above *low*, the two
            meeting being a threshold rather than a range.

    Returns:
        A uint8 array of the same shape. Over the full range, *image* itself:
        the control costs nothing until it is used.
    """
    import numpy as np

    if (low, high) == DEFAULT_INTENSITY_RANGE:
        return image
    low = int(np.clip(low, 0, 254))
    high = int(np.clip(high, low + 1, 255))
    levels = np.arange(256, dtype=np.float32)
    table = np.rint(
        np.clip((levels - low) * (255.0 / (high - low)), 0, 255)
    ).astype(np.uint8)
    return table[image]


def downscale_to_fit(
    array: np.ndarray, max_edge: int, *, nearest: bool = False
) -> np.ndarray:
    """Reduce *array* so its longest edge is at most *max_edge* pixels.

    Arrays already within the limit are returned unchanged; this never
    upscales, because enlarging cannot add detail and only costs memory.

    Args:
        array: 2D or 3D image array.
        max_edge: Maximum length of the longer edge; values <= 0 disable
            downscaling.
        nearest: Use nearest-neighbour resampling. Required for label arrays,
            whose values are class indices and must not be averaged.

    Returns:
        The reduced array, or *array* itself when no reduction is needed.
    """
    import cv2

    height, width = array.shape[:2]
    longest = max(height, width)
    if max_edge <= 0 or longest <= max_edge:
        return array

    scale = max_edge / longest
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(array, target, interpolation=interpolation)


def fit_within(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Reduce *array* to fit inside ``(width, height)``, preserving detail.

    Fitting a box rather than a single edge matters when the result is to be
    shown without any further scaling: constraining only the longer edge can
    still leave the other one overflowing a pane that is not the same shape.

    Never upscales -- enlarging cannot add detail, and a viewer showing the
    result larger than life is better off doing that itself.
    """
    max_width, max_height = size
    height, width = array.shape[:2]
    if max_width <= 0 or max_height <= 0:
        return array

    scale = min(max_width / width, max_height / height)
    if scale >= 1.0:
        return array
    return _resize_preserving_detail(
        array, max(1, round(width * scale)), max(1, round(height * scale))
    )


def downscale_for_detail(array: np.ndarray, max_edge: int) -> np.ndarray:
    """Reduce *array* preserving fine detail such as rendered text.

    Plain area averaging is correct but soft, which is fine for photographic
    data and poor for the axis labels and annotations in a rendered figure.
    This decimates most of the way with area averaging -- which is what stops
    the result aliasing -- and takes the last factor of two with a sharp
    filter, giving noticeably crisper small text at a modest extra cost.

    Used for summary plots. Acquisitions keep plain area averaging: a sharp
    filter can ring around hard edges, and inventing halos in scientific image
    data to make it look crisper is not a trade worth making.
    """
    height, width = array.shape[:2]
    longest = max(height, width)
    if max_edge <= 0 or longest <= max_edge:
        return array

    scale = max_edge / longest
    return _resize_preserving_detail(
        array, max(1, round(width * scale)), max(1, round(height * scale))
    )


def _resize_preserving_detail(
    array: np.ndarray, target_width: int, target_height: int
) -> np.ndarray:
    """Resize to an exact size, keeping fine detail as sharp as possible."""
    import cv2

    height, width = array.shape[:2]
    scale = min(target_width / width, target_height / height)

    # Only worth two passes when there is a real reduction to make.
    if scale > 0.5:
        return cv2.resize(
            array, (target_width, target_height), interpolation=cv2.INTER_AREA
        )

    intermediate = (min(width, target_width * 2), min(height, target_height * 2))
    reduced = cv2.resize(array, intermediate, interpolation=cv2.INTER_AREA)
    return cv2.resize(
        reduced, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4
    )


def resize_label_to(label: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Resample a label array to *shape* using nearest-neighbour.

    Class indices are categorical, so interpolation would invent classes that
    were never predicted. Returns *label* unchanged when it already matches.
    """
    import cv2

    height, width = shape[:2]
    if label.shape[:2] == (height, width):
        return label
    return cv2.resize(label, (width, height), interpolation=cv2.INTER_NEAREST)


def is_empty_label(label: np.ndarray) -> bool:
    """Return whether a label contains nothing but background.

    The external segmentation software writes an empty label for every image
    in its input folder, so an existing label file is not evidence that an
    image has been annotated. This distinguishes the two.
    """
    return not bool((label != BACKGROUND_CLASS).any())


def to_rgb(array: np.ndarray) -> np.ndarray:
    """Convert a greyscale, RGB or RGBA array to plain 8-bit RGB.

    Transparency is composited over white, matching the background these plots
    are rendered against.
    """
    import numpy as np

    if array.dtype != np.uint8:
        array = to_display_8bit(array)

    if array.ndim == 2:
        return np.repeat(array[:, :, np.newaxis], 3, axis=2)
    if array.ndim == 3 and array.shape[2] == 3:
        return array
    if array.ndim == 3 and array.shape[2] == 4:
        colour = array[:, :, :3].astype(np.float32)
        alpha = array[:, :, 3:4].astype(np.float32) / 255.0
        composited = colour * alpha + 255.0 * (1.0 - alpha)
        return np.clip(composited, 0, 255).astype(np.uint8)
    raise ValueError(f"Cannot interpret array of shape {array.shape} as an image")


# ---------------------------------------------------------------------------
# Overlay composition
# ---------------------------------------------------------------------------

def compose_overlay(
    image: np.ndarray,
    label: np.ndarray,
    config: SegmentationConfig,
    *,
    alpha: float = DEFAULT_ALPHA,
    hide_background: bool = True,
    hide_image: bool = False,
) -> np.ndarray:
    """Draw a label over an image using the configured class colours.

    Args:
        image: 2D greyscale or ``(H, W, 3)`` RGB uint8 array.
        label: Label array with the same height and width as *image*.
        config: Supplies the colour for each class index.
        alpha: Opacity of the label colour, 0 (invisible) to 1 (opaque).
        hide_background: Leave the background class undrawn. On by default:
            background covers most of the image, and colouring it would hide
            the data being reviewed.
        hide_image: Draw the label over :data:`HIDDEN_IMAGE_GREY` rather than
            over the image. What makes a stray annotation of a few pixels
            findable -- against the data it is a speck on grey, and against a
            flat ground it is the only thing on screen. *alpha* still applies,
            so a label drawn faintly over that ground is faint.

    Returns:
        An ``(H, W, 3)`` uint8 RGB array.

    Raises:
        ValueError: If *alpha* is outside 0-1, or if the shapes do not match.
            Matching shapes are the caller's responsibility -- see
            :func:`resize_label_to` -- so a mismatch here is a bug, not data to
            be silently patched up.
    """
    import numpy as np

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    if image.shape[:2] != label.shape[:2]:
        raise ValueError(
            f"Image shape {image.shape[:2]} does not match label shape {label.shape[:2]}"
        )

    if hide_image:
        result = np.full((*image.shape[:2], 3), HIDDEN_IMAGE_GREY, dtype=np.uint8)
    elif image.ndim == 2:
        result = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    else:
        # Copy, so blending never writes through to the caller's array.
        result = image[:, :, :3].copy()

    drawn = _drawn_mask(label, hide_background)
    if alpha == 0.0 or not drawn.any():
        return result

    # Only the pixels being coloured are converted to float and blended: the
    # overlay usually covers a minority of the frame, and doing the whole image
    # in float costs several times more for no visible difference. Colours come
    # from a palette indexed by class value, so this is one vectorised pass
    # rather than one pass per class.
    palette = _build_palette(label, config, hide_background)
    colours = palette[label[drawn]]
    blended = result[drawn].astype(np.float32) * (1.0 - alpha) + colours * alpha
    result[drawn] = np.rint(np.clip(blended, 0, 255)).astype(np.uint8)

    return result


def _drawn_mask(label: np.ndarray, hide_background: bool) -> np.ndarray:
    """Return a boolean mask of the pixels the overlay should colour."""
    import numpy as np

    if hide_background:
        return label != BACKGROUND_CLASS
    return np.ones(label.shape, dtype=bool)


def _build_palette(
    label: np.ndarray, config: SegmentationConfig, hide_background: bool
) -> np.ndarray:
    """Build a class-index-to-colour lookup table for the classes in *label*.

    Only classes actually present are looked up, so a config warning is never
    emitted for a class the label does not contain.
    """
    import numpy as np

    present = np.unique(label)
    palette = np.zeros((int(present.max()) + 1, 3), dtype=np.float32)
    for class_value in present:
        class_index = int(class_value)
        if hide_background and class_index == BACKGROUND_CLASS:
            continue
        palette[class_index] = config.rgb(class_index)
    return palette


# ---------------------------------------------------------------------------
# Reading from disk
# ---------------------------------------------------------------------------

def prepare_display_image(
    image_path: Path,
    label_path: Path | None = None,
    *,
    max_edge: int = DEFAULT_MAX_EDGE,
) -> DisplayImage:
    """Read an acquisition and its label, reduced and ready to display.

    Intensities are not adjusted: 8-bit data is shown exactly as recorded, and
    higher bit depths are linearly mapped onto 8 bits so they can be displayed
    at all. See :func:`to_display_8bit`.

    A missing label is a normal outcome, not an error, and is reported through
    :attr:`DisplayImage.label_status`. Emptiness is judged on the label as
    stored, before any resampling, so that a small annotation cannot be lost to
    downscaling and misreported as an empty label.

    Args:
        image_path: The SEM or FIB acquisition to read.
        label_path: The matching label, if the record has one.
        max_edge: Longest edge to reduce the image to.

    Returns:
        The prepared :class:`DisplayImage`.

    Raises:
        RuntimeError: If the optional image stack is not installed.
        FileNotFoundError: If *image_path* does not exist.
        OSError: If the image exists but cannot be decoded.
    """
    require_image_stack("Displaying images")

    raw = to_2d(read_array(image_path))
    original_shape = (int(raw.shape[0]), int(raw.shape[1]))
    reduced = downscale_to_fit(raw, max_edge)
    image = to_display_8bit(reduced)
    scale = image.shape[0] / original_shape[0] if original_shape[0] else 1.0

    label = None
    label_status: LabelStatus = "missing"
    label_resized = False

    if label_path is not None and label_path.exists():
        raw_label = to_2d(read_array(label_path))
        label_resized = raw_label.shape[:2] != original_shape
        label_status = "empty" if is_empty_label(raw_label) else "present"
        label = resize_label_to(raw_label, image.shape)
        if label_resized:
            logger.debug(
                "Label %s (%s) does not match image %s (%s); resampled to match",
                label_path.name, raw_label.shape[:2], image_path.name, original_shape,
            )
    elif label_path is not None:
        logger.debug("Label file does not exist: %s", label_path)

    return DisplayImage(
        image=image,
        label=label,
        label_status=label_status,
        label_resized=label_resized,
        original_shape=original_shape,
        scale=scale,
        source=image_path,
    )


def prepare_plot(
    plot_path: Path,
    *,
    max_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Read a summary plot as an RGB array, as rendered unless bounded.

    Args:
        plot_path: The plot PNG to read.
        max_size: ``(width, height)`` to fit within, or None to read it whole.
            A plot carries axis labels and annotations that exist only at the
            resolution it was rendered at, so reducing it on read puts a
            ceiling on what looking closely can ever show; a caller drawing one
            into a fixed space says so here.

    Returns:
        An ``(H, W, 3)`` uint8 RGB array.

    Raises:
        RuntimeError: If the optional image stack is not installed.
        FileNotFoundError: If *plot_path* does not exist.
        OSError: If the plot exists but cannot be decoded.
    """
    require_image_stack("Displaying plots")
    plot = to_rgb(read_color_array(plot_path))
    return plot if max_size is None else fit_within(plot, max_size)
