"""Tests for the headless display pipeline.

Everything here runs on synthetic arrays: no display, no GUI toolkit, and only
tmp_path for the few tests that must read a real file.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from am_segmentation_manager.config import SegmentationConfig

HAS_IMAGE_STACK = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "tifffile", "cv2")
)

pytestmark = pytest.mark.skipif(
    not HAS_IMAGE_STACK, reason="Optional image libs not installed"
)

if HAS_IMAGE_STACK:
    import numpy as np
    import tifffile

    from am_segmentation_manager.display import (
        DEFAULT_MAX_EDGE,
        compose_overlay,
        downscale_to_fit,
        is_empty_label,
        prepare_display_image,
        prepare_plot,
        resize_label_to,
        to_display_8bit,
        to_rgb,
    )


@pytest.fixture
def config() -> SegmentationConfig:
    """A three-class config with unmistakable colours."""
    return SegmentationConfig(
        class_colors={0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 0, 255)},
        class_labels={0: "background", 1: "GIS", 2: "lamella"},
    )


def write_tiff(path: Path, array) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), array)
    return path


# ---------------------------------------------------------------------------
# Bit depth
# ---------------------------------------------------------------------------

def test_8bit_images_pass_through_unchanged():
    """Newer acquisitions are already 8-bit; reviewing must not alter them."""
    image = np.array([[0, 4, 128], [200, 250, 255]], dtype=np.uint8)

    result = to_display_8bit(image)

    assert result.dtype == np.uint8
    assert np.array_equal(result, image)


def test_16bit_images_are_mapped_onto_8_bits():
    """Older data may be 16-bit and cannot be displayed without conversion."""
    image = np.linspace(1000, 2000, 100, dtype=np.uint16).reshape(10, 10)

    result = to_display_8bit(image)

    assert result.dtype == np.uint8
    assert result.min() == 0
    assert result.max() == 255


def test_16bit_conversion_is_linear_and_clips_nothing():
    """The conversion is a format change, not a contrast adjustment."""
    image = np.array([[0, 16384, 32768, 49152, 65535]], dtype=np.uint16)

    result = to_display_8bit(image)

    assert list(result[0]) == [0, 64, 128, 191, 255]


def test_16bit_data_not_filling_its_container_still_displays():
    """12-bit values in a 16-bit file must not render as near-black."""
    image = np.array([[0, 2048, 4095]], dtype=np.uint16)

    result = to_display_8bit(image)

    assert result.max() == 255


def test_stretch_is_opt_in_only():
    """Acquisition already sets the range, so nothing is stretched by default."""
    image = np.array([[100, 110], [120, 130]], dtype=np.uint8)

    assert np.array_equal(to_display_8bit(image), image)

    stretched = to_display_8bit(image, stretch=True)
    assert (stretched.min(), stretched.max()) == (0, 255)


def test_constant_image_has_no_range_to_map():
    image = np.full((8, 8), 1234, dtype=np.uint16)

    result = to_display_8bit(image)

    assert result.dtype == np.uint8
    assert np.array_equal(result, np.zeros((8, 8), dtype=np.uint8))


def test_float_images_are_mapped_onto_8_bits():
    image = np.linspace(0.1, 0.9, 64, dtype=np.float32).reshape(8, 8)

    result = to_display_8bit(image)

    assert result.dtype == np.uint8
    assert (result.min(), result.max()) == (0, 255)


# ---------------------------------------------------------------------------
# Downscaling
# ---------------------------------------------------------------------------

def test_downscale_preserves_aspect_ratio():
    image = np.zeros((2048, 3072), dtype=np.uint8)

    result = downscale_to_fit(image, 1024)

    assert result.shape == (683, 1024)


def test_downscale_never_upscales():
    image = np.zeros((100, 200), dtype=np.uint8)

    result = downscale_to_fit(image, 1024)

    assert result.shape == (100, 200)


def test_downscale_disabled_by_non_positive_max_edge():
    image = np.zeros((2048, 3072), dtype=np.uint8)

    assert downscale_to_fit(image, 0).shape == (2048, 3072)


def test_labels_downscale_without_inventing_classes():
    """Averaging class indices would produce classes that were never predicted."""
    label = np.zeros((400, 400), dtype=np.uint8)
    label[:200] = 4

    result = downscale_to_fit(label, 100, nearest=True)

    assert set(np.unique(result)) == {0, 4}


def test_resize_label_matches_target_shape():
    label = np.ones((50, 80), dtype=np.uint8)

    result = resize_label_to(label, (25, 40))

    assert result.shape == (25, 40)


def test_resize_label_leaves_matching_shapes_alone():
    label = np.ones((25, 40), dtype=np.uint8)

    assert resize_label_to(label, (25, 40)) is label


# ---------------------------------------------------------------------------
# Empty label detection
# ---------------------------------------------------------------------------

def test_all_background_label_is_empty():
    """The segmentation software auto-creates these for every input image."""
    assert is_empty_label(np.zeros((10, 10), dtype=np.uint8)) is True


def test_label_with_any_foreground_is_not_empty():
    label = np.zeros((10, 10), dtype=np.uint8)
    label[5, 5] = 2

    assert is_empty_label(label) is False


# ---------------------------------------------------------------------------
# Overlay composition
# ---------------------------------------------------------------------------

def test_background_is_not_drawn(config):
    image = np.full((4, 4), 100, dtype=np.uint8)
    label = np.zeros((4, 4), dtype=np.uint8)

    result = compose_overlay(image, label, config, alpha=1.0)

    assert np.array_equal(result, np.dstack([image] * 3))


def test_foreground_classes_are_drawn_in_their_configured_colour(config):
    image = np.full((2, 2), 100, dtype=np.uint8)
    label = np.array([[0, 1], [2, 1]], dtype=np.uint8)

    result = compose_overlay(image, label, config, alpha=1.0)

    assert tuple(result[0, 0]) == (100, 100, 100)  # background, untouched
    assert tuple(result[0, 1]) == (255, 0, 0)      # class 1
    assert tuple(result[1, 0]) == (0, 0, 255)      # class 2


def test_alpha_blends_towards_the_class_colour(config):
    image = np.zeros((1, 1), dtype=np.uint8)
    label = np.ones((1, 1), dtype=np.uint8)

    result = compose_overlay(image, label, config, alpha=0.5)

    assert tuple(result[0, 0]) == (128, 0, 0)


def test_zero_alpha_leaves_the_image_untouched(config):
    image = np.full((2, 2), 77, dtype=np.uint8)
    label = np.ones((2, 2), dtype=np.uint8)

    result = compose_overlay(image, label, config, alpha=0.0)

    assert np.array_equal(result, np.dstack([image] * 3))


def test_background_can_be_drawn_when_requested(config):
    image = np.full((1, 1), 200, dtype=np.uint8)
    label = np.zeros((1, 1), dtype=np.uint8)

    result = compose_overlay(image, label, config, alpha=1.0, hide_background=False)

    assert tuple(result[0, 0]) == (0, 0, 0)


def test_unconfigured_class_uses_the_fallback_colour_rather_than_failing(config):
    image = np.zeros((1, 1), dtype=np.uint8)
    label = np.full((1, 1), 99, dtype=np.uint8)

    result = compose_overlay(image, label, config, alpha=1.0)

    assert tuple(result[0, 0]) == (255, 0, 255)


def test_overlay_accepts_an_rgb_image(config):
    image = np.full((2, 2, 3), 50, dtype=np.uint8)
    label = np.ones((2, 2), dtype=np.uint8)

    result = compose_overlay(image, label, config, alpha=1.0)

    assert result.shape == (2, 2, 3)
    assert tuple(result[0, 0]) == (255, 0, 0)


def test_hiding_the_image_leaves_only_the_label(config):
    image = np.full((2, 2), 200, dtype=np.uint8)
    label = np.zeros((2, 2), dtype=np.uint8)
    label[0, 0] = 1

    result = compose_overlay(image, label, config, alpha=1.0, hide_image=True)

    assert tuple(result[0, 0]) == (255, 0, 0), "the annotation still draws"
    assert tuple(result[1, 1]) == (0, 0, 0), (
        "and everything else is black, so a speck is the only thing on screen"
    )


def test_hiding_the_image_still_respects_the_opacity(config):
    image = np.full((2, 2), 200, dtype=np.uint8)
    label = np.ones((2, 2), dtype=np.uint8)

    result = compose_overlay(image, label, config, alpha=0.5, hide_image=True)

    assert tuple(result[0, 0]) == (128, 0, 0), "half of the class colour over black"


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_invalid_alpha_raises(config, alpha):
    with pytest.raises(ValueError, match="alpha"):
        compose_overlay(
            np.zeros((1, 1), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8),
            config, alpha=alpha,
        )


def test_shape_mismatch_is_a_programming_error(config):
    """Matching shapes is the caller's job; silently patching up hides bugs."""
    with pytest.raises(ValueError, match="does not match"):
        compose_overlay(
            np.zeros((4, 4), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8), config,
        )


# ---------------------------------------------------------------------------
# Colour conversion
# ---------------------------------------------------------------------------

def test_greyscale_becomes_rgb():
    result = to_rgb(np.full((2, 2), 40, dtype=np.uint8))

    assert result.shape == (2, 2, 3)
    assert tuple(result[0, 0]) == (40, 40, 40)


def test_rgba_is_composited_over_white():
    """Plots are rendered against white; a transparent pixel must not go black."""
    array = np.zeros((1, 2, 4), dtype=np.uint8)
    array[0, 0] = (255, 0, 0, 255)  # opaque red
    array[0, 1] = (255, 0, 0, 0)    # fully transparent

    result = to_rgb(array)

    assert result.shape == (1, 2, 3)
    assert tuple(result[0, 0]) == (255, 0, 0)
    assert tuple(result[0, 1]) == (255, 255, 255)


def test_rgb_passes_through():
    array = np.full((2, 2, 3), 10, dtype=np.uint8)

    assert to_rgb(array) is array


def test_unsupported_channel_count_raises():
    with pytest.raises(ValueError, match="Cannot interpret"):
        to_rgb(np.zeros((2, 2, 5), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Reading from disk
# ---------------------------------------------------------------------------

def test_prepare_display_image_reduces_and_pairs_the_label(tmp_path, config):
    image_path = write_tiff(tmp_path / "sem" / "img_000_SEM.tif",
                            np.full((2048, 3072), 120, dtype=np.uint8))
    label = np.zeros((2048, 3072), dtype=np.uint8)
    label[:100, :100] = 2
    label_path = write_tiff(tmp_path / "labels" / "img_000_SEM.tif", label)

    prepared = prepare_display_image(image_path, label_path)

    assert prepared.image.shape == (683, 1024)
    assert prepared.label.shape == prepared.image.shape
    assert prepared.label_status == "present"
    assert prepared.label_resized is False
    assert prepared.original_shape == (2048, 3072)
    assert prepared.scale == pytest.approx(683 / 2048)
    assert prepared.has_label is True
    # The prepared arrays compose without any further shape handling.
    assert compose_overlay(prepared.image, prepared.label, config).shape == (683, 1024, 3)


def test_prepare_display_image_without_a_label(tmp_path):
    image_path = write_tiff(tmp_path / "img.tif", np.zeros((64, 64), dtype=np.uint8))

    prepared = prepare_display_image(image_path)

    assert prepared.label is None
    assert prepared.label_status == "missing"
    assert prepared.has_label is False


def test_missing_label_file_is_not_an_error(tmp_path):
    image_path = write_tiff(tmp_path / "img.tif", np.zeros((64, 64), dtype=np.uint8))

    prepared = prepare_display_image(image_path, tmp_path / "absent.tif")

    assert prepared.label_status == "missing"


def test_auto_generated_empty_label_is_reported_as_empty(tmp_path):
    image_path = write_tiff(tmp_path / "img.tif", np.zeros((64, 64), dtype=np.uint8))
    label_path = write_tiff(tmp_path / "label.tif", np.zeros((64, 64), dtype=np.uint8))

    prepared = prepare_display_image(image_path, label_path)

    assert prepared.label_status == "empty"
    assert prepared.has_label is True


def test_label_of_a_different_size_is_matched_and_flagged(tmp_path):
    image_path = write_tiff(tmp_path / "img.tif", np.zeros((400, 400), dtype=np.uint8))
    label = np.zeros((100, 100), dtype=np.uint8)
    label[:50] = 1
    label_path = write_tiff(tmp_path / "label.tif", label)

    prepared = prepare_display_image(image_path, label_path, max_edge=200)

    assert prepared.label_resized is True
    assert prepared.label.shape == prepared.image.shape
    assert prepared.label_status == "present"


def test_small_annotation_survives_the_emptiness_check(tmp_path):
    """Emptiness is judged before downscaling, so a few pixels are not lost."""
    image_path = write_tiff(tmp_path / "img.tif", np.zeros((2000, 2000), dtype=np.uint8))
    label = np.zeros((2000, 2000), dtype=np.uint8)
    label[0, 0] = 3
    label_path = write_tiff(tmp_path / "label.tif", label)

    prepared = prepare_display_image(image_path, label_path, max_edge=64)

    assert prepared.label_status == "present"


def test_16bit_acquisition_is_readable(tmp_path):
    """Older projects hold 16-bit data; it must display without a flag."""
    data = np.linspace(2000, 3000, 4096, dtype=np.uint16).reshape(64, 64)
    image_path = write_tiff(tmp_path / "img.tif", data)

    prepared = prepare_display_image(image_path)

    assert prepared.image.dtype == np.uint8
    assert prepared.image.max() == 255


def test_missing_image_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_display_image(tmp_path / "absent.tif")


def test_prepare_plot_drops_alpha_and_keeps_the_rendered_size(tmp_path):
    """Read whole by default: the annotations on these figures exist only at
    the resolution they were rendered at."""
    import cv2

    plot = np.zeros((3000, 4500, 4), dtype=np.uint8)
    plot[:, :, 3] = 255
    plot_path = tmp_path / "plot.png"
    cv2.imwrite(str(plot_path), plot)

    result = prepare_plot(plot_path)

    assert result.ndim == 3 and result.shape[2] == 3
    assert result.shape[:2] == (3000, 4500)
    assert result.dtype == np.uint8


def test_default_max_edge_is_applied(tmp_path):
    image_path = write_tiff(tmp_path / "img.tif", np.zeros((4000, 4000), dtype=np.uint8))

    prepared = prepare_display_image(image_path)

    assert max(prepared.image.shape) == DEFAULT_MAX_EDGE


# ---------------------------------------------------------------------------
# Detail-preserving downscaling (used for rendered figures)
# ---------------------------------------------------------------------------

def _fine_detail_image(size: int = 800) -> np.ndarray:
    """A high-contrast pattern standing in for small rendered text."""
    image = np.full((size, size), 255, dtype=np.uint8)
    image[:, ::4] = 0
    image[::4, :] = 0
    return image


def test_downscale_for_detail_hits_the_target_size():
    from am_segmentation_manager.display import downscale_for_detail

    result = downscale_for_detail(np.zeros((3000, 4500, 3), dtype=np.uint8), 1070)

    assert max(result.shape[:2]) == 1070
    assert result.shape[2] == 3


def test_downscale_for_detail_never_upscales():
    from am_segmentation_manager.display import downscale_for_detail

    small = np.zeros((100, 200), dtype=np.uint8)

    assert downscale_for_detail(small, 1070) is small


def test_downscale_for_detail_keeps_more_contrast_than_area_averaging():
    """Area averaging is correct but soft; figures need the detail kept."""
    from am_segmentation_manager.display import downscale_for_detail

    source = _fine_detail_image()
    detailed = downscale_for_detail(source, 200)
    averaged = downscale_to_fit(source, 200)

    assert detailed.std() > averaged.std()


def test_small_reductions_use_a_single_pass():
    """Below a factor of two there is nothing for the second pass to improve."""
    from am_segmentation_manager.display import downscale_for_detail

    source = _fine_detail_image(300)

    assert np.array_equal(
        downscale_for_detail(source, 250), downscale_to_fit(source, 250)
    )


def test_prepare_plot_honours_the_requested_size(tmp_path):
    import cv2

    plot_path = tmp_path / "plot.png"
    cv2.imwrite(str(plot_path), np.zeros((1500, 2250, 3), dtype=np.uint8))

    assert prepare_plot(plot_path, max_size=(1070, 1070)).shape[:2] == (713, 1070)
    assert prepare_plot(plot_path, max_size=(600, 600)).shape[:2] == (400, 600)


def test_prepare_plot_fits_both_dimensions_of_the_box(tmp_path):
    """Constraining only the long edge can still overflow a differently
    shaped pane, which would force the view to rescale it again."""
    import cv2

    plot_path = tmp_path / "plot.png"
    cv2.imwrite(str(plot_path), np.zeros((1500, 2250, 3), dtype=np.uint8))

    result = prepare_plot(plot_path, max_size=(1000, 400))

    assert result.shape[0] <= 400 and result.shape[1] <= 1000


def test_fit_within_never_upscales():
    from am_segmentation_manager.display import fit_within

    small = np.zeros((50, 80, 3), dtype=np.uint8)

    assert fit_within(small, (1000, 1000)) is small


def test_fit_within_uses_the_limiting_dimension():
    from am_segmentation_manager.display import fit_within

    array = np.zeros((1000, 1000, 3), dtype=np.uint8)

    assert fit_within(array, (500, 200)).shape[:2] == (200, 200)


# ---------------------------------------------------------------------------
# The black/white range, which is a view of the data and never a change to it
# ---------------------------------------------------------------------------

def test_the_whole_range_hands_back_the_array_itself():
    """The control costs nothing until it is used: the cached array is what the
    panel draws, not a copy of it."""
    from am_segmentation_manager.display import (
        DEFAULT_INTENSITY_RANGE,
        apply_intensity_range,
    )

    image = np.full((8, 8), 60, dtype=np.uint8)

    assert apply_intensity_range(image, *DEFAULT_INTENSITY_RANGE) is image


def test_the_limits_map_to_black_and_white():
    from am_segmentation_manager.display import apply_intensity_range

    image = np.array([[60, 130, 200]], dtype=np.uint8)

    result = apply_intensity_range(image, 60, 200)

    assert result[0, 0] == 0
    assert result[0, 2] == 255
    assert 100 < result[0, 1] < 155, "and the middle lands in the middle"


def test_everything_outside_the_limits_is_clipped():
    """The point of it: an acquisition whose data sits in a narrow band is
    reviewed by throwing away the empty ends of the range."""
    from am_segmentation_manager.display import apply_intensity_range

    image = np.array([[0, 40, 210, 255]], dtype=np.uint8)

    result = apply_intensity_range(image, 60, 200)

    assert list(result[0]) == [0, 0, 255, 255]


def test_a_narrow_band_is_spread_across_the_whole_range():
    from am_segmentation_manager.display import apply_intensity_range

    image = np.array([[100, 105, 110]], dtype=np.uint8)

    assert np.ptp(apply_intensity_range(image, 100, 110)) == 255


def test_the_limits_never_meet():
    """Two sliders dragged past each other would divide by zero, and a picture
    that turned into a threshold would read as a fault rather than a setting."""
    from am_segmentation_manager.display import apply_intensity_range

    image = np.array([[0, 128, 255]], dtype=np.uint8)

    for low, high in ((100, 100), (200, 50), (255, 255)):
        result = apply_intensity_range(image, low, high)
        assert result.dtype == np.uint8
        assert 0 <= result.min() and result.max() <= 255


def test_a_shown_range_stays_inside_the_byte_range():
    from am_segmentation_manager.display import apply_intensity_range

    image = np.array([[0, 128, 255]], dtype=np.uint8)

    for low, high in ((0, 10), (200, 255), (10, 250)):
        result = apply_intensity_range(image, low, high)
        assert result.dtype == np.uint8
        assert 0 <= result.min() and result.max() <= 255
