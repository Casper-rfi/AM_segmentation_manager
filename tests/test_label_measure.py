"""Measuring labels in physical units.

The behaviour worth protecting here is the scaling. A pixel count means nothing
across magnifications, and a label converted from a prediction need not share
its image's dimensions -- so a measurement that assumed 1:1 would be wrong by a
constant factor, in a direction nothing in the output would reveal.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from am_segmentation_manager import parallel
from am_segmentation_manager.config import (
    DEFAULT_MIN_CRACK_AREA_UM2,
    SegmentationConfig,
    load_segmentation_config,
)
from am_segmentation_manager.image_metadata import ImageMetadata
from am_segmentation_manager.operations import label_measure
from am_segmentation_manager.operations.label_measure import (
    clear_cache,
    find_specks,
    label_pixel_size_nm,
    measure_label,
    read_label_pixels,
    read_many_label_pixels,
    records_with_specks,
)
from am_segmentation_manager.scanner import Scanner
from conftest import build_dataset

HAS_IMAGE_STACK = all(
    importlib.util.find_spec(name) is not None for name in ("numpy", "tifffile", "cv2")
)
pytestmark = pytest.mark.skipif(
    not HAS_IMAGE_STACK, reason="optional image libs not installed"
)

if HAS_IMAGE_STACK:
    import numpy as np
    import tifffile

CRACK_CLASS = 3
LAMELLA_CLASS = 2


@pytest.fixture
def config() -> SegmentationConfig:
    """The bundled configuration, which is what real runs use."""
    return load_segmentation_config()


@pytest.fixture(autouse=True)
def _clear_measurement_cache():
    """Keep cached counts from leaking between tests.

    The cache keys on mtime, and files written in quick succession inside one
    tmp_path can land on the same timestamp granularity.
    """
    clear_cache()
    yield
    clear_cache()


def write_label(path: Path, array) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), array)
    return path


def label_with_squares(shape=(100, 100), squares=((10, 10, 20),)) -> np.ndarray:
    """A label of *shape* holding square cracks given as ``(row, col, side)``."""
    array = np.zeros(shape, dtype=np.uint8)
    for row, col, side in squares:
        array[row : row + side, col : col + side] = CRACK_CLASS
    return array


# ---------------------------------------------------------------------------
# Pixel counting
# ---------------------------------------------------------------------------

def test_counts_pixels_per_class(tmp_path: Path):
    array = np.zeros((10, 20), dtype=np.uint8)
    array[:2] = LAMELLA_CLASS  # 40 pixels
    array[5:6] = CRACK_CLASS  # 20 pixels
    counts = read_label_pixels(write_label(tmp_path / "l.tif", array), CRACK_CLASS)

    assert counts is not None
    assert counts.width == 20
    assert counts.height == 10
    assert counts.class_pixels == {0: 140, LAMELLA_CLASS: 40, CRACK_CLASS: 20}
    assert sum(counts.class_pixels.values()) == counts.pixels


def test_a_label_wider_than_a_byte_is_still_counted(tmp_path: Path):
    """Counting takes cv2's histogram for uint8 and numpy's otherwise. A label
    with more than 256 classes is unlikely, but a wrong count would show up as
    a plausible area rather than as a failure."""
    array = np.zeros((10, 20), dtype=np.uint16)
    array[:2] = 300
    counts = read_label_pixels(write_label(tmp_path / "l.tif", array), None)

    assert counts is not None
    assert counts.class_pixels == {0: 160, 300: 40}


def test_counting_stays_exact_past_a_float32_mantissa(tmp_path: Path, monkeypatch):
    """cv2 returns a float32 histogram, which represents integers exactly only
    to 2**24. Past that the count has to come from numpy, or a class covering
    most of a large label comes back off by one."""
    from am_segmentation_manager.operations import label_measure

    # Standing in for a label too large to build in a test: the guard is on the
    # pixel count, so lowering it exercises the same branch.
    monkeypatch.setattr(label_measure, "_EXACT_IN_FLOAT32", 4)
    array = np.zeros((10, 20), dtype=np.uint8)
    array[:2] = LAMELLA_CLASS
    counts = read_label_pixels(write_label(tmp_path / "l.tif", array), None)

    assert counts is not None
    assert counts.class_pixels == {0: 160, LAMELLA_CLASS: 40}


def test_separate_regions_are_separate_components(tmp_path: Path):
    array = label_with_squares(squares=((0, 0, 10), (50, 50, 4)))
    counts = read_label_pixels(write_label(tmp_path / "l.tif", array), CRACK_CLASS)

    assert counts is not None
    assert counts.component_pixels == (100, 16)  # largest first


def test_diagonally_touching_regions_are_one_component(tmp_path: Path):
    """Eight-connected: a diagonal step is a pixel-grid artefact, not a crack."""
    array = label_with_squares(squares=((0, 0, 4), (4, 4, 4)))
    counts = read_label_pixels(write_label(tmp_path / "l.tif", array), CRACK_CLASS)

    assert counts is not None
    assert counts.component_pixels == (32,)


def test_no_crack_class_skips_component_labelling(tmp_path: Path):
    array = label_with_squares()
    counts = read_label_pixels(write_label(tmp_path / "l.tif", array), None)

    assert counts is not None
    assert counts.crack_class is None
    assert counts.component_pixels == ()
    assert counts.class_pixels[CRACK_CLASS] == 400  # still counted as area


def test_unreadable_label_is_none_not_empty(tmp_path: Path):
    missing = read_label_pixels(tmp_path / "absent.tif", CRACK_CLASS)
    assert missing is None

    corrupt = tmp_path / "corrupt.tif"
    corrupt.write_bytes(b"not a tiff")
    assert read_label_pixels(corrupt, CRACK_CLASS) is None


def test_non_integer_label_is_refused(tmp_path: Path):
    """Class indices are discrete; a float image is not a label mask."""
    array = np.zeros((8, 8), dtype=np.float32)
    assert read_label_pixels(write_label(tmp_path / "l.tif", array), CRACK_CLASS) is None


# ---------------------------------------------------------------------------
# Scale: the label has no header of its own
# ---------------------------------------------------------------------------

def test_pixel_size_matches_the_image_when_dimensions_agree():
    metadata = ImageMetadata(image_width=1000, pixel_size_nm=10.0)
    assert label_pixel_size_nm(metadata, 1000) == pytest.approx(10.0)


def test_pixel_size_scales_with_a_resized_label():
    """A label half as wide covers the same field, so its pixels are twice the size."""
    metadata = ImageMetadata(image_width=1000, pixel_size_nm=10.0)
    assert label_pixel_size_nm(metadata, 500) == pytest.approx(20.0)


def test_pixel_size_falls_back_to_field_width():
    """hfw_um needs nothing from the image's pixel grid."""
    metadata = ImageMetadata(hfw_um=20.0)
    assert label_pixel_size_nm(metadata, 1000) == pytest.approx(20.0)


def test_pixel_size_is_unknown_rather_than_assumed():
    assert label_pixel_size_nm(ImageMetadata(), 1000) is None
    assert label_pixel_size_nm(None, 1000) is None
    assert label_pixel_size_nm(ImageMetadata(pixel_size_nm=10.0), 0) is None


# ---------------------------------------------------------------------------
# Measurement in physical units
# ---------------------------------------------------------------------------

def test_areas_are_in_square_micrometres(tmp_path: Path, config: SegmentationConfig):
    # 100 nm pixels: one pixel is 0.01 um^2, so a 20x20 square is 4 um^2.
    array = label_with_squares(squares=((10, 10, 20),))
    path = write_label(tmp_path / "l.tif", array)
    metadata = ImageMetadata(image_width=100, pixel_size_nm=100.0)

    measured = measure_label(path, metadata, config)

    assert measured is not None
    assert measured.pixel_size_nm == pytest.approx(100.0)
    assert measured.areas_um2[CRACK_CLASS] == pytest.approx(4.0)
    assert measured.crack_count == 1
    assert measured.crack_areas_um2 == pytest.approx((4.0,))
    assert measured.has_crack


def test_a_resized_label_measures_the_same_feature(tmp_path: Path, config: SegmentationConfig):
    """The whole point of the scaling: the same crack, at half the label size."""
    metadata = ImageMetadata(image_width=200, pixel_size_nm=50.0)

    full = write_label(tmp_path / "full.tif", label_with_squares((200, 200), ((0, 0, 40),)))
    half = write_label(tmp_path / "half.tif", label_with_squares((100, 100), ((0, 0, 20),)))

    full_area = measure_label(full, metadata, config).crack_areas_um2[0]
    half_area = measure_label(half, metadata, config).crack_areas_um2[0]

    assert full_area == pytest.approx(4.0)
    assert half_area == pytest.approx(full_area)


def test_components_below_the_threshold_are_noise(tmp_path: Path, config: SegmentationConfig):
    # 100 nm pixels: the 20x20 square is 4 um^2, the 2x2 speck is 0.04 um^2,
    # below the 0.1 um^2 default.
    array = label_with_squares(squares=((10, 10, 20), (80, 80, 2)))
    path = write_label(tmp_path / "l.tif", array)
    metadata = ImageMetadata(image_width=100, pixel_size_nm=100.0)

    measured = measure_label(path, metadata, config)

    assert measured.pixels.component_pixels == (400, 4)  # both found
    assert measured.crack_count == 1  # one survives the threshold
    assert measured.min_crack_area_um2 == pytest.approx(DEFAULT_MIN_CRACK_AREA_UM2)


def test_threshold_can_be_overridden(tmp_path: Path, config: SegmentationConfig):
    array = label_with_squares(squares=((10, 10, 20), (80, 80, 2)))
    path = write_label(tmp_path / "l.tif", array)
    metadata = ImageMetadata(image_width=100, pixel_size_nm=100.0)

    measured = measure_label(path, metadata, config, min_crack_area_um2=0.01)

    assert measured.crack_count == 2
    assert measured.min_crack_area_um2 == pytest.approx(0.01)


def test_empty_label_is_measured_as_zero_cracks(tmp_path: Path, config: SegmentationConfig):
    path = write_label(tmp_path / "l.tif", np.zeros((50, 50), dtype=np.uint8))
    metadata = ImageMetadata(image_width=50, pixel_size_nm=100.0)

    measured = measure_label(path, metadata, config)

    assert measured.crack_count == 0  # measured, and nothing found
    assert not measured.has_crack
    assert measured.areas_um2 == {0: pytest.approx(25.0)}


def test_unknown_scale_gives_no_areas_rather_than_pixel_areas(
    tmp_path: Path, config: SegmentationConfig
):
    """An unmeasurable label must be visible as such, not silently in pixels."""
    path = write_label(tmp_path / "l.tif", label_with_squares())
    measured = measure_label(path, ImageMetadata(), config)

    assert measured is not None
    assert measured.pixel_size_nm is None
    assert measured.areas_um2 == {}
    assert measured.crack_count is None  # not zero: the threshold is physical
    assert not measured.has_crack
    assert measured.pixels.component_pixels == (400,)  # the pixel counts survive


def test_unreadable_label_measures_as_none(tmp_path: Path, config: SegmentationConfig):
    metadata = ImageMetadata(image_width=100, pixel_size_nm=100.0)
    assert measure_label(tmp_path / "absent.tif", metadata, config) is None


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def test_cache_returns_the_same_counts(tmp_path: Path):
    path = write_label(tmp_path / "l.tif", label_with_squares())

    first = read_label_pixels(path, CRACK_CLASS)
    second = read_label_pixels(path, CRACK_CLASS)

    assert first is second  # served from the cache, not re-read


def test_cache_notices_an_edited_label(tmp_path: Path):
    """Proofreading can change a label while this process is still running."""
    path = write_label(tmp_path / "l.tif", label_with_squares(squares=((0, 0, 10),)))
    assert read_label_pixels(path, CRACK_CLASS).component_pixels == (100,)

    import os

    write_label(path, label_with_squares(squares=((0, 0, 20),)))
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    assert read_label_pixels(path, CRACK_CLASS).component_pixels == (400,)


def test_cache_can_be_bypassed(tmp_path: Path):
    path = write_label(tmp_path / "l.tif", label_with_squares())

    first = read_label_pixels(path, CRACK_CLASS, use_cache=False)
    second = read_label_pixels(path, CRACK_CLASS, use_cache=False)

    assert first is not second
    assert first == second


# ---------------------------------------------------------------------------
# Specks: components too small to be real
# ---------------------------------------------------------------------------

def speckled():
    """A label holding one real structure and two components of a few pixels."""
    label = np.zeros((60, 80), dtype=np.uint8)
    label[10:40, 10:40] = LAMELLA_CLASS   # a real structure
    label[50, 70] = CRACK_CLASS           # one stray pixel
    label[52:54, 60:62] = LAMELLA_CLASS   # one pixel of an upscaled prediction
    return label


def test_a_stray_pixel_is_found():
    specks = find_specks(speckled())

    assert [(speck.class_index, speck.pixels) for speck in specks] == [
        (CRACK_CLASS, 1), (LAMELLA_CLASS, 4)
    ], "smallest first, so the least defensible is looked at first"


def test_a_real_structure_is_not_a_speck():
    assert all(speck.pixels <= 4 for speck in find_specks(speckled()))


def test_the_class_does_not_decide_whether_it_is_a_speck():
    """A stray pixel is a stray pixel whichever class it claims to be."""
    specks = find_specks(speckled())

    assert {speck.class_index for speck in specks} == {CRACK_CLASS, LAMELLA_CLASS}


def test_a_hole_inside_an_annotation_is_a_speck():
    label = np.full((20, 20), LAMELLA_CLASS, dtype=np.uint8)
    label[5, 5] = 0

    specks = find_specks(label)

    assert [(speck.class_index, speck.pixels) for speck in specks] == [(0, 1)], (
        "a stray pixel of vacuum in the lamella is the same mistake, seen from "
        "the other side"
    )


def test_the_background_of_a_frame_is_not_a_speck():
    """The annotation is drawn well past the threshold on purpose: what is
    being asked is whether the *background* around it is reported, and a block
    small enough to be a speck itself would answer a different question."""
    label = np.zeros((60, 60), dtype=np.uint8)
    label[10:40, 10:40] = LAMELLA_CLASS

    assert find_specks(label) == (), (
        "the background around an annotation is one enormous component"
    )


def test_a_diagonal_staircase_of_holes_is_found():
    """Eight-connected background would join these up and leak them out to the
    background around the block, which is how they went unreported."""
    label = np.full((20, 20), LAMELLA_CLASS, dtype=np.uint8)
    for step in range(5):
        label[5 + step, 5 + step] = 0

    specks = find_specks(label)

    assert len(specks) == 5
    assert all(speck.pixels == 1 for speck in specks)


def test_the_threshold_includes_its_own_size():
    label = np.zeros((20, 20), dtype=np.uint8)
    label[5:7, 5:7] = LAMELLA_CLASS

    assert len(find_specks(label, at_most=4)) == 1
    assert find_specks(label, at_most=3) == ()


def test_a_speck_says_where_to_look():
    label = np.zeros((60, 80), dtype=np.uint8)
    label[30, 50] = LAMELLA_CLASS

    speck = find_specks(label)[0]

    assert (speck.x, speck.y) == (50, 30)


def test_a_diagonal_pair_is_one_speck_not_two():
    """Eight-connected, as the crack statistics already are."""
    label = np.zeros((20, 20), dtype=np.uint8)
    label[5, 5] = LAMELLA_CLASS
    label[6, 6] = LAMELLA_CLASS

    assert [speck.pixels for speck in find_specks(label)] == [2]


def test_a_label_is_swept_once_however_many_images_share_it(tmp_path: Path):
    dataset = build_dataset(tmp_path, acquisitions=1, sem_variants=("sem_8_fi_1",))
    label = np.zeros((32, 40), dtype=np.uint8)
    label[10, 10] = LAMELLA_CLASS
    for path in (dataset.label("SEM", 0), dataset.label("FIB", 0)):
        tifffile.imwrite(str(path), label)
    records = [
        record for project in Scanner(tmp_path).build_index().all_projects()
        for lamella in project.lamellae
        for stage in lamella.stages
        for record in stage.images
    ]

    found = records_with_specks(records)

    assert len(records) > len(found), "the conditions of one scene share a label"
    assert set(found) == {dataset.label("SEM", 0), dataset.label("FIB", 0)}


def test_a_record_with_no_label_is_absent(tmp_path: Path):
    build_dataset(tmp_path, acquisitions=1, labels="none")
    records = [
        record for project in Scanner(tmp_path).build_index().all_projects()
        for lamella in project.lamellae
        for stage in lamella.stages
        for record in stage.images
    ]

    assert records_with_specks(records) == {}


# ---------------------------------------------------------------------------
# Counting many labels at once
# ---------------------------------------------------------------------------

def test_labels_counted_in_worker_processes_match_those_counted_here(
    tmp_path: Path, monkeypatch
):
    paths = [
        write_label(tmp_path / f"l{index}.tif", label_with_squares(squares=((index, 5, 4 + index),)))
        for index in range(3)
    ]
    corrupt = tmp_path / "corrupt.tif"
    corrupt.write_bytes(b"not a tiff")
    expected = {
        path: read_label_pixels(path, CRACK_CLASS, use_cache=False) for path in [*paths, corrupt]
    }
    parallel.shutdown_pool()
    monkeypatch.setattr(label_measure, "LABEL_POOL_FILES", 0)
    monkeypatch.setattr(parallel, "POOL_WORKERS", 2)
    seen: list[tuple[int, int]] = []
    try:
        counted = read_many_label_pixels(
            [*paths, corrupt], CRACK_CLASS, on_progress=lambda *step: seen.append(step)
        )
        assert parallel._pool is not None
    finally:
        parallel.shutdown_pool()

    assert counted == expected
    assert seen[-1] == (4, 4)


def test_labels_counted_together_fill_the_cache_one_is_read_from(
    tmp_path: Path, monkeypatch
):
    """Which is what lets a breakdown after the label filter open nothing."""
    path = write_label(tmp_path / "l.tif", label_with_squares())
    read_many_label_pixels([path], CRACK_CLASS)
    monkeypatch.setattr(
        label_measure, "_count_pixels", lambda *args: pytest.fail("read the label again")
    )
    seen: list[tuple[int, int]] = []

    assert read_label_pixels(path, CRACK_CLASS).component_pixels == (400,)
    assert read_many_label_pixels([path], CRACK_CLASS, on_progress=lambda *step: seen.append(step))
    assert seen == [], "a label the cache answers is not counted as read"
