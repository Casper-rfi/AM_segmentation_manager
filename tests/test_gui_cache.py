"""Tests for the display cache and its prefetch policy.

Loaders are injected throughout, so nothing here reads a file or needs the
optional image dependencies.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.frames import LamellaFrame
from am_segmentation_manager.gui.cache import (
    BoundedCache,
    FrameCache,
    LoadRequest,
    prefetch_order,
)
from am_segmentation_manager.models import ImageRecord


class RecordingLoader:
    """Stands in for the display pipeline and records what it was asked for."""

    def __init__(self, fail_for: set[str] | None = None) -> None:
        self.calls: list[Path] = []
        self.fail_for = fail_for or set()

    def __call__(self, path: Path, label_path: Path | None = None):
        self.calls.append(path)
        if path.name in self.fail_for:
            raise OSError(f"could not decode {path.name}")
        return f"prepared:{path.name}"


def make_frame(number: int, *, sem=True, fib=True, plot=True) -> LamellaFrame:
    return LamellaFrame(
        lamella="01-test",
        stage="adaptive_polish_1",
        stage_number="01",
        image_number=number,
        sem=ImageRecord(image_path=Path(f"sem/img_{number:03d}_SEM.tif"), modality="SEM")
        if sem else None,
        fib=ImageRecord(image_path=Path(f"fib/img_{number:03d}_FIB.tif"), modality="FIB")
        if fib else None,
        plot_path=Path(f"plots/img_{number:03d}_plot.png") if plot else None,
    )


def make_cache(count: int = 5, **kwargs) -> tuple[FrameCache, RecordingLoader, RecordingLoader]:
    image_loader = RecordingLoader()
    plot_loader = RecordingLoader()
    cache = FrameCache(image_loader, plot_loader, **kwargs)
    cache.set_frames([make_frame(n) for n in range(count)])
    return cache, image_loader, plot_loader


# ---------------------------------------------------------------------------
# BoundedCache
# ---------------------------------------------------------------------------

def test_bounded_cache_evicts_least_recently_used():
    cache: BoundedCache[str, int] = BoundedCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)

    assert "a" not in cache
    assert cache.keys() == ["b", "c"]


def test_reading_an_entry_protects_it_from_eviction():
    cache: BoundedCache[str, int] = BoundedCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.put("c", 3)

    assert "a" in cache
    assert "b" not in cache


def test_unbounded_cache_keeps_everything():
    cache: BoundedCache[int, int] = BoundedCache(None)
    for value in range(100):
        cache.put(value, value)

    assert len(cache) == 100


def test_cache_limit_must_be_positive():
    with pytest.raises(ValueError):
        BoundedCache(0)


def test_missing_key_returns_none():
    assert BoundedCache(2).get("absent") is None


# ---------------------------------------------------------------------------
# Prefetch order
# ---------------------------------------------------------------------------

def test_prefetch_order_alternates_outwards_from_the_current_frame():
    """A user may reverse at any point, so both directions are prepared."""
    assert prefetch_order(2, 5) == [2, 3, 1, 4, 0]


def test_prefetch_order_is_clipped_at_the_ends():
    assert prefetch_order(0, 3) == [0, 1, 2]
    assert prefetch_order(2, 3) == [2, 1, 0]


def test_prefetch_radius_limits_how_far_ahead():
    assert prefetch_order(5, 20, radius=2) == [5, 6, 4, 7, 3]


def test_prefetch_order_handles_an_empty_lamella():
    assert prefetch_order(0, 0) == []


def test_prefetch_order_clamps_an_out_of_range_current():
    assert prefetch_order(99, 3) == [2, 1, 0]


# ---------------------------------------------------------------------------
# Loading and caching
# ---------------------------------------------------------------------------

def test_image_is_prepared_once_and_reused():
    cache, image_loader, _ = make_cache()

    first = cache.image(0, "SEM")
    second = cache.image(0, "SEM")

    assert first is second
    assert first.display == "prepared:img_000_SEM.tif"
    assert len(image_loader.calls) == 1


def test_label_path_is_passed_to_the_loader():
    calls = []

    def loader(image_path, label_path=None):
        calls.append((image_path, label_path))
        return "prepared"

    cache = FrameCache(loader, lambda path: "plot")
    record = ImageRecord(
        image_path=Path("sem/img.tif"),
        labels=[Path("labels/img.tif")],
        modality="SEM",
    )
    cache.set_frames([
        LamellaFrame(lamella="l", stage="s", stage_number="01", image_number=0, sem=record)
    ])

    cache.image(0, "SEM")

    assert calls == [(Path("sem/img.tif"), Path("labels/img.tif"))]


def test_absent_modality_returns_none_without_loading():
    """Not every acquisition has both modalities; that is not a failure."""
    image_loader = RecordingLoader()
    cache = FrameCache(image_loader, RecordingLoader())
    cache.set_frames([make_frame(0, fib=False)])

    assert cache.image(0, "FIB") is None
    assert image_loader.calls == []


def test_absent_plot_returns_none():
    cache = FrameCache(RecordingLoader(), RecordingLoader())
    cache.set_frames([make_frame(0, plot=False)])

    assert cache.plot(0) is None


def test_out_of_range_index_returns_none():
    cache, _, _ = make_cache(3)

    assert cache.image(99, "SEM") is None
    assert cache.plot(-1) is None


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------

def test_unreadable_image_is_recorded_not_raised():
    """One bad file must not stop the user navigating past it."""
    image_loader = RecordingLoader(fail_for={"img_001_SEM.tif"})
    cache = FrameCache(image_loader, RecordingLoader())
    cache.set_frames([make_frame(n) for n in range(3)])

    result = cache.image(1, "SEM")

    assert result.failed is True
    assert "could not decode" in result.error
    assert result.display is None
    # Neighbouring frames are unaffected.
    assert cache.image(0, "SEM").failed is False


def test_failures_are_cached_so_slow_reads_are_not_retried():
    image_loader = RecordingLoader(fail_for={"img_000_SEM.tif"})
    cache = FrameCache(image_loader, RecordingLoader())
    cache.set_frames([make_frame(0)])

    cache.image(0, "SEM")
    cache.image(0, "SEM")

    assert len(image_loader.calls) == 1


# ---------------------------------------------------------------------------
# Lamella switching
# ---------------------------------------------------------------------------

def test_setting_frames_discards_the_previous_lamella():
    cache, image_loader, _ = make_cache(3)
    cache.image(0, "SEM")
    assert cache.cached_image(0, "SEM") is not None

    cache.set_frames([make_frame(0)])

    assert cache.cached_image(0, "SEM") is None
    cache.image(0, "SEM")
    assert len(image_loader.calls) == 2


def test_cached_image_does_not_trigger_a_load():
    """Lets a view show 'loading' instead of blocking on a slow read."""
    cache, image_loader, _ = make_cache()

    assert cache.cached_image(0, "SEM") is None
    assert image_loader.calls == []


def test_image_limit_evicts_distant_frames():
    cache, _, _ = make_cache(10, image_limit=3)

    for index in range(5):
        cache.image(index, "SEM")

    assert cache.cached_image(0, "SEM") is None
    assert cache.cached_image(4, "SEM") is not None


def test_plots_have_their_own_smaller_limit():
    """Plots cost several times more per entry than acquisitions."""
    cache, _, plot_loader = make_cache(10, plot_limit=2)

    for index in range(4):
        cache.plot(index)

    assert cache.cached_plot(0) is None
    assert cache.cached_plot(3) is not None
    assert len(plot_loader.calls) == 4


# ---------------------------------------------------------------------------
# Pending work
# ---------------------------------------------------------------------------

def test_pending_lists_the_current_frame_first():
    cache, _, _ = make_cache(5)

    pending = cache.pending(2, ["SEM"], radius=1)

    assert pending == [
        LoadRequest(2, "SEM"),
        LoadRequest(3, "SEM"),
        LoadRequest(1, "SEM"),
    ]


def test_pending_respects_modality_priority():
    cache, _, _ = make_cache(2)

    pending = cache.pending(0, ["FIB", "SEM"], radius=0)

    assert pending == [LoadRequest(0, "FIB"), LoadRequest(0, "SEM")]


def test_pending_omits_what_is_already_held():
    cache, _, _ = make_cache(5)
    cache.image(2, "SEM")

    pending = cache.pending(2, ["SEM"], radius=1)

    assert LoadRequest(2, "SEM") not in pending
    assert pending == [LoadRequest(3, "SEM"), LoadRequest(1, "SEM")]


def test_pending_omits_failures_so_they_are_not_retried_forever():
    image_loader = RecordingLoader(fail_for={"img_000_SEM.tif"})
    cache = FrameCache(image_loader, RecordingLoader())
    cache.set_frames([make_frame(0)])
    cache.image(0, "SEM")

    assert cache.pending(0, ["SEM"]) == []


def test_pending_skips_absent_modalities():
    cache = FrameCache(RecordingLoader(), RecordingLoader())
    cache.set_frames([make_frame(0, fib=False), make_frame(1)])

    pending = cache.pending(0, ["SEM", "FIB"], radius=1)

    assert LoadRequest(0, "FIB") not in pending
    assert LoadRequest(1, "FIB") in pending


def test_plots_are_excluded_from_pending_by_default():
    cache, _, _ = make_cache(3)

    assert all(request.modality is not None for request in cache.pending(0, ["SEM"]))


def test_plots_can_be_included_when_the_plot_view_is_on():
    cache, _, _ = make_cache(3)

    pending = cache.pending(0, ["SEM"], radius=0, include_plots=True)

    assert LoadRequest(0, None) in pending


def test_ensure_performs_the_requested_work():
    cache, image_loader, plot_loader = make_cache(3)

    cache.ensure(LoadRequest(1, "SEM"))
    cache.ensure(LoadRequest(1, None))

    assert cache.cached_image(1, "SEM") is not None
    assert cache.cached_plot(1) is not None
    assert len(image_loader.calls) == 1
    assert len(plot_loader.calls) == 1


# ---------------------------------------------------------------------------
# Plot sizing and whole-lamella preparation
# ---------------------------------------------------------------------------

def test_plot_size_change_discards_prepared_plots():
    """A plot prepared at the wrong size would be resampled again by the view."""
    cache, _, plot_loader = make_cache(3)
    cache.plot_max_edge = 1600
    cache.plot(0)
    assert cache.cached_plot(0) is not None

    assert cache.set_plot_max_edge(1070) is True
    assert cache.cached_plot(0) is None


def test_small_plot_size_changes_are_ignored():
    """Dragging a window edge must not keep throwing the lamella away."""
    cache, _, _ = make_cache(3)
    cache.plot_max_edge = 1000
    cache.plot(0)

    assert cache.set_plot_max_edge(1050) is False
    assert cache.cached_plot(0) is not None


def test_plot_size_is_passed_to_the_real_loader(tmp_path):
    import pytest

    pytest.importorskip("cv2")
    import cv2
    import numpy as np

    from am_segmentation_manager.frames import LamellaFrame

    plot_path = tmp_path / "img_000_plot.png"
    cv2.imwrite(str(plot_path), np.zeros((900, 1200, 3), dtype=np.uint8))

    cache = FrameCache()
    cache.set_frames([
        LamellaFrame(lamella="l", stage="s", stage_number="01", image_number=0,
                     plot_path=plot_path)
    ])
    cache.set_plot_max_edge(600)

    assert max(cache.plot(0).plot.shape[:2]) == 600


def test_whole_lamella_of_plots_can_be_held():
    """Holding an arrow key outruns any per-plot loading, so all are kept."""
    cache, _, _ = make_cache(40)

    for index in range(40):
        cache.plot(index)

    assert cache.cached_plot(0) is not None
    assert cache.cached_plot(39) is not None
