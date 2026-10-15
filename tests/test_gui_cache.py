"""Tests for the display cache and its prefetch policy.

Loaders are injected throughout, so nothing here reads a file or needs the
optional image dependencies.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.frames import LamellaFrame
from am_segmentation_manager.gui.cache import (
    MIN_PLOT_LIMIT,
    BoundedCache,
    FrameCache,
    LoadRequest,
    prefetch_order,
)
from am_segmentation_manager.io.atomic import fingerprint_of
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
        sem=(ImageRecord(image_path=Path(f"sem/img_{number:03d}_SEM.tif"), modality="SEM"),)
        if sem else (),
        fib=(ImageRecord(image_path=Path(f"fib/img_{number:03d}_FIB.tif"), modality="FIB"),)
        if fib else (),
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
        LamellaFrame(lamella="l", stage="s", stage_number="01", image_number=0, sem=(record,))
    ])

    cache.image(0, "SEM")

    assert calls == [(Path("sem/img.tif"), Path("labels/img.tif"))]


def labelled_cache(label: Path, loader) -> FrameCache:
    """A cache over one SEM frame whose label is *label*."""
    record = ImageRecord(image_path=Path("sem/img.tif"), labels=[label], modality="SEM")
    cache = FrameCache(loader, lambda path: "plot")
    cache.set_frames([
        LamellaFrame(lamella="l", stage="s", stage_number="01", image_number=0, sem=(record,))
    ])
    return cache


def test_the_label_is_fingerprinted_before_it_is_read(tmp_path):
    """A rewrite landing during the read must leave the frame looking stale."""
    label = tmp_path / "img.tif"
    label.write_bytes(b"before")

    def rewritten_while_reading(image_path, label_path=None):
        replacement = tmp_path / "replacement.tif"
        replacement.write_bytes(b"after, and longer")
        replacement.replace(label)
        return "prepared"

    cache = labelled_cache(label, rewritten_while_reading)
    prepared = cache.image(0, "SEM")

    assert prepared.label_fingerprint is not None
    assert prepared.label_fingerprint != fingerprint_of(label)


def test_a_frame_with_no_label_has_no_fingerprint():
    cache, _, _ = make_cache(1)

    assert cache.image(0, "SEM").label_fingerprint is None


def test_relabelling_takes_the_written_label_s_fingerprint(tmp_path):
    np = pytest.importorskip("numpy")
    from am_segmentation_manager.display import DisplayImage

    label = tmp_path / "img.tif"
    label.write_bytes(b"before")

    def load(image_path, label_path=None):
        return DisplayImage(
            image=np.zeros((4, 4), np.uint8), label=np.zeros((4, 4), np.uint8),
            label_status="empty", source=image_path,
        )

    cache = labelled_cache(label, load)
    cache.image(0, "SEM")
    label.write_bytes(b"written by this window")

    cache.relabel({label: np.ones((4, 4), np.uint8)})

    assert cache.cached_image(0, "SEM").label_fingerprint == fingerprint_of(label)


def test_a_frame_whose_label_was_rewritten_is_reported(tmp_path):
    label = tmp_path / "img.tif"
    label.write_bytes(b"before")
    cache = labelled_cache(label, lambda image_path, label_path=None: "prepared")
    cache.image(0, "SEM")
    assert not cache.read_before_rewrite(0, "SEM")

    label.write_bytes(b"rewritten, and longer")

    assert cache.read_before_rewrite(0, "SEM")


def test_a_label_named_since_the_read_is_reported(tmp_path):
    """What a conversion does: the record gains a label it did not name."""
    cache, _, _ = make_cache(1)
    cache.image(0, "SEM")
    label = tmp_path / "img.tif"
    label.write_bytes(b"converted")

    cache.frames[0].sem[0].labels.append(label)

    assert cache.read_before_rewrite(0, "SEM")


def test_a_failed_read_is_not_reported_however_its_label_changed(tmp_path):
    """Or a file that will not read is read again on every repaint."""
    label = tmp_path / "img.tif"
    label.write_bytes(b"before")

    def fail(image_path, label_path=None):
        raise OSError("will not decode")

    cache = labelled_cache(label, fail)
    cache.image(0, "SEM")
    label.write_bytes(b"rewritten, and longer")

    assert not cache.read_before_rewrite(0, "SEM")


def test_a_frame_not_held_is_not_reported():
    cache, _, _ = make_cache(1)

    assert not cache.read_before_rewrite(0, "SEM")


def test_a_dropped_frame_is_read_again():
    cache, loader, _ = make_cache(2)
    cache.image(0, "SEM")

    assert cache.drop_image(0, "SEM")
    assert not cache.drop_image(0, "SEM"), "nothing was held the second time"
    cache.image(0, "SEM")

    assert len(loader.calls) == 2


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

def test_a_plot_is_read_at_the_size_it_was_rendered(tmp_path):
    """Reducing it on read puts a ceiling on what zooming in can ever show,
    and these figures are looked at closely for their annotations."""
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

    assert cache.plot(0).plot.shape[:2] == (900, 1200)


def test_the_plot_being_looked_at_is_read_whole_however_small_the_stepping_copy(tmp_path):
    """Stepping holds every plot reduced, and zooming into one has to find the
    whole figure: reducing it on read puts a ceiling on what looking closely
    can show."""
    pytest.importorskip("cv2")
    import cv2
    import numpy as np

    from am_segmentation_manager.display import DEFAULT_PLOT_MAX_EDGE
    from am_segmentation_manager.frames import LamellaFrame

    plot_path = tmp_path / "img_000_plot.png"
    cv2.imwrite(str(plot_path), np.zeros((1800, 2400, 3), dtype=np.uint8))
    cache = FrameCache()
    cache.set_frames([
        LamellaFrame(lamella="l", stage="s", stage_number="01", image_number=0,
                     plot_path=plot_path)
    ])

    stepping = cache.plot(0).plot
    whole = cache.plot_detail(0).plot

    assert max(stepping.shape[:2]) == DEFAULT_PLOT_MAX_EDGE, "reduced, for stepping"
    assert whole.shape[:2] == (1800, 2400)
    assert cache.cached_plot_detail(0) is not None, "and it is held, not re-read"


def test_whole_lamella_of_plots_can_be_held():
    """Holding an arrow key outruns any per-plot read, so all are kept."""
    cache, _, _ = make_cache(40)

    for index in range(40):
        cache.plot(index)

    assert cache.cached_plot(0) is not None
    assert cache.cached_plot(39) is not None


def test_the_read_ahead_never_asks_for_more_than_it_can_keep():
    """The read that finishes last would evict the one that finished first, so
    a step in either direction lands on something prepared and thrown away."""
    cache, _, _ = make_cache(80)

    wanted = cache.pending(40, [], radius=cache.plot_radius, include_plots=True)

    stepping = [r for r in wanted if r.modality is None and not r.detail]
    assert len(stepping) <= cache._plots.max_entries


def test_a_lamella_that_fits_is_still_prepared_whole():
    """Bounding the read-ahead must not narrow the common case: almost every
    lamella is shorter than the cache."""
    cache, _, _ = make_cache(12)

    wanted = cache.pending(6, [], radius=cache.plot_radius, include_plots=True)

    stepping = [r for r in wanted if r.modality is None and not r.detail]
    assert len(stepping) == 12


def test_the_full_resolution_copy_is_asked_for_last_and_only_where_it_is_looked_at():
    """It is the slowest read here, and nothing on screen is waiting on it --
    so it must not come between a step and the plot that step draws."""
    cache, _, _ = make_cache(6)

    requests = cache.pending(2, [], include_plots=True)

    detail = [r for r in requests if r.detail]
    assert [r.frame_index for r in detail] == [2]
    assert requests[-1].detail is True


def test_only_a_couple_of_full_resolution_plots_are_kept():
    """Near 40 MB each: what is being looked at, and what was just left."""
    from am_segmentation_manager.gui.cache import DETAIL_PLOT_LIMIT

    cache, _, _ = make_cache(6)

    assert cache._details.max_entries == DETAIL_PLOT_LIMIT
    assert DETAIL_PLOT_LIMIT < MIN_PLOT_LIMIT


def test_plot_limit_follows_the_memory_budget():
    """Their size belongs to whatever rendered them, so a fixed count would
    either waste memory on small figures or exhaust it on large ones."""
    from am_segmentation_manager.gui.cache import (
    MAX_PLOT_LIMIT,
        MIN_PLOT_LIMIT,
        _plot_limit_for_bytes,
    )

    modest = _plot_limit_for_bytes(1068 * 836 * 3)
    large = _plot_limit_for_bytes(3200 * 2400 * 3)

    assert large < modest
    assert MIN_PLOT_LIMIT <= large <= MAX_PLOT_LIMIT
