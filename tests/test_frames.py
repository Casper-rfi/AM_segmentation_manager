"""Tests for building per-lamella chronological frames.

These tests construct model objects directly rather than scanning a folder
tree: build_lamella_frames is a pure transformation of the scanned model, and
testing it without the filesystem keeps the cases readable.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.frames import LamellaFrame, build_lamella_frames
from am_segmentation_manager.models import AdaptiveMillingStage, ImageRecord, Lamella


def make_record(name: str, modality: str) -> ImageRecord:
    """Build an image record for a file named *name*."""
    folder = "sem" if modality == "SEM" else "fib"
    return ImageRecord(image_path=Path(folder) / name, modality=modality)


def make_stage(
    name: str,
    number: str,
    images: list[ImageRecord] | None = None,
    plots: list[str] | None = None,
) -> AdaptiveMillingStage:
    return AdaptiveMillingStage(
        name=name,
        path=Path(name),
        number=number,
        images=images or [],
        plots=[Path("plots") / plot for plot in (plots or [])],
    )


def make_lamella(stages: list[AdaptiveMillingStage]) -> Lamella:
    return Lamella(name="01-alive-muskox", path=Path("01-alive-muskox"), number="01", stages=stages)


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

def test_sem_and_fib_of_one_acquisition_share_a_frame():
    stage = make_stage("adaptive_polish_1", "01", [
        make_record("lam_AP_img_000_SEM.tif", "SEM"),
        make_record("lam_AP_img_000_FIB.tif", "FIB"),
    ])

    frames = build_lamella_frames(make_lamella([stage]))

    assert len(frames) == 1
    assert frames[0].sem.image_path.name == "lam_AP_img_000_SEM.tif"
    assert frames[0].fib.image_path.name == "lam_AP_img_000_FIB.tif"
    assert frames[0].image_number == 0


def test_pairing_uses_image_number_not_filename():
    """Older projects name the file inside fib/ with an _SEM suffix too."""
    stage = make_stage("adaptive_polish_1", "01", [
        make_record("09-modern-dove_AP_img_004_SEM.tif", "SEM"),
        make_record("09-modern-dove_AP_img_004_SEM.tif", "FIB"),
    ])

    frames = build_lamella_frames(make_lamella([stage]))

    assert len(frames) == 1
    assert frames[0].sem is not None
    assert frames[0].fib is not None


def test_frame_with_only_one_modality():
    stage = make_stage("adaptive_polish_1", "01", [
        make_record("lam_AP_img_000_SEM.tif", "SEM"),
        make_record("lam_AP_img_001_FIB.tif", "FIB"),
    ])

    frames = build_lamella_frames(make_lamella([stage]))

    assert [(f.image_number, f.sem is None, f.fib is None) for f in frames] == [
        (0, False, True),
        (1, True, False),
    ]


def test_modalities_in_different_states_still_share_a_frame():
    """Curation deselects one modality at a time, so a frame's SEM and FIB
    images are often in different state folders. Grouping must not split them."""
    stage = make_stage("adaptive_polish_1", "01", [
        ImageRecord(
            image_path=Path("sem/deselected/lam_AP_img_000_SEM.tif"),
            modality="SEM", selected=False,
        ),
        ImageRecord(
            image_path=Path("fib/lam_AP_img_000_FIB.tif"), modality="FIB",
        ),
    ])

    frames = build_lamella_frames(make_lamella([stage]))

    assert len(frames) == 1
    assert frames[0].sem.selected is False
    assert frames[0].fib.selected is True


def test_records_property_lists_present_modalities():
    frame = LamellaFrame(
        lamella="01", stage="s", stage_number="01", image_number=0,
        sem=make_record("img_000_SEM.tif", "SEM"),
    )

    assert len(frame.records) == 1
    assert frame.record("FIB") is None
    assert frame.record("sem") is frame.sem


def test_record_rejects_unknown_modality():
    frame = LamellaFrame(lamella="01", stage="s", stage_number="01", image_number=0)

    with pytest.raises(ValueError):
        frame.record("EDX")


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

def test_frames_ordered_by_image_number_within_a_stage():
    stage = make_stage("adaptive_polish_1", "01", [
        make_record("lam_AP_img_010_SEM.tif", "SEM"),
        make_record("lam_AP_img_002_SEM.tif", "SEM"),
        make_record("lam_AP_img_001_SEM.tif", "SEM"),
    ])

    frames = build_lamella_frames(make_lamella([stage]))

    assert [frame.image_number for frame in frames] == [1, 2, 10]


def test_frames_follow_stage_order_across_the_lamella():
    """Stage order comes from the scanner; frames must not re-sort across stages."""
    first = make_stage("adaptive_polish_morning", "01", [
        make_record("lam_AP_img_005_SEM.tif", "SEM"),
    ])
    second = make_stage("adaptive_polish_afternoon", "02", [
        make_record("lam_AP_img_000_SEM.tif", "SEM"),
    ])

    frames = build_lamella_frames(make_lamella([first, second]))

    assert [(f.stage_number, f.image_number) for f in frames] == [("01", 5), ("02", 0)]


def test_unnumbered_images_come_last_ordered_by_name():
    stage = make_stage("adaptive_polish_1", "01", [
        make_record("zzz_reference_SEM.tif", "SEM"),
        make_record("aaa_reference_SEM.tif", "SEM"),
        make_record("lam_AP_img_003_SEM.tif", "SEM"),
    ])

    frames = build_lamella_frames(make_lamella([stage]))

    assert [f.image_number for f in frames] == [3, None, None]
    assert [f.sem.image_path.name for f in frames[1:]] == [
        "aaa_reference_SEM.tif",
        "zzz_reference_SEM.tif",
    ]


def test_unnumbered_sem_and_fib_pair_by_filename_stem():
    stage = make_stage("adaptive_polish_1", "01", [
        make_record("centring.tif", "SEM"),
        make_record("centring.tif", "FIB"),
    ])

    frames = build_lamella_frames(make_lamella([stage]))

    assert len(frames) == 1
    assert frames[0].sem is not None and frames[0].fib is not None


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def test_plots_matched_to_frames_by_image_number():
    stage = make_stage(
        "adaptive_polish_1", "01",
        [make_record("lam_AP_img_000_SEM.tif", "SEM"),
         make_record("lam_AP_img_001_SEM.tif", "SEM")],
        plots=["lam_AP_img_001_plot.png", "lam_AP_img_000_plot.png"],
    )

    frames = build_lamella_frames(make_lamella([stage]))

    assert frames[0].plot_path.name == "lam_AP_img_000_plot.png"
    assert frames[1].plot_path.name == "lam_AP_img_001_plot.png"


def test_missing_plot_is_normal():
    """Stages routinely hold fewer plots than images."""
    stage = make_stage(
        "adaptive_polish_1", "01",
        [make_record("lam_AP_img_000_SEM.tif", "SEM"),
         make_record("lam_AP_img_001_SEM.tif", "SEM")],
        plots=["lam_AP_img_000_plot.png"],
    )

    frames = build_lamella_frames(make_lamella([stage]))

    assert frames[0].plot_path is not None
    assert frames[1].plot_path is None


def test_plot_without_an_image_is_ignored():
    stage = make_stage(
        "adaptive_polish_1", "01",
        [make_record("lam_AP_img_000_SEM.tif", "SEM")],
        plots=["lam_AP_img_000_plot.png", "lam_AP_img_099_plot.png"],
    )

    frames = build_lamella_frames(make_lamella([stage]))

    assert len(frames) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_duplicate_modality_keeps_the_first_and_warns(caplog):
    """Legacy multi-condition layouts must not silently lose an acquisition."""
    stage = make_stage("adaptive_polish_1", "01", [
        make_record("cond_a_img_000_SEM.tif", "SEM"),
        make_record("cond_b_img_000_SEM.tif", "SEM"),
    ])

    with caplog.at_level("WARNING"):
        frames = build_lamella_frames(make_lamella([stage]))

    assert len(frames) == 1
    assert frames[0].sem.image_path.name == "cond_a_img_000_SEM.tif"
    assert "cond_b_img_000_SEM.tif" in caplog.text


def test_record_with_unknown_modality_is_skipped(caplog):
    stage = make_stage("adaptive_polish_1", "01", [
        ImageRecord(image_path=Path("other/img_000.tif"), modality=None),
        make_record("lam_AP_img_000_SEM.tif", "SEM"),
    ])

    with caplog.at_level("WARNING"):
        frames = build_lamella_frames(make_lamella([stage]))

    assert len(frames) == 1
    assert frames[0].sem is not None


def test_lamella_with_no_stages_yields_no_frames():
    assert build_lamella_frames(make_lamella([])) == []


def test_empty_stage_yields_no_frames():
    """A stage folder with no images is normal and must not break the stack."""
    stages = [
        make_stage("adaptive_polish_empty", "01"),
        make_stage("adaptive_polish_2", "02", [make_record("lam_AP_img_000_SEM.tif", "SEM")]),
    ]

    frames = build_lamella_frames(make_lamella(stages))

    assert [f.stage_number for f in frames] == ["02"]


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def test_display_name_uses_stage_and_image_number():
    stage = make_stage("adaptive_polish_1", "02", [
        make_record("lam_AP_img_013_SEM.tif", "SEM"),
    ])

    frames = build_lamella_frames(make_lamella([stage]))

    assert frames[0].display_name == "stage 02 - img 013"


def test_display_name_falls_back_to_filename_when_unnumbered():
    stage = make_stage("adaptive_polish_1", "02", [
        make_record("centring.tif", "SEM"),
    ])

    frames = build_lamella_frames(make_lamella([stage]))

    assert frames[0].display_name == "stage 02 - centring"
