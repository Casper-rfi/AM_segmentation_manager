"""Tests for grouping loose images into acquisitions and per-series timelines.

Outside an Adaptive Polishing stage a filename carries no acquisition number,
and the two beams of one moment differ only by their suffix. These check that
such a pair becomes one acquisition, that an Adaptive Polishing one is
untouched by that rule, and that a lamella's timeline can be taken one series
at a time -- a 2026 lamella holds three unrelated ones at once.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.frames import build_lamella_frames, group_by_acquisition
from am_segmentation_manager.models import ImageRecord, Lamella, Stage

STAGE = Path("/p/01-ruling-ibex")
ADAPTIVE_STAGE = STAGE / "Milling/Polishing/adaptive_polish_2026-07-28-11-34-24AM"


def loose(name: str, modality: str, series: str = "reference") -> ImageRecord:
    return ImageRecord(
        image_path=STAGE / name, modality=modality, series=series, stage_path=STAGE
    )


def adaptive(name: str, modality: str) -> ImageRecord:
    directory = "sem" if modality == "SEM" else "fib"
    return ImageRecord(
        image_path=ADAPTIVE_STAGE / directory / name,
        modality=modality,
        series="adaptive",
        stage_path=ADAPTIVE_STAGE,
    )


# ---------------------------------------------------------------------------
# Pairing the two beams of one moment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "stem",
    [
        "ref_Polishing_start",
        "ref_Setup Lamella Position_final_res_01",
        "ref_MillRough_final_high_res",
        "ref_milling_Fine-1_finished_1750981637_3822749",
        "Polishing-Fine-1_finished_11-04-43",
    ],
)
def test_both_beams_of_one_moment_become_one_acquisition(stem):
    records = [loose(f"{stem}_eb.tif", "SEM"), loose(f"{stem}_ib.tif", "FIB")]

    acquisitions = group_by_acquisition(records)

    assert len(acquisitions) == 1
    assert acquisitions[0].sem and acquisitions[0].fib


def test_different_moments_stay_apart():
    records = [
        loose("ref_Polishing_start_eb.tif", "SEM"),
        loose("ref_Polishing_final_res_01_eb.tif", "SEM"),
    ]

    assert len(group_by_acquisition(records)) == 2


def test_an_image_with_no_partner_is_an_acquisition_of_one():
    """``ref_PositionReady.tif`` is ion-beam only, one per lamella."""
    acquisitions = group_by_acquisition([loose("ref_PositionReady.tif", "FIB")])

    assert len(acquisitions) == 1
    assert not acquisitions[0].sem


def test_successive_alignment_attempts_are_separate_acquisitions():
    records = [
        loose("new_image_00.tif", "FIB", "alignment"),
        loose("new_image_01.tif", "FIB", "alignment"),
        loose("reference_image.tif", "FIB", "alignment"),
    ]

    assert len(group_by_acquisition(records)) == 3


def test_adaptive_images_still_pair_by_their_acquisition_number():
    """Their filenames do not share a stem, so the number is what matches them."""
    records = [
        adaptive("01-ruling-ibex_AP_img_013_SEM.tif", "SEM"),
        adaptive("01-ruling-ibex_AP_img_013_FIB.tif", "FIB"),
    ]

    acquisitions = group_by_acquisition(records)

    assert len(acquisitions) == 1
    assert acquisitions[0].image_number == 13


def test_an_acquisition_reports_the_series_of_its_images():
    records = [loose("ref_Polishing_start_eb.tif", "SEM"), loose("ref_Polishing_start_ib.tif", "FIB")]

    assert group_by_acquisition(records)[0].series == "reference"


# ---------------------------------------------------------------------------
# One timeline at a time
# ---------------------------------------------------------------------------

@pytest.fixture
def lamella() -> Lamella:
    """A 2026 lamella: three series across two stages."""
    loose_stage = Stage(
        name="01-ruling-ibex",
        path=STAGE,
        number="01",
        synthetic=True,
        images=[
            loose("ref_Polishing_start_eb.tif", "SEM"),
            loose("ref_Polishing_start_ib.tif", "FIB"),
            loose("ref_alignment_ib.tif", "FIB", "alignment"),
        ],
    )
    polish = Stage(
        name="adaptive_polish_2026-07-28-11-34-24AM",
        path=ADAPTIVE_STAGE,
        number="02",
        images=[
            adaptive("01-ruling-ibex_AP_img_000_SEM.tif", "SEM"),
            adaptive("01-ruling-ibex_AP_img_000_FIB.tif", "FIB"),
        ],
    )
    return Lamella(name="01-ruling-ibex", path=STAGE, stages=[loose_stage, polish])


def test_the_whole_timeline_holds_every_series(lamella: Lamella):
    frames = build_lamella_frames(lamella)

    assert [frame.series for frame in frames] == ["reference", "alignment", "adaptive"]


@pytest.mark.parametrize(
    "series, expected", [("adaptive", 1), ("reference", 1), ("alignment", 1)]
)
def test_a_timeline_can_be_taken_one_series_at_a_time(lamella, series, expected):
    frames = build_lamella_frames(lamella, series=series)

    assert len(frames) == expected
    assert all(frame.series == series for frame in frames)


def test_asking_for_a_series_a_lamella_lacks_gives_no_frames(lamella: Lamella):
    lamella.stages = lamella.stages[1:]

    assert build_lamella_frames(lamella, series="reference") == []
