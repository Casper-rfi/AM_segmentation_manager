"""Tests for putting reference and alignment frames in the order taken.

Nothing in a loose image's name orders it. 44% of them carry no timestamp at
all, and the epochs that do appear are when the milling stage started rather
than when the image was taken, so the header's acquisition time is what orders
them. Adaptive Polishing images are ordered by the number in their filename and
their header is never opened, which these also pin.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from am_segmentation_manager.frames import build_lamella_frames, group_by_acquisition
from am_segmentation_manager.layout import UNSEQUENCED, alignment_run_order
from am_segmentation_manager.models import ImageRecord, Lamella, Stage

LAMELLA = Path("01-ruling-ibex")
RUN = LAMELLA / "Alignment" / "01-ruling-ibex - Polishing-10-55-07"


def at(hour: int, minute: int, second: int) -> datetime:
    return datetime(2026, 7, 28, hour, minute, second)


def loose(name: str, when: datetime, directory: Path = LAMELLA) -> ImageRecord:
    return ImageRecord(
        image_path=directory / name,
        modality="FIB",
        series="reference" if name.startswith("ref_") else "alignment",
        acquired_at=when,
        stage_path=directory,
    )


def stage(path: Path, images: list[ImageRecord], number: str) -> Stage:
    return Stage(name=path.name, path=path, number=number, images=images)


# ---------------------------------------------------------------------------
# Within a stage
# ---------------------------------------------------------------------------

def test_reference_images_follow_the_workflow_not_the_alphabet():
    """Alphabetically ``Setup Lamella Position`` sorts last of the four tasks."""
    images = [
        loose("ref_Setup Lamella Position_start_ib.tif", at(10, 34, 50)),
        loose("ref_Mill Fiducial_start_ib.tif", at(10, 35, 53)),
        loose("ref_Rough Milling_start_ib.tif", at(10, 40, 27)),
        loose("ref_Polishing_start_ib.tif", at(12, 5, 3)),
    ]
    lamella = Lamella(name="lam", path=LAMELLA, stages=[stage(LAMELLA, images, "01")])

    order = [f.records[0].image_path.name for f in build_lamella_frames(lamella)]

    assert order == [
        "ref_Setup Lamella Position_start_ib.tif",
        "ref_Mill Fiducial_start_ib.tif",
        "ref_Rough Milling_start_ib.tif",
        "ref_Polishing_start_ib.tif",
    ]


def test_a_start_image_comes_before_the_task_finishes():
    """``final_res`` sorts before ``start`` alphabetically."""
    images = [
        loose("ref_Polishing_final_res_01_ib.tif", at(12, 16, 10)),
        loose("ref_Polishing_start_ib.tif", at(12, 5, 3)),
    ]
    lamella = Lamella(name="lam", path=LAMELLA, stages=[stage(LAMELLA, images, "01")])

    order = [f.records[0].image_path.name for f in build_lamella_frames(lamella)]

    assert order[0] == "ref_Polishing_start_ib.tif"


# ---------------------------------------------------------------------------
# Across stages
# ---------------------------------------------------------------------------

def test_a_series_is_ordered_across_its_stages():
    """A lamella's reference images span the session its tasks sit inside.

    No position for the lamella directory in a stage-by-stage list is right,
    so the frames of a series are ordered by when they were taken.
    """
    lamella = Lamella(
        name="lam",
        path=LAMELLA,
        stages=[
            stage(
                LAMELLA,
                [
                    loose("ref_Setup Lamella Position_start_ib.tif", at(10, 34, 50)),
                    loose("ref_Polishing_start_ib.tif", at(12, 5, 3)),
                ],
                "01",
            ),
            stage(
                LAMELLA / "Milling" / "Fiducial",
                [
                    loose(
                        "Fiducial_pre-task_10-35-26_ib.tif",
                        at(10, 35, 26),
                        LAMELLA / "Milling" / "Fiducial",
                    )
                ],
                "02",
            ),
        ],
    )

    times = [f.acquired_at for f in build_lamella_frames(lamella)]

    assert times == sorted(times)


def test_adaptive_frames_keep_the_order_their_numbers_give_them():
    """Their headers are never opened, so there is no time to sort by."""
    images = [
        ImageRecord(
            image_path=LAMELLA / "sem" / f"img_{n:03d}_SEM.tif",
            modality="SEM",
            series="adaptive",
            stage_path=LAMELLA,
        )
        for n in (2, 0, 1)
    ]
    lamella = Lamella(name="lam", path=LAMELLA, stages=[stage(LAMELLA, images, "01")])

    numbers = [f.image_number for f in build_lamella_frames(lamella)]

    assert numbers == [0, 1, 2]
    assert all(f.acquired_at is None for f in build_lamella_frames(lamella))


# ---------------------------------------------------------------------------
# Breaking a tie inside an alignment run
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "earlier, later",
    [
        ("reference_image.tif", "new_image_00.tif"),
        ("new_image_00.tif", "new_image_01.tif"),
        ("new_image_01.tif", "new_image_02.tif"),
        ("new_image_02.tif", "final_image.tif"),
    ],
)
def test_an_alignment_run_reads_in_the_order_it_ran(earlier, later):
    assert alignment_run_order(earlier) < alignment_run_order(later)


def test_a_name_outside_the_sequence_is_placed_after_it():
    assert alignment_run_order("ref_alignment_ib.tif") == UNSEQUENCED
    assert alignment_run_order("final_image.tif") < UNSEQUENCED


def test_images_taken_in_one_second_keep_the_run_order():
    """A header records whole seconds; a run takes several images inside one."""
    same = at(10, 55, 44)
    images = [
        loose("final_image.tif", same, RUN),
        loose("new_image_02.tif", same, RUN),
        loose("reference_image.tif", same, RUN),
        loose("new_image_00.tif", same, RUN),
    ]
    lamella = Lamella(name="lam", path=LAMELLA, stages=[stage(RUN, images, "01")])

    order = [f.records[0].image_path.name for f in build_lamella_frames(lamella)]

    assert order == [
        "reference_image.tif",
        "new_image_00.tif",
        "new_image_02.tif",
        "final_image.tif",
    ]


# ---------------------------------------------------------------------------
# What ordering by time must not disturb
# ---------------------------------------------------------------------------

def test_numbered_acquisitions_are_still_ordered_by_number():
    """Acquisition time never reaches an image whose filename carries a number."""
    records = [
        ImageRecord(
            image_path=LAMELLA / "sem" / f"img_{n:03d}_SEM.tif",
            modality="SEM",
            series="adaptive",
            stage_path=LAMELLA,
        )
        for n in (5, 1, 3)
    ]

    assert [a.image_number for a in group_by_acquisition(records)] == [1, 3, 5]
