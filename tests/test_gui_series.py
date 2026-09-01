"""Tests for showing one series' timeline at a time in Curate.

A 2026 lamella holds its reference images, its alignment runs and its Adaptive
Polishing stages -- three unrelated things across up to 21 stages. Reviewing
that as one list is not something anyone does, so the timeline is built for one
series and the control switches between them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.config import SegmentationConfig
from am_segmentation_manager.gui.cache import FrameCache
from am_segmentation_manager.gui.controller import DEFAULT_SERIES, CurationController
from am_segmentation_manager.models import ImageRecord, Lamella, Project, Stage
from am_segmentation_manager.scanner import ProjectIndex

CONFIG = SegmentationConfig(
    class_colors={0: (0, 0, 0), 1: (255, 0, 0)}, class_labels={0: "background", 1: "GIS"}
)

LAMELLA = Path("01-ruling-ibex")
POLISH = LAMELLA / "Milling/Polishing/adaptive_polish_2026-07-28-11-34-24AM"


def adaptive_stage(count: int) -> Stage:
    images = []
    for n in range(count):
        for modality, directory in (("SEM", "sem"), ("FIB", "fib")):
            images.append(
                ImageRecord(
                    image_path=POLISH / directory / f"img_{n:03d}_{modality}.tif",
                    modality=modality,
                    series="adaptive",
                    stage_path=POLISH,
                )
            )
    return Stage(name=POLISH.name, path=POLISH, number="02", images=images)


def loose_stage(names: list[tuple[str, str, str]]) -> Stage:
    """A lamella directory standing in for a stage of its own loose images."""
    return Stage(
        name=LAMELLA.name,
        path=LAMELLA,
        number="01",
        synthetic=True,
        images=[
            ImageRecord(
                image_path=LAMELLA / name,
                modality=modality,
                series=series,
                stage_path=LAMELLA,
            )
            for name, modality, series in names
        ],
    )


def make_controller(lamellae: list[Lamella]) -> CurationController:
    project = Project(name="bitmap", path=Path("bitmap"), lamellae=lamellae, number="001")

    def scan(path: Path) -> ProjectIndex:
        index = ProjectIndex(start_path=path)
        index.add(project)
        return index

    cache = FrameCache(lambda image, label=None: f"img:{image.name}", lambda plot: "plot")
    controller = CurationController(cache=cache, config=CONFIG, scan=scan)
    controller.scan(Path("/scan-root"))
    return controller


@pytest.fixture
def controller() -> CurationController:
    """A 2026 lamella: two reference pairs, one alignment image, two adaptive."""
    lamella = Lamella(
        name=LAMELLA.name,
        path=LAMELLA,
        number="01",
        stages=[
            loose_stage(
                [
                    ("ref_Polishing_start_eb.tif", "SEM", "reference"),
                    ("ref_Polishing_start_ib.tif", "FIB", "reference"),
                    ("ref_Rough Milling_start_eb.tif", "SEM", "reference"),
                    ("ref_Rough Milling_start_ib.tif", "FIB", "reference"),
                    ("ref_alignment_ib.tif", "FIB", "alignment"),
                ]
            ),
            adaptive_stage(2),
        ],
    )
    return make_controller([lamella])


# ---------------------------------------------------------------------------
# What the timeline opens on
# ---------------------------------------------------------------------------

def test_the_timeline_opens_on_adaptive_polishing(controller: CurationController):
    """So the tab shows what it always showed."""
    assert controller.series == DEFAULT_SERIES == "adaptive"
    assert len(controller.frames) == 2
    assert all(frame.series == "adaptive" for frame in controller.frames)


def test_the_control_offers_only_what_the_lamella_holds(controller: CurationController):
    assert controller.available_series() == ["adaptive", "reference", "alignment"]


def test_a_lamella_of_one_series_offers_no_choice():
    """A pre-Autolamella lamella has only Adaptive Polishing images."""
    lamella = Lamella(name="lam_003", path=Path("lam_003"), stages=[adaptive_stage(2)])

    assert make_controller([lamella]).available_series() == ["adaptive"]


# ---------------------------------------------------------------------------
# Switching
# ---------------------------------------------------------------------------

def test_switching_rebuilds_the_timeline(controller: CurationController):
    controller.set_series("reference")

    assert len(controller.frames) == 2
    assert all(frame.series == "reference" for frame in controller.frames)
    assert all(frame.sem and frame.fib for frame in controller.frames)


def test_switching_returns_to_the_start_of_the_new_timeline(controller):
    controller.select_frame(1)

    controller.set_series("alignment")

    assert controller.frame_position == 0
    assert len(controller.frames) == 1


def test_switching_clears_what_was_ticked(controller: CurationController):
    """A frame index only means anything within one timeline."""
    controller.selection.toggle(0)
    assert controller.selection.frames

    controller.set_series("reference")

    assert not controller.selection.frames


def test_switching_to_the_series_already_shown_changes_nothing(controller):
    controller.select_frame(1)
    controller.selection.toggle(1)

    controller.set_series("adaptive")

    assert controller.frame_position == 1
    assert controller.selection.frames


def test_an_unknown_series_is_refused(controller: CurationController):
    with pytest.raises(ValueError, match="Unknown series"):
        controller.set_series("refrence")


def test_the_cache_follows_the_new_timeline(controller: CurationController):
    controller.set_series("reference")

    assert list(controller.cache.frames) == controller.frames


# ---------------------------------------------------------------------------
# The view follows the beams the timeline holds
# ---------------------------------------------------------------------------

def test_a_timeline_imaged_with_one_beam_shows_that_beam(controller: CurationController):
    """An alignment run is ion-beam only, so opening one on the SEM view showed
    "no image for this modality" on every frame of it."""
    controller.set_series("alignment")

    assert controller.mode == "fib"
    assert controller.target == "FIB", "and a state change follows the eye"


def test_the_beam_that_was_chosen_comes_back(controller: CurationController):
    controller.set_series("alignment")

    controller.set_series("adaptive")

    assert controller.mode == "sem"


def test_choosing_a_beam_there_is_a_choice_of_its_own(controller: CurationController):
    """Switching the view by hand is a decision, and outlives the timeline it
    was made on -- unlike the fallback, which answers about one timeline."""
    controller.set_series("alignment")
    controller.set_mode("fib")

    controller.set_series("adaptive")

    assert controller.mode == "fib"


def test_the_plot_view_is_left_alone(controller: CurationController):
    """A figure is not an image of either beam."""
    controller.set_mode("plot")

    controller.set_series("alignment")

    assert controller.mode == "plot"


def test_a_timeline_holding_neither_beam_changes_nothing():
    """Nothing to follow, so the view stays where the reviewer left it."""
    controller = make_controller([Lamella(name="lam", path=Path("lam"), stages=[])])

    assert controller.frames == []
    assert controller.mode == "sem"


def test_the_chosen_series_survives_moving_between_lamellae():
    """Otherwise reviewing a project's reference images means reselecting per lamella."""
    lamellae = [
        Lamella(
            name=f"0{n}-lam",
            path=Path(f"0{n}-lam"),
            number=f"0{n}",
            stages=[
                loose_stage([("ref_Polishing_start_eb.tif", "SEM", "reference")]),
                adaptive_stage(2),
            ],
        )
        for n in (1, 2)
    ]
    controller = make_controller(lamellae)
    controller.set_series("reference")

    controller.select_lamella(1)

    assert controller.series == "reference"
    assert len(controller.frames) == 1


def test_a_lamella_without_the_chosen_series_shows_one_it_holds():
    """It used to show an empty timeline, with the control hidden for having
    one entry -- so there was no way to reach the series the lamella has."""
    lamellae = [
        Lamella(
            name="01-lam",
            path=Path("01-lam"),
            number="01",
            stages=[loose_stage([("ref_alignment_ib.tif", "FIB", "alignment")])],
        ),
        Lamella(name="02-lam", path=Path("02-lam"), number="02", stages=[adaptive_stage(2)]),
    ]
    controller = make_controller(lamellae)
    assert controller.series == "alignment", "nothing else to show"

    controller.select_lamella(1)

    assert controller.series == "adaptive"
    assert len(controller.frames) == 2


def test_the_choice_survives_a_lamella_that_cannot_show_it():
    """The fallback answers about one lamella rather than changing the choice,
    so a project reviewed for its reference images stays on them."""
    lamellae = [
        Lamella(
            name=f"0{n}-lam",
            path=Path(f"0{n}-lam"),
            number=f"0{n}",
            stages=(
                [loose_stage([("ref_Polishing_start_eb.tif", "SEM", "reference")])]
                if n != 2
                else [adaptive_stage(2)]
            ),
        )
        for n in (1, 2, 3)
    ]
    controller = make_controller(lamellae)
    controller.set_series("reference")

    controller.select_lamella(1)
    assert controller.series == "adaptive", "all this lamella holds"

    controller.select_lamella(2)
    assert controller.series == "reference"


# ---------------------------------------------------------------------------
# Which series a project holds
# ---------------------------------------------------------------------------

def test_a_project_reports_the_series_it_holds(controller: CurationController):
    """Read off the records, so a project can never claim a series whose
    timeline would open empty."""
    assert controller.project_series() == [["adaptive", "reference", "alignment"]]


def test_a_project_with_nothing_to_proofread_says_so():
    """A quarter of the reference archive holds reference and alignment images
    and no Adaptive Polishing ones at all. Nothing else on the row says it."""
    controller = make_controller([
        Lamella(
            name=LAMELLA.name,
            path=LAMELLA,
            number="01",
            stages=[
                loose_stage([
                    ("ref_MillRough_start_eb.tif", "SEM", "reference"),
                    ("ref_alignment_ib.tif", "FIB", "alignment"),
                ])
            ],
        )
    ])

    assert controller.project_series() == [["reference", "alignment"]]


def test_the_series_are_reported_in_display_order():
    """Read by position on the chip, so the order cannot be the order the
    records happened to be scanned in."""
    controller = make_controller([
        Lamella(
            name=LAMELLA.name,
            path=LAMELLA,
            number="01",
            stages=[
                loose_stage([
                    ("ref_alignment_ib.tif", "FIB", "alignment"),
                    ("ref_MillRough_start_eb.tif", "SEM", "reference"),
                ]),
                adaptive_stage(1),
            ],
        )
    ])

    assert controller.project_series() == [["adaptive", "reference", "alignment"]]


def test_a_project_holding_no_images_reports_nothing():
    controller = make_controller([
        Lamella(name=LAMELLA.name, path=LAMELLA, number="01", stages=[])
    ])

    assert controller.project_series() == [[]]
