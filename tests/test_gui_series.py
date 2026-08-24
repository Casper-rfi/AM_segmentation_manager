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


def test_a_lamella_without_the_chosen_series_shows_an_empty_timeline():
    """The control still offers what is there, which is the way out."""
    lamella = Lamella(name="lam", path=Path("lam"), stages=[adaptive_stage(2)])
    controller = make_controller([lamella])
    controller.series = "reference"
    controller.select_lamella(0)

    assert controller.frames == []
    assert controller.available_series() == ["adaptive"]
