"""Tests for the plot that describes a whole stage rather than one frame.

An alignment run writes one ``figure.png`` summarising the several images it
took to converge, so there is nothing to match it to a single acquisition --
unlike an Adaptive Polishing stage, which writes one plot per milling cycle
into ``plots/``. Every frame of the run shows the same figure, and says so.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.frames import build_lamella_frames
from am_segmentation_manager.layout import NO_CORRECTIONS
from am_segmentation_manager.models import ImageRecord, Lamella, Stage
from am_segmentation_manager.scanner import Scanner

RUN = Path("01-ruling-ibex/Alignment/01-ruling-ibex - Polishing-10-55-07")
POLISH = Path("01-ruling-ibex/adaptive_polish_2026-07-28-11-34-24AM")


def touch(directory: Path, *names: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).touch()
    return directory


def frames_of(stage: Stage) -> list:
    return build_lamella_frames(
        Lamella(name="lam", path=Path("01-ruling-ibex"), stages=[stage]), corrections=NO_CORRECTIONS
    )


# ---------------------------------------------------------------------------
# A plot covering the whole stage
# ---------------------------------------------------------------------------

def test_every_frame_of_a_run_shows_the_run_s_figure():
    stage = Stage(
        name=RUN.name,
        path=RUN,
        number="01",
        summary_plot=RUN / "figure.png",
        images=[
            ImageRecord(
                image_path=RUN / name,
                modality="FIB",
                series="alignment",
                stage_path=RUN,
            )
            for name in ("reference_image.tif", "new_image_00.tif", "final_image.tif")
        ],
    )

    frames = frames_of(stage)

    assert len(frames) == 3
    assert all(f.displayed_plot == RUN / "figure.png" for f in frames)
    assert all(f.plot_is_shared for f in frames)


def test_a_frame_with_its_own_plot_prefers_it():
    """An Adaptive Polishing plot describes one milling cycle."""
    stage = Stage(
        name=POLISH.name,
        path=POLISH,
        number="01",
        summary_plot=POLISH / "AP_summary_plots.png",
        plots=[POLISH / "plots" / "lam_AP_img_000_plot.png"],
        images=[
            ImageRecord(
                image_path=POLISH / "sem" / "lam_AP_img_000_SEM.tif",
                modality="SEM",
                stage_path=POLISH,
            )
        ],
    )

    frame = frames_of(stage)[0]

    assert frame.displayed_plot == POLISH / "plots" / "lam_AP_img_000_plot.png"
    assert not frame.plot_is_shared


def test_a_stage_with_neither_kind_shows_no_plot():
    stage = Stage(
        name=POLISH.name,
        path=POLISH,
        number="01",
        images=[
            ImageRecord(
                image_path=POLISH / "sem" / "lam_AP_img_000_SEM.tif",
                modality="SEM",
                stage_path=POLISH,
            )
        ],
    )

    frame = frames_of(stage)[0]

    assert frame.displayed_plot is None
    assert not frame.plot_is_shared


# ---------------------------------------------------------------------------
# Finding it on disk
# ---------------------------------------------------------------------------

@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "AutoLamella-2026-07-28-10-31"
    root.mkdir(parents=True)
    (root / "experiment.yaml").write_text("name: bitmap\n", encoding="utf-8")
    lamella = touch(root / "01-ruling-ibex", "ref_alignment_ib.tif")
    run = touch(
        lamella / "Alignment" / "01-ruling-ibex - Polishing-10-55-07",
        "reference_image.tif",
        "new_image_00.tif",
        "figure.png",
        "data.json",
    )
    stage = lamella / "adaptive_polish_2026-07-28-11-34-24AM"
    touch(stage / "sem", "lam_AP_img_000_SEM.tif")
    touch(stage, "AP_summary_plots.png", "centring.png")
    return root, run, stage


def stages_of(root: Path) -> dict[str, Stage]:
    index = Scanner(root.parent).build_index()
    return {
        stage.path.name: stage
        for lamella in index.find_by_name("bitmap").lamellae
        for stage in lamella.stages
    }


def test_an_alignment_run_s_figure_is_found(project):
    root, run, _ = project

    assert stages_of(root)[run.name].summary_plot == run / "figure.png"


def test_an_adaptive_stage_s_summary_is_found(project):
    root, _, stage = project

    assert stages_of(root)[stage.name].summary_plot == stage / "AP_summary_plots.png"


def test_the_centring_plot_is_not_taken_as_a_summary(project):
    """It records where the plugin placed its field of view, not the result."""
    root, _, stage = project
    (stage / "AP_summary_plots.png").unlink()

    assert stages_of(root)[stage.name].summary_plot is None


def test_a_stage_that_wrote_no_summary_has_none(project):
    root, _, _ = project

    lamella_stage = stages_of(root)["01-ruling-ibex"]

    assert lamella_stage.summary_plot is None
