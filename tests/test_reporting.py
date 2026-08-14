"""Tests for the reporting module."""
from __future__ import annotations

from pathlib import Path

from am_segmentation_manager.models import AdaptiveMillingStage, ImageRecord, Lamella, Project
from am_segmentation_manager.reporting import (
    filesystem_state,
    print_project_summary,
    summarize_lamella,
)


def test_filesystem_state():
    assert filesystem_state(Path("/stage/sem/deselected/img.tif")) == "deselected"
    assert filesystem_state(Path("/stage/sem/corrected/img.tif")) == "corrected"
    assert filesystem_state(Path("/stage/sem/img.tif")) == "selected"


def test_summarize_lamella():
    img_sel = ImageRecord(image_path=Path("/stage/sem/img1.tif"), modality="SEM")
    img_des = ImageRecord(image_path=Path("/stage/sem/deselected/img2.tif"), modality="SEM")
    img_cor = ImageRecord(image_path=Path("/stage/sem/corrected/img3.tif"), modality="SEM", corrected=True)

    stage = AdaptiveMillingStage(name="stage1", path=Path("/stage"), images=[img_sel, img_des, img_cor])
    lamella = Lamella(name="lam1", path=Path("/lam1"), stages=[stage])

    summary = summarize_lamella(lamella)
    assert summary["SEM images"]["selected"] == 1
    assert summary["SEM images"]["deselected"] == 1
    assert summary["SEM images"]["corrected"] == 1
    assert summary["SEM images"]["all"] == 3


def test_print_project_summary(caplog):
    import logging
    img = ImageRecord(image_path=Path("/stage/sem/img1.tif"), modality="SEM")
    stage = AdaptiveMillingStage(name="stage1", path=Path("/stage"), images=[img])
    lamella = Lamella(name="lam1", path=Path("/lam1"), stages=[stage])
    proj = Project(name="Proj1", path=Path("/proj1"), lamellae=[lamella])

    with caplog.at_level(logging.INFO):
        print_project_summary(proj)

    assert "Proj1" in caplog.text
    assert "Counts shown as: deselected / selected / corrected / all" in caplog.text
