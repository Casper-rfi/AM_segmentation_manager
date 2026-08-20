"""Tests for the reporting module."""
from __future__ import annotations

from pathlib import Path

from am_segmentation_manager.models import AdaptiveMillingStage, ImageRecord, Lamella, Project
from am_segmentation_manager.reporting import (
    print_project_summary,
    record_state,
    summarize_lamella,
)


def test_record_state():
    def record(**kwargs):
        return ImageRecord(image_path=Path("/stage/sem/img.tif"), **kwargs)

    assert record_state(record()) == "selected"
    assert record_state(record(selected=False)) == "deselected"
    assert record_state(record(selected=False, corrected=True)) == "corrected"


def test_summarize_lamella():
    img_sel = ImageRecord(image_path=Path("/stage/sem/img1.tif"), modality="SEM")
    img_des = ImageRecord(
        image_path=Path("/stage/sem/img2.tif"), modality="SEM", selected=False
    )
    img_cor = ImageRecord(
        image_path=Path("/stage/sem/img3.tif"), modality="SEM",
        selected=False, corrected=True,
    )

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
