"""Tests for the shared dataset builders in ``conftest``.

A fixture builder is test infrastructure, but a broken one does not fail
loudly -- it silently builds a tree that does not resemble the data, and every
test using it keeps passing while asserting nothing useful. So the builders get
tested like anything else, against what the scanner makes of what they produce.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.operations.label_io import is_label_file_empty
from am_segmentation_manager.scanner import Scanner
from conftest import LABEL_NAMING, SAMPLE_LAYOUTS, build_dataset


def scanned(root: Path):
    """Return the single project the scanner finds under *root*."""
    projects = Scanner(root).build_index().all_projects()
    assert len(projects) == 1, f"expected one project, got {len(projects)}"
    return projects[0]


def records_of(root: Path) -> list:
    project = scanned(root)
    return [
        record
        for lamella in project.lamellae
        for stage in lamella.stages
        for record in stage.images
    ]


@pytest.mark.parametrize("layout", SAMPLE_LAYOUTS)
def test_every_era_scans_to_one_indexable_project(tmp_path: Path, layout: str):
    build_dataset(tmp_path, layout=layout)
    project = scanned(tmp_path)

    assert project.lamellae
    assert project.lamellae[0].stages


@pytest.mark.parametrize("layout", SAMPLE_LAYOUTS)
def test_every_era_pairs_each_image_with_its_label(tmp_path: Path, layout: str):
    build_dataset(tmp_path, layout=layout, acquisitions=2)
    records = records_of(tmp_path)

    assert records
    for record in records:
        assert record.labels, f"{record.image_path} has no label"


def test_collapsed_eras_report_the_level_they_lack_as_synthetic(tmp_path: Path):
    """The point of these fixtures: the missing level is synthesised, not
    invented on disk."""
    build_dataset(tmp_path, layout="no_lamella_level")
    project = scanned(tmp_path)

    assert project.lamellae[0].synthetic is True
    assert project.lamellae[0].stages[0].synthetic is True


def test_a_lamella_directory_without_a_stage_is_synthetic_only_at_stage_level(tmp_path: Path):
    build_dataset(tmp_path, layout="no_stage_level")
    project = scanned(tmp_path)

    assert project.lamellae[0].synthetic is False
    assert project.lamellae[0].stages[0].synthetic is True


@pytest.mark.parametrize("naming", sorted(LABEL_NAMING))
def test_labels_are_found_under_every_naming_scheme(tmp_path: Path, naming: str):
    build_dataset(tmp_path, label_naming=naming)
    for record in records_of(tmp_path):
        assert record.labels, f"{naming}: {record.image_path} has no label"


def test_sem_variants_share_one_label(tmp_path: Path):
    """Several imaging conditions of one scene resolve to a single label, which
    is what makes the label rather than the image the unit of workflow state."""
    data = build_dataset(
        tmp_path, layout="no_stage_level", sem_variants=("SEM_8_fi_1", "SEM_8_fi_4"),
        modalities=("SEM",), acquisitions=1,
    )
    sem_records = [r for r in records_of(tmp_path) if r.modality == "SEM"]

    assert len(sem_records) == 3
    assert len({str(r.labels[0]) for r in sem_records}) == 1
    assert data.label_dir("SEM") == data.label_dir("SEM", "SEM_8_fi_1")
    assert all(r.label_shared for r in sem_records)


@pytest.mark.parametrize("layout", SAMPLE_LAYOUTS)
def test_a_dataset_can_lack_a_modality_in_any_era(tmp_path: Path, layout: str):
    """Modality presence is independent of layout era.

    Two documented pre-Autolamella datasets have SEM only, but that is about
    those datasets, not their layout -- every lamella of the oldest dataset in
    the current collection has FIB.
    """
    build_dataset(tmp_path, layout=layout, modalities=("SEM",))

    assert {r.modality for r in records_of(tmp_path)} == {"SEM"}


def test_labels_none_builds_images_without_labels(tmp_path: Path):
    build_dataset(tmp_path, labels="none")

    assert all(not record.labels for record in records_of(tmp_path))


def test_annotated_and_empty_labels_are_distinguishable(tmp_path: Path):
    """A label file existing is not evidence of annotation, so the builder has
    to be able to produce both kinds."""
    pytest.importorskip("tifffile")
    annotated = build_dataset(tmp_path / "a", labels="annotated")
    empty = build_dataset(tmp_path / "b", labels="empty")

    assert is_label_file_empty(annotated.label("SEM")) is False
    assert is_label_file_empty(empty.label("SEM")) is True


def test_multiple_lamellae_are_all_scanned(tmp_path: Path):
    build_dataset(tmp_path, lamellae=3)

    assert len(scanned(tmp_path).lamellae) == 3


def test_the_returned_paths_are_the_ones_written(tmp_path: Path):
    """Dataset exists so tests stop re-deriving paths; it has to be right."""
    data = build_dataset(tmp_path, acquisitions=2, plots=True, predictions=True)

    assert data.image("SEM", 0).exists()
    assert data.image("FIB", 1).exists()
    assert data.label("SEM", 0).exists()
    assert data.project.exists()
    assert (data.stage / "plots").is_dir()
    assert (data.stage / "predictions" / "clean").is_dir()


def test_marker_defaults_match_the_era(tmp_path: Path):
    """Legacy datasets predate experiment.yaml; AM_metadata.yaml is what makes
    them indexable at all."""
    modern = build_dataset(tmp_path / "m")
    legacy = build_dataset(tmp_path / "l", layout="no_lamella_level")

    assert (modern.project / "experiment.yaml").exists()
    assert (legacy.project / "AM_metadata.yaml").exists()
    assert not (legacy.project / "experiment.yaml").exists()


def test_an_unknown_era_is_rejected_rather_than_silently_building_nothing(tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown sample layout"):
        build_dataset(tmp_path, layout="v2027_speculative")


# ---------------------------------------------------------------------------
# Axes, varied independently of any sample
#
# The reason build_dataset takes axes rather than only preset names: the sample
# layouts are datasets that happen to have been observed, not a classification
# of what is possible. A test needing "a stage nested three levels down" should
# be able to say that, instead of hunting for the sample that currently nests.
# ---------------------------------------------------------------------------

def test_a_stage_can_be_nested_more_deeply_than_any_sample(tmp_path: Path):
    build_dataset(
        tmp_path,
        stage_parts=("Milling", "Polishing", "Fine", "adaptive_polish_2027-01-01-00-00-00AM"),
    )
    records = records_of(tmp_path)

    assert records
    assert all(record.labels for record in records)


def test_the_lamella_level_can_be_removed_from_any_sample(tmp_path: Path):
    """Absent levels are an axis, not a property of the oldest datasets."""
    build_dataset(tmp_path, layout="timestamped_stage", lamella_name=None)
    project = scanned(tmp_path)

    assert project.lamellae[0].synthetic is True


def test_the_stage_level_can_be_removed_from_any_sample(tmp_path: Path):
    build_dataset(tmp_path, layout="timestamped_stage", stage_parts=None)
    project = scanned(tmp_path)

    assert project.lamellae[0].synthetic is False
    assert project.lamellae[0].stages[0].synthetic is True


def test_label_naming_is_independent_of_layout(tmp_path: Path):
    """A collapsed dataset using the current label naming is a combination no
    observed dataset shows, and must still index."""
    build_dataset(
        tmp_path, layout="no_lamella_level", label_naming="convention",
        image_dirs={"SEM": "sem", "FIB": "fib"},
    )

    for record in records_of(tmp_path):
        assert record.labels


def test_variants_are_independent_of_layout(tmp_path: Path):
    """Imaging conditions happen to appear only in one observed dataset; nothing
    says a newer layout cannot have them."""
    build_dataset(
        tmp_path, layout="timestamped_stage", sem_variants=("sem_5kv",),
        modalities=("SEM",), acquisitions=1,
    )
    sem = [r for r in records_of(tmp_path) if r.modality == "SEM"]

    assert {r.variant for r in sem} == {None, "5kv"}
    # One acquisition, two conditions, one label between them.
    assert len(sem) == 2
    assert len({str(r.labels[0]) for r in sem}) == 1
