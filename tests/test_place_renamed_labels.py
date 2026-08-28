"""Tests for ``scripts/place_renamed_labels.py``.

Every tree here is built from synthetic TIFFs carrying a hand-written FEI XML
tag, so the pairing is exercised without any real acquisition. The point of the
script is that it refuses: most of these assert that a flawed batch stops with
nothing written.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tifffile")
pytest.importorskip("cv2")

import numpy as np
import tifffile
from place_renamed_labels import (
    GUID_TAG,
    PlacementConflict,
    acquisition_guid,
    main,
    plan_placement,
    source_pairs,
)

from am_segmentation_manager.state import AM_STATE_FILENAME, read_state

_XML = (
    '<?xml version="1.0"?><Metadata><Core><Guid>{guid}</Guid>'
    "<UserID>Supervisor</UserID></Core></Metadata>"
)


def guid_of(number: int) -> str:
    """Return a distinct well-formed GUID for *number*."""
    return f"{number:08x}-1111-2222-3333-444444444444"


def write_image(path: Path, guid: str | None, *, seed: int = 0) -> Path:
    """Write a small image carrying *guid*, its pixels determined by *seed*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((8, 6), seed % 256, dtype=np.uint16)
    extratags = [(GUID_TAG, "s", 0, _XML.format(guid=guid), True)] if guid else []
    tifffile.imwrite(str(path), array, extratags=extratags)
    return path


def write_label(path: Path, *, empty: bool = False, shape: tuple[int, int] = (8, 6)) -> Path:
    """Write a label of *shape*, annotated unless *empty*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros(shape, dtype=np.uint8)
    if not empty:
        array[2:4, 2:4] = 2
    tifffile.imwrite(str(path), array)
    return path


@pytest.fixture
def dataset(tmp_path: Path) -> dict[str, Path]:
    """Build a project of two images and a renamed copy of each, with labels.

    The renamed copies are deliberately in a different order from the
    originals, since a script that paired by position rather than by GUID would
    otherwise pass.
    """
    project = tmp_path / "project"
    write_image(project / "lam_000" / "SEM_8" / "Image_000.tif", guid_of(1), seed=1)
    write_image(project / "lam_000" / "SEM_8" / "Image_004.tif", guid_of(2), seed=2)
    # A directory the originals must not be looked for in.
    write_image(project / "lam_000" / "Plots" / "Image_000.tif", guid_of(9), seed=9)

    renamed = tmp_path / "renamed"
    write_image(renamed / "SEM" / "SEM_1.tif", guid_of(2), seed=2)
    write_image(renamed / "SEM" / "SEM_2.tif", guid_of(1), seed=1)
    write_label(renamed / "labels" / "SEM_1.tif")
    write_label(renamed / "labels" / "SEM_2.tif")

    return {
        "project": project,
        "images": renamed / "SEM",
        "labels": renamed / "labels",
    }


def plan(dataset: dict[str, Path], **kwargs):
    """Return the placement planned for *dataset*."""
    return plan_placement(
        [(dataset["images"], dataset["labels"])], dataset["project"], **kwargs
    )


def test_reads_the_guid_from_the_header(tmp_path: Path) -> None:
    image = write_image(tmp_path / "image.tif", guid_of(7))
    assert acquisition_guid(image) == guid_of(7)


def test_an_image_without_a_guid_has_none(tmp_path: Path) -> None:
    image = write_image(tmp_path / "image.tif", None)
    assert acquisition_guid(image) is None


def test_pairs_by_guid_across_differing_filenames(dataset: dict[str, Path]) -> None:
    placements = plan(dataset)
    sem_8 = dataset["project"] / "lam_000" / "SEM_8_labels"
    assert [(p.label.name, p.destination) for p in placements] == [
        ("SEM_2.tif", sem_8 / "Image_000.tif"),
        ("SEM_1.tif", sem_8 / "Image_004.tif"),
    ]


def test_labels_land_beside_the_image_directory_they_belong_to(
    dataset: dict[str, Path],
) -> None:
    for placement in plan(dataset):
        assert placement.destination.parent.name == "SEM_8_labels"


def test_an_existing_shared_label_directory_is_used(dataset: dict[str, Path]) -> None:
    (dataset["project"] / "lam_000" / "sem_labels").mkdir()
    for placement in plan(dataset):
        assert placement.destination.parent.name == "sem_labels"


def test_a_legacy_label_directory_is_refused(dataset: dict[str, Path]) -> None:
    (dataset["project"] / "lam_000" / "labels").mkdir()
    with pytest.raises(PlacementConflict) as caught:
        plan(dataset)
    assert any("rename_legacy_label_dirs" in line for line in caught.value.conflicts)


def test_an_unmatched_image_stops_the_batch(dataset: dict[str, Path]) -> None:
    write_image(dataset["images"] / "SEM_3.tif", guid_of(99), seed=3)
    write_label(dataset["labels"] / "SEM_3.tif")
    with pytest.raises(PlacementConflict) as caught:
        plan(dataset)
    assert any("matches no image" in line for line in caught.value.conflicts)


def test_every_problem_is_reported_not_just_the_first(dataset: dict[str, Path]) -> None:
    write_image(dataset["images"] / "SEM_3.tif", guid_of(98), seed=3)
    write_label(dataset["labels"] / "SEM_3.tif")
    write_image(dataset["images"] / "SEM_4.tif", guid_of(97), seed=4)
    write_label(dataset["labels"] / "SEM_4.tif")
    with pytest.raises(PlacementConflict) as caught:
        plan(dataset)
    assert len(caught.value.conflicts) == 2


def test_a_label_without_an_image_stops_the_batch(dataset: dict[str, Path]) -> None:
    write_image(dataset["images"] / "SEM_3.tif", guid_of(1), seed=1)
    with pytest.raises(PlacementConflict) as caught:
        plan(dataset)
    assert any("no label beside it" in line for line in caught.value.conflicts)


def test_a_duplicated_guid_in_the_project_stops_the_batch(dataset: dict[str, Path]) -> None:
    write_image(
        dataset["project"] / "lam_001" / "SEM_8" / "Image_000.tif", guid_of(1), seed=1
    )
    with pytest.raises(PlacementConflict) as caught:
        plan(dataset)
    assert any("matches several images" in line for line in caught.value.conflicts)


def test_an_altered_original_stops_the_batch(dataset: dict[str, Path]) -> None:
    write_image(
        dataset["project"] / "lam_000" / "SEM_8" / "Image_000.tif", guid_of(1), seed=200
    )
    with pytest.raises(PlacementConflict) as caught:
        plan(dataset)
    assert any("pixels differ" in line for line in caught.value.conflicts)


def test_an_empty_label_stops_the_batch(dataset: dict[str, Path]) -> None:
    write_label(dataset["labels"] / "SEM_1.tif", empty=True)
    with pytest.raises(PlacementConflict) as caught:
        plan(dataset)
    assert any("no annotation" in line for line in caught.value.conflicts)


def test_a_label_of_the_wrong_size_stops_the_batch(dataset: dict[str, Path]) -> None:
    write_label(dataset["labels"] / "SEM_1.tif", shape=(4, 4))
    with pytest.raises(PlacementConflict) as caught:
        plan(dataset)
    assert any("but the image is" in line for line in caught.value.conflicts)


def test_an_occupied_destination_stops_the_batch(dataset: dict[str, Path]) -> None:
    write_label(dataset["project"] / "lam_000" / "SEM_8_labels" / "Image_000.tif")
    with pytest.raises(PlacementConflict) as caught:
        plan(dataset)
    assert any("already holds a label" in line for line in caught.value.conflicts)


def test_sharing_writes_into_the_modality_directory(dataset: dict[str, Path]) -> None:
    for placement in plan(dataset, shared=True):
        assert placement.destination.parent.name == "sem_labels"


#: One acquisition imaged three ways: distinct files, one shared segmentation.
CONDITIONS = (("SEM_8", 1), ("SEM_8_fi_1", 2), ("SEM_8_fi_4", 3))


def test_conditions_sharing_one_label_place_it_once(tmp_path: Path) -> None:
    """Source conditions resolving to one label directory are not a collision."""
    source, project = tmp_path / "source", tmp_path / "project"
    for condition, number in CONDITIONS:
        write_image(source / "lam_000" / condition / "img_0000.tif", guid_of(number), seed=number)
        write_image(project / "lam_000" / condition / "img_0000.tif", guid_of(number), seed=number)
    # One label directory beside them all, as the source keeps it.
    write_label(source / "lam_000" / "labels" / "img_0000.tif")

    placements = plan_placement(source_pairs(source), project, shared=True)
    assert [p.destination for p in placements] == [
        project / "lam_000" / "sem_labels" / "img_0000.tif"
    ]


def test_two_different_labels_claiming_one_destination_stop_the_batch(
    tmp_path: Path,
) -> None:
    source, project = tmp_path / "source", tmp_path / "project"
    for condition, number in CONDITIONS[:2]:
        write_image(source / "lam_000" / condition / "img_0000.tif", guid_of(number), seed=number)
        write_image(project / "lam_000" / condition / "img_0000.tif", guid_of(number), seed=number)
        # A private label each, so sharing makes two different files want one path.
        write_label(source / "lam_000" / f"{condition}_labels" / "img_0000.tif")

    with pytest.raises(PlacementConflict) as caught:
        plan_placement(source_pairs(source), project, shared=True)
    assert any("already claimed by" in line for line in caught.value.conflicts)


def test_sharing_is_refused_when_a_private_directory_would_shadow_it(
    dataset: dict[str, Path],
) -> None:
    (dataset["project"] / "lam_000" / "SEM_8_labels").mkdir()
    with pytest.raises(PlacementConflict) as caught:
        plan(dataset, shared=True)
    assert any("would shadow the shared" in line for line in caught.value.conflicts)


@pytest.fixture
def source_project(tmp_path: Path) -> Path:
    """Build a source laid out as a project: two modalities, two lamellae."""
    root = tmp_path / "source"
    for lam, guids in (("lam_000", (11, 12)), ("lam_001", (13,))):
        for number in guids:
            write_image(root / lam / "SEM_8" / f"img_{number:04d}.tif", guid_of(number), seed=number)
            write_label(root / lam / "labels" / f"img_{number:04d}.tif")
            write_image(root / lam / "FIB" / f"img_{number:04d}.tif", guid_of(number + 50), seed=number + 50)
            write_label(root / lam / "labels_FIB" / f"img_{number:04d}.tif")
    # An image directory with no labels, which a source pairing must skip.
    write_image(root / "lam_002" / "SEM_8" / "img_0000.tif", guid_of(99), seed=99)
    return root


def test_source_pairs_finds_both_modalities(source_project: Path) -> None:
    pairs = source_pairs(source_project)
    assert [(i.parent.name, i.name, lab.name) for i, lab in pairs] == [
        ("lam_000", "FIB", "labels_FIB"),
        ("lam_000", "SEM_8", "labels"),
        ("lam_001", "FIB", "labels_FIB"),
        ("lam_001", "SEM_8", "labels"),
    ]


def test_a_source_project_places_every_pair_as_one_batch(
    source_project: Path, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    for lam, guids in (("lam_000", (11, 12)), ("lam_001", (13,))):
        for number in guids:
            write_image(project / lam / "SEM_8" / f"img_{number:04d}.tif", guid_of(number), seed=number)
            write_image(project / lam / "FIB" / f"img_{number:04d}.tif", guid_of(number + 50), seed=number + 50)

    assert main([
        "--source-project", str(source_project),
        "--project", str(project), "--shared", "--mark-corrected", "--apply",
    ]) == 0
    assert sorted(p.name for p in (project/"lam_000"/"sem_labels").iterdir()) == [
        "img_0011.tif", "img_0012.tif",
    ]
    assert sorted(p.name for p in (project/"lam_000"/"fib_labels").iterdir()) == [
        "img_0011.tif", "img_0012.tif",
    ]
    assert (project/"lam_001"/"sem_labels"/"img_0013.tif").is_file()
    # The state names the shared directory, which only exists once the copies
    # have landed -- so the records are built after, not before.
    assert read_state(project/"lam_001").states == {
        "sem_labels/img_0013.tif": "corrected",
        "fib_labels/img_0013.tif": "corrected",
    }


def test_source_project_and_images_together_are_refused(
    source_project: Path, dataset: dict[str, Path]
) -> None:
    with pytest.raises(SystemExit):
        main([
            "--source-project", str(source_project),
            "--images", str(dataset["images"]),
            "--project", str(dataset["project"]),
        ])


def run(dataset: dict[str, Path], *extra: str) -> int:
    """Run the script's command line over *dataset*."""
    return main([
        "--images", str(dataset["images"]),
        "--labels", str(dataset["labels"]),
        "--project", str(dataset["project"]),
        *extra,
    ])


def test_a_preview_writes_nothing(dataset: dict[str, Path]) -> None:
    assert run(dataset) == 0
    assert not (dataset["project"] / "lam_000" / "SEM_8_labels").exists()


def test_apply_copies_the_labels(dataset: dict[str, Path]) -> None:
    assert run(dataset, "--apply") == 0
    placed = dataset["project"] / "lam_000" / "SEM_8_labels"
    assert sorted(path.name for path in placed.iterdir()) == [
        "Image_000.tif", "Image_004.tif",
    ]
    original = tifffile.imread(str(dataset["labels"] / "SEM_2.tif"))
    assert np.array_equal(tifffile.imread(str(placed / "Image_000.tif")), original)


def test_compressing_preserves_the_label_values(dataset: dict[str, Path]) -> None:
    assert run(dataset, "--apply", "--compress") == 0
    placed = dataset["project"] / "lam_000" / "SEM_8_labels" / "Image_000.tif"
    original = tifffile.imread(str(dataset["labels"] / "SEM_2.tif"))
    assert np.array_equal(tifffile.imread(str(placed)), original)


def test_marking_corrected_declares_only_the_placed_groups(
    dataset: dict[str, Path],
) -> None:
    write_image(dataset["project"] / "lam_000" / "SEM_8" / "Image_008.tif", guid_of(3), seed=3)
    assert run(dataset, "--apply", "--mark-corrected") == 0
    declared = read_state(dataset["project"] / "lam_000")
    assert declared.states == {
        "SEM_8_labels/Image_000.tif": "corrected",
        "SEM_8_labels/Image_004.tif": "corrected",
    }


def test_a_failed_plan_copies_nothing_and_reports_it(dataset: dict[str, Path]) -> None:
    write_label(dataset["labels"] / "SEM_1.tif", empty=True)
    assert run(dataset, "--apply") == 2
    assert not (dataset["project"] / "lam_000" / "SEM_8_labels").exists()
    assert not (dataset["project"] / "lam_000" / AM_STATE_FILENAME).exists()
