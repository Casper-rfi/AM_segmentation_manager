"""Finding datasets that hold images but are not projects yet.

``init-project`` writes the marker file that makes a pre-Autolamella dataset
indexable. Discovering *where* to write it cannot use project discovery, which
requires the very marker being written -- so the datasets that most need one
were invisible to the command that creates it.

The risk in the other direction is writing a marker too high up: a marker
defines a project boundary, and one placed on a collection directory swallows
the projects beneath it. These tests pin both sides.
"""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from am_segmentation_manager.cli import main
from am_segmentation_manager.discovery import find_dataset_candidates
from am_segmentation_manager.scanner import Scanner


def make_marked_project(root: Path, name: str) -> Path:
    """An Autolamella project: has experiment.yaml, needs nothing."""
    project = root / name
    stage = project / "01-lamella" / "adaptive_polish"
    for directory in ("sem", "fib"):
        (stage / directory).mkdir(parents=True)
        (stage / directory / "img_000.tif").write_bytes(b"x")
    (project / "experiment.yaml").write_text(f"name: {name}\n", encoding="utf-8")
    return project


def make_legacy_dataset(root: Path, name: str = "20240228_adaptive_milling_ai") -> Path:
    """A pre-Autolamella dataset: images, lamellae, no marker of any kind."""
    dataset = root / name
    for lamella in ("lam_004", "lam_010"):
        for directory in ("SEM_8", "SEM_8_fi_1", "FIB", "labels", "labels_FIB"):
            (dataset / lamella / directory).mkdir(parents=True)
            (dataset / lamella / directory / "img_0004.tif").write_bytes(b"x")
    return dataset


# ---------------------------------------------------------------------------
# What counts as a candidate
# ---------------------------------------------------------------------------

def test_a_dataset_without_a_marker_is_found(tmp_path: Path):
    """Beside a marked project, as in the real collection.

    A marked project anchors which level counts as a project, which is what
    makes the unmarked dataset beside it unambiguous.
    """
    make_marked_project(tmp_path, "AutoLamella-2026")
    dataset = make_legacy_dataset(tmp_path)

    assert find_dataset_candidates(tmp_path) == [dataset]


def test_a_marked_project_is_not_a_candidate(tmp_path: Path):
    """It is already a project; init-project finds it the ordinary way."""
    make_marked_project(tmp_path, "AutoLamella-2026")
    assert find_dataset_candidates(tmp_path) == []


def test_a_collection_of_marked_projects_is_not_a_candidate(tmp_path: Path):
    """The dangerous false positive.

    A collection directory holds images beneath it too, but writing a marker
    there would make one project out of the several inside it.
    """
    collection = tmp_path / "20240806_auNP"
    collection.mkdir()
    make_marked_project(collection, "grid6")
    make_marked_project(collection, "grid8")

    assert find_dataset_candidates(tmp_path) == []


def test_a_collection_holding_both_kinds_finds_only_the_unmarked_one(tmp_path: Path):
    """The real collection: 20 marked projects beside one legacy dataset."""
    make_marked_project(tmp_path, "AutoLamella-2026")
    collection = tmp_path / "20240806_auNP"
    collection.mkdir()
    make_marked_project(collection, "grid6")
    dataset = make_legacy_dataset(tmp_path)

    assert find_dataset_candidates(tmp_path) == [dataset]


def test_pointing_at_a_dataset_proposes_that_dataset(tmp_path: Path):
    dataset = make_legacy_dataset(tmp_path)
    assert find_dataset_candidates(dataset) == [dataset]


def test_the_lamellae_of_a_dataset_are_not_proposed(tmp_path: Path):
    """Descending past the dataset would split it into one project per lamella.

    Each lamella holds image directories and has no marker, so a rule that kept
    descending would mark every one of them -- destroying exactly the hierarchy
    the marker is meant to establish.
    """
    make_marked_project(tmp_path, "AutoLamella-2026")
    dataset = make_legacy_dataset(tmp_path)

    candidates = find_dataset_candidates(tmp_path)

    assert candidates == [dataset]
    assert not any(c.name.startswith("lam_") for c in candidates)


def test_a_marked_project_does_not_propose_its_own_lamellae(tmp_path: Path):
    """Its contents belong to it, marker and all."""
    project = make_marked_project(tmp_path, "AutoLamella-2026")
    assert find_dataset_candidates(project) == []


def test_a_directory_with_no_images_is_not_a_candidate(tmp_path: Path):
    (tmp_path / "notes" / "drafts").mkdir(parents=True)
    (tmp_path / "notes" / "readme.txt").write_text("nothing here")

    assert find_dataset_candidates(tmp_path) == []


def test_loose_images_outside_an_image_directory_are_not_a_candidate(tmp_path: Path):
    """The collection has an 'overviews' folder of stray tif files."""
    overviews = tmp_path / "overviews"
    overviews.mkdir()
    (overviews / "grid_front.tif").write_bytes(b"x")

    assert find_dataset_candidates(tmp_path) == []


# ---------------------------------------------------------------------------
# init-project
# ---------------------------------------------------------------------------

def test_recursive_init_writes_a_marker_for_an_unmarked_dataset(tmp_path: Path):
    """The gap this closes: --recursive could only find existing projects."""
    dataset = make_legacy_dataset(tmp_path)
    make_marked_project(tmp_path, "AutoLamella-2026")

    assert main(["init-project", str(tmp_path), "--recursive"]) == 0

    assert (dataset / "AM_metadata.yaml").exists()


def test_the_dataset_becomes_indexable(tmp_path: Path):
    """The point of writing the marker at all."""
    make_marked_project(tmp_path, "AutoLamella-2026")
    make_legacy_dataset(tmp_path)
    assert [p.path.name for p in Scanner(tmp_path).build_index().all_projects()] == [
        "AutoLamella-2026"
    ]

    main(["init-project", str(tmp_path), "--recursive"])

    projects = {p.path.name: p for p in Scanner(tmp_path).build_index().all_projects()}
    assert "20240228_adaptive_milling_ai" in projects
    legacy = projects["20240228_adaptive_milling_ai"]
    assert [l.name for l in legacy.lamellae] == ["lam_004", "lam_010"]
    assert all(s.synthetic for l in legacy.lamellae for s in l.stages)


def test_an_unmarked_dataset_is_announced_before_being_written(tmp_path: Path, capsys):
    make_legacy_dataset(tmp_path)

    main(["init-project", str(tmp_path), "--recursive", "--dry-run"])

    assert "not a project yet" in capsys.readouterr().out


def test_a_dry_run_writes_no_marker_for_a_candidate(tmp_path: Path):
    dataset = make_legacy_dataset(tmp_path)

    main(["init-project", str(tmp_path), "--recursive", "--dry-run"])

    assert not (dataset / "AM_metadata.yaml").exists()


def test_recursive_init_leaves_a_collection_directory_alone(tmp_path: Path):
    """A marker here would merge the projects beneath it into one."""
    collection = tmp_path / "20240806_auNP"
    collection.mkdir()
    make_marked_project(collection, "grid6")
    make_marked_project(collection, "grid8")

    main(["init-project", str(tmp_path), "--recursive"])

    assert not (collection / "AM_metadata.yaml").exists()
    assert Scanner(tmp_path).build_index().all_projects().__len__() == 2


def test_a_lone_unmarked_dataset_proposes_its_container(tmp_path: Path):
    """The ambiguous case, pinned so the choice is deliberate.

    With no marked project to anchor which level is "the project",
    'collection/dataset/lamella' and 'dataset/lamella/stage' are structurally
    identical. The outermost qualifying directory is proposed; --dry-run shows
    which one before anything is written.
    """
    make_legacy_dataset(tmp_path)

    assert find_dataset_candidates(tmp_path) == [tmp_path]


def test_recursive_init_reports_when_there_is_nothing_at_all(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    assert main(["init-project", str(tmp_path), "--recursive"]) == 2
