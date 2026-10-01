"""Tests for the families view model behind the Add data tab's Families box.

Driven against scanned projects on disk, as the window drives it: what the tree
shows is what the scan and ``AM_families.yaml`` say, and nothing else.
"""
from __future__ import annotations

from pathlib import Path

from am_segmentation_manager.gui.families_controller import FamiliesController
from am_segmentation_manager.models import Project
from am_segmentation_manager.scanner import Scanner


def scan(root: Path, families: str | None = None, **stages: tuple[str, ...]) -> list[Project]:
    """Scan one project of *stages*, each the image directories it holds."""
    project = root / "project"
    for stage, directories in stages.items():
        for directory in directories:
            (project / stage / directory).mkdir(parents=True)
            (project / stage / directory / "img_0004.tif").write_bytes(b"")
    (project / "AM_metadata.yaml").write_text("metadata:\n", encoding="utf-8")
    if families is not None:
        (project / "AM_families.yaml").write_text(families, encoding="utf-8")
    return Scanner(root).build_index().all_projects()


def rows(controller: FamiliesController) -> list[tuple[str, str, list[tuple[str, int, str]]]]:
    """Return the tree as ``(beam, family, [(condition, stages, note)])``."""
    return [
        (
            node.modality,
            node.family,
            [(row.name, row.stages, row.note) for row in node.conditions],
        )
        for node in controller.tree()
    ]


def test_the_tree_lists_each_beams_families_and_their_conditions(tmp_path: Path):
    projects = scan(
        tmp_path,
        lam_010=("sem", "sem_fi4", "sem-5kV", "FIB"),
        lam_011=("sem", "sem-5kV", "sem-5kV_fi4", "FIB"),
    )
    controller = FamiliesController()

    controller.set_scope(projects)

    assert rows(controller) == [
        ("SEM", "sem", [("sem", 2, "primary (default)"), ("sem_fi4", 1, "")]),
        ("SEM", "sem-5kV", [("sem-5kV", 2, "primary (default)"), ("sem-5kV_fi4", 1, "")]),
        ("FIB", "FIB", [("FIB", 2, "primary (default)")]),
    ]
    assert controller.status() == ""


def test_a_family_without_a_base_has_its_first_variant_as_primary(tmp_path: Path):
    projects = scan(tmp_path, lam_010=("SEM_8_fi_4", "SEM_8", "SEM_8_fi_16"))
    controller = FamiliesController()

    controller.set_scope(projects)

    [(_, family, conditions)] = rows(controller)
    assert family == "SEM"
    assert conditions[0] == ("SEM_8", 1, "primary (default)")


def test_the_file_moves_conditions_and_names_primaries(tmp_path: Path):
    projects = scan(
        tmp_path,
        families=(
            "version: 1\nfamilies:\n  sem_b: sem-5kV\nprimary:\n  sem-5kV: sem_b\n"
        ),
        lam_010=("sem", "sem_b", "sem-5kV"),
    )
    controller = FamiliesController()

    controller.set_scope(projects)

    assert rows(controller) == [
        ("SEM", "sem", [("sem", 1, "primary (default)")]),
        ("SEM", "sem-5kV", [
            ("sem-5kV", 1, ""),
            ("sem_b", 1, "primary, moved by AM_families.yaml"),
        ]),
    ]


def test_several_projects_show_no_families(tmp_path: Path):
    projects = scan(tmp_path, lam_010=("sem",))
    controller = FamiliesController()

    controller.set_scope([*projects, *projects])

    assert controller.tree() == ()
    assert controller.status() == "Choose one project to see its families."


def test_a_damaged_file_is_named_and_the_folder_names_shown(tmp_path: Path):
    projects = scan(tmp_path, families="version: 1\nfamilies: [", lam_010=("sem", "sem_fi4"))
    controller = FamiliesController()

    controller.set_scope(projects)

    assert controller.status().startswith("Shown as the folder names give them.")
    assert "correct the file by hand first" in controller.status()
    assert [node.family for node in controller.tree()] == ["sem"]


def test_nothing_selected_says_so():
    controller = FamiliesController()

    controller.set_scope([])

    assert controller.status() == "No project selected."
