"""Tests for the families view model behind the Add data tab's Families box.

Driven against scanned projects on disk, as the window drives it: what the tree
shows is what the scan and ``AM_families.yaml`` say, and nothing else.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.families import AM_FAMILIES_FILENAME, StaleFamilies, read_families
from am_segmentation_manager.gui.attribution import Attribution
from am_segmentation_manager.gui.families_controller import (
    FamiliesController,
    leaving_corrected,
)
from am_segmentation_manager.history import Recorder, read_log
from am_segmentation_manager.models import Project
from am_segmentation_manager.operations.families_edit import FamiliesValidationError
from am_segmentation_manager.operations.label_groups import primary_record
from am_segmentation_manager.scanner import Scanner
from am_segmentation_manager.state import read_state


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


@pytest.mark.parametrize(
    ("directories", "families"),
    [
        (("SEM_8_fi_4", "SEM_8", "SEM_8_fi_16"), None),
        (("sem", "sem_b", "sem-5kV"), "families:\n  sem_b: sem-5kV\n"),
        (("sem", "sem_b", "sem-5kV"), "families:\n  sem_b: sem-5kV\nprimary:\n  sem-5kV: sem_b\n"),
    ],
)
def test_the_starred_primary_is_the_one_predicted(tmp_path: Path, directories, families):
    """The tab and the operations choose a family's primary by one rule."""
    projects = scan(tmp_path, families=families, lam_010=directories)
    controller = FamiliesController()
    controller.set_scope(projects)
    by_family: dict[str, list] = {}
    for record in projects[0].image_records:
        by_family.setdefault(record.family.lower(), []).append(record)

    for node in controller.tree():
        predicted = primary_record(by_family[node.family.lower()])
        assert node.primary is not None
        assert node.primary.name == predicted.image_path.parent.name


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


def test_a_file_with_lamella_sections_is_shown_as_the_scan_read_it(tmp_path: Path):
    projects = scan(
        tmp_path,
        families="version: 2\nlamellae:\n  lam_010:\n    families:\n      sem_fi4: sem-x\n",
        lam_010=("sem", "sem_fi4"),
    )
    controller = FamiliesController()

    controller.set_scope(projects)

    assert controller.editable
    assert {node.family for node in controller.tree()} == {"sem", "sem-x"}


# ---------------------------------------------------------------------------
# A change confined to the lamella in view
# ---------------------------------------------------------------------------

def two_lamellae(tmp_path: Path, families: str | None = None) -> tuple[FamiliesController, Project]:
    """Return a controller with lam_011 in view, editing on, and the project."""
    projects = scan(
        tmp_path, families,
        lam_010=("sem", "sem_fi4", "sem_b"), lam_011=("sem", "sem_fi4", "sem_b"),
    )
    [project] = projects
    controller = FamiliesController(Attribution(Recorder()))
    controller.set_scope(projects, project.lamellae[1])
    controller.editing_enabled = True
    return controller, project


def test_a_change_confined_to_the_lamella_in_view_writes_its_section(tmp_path: Path):
    controller, project = two_lamellae(tmp_path)

    controller.set_target(True)
    controller.set_family("sem_b", "sem-5kV")
    controller.apply(controller.plan())

    on_file = read_families(project.path)
    assert on_file.project.families == {}
    assert dict(on_file.lamellae["lam_011"].families) == {"sem_b": "sem-5kV"}


def test_in_a_lamella_the_projects_entry_is_what_a_condition_is_given(tmp_path: Path):
    controller, project = two_lamellae(tmp_path, "version: 1\nfamilies:\n  sem_b: sem-5kV\n")
    controller.set_target(True)

    assert controller.family_of("sem_b") == "sem-5kV", "the project's, not the name's"
    controller.set_family("sem_b", "sem")
    assert controller.has_changes, "undoing the project's move here is an entry"
    controller.set_family("sem_b", "sem-5kV")
    assert not controller.has_changes, "back to the project's is none"


def test_held_changes_keep_their_choice(tmp_path: Path):
    controller, project = two_lamellae(tmp_path)
    controller.set_target(True)
    controller.set_family("sem_b", "sem-5kV")

    with pytest.raises(ValueError, match="Save or discard"):
        controller.set_target(False)
    controller.set_scope([project], project.lamellae[0])
    assert controller.target is project.lamellae[1], "the lamella they were made in"

    controller.discard()
    controller.set_scope([project], project.lamellae[0])
    assert controller.target is project.lamellae[0], "with none held it follows the view"


def test_a_lamella_with_no_folder_offers_no_choice(tmp_path: Path):
    project_root = tmp_path / "project"
    for directory in ("sem", "sem_b"):
        (project_root / directory).mkdir(parents=True)
        (project_root / directory / "img_0004.tif").write_bytes(b"")
    (project_root / "AM_metadata.yaml").write_text("metadata:\n", encoding="utf-8")
    [project] = Scanner(tmp_path).build_index().all_projects()
    controller = FamiliesController()

    controller.set_scope([project], project.lamellae[0])

    assert controller.lamella_choice is None
    with pytest.raises(ValueError, match="no folder of its own"):
        controller.set_target(True)


def test_nothing_selected_says_so():
    controller = FamiliesController()

    controller.set_scope([])

    assert controller.status() == "No project selected."


# ---------------------------------------------------------------------------
# Changes held until saved
# ---------------------------------------------------------------------------

def two_families(tmp_path: Path, families: str | None = None) -> FamiliesController:
    """Return a controller on a project of the sem and sem-5kV families, editing on."""
    projects = scan(tmp_path, families, lam_010=("sem", "sem_fi4", "sem_b", "sem-5kV", "FIB"))
    controller = FamiliesController(Attribution(Recorder()))
    controller.set_scope(projects)
    controller.editing_enabled = True
    return controller


def members(controller: FamiliesController, family: str) -> list[tuple[str, str]]:
    """Return ``(condition, note)`` for each condition of *family* in the tree."""
    node = next(node for node in controller.tree() if node.family == family)
    return [(row.name, row.note) for row in node.conditions]


def test_a_moved_condition_shows_in_its_new_family_until_saved(tmp_path: Path):
    controller = two_families(tmp_path)

    controller.set_family("sem_b", "sem-5kV")

    assert members(controller, "sem-5kV") == [
        ("sem-5kV", "primary (default)"),
        ("sem_b", "moved by AM_families.yaml, not saved"),
    ]
    assert [name for name, _ in members(controller, "sem")] == ["sem", "sem_fi4"]
    assert controller.has_changes
    assert not (tmp_path / "project" / AM_FAMILIES_FILENAME).exists(), "nothing written"


def test_a_condition_can_form_a_family_of_its_own(tmp_path: Path):
    controller = two_families(tmp_path)

    controller.set_family("sem_b", "sem-b")

    assert members(controller, "sem-b") == [
        ("sem_b", "primary (default), moved by AM_families.yaml, not saved"),
    ]


def test_moving_back_to_where_it_is_saved_is_no_change(tmp_path: Path):
    controller = two_families(tmp_path)
    controller.set_family("sem_b", "sem-5kV")

    controller.set_family("sem_b", "SEM")

    assert not controller.has_changes
    assert controller.family_of("sem_b") == "sem"


def test_a_saved_move_is_undone_by_choosing_the_names_own_family(tmp_path: Path):
    controller = two_families(tmp_path, "version: 1\nfamilies:\n  sem_b: sem-5kV\n")

    controller.set_family("sem_b", "sem")

    assert ("sem_b", "not saved") in members(controller, "sem")
    assert controller.plan().after.project.families == {}


def test_a_family_a_condition_cannot_join_is_refused(tmp_path: Path):
    controller = two_families(tmp_path)

    with pytest.raises(ValueError, match="cannot join the FIB family"):
        controller.set_family("sem_b", "fib-x")
    with pytest.raises(ValueError, match="not a family name"):
        controller.set_family("sem_b", "sem_x")
    assert not controller.has_changes


def test_the_options_are_the_beams_families_and_the_names_own(tmp_path: Path):
    controller = two_families(tmp_path)

    assert controller.family_options("sem_b") == [("sem (its name)", "sem"), ("sem-5kV", "sem-5kV")]


def test_a_primary_named_here_shows_until_saved(tmp_path: Path):
    controller = two_families(tmp_path)

    controller.set_primary("sem", "sem_fi4")

    assert members(controller, "sem") == [
        ("sem", ""), ("sem_b", ""), ("sem_fi4", "primary, not saved"),
    ]
    controller.set_primary("sem", None)
    assert not controller.has_changes


def test_moving_a_named_primary_out_gives_up_its_primary(tmp_path: Path):
    controller = two_families(tmp_path, "version: 1\nprimary:\n  sem: sem_fi4\n")

    controller.set_family("sem_fi4", "sem-5kV")

    assert controller.plan().after.project.primary == {}


def test_changes_are_kept_while_the_project_stays_and_dropped_when_it_changes(tmp_path: Path):
    controller = two_families(tmp_path)
    controller.set_family("sem_b", "sem-5kV")
    project = controller.project
    assert project is not None

    controller.set_scope([project])
    assert controller.has_changes
    controller.set_scope([])
    assert not controller.has_changes


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def test_saving_writes_the_file_records_it_and_forgets_the_changes(tmp_path: Path):
    controller = two_families(tmp_path)
    controller.set_family("sem_b", "sem-5kV")

    change = controller.apply(controller.plan())

    assert change.applied
    project = tmp_path / "project"
    assert read_families(project).project.families == {"sem_b": "sem-5kV"}
    assert [
        (event.data["key"], event.data["new"])
        for event in read_log(project).events if event.kind == "family-changed"
    ] == [("sem_b", "sem-5kV")]
    assert not controller.has_changes


def test_saving_is_refused_until_editing_is_enabled(tmp_path: Path):
    controller = two_families(tmp_path)
    controller.set_family("sem_b", "sem-5kV")
    change = controller.plan()
    controller.editing_enabled = False

    with pytest.raises(PermissionError):
        controller.apply(change)


def test_a_file_changed_since_the_plan_is_not_overwritten(tmp_path: Path):
    controller = two_families(tmp_path)
    controller.set_family("sem_b", "sem-5kV")
    change = controller.plan()
    (tmp_path / "project" / AM_FAMILIES_FILENAME).write_text(
        "version: 1\nfamilies:\n  sem_fi4: sem-fi4\n", encoding="utf-8"
    )

    with pytest.raises(StaleFamilies):
        controller.apply(change)
    assert read_families(tmp_path / "project").project.families == {"sem_fi4": "sem-fi4"}


def test_a_refused_change_carries_every_reason(tmp_path: Path):
    """Swapping one labelled directory for another is the planner's to refuse."""
    controller = two_families(tmp_path)
    for name in ("sem_labels", "sem-5kV_labels"):
        labels = tmp_path / "project" / "lam_010" / name
        labels.mkdir()
        (labels / "img_0004.tif").write_bytes(b"")
    controller.set_family("sem_b", "sem-5kV")

    with pytest.raises(FamiliesValidationError) as refused:
        controller.plan()
    assert refused.value.problems


def test_images_leaving_a_corrected_group_are_counted(tmp_path: Path):
    controller = two_families(tmp_path)
    stage = tmp_path / "project" / "lam_010"
    (stage / "sem_labels").mkdir()
    (stage / "sem_labels" / "img_0004.tif").write_bytes(b"")
    (stage / "AM_state.yaml").write_text(
        "version: 1\nstate:\n  sem_labels/img_0004.tif: corrected\n", encoding="utf-8"
    )
    controller.set_scope(Scanner(tmp_path).build_index().all_projects())
    controller.set_family("sem_b", "sem-b")

    change = controller.plan()

    assert leaving_corrected(change) == 1
    controller.apply(change)
    assert read_state(stage).states == {"sem_labels/img_0004.tif": "corrected"}
