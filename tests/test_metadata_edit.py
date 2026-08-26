"""Tests for editing a project's AM_metadata.yaml.

The file is meant to be hand-edited: `init-project` writes it with comments
explaining each field. So the tests here care as much about what an edit
*preserves* as about what it changes -- losing a comment or a key someone added
by hand is the worst failure this module could have, and it is a silent one.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from am_segmentation_manager.cli import main
from am_segmentation_manager.metadata import TODO_MARKER
from am_segmentation_manager.operations.metadata_edit import (
    MetadataValidationError,
    plan_metadata_update,
    update_project_metadata,
)

TEMPLATE = """\
# Metadata for proj
#
# Written by 'am-seg init-project'. Values here override experiment.yaml.
metadata:
  # What the experiment was for
  description: TODO
  organisation: Rosalind Franklin Institute
  # Cell type or organism
  sample: TODO
  lamella_type: TODO
  hand_written_field: keep me

# variants:
#   5kv: 5 kV landing energy
custom_top_level:
  nested: also keep me
"""


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project with both metadata files, the AM one hand-annotated."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "experiment.yaml").write_text(
        "name: proj\nmethod: AutoLiftout\nmetadata:\n  user: Autolamella User\n",
        encoding="utf-8",
    )
    (root / "AM_metadata.yaml").write_text(TEMPLATE, encoding="utf-8")
    return root


def block(root: Path) -> dict:
    return yaml.safe_load((root / "AM_metadata.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# What an edit must preserve
# ---------------------------------------------------------------------------

def test_comments_survive_an_edit(project: Path):
    """A YAML round-trip would delete every one of these. The file exists to be
    read and edited by a person, so the guidance has to survive."""
    update_project_metadata(project, {"sample": "HeLa"})
    text = (project / "AM_metadata.yaml").read_text(encoding="utf-8")

    assert "# What the experiment was for" in text
    assert "# Cell type or organism" in text
    assert "# Written by 'am-seg init-project'" in text
    assert "#   5kv: 5 kV landing energy" in text


def test_keys_the_package_does_not_model_survive(project: Path):
    """Losing something a user added by hand would be silent and unrecoverable."""
    update_project_metadata(project, {"sample": "HeLa"})
    data = block(project)

    assert data["metadata"]["hand_written_field"] == "keep me"
    assert data["custom_top_level"] == {"nested": "also keep me"}


def test_key_order_is_not_shuffled(project: Path):
    update_project_metadata(project, {"sample": "HeLa"})
    fields = list(block(project)["metadata"])

    assert fields == ["description", "organisation", "sample", "lamella_type", "hand_written_field"]


def test_experiment_yaml_is_never_written(project: Path):
    """It belongs to Autolamella. The per-key override exists so nothing here
    has to touch it."""
    before = (project / "experiment.yaml").read_bytes()
    update_project_metadata(project, {"sample": "HeLa", "user": "C Berger"})

    assert (project / "experiment.yaml").read_bytes() == before


# ---------------------------------------------------------------------------
# Setting, clearing and marking outstanding
# ---------------------------------------------------------------------------

def test_a_value_is_written_and_reads_back(project: Path):
    update_project_metadata(project, {"sample": "S. cerevisiae"})

    assert block(project)["metadata"]["sample"] == "S. cerevisiae"


def test_a_field_absent_from_the_file_is_added(project: Path):
    update_project_metadata(project, {"project": "Adaptive milling"})

    assert block(project)["metadata"]["project"] == "Adaptive milling"


def test_clearing_falls_back_to_experiment_yaml(project: Path):
    """Removing a field is not the same as blanking it: the value underneath
    applies again."""
    update_project_metadata(project, {"user": "Someone Else"})
    assert block(project)["metadata"]["user"] == "Someone Else"

    update_project_metadata(project, {"user": None})
    assert "user" not in block(project)["metadata"]

    from am_segmentation_manager.scanner import Scanner
    scanned = Scanner(project.parent).build_index().all_projects()[0]
    assert scanned.experiment_metadata.user == "Autolamella User"


def test_marking_a_field_outstanding_differs_from_clearing(project: Path):
    """TODO says "someone must fill this in", which inspect-layout reports.
    Clearing says "this file has nothing to say about it"."""
    update_project_metadata(project, {"project": TODO_MARKER})

    assert block(project)["metadata"]["project"] == TODO_MARKER


def test_values_needing_quotes_round_trip(project: Path):
    """A colon, a leading hash or a YAML keyword each need different handling,
    which is why quoting is delegated to the dumper."""
    awkward = {
        "description": "Session 3: 30 kV, #2 grid",
        "sample": "yes",
        "project": "  padded  ",
    }
    update_project_metadata(project, awkward)
    data = block(project)["metadata"]

    assert data["description"] == "Session 3: 30 kV, #2 grid"
    assert data["sample"] == "yes"          # not parsed back as True
    assert isinstance(data["sample"], str)


def test_creating_the_file_uses_the_commented_template(tmp_path: Path):
    """Editing a project with no AM_metadata.yaml creates one, with the
    guidance comments, rather than a bare mapping."""
    root = tmp_path / "fresh"
    root.mkdir()
    (root / "experiment.yaml").write_text("name: fresh\n", encoding="utf-8")

    change = update_project_metadata(root, {"sample": "HeLa"})
    text = (root / "AM_metadata.yaml").read_text(encoding="utf-8")

    assert change.created is True
    assert block(root)["metadata"]["sample"] == "HeLa"
    assert "# Written by" in text
    # Fields not supplied stay outstanding rather than being invented.
    assert block(root)["metadata"]["description"] == TODO_MARKER


# ---------------------------------------------------------------------------
# Validation, before anything is written
# ---------------------------------------------------------------------------

def test_an_unknown_field_is_rejected(project: Path):
    with pytest.raises(MetadataValidationError, match="unknown field"):
        update_project_metadata(project, {"smaple": "typo"})


def test_an_invalid_lamella_type_is_rejected(project: Path):
    with pytest.raises(MetadataValidationError, match="lamella_type"):
        update_project_metadata(project, {"lamella_type": "banana"})


def test_a_grid_type_is_free_text(project: Path):
    """Unlike lamella_type. Material, mesh and film combine in more ways than
    a vocabulary would survive, and a rejected real grid is worse than an
    inconsistent one -- the Add data tab offers the spellings already in use."""
    update_project_metadata(project, {"grid_type": "Quantifoil R2/2 Cu 200 mesh"})

    assert block(project)["metadata"]["grid_type"] == "Quantifoil R2/2 Cu 200 mesh"


def test_every_problem_is_reported_at_once(project: Path):
    """Correcting a command line one error per run is miserable."""
    with pytest.raises(MetadataValidationError) as raised:
        update_project_metadata(project, {"nonsense": "x", "lamella_type": "banana"})

    assert len(raised.value.problems) == 2


def test_a_rejected_edit_writes_nothing(project: Path):
    before = (project / "AM_metadata.yaml").read_bytes()
    with pytest.raises(MetadataValidationError):
        update_project_metadata(project, {"sample": "HeLa", "lamella_type": "banana"})

    assert (project / "AM_metadata.yaml").read_bytes() == before


def test_an_empty_value_is_rejected_rather_than_blanking_a_field(project: Path):
    with pytest.raises(MetadataValidationError, match="--clear"):
        update_project_metadata(project, {"sample": "   "})


# ---------------------------------------------------------------------------
# Previewing
# ---------------------------------------------------------------------------

def test_a_dry_run_leaves_the_file_byte_identical(project: Path):
    before = (project / "AM_metadata.yaml").read_bytes()
    change = update_project_metadata(project, {"sample": "HeLa"}, dry_run=True)

    assert (project / "AM_metadata.yaml").read_bytes() == before
    assert change.applied is False
    assert not change.is_empty


def test_the_plan_reports_where_the_old_value_came_from(project: Path):
    change = plan_metadata_update(project, {"user": "C Berger"})

    assert len(change.changes) == 1
    assert change.changes[0].old == "Autolamella User"
    assert change.changes[0].old_source.name == "experiment.yaml"


def test_setting_a_field_to_what_it_already_holds_is_not_a_change(project: Path):
    change = plan_metadata_update(project, {"organisation": "Rosalind Franklin Institute"})

    assert change.is_empty


def test_a_value_derived_from_experiment_yaml_is_not_a_change(project: Path):
    """method: AutoLiftout already implies lift-out, so setting it changes
    nothing."""
    assert plan_metadata_update(project, {"lamella_type": "lift-out"}).is_empty


# ---------------------------------------------------------------------------
# Through the CLI
# ---------------------------------------------------------------------------

def collection(tmp_path: Path) -> Path:
    root = tmp_path / "collection"
    for name in ("projA", "projB"):
        project_root = root / name
        stage = project_root / "01-lam" / "adaptive_polish" / "sem"
        stage.mkdir(parents=True)
        (stage / "img_000_SEM.tif").write_bytes(b"")
        (project_root / "experiment.yaml").write_text(f"name: {name}\n", encoding="utf-8")
    return root


def test_cli_sets_a_field_across_every_project(tmp_path: Path):
    root = collection(tmp_path)

    assert main(["edit-metadata", str(root), "--all", "--set", "organisation=RFI"]) == 0
    for name in ("projA", "projB"):
        assert block(root / name)["metadata"]["organisation"] == "RFI"


def test_cli_requires_an_explicit_choice_of_projects(tmp_path: Path):
    """--all is opt-in: editing every project in a collection by default would
    be a surprising thing for a bare command to do."""
    root = collection(tmp_path)

    assert main(["edit-metadata", str(root), "--set", "organisation=RFI"]) == 2
    assert not (root / "projA" / "AM_metadata.yaml").exists()


def test_cli_dry_run_writes_nothing(tmp_path: Path):
    root = collection(tmp_path)

    assert main(["edit-metadata", str(root), "--all", "--set", "sample=HeLa", "--dry-run"]) == 0
    assert not (root / "projA" / "AM_metadata.yaml").exists()


def test_cli_rejects_a_bad_edit_before_touching_any_project(tmp_path: Path):
    """Validation runs against every project first, so a bad field name cannot
    leave half the collection edited."""
    root = collection(tmp_path)

    assert main(["edit-metadata", str(root), "--all", "--set", "lamella_type=banana"]) == 2
    assert not (root / "projA" / "AM_metadata.yaml").exists()
    assert not (root / "projB" / "AM_metadata.yaml").exists()


def test_cli_clear_and_todo(tmp_path: Path):
    root = collection(tmp_path)
    main(["edit-metadata", str(root), "--all", "--set", "sample=HeLa"])

    assert main(["edit-metadata", str(root), "--project", "projA", "--clear", "sample"]) == 0
    assert "sample" not in block(root / "projA")["metadata"]
    assert block(root / "projB")["metadata"]["sample"] == "HeLa"

    assert main(["edit-metadata", str(root), "--project", "projA", "--todo", "sample"]) == 0
    assert block(root / "projA")["metadata"]["sample"] == TODO_MARKER


def test_cli_show_metadata_names_the_source_file(tmp_path: Path, capsys):
    root = collection(tmp_path)
    main(["edit-metadata", str(root), "--project", "projA", "--set", "sample=HeLa"])
    capsys.readouterr()

    assert main(["show-metadata", str(root), "--project", "projA"]) == 0
    output = capsys.readouterr().out

    assert "sample" in output
    assert "HeLa" in output
    assert "AM_metadata.yaml" in output


def test_cli_reports_no_match_rather_than_doing_nothing_quietly(tmp_path: Path):
    root = collection(tmp_path)

    assert main(["edit-metadata", str(root), "--project", "nope", "--set", "sample=HeLa"]) == 2


# ---------------------------------------------------------------------------
# Numeric fields
# ---------------------------------------------------------------------------

def test_a_numeric_field_rejects_something_that_is_not_a_number(project: Path):
    """Caught where somebody can still fix it, rather than in the CSV that
    was deposited in a repository."""
    with pytest.raises(MetadataValidationError, match="not a number"):
        update_project_metadata(project, {"gis_deposition_time_s": "3o"})


def test_a_numeric_field_is_written_as_a_number(project: Path):
    """Quoted, it would read back as a string that merely looks like one, and
    the file would not say what it holds."""
    update_project_metadata(project, {"gis_deposition_time_s": "30"})

    text = (project / "AM_metadata.yaml").read_text(encoding="utf-8")
    assert "gis_deposition_time_s: 30" in text
    assert block(project)["metadata"]["gis_deposition_time_s"] == 30


def test_a_numeric_field_keeps_a_fraction(project: Path):
    update_project_metadata(project, {"gis_deposition_time_s": "2.5"})

    assert block(project)["metadata"]["gis_deposition_time_s"] == 2.5


def test_a_numeric_field_can_still_be_marked_outstanding(project: Path):
    """TODO is no more a number than it is a sample, and marking a field as
    unfilled must not be the one edit a numeric field refuses."""
    update_project_metadata(project, {"gis_deposition_time_s": "TODO"})

    assert block(project)["metadata"]["gis_deposition_time_s"] == "TODO"
