"""Tests for ``scripts/share_variant_labels.py``.

The script never decodes an image, so the trees here are built from empty
files: what it acts on is names, and what it must not get wrong is the workflow
state that a label directory's name is part of.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest
from share_variant_labels import ShareConflict, main, plan_sharing

from am_segmentation_manager.state import AM_STATE_FILENAME, read_state, write_state


def touch(path: Path) -> Path:
    """Create an empty file, and the directories above it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def declare(stage: Path, states: dict[str, str]) -> None:
    """Write *stage*'s state file."""
    write_state(read_state(stage), states)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A lamella with two SEM conditions, the second repeating its variant.

    ``SEM_TLD`` names its files ``Image_000_TLD.tif``, so they pair with no
    label; the labels sit privately beside ``SEM_8`` and are declared corrected.
    """
    root = tmp_path / "project"
    lam = root / "lam_016"
    for number in ("000", "001"):
        touch(lam / "SEM_8" / f"Image_{number}.tif")
        touch(lam / "FIB" / f"Image_{number}.tif")
        touch(lam / "SEM_TLD" / f"Image_{number}_TLD.tif")
        touch(lam / "SEM_8_labels" / f"Image_{number}.tif")
    declare(lam, {
        "SEM_8_labels/Image_000.tif": "corrected",
        "SEM_8_labels/Image_001.tif": "corrected",
    })
    return root


def run(project: Path, *extra: str) -> int:
    """Run the script's command line over *project*."""
    return main([str(project), "--suffix", "_TLD", *extra])


def test_the_suffix_is_dropped_and_the_labels_become_shared(project: Path) -> None:
    assert run(project, "--apply", "--manifest", str(project.parent / "m.csv")) == 0
    lam = project / "lam_016"
    assert sorted(p.name for p in (lam / "SEM_TLD").iterdir()) == [
        "Image_000.tif", "Image_001.tif",
    ]
    assert not (lam / "SEM_8_labels").exists()
    assert sorted(p.name for p in (lam / "sem_labels").iterdir()) == [
        "Image_000.tif", "Image_001.tif",
    ]


def test_the_state_keys_follow_the_directory(project: Path) -> None:
    """The group keeps its declaration: the key names the label's new path."""
    assert run(project, "--apply", "--manifest", str(project.parent / "m.csv")) == 0
    assert read_state(project / "lam_016").states == {
        "sem_labels/Image_000.tif": "corrected",
        "sem_labels/Image_001.tif": "corrected",
    }


def test_entries_naming_another_directory_are_carried_through(project: Path) -> None:
    declare(project / "lam_016", {
        "SEM_8_labels/Image_000.tif": "corrected",
        "SEM_8_labels/Image_001.tif": "corrected",
        "FIB_labels/Image_000.tif": "deselected",
    })

    assert run(project, "--apply", "--manifest", str(project.parent / "m.csv")) == 0

    assert read_state(project / "lam_016").states == {
        "sem_labels/Image_000.tif": "corrected",
        "sem_labels/Image_001.tif": "corrected",
        "FIB_labels/Image_000.tif": "deselected",
    }


def test_a_preview_changes_nothing(project: Path) -> None:
    assert run(project) == 0
    lam = project / "lam_016"
    assert (lam / "SEM_TLD" / "Image_000_TLD.tif").exists()
    assert (lam / "SEM_8_labels").exists()
    assert read_state(lam).states == {
        "SEM_8_labels/Image_000.tif": "corrected",
        "SEM_8_labels/Image_001.tif": "corrected",
    }


def test_the_manifest_records_every_rename(project: Path) -> None:
    manifest = project.parent / "manifest.csv"
    assert run(project, "--apply", "--manifest", str(manifest)) == 0

    rows = list(csv.DictReader(manifest.open()))
    assert [row["kind"] for row in rows] == ["file", "file", "directory"]
    assert Path(rows[0]["from"]).name == "Image_000_TLD.tif"
    assert Path(rows[0]["to"]).name == "Image_000.tif"
    assert Path(rows[2]["to"]).name == "sem_labels"


def test_a_partly_suffixed_directory_stops_the_batch(project: Path) -> None:
    touch(project / "lam_016" / "SEM_TLD" / "Image_002.tif")
    with pytest.raises(ShareConflict) as caught:
        plan_sharing(project, "_TLD")
    assert any("of 3 files carry _TLD" in line for line in caught.value.conflicts)


def test_a_rename_onto_an_existing_name_stops_the_batch(project: Path) -> None:
    """Dropping the suffix from this one would land on the file beside it."""
    touch(project / "lam_016" / "SEM_TLD" / "Image_000_TLD_TLD.tif")
    with pytest.raises(ShareConflict) as caught:
        plan_sharing(project, "_TLD")
    assert any("Image_000_TLD.tif is already there" in line
               for line in caught.value.conflicts)


def test_an_existing_shared_directory_stops_the_batch(project: Path) -> None:
    (project / "lam_016" / "sem_labels").mkdir()
    with pytest.raises(ShareConflict) as caught:
        plan_sharing(project, "_TLD")
    assert any("merging them is not something this does" in line
               for line in caught.value.conflicts)


def test_two_private_directories_becoming_one_stop_the_batch(project: Path) -> None:
    touch(project / "lam_016" / "SEM_TLD_labels" / "Image_000.tif")
    with pytest.raises(ShareConflict) as caught:
        plan_sharing(project, "_TLD")
    assert any("several label directories would become sem_labels" in line
               for line in caught.value.conflicts)


def test_a_suffix_nothing_carries_is_refused(project: Path) -> None:
    with pytest.raises(ShareConflict) as caught:
        plan_sharing(project, "_T2")
    assert any("carries _T2" in line for line in caught.value.conflicts)


def test_a_stage_with_no_labels_is_only_renamed(tmp_path: Path) -> None:
    root = tmp_path / "project"
    touch(root / "lam_020" / "SEM_8" / "Image_000.tif")
    touch(root / "lam_020" / "SEM_TLD" / "Image_000_TLD.tif")

    plan = plan_sharing(root, "_TLD")

    assert len(plan.renames) == 1 and plan.shares == ()
    assert main([str(root), "--suffix", "_TLD", "--apply",
                 "--manifest", str(tmp_path / "m.csv")]) == 0
    assert (root / "lam_020" / "SEM_TLD" / "Image_000.tif").exists()
    # No state was declared here, so none is invented.
    assert not (root / "lam_020" / AM_STATE_FILENAME).exists()


def test_only_the_stages_a_rename_touched_are_shared(project: Path) -> None:
    """A private label directory with no second condition is left alone."""
    other = project / "lam_003"
    touch(other / "SEM_8" / "Image_000.tif")
    touch(other / "SEM_8_labels" / "Image_000.tif")
    declare(other, {"SEM_8_labels/Image_000.tif": "corrected"})

    plan = plan_sharing(project, "_TLD")

    assert [share.stage_directory.name for share in plan.shares] == ["lam_016"]
