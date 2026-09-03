"""Tests for copying a selection of images and labels into a folder of its own.

The property the whole design turns on is that **nothing is overwritten and
nothing is half-written**: the destinations are resolved and classified before
a byte is copied, and each file lands by rename. So these mostly assert what a
plan refuses, and what a second run makes of what the first one wrote.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from am_segmentation_manager.operations.copy_export import (
    PROJECT_FILES,
    CopyPlan,
    DestinationInsideSource,
    copy_export,
    plan_copy,
)
from am_segmentation_manager.progress import Cancellation
from am_segmentation_manager.scanner import Scanner
from conftest import build_dataset

HAS_IMAGE_STACK = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "tifffile", "cv2")
)

pytestmark = pytest.mark.skipif(
    not HAS_IMAGE_STACK, reason="Optional image libs not installed"
)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """A scanned collection: one project, two lamellae, labels beside them."""
    root = tmp_path / "collection"
    root.mkdir()
    build_dataset(root, lamellae=2, acquisitions=2, labels="annotated")
    return root


def scanned(root: Path):
    """Return the records and projects of a scan, as a caller hands them over."""
    projects = Scanner(root).build_index().all_projects()
    return [record for project in projects for record in project.image_records], projects


def plan_for(source: Path, destination: Path) -> CopyPlan:
    records, projects = scanned(source)
    return plan_copy(records, destination, source, projects=projects)


def relative_destinations(plan: CopyPlan) -> set[str]:
    return {
        str(item.destination.relative_to(plan.destination)) for item in plan.items
    }


# ---------------------------------------------------------------------------
# What a copy takes
# ---------------------------------------------------------------------------

def test_the_hierarchy_is_mirrored(source: Path, tmp_path: Path):
    """The copy is the same collection with fewer images in it, which is what
    lets am-seg read it and proofreading continue there."""
    plan = plan_for(source, tmp_path / "out")

    for item in plan.items:
        assert item.destination.relative_to(plan.destination) == (
            item.source.relative_to(source.resolve())
        )


def test_each_image_brings_its_label(source: Path, tmp_path: Path):
    plan = plan_for(source, tmp_path / "out")

    kinds = [item.kind for item in plan.items]
    assert kinds.count("image") == 8
    assert kinds.count("label") == 8


def test_the_files_describing_the_project_come_too(source: Path, tmp_path: Path):
    """Without them the copy is a pile of images: experiment.yaml is what makes
    it indexable at all, and the other two say what it is and how it was made."""
    (source / "project" / "protocol.yaml").write_text("stages: []\n", encoding="utf-8")
    (source / "project" / "AM_metadata.yaml").write_text(
        "metadata:\n  sample: HeLa\n", encoding="utf-8"
    )
    # The history must travel or a merge back has no common ancestor to
    # compare against, which is why it is among PROJECT_FILES at all.
    (source / "project" / "AM_history.jsonl").write_text("", encoding="utf-8")

    plan = plan_for(source, tmp_path / "out")

    copied = relative_destinations(plan)
    for name in PROJECT_FILES:
        assert f"project/{name}" in copied


def test_a_project_file_that_does_not_exist_is_not_an_error(
    source: Path, tmp_path: Path
):
    """Only some projects have a protocol.yaml, and only this package writes
    AM_metadata.yaml."""
    plan = plan_for(source, tmp_path / "out")

    assert not [item for item in plan.items if item.source.name == "protocol.yaml"]
    assert plan.to_copy


def test_a_stage_brings_its_state_file(source: Path, tmp_path: Path):
    """So the copy declares the same workflow state as the collection it came
    from, rather than reading as freshly selected."""
    records, projects = scanned(source)
    stage = records[0].stage_path
    (stage / "AM_state.yaml").write_text(
        "version: 1\nstate:\n  labels/img_000_SEM.tif: deselected\n", encoding="utf-8"
    )

    plan = plan_copy(records, tmp_path / "out", source, projects=projects)

    assert any(item.kind == "state" for item in plan.items)


def test_the_copy_can_be_scanned_as_a_collection(source: Path, tmp_path: Path):
    destination = tmp_path / "out"
    plan = plan_for(source, destination)

    copy_export(plan)

    before, _ = scanned(source)
    after, _ = scanned(destination)
    assert len(after) == len(before)


# ---------------------------------------------------------------------------
# Nothing is overwritten
# ---------------------------------------------------------------------------

def test_a_second_run_copies_nothing_and_refuses_nothing(source: Path, tmp_path: Path):
    """What makes an interrupted export resumable: the modification time is
    kept, so a file already written is recognised rather than rewritten."""
    destination = tmp_path / "out"
    first = plan_for(source, destination)
    copy_export(first)

    again = plan_for(source, destination)

    assert again.to_copy == ()
    assert len(again.already_there) == len(first.to_copy)
    assert not again.is_refused


def test_a_destination_holding_something_else_refuses_the_whole_run(
    source: Path, tmp_path: Path
):
    """The rule the batch validation exists for: one file in the way stops
    everything, rather than being discovered halfway through."""
    destination = tmp_path / "out"
    copy_export(plan_for(source, destination))
    intruder = next(iter(sorted(destination.rglob("*.tif"))))
    intruder.write_bytes(b"not what was copied")

    plan = plan_for(source, destination)

    assert len(plan.conflicts) == 1
    assert plan.is_refused
    with pytest.raises(ValueError, match="refused"):
        copy_export(plan)
    assert intruder.read_bytes() == b"not what was copied"


def test_a_destination_inside_the_source_is_refused(source: Path):
    """The copy would be walking into its own output, and the result would be
    indexed as part of the collection."""
    records, projects = scanned(source)

    with pytest.raises(DestinationInsideSource):
        plan_copy(records, source / "export", source, projects=projects)


def test_a_destination_containing_the_source_is_refused(source: Path):
    records, projects = scanned(source)

    with pytest.raises(DestinationInsideSource):
        plan_copy(records, source.parent, source, projects=projects)


def test_a_full_disk_refuses_before_anything_is_written(
    source: Path, tmp_path: Path, monkeypatch
):
    """Asked before the first file rather than discovered at the last."""
    monkeypatch.setattr(
        "am_segmentation_manager.operations.copy_export._free_space", lambda _p: 10
    )

    plan = plan_for(source, tmp_path / "out")

    assert plan.short_by == plan.bytes_to_copy - 10
    assert plan.is_refused
    with pytest.raises(ValueError, match="refused"):
        copy_export(plan)
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------
# Stopping
# ---------------------------------------------------------------------------

def test_stopping_keeps_what_was_already_copied(source: Path, tmp_path: Path):
    """A run cancelled after ten of a hundred files did copy ten, and the next
    run carries on from there rather than starting again."""
    destination = tmp_path / "out"
    plan = plan_for(source, destination)
    cancellation = Cancellation()

    def stop_after_three(done: int, total: int) -> None:
        if done == 3:
            cancellation.request()

    report = copy_export(plan, on_progress=stop_after_three, cancellation=cancellation)

    assert report.stopped
    assert len(report.copied) == 3
    assert len(list(destination.rglob("*"))) > 0
    assert len(plan_for(source, destination).already_there) == 3


def test_progress_is_reported_per_file(source: Path, tmp_path: Path):
    seen: list[tuple[int, int]] = []
    plan = plan_for(source, tmp_path / "out")

    copy_export(plan, on_progress=lambda done, total: seen.append((done, total)))

    assert seen[0] == (1, len(plan.to_copy))
    assert seen[-1] == (len(plan.to_copy), len(plan.to_copy))


# ---------------------------------------------------------------------------
# From the command line
# ---------------------------------------------------------------------------

SCOPE = ["--project", "all", "--lamella", "all", "--stage", "all", "--image", "all"]


def run_cli(*arguments: str) -> int:
    from am_segmentation_manager.cli import main

    return main(["export-files", *arguments])


def test_the_command_copies_the_selection(source: Path, tmp_path: Path):
    destination = tmp_path / "out"

    assert run_cli(str(source), *SCOPE, "-o", str(destination)) == 0

    after, _ = scanned(destination)
    assert len(after) == 8


def test_a_dry_run_writes_nothing(source: Path, tmp_path: Path):
    destination = tmp_path / "out"

    assert run_cli(str(source), *SCOPE, "-o", str(destination), "--dry-run") == 0

    assert not destination.exists()


def test_the_command_refuses_a_destination_inside_the_source(source: Path):
    assert run_cli(str(source), *SCOPE, "-o", str(source / "out")) == 2


def test_the_command_refuses_rather_than_overwriting(source: Path, tmp_path: Path):
    destination = tmp_path / "out"
    run_cli(str(source), *SCOPE, "-o", str(destination))
    intruder = next(iter(sorted(destination.rglob("*.tif"))))
    intruder.write_bytes(b"not what was copied")

    assert run_cli(str(source), *SCOPE, "-o", str(destination)) == 2
    assert intruder.read_bytes() == b"not what was copied"
