"""The conflict resolution model: where each conflict sits, and what was decided.

Qt-free, so these run without a display. Rows are built by hand rather than by
a merge, so the arrangement and the decision rules are tested apart from how
a merge finds its conflicts; `test_gui_operations.py` joins the two.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.gui.conflict_resolution import (
    ALL,
    LEVEL_CONFLICT,
    LEVEL_FOLDER,
    LEVEL_LAMELLA,
    LEVEL_PROJECT,
    NONE,
    SOME,
    ConflictResolution,
)
from am_segmentation_manager.gui.report import ConflictRow
from am_segmentation_manager.metadata import FIELDS_BY_NAME
from am_segmentation_manager.models import ImageRecord
from am_segmentation_manager.operations.merge import OURS, THEIRS

ROOT = Path("/collection")
P = "session_a/Auto-1"
Q = "session_a/Auto-2"


def _group(
    project: str, lamella: str, number: int, beam: str = "SEM", *, corrected=False
) -> ConflictRow:
    """One label group in conflict, named the way Adaptive Polishing names it."""
    stage = ROOT / project / lamella / "adaptive_polish_2025-09-05-01-35-01PM"
    name = f"{lamella}_AP_img_{number:03d}_{beam}.tif"
    record = ImageRecord(
        image_path=stage / beam.lower() / name,
        lamella=lamella,
        modality=beam,
        stage_path=stage,
    )
    key = f"{project}/{lamella}/{stage.name}/{beam.lower()}_labels/{name}"
    return ConflictRow(
        key=key, title=key, corrected=corrected,
        project=project, lamella=lamella, record=record,
    )


def _field(project: str, name: str) -> ConflictRow:
    """One metadata field in conflict."""
    return ConflictRow(
        key=f"{project}:{name}", title=name, project=project, metadata_field=name
    )


def _node(nodes, *captions):
    """Return the row reached by following *captions* down from *nodes*."""
    node = None
    for caption in captions:
        node = next(candidate for candidate in nodes if candidate.caption == caption)
        nodes = node.children
    return node


# ---------------------------------------------------------------------------
# Where each conflict sits
# ---------------------------------------------------------------------------

def test_groups_are_arranged_by_folder_project_and_lamella():
    resolution = ConflictResolution([
        _group("session_b/Auto-3", "01-cat", 1),
        _group(P, "02-dog", 1),
        _group(P, "01-cat", 1),
        _group("Auto-root", "01-cat", 1),
    ])

    assert [(node.level, node.caption) for node in resolution.labels] == [
        (LEVEL_PROJECT, "Auto-root"),
        (LEVEL_FOLDER, "session_a"),
        (LEVEL_FOLDER, "session_b"),
    ], "a project at the scan root hangs from the top, as in the project list"
    project = _node(resolution.labels, "session_a", "Auto-1")
    assert project.level == LEVEL_PROJECT
    assert [(n.level, n.caption) for n in project.children] == [
        (LEVEL_LAMELLA, "01-cat"),
        (LEVEL_LAMELLA, "02-dog"),
    ]
    assert project.children[0].children[0].level == LEVEL_CONFLICT


def test_a_lamella_lists_its_conflicts_in_acquisition_order():
    """SEM before FIB within one acquisition, as the timeline has it --
    which is not the order their names sort in."""
    resolution = ConflictResolution([
        _group(P, "01-cat", 2),
        _group(P, "01-cat", 1, "FIB"),
        _group(P, "01-cat", 1),
    ])

    lamella = _node(resolution.labels, "session_a", "Auto-1", "01-cat")

    assert [node.caption for node in lamella.children] == [
        "01-cat_AP_img_001_SEM.tif",
        "01-cat_AP_img_001_FIB.tif",
        "01-cat_AP_img_002_SEM.tif",
    ]


def test_a_group_that_cannot_be_placed_follows_those_that_can():
    loose = ConflictRow(
        key=f"{P}/01-cat/aaa_loose.tif", title="", project=P, lamella="01-cat"
    )
    placed = _group(P, "01-cat", 1)

    lamella = _node(
        ConflictResolution([loose, placed]).labels, "session_a", "Auto-1", "01-cat"
    )

    assert lamella.keys == (placed.key, loose.key)


def test_fields_are_a_tree_of_their_own_in_the_order_of_the_form():
    group = _group(P, "01-cat", 1)
    resolution = ConflictResolution(
        [_field(P, "grid_id"), group, _field(P, "description")]
    )

    project = _node(resolution.fields, "session_a", "Auto-1")

    assert [node.caption for node in project.children] == [
        FIELDS_BY_NAME["description"].label,
        FIELDS_BY_NAME["grid_id"].label,
    ]
    assert resolution.labels[0].keys == (group.key,)


# ---------------------------------------------------------------------------
# Deciding
# ---------------------------------------------------------------------------

def test_nothing_is_decided_in_advance():
    rows = [_group(P, "01-cat", 1), _field(P, "description")]
    resolution = ConflictResolution(rows)

    assert resolution.resolutions == {}
    assert resolution.outstanding() == 2
    assert resolution.side(rows[0].key) is None


def test_deciding_a_row_decides_every_conflict_under_it_and_nothing_else():
    rows = [_group(P, "01-cat", 1), _group(P, "02-dog", 1), _group(Q, "01-cat", 1)]
    resolution = ConflictResolution(rows)

    resolution.decide(_node(resolution.labels, "session_a", "Auto-1"), THEIRS)

    assert resolution.resolutions == {rows[0].key: THEIRS, rows[1].key: THEIRS}
    assert resolution.outstanding() == 1


def test_a_heading_shows_how_much_of_it_each_side_holds():
    """Asked per side, so a heading decided both ways shows some of each --
    the mixed state a single value per heading could not hold."""
    rows = [_group(P, "01-cat", n) for n in (1, 2, 3)]
    resolution = ConflictResolution(rows)
    lamella = _node(resolution.labels, "session_a", "Auto-1", "01-cat")
    assert resolution.coverage(lamella, OURS) == NONE

    resolution.decide(resolution.conflict(rows[0].key), OURS)
    resolution.decide(resolution.conflict(rows[1].key), THEIRS)
    assert resolution.coverage(lamella, OURS) == SOME
    assert resolution.coverage(lamella, THEIRS) == SOME

    resolution.decide(lamella, THEIRS)
    assert resolution.coverage(lamella, THEIRS) == ALL
    assert resolution.coverage(lamella, OURS) == NONE


def test_undeciding_puts_every_conflict_under_a_row_back():
    rows = [_group(P, "01-cat", n) for n in (1, 2)]
    resolution = ConflictResolution(rows)
    lamella = _node(resolution.labels, "session_a", "Auto-1", "01-cat")

    resolution.decide(lamella, OURS)
    resolution.decide(lamella, None)

    assert resolution.resolutions == {}
    assert resolution.outstanding() == 2


def test_a_side_that_is_neither_is_refused():
    resolution = ConflictResolution([_group(P, "01-cat", 1)])

    with pytest.raises(ValueError):
        resolution.decide(resolution.labels[0], "both")
    assert resolution.resolutions == {}


def test_deciding_a_whole_tree_leaves_the_other_alone():
    """Labels and fields are decided on separate pages, and a button deciding
    every label must not answer a field nobody has looked at."""
    resolution = ConflictResolution(
        [_group(P, "01-cat", 1), _group(Q, "01-cat", 1), _field(P, "description")]
    )

    resolution.decide_all(resolution.labels, OURS)

    assert resolution.outstanding(resolution.labels) == 0
    assert resolution.outstanding(resolution.fields) == 1


def test_a_heading_counts_the_proofread_work_under_it():
    resolution = ConflictResolution([
        _group(P, "01-cat", 1, corrected=True),
        _group(P, "01-cat", 2),
        _group(Q, "01-cat", 1, corrected=True),
    ])

    assert resolution.proofread(_node(resolution.labels, "session_a")) == 2
    assert resolution.proofread(_node(resolution.labels, "session_a", "Auto-1")) == 1


# ---------------------------------------------------------------------------
# Moving
# ---------------------------------------------------------------------------

def test_stepping_within_a_lamella_stops_at_its_ends():
    rows = [_group(P, "01-cat", 1), _group(P, "01-cat", 2), _group(P, "02-dog", 1)]
    first, second, elsewhere = (row.key for row in rows)
    resolution = ConflictResolution(rows)

    assert resolution.step_within(first, 1) == second
    assert resolution.step_within(second, 1) is None
    assert resolution.step_within(first, -1) is None
    assert resolution.step_within(elsewhere, 1) is None


def test_stepping_across_reaches_the_next_lamella_even_in_another_project():
    rows = [
        _group(P, "01-cat", 1),
        _group(P, "01-cat", 2),
        _group(P, "02-dog", 1),
        _group(Q, "01-cat", 1),
    ]
    resolution = ConflictResolution(rows)

    assert resolution.step_across(rows[1].key, 1) == rows[2].key
    assert resolution.step_across(rows[2].key, 1) == rows[3].key
    assert resolution.step_across(rows[3].key, 1) is None
    assert resolution.step_across(rows[2].key, -1) == rows[0].key


def test_stepping_across_fields_moves_between_projects():
    rows = [_field(P, "description"), _field(P, "grid_id"), _field(Q, "description")]
    resolution = ConflictResolution(rows)

    assert resolution.step_across(rows[1].key, 1) == rows[2].key
    assert resolution.step_across(rows[2].key, -1) == rows[0].key


def test_the_next_undecided_conflict_wraps_round_and_runs_out():
    rows = [_group(P, "01-cat", n) for n in (1, 2, 3)]
    first, second, third = (row.key for row in rows)
    resolution = ConflictResolution([*rows, _field(P, "description")])

    resolution.decide(resolution.conflict(second), OURS)
    assert resolution.next_undecided(first) == third
    assert resolution.next_undecided(third) == first
    assert resolution.first_undecided(resolution.labels) == first

    resolution.decide_all(resolution.labels, THEIRS)
    assert resolution.next_undecided(first) is None, (
        "an undecided field is on the other page, not the next conflict here"
    )
    assert resolution.first_undecided(resolution.labels) is None
