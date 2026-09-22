"""A write that could replace a corrected label asks the state file, not the record.

A record holds what its stage's ``AM_state.yaml`` said when the collection was
scanned. Two ways that goes wrong: the file is damaged, which the scanner reads
as everything selected so a collection still opens; and a second window, or
the command line, marks a group corrected after this one scanned. Either way a
conversion or a save trusting the record would write over proofread work.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

pytest.importorskip("tifffile")
pytest.importorskip("cv2")
pytest.importorskip("numpy")

import numpy as np

from am_segmentation_manager.cli import main
from am_segmentation_manager.io.atomic import fingerprint_of
from am_segmentation_manager.operations.label_compression import compress_labels
from am_segmentation_manager.operations.label_save import (
    LabelToSave,
    plan_saves,
    save_labels,
)
from am_segmentation_manager.operations.predictions import plan_conversions
from am_segmentation_manager.operations.selection import (
    declared_states,
    transition_records,
)
from am_segmentation_manager.scanner import Scanner
from am_segmentation_manager.state import StateFileError, state_file_path
from conftest import build_dataset


def _sem_records(root: Path):
    index = Scanner(root).build_index()
    return [
        record
        for project in index.projects.values()
        for record in project.image_records
        if record.modality == "SEM"
    ]


def _damage(stage: Path) -> None:
    """Append a line YAML cannot parse, as a hand edit might."""
    path = state_file_path(stage)
    path.write_text(path.read_text() + "  broken: [unclosed\n")


@pytest.fixture
def corrected(tmp_path: Path):
    """A project whose SEM labels are all declared corrected, and its dataset."""
    dataset = build_dataset(tmp_path, predictions=True)
    transition_records(_sem_records(tmp_path), "corrected")
    return dataset


@pytest.fixture
def scanned_then_corrected(tmp_path: Path):
    """SEM records scanned while selected, then marked corrected elsewhere."""
    dataset = build_dataset(tmp_path, predictions=True)
    records = _sem_records(tmp_path)
    transition_records(_sem_records(tmp_path), "corrected")
    assert all(record.state == "selected" for record in records)
    return dataset, records


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def test_declared_states_reads_what_the_file_says_now(scanned_then_corrected):
    dataset, records = scanned_then_corrected
    labels = [record.labels[0] for record in records]

    assert set(declared_states(labels).values()) == {"corrected"}


def test_a_stage_with_no_state_file_declares_everything_selected(tmp_path: Path):
    dataset = build_dataset(tmp_path)

    assert declared_states([dataset.label()]) == {dataset.label(): "selected"}


def test_declared_states_refuses_a_damaged_file(corrected):
    _damage(corrected.stage)

    with pytest.raises(StateFileError):
        declared_states([corrected.label()])


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def test_a_damaged_state_file_stops_a_conversion_over_corrected_labels(corrected, caplog):
    _damage(corrected.stage)
    with caplog.at_level(logging.CRITICAL):
        records = _sem_records(corrected.root)
    assert all(record.state == "selected" for record in records), "the scan reads it as selected"

    with pytest.raises(StateFileError):
        plan_conversions(records, overwrite=True)


def test_a_group_corrected_since_the_scan_is_refused_conversion(scanned_then_corrected):
    _, records = scanned_then_corrected

    plan = plan_conversions(records, overwrite=True)

    assert not plan.conversions
    assert len(plan.refused) == len(records)


def test_the_command_line_converts_nothing_past_a_damaged_state_file(corrected):
    _damage(corrected.stage)
    before = corrected.label().read_bytes()

    code = main([
        "convert-predictions", "--start-path", str(corrected.root),
        "--project", "all", "--lamella", "all", "--stage", "all",
        "--image", "all", "--modality", "SEM", "--overwrite",
    ])

    assert code == 2
    assert corrected.label().read_bytes() == before


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def _edit(label: Path) -> LabelToSave:
    """An edit to *label* whose opener believed it was not corrected."""
    return LabelToSave(
        destination=label,
        label=np.full((16, 16), 2, dtype=np.uint8),
        fingerprint=fingerprint_of(label),
        corrected=False,
    )


def test_a_save_over_a_group_corrected_since_it_was_opened_needs_consent(
    scanned_then_corrected,
):
    dataset, _ = scanned_then_corrected
    label = dataset.label()
    before = label.read_bytes()

    assert len(plan_saves([_edit(label)]).corrected) == 1
    with pytest.raises(PermissionError):
        save_labels([_edit(label)])
    assert label.read_bytes() == before


def test_a_save_into_a_stage_with_a_damaged_state_file_writes_nothing(corrected):
    _damage(corrected.stage)
    label = corrected.label()
    before = label.read_bytes()

    with pytest.raises(StateFileError):
        save_labels([_edit(label)], allow_corrected=False)
    assert label.read_bytes() == before


# ---------------------------------------------------------------------------
# Compressing
# ---------------------------------------------------------------------------

def test_compressing_a_group_corrected_since_the_scan_needs_consent(scanned_then_corrected):
    dataset, records = scanned_then_corrected
    before = dataset.label().read_bytes()

    with pytest.raises(PermissionError):
        compress_labels(records)
    assert dataset.label().read_bytes() == before


def test_compressing_past_a_damaged_state_file_changes_nothing(corrected, caplog):
    _damage(corrected.stage)
    with caplog.at_level(logging.CRITICAL):
        records = _sem_records(corrected.root)
    before = corrected.label().read_bytes()

    with pytest.raises(StateFileError):
        compress_labels(records)
    assert corrected.label().read_bytes() == before
