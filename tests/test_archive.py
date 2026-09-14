"""Tests for the archive of labels an operation replaced.

The correctness requirement is the reserved name: indexed, a superseded version
is read as a current label and reaches a training CSV, which is a corrupted
training set rather than a slow scan. That is pinned first.

The rest is the decision about what earns keeping, which asks where the pixels
came from and never what state claims -- an untouched pseudo-label and one
somebody proofread for an hour are both ``annotated`` and both ``selected``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.history import Recorder, read_log
from am_segmentation_manager.io.atomic import fingerprint_of
from am_segmentation_manager.layout import ARCHIVE_DIRECTORY
from am_segmentation_manager.operations.archive import (
    Archivist,
    apply_prune,
    archive_replaced,
    archive_root,
    archived_versions,
    plan_prune,
    restore,
    versions_of,
)
from am_segmentation_manager.operations.label_io import (
    label_digest,
    label_fields,
    read_label,
    replacement_fields,
)
from am_segmentation_manager.operations.label_save import LabelToSave, save_labels
from conftest import build_dataset, start_history

np = pytest.importorskip("numpy")
pytest.importorskip("tifffile")


@pytest.fixture
def recorder() -> Recorder:
    """A recorder attributing everything to a named author."""
    return Recorder(by="casper", by_source="flag")


def logged(project: Path) -> Path:
    """Start a log in *project*, so operations there record and archive."""
    start_history(project)
    return project


def label(value: int = 2, shape: tuple[int, int] = (16, 16)):
    """Return a small label holding one square of *value*."""
    array = np.zeros(shape, dtype=np.uint8)
    array[2:6, 2:6] = value
    return array


def write_label(path: Path, array) -> Path:
    """Write *array* to *path* as a label file."""
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, array)
    return path


def save(destination: Path, array, recorder: Recorder, **kwargs):
    """Save *array* over *destination* the way an editor would."""
    return save_labels(
        [
            LabelToSave(
                destination=destination,
                label=array,
                created=not destination.exists(),
                fingerprint=fingerprint_of(destination),
                **kwargs,
            )
        ],
        recorder=recorder,
        allow_corrected=kwargs.get("corrected", False),
    )


# ---------------------------------------------------------------------------
# The reserved name
# ---------------------------------------------------------------------------

def test_an_archive_never_reaches_a_training_export(tmp_path: Path, recorder):
    """The failure this pins is a corrupted training set, not a slow scan."""
    from am_segmentation_manager.operations.csv import build_training_rows
    from am_segmentation_manager.scanner import Scanner

    dataset = build_dataset(tmp_path, labels="annotated")
    logged(dataset.project)
    before = build_training_rows(_records(tmp_path), tmp_path, paired=False)

    condition = next(iter(dataset.label_dirs))
    existing = next(dataset.label_dirs[condition].glob("*.tif"))
    save(existing, label(3), recorder)

    assert archive_root(dataset.project).is_dir()
    assert build_training_rows(_records(tmp_path), tmp_path, paired=False) == before
    assert Scanner(tmp_path).build_index()  # the scan still works


def test_an_archive_adds_no_image_sets_and_no_findings(tmp_path: Path, recorder):
    from am_segmentation_manager.survey import survey_collection

    dataset = build_dataset(tmp_path, labels="annotated")
    logged(dataset.project)
    before = _survey_categories(tmp_path)

    condition = next(iter(dataset.label_dirs))
    save(next(dataset.label_dirs[condition].glob("*.tif")), label(3), recorder)

    assert _survey_categories(tmp_path) == before
    assert survey_collection(tmp_path) is not None


def test_the_archive_directory_is_not_read_as_an_image_directory():
    from am_segmentation_manager.layout import (
        classify_image_directory,
        directory_role,
    )

    assert classify_image_directory(ARCHIVE_DIRECTORY) is None
    assert directory_role(ARCHIVE_DIRECTORY) == "archive"


# ---------------------------------------------------------------------------
# What earns keeping
# ---------------------------------------------------------------------------

def test_a_replaced_label_is_kept_whole(tmp_path: Path, recorder):
    import tifffile

    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    before = label()
    write_label(destination, before)

    save(destination, label(3), recorder)

    (version,) = versions_of(tmp_path, "sem_labels/i.tif")
    assert np.array_equal(tifffile.imread(version.path), before)
    assert version.event is not None and version.event.kind == "label-saved"


def test_a_new_label_archives_nothing(tmp_path: Path, recorder):
    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    destination.parent.mkdir(parents=True)

    save(destination, label(), recorder)

    assert not archive_root(tmp_path).exists()


def test_an_empty_placeholder_is_not_kept(tmp_path: Path, recorder):
    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    write_label(destination, np.zeros((16, 16), dtype=np.uint8))

    save(destination, label(), recorder)

    assert versions_of(tmp_path, "sem_labels/i.tif") == []


def test_the_same_pixels_written_again_are_not_kept(tmp_path: Path, recorder):
    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    pixels = label()
    write_label(destination, pixels)

    save(destination, pixels.copy(), recorder)

    assert versions_of(tmp_path, "sem_labels/i.tif") == []


def test_untouched_machine_output_is_not_kept(tmp_path: Path, recorder):
    """A pseudo-label nobody has touched is regenerable from the model."""
    from am_segmentation_manager.operations.label_io import label_fields

    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    machine = label()
    write_label(destination, machine)
    recorder.record(
        destination, "label-converted", model="fpn_v4", prediction_kind="am_seg",
        **label_fields(machine),
    )

    save(destination, label(3), recorder)

    assert versions_of(tmp_path, "sem_labels/i.tif") == []


def test_machine_output_somebody_edited_is_kept(tmp_path: Path, recorder):
    """The case the state cannot tell apart: both are annotated and selected."""
    from am_segmentation_manager.operations.label_io import label_fields

    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    machine = label()
    write_label(destination, machine)
    recorder.record(
        destination, "label-converted", model="fpn_v4", prediction_kind="am_seg",
        **label_fields(machine),
    )

    edited = machine.copy()
    edited[10:12, 10:12] = 1
    write_label(destination, edited)
    save(destination, label(3), recorder)

    (version,) = versions_of(tmp_path, "sem_labels/i.tif")
    assert version.event is not None


def test_a_label_of_unknown_provenance_is_kept(tmp_path: Path, recorder):
    """Nothing says where these pixels came from, so they are treated as work."""
    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    write_label(destination, label())

    save(destination, label(3), recorder)

    assert len(versions_of(tmp_path, "sem_labels/i.tif")) == 1


def test_a_project_with_no_log_gains_no_archive(tmp_path: Path, recorder):
    destination = tmp_path / "sem_labels" / "i.tif"
    write_label(destination, label())

    save(destination, label(3), recorder)

    assert not archive_root(tmp_path).exists()


def test_the_entry_says_where_the_version_went(tmp_path: Path, recorder):
    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    write_label(destination, label())

    save(destination, label(3), recorder)

    (event,) = [e for e in read_log(tmp_path).events if e.kind == "label-saved"]
    assert event.data["archived_as"].startswith(f"{ARCHIVE_DIRECTORY}/")
    assert (tmp_path / event.data["archived_as"]).is_file()


def test_an_unwritable_archive_does_not_stop_the_save(
    tmp_path: Path, recorder, monkeypatch
):
    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    write_label(destination, label())

    def refuse(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr("shutil.copy2", refuse)
    report = save(destination, label(3), recorder)

    assert report.written == (destination,)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def test_a_conversion_keeps_the_label_it_writes_over(tmp_path: Path, recorder):
    """The case the archive exists for: a machine guess over somebody's work."""
    from am_segmentation_manager.operations.archive import Archivist as _A

    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    write_label(destination, label())

    archivist = _A(recorder)
    kept = archivist.keep(destination, "deadbeefdeadbeef", empty=False)

    assert kept is not None and (tmp_path / kept).is_file()


# ---------------------------------------------------------------------------
# Restoring
# ---------------------------------------------------------------------------

def test_restoring_puts_the_pixels_back_and_keeps_what_it_replaced(
    tmp_path: Path, recorder
):
    import tifffile

    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    original = label()
    write_label(destination, original)
    save(destination, label(3), recorder)
    (version,) = versions_of(tmp_path, "sem_labels/i.tif")

    restore(version, destination, recorder=recorder)

    assert np.array_equal(tifffile.imread(destination), original)
    assert len(versions_of(tmp_path, "sem_labels/i.tif")) == 2


def test_a_restore_is_recorded_as_one_and_names_what_it_came_from(
    tmp_path: Path, recorder
):
    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    write_label(destination, label())
    save(destination, label(3), recorder)
    (version,) = versions_of(tmp_path, "sem_labels/i.tif")

    restore(version, destination, recorder=recorder)

    (event,) = [e for e in read_log(tmp_path).events if e.kind == "label-restored"]
    assert event.data["restored_from"]
    assert event.target == "sem_labels/i.tif"


def test_restoring_over_proofread_work_needs_saying_so(tmp_path: Path, recorder):
    """Restoring is the ordinary save, so the ordinary guard applies."""
    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    write_label(destination, label())
    save(destination, label(3), recorder)
    (version,) = versions_of(tmp_path, "sem_labels/i.tif")

    with pytest.raises(PermissionError):
        restore(version, destination, recorder=recorder, corrected=True)

    restore(
        version, destination, recorder=recorder, corrected=True, allow_corrected=True
    )


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

#: Nothing is left open, so the round a test wrote is closed and collapsible.
#: Pruning only ever collapses a closed round.
NOTHING_OPEN: set[str] = set()


def test_pruning_collapses_the_saves_of_a_closed_round(tmp_path: Path, recorder):
    """The gradient goes; the furthest back anybody can go stays."""
    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    write_label(destination, label(1))
    for value in (2, 3, 4, 5):
        save(destination, label(value), recorder)
    assert len(versions_of(tmp_path, "sem_labels/i.tif")) == 4

    plan = plan_prune(tmp_path, open_round=NOTHING_OPEN)

    assert len(plan.remove) == 3
    assert plan.kept == 1


def test_nothing_is_removed_until_a_savepoint_closes_a_round(tmp_path: Path, recorder):
    """Everything kept is the work in hand until somebody names its end.

    The honest consequence of the savepoints being the policy: with none taken,
    the whole archive is one open round.
    """
    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    write_label(destination, label(1))
    for value in (2, 3, 4, 5):
        save(destination, label(value), recorder)

    plan = plan_prune(tmp_path)

    assert not plan.remove
    assert plan.kept == 4


def test_work_done_since_the_last_savepoint_is_never_collapsed(tmp_path: Path, recorder):
    """Stepping back through recent edits is when it is most likely to be wanted."""
    from am_segmentation_manager.history import append_event, build_event

    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    write_label(destination, label(1))
    save(destination, label(2), recorder)
    save(destination, label(3), recorder)

    append_event(tmp_path, build_event(
        "checkpoint", by="casper", by_source="flag",
        savepoint="9f2c4b11", title="round 3",
    ))
    save(destination, label(4), recorder)
    save(destination, label(5), recorder)

    from am_segmentation_manager.operations.savepoint import open_round

    still_open = open_round(tmp_path)
    plan = plan_prune(tmp_path, open_round=still_open)

    assert len(still_open) == 2, "both saves after the savepoint kept a version"
    kept = {v.path.relative_to(tmp_path).as_posix() for v in plan.protected}
    assert still_open <= kept, "and neither is offered up"
    assert not {v.path.relative_to(tmp_path).as_posix() for v in plan.remove} & still_open


def test_pruning_never_removes_a_version_a_savepoint_names(tmp_path: Path, recorder):
    """The savepoints are the policy: a state you named stays reachable."""
    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    write_label(destination, label(1))
    for value in (2, 3, 4, 5):
        save(destination, label(value), recorder)
    kept = versions_of(tmp_path, "sem_labels/i.tif")
    named = kept[2].path.relative_to(tmp_path).as_posix()

    plan = plan_prune(tmp_path, reached=[named], open_round=NOTHING_OPEN)

    assert named not in {
        version.path.relative_to(tmp_path).as_posix() for version in plan.remove
    }
    assert any("savepoint" in _why(plan, version) for version in plan.protected)


def _why(plan, version) -> str:
    """Return why a protected version was protected, by re-asking the rule."""
    from am_segmentation_manager.operations.archive import _protection, _relative

    return _protection(version, is_oldest=False, needed={_relative(version)}) or ""


def test_pruning_never_removes_the_version_before_a_model_wrote(
    tmp_path: Path, recorder
):
    from am_segmentation_manager.operations.label_io import label_fields

    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    write_label(destination, label(1))
    save(destination, label(2), recorder)          # oldest, protected anyway
    save(destination, label(3), recorder)
    # A conversion over the label keeps a version, recorded as label-converted.
    previous = label(3)
    Archivist(recorder).keep(destination, label_fields(previous)["digest"], False)
    recorder.record(
        destination, "label-converted", model="fpn_v4", prediction_kind="am_seg",
        archived_as=versions_of(tmp_path, "sem_labels/i.tif")[-1]
        .path.relative_to(tmp_path).as_posix(),
        **label_fields(label(9)),
    )

    plan = plan_prune(tmp_path, open_round=NOTHING_OPEN)

    protected_kinds = {v.event.kind for v in plan.protected if v.event}
    assert "label-converted" in protected_kinds
    assert all(v.event is None or v.event.kind != "label-converted" for v in plan.remove)


def test_a_version_the_history_does_not_account_for_is_never_removed(
    tmp_path: Path, recorder
):
    logged(tmp_path)
    stray = archive_root(tmp_path) / "sem_labels" / "i__20260101T000000Z__abcdef.tif"
    write_label(stray, label())

    plan = plan_prune(tmp_path, open_round=NOTHING_OPEN)

    assert not plan.remove
    assert len(plan.protected) == 0 or all(v.event is None for v in plan.protected)


def test_pruning_removes_the_files_and_leaves_the_record(tmp_path: Path, recorder):
    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    write_label(destination, label(1))
    for value in (2, 3, 4, 5):
        save(destination, label(value), recorder)
    entries_before = len(read_log(tmp_path).events)

    plan = plan_prune(tmp_path, open_round=NOTHING_OPEN)
    removed = apply_prune(plan)

    assert removed == len(plan.remove)
    assert all(not version.path.exists() for version in plan.remove)
    assert len(read_log(tmp_path).events) == entries_before


def test_planning_a_prune_removes_nothing(tmp_path: Path, recorder):
    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    write_label(destination, label(1))
    for value in (2, 3, 4):
        save(destination, label(value), recorder)

    plan = plan_prune(tmp_path, open_round=NOTHING_OPEN)

    assert all(version.path.exists() for version in plan.remove)


# ---------------------------------------------------------------------------
# The commands
# ---------------------------------------------------------------------------

def test_the_archive_command_reports_what_is_kept(tmp_path: Path, recorder, capsys):
    from am_segmentation_manager.cli import main

    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    write_label(destination, label())
    save(destination, label(3), recorder)

    assert main(["archive", str(tmp_path)]) == 0

    printed = capsys.readouterr().out
    assert "sem_labels/i.tif" in printed and "1 version(s)" in printed


def test_the_prune_command_previews_by_default(tmp_path: Path, recorder, capsys):
    from am_segmentation_manager.cli import main

    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    write_label(destination, label(1))
    for value in (2, 3, 4, 5):
        save(destination, label(value), recorder)
    before = len(archived_versions(tmp_path))

    assert main(["prune-archive", str(tmp_path)]) == 0

    printed = capsys.readouterr().out
    assert "no savepoint" in printed.lower(), "and it says why nothing would go"
    assert len(archived_versions(tmp_path)) == before


def test_the_prune_command_keeps_what_a_savepoint_names(tmp_path: Path, recorder):
    """The savepoints are the policy, so the command needs no rule of its own."""
    from am_segmentation_manager.cli import main
    from am_segmentation_manager.operations.savepoint import (
        apply_savepoint,
        plan_savepoint,
    )
    from am_segmentation_manager.scanner import Scanner
    from conftest import write_marker

    write_marker(tmp_path)
    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    write_label(destination, label(1))
    save(destination, label(2), recorder)
    projects = Scanner(tmp_path).build_index().all_projects()
    apply_savepoint(plan_savepoint(projects, "round 3"), recorder=recorder)
    for value in (3, 4, 5):
        save(destination, label(value), recorder)
    named = {
        version.path
        for version in versions_of(tmp_path, "sem_labels/i.tif")
        if _digest_of(version.path) == _digest_of_label(label(2))
    }

    assert main(["prune-archive", str(tmp_path), "--apply"]) == 0

    assert all(path.exists() for path in named), "a named state stays reachable"


def _digest_of(path: Path) -> str:
    from am_segmentation_manager.io.image import read_array
    from am_segmentation_manager.operations.label_io import label_fields

    return str(label_fields(read_array(path))["digest"])


def _digest_of_label(array) -> str:
    from am_segmentation_manager.operations.label_io import label_fields

    return str(label_fields(array)["digest"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _records(root: Path):
    from am_segmentation_manager.scanner import Scanner

    index = Scanner(root).build_index()
    return [
        record
        for project in index.all_projects()
        for lamella in project.lamellae
        for record in lamella.image_records
    ]


def _survey_categories(root: Path):
    from am_segmentation_manager.survey import survey_collection

    survey = survey_collection(root)
    return sorted(
        (finding.category, str(finding.path))
        for project in survey.projects
        for finding in project.findings
    )


# ---------------------------------------------------------------------------
# The bookkeeping every write replacing a label shares
# ---------------------------------------------------------------------------

class _KeepingEverything:
    """An archivist that keeps whatever it is offered, and remembers what."""

    def __init__(self) -> None:
        self.kept: list[str] = []

    def keep(self, label_path: Path, previous_digest: str, empty: bool) -> str:
        self.kept.append(previous_digest)
        return "AM_archive/kept.tif"


def test_a_replacement_holding_the_same_pixels_is_not_archived(tmp_path: Path):
    archivist = _KeepingEverything()

    archived = archive_replaced(
        archivist, tmp_path / "label.tif", label(2), replacement=label(2)
    )

    assert archived is None
    assert archivist.kept == []


def test_a_write_naming_no_replacement_archives_even_unchanged_pixels(tmp_path: Path):
    archivist = _KeepingEverything()

    archived = archive_replaced(archivist, tmp_path / "label.tif", label(2))

    assert archived == "AM_archive/kept.tif"
    assert archivist.kept == [label_fields(label(2))["digest"]]


def test_nothing_replaced_archives_nothing(tmp_path: Path):
    archivist = _KeepingEverything()

    assert archive_replaced(archivist, tmp_path / "label.tif", None) is None
    assert archive_replaced(None, tmp_path / "label.tif", label(2)) is None
    assert archivist.kept == []


def test_a_replacement_is_described_by_what_it_replaced(tmp_path: Path):
    before, after = label(2), label(3)

    fields = replacement_fields(tmp_path / "label.tif", before, after, "AM_archive/v.tif")

    assert fields == {
        "parent_digest": label_fields(before)["digest"],
        "changed_fraction": pytest.approx(16 / 256),
        "archived_as": "AM_archive/v.tif",
    }


def test_a_label_written_where_none_was_describes_no_parent(tmp_path: Path):
    assert replacement_fields(tmp_path / "label.tif", None, label(2), None) == {}


def test_a_resized_replacement_names_its_parent_but_no_changed_fraction(tmp_path: Path):
    fields = replacement_fields(
        tmp_path / "label.tif", label(2, shape=(8, 8)), label(2), None
    )

    assert fields["parent_digest"] == label_fields(label(2, shape=(8, 8)))["digest"]
    assert fields["changed_fraction"] is None


def test_a_label_that_will_not_open_reads_as_nothing(tmp_path: Path, caplog):
    damaged = tmp_path / "label.tif"
    damaged.write_bytes(b"not a tiff")

    assert read_label(damaged) is None
    assert label_digest(damaged) is None
    assert "Could not read label" in caplog.text


def test_a_missing_label_has_no_digest(tmp_path: Path):
    assert label_digest(None) is None
    assert label_digest(tmp_path / "absent.tif") is None
    assert label_digest(write_label(tmp_path / "label.tif", label(2))) == (
        label_fields(label(2))["digest"]
    )
