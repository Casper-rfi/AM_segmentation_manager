"""Tests for the command that gives a collection its first history.

Two things matter more than the entries themselves. **It writes nothing but
logs** -- a collection is 211 GB of irreplaceable images and this walks all of
it, so the tests compare the whole tree before and after. And **it is
re-runnable**: labels arrive after a baseline, and a second run must add those
without duplicating what it already said, or the record doubles every time
somebody runs it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from am_segmentation_manager.history import (
    Recorder,
    history_file_path,
    read_log,
)
from am_segmentation_manager.operations.init_history import (
    BASELINE_BASIS,
    apply_baseline,
    plan_baseline,
)
from am_segmentation_manager.scanner import Scanner
from conftest import build_dataset, start_history

np = pytest.importorskip("numpy")
pytest.importorskip("tifffile")


@pytest.fixture
def recorder() -> Recorder:
    """A recorder attributing everything to a named author."""
    return Recorder(by="casper", by_source="flag")


def projects_under(root: Path):
    """Return every project the scanner finds under *root*."""
    return Scanner(root).build_index().all_projects()


def baseline(root: Path, recorder: Recorder):
    """Plan and apply a baseline over everything under *root*."""
    return apply_baseline(plan_baseline(projects_under(root)), recorder=recorder)


def snapshot(root: Path) -> dict[str, str]:
    """Return every file under *root* and the hash of its bytes."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def baselines(project: Path):
    """Return the baseline entries recorded for *project*."""
    return [event for event in read_log(project).events if event.kind == "baseline"]


# ---------------------------------------------------------------------------
# What it writes, and what it must not
# ---------------------------------------------------------------------------

def test_it_writes_logs_and_touches_nothing_else(tmp_path: Path, recorder):
    dataset = build_dataset(tmp_path, labels="annotated")
    before = snapshot(tmp_path)

    baseline(tmp_path, recorder)

    after = snapshot(tmp_path)
    added = set(after) - set(before)
    assert added == {str(history_file_path(dataset.project).relative_to(tmp_path))}
    assert {name: after[name] for name in before} == before


def test_a_log_opens_by_saying_which_project_it_belongs_to(tmp_path: Path, recorder):
    dataset = build_dataset(tmp_path, labels="annotated")

    baseline(tmp_path, recorder)

    first = read_log(dataset.project).events[0]
    assert first.kind == "log-created"
    assert first.target is None
    assert first.data["project_id"]
    assert "am_seg_version" in first.data


def test_a_baseline_says_it_observed_rather_than_when_the_work_was_done(
    tmp_path: Path, recorder
):
    dataset = build_dataset(tmp_path, labels="annotated")

    baseline(tmp_path, recorder)

    for event in baselines(dataset.project):
        assert event.data["basis"] == BASELINE_BASIS


def test_an_annotated_label_is_recorded_with_what_it_holds(tmp_path: Path, recorder):
    dataset = build_dataset(tmp_path, labels="annotated")

    baseline(tmp_path, recorder)

    recorded = baselines(dataset.project)
    assert recorded
    for event in recorded:
        assert event.data["empty"] is False
        assert event.data["digest"] and event.data["shape"]
        assert event.data["state"] == "selected"


def test_an_empty_label_is_recorded_as_empty(tmp_path: Path, recorder):
    dataset = build_dataset(tmp_path, labels="empty")

    baseline(tmp_path, recorder)

    recorded = baselines(dataset.project)
    assert recorded and all(event.data["empty"] is True for event in recorded)


def test_a_group_with_no_label_and_nothing_declared_is_not_recorded(
    tmp_path: Path, recorder
):
    """Otherwise a baseline is one entry per unannotated image in the archive."""
    build_dataset(tmp_path, labels="none")

    report = baseline(tmp_path, recorder)

    assert report.entries_written == 0


def test_a_deselected_group_with_no_label_is_recorded_for_its_state(
    tmp_path: Path, recorder
):
    from am_segmentation_manager.operations.selection import transition_records

    dataset = build_dataset(tmp_path, labels="none")
    records = projects_under(tmp_path)[0].lamellae[0].image_records
    transition_records(records[:1], "deselected")

    baseline(tmp_path, recorder)

    deselected = [
        event for event in baselines(dataset.project)
        if event.data["state"] == "deselected"
    ]
    assert len(deselected) == 1
    assert "digest" not in deselected[0].data


def test_a_label_that_will_not_open_is_recorded_as_contents_unknown(
    tmp_path: Path, recorder
):
    """Left out, it would look like a group nobody annotated -- opposite actions."""
    dataset = build_dataset(tmp_path, labels="annotated")
    broken = next(iter(dataset.label_dirs.values())).glob("*.tif")
    target = next(broken)
    target.write_bytes(b"not a tiff at all")

    report = baseline(tmp_path, recorder)

    assert target in report.unreadable
    unknown = [event for event in baselines(dataset.project) if "digest" not in event.data]
    assert len(unknown) == 1


# ---------------------------------------------------------------------------
# Running it again
# ---------------------------------------------------------------------------

def test_a_second_run_adds_nothing(tmp_path: Path, recorder):
    dataset = build_dataset(tmp_path, labels="annotated")
    baseline(tmp_path, recorder)
    after_first = history_file_path(dataset.project).read_text(encoding="utf-8")

    report = baseline(tmp_path, recorder)

    assert report.entries_written == 0 and report.logs_started == 0
    assert history_file_path(dataset.project).read_text(encoding="utf-8") == after_first


def test_a_label_arriving_later_is_baselined_by_the_next_run(tmp_path: Path, recorder):
    dataset = build_dataset(tmp_path, labels="annotated")
    baseline(tmp_path, recorder)
    before = len(baselines(dataset.project))

    condition = next(iter(dataset.image_dirs))
    _write_tif(dataset.image_dirs[condition] / "extra_SEM.tif")
    _write_tif(dataset.label_dirs[condition] / "extra_SEM.tif", value=2)

    report = baseline(tmp_path, recorder)

    assert report.entries_written == 1
    assert len(baselines(dataset.project)) == before + 1


def test_a_group_already_baselined_is_not_baselined_again_when_it_changes(
    tmp_path: Path, recorder
):
    """A second baseline would paper over a change that belongs in the record."""
    dataset = build_dataset(tmp_path, labels="annotated")
    baseline(tmp_path, recorder)
    first = {event.data["digest"] for event in baselines(dataset.project)}

    changed = next(iter(dataset.label_dirs.values())).glob("*.tif")
    _write_tif(next(changed), value=7)

    report = baseline(tmp_path, recorder)

    assert report.entries_written == 0
    assert {event.data["digest"] for event in baselines(dataset.project)} == first


def test_a_project_keeps_the_identity_its_log_already_declares(tmp_path: Path, recorder):
    dataset = build_dataset(tmp_path, labels="annotated")
    baseline(tmp_path, recorder)
    declared = read_log(dataset.project).events[0].data["project_id"]

    plan = plan_baseline(projects_under(tmp_path))

    assert plan.projects[0].project_id == declared
    assert not plan.projects[0].creates_log


def test_a_log_started_by_hand_gains_the_entry_that_names_its_project(
    tmp_path: Path, recorder
):
    """Stage 2 opts a project in by creating the file; this completes it."""
    dataset = build_dataset(tmp_path, labels="annotated")
    start_history(dataset.project)

    report = baseline(tmp_path, recorder)

    assert report.logs_started == 1
    assert read_log(dataset.project).events[0].kind == "log-created"


# ---------------------------------------------------------------------------
# Planning without doing
# ---------------------------------------------------------------------------

def test_planning_writes_nothing(tmp_path: Path):
    dataset = build_dataset(tmp_path, labels="annotated")
    before = snapshot(tmp_path)

    plan = plan_baseline(projects_under(tmp_path))

    assert plan.entry_count > 0 and plan.log_count == 1
    assert snapshot(tmp_path) == before
    assert not history_file_path(dataset.project).exists()


def test_a_plan_says_what_each_project_would_get(tmp_path: Path):
    build_dataset(tmp_path, labels="annotated", name="one")
    build_dataset(tmp_path, labels="annotated", name="two")

    described = plan_baseline(projects_under(tmp_path)).describe()

    assert len(described) == 2
    assert all("starting a log" in line for line in described)


# ---------------------------------------------------------------------------
# Through the command
# ---------------------------------------------------------------------------

def test_the_command_previews_by_default(tmp_path: Path, capsys):
    from am_segmentation_manager.cli import main

    dataset = build_dataset(tmp_path, labels="annotated")

    assert main(["init-history", "--start-path", str(tmp_path)]) == 0

    assert "Re-run with --apply" in capsys.readouterr().out
    assert not history_file_path(dataset.project).exists()


def test_the_command_writes_when_asked(tmp_path: Path):
    from am_segmentation_manager.cli import main

    dataset = build_dataset(tmp_path, labels="annotated")

    assert main(["init-history", "--start-path", str(tmp_path), "--apply"]) == 0

    assert baselines(dataset.project)


def test_the_command_attributes_entries_to_the_name_it_was_given(tmp_path: Path):
    from am_segmentation_manager.cli import main

    dataset = build_dataset(tmp_path, labels="annotated")

    main(["--user", "anna", "init-history", "--start-path", str(tmp_path), "--apply"])

    assert {event.by for event in read_log(dataset.project).events} == {"anna"}


def test_the_command_says_so_where_there_are_no_projects(tmp_path: Path):
    from am_segmentation_manager.cli import main

    assert main(["init-history", "--start-path", str(tmp_path)]) == 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tif(path: Path, value: int = 0) -> Path:
    """Write a tiny label or image file at *path*."""
    import tifffile

    array = np.zeros((8, 8), dtype=np.uint8)
    if value:
        array[1:4, 1:4] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, array)
    return path


# ---------------------------------------------------------------------------
# Crediting work done elsewhere
# ---------------------------------------------------------------------------

def test_a_dataset_annotated_elsewhere_is_credited_to_whoever_annotated_it(
    tmp_path: Path, recorder
):
    """The one place this design would otherwise record something untrue."""
    dataset = build_dataset(tmp_path, labels="annotated")

    apply_baseline(
        plan_baseline(projects_under(tmp_path)),
        recorder=recorder,
        annotator=recorder.attributed_to("Anna Example"),
    )

    events = read_log(dataset.project).events
    baselines = [event for event in events if event.kind == "baseline"]
    assert baselines
    assert {event.by for event in baselines} == {"Anna Example"}
    assert {event.by_source for event in baselines} == {"declared"}


def test_starting_the_log_stays_with_whoever_started_it(tmp_path: Path, recorder):
    """Crediting the annotator must not replace one false record with another.

    Anna never touched this machine; the log was created here, by the person
    adding her dataset to the collection.
    """
    dataset = build_dataset(tmp_path, labels="annotated")

    apply_baseline(
        plan_baseline(projects_under(tmp_path)),
        recorder=recorder,
        annotator=recorder.attributed_to("Anna Example"),
    )

    (created,) = [
        event for event in read_log(dataset.project).events
        if event.kind == "log-created"
    ]
    assert created.by == recorder.by
    assert created.by_source == recorder.by_source


def test_a_declared_name_is_not_read_as_the_person_speaking_for_themselves(
    tmp_path: Path, recorder
):
    """`by_source` is what keeps attribution from being read as more than it is."""
    build_dataset(tmp_path, labels="annotated")

    credited = recorder.attributed_to("Anna Example")

    assert credited.by_source == "declared"
    assert credited.by_source != recorder.by_source
    assert credited.batch == recorder.batch, "one run stays one run"


def test_without_a_name_the_baseline_is_attributed_to_whoever_ran_it(
    tmp_path: Path, recorder
):
    dataset = build_dataset(tmp_path, labels="annotated")

    apply_baseline(plan_baseline(projects_under(tmp_path)), recorder=recorder)

    assert {
        event.by for event in read_log(dataset.project).events
    } == {recorder.by}


def test_the_command_credits_the_annotator(tmp_path: Path):
    from am_segmentation_manager.cli import main

    dataset = build_dataset(tmp_path, labels="annotated")

    assert main([
        "--user", "Casper", "init-history", "--start-path", str(tmp_path),
        "--as", "Anna Example", "--apply",
    ]) == 0

    events = read_log(dataset.project).events
    by_kind = {event.kind: event.by for event in events}
    assert by_kind["baseline"] == "Anna Example"
    assert by_kind["log-created"] == "Casper"


# ---------------------------------------------------------------------------
# Bringing a project into a record the rest of the collection is already in
# ---------------------------------------------------------------------------

def test_a_project_that_arrives_later_is_brought_into_the_record(
    tmp_path: Path, recorder
):
    """Data arrives manually, and an Autolamella project is indexable at once."""
    from am_segmentation_manager.operations.init_history import ensure_recording

    first = build_dataset(tmp_path, name="already_here", labels="annotated")
    start_history(first.project)
    apply_baseline(plan_baseline(projects_under(tmp_path)), recorder=recorder)
    later = build_dataset(tmp_path, name="arrived_later", labels="annotated")

    report = ensure_recording(projects_under(tmp_path), recorder=recorder)

    assert report.logs_started == 1
    assert report.entries_written
    assert any(
        event.kind == "baseline" for event in read_log(later.project).events
    )


def test_it_does_not_opt_a_collection_in(tmp_path: Path, recorder):
    """Opting in is one considered act on real data, and stays one.

    This is what keeps it safe to open a collection nobody has opted in --
    including one this package is only ever allowed to read.
    """
    from am_segmentation_manager.history import history_file_path
    from am_segmentation_manager.operations.init_history import ensure_recording

    dataset = build_dataset(tmp_path, labels="annotated")

    report = ensure_recording(projects_under(tmp_path), recorder=recorder)

    assert report.entries_written == 0 and report.logs_started == 0
    assert not history_file_path(dataset.project).is_file(), "nothing was written"


def test_bringing_a_project_in_twice_records_it_once(tmp_path: Path, recorder):
    from am_segmentation_manager.operations.init_history import ensure_recording

    first = build_dataset(tmp_path, name="already_here", labels="annotated")
    start_history(first.project)
    build_dataset(tmp_path, name="arrived_later", labels="annotated")

    ensure_recording(projects_under(tmp_path), recorder=recorder)
    before = sum(len(read_log(p.path).events) for p in projects_under(tmp_path))
    ensure_recording(projects_under(tmp_path), recorder=recorder)

    assert sum(len(read_log(p.path).events) for p in projects_under(tmp_path)) == before


def test_a_group_a_save_already_recorded_is_not_baselined_after_it(
    tmp_path: Path, recorder
):
    """A baseline observes what is on disk, so one written after a change would
    claim to have seen something that had already happened."""
    import tifffile

    from am_segmentation_manager.io.atomic import fingerprint_of
    from am_segmentation_manager.operations.init_history import ensure_recording
    from am_segmentation_manager.operations.label_save import LabelToSave, save_labels

    dataset = build_dataset(tmp_path, labels="annotated")
    start_history(dataset.project)
    label = sorted(tmp_path.rglob("*labels/*.tif"))[0]
    pixels = tifffile.imread(label)
    pixels[4:8, 4:8] = 3
    save_labels(
        [LabelToSave(destination=label, label=pixels, fingerprint=fingerprint_of(label))],
        recorder=recorder,
    )

    ensure_recording(projects_under(tmp_path), recorder=recorder)

    events = read_log(dataset.project).events
    target = next(e.target for e in events if e.kind == "label-saved")
    kinds = [e.kind for e in events if e.target == target]
    assert "baseline" not in kinds, "the save already says what it held"
