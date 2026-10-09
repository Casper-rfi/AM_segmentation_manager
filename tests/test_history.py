"""Tests for the append-only history log.

The log is the only record of who changed what, and it is written by several
processes at once, so what is pinned here is mostly what it survives: a
damaged line, a line from a version that knows more, two processes appending
together, and the same operation run twice. The digest is pinned against the
one thing that would make it useless -- moving when a label is recompressed.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from am_segmentation_manager.history import (
    AM_HISTORY_FILENAME,
    BY_SOURCES,
    FILE_VERSION,
    KINDS,
    MAX_CLASS_VALUE,
    USER_ENV_VAR,
    Event,
    HistoryError,
    append_event,
    build_event,
    containing_project,
    events_for,
    history_file_path,
    latest_for,
    pixel_digest,
    read_log,
    resolve_author,
    union,
)
from conftest import start_history, write_marker

np = pytest.importorskip("numpy")


AUTHOR = {"by": "casper", "by_source": "flag"}


def saved(target: str, digest: str = "aaaa", at: str | None = None, **overrides) -> Event:
    """Return a label-saved entry, with the fields every one of them needs."""
    fields = {"digest": digest, "dtype": "uint8", "shape": [4, 4], "empty": False}
    fields.update(overrides)
    return build_event("label-saved", target=target, at=at, **AUTHOR, **fields)


def label(value: int = 2, dtype=np.uint8, shape: tuple[int, int] = (8, 8)):
    """Return a small label array holding one square of *value*."""
    array = np.zeros(shape, dtype=dtype)
    array[2:5, 2:5] = value
    return array


# ---------------------------------------------------------------------------
# The digest
# ---------------------------------------------------------------------------

def test_the_digest_is_over_pixels_so_a_recompression_does_not_move_it(tmp_path: Path):
    tifffile = pytest.importorskip("tifffile")
    array = label()
    plain, packed = tmp_path / "a.tif", tmp_path / "b.tif"
    tifffile.imwrite(plain, array)
    tifffile.imwrite(packed, array, compression="zlib")

    assert plain.stat().st_size != packed.stat().st_size
    assert pixel_digest(tifffile.imread(plain)) == pixel_digest(tifffile.imread(packed))


def test_the_same_classes_digest_alike_whatever_width_they_are_stored_in():
    assert pixel_digest(label(dtype=np.uint8)) == pixel_digest(label(dtype=np.uint16))


def test_one_changed_pixel_changes_the_digest():
    array = label()
    moved = array.copy()
    moved[0, 0] = 1
    assert pixel_digest(array) != pixel_digest(moved)


def test_the_same_pixels_in_a_different_shape_do_not_digest_alike():
    flat = np.zeros((4, 16), dtype=np.uint8)
    square = flat.reshape((8, 8))
    assert pixel_digest(flat) != pixel_digest(square)


def test_a_value_too_large_to_be_a_class_is_refused_rather_than_wrapped():
    array = np.full((4, 4), MAX_CLASS_VALUE + 1, dtype=np.uint32)
    with pytest.raises(ValueError, match="class index"):
        pixel_digest(array)


def test_a_label_that_is_not_whole_numbers_is_refused():
    with pytest.raises(ValueError, match="whole class indices"):
        pixel_digest(np.zeros((4, 4), dtype=np.float32))


# ---------------------------------------------------------------------------
# Building an entry
# ---------------------------------------------------------------------------

def test_an_id_is_the_same_for_the_same_content_however_the_fields_were_ordered():
    one = build_event(
        "state-changed", target="a/b.tif", at="2026-01-01T00:00:00.000Z",
        old="selected", new="corrected", **AUTHOR,
    )
    other = build_event(
        "state-changed", target="a/b.tif", at="2026-01-01T00:00:00.000Z",
        new="corrected", old="selected", **AUTHOR,
    )
    assert one.id == other.id


def test_a_different_author_is_a_different_entry():
    common = dict(target="a/b.tif", at="2026-01-01T00:00:00.000Z",
                  old="selected", new="corrected")
    mine = build_event("state-changed", by="casper", by_source="flag", **common)
    theirs = build_event("state-changed", by="anna", by_source="flag", **common)
    assert mine.id != theirs.id


def test_an_entry_missing_a_field_its_kind_needs_is_refused():
    with pytest.raises(HistoryError, match="needs digest"):
        build_event("label-saved", target="a/b.tif", dtype="uint8", shape=[4, 4],
                    empty=False, **AUTHOR)


def test_an_entry_carrying_a_field_its_kind_does_not_declare_is_refused():
    with pytest.raises(HistoryError, match="does not carry"):
        saved("a/b.tif", invented=1)


def test_an_unknown_kind_is_never_written():
    with pytest.raises(HistoryError, match="not an entry this version writes"):
        build_event("label-improved", target="a/b.tif", **AUTHOR)


def test_an_entry_about_a_file_must_say_which():
    with pytest.raises(HistoryError, match="must say what it is about"):
        build_event("state-changed", old="selected", new="corrected", **AUTHOR)


def test_an_entry_about_the_log_itself_names_no_file():
    with pytest.raises(HistoryError, match="not about a particular file"):
        build_event("log-created", target="a/b.tif", project_id="x",
                    am_seg_version="0.0.0", **AUTHOR)


def test_every_kind_declares_only_fields_the_envelope_does_not_own():
    from am_segmentation_manager.history import ENVELOPE_KEYS

    for kind, spec in KINDS.items():
        overlap = (set(spec.required) | set(spec.optional)) & ENVELOPE_KEYS
        assert not overlap, f"{kind} redeclares {overlap}"


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------

def test_an_entry_survives_being_written_and_read_back(tmp_path: Path):
    written = append_event(tmp_path, saved("sem_labels/i.tif", changed_fraction=0.25))
    (read_back,) = read_log(tmp_path).events
    assert read_back == written


def test_a_project_nothing_has_been_recorded_about_reads_as_an_empty_log(tmp_path: Path):
    log = read_log(tmp_path)
    assert log.events == () and log.damaged == () and log.unrecognised == ()
    assert log.path == history_file_path(tmp_path)


def test_entries_come_back_in_the_order_they_were_written(tmp_path: Path):
    for index in range(5):
        append_event(tmp_path, saved(f"sem_labels/{index}.tif"))
    kinds = [event.target for event in read_log(tmp_path).events]
    assert kinds == [f"sem_labels/{index}.tif" for index in range(5)]


def test_one_labels_history_is_the_entries_naming_it(tmp_path: Path):
    append_event(tmp_path, saved("sem_labels/a.tif"))
    append_event(tmp_path, saved("sem_labels/b.tif"))
    append_event(tmp_path, saved("sem_labels/a.tif", digest="bbbb"))

    log = read_log(tmp_path)
    assert [event.data["digest"] for event in events_for(log, "sem_labels/a.tif")] == [
        "aaaa", "bbbb",
    ]
    assert latest_for(log, "sem_labels/a.tif").data["digest"] == "bbbb"
    assert latest_for(log, "sem_labels/never.tif") is None


# ---------------------------------------------------------------------------
# What a reader survives
# ---------------------------------------------------------------------------

def test_a_damaged_line_costs_its_own_contents_and_not_the_files(tmp_path: Path):
    append_event(tmp_path, saved("sem_labels/a.tif"))
    with history_file_path(tmp_path).open("a", encoding="utf-8") as handle:
        handle.write('{"v":1,"kind":"label-saved",\n')
    append_event(tmp_path, saved("sem_labels/b.tif"))

    log = read_log(tmp_path)
    assert [event.target for event in log.events] == [
        "sem_labels/a.tif", "sem_labels/b.tif",
    ]
    assert len(log.damaged) == 1 and log.damaged[0].number == 2


def test_an_entry_from_a_later_version_is_kept_apart_rather_than_refused(tmp_path: Path):
    append_event(tmp_path, saved("sem_labels/a.tif"))
    later = {"v": FILE_VERSION + 1, "id": "ffff", "kind": "label-tweaked",
             "at": "2027-01-01T00:00:00.000Z", "by": "anna", "by_source": "flag"}
    with history_file_path(tmp_path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(later) + "\n")

    log = read_log(tmp_path)
    assert len(log.events) == 1
    assert len(log.unrecognised) == 1 and not log.damaged
    assert log.unrecognised[0].raw == json.dumps(later)


def test_a_kind_this_version_does_not_know_is_kept_apart_too(tmp_path: Path):
    unknown = {"v": FILE_VERSION, "id": "ffff", "kind": "label-tweaked",
               "at": "2027-01-01T00:00:00.000Z", "by": "anna", "by_source": "flag",
               "target": "a/b.tif"}
    history_file_path(tmp_path).write_text(json.dumps(unknown) + "\n", encoding="utf-8")

    log = read_log(tmp_path)
    assert not log.events and not log.damaged and len(log.unrecognised) == 1


def test_a_target_that_leaves_the_project_is_not_read_as_an_entry(tmp_path: Path):
    escaping = saved("sem_labels/a.tif").document()
    escaping["target"] = "../elsewhere/a.tif"
    history_file_path(tmp_path).write_text(
        json.dumps(escaping) + "\n", encoding="utf-8"
    )

    log = read_log(tmp_path)
    assert not log.events and len(log.damaged) == 1


def test_an_unrecognised_line_is_preserved_verbatim_when_the_log_grows(tmp_path: Path):
    later = {"v": FILE_VERSION + 1, "id": "ffff", "kind": "label-tweaked",
             "at": "2027-01-01T00:00:00.000Z", "by": "anna", "by_source": "flag"}
    history_file_path(tmp_path).write_text(json.dumps(later) + "\n", encoding="utf-8")
    append_event(tmp_path, saved("sem_labels/a.tif"))

    assert json.dumps(later) in history_file_path(tmp_path).read_text(encoding="utf-8")


def test_a_blank_line_is_not_a_problem(tmp_path: Path):
    append_event(tmp_path, saved("sem_labels/a.tif"))
    with history_file_path(tmp_path).open("a", encoding="utf-8") as handle:
        handle.write("\n   \n")

    log = read_log(tmp_path)
    assert len(log.events) == 1 and not log.damaged


# ---------------------------------------------------------------------------
# Several writers
# ---------------------------------------------------------------------------

def test_a_line_ends_the_same_on_every_platform(tmp_path: Path):
    """A log is carried between machines; text mode on Windows would end its
    lines in \\r\\n and a copy made there would differ byte for byte."""
    append_event(tmp_path, saved("sem_labels/a.tif"))

    written = history_file_path(tmp_path).read_bytes()
    assert written.endswith(b"}\n") and b"\r" not in written


def test_two_processes_recording_at_once_both_land(tmp_path: Path):
    program = (
        "import sys;"
        "from pathlib import Path;"
        "from am_segmentation_manager.history import append_event, build_event;"
        "root, who = Path(sys.argv[1]), sys.argv[2];"
        "[append_event(root, build_event('state-changed',"
        " target=f'{who}/{n}.tif', by=who, by_source='flag',"
        " old='selected', new='corrected')) for n in range(40)]"
    )
    running = [
        subprocess.Popen([sys.executable, "-c", program, str(tmp_path), who])
        for who in ("casper", "anna")
    ]
    for process in running:
        assert process.wait(timeout=120) == 0

    log = read_log(tmp_path)
    assert len(log.events) == 80 and not log.damaged
    assert len({event.by for event in log.events}) == 2


# ---------------------------------------------------------------------------
# Combining logs
# ---------------------------------------------------------------------------

def test_combining_two_logs_keeps_each_entry_once_whichever_order_they_come_in():
    shared = saved("shared.tif", at="2026-01-01T00:00:00.000Z")
    theirs = [shared, saved("b.tif", at="2026-01-02T00:00:00.000Z")]
    mine = [shared, saved("c.tif", at="2026-01-03T00:00:00.000Z")]

    forwards = union([mine, theirs])
    backwards = union([theirs, mine])
    assert [event.id for event in forwards] == [event.id for event in backwards]
    assert len(forwards) == 3


def test_combining_a_log_with_itself_changes_nothing():
    entries = [saved("a.tif", at="2026-01-01T00:00:00.000Z")]
    assert union([entries, entries]) == entries


def test_the_same_operation_recorded_twice_is_one_entry():
    once = build_event("state-changed", target="a.tif", at="2026-01-01T00:00:00.000Z",
                       old="selected", new="corrected", **AUTHOR)
    again = build_event("state-changed", target="a.tif", at="2026-01-01T00:00:00.000Z",
                        old="selected", new="corrected", **AUTHOR)
    assert len(union([[once], [again]])) == 1


# ---------------------------------------------------------------------------
# Who the author is
# ---------------------------------------------------------------------------

def test_a_name_passed_in_beats_everything_else(monkeypatch):
    monkeypatch.setenv(USER_ENV_VAR, "anna")
    assert resolve_author("casper") == ("casper", "flag")


def test_the_environment_is_used_when_nothing_was_passed_in(monkeypatch):
    monkeypatch.setenv(USER_ENV_VAR, "anna")
    assert resolve_author() == ("anna", "env")


def test_the_name_in_the_config_file_is_used_next(monkeypatch, tmp_path: Path):
    monkeypatch.delenv(USER_ENV_VAR, raising=False)
    monkeypatch.setattr("am_segmentation_manager.history.CONFIG_DIRECTORY", tmp_path)
    (tmp_path / "user.yaml").write_text("user: casper\n", encoding="utf-8")
    assert resolve_author() == ("casper", "config")


def test_no_name_is_resolved_when_nobody_stated_one(monkeypatch, tmp_path: Path):
    """An account name is not a chosen identity, and a log leaves with the data.

    Depositing a collection deposits its logs, so a guess written into one is a
    name nobody agreed to publish.
    """
    monkeypatch.delenv(USER_ENV_VAR, raising=False)
    monkeypatch.setattr("am_segmentation_manager.history.CONFIG_DIRECTORY", tmp_path)

    assert resolve_author() == (None, None)


def test_a_blank_statement_is_not_a_statement(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(USER_ENV_VAR, "   ")
    monkeypatch.setattr("am_segmentation_manager.history.CONFIG_DIRECTORY", tmp_path)

    assert resolve_author() == (None, None)


def test_every_source_left_is_a_deliberate_statement(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("am_segmentation_manager.history.CONFIG_DIRECTORY", tmp_path)
    monkeypatch.delenv(USER_ENV_VAR, raising=False)

    assert resolve_author("Anna") == ("Anna", "flag")
    monkeypatch.setenv(USER_ENV_VAR, "Bo")
    assert resolve_author() == ("Bo", "env")
    assert "os" not in BY_SOURCES, "there is no value for a guess"


def test_an_entry_with_no_stated_author_carries_neither_field(tmp_path: Path):
    """It says what happened and when. That is the whole of it."""
    event = build_event("state-changed", target="a.tif", old="selected", new="corrected")

    assert "by" not in event.document()
    assert "by_source" not in event.document()

    append_event(tmp_path, event)
    (read_back,) = read_log(tmp_path).events
    assert read_back.by is None and read_back.by_source is None


def test_a_name_is_flattened_onto_one_line_so_it_cannot_break_the_file(tmp_path: Path):
    event = build_event("state-changed", target="a.tif", by="cas\nper",
                        by_source="flag", old="selected", new="corrected")
    append_event(tmp_path, event)
    assert len(read_log(tmp_path).events) == 1


# ---------------------------------------------------------------------------
# Finding the log
# ---------------------------------------------------------------------------

def test_a_label_finds_the_log_of_the_project_holding_it(tmp_path: Path):
    label_path = tmp_path / "lamella_01" / "sem_labels" / "i.tif"
    label_path.parent.mkdir(parents=True)
    label_path.touch()
    start_history(tmp_path)

    assert containing_project(label_path) == tmp_path


def test_a_path_with_no_log_above_it_finds_none(tmp_path: Path):
    assert containing_project(tmp_path) is None


def test_the_nearest_log_wins_where_one_project_sits_inside_another(tmp_path: Path):
    inner = tmp_path / "inner"
    inner.mkdir()
    start_history(tmp_path)
    start_history(inner)

    assert containing_project(inner) == inner


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------

def test_the_command_prints_a_projects_entries(tmp_path: Path, capsys):
    from am_segmentation_manager.cli import main

    start_history(tmp_path)

    append_event(tmp_path, saved("sem_labels/a.tif"))
    assert main(["history", str(tmp_path)]) == 0

    printed = capsys.readouterr().out
    assert "label-saved" in printed and "sem_labels/a.tif" in printed


def test_the_command_pointed_at_a_label_shows_only_that_label(tmp_path: Path, capsys):
    from am_segmentation_manager.cli import main

    start_history(tmp_path)

    append_event(tmp_path, saved("sem_labels/a.tif"))
    append_event(tmp_path, saved("sem_labels/b.tif"))
    wanted = tmp_path / "sem_labels" / "a.tif"
    wanted.parent.mkdir(parents=True)
    wanted.touch()

    assert main(["history", str(wanted)]) == 0
    printed = capsys.readouterr().out
    assert "sem_labels/a.tif" in printed and "sem_labels/b.tif" not in printed


def test_the_command_says_so_where_nothing_has_been_recorded(tmp_path: Path, capsys):
    from am_segmentation_manager.cli import main

    assert main(["history", str(tmp_path)]) == 1
    # The commands log to stdout, that being the command's own output.
    assert AM_HISTORY_FILENAME in capsys.readouterr().out


def test_a_name_given_on_somebody_else_s_behalf_is_shown_as_one(tmp_path: Path, capsys):
    """`declared` is the one source nobody can verify, so it is never silent."""
    from am_segmentation_manager.cli import main

    start_history(tmp_path)

    append_event(tmp_path, build_event(
        "state-changed", target="a.tif", by="Anna", by_source="declared",
        old="selected", new="corrected",
    ))
    main(["history", str(tmp_path)])

    assert "Anna (on somebody's say-so)" in capsys.readouterr().out


def test_an_entry_with_no_author_prints_without_one(tmp_path: Path, capsys):
    from am_segmentation_manager.cli import main

    start_history(tmp_path)
    append_event(tmp_path, build_event(
        "state-changed", target="a.tif", old="selected", new="corrected",
    ))

    main(["history", str(tmp_path)])

    printed = capsys.readouterr().out
    assert "state-changed" in printed and "None" not in printed


def test_entries_a_merge_appended_are_read_in_the_order_things_happened(tmp_path: Path):
    """A merge appends the copy's entries whole, and they are older than ours.

    Everything asking what happened *last* to a label reads this order: what the
    archive treats as machine output, and the version a revert puts back.
    """
    append_event(tmp_path, saved("sem_labels/a.tif", at="2026-09-04T10:00:00.000Z"))
    append_event(tmp_path, saved("sem_labels/a.tif", at="2026-08-01T09:00:00.000Z"))

    log = read_log(tmp_path)

    assert [event.at for event in log.events] == [
        "2026-08-01T09:00:00.000Z", "2026-09-04T10:00:00.000Z"
    ]
    assert latest_for(log, "sem_labels/a.tif").at == "2026-09-04T10:00:00.000Z"


def test_entries_sharing_a_millisecond_keep_the_order_they_were_written(tmp_path: Path):
    """A batch writes many entries in one millisecond; a hash must not reorder them."""
    for index in range(6):
        append_event(
            tmp_path, saved(f"sem_labels/{index}.tif", at="2026-09-04T10:00:00.000Z")
        )

    targets = [event.target for event in read_log(tmp_path).events]

    assert targets == [f"sem_labels/{index}.tif" for index in range(6)]


def test_a_log_at_a_directory_that_is_not_a_project_records_nothing(tmp_path: Path):
    """One dropped at a collection root would swallow every project below it.

    Its targets would be ``project_a/lam/sem_labels/img.tif`` where every other
    reader uses the per-project key, and it would travel with nothing.
    """
    from am_segmentation_manager.history import history_file_path

    history_file_path(tmp_path).touch()          # a collection root, not a project
    project = tmp_path / "project_a"
    write_marker(project)
    label = project / "lam" / "sem_labels" / "img.tif"
    label.parent.mkdir(parents=True)
    label.touch()

    assert containing_project(label) is None


def test_a_log_beside_a_marker_is_the_project_it_belongs_to(tmp_path: Path):
    project = tmp_path / "project_a"
    start_history(project)
    label = project / "lam" / "sem_labels" / "img.tif"
    label.parent.mkdir(parents=True)
    label.touch()

    assert containing_project(label) == project.resolve()


def test_the_nearest_project_wins_over_one_further_up(tmp_path: Path):
    outer = tmp_path / "outer"
    inner = outer / "inner"
    start_history(outer)
    start_history(inner, kind="am_metadata")

    assert containing_project(inner / "sem_labels" / "img.tif") == inner.resolve()


def test_a_stamp_is_utc_ends_in_z_and_is_to_the_second_unless_asked():
    import re
    from datetime import UTC, datetime

    from am_segmentation_manager.history import utc_stamp

    before = datetime.now(UTC).replace(microsecond=0)
    stamp, fine = utc_stamp(), utc_stamp("milliseconds")

    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", stamp)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", fine)
    assert datetime.fromisoformat(stamp) >= before
