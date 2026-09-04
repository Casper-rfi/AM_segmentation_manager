"""Writing the first history a project has, for the data already on disk.

Recording only happens where a log exists, so this is what a collection opts in
with: a ``log-created`` entry naming the project, then one ``baseline`` entry
per label group that has a label or a state other than selected.

**A baseline observes; it does not claim.** When each label was annotated is not
recoverable -- the collection has been copied, so no timestamp on disk means
anything -- so an entry records when it was *seen*, marked ``baseline-scan``,
and nothing here should be read as saying when the work was done.

**Who** is a different matter, because a name reaches a deposition where a
timestamp does not. By default a baseline is attributed to whoever runs it,
which is right for a collection they annotated themselves and wrong for one
that arrived annotated. So the annotator can be named, and is then recorded as
``declared`` -- somebody stating a name on another person's behalf, which this
package cannot verify. Starting the log stays attributed to whoever did it.

Re-running adds only what is missing, so a project gains baselines for labels
that arrived since the last run. A group already baselined is left alone even
if its label has changed since: a second baseline would quietly paper over a
change that belongs in the record as a change.
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .. import __version__
from ..history import (
    Recorder,
    history_file_path,
    read_log,
)
from ..models import ImageRecord, Project
from ..state import DEFAULT_STATE, RecordState
from .label_groups import build_groups
from .selection import current_state

logger = logging.getLogger(__name__)

__all__ = [
    "BASELINE_BASIS",
    "BaselineEntry",
    "BaselinePlan",
    "BaselineReport",
    "ProjectBaseline",
    "apply_baseline",
    "plan_baseline",
]

#: What a baseline entry says it is. An observation of what was on disk, never
#: a statement about when the annotation happened.
BASELINE_BASIS: str = "baseline-scan"


@dataclass(frozen=True)
class BaselineEntry:
    """One label group a baseline would record.

    Attributes:
        target: The group's key, relative to the project root.
        label_path: The label to read, or None where the group has none and is
            being recorded for its state alone.
        state: The workflow state declared for it.
    """

    target: str
    label_path: Path | None
    state: RecordState


@dataclass(frozen=True)
class ProjectBaseline:
    """What one project needs before its history is complete.

    Attributes:
        project: The project.
        project_id: Its identity, read from an existing log or newly minted.
        creates_log: Whether a ``log-created`` entry is still to be written.
        entries: The groups with nothing recorded about them yet.
        recorded: How many groups already have a baseline.
    """

    project: Project
    project_id: str
    creates_log: bool
    entries: tuple[BaselineEntry, ...]
    recorded: int = 0

    @property
    def is_empty(self) -> bool:
        """Return whether this project needs nothing written."""
        return not self.creates_log and not self.entries


@dataclass(frozen=True)
class BaselinePlan:
    """What baselining a collection would write, having written nothing."""

    projects: tuple[ProjectBaseline, ...]

    @property
    def entry_count(self) -> int:
        """Return how many baseline entries would be written."""
        return sum(len(project.entries) for project in self.projects)

    @property
    def log_count(self) -> int:
        """Return how many projects would gain a log."""
        return sum(1 for project in self.projects if project.creates_log)

    def describe(self) -> list[str]:
        """Return one line per project with something to write."""
        lines = []
        for entry in self.projects:
            if entry.is_empty:
                continue
            what = f"{len(entry.entries)} group(s)"
            if entry.creates_log:
                what += ", starting a log"
            if entry.recorded:
                what += f", {entry.recorded} already recorded"
            lines.append(f"{entry.project.key}: {what}")
        return lines


@dataclass(frozen=True)
class BaselineReport:
    """What baselining actually wrote.

    Attributes:
        logs_started: Projects that gained a log.
        entries_written: Baseline entries recorded.
        unreadable: Labels that would not open, recorded without their
            contents. Kept apart because they are the ones worth looking at.
    """

    logs_started: int = 0
    entries_written: int = 0
    unreadable: tuple[Path, ...] = ()


def plan_baseline(projects: Iterable[Project]) -> BaselinePlan:
    """Return what baselining *projects* would write, reading no labels.

    Deliberately cheap: a dry run answers how much there is to do without the
    per-label decode that doing it costs.
    """
    planned = []
    for project in projects:
        log = read_log(project.path)
        recorded = {
            event.target
            for event in log.events
            if event.kind == "baseline" and event.target
        }
        started = any(event.kind == "log-created" for event in log.events)

        entries = [
            entry
            for entry in _entries_of(project)
            if entry.target not in recorded
        ]
        planned.append(
            ProjectBaseline(
                project=project,
                project_id=_project_id(log.events) or uuid.uuid4().hex,
                creates_log=not started,
                entries=tuple(entries),
                recorded=len(recorded),
            )
        )
    return BaselinePlan(tuple(planned))


def apply_baseline(
    plan: BaselinePlan,
    *,
    recorder: Recorder,
    annotator: Recorder | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> BaselineReport:
    """Write *plan*, reading each label to record what it holds.

    Each project's log is started before its groups are recorded, so a run
    stopped partway leaves projects whose history begins properly rather than
    entries belonging to no declared project.

    Args:
        recorder: Who is running this. Starting a log is their act, whoever
            made the labels.
        annotator: Who to credit the labels to, where that is somebody else --
            a dataset annotated at another institute. Defaults to *recorder*.
            **The two are kept apart deliberately**: attributing the log's
            creation to a person who never touched this machine would replace
            one false record with another.
    """
    crediting = annotator if annotator is not None else recorder
    logs, written, unreadable = 0, 0, []
    total = plan.entry_count
    done = 0

    for project in plan.projects:
        if project.creates_log:
            _start_log(project, recorder)
            logs += 1
        for entry in project.entries:
            fields, failed = _contents_of(entry.label_path)
            if failed is not None:
                unreadable.append(failed)
            recorded = crediting.record(
                project.project.path / entry.target,
                "baseline",
                state=entry.state,
                basis=BASELINE_BASIS,
                **fields,
            )
            if recorded is not None:
                written += 1
            done += 1
            if on_progress is not None:
                on_progress(done, total)

    logger.info("Recorded %s baseline entr(ies) across %s log(s)", written, logs)
    return BaselineReport(
        logs_started=logs, entries_written=written, unreadable=tuple(unreadable)
    )


# ---------------------------------------------------------------------------
# Working out what needs recording
# ---------------------------------------------------------------------------

def _entries_of(project: Project) -> list[BaselineEntry]:
    """Return every group of *project* worth recording.

    A group with no label and nothing declared about it has nothing to record:
    it is an image nobody has reached yet, and an entry saying so would be one
    per unannotated image in the collection.
    """
    entries = []
    for group in build_groups(_records_of(project)):
        identity = group.label_path or group.identity
        if identity is None:
            continue
        state = current_state(group.records[0])
        if group.label_path is None and state == DEFAULT_STATE:
            continue
        try:
            target = identity.resolve().relative_to(project.path.resolve()).as_posix()
        except ValueError:
            logger.warning("%s is not inside %s; not recording", identity, project.path)
            continue
        entries.append(
            BaselineEntry(target=target, label_path=group.label_path, state=state)
        )
    return sorted(entries, key=lambda entry: entry.target)


def _records_of(project: Project) -> list[ImageRecord]:
    """Return every image record of *project*."""
    return [
        record for lamella in project.lamellae for record in lamella.image_records
    ]


def _project_id(events: Sequence) -> str | None:
    """Return the identity an existing log declares, if it declares one."""
    for event in events:
        if event.kind == "log-created":
            declared = event.data.get("project_id")
            if isinstance(declared, str) and declared:
                return declared
    return None


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _start_log(project: ProjectBaseline, recorder: Recorder) -> None:
    """Create the log and record what it belongs to.

    The identity goes in the log rather than in ``AM_metadata.yaml``: a project
    holding only Autolamella's ``experiment.yaml`` has no file of ours to put
    it in, and writing Autolamella's would misrepresent whose it is. The file
    is created first, because a recorder writes only where a log already
    exists -- which is the same rule that keeps a project nobody baselined
    untouched.
    """
    history_file_path(project.project.path).touch()
    recorder.record_project(
        project.project.path,
        "log-created",
        project_id=project.project_id,
        am_seg_version=__version__,
        note="Append-only history written by am-seg. A record, not a control file.",
    )


def _contents_of(label_path: Path | None) -> tuple[dict[str, object], Path | None]:
    """Return what a label holds, and the path if it could not be read.

    A label that will not open is still recorded, as a group whose contents are
    unknown: leaving it out would make it indistinguishable from a group nobody
    has annotated, and those need opposite actions.
    """
    if label_path is None:
        return {}, None
    try:
        from ..io.image import read_array
        from .label_io import label_fields

        return label_fields(read_array(label_path)), None
    except Exception as error:
        logger.warning("Could not read %s: %s", label_path, error)
        return {}, label_path
