"""The append-only record of what changed in a project, and who changed it.

``AM_history.jsonl`` sits at each project root, one JSON object per line,
appended and never rewritten. An entry names its target by the path relative to
that root -- for a label, the same group key the state file uses -- so the
several imaging conditions of one acquisition share one history.

The log is a record, never the authority on what a file currently holds: a
reader that disagrees with the filesystem reports the difference rather than
trusting either. Because nothing is ever rewritten, a line this version does
not understand is kept and reported instead of refused, and a damaged line
costs its own contents rather than the file's. Resolving who the author is
lives here too, being the one thing every entry needs.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, get_args

from .io.path import checked_relative_key

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "AM_HISTORY_FILENAME",
    "CONFIG_DIRECTORY",
    "FILE_VERSION",
    "KINDS",
    "BySource",
    "Event",
    "EventKind",
    "Recorder",
    "HistoryError",
    "HistoryLog",
    "UnusableLine",
    "append_event",
    "build_event",
    "containing_project",
    "events_for",
    "history_file_path",
    "latest_for",
    "new_batch",
    "pixel_digest",
    "resolve_author",
    "read_log",
    "union",
]

#: The log, one per project directory.
AM_HISTORY_FILENAME: Final[str] = "AM_history.jsonl"

#: Where this package keeps a person's own settings, as opposed to anything
#: belonging to a collection. The window's log file lives here too.
CONFIG_DIRECTORY: Final[Path] = Path.home() / ".am_segmentation_manager"

#: The file naming the author, when nothing more direct does.
USER_FILENAME: Final[str] = "user.yaml"

#: Environment variable naming the author.
USER_ENV_VAR: Final[str] = "AM_SEG_USER"

#: Written into every entry. Reserved for a format break: a *later* version on
#: one line is kept and reported, not refused, because the file is only ever
#: appended to and refusing it would lock a collaborator out of their own work.
FILE_VERSION: Final[int] = 1

#: What an entry records. Adding one is a row in :data:`KINDS`.
EventKind = Literal[
    "log-created",
    "baseline",
    "label-saved",
    "label-converted",
    "label-compressed",
    "label-restored",
    "state-changed",
    "metadata-changed",
    "renamed",
    "merged",
]

#: How the author's name was arrived at. ``os`` is a guess about who was at the
#: keyboard; the rest are statements. Recorded so attribution is never read as
#: more certain than it is.
BySource = Literal["flag", "env", "config", "os"]

BY_SOURCES: Final[tuple[BySource, ...]] = get_args(BySource)

#: Longest author name accepted. A name is written into every entry and into a
#: deposition, not used as a key.
MAX_AUTHOR_LENGTH: Final[int] = 64

#: Largest class index a label may hold. Digests normalise to 16-bit, so a
#: value above this is not a class-index label and is refused rather than
#: silently wrapped.
MAX_CLASS_VALUE: Final[int] = 65535

#: How many hex characters of each hash are kept. 64 bits: collision odds are
#: about 3e-8 at a million entries, against a realistic ceiling far below that.
DIGEST_LENGTH: Final[int] = 16

#: Prefixes keeping the two hashes in separate spaces, so a label digest can
#: never be mistaken for an entry id.
_LABEL_DOMAIN: Final[bytes] = b"amseg-label-v1\0"
_EVENT_DOMAIN: Final[bytes] = b"amseg-event-v1\0"

#: Keys the envelope owns. A kind may not declare a field named one of these.
ENVELOPE_KEYS: Final[frozenset[str]] = frozenset(
    {"v", "id", "kind", "at", "by", "by_source", "batch", "target", "note"}
)


class HistoryError(RuntimeError):
    """Raised when an entry cannot be built, or a log cannot be written."""


@dataclass(frozen=True)
class KindSpec:
    """What one kind of entry must and may carry beyond the envelope.

    Attributes:
        required: Fields without which the entry means nothing.
        optional: Fields that may be absent.
        needs_target: Whether the entry is about a particular file.
    """

    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    needs_target: bool = True


#: What each kind carries. One row per kind, so adding one is an edit here
#: rather than in every validator -- the shape ``metadata.FIELDS`` uses for the
#: same reason. Unknown fields are refused on the way in and kept on the way
#: out: writing is ours to control, reading is not.
KINDS: Final[dict[str, KindSpec]] = {
    "log-created": KindSpec(
        required=("project_id", "am_seg_version"), needs_target=False
    ),
    # The label fields are optional because a group can be deselected without
    # ever having had a label, and a label that will not open is worth
    # recording as a group whose contents are unknown rather than not at all.
    "baseline": KindSpec(
        required=("state", "basis"),
        optional=("digest", "dtype", "shape", "empty"),
    ),
    "label-saved": KindSpec(
        required=("digest", "dtype", "shape", "empty"),
        optional=("parent_digest", "changed_fraction", "archived_as"),
    ),
    "label-converted": KindSpec(
        required=("digest", "dtype", "shape", "empty", "model", "prediction_kind"),
        optional=("parent_digest", "changed_fraction", "archived_as"),
    ),
    "label-compressed": KindSpec(
        required=("digest", "parent_digest", "bytes_before", "bytes_after"),
    ),
    "label-restored": KindSpec(
        required=("digest", "parent_digest", "restored_from", "archived_as"),
        optional=("changed_fraction", "dtype", "shape", "empty"),
    ),
    "state-changed": KindSpec(required=("old", "new")),
    "metadata-changed": KindSpec(required=("field", "old", "new")),
    "renamed": KindSpec(required=("renamed_from", "scope")),
    "merged": KindSpec(
        required=("resolution", "source_project_id", "source_event", "digest"),
        optional=("parent_digest", "archived_as"),
    ),
}


@dataclass(frozen=True)
class Event:
    """One recorded change.

    Attributes:
        kind: What happened. One of :data:`KINDS`.
        at: When, as UTC to the millisecond.
        by: The author's chosen name.
        by_source: How that name was arrived at.
        data: The fields this kind carries, validated against its row.
        target: The file it is about, relative to the project root, or None
            for an entry about the log itself.
        batch: Groups the entries of one run, or None.
        note: Free text, never parsed.
        id: Hash of everything above, so two runs of one operation produce one
            entry and a union of two logs needs no coordination to deduplicate.
    """

    kind: str
    at: str
    by: str
    by_source: str
    data: Mapping[str, Any] = field(default_factory=dict)
    target: str | None = None
    batch: str | None = None
    note: str | None = None
    id: str = ""

    def document(self) -> dict[str, Any]:
        """Return the entry as it is written, envelope first then its fields."""
        document: dict[str, Any] = {"v": FILE_VERSION, "id": self.id, "kind": self.kind}
        document["at"] = self.at
        document["by"] = self.by
        document["by_source"] = self.by_source
        if self.batch is not None:
            document["batch"] = self.batch
        if self.target is not None:
            document["target"] = self.target
        if self.note is not None:
            document["note"] = self.note
        document.update(dict(sorted(self.data.items())))
        return document


@dataclass(frozen=True)
class UnusableLine:
    """A line a reader could not turn into an entry.

    Attributes:
        number: Which line it was, counting from one.
        reason: What was wrong with it.
        raw: The line itself, kept so a later version can still use it.
    """

    number: int
    reason: str
    raw: str


@dataclass(frozen=True)
class HistoryLog:
    """What one project's log holds.

    Attributes:
        path: The file, whether or not it exists.
        events: The entries understood, in the order written.
        damaged: Lines that are not usable entries at all.
        unrecognised: Lines written by a version that knows more than this one.
            Kept apart from *damaged* because they are somebody's real record.
    """

    path: Path
    events: tuple[Event, ...] = ()
    damaged: tuple[UnusableLine, ...] = ()
    unrecognised: tuple[UnusableLine, ...] = ()


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _hash(domain: bytes, payload: bytes) -> str:
    """Return the short hex digest of *payload* within *domain*."""
    return hashlib.sha256(domain + payload).hexdigest()[:DIGEST_LENGTH]


def pixel_digest(array: np.ndarray) -> str:
    """Return the digest of a label's pixels.

    Taken over the decoded array and its shape, **never the file bytes**:
    recompressing a label rewrites the file without changing the segmentation,
    and a digest that moved would report every compression run as an edit.

    Values are normalised to 16-bit for hashing, so the same classes stored as
    ``uint8`` and as ``uint16`` digest alike; the stored dtype is recorded in
    the entry instead of being folded in here.

    Raises:
        ValueError: If the array is not integral, holds a negative value, or
            holds one above :data:`MAX_CLASS_VALUE` -- none of which is a
            label of class indices, and all of which 16-bit would silently
            corrupt.
    """
    import numpy as np

    from .io.image import to_2d

    plane = np.ascontiguousarray(to_2d(array))
    if plane.dtype.kind not in ("u", "i", "b"):
        raise ValueError(
            f"a label must hold whole class indices, not {plane.dtype}"
        )
    if plane.size:
        if plane.dtype.kind == "i" and int(plane.min()) < 0:
            raise ValueError("a label may not hold a negative class index")
        if int(plane.max()) > MAX_CLASS_VALUE:
            raise ValueError(
                f"a label may not hold a class index above {MAX_CLASS_VALUE}"
            )
    payload = repr(tuple(plane.shape)).encode("utf-8") + b"\0"
    return _hash(_LABEL_DOMAIN, payload + plane.astype("<u2", copy=False).tobytes())


def _event_id(document: Mapping[str, Any]) -> str:
    """Return the id of an entry, hashed over its body without the id.

    Deterministic on content, so re-running an operation that changed nothing
    records nothing new. That is what makes the baseline idempotent and a union
    of two logs correct without either knowing about the other.
    """
    body = {key: value for key, value in document.items() if key != "id"}
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return _hash(_EVENT_DOMAIN, canonical.encode("utf-8"))


# ---------------------------------------------------------------------------
# Who the author is
# ---------------------------------------------------------------------------

def resolve_author(explicit: str | None = None) -> tuple[str, str]:
    """Return the author's name and how it was arrived at.

    Tried in order: what the caller was told, ``AM_SEG_USER``, the name in
    ``user.yaml``, then the account running the process. **Never fails**: an
    entry attributed to an account name is worth more than a refused save.

    Returns:
        The name, and one of :data:`BY_SOURCES`.
    """
    for candidate, source in (
        (explicit, "flag"),
        (os.environ.get(USER_ENV_VAR), "env"),
        (_author_from_config(), "config"),
    ):
        if candidate and candidate.strip():
            return _checked_author(candidate.strip()), source
    return _checked_author(_account_name()), "os"


def _author_from_config() -> str | None:
    """Return the name in ``user.yaml``, or None if there is not one."""
    path = CONFIG_DIRECTORY / USER_FILENAME
    if not path.is_file():
        return None
    try:
        from .io.yaml import read_yaml

        document = read_yaml(path)
    except Exception as error:  # pragma: no cover - unreadable config
        logger.warning("Could not read %s: %s", path, error)
        return None
    value = document.get("user") if isinstance(document, dict) else None
    return value if isinstance(value, str) else None


def _account_name() -> str:
    """Return the account running this process, or a placeholder."""
    import getpass

    try:
        return getpass.getuser()
    except Exception:  # pragma: no cover - no account name anywhere
        return "unknown"


def _checked_author(name: str) -> str:
    """Return *name* trimmed to something that can be written on one line."""
    flattened = " ".join(name.split())
    if not flattened:
        return "unknown"
    return flattened[:MAX_AUTHOR_LENGTH]


def new_batch() -> str:
    """Return an id grouping the entries of one run."""
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def _now() -> str:
    """Return the current time as UTC to the millisecond."""
    stamp = datetime.now(UTC).isoformat(timespec="milliseconds")
    return stamp.replace("+00:00", "Z")


def build_event(
    kind: str,
    *,
    by: str,
    by_source: str,
    target: str | None = None,
    batch: str | None = None,
    note: str | None = None,
    at: str | None = None,
    **data: Any,
) -> Event:
    """Return a validated entry, with its id computed.

    Writing is this package's to control, so an unknown kind, a missing
    required field or a field the kind does not declare is refused here rather
    than written and puzzled over later.

    Raises:
        HistoryError: If the kind is unknown, a required field is missing, an
            undeclared field is passed, or a target is required and absent.
    """
    spec = KINDS.get(kind)
    if spec is None:
        raise HistoryError(f"{kind!r} is not an entry this version writes")
    if by_source not in BY_SOURCES:
        raise HistoryError(f"{by_source!r} is not one of {', '.join(BY_SOURCES)}")

    if spec.needs_target and target is None:
        raise HistoryError(f"a {kind} entry must say what it is about")
    if not spec.needs_target and target is not None:
        raise HistoryError(f"a {kind} entry is not about a particular file")

    missing = [name for name in spec.required if name not in data]
    if missing:
        raise HistoryError(f"a {kind} entry needs {', '.join(sorted(missing))}")
    allowed = set(spec.required) | set(spec.optional)
    unexpected = sorted(set(data) - allowed)
    if unexpected:
        raise HistoryError(f"a {kind} entry does not carry {', '.join(unexpected)}")
    collisions = sorted(set(data) & ENVELOPE_KEYS)
    if collisions:
        raise HistoryError(f"{', '.join(collisions)} belongs to every entry")

    event = Event(
        kind=kind,
        at=at or _now(),
        by=_checked_author(by),
        by_source=by_source,
        data=dict(data),
        target=target,
        batch=batch,
        note=note,
    )
    return Event(**{**event.__dict__, "id": _event_id(event.document())})


# ---------------------------------------------------------------------------
# Reading and writing
# ---------------------------------------------------------------------------

def history_file_path(project_directory: Path) -> Path:
    """Return where *project_directory*'s log lives, existing or not."""
    return project_directory / AM_HISTORY_FILENAME


def containing_project(path: Path) -> Path | None:
    """Return the nearest directory at or above *path* that holds a log.

    Lets a command be pointed at a label and find the log that records it,
    without needing the scanner to say which project the label belongs to.

    **Resolved before the walk**, so the answer can be subtracted from a
    resolved path to give a target. A root found through a symlinked parent --
    ``/tmp`` being one on macOS -- is not a prefix of the resolved file, and
    every entry about it would be quietly dropped.
    """
    resolved = path.resolve()
    start = resolved if resolved.is_dir() else resolved.parent
    for directory in (start, *start.parents):
        if history_file_path(directory).is_file():
            return directory
    return None


def append_event(project_directory: Path, event: Event) -> Event:
    """Append *event* to *project_directory*'s log and return it.

    One line, written in a single append, so two processes recording at once
    interleave rather than one discarding the other's entry. There is no
    read-modify-write and so nothing to be stale against.

    Raises:
        HistoryError: If the entry cannot be serialised onto one line.
    """
    line = json.dumps(event.document(), ensure_ascii=False, separators=(",", ":"))
    if "\n" in line:
        raise HistoryError("an entry must fit on one line")
    path = history_file_path(project_directory)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    logger.debug("Recorded %s in %s", event.kind, path)
    return event


def read_log(project_directory: Path) -> HistoryLog:
    """Read *project_directory*'s log.

    A missing file reads as an empty log, which is a project nothing has been
    recorded about yet -- the same answer, so a caller need not ask which.

    **Nothing here raises over the contents.** A line that is not a usable
    entry is collected in :attr:`HistoryLog.damaged`, and one written by a
    later version in :attr:`HistoryLog.unrecognised`; refusing four thousand
    good entries because of one bad line would destroy more than it protects.
    """
    path = history_file_path(project_directory)
    if not path.is_file():
        return HistoryLog(path=path)

    events: list[Event] = []
    damaged: list[UnusableLine] = []
    unrecognised: list[UnusableLine] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            outcome, detail = _parse_line(line, project_directory)
            if isinstance(outcome, Event):
                events.append(outcome)
            elif outcome == "unrecognised":
                unrecognised.append(UnusableLine(number, detail, line))
            else:
                damaged.append(UnusableLine(number, detail, line))

    if damaged:
        logger.warning("%s unusable line(s) in %s", len(damaged), path)
    if unrecognised:
        logger.info(
            "%s entr(ies) in %s were written by a newer am-seg", len(unrecognised), path
        )
    return HistoryLog(
        path=path,
        events=tuple(events),
        damaged=tuple(damaged),
        unrecognised=tuple(unrecognised),
    )


def _parse_line(line: str, project_directory: Path) -> tuple[Event | str, str]:
    """Return the entry a line holds, or why it is not one."""
    try:
        document = json.loads(line)
    except ValueError as error:
        return "damaged", f"not JSON: {error}"
    if not isinstance(document, dict):
        return "damaged", "not an object"

    version = document.get("v")
    if not isinstance(version, int) or isinstance(version, bool):
        return "damaged", "no version"
    if version > FILE_VERSION:
        return "unrecognised", f"written to file version {version}"

    kind = document.get("kind")
    if not isinstance(kind, str):
        return "damaged", "no kind"
    spec = KINDS.get(kind)
    if spec is None:
        return "unrecognised", f"{kind!r} is not an entry this version knows"

    for name in ("id", "at", "by", "by_source"):
        if not isinstance(document.get(name), str) or not document[name]:
            return "damaged", f"no {name}"

    target = document.get("target")
    if spec.needs_target:
        if not isinstance(target, str):
            return "damaged", "no target"
        try:
            checked_relative_key(target, project_directory)
        except ValueError as error:
            return "damaged", str(error)
    elif target is not None:
        return "damaged", f"a {kind} entry is not about a particular file"

    data = {
        key: value for key, value in document.items() if key not in ENVELOPE_KEYS
    }
    missing = [name for name in spec.required if name not in data]
    if missing:
        return "damaged", f"missing {', '.join(sorted(missing))}"

    note = document.get("note")
    batch = document.get("batch")
    return (
        Event(
            kind=kind,
            at=document["at"],
            by=document["by"],
            by_source=document["by_source"],
            data=data,
            target=target if spec.needs_target else None,
            batch=batch if isinstance(batch, str) else None,
            note=note if isinstance(note, str) else None,
            id=document["id"],
        ),
        "",
    )


# ---------------------------------------------------------------------------
# Recording, from an operation that changed something
# ---------------------------------------------------------------------------

@dataclass
class Recorder:
    """Records the changes of one run into whichever logs already exist.

    An operation is given one of these or nothing at all, so what records and
    what does not is visible at the call site rather than ambient. **A project
    with no log is left without one**: recording begins when somebody asks for
    it, which is what lets this ship without changing a collection that has
    not opted in.

    **Recording never fails an operation.** A change that happened and went
    unrecorded is a gap in the record; a save refused because its entry could
    not be written is lost work.

    Attributes:
        by: The author every entry is attributed to.
        by_source: How that name was arrived at.
        batch: Groups this run's entries.
    """

    by: str
    by_source: str
    batch: str = field(default_factory=new_batch)
    _roots: dict[Path, Path | None] = field(default_factory=dict, repr=False)

    @classmethod
    def for_run(cls, user: str | None = None) -> Recorder:
        """Return a recorder for one run, resolving who is running it."""
        by, by_source = resolve_author(user)
        return cls(by=by, by_source=by_source)

    def for_batch(self) -> Recorder:
        """Return a recorder for one operation, under this same author.

        A batch groups what a single action changed, so a window open all day
        starts a new one per save rather than gathering a session into one.
        The remembered project roots are shared, those being a property of the
        collection rather than of the run.
        """
        return Recorder(
            by=self.by, by_source=self.by_source, batch=new_batch(), _roots=self._roots
        )

    def record(self, path: Path, kind: str, **data: Any) -> Event | None:
        """Record *kind* against the file *path*, if its project keeps a log.

        Returns:
            The entry recorded, or None where there was no log to record into
            or the entry could not be written.
        """
        resolved = path.resolve()
        root = self.project_of(resolved)
        if root is None:
            return None
        try:
            target = resolved.relative_to(root).as_posix()
        except ValueError:
            logger.warning("%s is not inside the project %s; not recording", path, root)
            return None
        return self._append(root, kind, target, **data)

    def record_project(self, project_root: Path, kind: str, **data: Any) -> Event | None:
        """Record *kind* against the project itself rather than one of its files."""
        if not history_file_path(project_root).is_file():
            return None
        return self._append(project_root, kind, None, **data)

    def record_directory_rename(self, source: Path, destination: Path) -> int:
        """Record that every file now in *destination* used to be in *source*.

        A group is keyed by its label's path, so renaming a label directory
        changes the key of everything in it. Without this the history before
        the rename is orphaned -- and unlike a state file, which can be
        re-declared by hand, an orphaned history cannot be reattached.

        Returns:
            How many entries were recorded.
        """
        if not destination.is_dir():
            return 0
        recorded = 0
        for path in sorted(destination.iterdir()):
            if not path.is_file():
                continue
            resolved = path.resolve()
            root = self.project_of(resolved)
            if root is None:
                continue
            try:
                was = (source.resolve() / path.name).relative_to(root).as_posix()
                now = resolved.relative_to(root).as_posix()
            except ValueError:
                logger.warning("%s is not inside the project %s; not recording", path, root)
                continue
            if self._append(root, "renamed", now, renamed_from=was, scope="directory"):
                recorded += 1
        return recorded

    def _append(
        self, root: Path, kind: str, target: str | None, **data: Any
    ) -> Event | None:
        """Build and append an entry, reporting rather than raising on failure."""
        try:
            event = build_event(
                kind, by=self.by, by_source=self.by_source,
                target=target, batch=self.batch, **data,
            )
            return append_event(root, event)
        except (HistoryError, OSError, ValueError) as error:
            logger.warning("Could not record %s for %s: %s", kind, target or root, error)
            return None

    def project_of(self, path: Path) -> Path | None:
        """Return the project holding *path*, or None where none keeps a log.

        Public because archiving asks the same question before it writes
        anything: a project this would not record into is one it must not leave
        an archive in either. A batch touches a handful of directories over
        hundreds of files, so the walk up the tree happens once per directory
        rather than once per file.
        """
        directory = path.parent
        if directory not in self._roots:
            self._roots[directory] = containing_project(path)
        return self._roots[directory]


# ---------------------------------------------------------------------------
# Reading a log back
# ---------------------------------------------------------------------------

def events_for(log: HistoryLog, target: str) -> list[Event]:
    """Return the entries about *target*, oldest first."""
    return [event for event in log.events if event.target == target]


def latest_for(log: HistoryLog, target: str) -> Event | None:
    """Return the last entry recorded about *target*, or None if there is none."""
    found = events_for(log, target)
    return found[-1] if found else None


def union(logs: Iterable[Sequence[Event]]) -> list[Event]:
    """Return the entries of several logs, each kept once.

    Entries are addressed by their content, so combining two logs is a union
    rather than a reconciliation: order does not matter, running it twice
    changes nothing, and neither log has to know the other exists. The result
    is ordered by time and then by id, so two callers agree on it.
    """
    seen: dict[str, Event] = {}
    for events in logs:
        for event in events:
            seen.setdefault(event.id, event)
    return sorted(seen.values(), key=lambda event: (event.at, event.id))
